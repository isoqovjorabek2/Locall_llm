"""GPU benchmark: Gemma 4 26B-A4B via transformers on the A40s.

Track B ("best realistic GPU deployment"): bf16 weights sharded across both
A40s with accelerate's device_map. Optionally 8-bit/4-bit to fit a single card.

Usage:
    python bench/bench_hf_gpu.py                     # bf16, both GPUs
    python bench/bench_hf_gpu.py --gpus 1            # bf16, GPU 1 only (needs quant)
    python bench/bench_hf_gpu.py --load-4bit --gpus 1
    python bench/bench_hf_gpu.py --reserve-mib 8000  # leave room for other users
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from threading import Thread

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.common import (  # noqa: E402
    BENCH_CASES,
    MODEL_ID,
    Telemetry,
    build_prompt,
    gpu_inventory,
    host_info,
    make_record,
    require_gemma4_support,
    write_meta,
    write_records,
)


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark Gemma 4 26B-A4B on GPU")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument(
        "--gpus",
        default="0,1",
        help="Comma-separated CUDA device indices to expose (default: 0,1)",
    )
    p.add_argument(
        "--reserve-mib",
        type=int,
        default=6000,
        help="VRAM per GPU to leave free for other users on this shared box "
        "(default: 6000). Subtracted from each card's free memory.",
    )
    p.add_argument("--load-8bit", action="store_true")
    p.add_argument("--load-4bit", action="store_true")
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3, help="Timed runs per case")
    p.add_argument("--out", default="gpu_transformers.jsonl")
    return p.parse_args()


def build_max_memory(gpu_indices, reserve_mib):
    """Cap per-GPU usage at (free - reserve) so we do not evict other jobs.

    The A40s on this box are shared; nvidia-smi already showed resident
    python3 processes on both cards. Blindly taking all memory would OOM them.
    """
    inv = {g["index"]: g for g in gpu_inventory()}
    max_memory = {}
    for local_idx, phys_idx in enumerate(gpu_indices):
        g = inv.get(phys_idx)
        if g is None:
            continue
        free = g["memory_total_mib"] - g["memory_used_mib_at_start"]
        budget = max(free - reserve_mib, 1024)
        # Keys are indices into CUDA_VISIBLE_DEVICES, i.e. re-based to 0..N-1.
        max_memory[local_idx] = str(budget) + "MiB"
    max_memory["cpu"] = "0GiB"  # fail loudly rather than silently offload to RAM
    return max_memory


def main():
    args = parse_args()

    gpu_indices = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    max_memory = build_max_memory(gpu_indices, args.reserve_mib)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_indices)

    require_gemma4_support()

    import torch  # noqa: E402  (import after CUDA_VISIBLE_DEVICES is set)
    from transformers import AutoProcessor, TextIteratorStreamer
    from transformers import Gemma4ForConditionalGeneration as ModelCls

    if not torch.cuda.is_available():
        sys.exit("CUDA is not available -- this script is the GPU track.")

    precision = "bf16"
    quant_cfg = None
    if args.load_4bit or args.load_8bit:
        from transformers import BitsAndBytesConfig

        if args.load_4bit:
            precision = "nf4"
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            precision = "int8"
            quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

    print("[info] loading " + args.model + " (" + precision + ") onto GPUs " + args.gpus)
    print("[info] per-GPU budget: " + str(max_memory))
    load_start = time.perf_counter()

    processor = AutoProcessor.from_pretrained(args.model)
    load_kwargs = dict(
        device_map="auto",
        max_memory=max_memory,
        attn_implementation=args.attn,
    )
    if quant_cfg is not None:
        load_kwargs["quantization_config"] = quant_cfg
    else:
        load_kwargs["dtype"] = torch.bfloat16

    model = ModelCls.from_pretrained(args.model, **load_kwargs)
    model.eval()
    load_s = time.perf_counter() - load_start
    print("[info] loaded in " + str(round(load_s, 1)) + "s")
    print("[info] device map: " + str(getattr(model, "hf_device_map", "n/a")))

    tokenizer = getattr(processor, "tokenizer", processor)

    def encode(prompt_text):
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
        # Thinking mode is disabled so token counts reflect the answer only.
        for kwargs in (
            dict(add_generation_prompt=True, tokenize=True, return_dict=True,
                 return_tensors="pt", enable_thinking=False),
            dict(add_generation_prompt=True, tokenize=True, return_dict=True,
                 return_tensors="pt"),
        ):
            try:
                return processor.apply_chat_template(messages, **kwargs)
            except (TypeError, ValueError):
                continue
        raise RuntimeError("Could not apply chat template")

    def run_once(prompt_text, max_new_tokens):
        inputs = encode(prompt_text)
        inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, "to")}
        n_prompt = int(inputs["input_ids"].shape[-1])

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,  # fixed length => comparable timings
            do_sample=False,
            streamer=streamer,
        )
        torch.cuda.synchronize()
        start = time.perf_counter()
        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        ttft = None
        pieces = []
        for chunk in streamer:
            if ttft is None:
                ttft = time.perf_counter() - start
            pieces.append(chunk)
        thread.join()
        torch.cuda.synchronize()
        total = time.perf_counter() - start

        text = "".join(pieces)
        n_gen = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        return n_prompt, max(n_gen, 1), (ttft or total), total, text

    # Warmup: first call pays CUDA graph/kernel autotune costs.
    for _ in range(args.warmup):
        run_once(build_prompt(1), 32)

    records = []
    telemetry_labels = list(range(len(gpu_indices)))
    for case_name, repeats, max_new in BENCH_CASES:
        prompt = build_prompt(repeats)
        for run_idx in range(args.repeats):
            tele = Telemetry(gpu_indices=[gpu_indices[i] for i in telemetry_labels]).start()
            n_prompt, n_gen, ttft, total, _text = run_once(prompt, max_new)
            tele.stop()
            rec = make_record(
                track="B_best_realistic",
                backend="transformers",
                device="gpu",
                precision=precision,
                case=case_name,
                prompt_tokens=n_prompt,
                generated_tokens=n_gen,
                ttft_s=ttft,
                total_s=total,
                telemetry=tele.summary(),
                extra={
                    "run_index": run_idx,
                    "gpus": gpu_indices,
                    "attn": args.attn,
                    "load_seconds": round(load_s, 1),
                    "device_map": str(getattr(model, "hf_device_map", "")),
                },
            )
            records.append(rec)
            print(
                "  " + case_name + " run" + str(run_idx)
                + ": prefill " + str(rec["prefill_tok_s"]) + " tok/s, "
                + "decode " + str(rec["decode_tok_s"]) + " tok/s, "
                + "TTFT " + str(rec["ttft_s"]) + "s"
            )

    path = write_records(records, args.out)
    write_meta(host_info(), "host_gpu.json")
    print("[done] wrote " + str(path))


if __name__ == "__main__":
    main()
