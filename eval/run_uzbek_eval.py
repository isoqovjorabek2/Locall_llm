"""Run the Uzbek suite against Gemma 4 and score the output.

Two backends, so the same suite can be pointed at either deployment:

    # GPU, bf16, straight transformers
    python eval/run_uzbek_eval.py --backend hf --tag gpu-bf16

    # anything served by llama-server (CPU or GPU, any quant)
    python eval/run_uzbek_eval.py --backend server --url http://127.0.0.1:8080 --tag cpu-q4

The interesting comparison is not CPU-vs-GPU quality (identical weights give
identical text). It is bf16-vs-Q4 quality: low-resource languages tend to
degrade under quantisation faster than English does, and Uzbek is exactly the
kind of language where that shows up first.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.uz_scoring import (  # noqa: E402
    build_trigram_model,
    load_suite,
    score_item,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "raw"
MODEL_ID = "google/gemma-4-26B-A4B-it"


def parse_args():
    p = argparse.ArgumentParser(description="Uzbek quality eval for Gemma 4")
    p.add_argument("--backend", choices=["hf", "server"], default="hf")
    p.add_argument("--model", default=MODEL_ID, help="HF model id (backend=hf)")
    p.add_argument("--url", default="http://127.0.0.1:8080",
                   help="llama-server base URL (backend=server)")
    p.add_argument("--tag", required=True,
                   help="Label for this run, e.g. gpu-bf16 or cpu-q4km")
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--reserve-mib", type=int, default=6000)
    p.add_argument("--load-4bit", action="store_true")
    p.add_argument("--load-8bit", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--categories", default="",
                   help="Comma-separated category filter (default: all)")
    p.add_argument("--suite", default="")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
def make_hf_generator(args):
    import os

    gpu_indices = [int(x) for x in args.gpus.split(",") if x.strip()]
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from bench.bench_hf_gpu import build_max_memory

    max_memory = build_max_memory(gpu_indices, args.reserve_mib)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_indices)

    import torch
    from transformers import AutoProcessor

    try:
        from transformers import Gemma4ForConditionalGeneration as ModelCls
    except ImportError:
        from transformers import AutoModelForCausalLM as ModelCls

    load_kwargs = dict(device_map="auto", max_memory=max_memory, attn_implementation="sdpa")
    precision = "bf16"
    if args.load_4bit or args.load_8bit:
        from transformers import BitsAndBytesConfig

        if args.load_4bit:
            precision = "nf4"
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            )
        else:
            precision = "int8"
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        load_kwargs["dtype"] = torch.bfloat16

    print("[info] loading " + args.model + " (" + precision + ")")
    processor = AutoProcessor.from_pretrained(args.model)
    model = ModelCls.from_pretrained(args.model, **load_kwargs)
    model.eval()
    tokenizer = getattr(processor, "tokenizer", processor)

    def generate(prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = None
        for kwargs in (
            dict(add_generation_prompt=True, tokenize=True, return_dict=True,
                 return_tensors="pt", enable_thinking=False),
            dict(add_generation_prompt=True, tokenize=True, return_dict=True,
                 return_tensors="pt"),
        ):
            try:
                inputs = processor.apply_chat_template(messages, **kwargs)
                break
            except (TypeError, ValueError):
                continue
        if inputs is None:
            raise RuntimeError("Could not apply chat template")

        inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, "to")}
        n_prompt = int(inputs["input_ids"].shape[-1])
        gen_kwargs = dict(max_new_tokens=args.max_new_tokens)
        if args.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=args.temperature)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][n_prompt:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return generate, precision


def make_server_generator(args):
    import requests

    base = args.url.rstrip("/")

    def generate(prompt: str) -> str:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "stream": False,
        }
        resp = requests.post(base + "/v1/chat/completions", json=payload, timeout=1800)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    return generate, "server"


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    suite = load_suite(args.suite or None)
    trigram_model = build_trigram_model(suite)

    items = suite["items"]
    if args.categories:
        wanted = {c.strip() for c in args.categories.split(",") if c.strip()}
        items = [it for it in items if it.get("category") in wanted]
    if not items:
        sys.exit("No items matched the category filter.")

    if args.backend == "hf":
        generate, precision = make_hf_generator(args)
    else:
        generate, precision = make_server_generator(args)

    print("[info] running " + str(len(items)) + " items, tag=" + args.tag)
    rows = []
    for idx, item in enumerate(items, 1):
        start = time.perf_counter()
        try:
            output = generate(item["prompt"])
            error = None
        except Exception as exc:  # keep going; a single failure should not kill the run
            output = ""
            error = repr(exc)
            print("  [error] " + item["id"] + ": " + error, file=sys.stderr)
        elapsed = time.perf_counter() - start

        scores = score_item(item, output, trigram_model)
        rows.append({
            "tag": args.tag,
            "backend": args.backend,
            "precision": precision,
            "id": item["id"],
            "category": item["category"],
            "prompt": item["prompt"],
            "reference": item.get("reference"),
            "output": output,
            "error": error,
            "seconds": round(elapsed, 2),
            "scores": scores,
        })
        flag = "ok " if scores.get("uzbek_ok") is not False else "FLAG"
        print(
            "  [" + str(idx) + "/" + str(len(items)) + "] " + flag + " " + item["id"]
            + "  script=" + str(scores.get("script"))
            + " known_word=" + str(scores.get("known_word_rate"))
            + " unseen_tri=" + str(scores.get("unseen_trigram_rate"))
            + " chrf=" + str(scores.get("chrf"))
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / ("uzbek_eval_" + args.tag + ".jsonl")
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("[done] wrote " + str(out_path))
    print("Now build the comparison: python analyze/make_report.py")


if __name__ == "__main__":
    main()
