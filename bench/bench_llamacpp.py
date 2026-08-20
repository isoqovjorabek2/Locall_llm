"""llama.cpp benchmark: same GGUF weights on CPU and GPU.

This is Track A, the controlled comparison. Because the engine, the
quantisation, and the tokenizer are all held constant and only `-ngl`
changes, the CPU/GPU ratio here is a clean hardware measurement.

(bench_hf_gpu.py is Track B: bf16 transformers, i.e. the best the GPU can
actually do in a real deployment. Comparing that directly against CPU Q4
conflates hardware with quantisation, so we report both tracks separately.)

Usage:
    python bench/bench_llamacpp.py --gguf /models/gemma-4-26b-a4b-it-Q4_K_M.gguf
    python bench/bench_llamacpp.py --gguf ... --devices cpu
    python bench/bench_llamacpp.py --gguf ... --threads 32
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.common import (  # noqa: E402
    Telemetry,
    host_info,
    physical_cores,
    write_meta,
    write_records,
)

# (label, n_prompt, n_gen) -- llama-bench measures prefill and decode separately.
LLAMA_BENCH_CASES = [
    ("pp512", 512, 0),
    ("pp4096", 4096, 0),
    ("tg128", 0, 128),
    ("tg512", 0, 512),
]


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark Gemma 4 GGUF on CPU and GPU")
    p.add_argument("--gguf", required=True, help="Path to the .gguf file")
    p.add_argument(
        "--llama-bench",
        default="llama-bench",
        help="Path to the llama-bench binary (default: found on PATH)",
    )
    p.add_argument(
        "--devices",
        default="cpu,gpu",
        help="Which devices to test: cpu, gpu, or cpu,gpu (default: both)",
    )
    p.add_argument("--threads", type=int, default=0, help="0 = auto (physical cores)")
    p.add_argument("--gpu-layers", type=int, default=99, help="-ngl for the GPU run")
    p.add_argument("--gpu-index", type=int, default=1,
                   help="Which CUDA device for the GPU run (default 1, the emptier A40)")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", default="llamacpp.jsonl")
    return p.parse_args()


def current_vram_mib(gpu_index):
    """Instantaneous VRAM use on one card, for the offload sanity check."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--id=" + str(gpu_index),
             "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return float(out.stdout.strip().splitlines()[0])
    except (subprocess.SubprocessError, OSError, ValueError, IndexError):
        return None
    return None


def has_cuda_backend(binary):
    """Was this llama.cpp actually built with CUDA?

    A CPU-only build still accepts -ngl and still runs -- it just ignores the
    flag. Without this check the 'GPU' row would silently be CPU numbers.
    """
    try:
        out = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=30)
        blob = (out.stdout + out.stderr).lower()
        if "cuda" in blob:
            return True
    except (subprocess.SubprocessError, OSError):
        pass
    return None  # unknown; the VRAM delta check below is the real arbiter


def run_llama_bench(binary, gguf, n_prompt, n_gen, ngl, threads, repeats, gpu_index):
    """Invoke llama-bench once, return (parsed_rows, telemetry_summary, wall)."""
    cmd = [
        binary,
        "-m", gguf,
        "-p", str(n_prompt),
        "-n", str(n_gen),
        "-ngl", str(ngl),
        "-t", str(threads),
        "-r", str(repeats),
        "-o", "json",
    ]
    env = None
    gpu_ids = []
    if ngl > 0:
        import os

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        gpu_ids = [gpu_index]

    baseline_vram = current_vram_mib(gpu_index) if ngl > 0 else None

    tele = Telemetry(gpu_indices=gpu_ids).start()
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    wall = time.perf_counter() - start
    tele.stop()

    summary = tele.summary()

    # Did the GPU run actually touch the GPU? A CPU-only llama.cpp accepts -ngl
    # and reports success while quietly running everything on the CPU.
    if ngl > 0:
        peak = summary.get("peak_vram_mib")
        if peak is not None and baseline_vram is not None:
            delta = peak - baseline_vram
            summary["vram_delta_mib"] = round(delta, 1)
            summary["gpu_actually_used"] = delta > 500
            if delta <= 500:
                print(
                    "[ERROR] -ngl " + str(ngl) + " was requested but VRAM on GPU "
                    + str(gpu_index) + " rose only " + str(round(delta, 1)) + " MiB.\n"
                    "        This llama.cpp is almost certainly a CPU-only build, so these\n"
                    "        'GPU' numbers are really CPU numbers. Do not report them.\n"
                    "        Rebuild with CUDA: bash scripts/setup_server.sh llamacpp",
                    file=sys.stderr,
                )
        else:
            summary["gpu_actually_used"] = None

    if proc.returncode != 0:
        print("[error] llama-bench failed:\n" + proc.stderr[-4000:], file=sys.stderr)
        return [], summary, wall

    # llama-bench -o json prints a JSON array; stderr carries the load logs.
    text = proc.stdout.strip()
    brace = text.find("[")
    if brace == -1:
        print("[error] no JSON in llama-bench output:\n" + text[:2000], file=sys.stderr)
        return [], summary, wall
    try:
        rows = json.loads(text[brace:])
    except json.JSONDecodeError as exc:
        print("[error] could not parse llama-bench JSON: " + str(exc), file=sys.stderr)
        return [], summary, wall
    return rows, summary, wall


def main():
    args = parse_args()

    gguf = Path(args.gguf).expanduser()
    if not gguf.exists():
        sys.exit("GGUF not found: " + str(gguf))

    binary = args.llama_bench
    if shutil.which(binary) is None and not Path(binary).exists():
        sys.exit(
            "llama-bench not found at '" + binary + "'.\n"
            "Build it first (see scripts/setup_server.sh) or pass --llama-bench /path/to/llama-bench"
        )

    threads = args.threads or physical_cores()
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    print("[info] gguf=" + gguf.name + " threads=" + str(threads) + " devices=" + str(devices))

    if "gpu" in devices and has_cuda_backend(binary) is not True:
        print(
            "[warn] Could not confirm this llama-bench was built with CUDA.\n"
            "       If it was not, the GPU rows will silently be CPU numbers.\n"
            "       The VRAM check after each GPU run will tell you for certain.",
            file=sys.stderr,
        )

    records = []
    for device in devices:
        ngl = args.gpu_layers if device == "gpu" else 0
        for label, n_prompt, n_gen in LLAMA_BENCH_CASES:
            print("[run] " + device + " " + label + " ...")
            rows, tele, wall = run_llama_bench(
                binary, str(gguf), n_prompt, n_gen, ngl, threads, args.repeats, args.gpu_index
            )
            for row in rows:
                avg_ts = row.get("avg_ts")
                if avg_ts is None:
                    continue
                is_prefill = int(row.get("n_prompt", 0)) > 0
                n_tokens = int(row.get("n_prompt", 0) or row.get("n_gen", 0))
                # llama-bench reports throughput directly; derive a time from it
                # so the record schema matches the transformers track.
                seconds = n_tokens / avg_ts if avg_ts > 0 else 0.0
                rec = {
                    "track": "A_same_engine",
                    "backend": "llama.cpp",
                    "device": device,
                    "precision": row.get("model_type", "Q4_K_M"),
                    "case": label,
                    "model": gguf.name,
                    "phase": "prefill" if is_prefill else "decode",
                    "prompt_tokens": int(row.get("n_prompt", 0)),
                    "generated_tokens": int(row.get("n_gen", 0)),
                    "tok_s": round(avg_ts, 2),
                    "tok_s_stddev": round(row.get("stddev_ts", 0.0), 2),
                    "prefill_tok_s": round(avg_ts, 2) if is_prefill else None,
                    "decode_tok_s": round(avg_ts, 2) if not is_prefill else None,
                    "seconds": round(seconds, 4),
                    "ttft_s": None,
                    "total_s": round(wall, 3),
                    "telemetry": tele,
                    "extra": {
                        "n_gpu_layers": row.get("n_gpu_layers"),
                        "n_threads": row.get("n_threads"),
                        "model_size_bytes": row.get("model_size"),
                        "model_n_params": row.get("model_n_params"),
                        "build_commit": row.get("build_commit"),
                        "repeats": args.repeats,
                    },
                }
                records.append(rec)
                print(
                    "    " + label + " " + device + ": " + str(rec["tok_s"])
                    + " tok/s (+/- " + str(rec["tok_s_stddev"]) + ")"
                )

    if not records:
        sys.exit("No results collected -- check the llama-bench errors above.")

    path = write_records(records, args.out)
    write_meta(host_info(), "host_llamacpp.json")
    print("[done] wrote " + str(path))


if __name__ == "__main__":
    main()
