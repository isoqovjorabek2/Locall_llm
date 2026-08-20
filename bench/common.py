"""Shared benchmark plumbing: telemetry sampling, timing, result records.

Both the GPU (transformers) and CPU/GPU (llama.cpp) tracks emit the *same*
record schema so analyze/make_report.py can compare them directly.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "raw"

MODEL_ID = "google/gemma-4-26B-A4B-it"

# ---------------------------------------------------------------------------
# Benchmark prompt set
# ---------------------------------------------------------------------------
# Two axes matter for an MoE: prompt length (prefill is compute-bound and, at
# batch sizes above 1 token, touches *most* experts) and generation length
# (decode is memory-bound and touches only the top-8 routed experts per token).
# We sweep both so the CPU/GPU gap can be read separately for each phase.

_PARAGRAPH = (
    "The Amu Darya and Syr Darya rivers have shaped settlement patterns across "
    "Central Asia for millennia, feeding the oasis cities of Samarkand, Bukhara, "
    "and Khiva that anchored the Silk Road trade network. Irrigation systems drawn "
    "from these rivers supported dense agriculture in an otherwise arid basin. "
)

# name, prompt_repeats, max_new_tokens
BENCH_CASES = [
    ("short_prompt_short_gen", 1, 128),
    ("short_prompt_long_gen", 1, 512),
    ("long_prompt_short_gen", 24, 128),
    ("long_prompt_long_gen", 24, 512),
]


def build_prompt(repeats: int) -> str:
    body = _PARAGRAPH * repeats
    return (
        body
        + "\n\nBased on the passage above, write a clear analytical summary "
        "explaining how river geography influenced the development of these cities."
    )


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------
@dataclass
class Telemetry:
    """Background sampler for GPU memory/power/util and process RSS."""

    interval: float = 0.25
    gpu_indices: list = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: Any = field(default=None, repr=False)
    vram_mib: list = field(default_factory=list)
    power_w: list = field(default_factory=list)
    gpu_util: list = field(default_factory=list)
    rss_mib: list = field(default_factory=list)

    def _sample_gpu(self) -> None:
        if not shutil.which("nvidia-smi") or not self.gpu_indices:
            return
        ids = ",".join(str(i) for i in self.gpu_indices)
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--id=" + ids,
                    "--query-gpu=memory.used,power.draw,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (subprocess.SubprocessError, OSError):
            return
        if out.returncode != 0:
            return
        mem_total = 0.0
        pwr_total = 0.0
        util_total = 0.0
        rows = 0
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            try:
                mem_total += float(parts[0])
                pwr_total += float(parts[1])
                util_total += float(parts[2])
            except ValueError:
                continue
            rows += 1
        if rows:
            self.vram_mib.append(mem_total)
            self.power_w.append(pwr_total)
            self.gpu_util.append(util_total / rows)

    def _sample_rss(self) -> None:
        # /proc is the dependency-free route on Linux; skip elsewhere.
        try:
            with open("/proc/" + str(os.getpid()) + "/status", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        self.rss_mib.append(float(line.split()[1]) / 1024.0)
                        return
        except (OSError, ValueError, IndexError):
            return

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample_gpu()
            self._sample_rss()
            self._stop.wait(self.interval)

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def summary(self) -> dict:
        def _peak(xs):
            return round(max(xs), 1) if xs else None

        def _mean(xs):
            return round(statistics.fmean(xs), 1) if xs else None

        return {
            "peak_vram_mib": _peak(self.vram_mib),
            "mean_power_w": _mean(self.power_w),
            "peak_power_w": _peak(self.power_w),
            "mean_gpu_util_pct": _mean(self.gpu_util),
            "peak_host_rss_mib": _peak(self.rss_mib),
        }


# ---------------------------------------------------------------------------
# Host description
# ---------------------------------------------------------------------------
def cpu_model_name() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def physical_cores() -> int:
    """Physical cores, not SMT threads -- llama.cpp throughput drops past them."""
    try:
        out = subprocess.run(
            ["lscpu", "-p=Core,Socket"], capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0:
            pairs = set()
            for line in out.stdout.splitlines():
                if line and not line.startswith("#"):
                    pairs.add(line.strip())
            if pairs:
                return len(pairs)
    except (subprocess.SubprocessError, OSError):
        pass
    return os.cpu_count() or 8


def gpu_inventory() -> list:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mib": int(parts[2]),
                    "memory_used_mib_at_start": int(parts[3]),
                    "driver": parts[4],
                }
            )
        except ValueError:
            continue
    return gpus


def host_info() -> dict:
    return {
        "hostname": platform.node(),
        "os": platform.system() + " " + platform.release(),
        "cpu": cpu_model_name(),
        "physical_cores": physical_cores(),
        "logical_cores": os.cpu_count(),
        "gpus": gpu_inventory(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def make_record(
    track: str,
    backend: str,
    device: str,
    precision: str,
    case: str,
    prompt_tokens: int,
    generated_tokens: int,
    ttft_s: float,
    total_s: float,
    telemetry: dict,
    extra: dict = None,
) -> dict:
    """One measured generation.

    ttft_s is time-to-first-token, which is dominated by prefill.
    decode_s is the remainder, during which generated_tokens-1 tokens land.
    """
    decode_s = max(total_s - ttft_s, 1e-9)
    decode_tokens = max(generated_tokens - 1, 1)
    return {
        "track": track,          # "A_same_engine" | "B_best_realistic"
        "backend": backend,      # "transformers" | "llama.cpp"
        "device": device,        # "gpu" | "cpu"
        "precision": precision,  # "bf16" | "Q4_K_M" | "int4" ...
        "case": case,
        "model": MODEL_ID,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "ttft_s": round(ttft_s, 4),
        "total_s": round(total_s, 4),
        "prefill_tok_s": round(prompt_tokens / ttft_s, 2) if ttft_s > 0 else None,
        "decode_tok_s": round(decode_tokens / decode_s, 2),
        "telemetry": telemetry,
        "extra": extra or {},
    }


def write_records(records: list, filename: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def write_meta(info: dict, filename: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
