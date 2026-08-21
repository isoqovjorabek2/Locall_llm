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


def _root_cause(exc):
    """Walk to the original exception.

    transformers catches a peer dependency's ImportError and re-raises its own
    "Could not import module 'X'", which drops exc.name and names the class it
    was building rather than the module that was actually absent.
    """
    seen = set()
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        nxt = cur.__cause__ or cur.__context__
        if nxt is None:
            break
        cur = nxt
    return cur


def _dependency_report(exc):
    """Human-readable diagnosis for an import failure inside transformers."""
    root = _root_cause(exc)
    missing = getattr(root, "name", None)
    if not missing:
        text = str(root)
        for candidate in ("torchvision", "torchaudio", "PIL", "timm", "av",
                          "sentencepiece", "protobuf", "einops",
                          "causal_conv1d", "flash_attn"):
            if candidate in text:
                missing = candidate
                break

    probe_lines = []
    for mod in ("torch", "torchvision", "PIL", "accelerate"):
        try:
            imported = __import__(mod)
            probe_lines.append(
                "    " + mod.ljust(14) + " "
                + str(getattr(imported, "__version__", "installed"))
            )
        except Exception as probe_exc:  # noqa: BLE001 - reporting, not handling
            probe_lines.append(
                "    " + mod.ljust(14) + " MISSING (" + type(probe_exc).__name__ + ")"
            )

    root_text = str(root) + " " + str(exc)
    abi_mismatch = (
        "does not exist" in root_text and "torchvision::" in root_text
    ) or "undefined symbol" in root_text

    if abi_mismatch:
        hint = (
            "\nThis is a torch/torchvision ABI MISMATCH, not a missing package.\n"
            "Install the pair together from one index:\n"
            "  bash scripts/setup_server.sh torch\n"
        )
    elif missing:
        hint = (
            "\nInstall the missing dependency:\n"
            "  source .venv/bin/activate && pip install " + str(missing) + "\n"
            "(if it is torch or torchvision: bash scripts/setup_server.sh torch)\n"
        )
    else:
        hint = ""

    return (
        "  transformers said: " + str(exc) + "\n"
        "  root cause:        " + type(root).__name__ + ": " + str(root) + "\n"
        "  missing module:    " + str(missing or "see root cause above") + "\n\n"
        "  currently importable:\n" + "\n".join(probe_lines) + "\n" + hint
    )


def require_transformers():
    """Import transformers, or exit with a message naming the real problem."""
    import sys

    try:
        import transformers
    except ImportError as exc:
        sys.exit(
            "transformers could not be imported: " + str(exc) + "\n"
            "  source .venv/bin/activate && pip install -r requirements.txt"
        )
    except Exception as exc:  # installed but broken
        sys.exit(
            "transformers is installed but fails to import:\n"
            "  " + type(exc).__name__ + ": " + str(exc) + "\n\n"
            "Reinstall it inside the venv:\n"
            "  source .venv/bin/activate && pip install -r requirements.txt"
        )
    return getattr(transformers, "__version__", "unknown")


def resolve_model_class(model_id, trust_remote_code=False):
    """Pick the right model class for ANY model, from its own config.

    Reads architectures[0] out of the checkpoint's config and looks it up in
    transformers. This is what makes the harness model-agnostic: Gemma 4 needs
    Gemma4ForConditionalGeneration, Qwen3.5 needs Qwen3_5ForConditionalGeneration,
    a plain Llama needs LlamaForCausalLM -- all resolved the same way, with no
    per-model branching to maintain.

    Returns (ModelCls, config, architecture_name).
    """
    import sys

    version = require_transformers()
    import transformers
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
    except Exception as exc:  # noqa: BLE001
        root = _root_cause(exc)
        extra = ""
        if "trust_remote_code" in (str(exc) + str(root)):
            extra = (
                "\nThis checkpoint ships custom modelling code. Re-run with\n"
                "  --trust-remote-code\n"
                "only if you trust the publisher -- it executes their Python.\n"
            )
        sys.exit(
            "Could not read the config for " + str(model_id) + "\n\n"
            "  transformers: " + version + "\n"
            "  error:        " + type(exc).__name__ + ": " + str(exc) + "\n" + extra
            + "\nIf the id is wrong, check it on huggingface.co. If the model is\n"
            "newer than your transformers, upgrade:\n"
            "  source .venv/bin/activate && pip install -U transformers"
        )

    arch_list = getattr(config, "architectures", None) or []
    arch = arch_list[0] if arch_list else None
    model_type = getattr(config, "model_type", "unknown")

    if arch and hasattr(transformers, arch):
        try:
            return getattr(transformers, arch), config, arch
        except Exception as exc:  # noqa: BLE001 - lazy resolution fails here
            sys.exit(
                "transformers " + version + " lists " + str(arch)
                + " but could not import it.\n\n" + _dependency_report(exc)
            )

    # Architecture unknown to this transformers: usually too old for the model.
    try:
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM, config, (arch or "AutoModelForCausalLM")
    except Exception:  # noqa: BLE001
        pass

    sys.exit(
        "transformers " + version + " does not know how to build this model.\n\n"
        "  model:         " + str(model_id) + "\n"
        "  model_type:    " + str(model_type) + "\n"
        "  architectures: " + str(arch_list or "(none listed)") + "\n\n"
        "That architecture is not exposed by this transformers, which usually\n"
        "means the library predates the model. Upgrade:\n"
        "  source .venv/bin/activate && pip install -U transformers\n\n"
        "If the checkpoint ships its own modelling code, pass --trust-remote-code\n"
        "(only if you trust the publisher -- it executes their Python)."
    )


def load_processor(model_id, trust_remote_code=False):
    """Return (processor_or_tokenizer, tokenizer).

    Multimodal checkpoints expose an AutoProcessor; text-only ones only have a
    tokenizer. Try the processor first and fall back, so both work unchanged.
    """
    import sys

    require_transformers()
    from transformers import AutoProcessor, AutoTokenizer

    processor = None
    try:
        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
    except Exception as exc:  # noqa: BLE001
        root_text = str(_root_cause(exc)) + " " + str(exc)
        # A real dependency failure must not be mistaken for "text-only".
        if "torchvision" in root_text or "undefined symbol" in root_text:
            sys.exit(
                "The processor for " + str(model_id) + " could not be loaded.\n\n"
                + _dependency_report(exc)
            )

    if processor is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=trust_remote_code
            )
        except Exception as exc:  # noqa: BLE001
            sys.exit(
                "Could not load a processor or tokenizer for " + str(model_id)
                + "\n\n" + _dependency_report(exc)
            )
        return tokenizer, tokenizer

    return processor, getattr(processor, "tokenizer", processor)


def apply_chat(processor, prompt_text):
    """Build model inputs from one user turn, across processor/tokenizer APIs.

    Tries the multimodal content-list form first, then plain string content,
    and disables thinking mode where supported so token counts reflect the
    answer rather than hidden reasoning.
    """
    multimodal = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    plain = [{"role": "user", "content": prompt_text}]

    attempts = []
    for messages in (multimodal, plain):
        for extra in ({"enable_thinking": False}, {}):
            kwargs = dict(
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            kwargs.update(extra)
            attempts.append((messages, kwargs))

    last_error = None
    for messages, kwargs in attempts:
        try:
            out = processor.apply_chat_template(messages, **kwargs)
        except Exception as exc:  # noqa: BLE001 - probing API shapes
            last_error = exc
            continue
        if hasattr(out, "keys"):
            return out
        return {"input_ids": out}

    raise RuntimeError(
        "Could not apply this model's chat template: " + repr(last_error)
    )


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
    model: str = None,
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
        "model": model or MODEL_ID,
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
