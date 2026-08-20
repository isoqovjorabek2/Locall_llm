#!/usr/bin/env bash
# Run the full comparison end to end and build results/REPORT.md.
#
#   bash scripts/run_all.sh
#
# Roughly: Track A (llama.cpp CPU+GPU) ~20-40 min, Track B (transformers bf16)
# ~15 min plus a slow first load, Uzbek eval ~10-20 min per configuration.
# Run it under tmux -- do not let an SSH drop kill a two-hour job.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-script}")/.." && pwd)"
cd "${REPO_ROOT}"

MODELS_DIR="${REPO_ROOT}/models"
LLAMA_DIR="${LLAMA_CPP_DIR:-${HOME}/llama.cpp}"
LLAMA_BENCH="${LLAMA_BENCH:-${LLAMA_DIR}/build/bin/llama-bench}"
LLAMA_SERVER="${LLAMA_SERVER:-${LLAMA_DIR}/build/bin/llama-server}"
SERVER_PORT="${SERVER_PORT:-8080}"

# GPU 1 was the emptier A40 in the nvidia-smi snapshot; override if that changed.
GPU_FOR_LLAMACPP="${GPU_FOR_LLAMACPP:-1}"

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31m[stop] %s\033[0m\n' "$*"; exit 1; }

# A failing pipeline inside a command substitution aborts `set -e` scripts
# silently. Report where, instead of the run appearing to stop for no reason.
trap 'rc=$?; [ $rc -ne 0 ] && printf "\n\033[1;31m[error] aborted at %s line %s (exit %s)\033[0m\n  Re-run with: bash -x scripts/run_all.sh\n" "${BASH_SOURCE[0]:-script}" "${LINENO:-?}" "$rc"; exit $rc' ERR

# Under sudo, HOME becomes /root, so LLAMA_DIR resolves to /root/llama.cpp and
# the whole run looks for binaries and caches in the wrong place.
if [ -n "${SUDO_USER:-}" ] || [ "$(id -u)" -eq 0 ]; then
  die "Do not run the benchmarks with sudo.

  HOME becomes /root, so llama.cpp is looked up at /root/llama.cpp and Hugging
  Face reads from /root/.cache -- neither is where setup put things.

  Run as yourself:
      bash scripts/run_all.sh"
fi

if [ ! -x "${REPO_ROOT}/.venv/bin/python" ]; then
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    die "Your shell shows an active venv (${VIRTUAL_ENV}) but ${REPO_ROOT}/.venv/bin/python
  does not exist -- it was deleted while still activated, so the prompt and PATH
  are stale and everything falls through to the system python.

      deactivate
      bash scripts/setup_server.sh python"
  fi
  die "No venv at ${REPO_ROOT}/.venv
      Run first: bash scripts/setup_server.sh"
fi
# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"

# Fail here, with the fix, rather than deep inside AutoProcessor.
python - <<'PY' || exit 1
import sys
sys.path.insert(0, ".")
from bench.common import require_gemma4_support
print("[info] transformers", require_gemma4_support())
PY

GGUF="$(find "${MODELS_DIR}" -name '*.gguf' 2>/dev/null | head -1 || true)"

# The HF weights live in the shared cache, so check for a real local snapshot
# rather than assuming; an empty cache means run_all would do nothing useful.
HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"
if ! find "${HF_CACHE}" -type d -name '*gemma-4-26B-A4B-it*' 2>/dev/null | grep -q .; then
  warn "No local snapshot of google/gemma-4-26B-A4B-it under ${HF_CACHE}."
  warn "The GPU track and the bf16 Uzbek eval will download ~52 GB on first use,"
  warn "or fail if the disk is short. Pre-fetch with: bash scripts/download_models.sh"
fi
if [ -z "${GGUF}" ]; then
  warn "No .gguf under ${MODELS_DIR} -- the CPU track and the Q4 eval will be skipped."
  warn "Fetch it with: bash scripts/download_models.sh gguf"
fi
if [ -z "${GGUF}" ] && ! find "${HF_CACHE}" -type d -name '*gemma-4-26B-A4B-it*' 2>/dev/null | grep -q .; then
  die "Neither weight set is present -- there is nothing to benchmark.
      Run first: bash scripts/download_models.sh"
fi

log "Current GPU state (check you are not about to evict someone's job)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv || true

# ---------------------------------------------------------------------------
log "Track A: llama.cpp, identical GGUF on CPU and GPU"
if [ -n "${GGUF}" ] && [ -x "${LLAMA_BENCH}" ]; then
  python bench/bench_llamacpp.py \
    --gguf "${GGUF}" \
    --llama-bench "${LLAMA_BENCH}" \
    --devices cpu,gpu \
    --gpu-index "${GPU_FOR_LLAMACPP}" \
    --repeats 3
else
  echo "[skip] need a .gguf under models/ and llama-bench at ${LLAMA_BENCH}"
fi

# ---------------------------------------------------------------------------
log "Track B: transformers bf16 across both A40s"
python bench/bench_hf_gpu.py --gpus 0,1 --reserve-mib 6000 --repeats 3 \
  || echo "[warn] GPU benchmark failed -- see the traceback above"

# ---------------------------------------------------------------------------
log "Uzbek eval -- GPU bf16"
python eval/run_uzbek_eval.py --backend hf --tag gpu-bf16 --gpus 0,1 \
  || echo "[warn] bf16 Uzbek eval failed"

# ---------------------------------------------------------------------------
log "Uzbek eval -- CPU Q4_K_M (via llama-server)"
if [ -n "${GGUF}" ] && [ -x "${LLAMA_SERVER}" ]; then
  "${LLAMA_SERVER}" -m "${GGUF}" -ngl 0 -c 4096 --port "${SERVER_PORT}" \
    -t "$(nproc --all)" > /tmp/llama-server-cpu.log 2>&1 &
  SERVER_PID=$!
  trap 'kill ${SERVER_PID} 2>/dev/null || true' EXIT

  echo "Waiting for llama-server on port ${SERVER_PORT} ..."
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null 2>&1; then
      echo "ready"; break
    fi
    sleep 2
  done

  python eval/run_uzbek_eval.py --backend server \
    --url "http://127.0.0.1:${SERVER_PORT}" --tag cpu-q4km \
    || echo "[warn] Q4 Uzbek eval failed (server log: /tmp/llama-server-cpu.log)"

  kill "${SERVER_PID}" 2>/dev/null || true
  trap - EXIT
else
  echo "[skip] need a .gguf and llama-server at ${LLAMA_SERVER}"
fi

# ---------------------------------------------------------------------------
log "Building the report"
python analyze/make_report.py

log "Done -- read results/REPORT.md"
