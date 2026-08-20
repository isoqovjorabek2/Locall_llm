#!/usr/bin/env bash
# Run the full comparison end to end and build results/REPORT.md.
#
#   bash scripts/run_all.sh
#
# Roughly: Track A (llama.cpp CPU+GPU) ~20-40 min, Track B (transformers bf16)
# ~15 min plus a slow first load, Uzbek eval ~10-20 min per configuration.
# Run it under tmux -- do not let an SSH drop kill a two-hour job.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MODELS_DIR="${REPO_ROOT}/models"
LLAMA_DIR="${LLAMA_CPP_DIR:-${HOME}/llama.cpp}"
LLAMA_BENCH="${LLAMA_BENCH:-${LLAMA_DIR}/build/bin/llama-bench}"
LLAMA_SERVER="${LLAMA_SERVER:-${LLAMA_DIR}/build/bin/llama-server}"
SERVER_PORT="${SERVER_PORT:-8080}"

# GPU 1 was the emptier A40 in the nvidia-smi snapshot; override if that changed.
GPU_FOR_LLAMACPP="${GPU_FOR_LLAMACPP:-1}"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

if [ -d "${REPO_ROOT}/.venv" ]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

GGUF="$(find "${MODELS_DIR}" -name '*.gguf' 2>/dev/null | head -1 || true)"

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
