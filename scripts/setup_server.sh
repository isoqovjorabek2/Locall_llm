#!/usr/bin/env bash
# Set up the gpu-sqb2 box (2x A40, CUDA 13.2) for the Gemma 4 26B-A4B benchmarks.
#
#   bash scripts/setup_server.sh            # everything
#   bash scripts/setup_server.sh python     # just the venv + pip deps
#   bash scripts/setup_server.sh llamacpp   # just build llama.cpp with CUDA
#
# Nothing here needs root. Everything lands under this repo or ~/.cache.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${REPO_ROOT}/.venv"
LLAMA_DIR="${LLAMA_CPP_DIR:-${HOME}/llama.cpp}"
STAGE="${1:-all}"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

setup_python() {
  log "Creating virtualenv at ${VENV}"
  python3 -m venv "${VENV}"
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  pip install --upgrade pip wheel

  log "Installing PyTorch (CUDA 12.8 wheels work on the 595.x driver)"
  pip install torch --index-url https://download.pytorch.org/whl/cu128

  log "Installing the rest of requirements.txt"
  pip install -r "${REPO_ROOT}/requirements.txt"

  log "Verifying GPU visibility"
  python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name}  {p.total_memory / 1024**3:.1f} GiB  sm_{p.major}{p.minor}")
PY
}

setup_llamacpp() {
  log "Building llama.cpp with CUDA at ${LLAMA_DIR}"
  if [ ! -d "${LLAMA_DIR}/.git" ]; then
    git clone https://github.com/ggml-org/llama.cpp "${LLAMA_DIR}"
  else
    git -C "${LLAMA_DIR}" pull --ff-only
  fi

  # A40 is Ampere = sm_86. Building only that arch keeps compile time sane.
  cmake -S "${LLAMA_DIR}" -B "${LLAMA_DIR}/build" \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=ON

  cmake --build "${LLAMA_DIR}/build" --config Release -j "$(nproc)"

  log "Built binaries"
  ls -la "${LLAMA_DIR}/build/bin/" | grep -E 'llama-(bench|server|cli)' || true
  echo
  echo "Add to your shell profile:"
  echo "  export PATH=\"${LLAMA_DIR}/build/bin:\$PATH\""
}

case "${STAGE}" in
  python)   setup_python ;;
  llamacpp) setup_llamacpp ;;
  all)      setup_python; setup_llamacpp ;;
  *)        echo "Unknown stage: ${STAGE} (use: python | llamacpp | all)"; exit 1 ;;
esac

log "Setup complete"
cat <<EOF

Next:
  1. Download weights:   bash scripts/download_models.sh
  2. Run everything:     bash scripts/run_all.sh
EOF
