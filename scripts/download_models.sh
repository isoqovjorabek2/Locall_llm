#!/usr/bin/env bash
# Fetch Gemma 4 26B-A4B weights: bf16 safetensors (GPU track) + Q4_K_M GGUF (CPU track).
#
#   bash scripts/download_models.sh          # both
#   bash scripts/download_models.sh hf       # safetensors only  (~52 GB)
#   bash scripts/download_models.sh gguf     # GGUF only         (~15 GB)
#
# Weights go to $HF_HOME (default ~/.cache/huggingface) and ./models.
# Check you have the disk first: df -h ~

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${REPO_ROOT}/models"
HF_MODEL="google/gemma-4-26B-A4B-it"
GGUF_REPO="${GGUF_REPO:-ggml-org/gemma-4-26b-a4b-it-GGUF}"
GGUF_QUANT="${GGUF_QUANT:-Q4_K_M}"
STAGE="${1:-all}"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

if [ -d "${REPO_ROOT}/.venv" ]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

command -v hf >/dev/null 2>&1 || pip install --quiet "huggingface_hub[cli]"

fetch_hf() {
  log "Downloading ${HF_MODEL} (bf16 safetensors, ~52 GB)"
  echo "Disk free on \$HOME:"; df -h "${HOME}" | tail -1
  hf download "${HF_MODEL}" --exclude "*.gguf"
  echo "Cached under: ${HF_HOME:-${HOME}/.cache/huggingface}"
}

fetch_gguf() {
  log "Downloading ${GGUF_REPO} ${GGUF_QUANT} GGUF"
  mkdir -p "${MODELS_DIR}"
  hf download "${GGUF_REPO}" \
    --include "*${GGUF_QUANT}*.gguf" \
    --local-dir "${MODELS_DIR}"
  echo
  echo "GGUF files:"
  find "${MODELS_DIR}" -name '*.gguf' -printf '%p  %s bytes\n' 2>/dev/null || ls -la "${MODELS_DIR}"
}

case "${STAGE}" in
  hf)   fetch_hf ;;
  gguf) fetch_gguf ;;
  all)  fetch_hf; fetch_gguf ;;
  *)    echo "Unknown stage: ${STAGE} (use: hf | gguf | all)"; exit 1 ;;
esac

log "Downloads complete"
