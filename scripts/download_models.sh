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
die() { printf '\n\033[1;31m[stop] %s\033[0m\n' "$*"; exit 1; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }

# Under sudo these ~67 GB land in /root/.cache, where nothing else looks.
if [ -n "${SUDO_USER:-}" ] || [ "$(id -u)" -eq 0 ]; then
  die "Do not download with sudo -- ~67 GB would go to /root/.cache instead of
  your home directory, and the benchmarks would not find it.

  Run as yourself:
      bash scripts/download_models.sh"
fi

if [ -d "${REPO_ROOT}/.venv" ]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

command -v hf >/dev/null 2>&1 || pip install --quiet "huggingface_hub[cli]"

HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"

# The download fails deep inside a thread pool, and the traceback that surfaces
# points at hf_thread_map rather than the cause. Check the usual causes here.
preflight_download() {
  local need_gib="$1"

  log "Checking space and permissions before downloading"

  # 1. Ownership. An earlier sudo run leaves a root-owned cache, and the
  #    resulting PermissionError appears only at the bottom of a long traceback.
  mkdir -p "${HF_CACHE}" 2>/dev/null || true
  if [ -e "${HF_CACHE}" ]; then
    local owner
    owner="$(stat -c '%U' "${HF_CACHE}" 2>/dev/null || echo '?')"
    if [ ! -w "${HF_CACHE}" ]; then
      die "${HF_CACHE} is not writable by $(id -un) (owner: ${owner}).

  An earlier sudo run created it as root. Hand it back:
      sudo bash scripts/setup_server.sh fix-perms"
    fi
    echo "  cache:  ${HF_CACHE}  (owner ${owner}, writable)"
  fi

  # 2. Disk. 52 GB of safetensors plus 15 GB of GGUF, and HF needs room for
  #    partial .incomplete files alongside the finished blobs.
  local avail_gib
  avail_gib="$(df -PBG "${HF_CACHE}" 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4}')"
  if [ -n "${avail_gib}" ]; then
    echo "  disk:   ${avail_gib} GiB free on $(df -P "${HF_CACHE}" | awk 'NR==2 {print $6}')"
    if [ "${avail_gib}" -lt "${need_gib}" ]; then
      die "Not enough disk: ${avail_gib} GiB free, need about ${need_gib} GiB.

  Free space, or point the cache at a bigger volume:
      export HF_HOME=/path/with/room/huggingface
      bash scripts/download_models.sh"
    fi
  fi

  # 3. Reachability. Fail here rather than inside a worker thread.
  if ! curl -sfI --max-time 20 https://huggingface.co >/dev/null 2>&1; then
    die "Cannot reach huggingface.co.

  If this box needs a proxy:
      export HTTPS_PROXY=http://proxy:port
  Or use a mirror:
      export HF_ENDPOINT=https://hf-mirror.com"
  fi
  echo "  network: huggingface.co reachable"
  echo
}

# hf download resumes automatically, so a retry after a dropped connection
# picks up where it stopped rather than restarting 52 GB.
retry_download() {
  local attempt
  for attempt in 1 2 3; do
    if "$@"; then
      return 0
    fi
    if [ "${attempt}" -lt 3 ]; then
      warn "Download attempt ${attempt} failed. Retrying in 15s (it resumes, "
      warn "so nothing already fetched is downloaded twice)."
      sleep 15
    fi
  done
  die "Download failed three times. The real cause is the LAST line of the
  traceback above, below the hf_thread_map frames. Common ones:

    OSError: [Errno 28] No space left on device   -> free disk, or set HF_HOME
    PermissionError                               -> sudo bash scripts/setup_server.sh fix-perms
    401 / GatedRepoError                          -> hf auth login
    ConnectionError / timeout                     -> set HTTPS_PROXY or HF_ENDPOINT

  To reduce concurrency (helps on flaky links):
      hf download ${HF_MODEL} --exclude '*.gguf' --max-workers 2"
}

fetch_hf() {
  preflight_download 60
  log "Downloading ${HF_MODEL} (bf16 safetensors, ~52 GB)"
  retry_download hf download "${HF_MODEL}" --exclude "*.gguf"
  echo "Cached under: ${HF_CACHE}"
}

fetch_gguf() {
  preflight_download 20
  log "Downloading ${GGUF_REPO} ${GGUF_QUANT} GGUF (~15 GB)"
  mkdir -p "${MODELS_DIR}"
  retry_download hf download "${GGUF_REPO}" \
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
