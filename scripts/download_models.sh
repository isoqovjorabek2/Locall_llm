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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-script}")/.." && pwd)"
MODELS_DIR="${REPO_ROOT}/models"
HF_MODEL="google/gemma-4-26B-A4B-it"
GGUF_REPO="${GGUF_REPO:-ggml-org/gemma-4-26b-a4b-it-GGUF}"
GGUF_QUANT="${GGUF_QUANT:-Q4_K_M}"
STAGE="${1:-all}"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m[stop] %s\033[0m\n' "$*"; exit 1; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }

# `set -e` aborts on any unchecked failure, including inside a command
# substitution, and does it without printing anything. This turns every such
# exit into a located one instead of the script appearing to "just end".
trap 'rc=$?; [ $rc -ne 0 ] && printf "\n\033[1;31m[error] aborted at %s line %s (exit %s)\033[0m\n  The command on that line failed. Re-run with: bash -x scripts/download_models.sh\n" "${BASH_SOURCE[0]:-script}" "${LINENO:-?}" "$rc"; exit $rc' ERR

# Under sudo these ~67 GB land in /root/.cache, where nothing else looks.
if [ -n "${SUDO_USER:-}" ] || [ "$(id -u)" -eq 0 ]; then
  die "Do not download with sudo -- ~67 GB would go to /root/.cache instead of
  your home directory, and the benchmarks would not find it.

  Run as yourself:
      bash scripts/download_models.sh"
fi

# A "(.venv)" prompt does NOT prove the venv exists -- if it was deleted while
# still activated (fix-perms does exactly that), the prompt and PATH survive but
# every tool silently falls through to the system python. That is what produces
# pip's "externally-managed-environment" error on Debian/Ubuntu.
VENV_PY="${REPO_ROOT}/.venv/bin/python"

if [ ! -x "${VENV_PY}" ]; then
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    die "Your shell shows an active venv (${VIRTUAL_ENV}) but ${VENV_PY}
  does not exist -- the venv was deleted while still activated, so pip and
  python are silently resolving to the SYSTEM python. That is the
  'externally-managed-environment' error.

  Leave the stale environment and rebuild:
      deactivate
      bash scripts/setup_server.sh python
      bash scripts/download_models.sh"
  fi
  die "No virtualenv at ${REPO_ROOT}/.venv
      Run first: bash scripts/setup_server.sh"
fi

# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"

# Always go through the venv's interpreter explicitly. `pip` alone can resolve
# to /usr/bin/pip if PATH is stale, and on Ubuntu 24.04 that is PEP 668-blocked.
if ! "${VENV_PY}" -c "import huggingface_hub" >/dev/null 2>&1; then
  log "Installing huggingface_hub into the venv"
  "${VENV_PY}" -m pip install --quiet "huggingface_hub[cli]"
fi

# Same reason: call the venv's own hf binary by absolute path rather than
# trusting PATH. (`python -m huggingface_hub.cli.hf` is not a reliable entry
# point -- the module has no __main__ guard in every release.)
HF_BIN="${REPO_ROOT}/.venv/bin/hf"
if [ ! -x "${HF_BIN}" ]; then
  die "huggingface_hub is installed but ${HF_BIN} is missing.
      Reinstall the CLI extra:
          ${VENV_PY} -m pip install -U 'huggingface_hub[cli]'"
fi
hf() { "${HF_BIN}" "$@"; }

HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"

# The download fails deep inside a thread pool, and the traceback that surfaces
# points at hf_thread_map rather than the cause. Check the usual causes here.
preflight_download() {
  local need_gib="$1"

  log "Checking space and permissions before downloading"

  # 1. Ownership. An earlier sudo run leaves a root-owned cache, and the
  #    resulting PermissionError appears only at the bottom of a long traceback.
  # Show what mkdir actually said. Swallowing its stderr turns a precise error
  # ("Permission denied", "Disk quota exceeded", "Read-only file system") into
  # a guess, and the reported "parent" is misleading when the parent is also
  # missing -- dirname of a missing path is not the thing that blocked you.
  local mkerr=""
  if ! mkerr="$(mkdir -p "${HF_CACHE}" 2>&1)"; then
    local probe="${HF_CACHE}" existing=""
    while [ "${probe}" != "/" ] && [ -n "${probe}" ]; do
      if [ -e "${probe}" ]; then existing="${probe}"; break; fi
      probe="$(dirname "${probe}")"
    done
    die "Cannot create ${HF_CACHE}

  mkdir said:        ${mkerr:-(no message)}
  deepest existing:  ${existing:-/}  (owner $(stat -c '%U' "${existing:-/}" 2>/dev/null || echo '?'), \
mode $(stat -c '%a' "${existing:-/}" 2>/dev/null || echo '?'))
  you are:           $(id -un) ($(id -u)), groups: $(id -Gn 2>/dev/null || echo '?')

  If that directory is root-owned:
      sudo bash scripts/setup_server.sh fix-perms
  If it is a quota or read-only mount, put the cache somewhere you can write:
      export HF_HOME=/some/writable/path/huggingface"
  fi

  local owner="?"
  owner="$(stat -c '%U' "${HF_CACHE}" 2>/dev/null)" || owner="?"
  if [ ! -w "${HF_CACHE}" ]; then
    die "${HF_CACHE} is not writable by $(id -un) (owner: ${owner}).

  An earlier sudo run created it as root. Hand it back:
      sudo bash scripts/setup_server.sh fix-perms"
  fi
  echo "  cache:   ${HF_CACHE}  (owner ${owner}, writable)"

  # 2. Disk. 52 GB of safetensors plus 15 GB of GGUF, and HF needs room for
  #    partial .incomplete files alongside the finished blobs.
  #    df -Pk is POSIX and portable; -BG is not, and its failure under
  #    `set -e` + pipefail used to abort this script with no message at all.
  local avail_kb="" avail_gib="" mount=""
  avail_kb="$(df -Pk "${HF_CACHE}" 2>/dev/null | awk 'NR==2 {print $4}')" || avail_kb=""
  mount="$(df -Pk "${HF_CACHE}" 2>/dev/null | awk 'NR==2 {print $6}')" || mount=""

  if [ -n "${avail_kb}" ] && [ "${avail_kb}" -eq "${avail_kb}" ] 2>/dev/null; then
    avail_gib=$(( avail_kb / 1048576 ))
    echo "  disk:    ${avail_gib} GiB free on ${mount:-?}"
    if [ "${avail_gib}" -lt "${need_gib}" ]; then
      die "Not enough disk: ${avail_gib} GiB free, need about ${need_gib} GiB.

  Free space, or point the cache at a bigger volume:
      export HF_HOME=/path/with/room/huggingface
      bash scripts/download_models.sh"
    fi
  else
    warn "Could not read free space for ${HF_CACHE}; continuing without the check."
  fi

  # 3. Reachability. Fail here rather than inside a worker thread.
  if curl -sfI --max-time 20 https://huggingface.co >/dev/null 2>&1; then
    echo "  network: huggingface.co reachable"
  elif curl -sfI --max-time 20 "${HF_ENDPOINT:-https://huggingface.co}" >/dev/null 2>&1; then
    echo "  network: ${HF_ENDPOINT} reachable"
  else
    die "Cannot reach huggingface.co.

  If this box needs a proxy:
      export HTTPS_PROXY=http://proxy:port
  Or use a mirror:
      export HF_ENDPOINT=https://hf-mirror.com"
  fi
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
