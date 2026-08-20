#!/usr/bin/env bash
# Set up the gpu-sqb2 box (2x A40, CUDA 13.2) for the Gemma 4 26B-A4B benchmarks.
#
#   bash scripts/setup_server.sh             # everything
#   bash scripts/setup_server.sh preflight   # just check what is missing
#   bash scripts/setup_server.sh python      # venv + pip deps
#   bash scripts/setup_server.sh llamacpp    # build llama.cpp (CUDA if possible)
#   bash scripts/setup_server.sh prebuilt    # skip building: fetch CPU-only binaries
#   sudo bash scripts/setup_server.sh cuda       # install CUDA toolkit (needs root)
#   sudo bash scripts/setup_server.sh fix-perms  # undo an earlier sudo run
#   sudo bash scripts/setup_server.sh reclaim    # move root-downloaded weights to you
#
# Run every stage as YOURSELF. Only `cuda` and `fix-perms` take sudo -- the others
# refuse to run as root, because a root-created venv breaks later pip installs and
# sends ~52 GB of weights to /root/.cache. cmake and ninja come from PyPI (those
# wheels ship real binaries) rather than apt, so a locked-down server is fine.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-script}")/.." && pwd)"
VENV="${REPO_ROOT}/.venv"
LLAMA_DIR="${LLAMA_CPP_DIR:-${HOME}/llama.cpp}"
STAGE="${1:-all}"

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m  ok\033[0m   %s\n' "$*"; }
bad()  { printf '\033[1;31m  MISSING\033[0m %s\n' "$*"; }
die()  { printf '\n\033[1;31m[stop] %s\033[0m\n' "$*"; exit 1; }

# Under `set -e` with pipefail, a failing pipeline inside a command substitution
# aborts the script silently. This turns any such exit into a located one.
trap 'rc=$?; [ $rc -ne 0 ] && printf "\n\033[1;31m[error] aborted at %s line %s (exit %s)\033[0m\n  Re-run with: bash -x %s %s\n" "${BASH_SOURCE[0]:-script}" "${LINENO:-?}" "$rc" "${BASH_SOURCE[0]:-script}" "${STAGE:-}"; exit $rc' ERR

# First line of a command's output, never failing the caller. Version probes
# like `foo --version | head -1` are informational; they must not kill the run.
brief() { { "$@" 2>&1 || true; } | head -1 || true; }

# grep exits 1 when it matches nothing, which under `set -e` aborted preflight
# on any nvcc whose --version wording differs.
nvcc_release() {
  local out
  out="$( { "$1" --version 2>&1 || true; } | grep -o 'release [0-9.]*' | head -1 )" || out=""
  printf '%s' "${out:-version unknown}"
}

# Do NOT create the venv or download weights as root. Under sudo the venv ends up
# root-owned (later non-sudo pip installs fail) and ~52 GB of Hugging Face weights
# land in /root/.cache instead of your home. Only the `cuda` stage needs sudo.
refuse_sudo() {
  if [ -n "${SUDO_USER:-}" ] || [ "$(id -u)" -eq 0 ]; then
    die "Do not run the '${STAGE}' stage with sudo.

  The venv would be created root-owned and Hugging Face would cache ~52 GB of
  weights into /root/.cache instead of your home directory.

  Run it as yourself:
      bash scripts/setup_server.sh ${STAGE}

  Only the CUDA toolkit install needs root:
      sudo bash scripts/setup_server.sh cuda"
  fi
}

# Use the venv whenever it exists, so `llamacpp` works as a standalone stage.
activate_venv_if_present() {
  if [ -f "${VENV}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
  fi
}

# pip target: inside the venv if we have one, otherwise --user.
pip_install() {
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    pip install --quiet "$@"
  else
    pip install --quiet --user "$@"
    export PATH="${HOME}/.local/bin:${PATH}"
  fi
}

# ---------------------------------------------------------------------------
# CUDA toolkit discovery
# ---------------------------------------------------------------------------
# The 595.x driver reporting "CUDA 13.2" in nvidia-smi is the driver's *runtime*
# capability. It does NOT mean nvcc is installed. Building llama.cpp with
# GGML_CUDA=ON needs the actual toolkit.
find_nvcc() {
  if command -v nvcc >/dev/null 2>&1; then
    command -v nvcc
    return 0
  fi
  local candidate
  for candidate in \
      "${CUDA_HOME:-}/bin/nvcc" \
      "${CUDA_PATH:-}/bin/nvcc" \
      /usr/local/cuda/bin/nvcc \
      /usr/local/cuda-*/bin/nvcc \
      /opt/cuda/bin/nvcc; do
    if [ -x "${candidate}" ]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

find_compiler() {
  local cc
  for cc in "${CXX:-}" g++ c++ clang++; do
    if [ -n "${cc}" ] && command -v "${cc}" >/dev/null 2>&1; then
      command -v "${cc}"
      return 0
    fi
  done
  return 1
}

# ---------------------------------------------------------------------------
preflight() {
  log "Preflight: checking the toolchain"
  activate_venv_if_present
  local missing=0

  if command -v python3 >/dev/null 2>&1; then
    ok "python3  $(python3 --version 2>&1)"
  else
    bad "python3"; missing=1
  fi

  if command -v git >/dev/null 2>&1; then ok "git      $(git --version)"; else bad "git"; missing=1; fi

  if command -v cmake >/dev/null 2>&1; then
    ok "cmake    $(brief cmake --version)"
  else
    bad "cmake  -> will be installed from PyPI (no root needed)"
  fi

  if command -v ninja >/dev/null 2>&1; then
    ok "ninja    $(ninja --version)"
  elif command -v make >/dev/null 2>&1; then
    ok "make     $(brief make --version)  (ninja will be installed for speed)"
  else
    bad "ninja/make -> ninja will be installed from PyPI"
  fi

  local cxx
  if cxx="$(find_compiler)"; then
    ok "C++      ${cxx}  ($(brief "${cxx}" --version))"
  else
    bad "g++/clang++  -- a C++ compiler CANNOT be pip-installed."
    warn "Ask the admin for build-essential, or load a module: 'module avail gcc'"
    missing=1
  fi

  local nvcc
  if nvcc="$(find_nvcc)"; then
    ok "nvcc     ${nvcc}  ($(nvcc_release "${nvcc}"))"
  else
    bad "nvcc  -- CUDA toolkit not found"
    warn "nvidia-smi showing 'CUDA 13.2' is the DRIVER version, not a toolkit."
    warn "Only llama.cpp's CUDA backend needs this. transformers gets CUDA from"
    warn "its pip wheels, so Track B and both Uzbek evals work without it."
    warn "To get Track A's GPU half, in order of preference:"
    warn "  1. module avail cuda && module load cuda"
    warn "  2. sudo bash scripts/setup_server.sh cuda    (installs toolkit only,"
    warn "     never the driver, so other users' jobs are untouched)"
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    ok "nvidia-smi present"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true
  else
    bad "nvidia-smi"
  fi

  echo
  if [ "${missing}" -eq 1 ]; then
    warn "Something essential is missing -- see above."
    return 1
  fi
  log "Preflight OK"
}

# ---------------------------------------------------------------------------
# A previous sudo run leaves root-owned files behind, and the failure that
# produces later is an opaque "Permission denied" from venv/pip. Catch it here
# and say what to do instead.
check_writable() {
  local stale=""

  note_stale() {
    local path="$1"
    stale+=$'\n'"    ${path}  (owner: $(stat -c '%U' "${path}" 2>/dev/null || echo '?'))"
  }

  [ -e "${VENV}" ] && [ ! -w "${VENV}" ] && note_stale "${VENV}"
  [ ! -w "${REPO_ROOT}" ] && note_stale "${REPO_ROOT}"
  [ -e "${LLAMA_DIR}" ] && [ ! -w "${LLAMA_DIR}" ] && note_stale "${LLAMA_DIR}"
  [ -e "${HOME}/.cache/huggingface" ] && [ ! -w "${HOME}/.cache/huggingface" ] \
    && note_stale "${HOME}/.cache/huggingface"

  if [ -n "${stale}" ]; then
    die "These paths are not writable by $(id -un):
${stale}

  They were created by an earlier sudo run. Hand them back:
      sudo bash scripts/setup_server.sh fix-perms

  Then re-run this stage as yourself (no sudo):
      bash scripts/setup_server.sh ${STAGE}"
  fi
  return 0
}

setup_python() {
  check_writable

  # A root-owned venv from a previous sudo run is unusable; replace it.
  if [ -d "${VENV}" ] && [ ! -w "${VENV}/bin" ] 2>/dev/null; then
    die "Existing venv at ${VENV} is not writable. Run: sudo bash scripts/setup_server.sh fix-perms"
  fi

  # Rebuilding the venv you are standing inside leaves the shell pointing at
  # paths that are about to be replaced.
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    warn "A venv is active in this shell (${VIRTUAL_ENV})."
    warn "Run 'deactivate' first if anything behaves oddly after this."
  fi

  log "Creating virtualenv at ${VENV}"
  python3 -m venv "${VENV}"
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  pip install --upgrade pip wheel

  log "Installing PyTorch + torchvision (CUDA 12.8 wheels work on the 595.x driver)"
  # torchvision is not optional: Gemma 4 is multimodal and its processor imports it.
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

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

# ---------------------------------------------------------------------------
ensure_build_tools() {
  activate_venv_if_present

  if ! command -v cmake >/dev/null 2>&1; then
    log "cmake not found -- installing it from PyPI (ships a real binary, no root)"
    pip_install cmake
    hash -r
    command -v cmake >/dev/null 2>&1 || {
      warn "pip installed cmake but it is not on PATH."
      warn "Try: export PATH=\"\$(python3 -c 'import sysconfig;print(sysconfig.get_path(\"scripts\"))'):\$PATH\""
      return 1
    }
  fi
  ok "cmake  $(brief cmake --version)"

  if ! command -v ninja >/dev/null 2>&1 && ! command -v make >/dev/null 2>&1; then
    log "No ninja or make -- installing ninja from PyPI"
    pip_install ninja
    hash -r
  fi

  find_compiler >/dev/null 2>&1 || {
    warn "No C++ compiler found. This is the one thing pip cannot provide."
    warn "Options:  module load gcc   |   ask the admin for build-essential"
    warn "Meanwhile you can use prebuilt CPU binaries:"
    warn "    bash scripts/setup_server.sh prebuilt"
    return 1
  }
}

setup_llamacpp() {
  ensure_build_tools || return 1

  log "Fetching llama.cpp into ${LLAMA_DIR}"
  if [ ! -d "${LLAMA_DIR}/.git" ]; then
    git clone https://github.com/ggml-org/llama.cpp "${LLAMA_DIR}"
  else
    git -C "${LLAMA_DIR}" pull --ff-only
  fi

  local cuda_flags=()
  local nvcc
  if nvcc="$(find_nvcc)"; then
    local cuda_root
    cuda_root="$(dirname "$(dirname "${nvcc}")")"
    log "CUDA toolkit found at ${cuda_root} -- building with GPU support"
    # A40 is Ampere = sm_86. One arch keeps the compile to minutes, not an hour.
    cuda_flags=(
      -DGGML_CUDA=ON
      -DCMAKE_CUDA_ARCHITECTURES=86
      -DCMAKE_CUDA_COMPILER="${nvcc}"
    )
    export CUDA_HOME="${cuda_root}"
    export PATH="${cuda_root}/bin:${PATH}"
  else
    warn "No CUDA toolkit -- building CPU-only."
    warn "Track A's GPU half will be unavailable; the CPU half and Track B (transformers) still work."
    warn "To get the GPU half: 'module load cuda' then re-run this stage."
    cuda_flags=(-DGGML_CUDA=OFF)
  fi

  local generator=()
  command -v ninja >/dev/null 2>&1 && generator=(-G Ninja)

  log "Configuring"
  cmake -S "${LLAMA_DIR}" -B "${LLAMA_DIR}/build" \
    "${generator[@]}" \
    "${cuda_flags[@]}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=OFF

  log "Building (this takes a few minutes)"
  cmake --build "${LLAMA_DIR}/build" --config Release -j "$(nproc)"

  log "Built binaries"
  ls -la "${LLAMA_DIR}/build/bin/" | grep -E 'llama-(bench|server|cli)' || true
  echo
  echo "Add to your shell profile:"
  echo "  export PATH=\"${LLAMA_DIR}/build/bin:\$PATH\""
}

# ---------------------------------------------------------------------------
# Escape hatch: no compiler at all. Prebuilt CPU binaries unblock the CPU track.
# (ggml-org publishes no Linux CUDA build, so GPU still needs a source build.)
setup_prebuilt() {
  log "Fetching prebuilt llama.cpp binaries (CPU-only)"
  local dest="${LLAMA_DIR}/build/bin"
  mkdir -p "${dest}"

  local tag url tmp
  tag="$(curl -sfL https://api.github.com/repos/ggml-org/llama.cpp/releases/latest \
         | grep -o '"tag_name": *"[^"]*"' | head -1 | cut -d'"' -f4)"
  [ -n "${tag}" ] || { warn "Could not resolve the latest release tag"; return 1; }

  url="https://github.com/ggml-org/llama.cpp/releases/download/${tag}/llama-${tag}-bin-ubuntu-x64.tar.gz"
  tmp="$(mktemp -d)"
  log "Downloading ${tag}"
  curl -fL --progress-bar "${url}" -o "${tmp}/llama.tar.gz"
  tar -xzf "${tmp}/llama.tar.gz" -C "${tmp}"

  find "${tmp}" -type f \( -name 'llama-*' -o -name '*.so*' \) -exec cp -f {} "${dest}/" \;
  chmod +x "${dest}"/llama-* 2>/dev/null || true
  rm -rf "${tmp}"

  log "Installed to ${dest}"
  ls -la "${dest}" | grep -E 'llama-(bench|server|cli)' || true
  warn "These are CPU-only. Run Track A with --devices cpu; use Track B for GPU numbers."
}

# ---------------------------------------------------------------------------
# Install the CUDA toolkit (needs root). Only llama.cpp's CUDA backend requires
# this -- transformers gets CUDA from its pip wheels and never needs nvcc.
setup_cuda() {
  if [ "$(id -u)" -ne 0 ]; then
    die "This stage installs system packages and needs root:
      sudo bash scripts/setup_server.sh cuda"
  fi

  local version="${CUDA_VERSION:-13-2}"
  local dotted="${version//-/.}"

  log "Installing cuda-toolkit-${version} from NVIDIA's Ubuntu 24.04 repository"
  warn "Installing the TOOLKIT ONLY (cuda-toolkit-${version})."
  warn "Deliberately NOT the 'cuda' metapackage -- that one pulls a new DRIVER,"
  warn "which would disrupt the other users' jobs running on these A40s."
  echo

  if command -v lsb_release >/dev/null 2>&1 && [ "$(lsb_release -rs)" != "24.04" ]; then
    warn "Expected Ubuntu 24.04, found $(lsb_release -rs). Check the repo URL suits it."
  fi

  local tmp keyring
  tmp="$(mktemp -d)"
  keyring="cuda-keyring_1.1-1_all.deb"
  curl -fL -o "${tmp}/${keyring}" \
    "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/${keyring}"
  dpkg -i "${tmp}/${keyring}"
  rm -rf "${tmp}"

  apt-get update
  apt-get install -y "cuda-toolkit-${version}"

  if [ -x "/usr/local/cuda-${dotted}/bin/nvcc" ]; then
    log "Installed $(nvcc_release "/usr/local/cuda-${dotted}/bin/nvcc")"
    echo
    echo "Now, as yourself (NOT root):"
    echo "  export CUDA_HOME=/usr/local/cuda-${dotted}"
    echo "  export PATH=\"\$CUDA_HOME/bin:\$PATH\""
    echo "  bash scripts/setup_server.sh llamacpp"
  else
    warn "nvcc not found at /usr/local/cuda-${dotted}/bin/nvcc"
    warn "Check: ls /usr/local/ | grep cuda"
  fi
}

# Undo the damage from an earlier sudo run: hand every path this project uses
# back to the real user. Needs root (only root can chown away from root).
fix_perms() {
  if [ "$(id -u)" -ne 0 ]; then
    die "This stage needs root in order to chown files away from root:
      sudo bash scripts/setup_server.sh fix-perms"
  fi

  local target_user="${SUDO_USER:-}"
  if [ -z "${target_user}" ] || [ "${target_user}" = "root" ]; then
    die "Cannot tell which user to hand ownership to.
  Invoke this via sudo from your normal account, not as a root login shell:
      sudo bash scripts/setup_server.sh fix-perms"
  fi

  local target_group target_home
  target_group="$(id -gn "${target_user}" 2>/dev/null)" || target_group="${target_user}"
  target_home="$(getent passwd "${target_user}" 2>/dev/null | cut -d: -f6)" || target_home=""
  [ -n "${target_home}" ] || target_home="/home/${target_user}"
  [ -d "${target_home}" ] || die "Home directory for ${target_user} not found at ${target_home}"

  log "Repairing ownership for ${target_user}:${target_group}"

  local path changed=0 venv_was_root=0
  # ~/.cache itself matters: if the parent is root-owned, creating
  # ~/.cache/huggingface fails even when the subdirectory does not exist yet.
  # Missing paths are simply skipped -- absent is not the same as broken.
  for path in \
      "${REPO_ROOT}" \
      "${target_home}/llama.cpp" \
      "${target_home}/.cache" \
      "${target_home}/.cache/huggingface" \
      "${target_home}/.cache/pip" \
      "${target_home}/.local"; do
    if [ ! -e "${path}" ]; then
      printf '  --   absent, nothing to do: %s\n' "${path}"
      continue
    fi
    local owner
    owner="$(stat -c '%U' "${path}" 2>/dev/null)" || owner="?"
    if [ "${owner}" = "${target_user}" ]; then
      ok "already ${target_user}: ${path}"
    else
      chown -R "${target_user}:${target_group}" "${path}"
      ok "chowned (was ${owner}): ${path}"
      changed=1
      [ "${path}" = "${REPO_ROOT}" ] && venv_was_root=1
    fi
  done

  # Only discard a venv that root actually created. A healthy user-owned venv
  # must be left alone: deleting it while it is activated in some shell leaves
  # $VIRTUAL_ENV and $PATH pointing at nothing, and every later python/pip call
  # falls through to the system interpreter.
  if [ -d "${REPO_ROOT}/.venv" ]; then
    local venv_owner
    venv_owner="$(stat -c '%U' "${REPO_ROOT}/.venv" 2>/dev/null)" || venv_owner="?"
    if [ "${venv_owner}" = "root" ] || [ "${venv_was_root}" -eq 1 ]; then
      warn "Removing ${REPO_ROOT}/.venv -- it was created by root, and chown alone"
      warn "does not reliably repair a venv. Rebuild it with:"
      warn "    bash scripts/setup_server.sh python"
      warn "If it is active in any shell, run 'deactivate' there first."
      rm -rf "${REPO_ROOT}/.venv"
      changed=1
    else
      ok "venv is ${venv_owner}-owned and healthy: left untouched"
    fi
  fi

  # Create ~/.cache while we still have root. If the user could not create it
  # themselves, printing advice does not help; making it and handing it over does.
  if [ ! -e "${target_home}/.cache" ]; then
    if mkdir -p "${target_home}/.cache" 2>/dev/null; then
      chown "${target_user}:${target_group}" "${target_home}/.cache"
      ok "created ${target_home}/.cache and gave it to ${target_user}"
      changed=1
    else
      warn "Could not create ${target_home}/.cache even as root."
      warn "Check the mount: $(findmnt -no TARGET,OPTIONS --target "${target_home}" 2>/dev/null || echo '?')"
    fi
  fi

  # Weights in /root/.cache are stranded: not the user's, and not where the
  # scripts look. Report them; `reclaim` moves them.
  if [ -d /root/.cache/huggingface ]; then
    local sz
    sz="$(du -sh /root/.cache/huggingface 2>/dev/null | cut -f1)" || sz="?"
    warn "/root/.cache/huggingface exists (${sz}) -- weights downloaded as root."
    warn "Move them to ${target_user} instead of downloading again:"
    warn "    sudo bash scripts/setup_server.sh reclaim"
  fi

  echo
  if [ "${changed}" -eq 0 ]; then
    log "Nothing needed repairing -- no root-owned files found."
    echo "  Permissions are not your problem. Carry on as yourself:"
    echo "      bash scripts/download_models.sh"
  else
    log "Done. Now run as yourself, without sudo:"
    echo "      bash scripts/setup_server.sh"
  fi
}

# Move weights that were downloaded as root into the user's cache, rather than
# re-downloading tens of GB. Needs root: only root can move out of /root.
reclaim() {
  if [ "$(id -u)" -ne 0 ]; then
    die "This stage moves files out of /root and needs root:
      sudo bash scripts/setup_server.sh reclaim"
  fi

  local target_user="${SUDO_USER:-}"
  if [ -z "${target_user}" ] || [ "${target_user}" = "root" ]; then
    die "Cannot tell which user to hand the weights to.
      Invoke via sudo from your normal account."
  fi

  local target_group target_home src dst
  target_group="$(id -gn "${target_user}" 2>/dev/null)" || target_group="${target_user}"
  target_home="$(getent passwd "${target_user}" 2>/dev/null | cut -d: -f6)" || target_home=""
  [ -n "${target_home}" ] || target_home="/home/${target_user}"

  src="/root/.cache/huggingface"
  dst="${target_home}/.cache/huggingface"

  [ -d "${src}" ] || die "Nothing to reclaim: ${src} does not exist."

  local sz
  sz="$(du -sh "${src}" 2>/dev/null | cut -f1)" || sz="?"
  log "Reclaiming ${sz} from ${src}"

  mkdir -p "${target_home}/.cache"
  chown "${target_user}:${target_group}" "${target_home}/.cache"

  if [ -d "${dst}" ]; then
    # Never clobber an existing cache -- merge, letting the user's copy win.
    warn "${dst} already exists; merging without overwriting what is there."
    cp -an "${src}/." "${dst}/" 2>/dev/null || true
    rm -rf "${src}"
  else
    mv "${src}" "${dst}"
  fi

  chown -R "${target_user}:${target_group}" "${dst}"
  ok "moved to ${dst}, owned by ${target_user}:${target_group}"

  echo
  log "Done. As yourself, verify the snapshot is visible:"
  echo "      ls ${dst}/hub"
  echo "      bash scripts/run_all.sh"
}

# ---------------------------------------------------------------------------
case "${STAGE}" in
  # `|| exit` keeps the ERR trap quiet: preflight returning 1 means "something
  # is missing", which is a reportable result, not a crash.
  preflight)  preflight || exit $? ;;
  cuda)       setup_cuda ;;
  fix-perms)  fix_perms ;;
  reclaim)    reclaim ;;
  python)    refuse_sudo; setup_python ;;
  llamacpp)  refuse_sudo; setup_llamacpp ;;
  prebuilt)  refuse_sudo; setup_prebuilt ;;
  all)
    refuse_sudo
    preflight || warn "Continuing despite preflight warnings"
    setup_python
    setup_llamacpp || warn "llama.cpp build failed -- try: bash scripts/setup_server.sh prebuilt"
    ;;
  *) echo "Unknown stage: ${STAGE} (use: preflight | fix-perms | reclaim | cuda | python | llamacpp | prebuilt | all)"; exit 1 ;;
esac

log "Setup complete"
cat <<EOF

Next:
  1. Download weights:   bash scripts/download_models.sh
  2. Run everything:     bash scripts/run_all.sh
EOF
