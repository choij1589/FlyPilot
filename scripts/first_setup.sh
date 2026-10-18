#!/usr/bin/env bash
# Execute this installer; use the root setup.sh for interactive Zsh activation.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root_prefix="${MAMBA_ROOT_PREFIX:-${HOME}/micromamba}"
backend=auto
torch_version=2.7.1

usage() {
    printf '%s\n' 'Usage: bash scripts/first_setup.sh [--backend auto|cpu|mps|cuda] [--root-prefix /absolute/path]' \
        'Default: Apple Silicon MPS; Linux x86-64 CPU. CUDA uses PyTorch CUDA 12.6 wheels.' \
        'Creates/reuses the flypilot environment. Does not edit shell startup files.'
}
fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend|--root-prefix)
            [[ $# -ge 2 && -n "$2" ]] || fail "$1 requires a value"
            if [[ "$1" == --backend ]]; then backend="$2"; else root_prefix="$2"; fi
            shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown argument: $1" ;;
    esac
done
[[ "$root_prefix" == /* && "$root_prefix" != / ]] || fail 'Root prefix must be an absolute directory other than /.'
case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) platform=osx-arm64; [[ "$backend" != auto ]] || backend=mps ;;
    Linux-x86_64) platform=linux-64; [[ "$backend" != auto ]] || backend=cpu ;;
    *) fail 'Supported platforms: Apple Silicon macOS and x86-64 Linux (glibc).' ;;
esac
case "$platform:$backend" in
    osx-arm64:mps|osx-arm64:cpu|linux-64:cpu|linux-64:cuda) ;;
    *) fail "Unsupported backend $backend for $platform" ;;
esac

export MAMBA_ROOT_PREFIX="$root_prefix"
export PYTHONNOUSERSITE=1
# Avoid an inherited Python module/environment contaminating this installation.
unset PYTHONPATH PYTHONHOME
if [[ -n "${MAMBA_EXE:-}" && -x "$MAMBA_EXE" ]]; then
    mamba_exe="$MAMBA_EXE"
elif command -v micromamba >/dev/null 2>&1; then
    mamba_exe="$(command -v micromamba)"
elif [[ -x "$root_prefix/bin/micromamba" ]]; then
    mamba_exe="$root_prefix/bin/micromamba"
else
    command -v curl >/dev/null || fail 'curl is required to download micromamba.'
    command -v tar >/dev/null || fail 'tar with bzip2 support is required.'
    download_dir="$(mktemp -d)"
    # Keep downloads on failure for diagnosis; never remove an existing environment.
    printf 'Downloading micromamba 2.3.2 for %s into %s\n' "$platform" "$download_dir"
    curl --fail --location --retry 3 "https://micro.mamba.pm/api/micromamba/$platform/2.3.2" -o "$download_dir/micromamba.tar.bz2"
    tar -xjf "$download_dir/micromamba.tar.bz2" -C "$download_dir" bin/micromamba
    mkdir -p "$root_prefix/bin"
    install -m 755 "$download_dir/bin/micromamba" "$root_prefix/bin/micromamba"
    mamba_exe="$root_prefix/bin/micromamba"
fi

env_prefix="$root_prefix/envs/flypilot"
if [[ -d "$env_prefix/conda-meta" ]]; then
    "$mamba_exe" run -p "$env_prefix" python -c \
        'import sys; assert sys.version_info[:2] == (3, 11), "Existing flypilot requires Python 3.11; select a different root prefix."'
elif [[ -e "$env_prefix" ]]; then
    fail "$env_prefix already exists but is not a conda environment; choose another root prefix."
else
    "$mamba_exe" create --yes --override-channels -c conda-forge -p "$env_prefix" python=3.11.13 pip=25.1.1
fi
run_python() { "$mamba_exe" run -p "$env_prefix" python "$@"; }
# PyROOT needs platform binaries; keep it in conda rather than pip.
"$mamba_exe" install --yes --override-channels -c conda-forge -p "$env_prefix" \
    root_base=6.34.10 python=3.11.13 numpy=2.2.6 pip=25.1.1
run_python -m pip --isolated install --index-url https://pypi.org/simple -r "$script_dir/requirements.txt"
case "$platform:$backend" in
    linux-64:cuda) torch_spec="torch==${torch_version}+cu126"; torch_index=https://download.pytorch.org/whl/cu126 ;;
    linux-64:cpu) torch_spec="torch==${torch_version}+cpu"; torch_index=https://download.pytorch.org/whl/cpu ;;
    *) torch_spec="torch==${torch_version}"; torch_index=https://pypi.org/simple ;;
esac
run_python -m pip --isolated install --index-url "$torch_index" "$torch_spec"
run_python -m pip check
run_python -m ipykernel install --user --name flypilot --display-name 'Python (flypilot)'
# Login nodes may lack accelerators: validate the installed build and CPU here.
run_python "$script_dir/../python/check_environment.py" --device cpu --expected-backend "$backend"
printf '\nInstalled flypilot (%s). In Zsh run:\n' "$backend"
printf 'export MAMBA_ROOT_PREFIX=%q\nsource %q\n' "$root_prefix" "$script_dir/../setup.sh"
printf 'Validate accelerator access on the target machine/allocation with:\n%q check-environment --device %s\n' "$script_dir/../doThis.sh" "$backend"
