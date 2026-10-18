#!/usr/bin/env zsh
# Source this file in Zsh. No installation or shell startup-file changes occur.
if [ -z "${ZSH_VERSION:-}" ]; then
    printf '%s\n' 'Use Zsh: source /path/to/FlyPilot/setup.sh' >&2
    return 1 2>/dev/null || exit 1
fi
if [[ "$ZSH_EVAL_CONTEXT" != *:file* ]]; then
    print -u2 'Source this script to activate the current shell: source setup.sh'
    exit 1
fi

function _flypilot_activate() {
    # Conda compiler activation hooks probe unset optional variables.
    # Keep this exception local so callers retain their original shell options.
    setopt localoptions
    unsetopt nounset
    local root_prefix="${MAMBA_ROOT_PREFIX:-${HOME}/micromamba}"
    local mamba_exe hook
    if [[ -n "${MAMBA_EXE:-}" && -x "$MAMBA_EXE" ]]; then
        mamba_exe="$MAMBA_EXE"
    elif (( $+commands[micromamba] )); then
        mamba_exe="${commands[micromamba]}"
    elif [[ -x "$root_prefix/bin/micromamba" ]]; then
        mamba_exe="$root_prefix/bin/micromamba"
    else
        print -u2 'micromamba not found. Run bash scripts/first_setup.sh first.'
        return 1
    fi
    if [[ ! -d "$root_prefix/envs/flypilot/conda-meta" ]]; then
        print -u2 "flypilot is missing under $root_prefix. Run first_setup.sh with the same root prefix."
        return 1
    fi
    hook="$("$mamba_exe" shell hook --shell zsh --root-prefix "$root_prefix")" || return 1
    export MAMBA_ROOT_PREFIX="$root_prefix"
    eval "$hook" || return 1
    micromamba activate "$root_prefix/envs/flypilot" || return 1
    export PYTHONNOUSERSITE=1
    if [[ -n "${PYTHONPATH:-}${PYTHONHOME:-}" ]]; then
        print -u2 'Warning: PYTHONPATH/PYTHONHOME is set; unset these if imports come from outside flypilot.'
    fi
}
_flypilot_activate
