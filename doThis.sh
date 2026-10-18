#!/usr/bin/env zsh
# Single CLI for Python tasks. Add one named dispatch branch per new task.
set -eu
repo_root="${0:A:h}"
usage() {
    print -r -- 'Usage: ./doThis.sh COMMAND [OPTIONS]

  check-environment  Check imports, Pendulum, and a device (--device cpu|mps|cuda)
  register-kernel   Register Python (flypilot) for Jupyter and notebook editors
  compute-sanity     Benchmark compute (--devices cpu mps; use --help for options)
  test              Run correctness tests (remaining arguments go to pytest)
  render-notebook   Render step 01 from saved data (--run-dir results/<run-id>)

Runs in the flypilot environment. No arguments or --help displays this help.
Relative argument paths are interpreted from your current directory.'
}
if (( $# == 0 )); then usage; exit 0; fi
task="$1"
shift
case "$task" in
    -h|--help|help) usage; exit 0 ;;
    check-environment|register-kernel|compute-sanity|test|render-notebook) ;;
    *) print -u2 -- "Unknown command: $task"; usage >&2; exit 2 ;;
esac
source "$repo_root/setup.sh"
export PYTHONPATH="$repo_root/python${PYTHONPATH:+:$PYTHONPATH}"
case "$task" in
    check-environment) exec python "$repo_root/python/check_environment.py" "$@" ;;
    register-kernel) exec python -m ipykernel install --user --name flypilot --display-name 'Python (flypilot)' "$@" ;;
    compute-sanity) exec python "$repo_root/python/compute_sanity.py" "$@" ;;
    test) exec python -m pytest "$repo_root/tests" "$@" ;;
    render-notebook) exec python "$repo_root/python/render_notebook.py" "$@" ;;
esac
