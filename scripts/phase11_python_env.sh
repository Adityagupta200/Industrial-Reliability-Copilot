#!/usr/bin/env bash
# Resolve the project Python interpreter for bash-based Phase 11 validation.
# Source this file before using "$PY" in Git Bash, WSL, Linux, or macOS shells.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "source scripts/phase11_python_env.sh instead of executing it directly" >&2
  exit 2
fi

_phase11_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(cd -- "$_phase11_script_dir/.." && pwd)}"

_phase11_command_exists() {
  command -v "$1" >/dev/null 2>&1
}

_phase11_python_is_usable() {
  local candidate="$1"
  if [[ "$candidate" == */* || "$candidate" == *\\* ]]; then
    [ -x "$candidate" ] || return 1
  else
    _phase11_command_exists "$candidate" || return 1
  fi

  "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
}

_phase11_candidates=()
if [ -n "${PY:-}" ]; then
  _phase11_candidates+=("$PY")
fi
_phase11_candidates+=(
  "$REPO_ROOT/.venv/bin/python"
  "$REPO_ROOT/.venv/Scripts/python.exe"
  "python3"
  "python"
)

_phase11_resolved_python=""
for _phase11_candidate in "${_phase11_candidates[@]}"; do
  if _phase11_python_is_usable "$_phase11_candidate"; then
    _phase11_resolved_python="$_phase11_candidate"
    break
  fi
done

if [ -z "$_phase11_resolved_python" ]; then
  echo "Could not find a usable Python >=3.11 interpreter." >&2
  echo "Create the project virtualenv first, then rerun: python -m venv .venv" >&2
  return 2
fi

export PY="$_phase11_resolved_python"

case ":${PYTHONPATH:-}:" in
  *":$REPO_ROOT/src:"* | *":src:"*) ;;
  *) export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

unset _phase11_candidate
unset _phase11_candidates
unset _phase11_resolved_python
unset _phase11_script_dir
