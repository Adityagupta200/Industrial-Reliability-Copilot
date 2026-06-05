#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/phase11_python_env.sh
source "$script_dir/phase11_python_env.sh"

cd "$REPO_ROOT"
export ORCHESTRATOR_URL="${ORCHESTRATOR_URL:-http://127.0.0.1:8000/query}"

if [ "${SKIP_EVAL_HEALTH_CHECK:-false}" != "true" ]; then
  "$PY" - <<'PY'
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

query_url = os.environ["ORCHESTRATOR_URL"]
parts = urllib.parse.urlsplit(query_url)
if not parts.scheme or not parts.netloc:
    raise SystemExit(f"ORCHESTRATOR_URL must be an absolute URL, got: {query_url!r}")

health_url = urllib.parse.urlunsplit((parts.scheme, parts.netloc, "/health/ready", "", ""))
try:
    with urllib.request.urlopen(health_url, timeout=10) as response:
        if response.status >= 400:
            raise RuntimeError(f"HTTP {response.status}")
except (OSError, RuntimeError, urllib.error.URLError) as exc:
    print(
        "Orchestrator health check failed before offline evaluation.\n"
        f"  health_url={health_url}\n"
        f"  error={exc}\n"
        "Start the stack first, or set SKIP_EVAL_HEALTH_CHECK=true only when "
        "intentionally evaluating a remote endpoint without /health/ready.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
fi

"$PY" src/evaluation/offline/ragas_eval.py
"$PY" scripts/check_thresholds.py
