#!/usr/bin/env bash
set -Eeuo pipefail

API_BASE="${API_BASE:-http://127.0.0.1:8000}"
RAG_BASE="${RAG_BASE:-http://127.0.0.1:8002}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-60}"
POLL_SECONDS="${POLL_SECONDS:-2}"
CURL_TIMEOUT="${CURL_TIMEOUT:-20}"
REQUIRE_LLM_JUDGE_AUDIT="${REQUIRE_LLM_JUDGE_AUDIT:-true}"
VERIFY_TRACE_INSTRUMENTATION="${VERIFY_TRACE_INSTRUMENTATION:-true}"
WARM_RETRIEVAL_BEFORE_TRACE="${WARM_RETRIEVAL_BEFORE_TRACE:-true}"
WARMUP_ATTEMPTS="${WARMUP_ATTEMPTS:-4}"
WARMUP_MAX_LATENCY_MS="${WARMUP_MAX_LATENCY_MS:-3000}"
AUDIT_TRACE_LATENCY_NOTE_MS="${AUDIT_TRACE_LATENCY_NOTE_MS:-10000}"

if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: curl is required." >&2
  exit 127
fi

PYTHON_CMD=()

python_candidate_works() {
  local candidate="$1"
  local -a parts
  read -r -a parts <<<"${candidate}"

  if ((${#parts[@]} == 0)); then
    return 1
  fi

  if [[ "${parts[0]}" != */* ]] && ! command -v "${parts[0]}" >/dev/null 2>&1; then
    return 1
  fi

  "${parts[@]}" -c 'import json, sys; sys.exit(0)' >/dev/null 2>&1
}

set_python_cmd() {
  local candidate="$1"
  read -r -a PYTHON_CMD <<<"${candidate}"
}

discover_python() {
  local -a candidates=()

  if [[ -n "${PYTHON_BIN:-}" ]]; then
    candidates+=("${PYTHON_BIN}")
  fi
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    candidates+=(
      "${VIRTUAL_ENV}/Scripts/python.exe"
      "${VIRTUAL_ENV}/bin/python"
    )
  fi

  candidates+=(
    "./.venv/Scripts/python.exe"
    "./myenv/Scripts/python.exe"
    "py -3"
    "py"
    "python3"
    "python"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if python_candidate_works "${candidate}"; then
      set_python_cmd "${candidate}"
      return 0
    fi
  done

  echo "ERROR: could not find a usable Python interpreter for JSON parsing." >&2
  echo "Set PYTHON_BIN=./.venv/Scripts/python.exe or run from an activated project venv." >&2
  exit 127
}

discover_python
echo "Using Python interpreter: ${PYTHON_CMD[*]}"

tmp_files=()

make_tmpfile() {
  local var_name="$1"
  local f
  f="$(mktemp)"
  tmp_files+=("${f}")
  printf -v "${var_name}" '%s' "${f}"
}

cleanup() {
  if ((${#tmp_files[@]})); then
    rm -f "${tmp_files[@]}"
  fi
}
trap cleanup EXIT

json_get() {
  local path="$1"
  "${PYTHON_CMD[@]}" -c '
import json
import sys

try:
    data = json.load(sys.stdin)
except Exception:
    print("")
    raise SystemExit(0)

for key in sys.argv[1].split("."):
    if not key:
        continue
    if isinstance(data, dict):
        data = data.get(key, "")
    elif isinstance(data, list) and key.isdigit():
        idx = int(key)
        data = data[idx] if idx < len(data) else ""
    else:
        data = ""
        break

print("" if data is None else data)
' "${path}"
}

number_lte() {
  local value="${1:-0}"
  local threshold="${2:-0}"
  "${PYTHON_CMD[@]}" -c '
import sys

try:
    value = float(sys.argv[1] or 0)
    threshold = float(sys.argv[2] or 0)
except ValueError:
    raise SystemExit(1)

raise SystemExit(0 if value <= threshold else 1)
' "${value}" "${threshold}"
}

number_gt() {
  local value="${1:-0}"
  local threshold="${2:-0}"
  "${PYTHON_CMD[@]}" -c '
import sys

try:
    value = float(sys.argv[1] or 0)
    threshold = float(sys.argv[2] or 0)
except ValueError:
    raise SystemExit(1)

raise SystemExit(0 if value > threshold else 1)
' "${value}" "${threshold}"
}

curl_status_to_file() {
  local output_file="$1"
  shift
  local err_file="${output_file}.curlerr"
  local http_code
  tmp_files+=("${err_file}")

  if http_code="$(
    curl -sS \
      --connect-timeout 5 \
      --max-time "${CURL_TIMEOUT}" \
      -o "${output_file}" \
      -w "%{http_code}" \
      "$@" 2>"${err_file}"
  )"; then
    printf '%s' "${http_code}"
    return 0
  fi

  local curl_exit=$?
  local error_text=""
  if [[ -s "${err_file}" ]]; then
    error_text="$(tr '\n' ' ' <"${err_file}")"
  fi

  printf 'curl_exit=%s' "${curl_exit}" >"${output_file}"
  if [[ -n "${error_text}" ]]; then
    printf ': %s' "${error_text}" >>"${output_file}"
  fi
  printf '000'
  return 0
}

wait_for_http_200() {
  local name="$1"
  local url="$2"
  local attempts="${3:-${MAX_ATTEMPTS}}"
  local seconds="${4:-${POLL_SECONDS}}"
  local body
  local code
  local response

  for attempt in $(seq 1 "${attempts}"); do
    make_tmpfile body
    code="$(curl_status_to_file "${body}" "${url}")"
    response="$(<"${body}")"

    if [[ "${code}" == "200" ]]; then
      echo "${name} is ready (HTTP 200)"
      return 0
    fi

    echo "Attempt ${attempt}/${attempts}: ${name} not ready. HTTP ${code}: ${response:-<empty>}" >&2
    sleep "${seconds}"
  done

  echo "ERROR: ${name} did not become ready after ${attempts} attempts: ${url}" >&2
  echo "Last response: ${response:-<empty>}" >&2
  exit 1
}

langsmith_key_present() {
  if [[ -n "${LANGCHAIN_API_KEY:-}" && "${LANGCHAIN_API_KEY}" != "your_langsmith_key" ]]; then
    return 0
  fi

  if [[ -f ".env" ]]; then
    "${PYTHON_CMD[@]}" -c '
from pathlib import Path
import sys

env_path = Path(".env")
for raw_line in env_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    if key.strip() == "LANGCHAIN_API_KEY":
        value = value.strip().strip("\"").strip(chr(39))
        if value and value not in {"your_langsmith_key", "your_langsmith_key_here"}:
            raise SystemExit(0)
raise SystemExit(1)
'
    return $?
  fi

  return 1
}

env_file_value() {
  local wanted_key="$1"
  if [[ ! -f ".env" ]]; then
    return 0
  fi

  "${PYTHON_CMD[@]}" -c '
from pathlib import Path
import sys

wanted_key = sys.argv[1]
for raw_line in Path(".env").read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    if key.strip() == wanted_key:
        print(value.strip().strip("\"").strip(chr(39)))
        break
' "${wanted_key}"
}

normalize_judge_mode() {
  local raw
  raw="$(printf '%s' "${1:-fallback}" | tr '[:upper:]' '[:lower:]')"
  case "${raw}" in
    1|true|always)
      printf 'audit'
      ;;
    *)
      printf '%s' "${raw}"
      ;;
  esac
}

judge_mode_has_real_llm_span() {
  local normalized
  normalized="$(normalize_judge_mode "${1:-fallback}")"
  [[ "${normalized}" == "audit" || "${normalized}" == "strict" ]]
}

compose_service_env_value() {
  local service="$1"
  local key="$2"
  if ! command -v docker >/dev/null 2>&1; then
    return 1
  fi

  local value
  if ! value="$(docker compose exec -T "${service}" printenv "${key}" 2>/dev/null)"; then
    return 1
  fi

  value="${value//$'\r'/}"
  value="${value//$'\n'/}"
  if [[ -z "${value}" ]]; then
    return 1
  fi

  printf '%s' "${value}"
}

verify_llm_judge_audit_mode() {
  case "${REQUIRE_LLM_JUDGE_AUDIT}" in
    1|true|TRUE|yes|YES|on|ON) ;;
    *)
      echo "Skipping LLM judge audit-mode verification because REQUIRE_LLM_JUDGE_AUDIT=${REQUIRE_LLM_JUDGE_AUDIT}"
      return 0
      ;;
  esac

  local requested_mode
  local live_mode
  requested_mode="${OUTPUT_GUARDRAILS_LLM_JUDGE_MODE:-$(env_file_value OUTPUT_GUARDRAILS_LLM_JUDGE_MODE)}"
  requested_mode="${requested_mode:-fallback}"

  if live_mode="$(compose_service_env_value llm-orchestrator OUTPUT_GUARDRAILS_LLM_JUDGE_MODE)"; then
    if judge_mode_has_real_llm_span "${live_mode}"; then
      echo "Groundedness judge trace mode verified from running container: OUTPUT_GUARDRAILS_LLM_JUDGE_MODE=${live_mode}"
      return 0
    fi

    echo "ERROR: running llm-orchestrator container has OUTPUT_GUARDRAILS_LLM_JUDGE_MODE='${live_mode:-<unset>}'." >&2
    echo "Recreate it with:" >&2
    echo "  OUTPUT_GUARDRAILS_LLM_JUDGE_MODE=audit ROOT_CAUSE_FAST_PATH_ENABLED=false docker compose up -d --build --force-recreate llm-orchestrator api-gateway" >&2
    exit 1
  fi

  if judge_mode_has_real_llm_span "${requested_mode}"; then
    echo "Groundedness judge trace mode verified from shell/.env: OUTPUT_GUARDRAILS_LLM_JUDGE_MODE=${requested_mode}"
    return 0
  fi

  echo "ERROR: LangSmith screenshot mode requires OUTPUT_GUARDRAILS_LLM_JUDGE_MODE=audit or strict." >&2
  echo "The shell/.env value is '${requested_mode}', and the running llm-orchestrator container could not be inspected for an audit-mode override." >&2
  echo "Run:" >&2
  echo "  OUTPUT_GUARDRAILS_LLM_JUDGE_MODE=audit ROOT_CAUSE_FAST_PATH_ENABLED=false docker compose up -d --build --force-recreate llm-orchestrator api-gateway" >&2
  echo "Then rerun this script. If Docker is unavailable to this shell, use:" >&2
  echo "  OUTPUT_GUARDRAILS_LLM_JUDGE_MODE=audit bash scripts/phase10_langsmith_trace_smoke.sh" >&2
  exit 1
}

verify_fast_path_disabled() {
  local live_fast_path
  if ! live_fast_path="$(compose_service_env_value llm-orchestrator ROOT_CAUSE_FAST_PATH_ENABLED)"; then
    return 0
  fi

  case "$(printf '%s' "${live_fast_path}" | tr '[:upper:]' '[:lower:]')" in
    0|false|no|off)
      echo "Root-cause fast path verified disabled in running container."
      ;;
    *)
      echo "ERROR: running llm-orchestrator container has ROOT_CAUSE_FAST_PATH_ENABLED='${live_fast_path}'." >&2
      echo "The LangSmith screenshot must show the LLM-backed path, not the rules+retrieval fast path." >&2
      echo "Recreate it with:" >&2
      echo "  OUTPUT_GUARDRAILS_LLM_JUDGE_MODE=audit ROOT_CAUSE_FAST_PATH_ENABLED=false docker compose up -d --build --force-recreate llm-orchestrator api-gateway" >&2
      exit 1
      ;;
  esac
}

verify_trace_instrumentation_version() {
  case "${VERIFY_TRACE_INSTRUMENTATION}" in
    1|true|TRUE|yes|YES|on|ON) ;;
    *)
      echo "Skipping trace instrumentation verification because VERIFY_TRACE_INSTRUMENTATION=${VERIFY_TRACE_INSTRUMENTATION}"
      return 0
      ;;
  esac

  if ! command -v docker >/dev/null 2>&1; then
    echo "WARN: docker is not available; cannot verify running container instrumentation." >&2
    return 0
  fi

  local verification_output
  if verification_output="$(
    docker compose exec -T llm-orchestrator python -c '
from llm_orchestrator.guardrails.output_filters import OutputGuardrails

required = (
    "_deterministic_groundedness_trace",
    "_run_llm_groundedness_judge",
)
missing = [name for name in required if not hasattr(OutputGuardrails, name)]
if missing:
    raise SystemExit("missing instrumentation: " + ", ".join(missing))

print("Guardrail trace instrumentation verified in running container.")
'
  )"; then
    echo "${verification_output}"
    return 0
  fi

  echo "ERROR: running llm-orchestrator container does not contain the Phase 10 guardrail trace instrumentation." >&2
  echo "This usually means the service was recreated from a stale image. Rebuild/recreate it with:" >&2
  echo "  OUTPUT_GUARDRAILS_LLM_JUDGE_MODE=audit ROOT_CAUSE_FAST_PATH_ENABLED=false docker compose up -d --build --force-recreate llm-orchestrator api-gateway" >&2
  echo "Then rerun this script and retake docs/assets/screenshots/langsmith-trace.png." >&2
  if [[ -n "${verification_output:-}" ]]; then
    echo "Container verification output: ${verification_output}" >&2
  fi
  exit 1
}

build_retrieval_warmup_payload() {
  local endpoint_kind="$1"
  local retrieval_query
  retrieval_query="$(printf '%s\n%s\n%s\n%s' \
    "Why did pump P-23 trigger anomaly at 03:41?" \
    "Equipment: pump_P-23" \
    "Anomaly: Pump P-23 triggered a high-vibration anomaly at 03:41 with vibration RMS above the alarm threshold and no corresponding pressure drop." \
    "Observed signals: bearing failure, bearing wear, lubrication, relubrication, high vibration")"

  "${PYTHON_CMD[@]}" -c '
import json
import sys

endpoint_kind = sys.argv[1]
retrieval_query = sys.argv[2]
payload = {
    "query": retrieval_query,
    "filters": {"equipment_id": "pump_P-23"},
}
if endpoint_kind == "hybrid":
    payload.update({"semantic_k": 15, "keyword_k": 15, "out_k": 8, "rrf_k": 60})
else:
    payload.update({"k": 4})
print(json.dumps(payload, separators=(",", ":")))
' "${endpoint_kind}" "${retrieval_query}"
}

warm_retrieval_endpoint() {
  local name="$1"
  local path="$2"
  local endpoint_kind="$3"
  local payload
  local body
  local code
  local response
  local count
  local latency_ms

  payload="$(build_retrieval_warmup_payload "${endpoint_kind}")"

  for attempt in $(seq 1 "${WARMUP_ATTEMPTS}"); do
    make_tmpfile body
    code="$(
      curl_status_to_file \
        "${body}" \
        -X POST "${RAG_BASE}${path}" \
        -H "Content-Type: application/json" \
        -d "${payload}"
    )"
    response="$(<"${body}")"

    if [[ ! "${code}" =~ ^2 ]]; then
      echo "Attempt ${attempt}/${WARMUP_ATTEMPTS}: ${name} warm-up failed. HTTP ${code}: ${response:-<empty>}" >&2
      sleep "${POLL_SECONDS}"
      continue
    fi

    count="$(printf '%s' "${response}" | json_get "count")"
    latency_ms="$(printf '%s' "${response}" | json_get "latency_ms")"
    count="${count:-0}"
    latency_ms="${latency_ms:-0}"

    if number_gt "${count}" 0 && number_lte "${latency_ms}" "${WARMUP_MAX_LATENCY_MS}"; then
      echo "${name} warm-up verified: count=${count}, latency_ms=${latency_ms}"
      return 0
    fi

    echo "Attempt ${attempt}/${WARMUP_ATTEMPTS}: ${name} warming. count=${count}, latency_ms=${latency_ms}, target<=${WARMUP_MAX_LATENCY_MS}ms"
    sleep "${POLL_SECONDS}"
  done

  echo "ERROR: ${name} did not warm below ${WARMUP_MAX_LATENCY_MS}ms with non-empty retrieval results." >&2
  echo "This screenshot would show cold or degraded retrieval latency. Inspect RAG logs before capturing LangSmith evidence:" >&2
  echo "  docker compose logs --tail=120 rag-service" >&2
  exit 1
}

warm_retrieval_before_trace() {
  case "${WARM_RETRIEVAL_BEFORE_TRACE}" in
    1|true|TRUE|yes|YES|on|ON) ;;
    *)
      echo "Skipping retrieval warm-up because WARM_RETRIEVAL_BEFORE_TRACE=${WARM_RETRIEVAL_BEFORE_TRACE}"
      return 0
      ;;
  esac

  echo "Checking RAG service readiness at ${RAG_BASE}/health/ready"
  wait_for_http_200 "RAG service" "${RAG_BASE}/health/ready"
  echo "Warming real RAG retrieval paths before LangSmith trace capture"
  warm_retrieval_endpoint "Hybrid retrieval" "/retrieve/hybrid" "hybrid"
  warm_retrieval_endpoint "Procedure retrieval" "/retrieve/procedures" "procedure"
}

if ! langsmith_key_present; then
  echo "ERROR: LANGCHAIN_API_KEY is not configured. Add a real LangSmith key to .env or export it." >&2
  echo "Do not fake langsmith-trace.png; generate it only after tracing is enabled." >&2
  exit 1
fi

verify_llm_judge_audit_mode
verify_fast_path_disabled
verify_trace_instrumentation_version
warm_retrieval_before_trace

echo "Checking API gateway readiness at ${API_BASE}/health/ready"
wait_for_http_200 "API gateway" "${API_BASE}/health/ready"

payload="$(printf '%s' '{"chain":"root_cause","bypass_cache":true,"root_cause":{"user_query":"Why did pump P-23 trigger anomaly at 03:41?","equipment_id":"pump_P-23","anomaly_description":"Pump P-23 triggered a high-vibration anomaly at 03:41 with vibration RMS above the alarm threshold and no corresponding pressure drop.","sensor_data":{"vibration_rms":8.4,"temp_c":74.2,"pressure_bar":5.2,"flow_rate_lpm":176.0}}}')"
trace_id="${TRACE_ID:-phase10-langsmith-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}}"

echo "Submitting LangSmith trace demo query with trace_id=${trace_id}"
make_tmpfile query_body
query_code="$(
  curl_status_to_file \
    "${query_body}" \
    -X POST "${API_BASE}/query" \
    -H "Content-Type: application/json" \
    -H "X-Trace-ID: ${trace_id}" \
    -d "${payload}"
)"

query_response="$(<"${query_body}")"
if [[ ! "${query_code}" =~ ^2 ]]; then
  echo "ERROR: query submission failed. HTTP ${query_code}: ${query_response}" >&2
  exit 1
fi

job_id="$(printf '%s' "${query_response}" | json_get "job_id")"
if [[ -z "${job_id}" ]]; then
  echo "ERROR: query response did not include job_id: ${query_response}" >&2
  exit 1
fi

status="processing"
status_response=""
echo "Polling query job_id=${job_id}"
for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
  make_tmpfile status_body
  status_code="$(curl_status_to_file "${status_body}" "${API_BASE}/query/${job_id}")"
  status_response="$(<"${status_body}")"

  if [[ "${status_code}" != "200" ]]; then
    echo "Attempt ${attempt}/${MAX_ATTEMPTS}: query status HTTP ${status_code}" >&2
    sleep "${POLL_SECONDS}"
    continue
  fi

  status="$(printf '%s' "${status_response}" | json_get "status")"
  if [[ "${status}" == "completed" ]]; then
    break
  fi
  if [[ "${status}" == "failed" ]]; then
    error="$(printf '%s' "${status_response}" | json_get "error")"
    echo "ERROR: query failed: ${error}" >&2
    exit 1
  fi

  echo "Attempt ${attempt}/${MAX_ATTEMPTS}: status=${status:-unknown}"
  sleep "${POLL_SECONDS}"
done

if [[ "${status}" != "completed" ]]; then
  echo "ERROR: query did not complete after ${MAX_ATTEMPTS} attempts. Last response: ${status_response}" >&2
  exit 1
fi

provider="$(printf '%s' "${status_response}" | json_get "result.model_provider")"
model="$(printf '%s' "${status_response}" | json_get "result.model_name")"
latency_ms="$(printf '%s' "${status_response}" | json_get "result.latency_ms")"

if [[ "${provider}" == "rules+retrieval" ]]; then
  echo "ERROR: trace demo still used the fast path. Recreate with ROOT_CAUSE_FAST_PATH_ENABLED=false and judge audit mode." >&2
  echo "Run: OUTPUT_GUARDRAILS_LLM_JUDGE_MODE=audit ROOT_CAUSE_FAST_PATH_ENABLED=false docker compose up -d --build --force-recreate llm-orchestrator api-gateway" >&2
  exit 1
fi

echo "LangSmith guardrail audit trace completed: provider=${provider}, model=${model}, audit_latency_ms=${latency_ms}"
if number_gt "${latency_ms}" "${AUDIT_TRACE_LATENCY_NOTE_MS}"; then
  echo "NOTE: audit_latency_ms includes the intentionally forced LLM path plus the extra groundedness judge call."
  echo "Do not use this audit trace as the p95 serving-latency screenshot; restore fallback judge mode and the root-cause fast path for normal latency evidence."
fi
echo "Open https://smith.langchain.com/ and capture project industrial-reliability-copilot."
echo "Expected trace tree: Process_Query_Background -> Input_Guardrails -> retrieval -> Prompt_Model_Call -> Output_Guardrails -> Deterministic_Groundedness_Check -> Groundedness_LLM_Judge -> Prompt_Model_Call(is_judge=true)."
