#!/usr/bin/env bash
set -Eeuo pipefail

API_BASE="${API_BASE:-http://127.0.0.1:8000}"
PROM_BASE="${PROM_BASE:-http://127.0.0.1:9090}"
GRAFANA_BASE="${GRAFANA_BASE:-http://127.0.0.1:3000}"
GRAFANA_USER="${GRAFANA_USER:-admin}"
GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-admin}"
VERIFY_GRAFANA_DASHBOARDS="${VERIFY_GRAFANA_DASHBOARDS:-true}"
EXPECTED_PROMETHEUS_JOBS="${EXPECTED_PROMETHEUS_JOBS:-prometheus api-gateway llm-orchestrator rag-service anomaly-service}"
SCORE="${SCORE:-5}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-45}"
POLL_SECONDS="${POLL_SECONDS:-2}"
VERIFY_ATTEMPTS="${VERIFY_ATTEMPTS:-20}"
VERIFY_SECONDS="${VERIFY_SECONDS:-3}"
CURL_TIMEOUT="${CURL_TIMEOUT:-20}"

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
  echo "Tried repo virtualenvs, VIRTUAL_ENV, py.exe, python3, and python." >&2
  echo "Set PYTHON_BIN=./.venv/Scripts/python.exe or run from an activated project venv." >&2
  exit 127
}

discover_python
echo "Using Python interpreter: ${PYTHON_CMD[*]}"

if [[ ! "${SCORE}" =~ ^[1-5]$ ]]; then
  echo "ERROR: SCORE must be an integer from 1 to 5. Received: ${SCORE}" >&2
  exit 2
fi

if (( SCORE >= 4 )); then
  RATING="positive"
elif (( SCORE <= 2 )); then
  RATING="negative"
else
  RATING="neutral"
fi

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

json_find_dashboard_uid() {
  local title="$1"
  "${PYTHON_CMD[@]}" -c '
import json
import sys

title = sys.argv[1]
try:
    data = json.load(sys.stdin)
except Exception:
    print("")
    raise SystemExit(0)

if isinstance(data, dict):
    dashboards = data.get("dashboards") or data.get("results") or []
else:
    dashboards = data

if not isinstance(dashboards, list):
    print("")
    raise SystemExit(0)

for item in dashboards:
    if isinstance(item, dict) and item.get("title") == title and item.get("uid"):
        print(item["uid"])
' "${title}"
}

number_greater_than() {
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

number_subtract() {
  local value="${1:-0}"
  local baseline="${2:-0}"
  "${PYTHON_CMD[@]}" -c '
import sys

try:
    value = float(sys.argv[1] or 0)
    baseline = float(sys.argv[2] or 0)
except ValueError:
    print("0")
    raise SystemExit(0)

delta = value - baseline
if abs(delta - round(delta)) < 1e-9:
    print(int(round(delta)))
else:
    print(f"{delta:.6f}")
' "${value}" "${baseline}"
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

grafana_status_to_file() {
  local output_file="$1"
  shift
  local err_file="${output_file}.curlerr"
  local http_code
  tmp_files+=("${err_file}")

  if http_code="$(
    curl -sS \
      --connect-timeout 5 \
      --max-time "${CURL_TIMEOUT}" \
      -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" \
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

prometheus_query_value() {
  local query="$1"
  local body
  local code
  local response

  make_tmpfile body
  code="$(
    curl_status_to_file \
      "${body}" \
      --get "${PROM_BASE}/api/v1/query" \
      --data-urlencode "query=${query}"
  )"
  response="$(<"${body}")"

  if [[ "${code}" != "200" ]]; then
    echo ""
    return 1
  fi

  printf '%s' "${response}" | json_get "data.result.0.value.1"
}

verify_prometheus_targets() {
  local targets_body
  local targets_code
  local targets_response

  echo "Verifying Prometheus scrape targets: ${EXPECTED_PROMETHEUS_JOBS}"
  make_tmpfile targets_body
  targets_code="$(
    curl_status_to_file \
      "${targets_body}" \
      --get "${PROM_BASE}/api/v1/targets" \
      --data-urlencode "state=active"
  )"
  targets_response="$(<"${targets_body}")"

  if [[ "${targets_code}" != "200" ]]; then
    echo "ERROR: Prometheus targets API failed. HTTP ${targets_code}: ${targets_response}" >&2
    exit 1
  fi

  printf '%s' "${targets_response}" | "${PYTHON_CMD[@]}" -c '
import json
import sys

expected = sys.argv[1].split()

try:
    payload = json.load(sys.stdin)
except Exception as exc:
    print(f"ERROR: Prometheus targets response was not valid JSON: {exc}", file=sys.stderr)
    raise SystemExit(1)

targets = payload.get("data", {}).get("activeTargets", [])
by_job = {}
errors = []

for target in targets:
    labels = target.get("labels", {}) or {}
    job = labels.get("job", "")
    if not job:
        continue
    by_job.setdefault(job, []).append(target)

missing = [job for job in expected if job not in by_job]
if missing:
    errors.append("missing jobs: " + ", ".join(missing))

for job in expected:
    job_targets = by_job.get(job, [])
    unhealthy = [
        target
        for target in job_targets
        if target.get("health") != "up"
    ]
    if unhealthy:
        details = []
        for target in unhealthy:
            labels = target.get("labels", {}) or {}
            instance = labels.get("instance") or target.get("scrapeUrl") or "<unknown>"
            health = target.get("health", "<unknown>")
            last_error = target.get("lastError") or "no error message"
            details.append(f"{instance} health={health} error={last_error}")
        errors.append(f"{job}: " + "; ".join(details))

if errors:
    print("ERROR: Prometheus target health check failed:", file=sys.stderr)
    for error in errors:
        print(f"- {error}", file=sys.stderr)
    raise SystemExit(1)

summary = []
for job in expected:
    job_targets = by_job[job]
    instances = sorted(
        (target.get("labels", {}) or {}).get("instance", "<unknown>")
        for target in job_targets
    )
    instances_text = ", ".join(instances)
    summary.append(f"{job}={len(job_targets)} up ({instances_text})")

print("Prometheus targets verified: " + "; ".join(summary))
' "${EXPECTED_PROMETHEUS_JOBS}"
}

reload_and_verify_grafana_dashboards() {
  case "${VERIFY_GRAFANA_DASHBOARDS}" in
    1|true|TRUE|yes|YES|on|ON) ;;
    *)
      echo "Skipping Grafana dashboard reload/verification because VERIFY_GRAFANA_DASHBOARDS=${VERIFY_GRAFANA_DASHBOARDS}"
      return 0
      ;;
  esac

  echo "Checking Grafana readiness at ${GRAFANA_BASE}/api/health"
  make_tmpfile grafana_health_body
  grafana_health_code="$(grafana_status_to_file "${grafana_health_body}" "${GRAFANA_BASE}/api/health")"
  if [[ "${grafana_health_code}" != "200" ]]; then
    echo "ERROR: Grafana is not reachable or credentials are invalid. HTTP ${grafana_health_code}: $(<"${grafana_health_body}")" >&2
    echo "Set GRAFANA_USER/GRAFANA_PASSWORD or VERIFY_GRAFANA_DASHBOARDS=false if Grafana verification is intentionally skipped." >&2
    exit 1
  fi

  echo "Reloading Grafana provisioned dashboards"
  make_tmpfile grafana_reload_body
  grafana_reload_code="$(
    grafana_status_to_file \
      "${grafana_reload_body}" \
      -X POST "${GRAFANA_BASE}/api/admin/provisioning/dashboards/reload"
  )"
  if [[ ! "${grafana_reload_code}" =~ ^2 ]]; then
    echo "ERROR: Grafana dashboard provisioning reload failed. HTTP ${grafana_reload_code}: $(<"${grafana_reload_body}")" >&2
    exit 1
  fi

  verify_grafana_dashboard_current \
    "RAG Quality Metrics" \
    "Online Groundedness Proxy (Range Avg)" \
    "User Feedback Ratings (Selected Range)" \
    '$__range'
  verify_grafana_dashboard_current \
    "System Health - LLM Orchestrator" \
    "Completed Queries" \
    '$__rate_interval'
  verify_grafana_dashboard_current \
    "LLM Cost & Token Usage" \
    "Fast-Path Share" \
    '$__range'
}

verify_grafana_dashboard_current() {
  local title="$1"
  shift
  local search_body
  local search_code
  local search_response
  local uids
  local uid
  local dashboard_body
  local dashboard_code
  local needle
  local dashboard_current

  make_tmpfile search_body
  search_code="$(
    grafana_status_to_file \
      "${search_body}" \
      --get "${GRAFANA_BASE}/api/search" \
      --data-urlencode "query=${title}"
  )"
  search_response="$(<"${search_body}")"
  if [[ "${search_code}" != "200" ]]; then
    echo "ERROR: Grafana dashboard search failed for '${title}'. HTTP ${search_code}: ${search_response}" >&2
    exit 1
  fi

  uids="$(printf '%s' "${search_response}" | json_find_dashboard_uid "${title}")"
  if [[ -z "${uids}" ]]; then
    echo "ERROR: Grafana dashboard '${title}' was not found after provisioning reload." >&2
    echo "Search response: ${search_response}" >&2
    exit 1
  fi

  dashboard_current="false"
  while IFS= read -r uid; do
    [[ -z "${uid}" ]] && continue

    make_tmpfile dashboard_body
    dashboard_code="$(grafana_status_to_file "${dashboard_body}" "${GRAFANA_BASE}/api/dashboards/uid/${uid}")"
    if [[ "${dashboard_code}" != "200" ]]; then
      echo "WARN: Grafana dashboard '${title}' uid=${uid} could not be read. HTTP ${dashboard_code}: $(<"${dashboard_body}")" >&2
      continue
    fi

    dashboard_current="true"
    for needle in "$@"; do
      if ! grep -Fq "${needle}" "${dashboard_body}"; then
        dashboard_current="false"
        break
      fi
    done

    if [[ "${dashboard_current}" == "true" ]]; then
      echo "Grafana verified: ${title} (uid=${uid})"
      return 0
    fi
  done <<<"${uids}"

  echo "ERROR: Grafana dashboard '${title}' is stale; no matching dashboard contained all expected markers." >&2
  echo "Expected markers: $*" >&2
  echo "Try: docker compose restart grafana" >&2
  exit 1
}

PAYLOAD="${PAYLOAD:-$(printf '%s' '{"chain":"root_cause","root_cause":{"user_query":"Why did pump P-23 trigger anomaly at 03:41?","equipment_id":"pump_P-23","anomaly_description":"Pump P-23 triggered a high-vibration anomaly at 03:41 with vibration RMS above the alarm threshold and no corresponding pressure drop.","sensor_data":{"vibration_rms":8.4,"temp_c":74.2,"pressure_bar":5.2,"flow_rate_lpm":176.0}}}')}"
TRACE_ID="${TRACE_ID:-phase10-feedback-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}}"

echo "Checking API gateway readiness at ${API_BASE}/health/ready"
wait_for_http_200 "API gateway" "${API_BASE}/health/ready"

echo "Checking Prometheus readiness at ${PROM_BASE}/-/ready"
wait_for_http_200 "Prometheus" "${PROM_BASE}/-/ready"

verify_prometheus_targets
reload_and_verify_grafana_dashboards

echo "Submitting root-cause query with trace_id=${TRACE_ID}"
make_tmpfile query_body
query_code="$(
  curl_status_to_file \
    "${query_body}" \
    -X POST "${API_BASE}/query" \
    -H "Content-Type: application/json" \
    -H "X-Trace-ID: ${TRACE_ID}" \
    -d "${PAYLOAD}"
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

echo "Polling query job_id=${job_id}"
status="processing"
status_response=""
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

chain="$(printf '%s' "${status_response}" | json_get "result.chain")"
provider="$(printf '%s' "${status_response}" | json_get "result.model_provider")"
model="$(printf '%s' "${status_response}" | json_get "result.model_name")"
latency_ms="$(printf '%s' "${status_response}" | json_get "result.latency_ms")"
echo "Query completed: chain=${chain:-unknown}, provider=${provider:-unknown}, model=${model:-unknown}, latency_ms=${latency_ms:-unknown}"

counter_query="(sum(user_feedback_total{rating=\"${RATING}\"}) OR on() vector(0))"
feedback_counter_before="$(prometheus_query_value "${counter_query}" || true)"
feedback_counter_before="${feedback_counter_before:-0}"
echo "Feedback counter before submit: rating=${RATING}, value=${feedback_counter_before}"

feedback_payload="$(printf '{"query_id":"%s","score":%s}' "${job_id}" "${SCORE}")"
feedback_response=""
feedback_code=""

for attempt in $(seq 1 10); do
  make_tmpfile feedback_body
  feedback_code="$(
    curl_status_to_file \
      "${feedback_body}" \
      -X POST "${API_BASE}/feedback" \
      -H "Content-Type: application/json" \
      -H "X-Trace-ID: ${TRACE_ID}" \
      -d "${feedback_payload}"
  )"
  feedback_response="$(<"${feedback_body}")"

  if [[ "${feedback_code}" =~ ^2 ]]; then
    break
  fi
  if [[ "${feedback_code}" == "409" ]]; then
    echo "Feedback not ready yet; retrying (${attempt}/10). Response: ${feedback_response}"
    sleep 2
    continue
  fi

  echo "ERROR: feedback submission failed. HTTP ${feedback_code}: ${feedback_response}" >&2
  exit 1
done

if [[ ! "${feedback_code}" =~ ^2 ]]; then
  echo "ERROR: feedback was not accepted after retries. Last response: ${feedback_response}" >&2
  exit 1
fi

recorded_rating="$(printf '%s' "${feedback_response}" | json_get "rating")"
echo "Feedback accepted: query_id=${job_id}, score=${SCORE}, rating=${recorded_rating:-unknown}"

metric_value="${feedback_counter_before}"
metric_delta="0"

echo "Verifying Prometheus feedback counter for rating=${RATING}"
for attempt in $(seq 1 "${VERIFY_ATTEMPTS}"); do
  metric_value="$(prometheus_query_value "${counter_query}" || true)"
  metric_value="${metric_value:-0}"
  metric_delta="$(number_subtract "${metric_value}" "${feedback_counter_before}")"

  if number_greater_than "${metric_value}" "${feedback_counter_before}"; then
    echo "Prometheus verified: user_feedback_total{rating=\"${RATING}\"} ${feedback_counter_before} -> ${metric_value} (delta +${metric_delta})"
    echo "Grafana dashboard: http://127.0.0.1:3000"
    exit 0
  fi

  echo "Attempt ${attempt}/${VERIFY_ATTEMPTS}: Prometheus counter=${metric_value} (delta +${metric_delta}); waiting for scrape"
  sleep "${VERIFY_SECONDS}"
done

echo "ERROR: feedback was accepted but Prometheus did not expose it within the verification window." >&2
echo "Last Prometheus counter for rating=${RATING}: ${metric_value:-0} (baseline ${feedback_counter_before})" >&2
exit 1
