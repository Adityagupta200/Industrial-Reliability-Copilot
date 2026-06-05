#!/usr/bin/env bash
set -euo pipefail

NS="${NS:-staging}"
AWS_REGION="${AWS_REGION:-ap-south-1}"
IMAGE_TAG="${IMAGE_TAG:-}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-}"
ECR_REGISTRY="${ECR_REGISTRY:-}"

REPORT_DIR="data/phase11/reports/eks"
RENDER_DIR="rendered/${NS}"
BOOTSTRAP_TIMEOUT_SECONDS="${BOOTSTRAP_TIMEOUT_SECONDS:-7200}"
BOOTSTRAP_POLL_SECONDS="${BOOTSTRAP_POLL_SECONDS:-10}"
LOG_ATTACH_TIMEOUT_SECONDS="${LOG_ATTACH_TIMEOUT_SECONDS:-1200}"

mkdir -p "${REPORT_DIR}" "${RENDER_DIR}"

require_command() {
  local command_name="$1"
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing required command: ${command_name}" >&2
    exit 1
  fi
}

require_command aws
require_command docker
require_command git
require_command kubectl
require_command sed

if [ -z "${IMAGE_TAG}" ]; then
  IMAGE_TAG="$(git rev-parse HEAD)"
fi

if [ -z "${AWS_ACCOUNT_ID}" ]; then
  AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
fi

if [ -z "${ECR_REGISTRY}" ]; then
  ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
fi

BOOTSTRAP_IMAGE="${ECR_REGISTRY}/rag-service:${IMAGE_TAG}-bootstrap"
RAG_RUNTIME_IMAGE="${ECR_REGISTRY}/rag-service:${IMAGE_TAG}"
log_pid=""

cleanup() {
  if [ -n "${log_pid}" ]; then
    kill "${log_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "Phase 11 bootstrap target: namespace=${NS}, region=${AWS_REGION}, image_tag=${IMAGE_TAG}"
kubectl get namespace "${NS}" >/dev/null

job_condition() {
  { kubectl -n "${NS}" get job phase11-data-bootstrap \
    -o jsonpath='{range .status.conditions[*]}{.type}={.status}{"\n"}{end}' 2>/dev/null || true; } \
    | awk -F= '
        $1 == "Complete" && $2 == "True" { print "Complete"; found=1; exit }
        $1 == "Failed" && $2 == "True" { print "Failed"; found=1; exit }
        END { if (!found) print "Running" }
      '
}

pod_state() {
  local pod_name="$1"
  kubectl -n "${NS}" get pod "${pod_name}" \
    -o jsonpath='{.status.phase}{"|"}{.status.containerStatuses[0].state.waiting.reason}{"|"}{.status.containerStatuses[0].state.running.startedAt}{"|"}{.status.containerStatuses[0].state.terminated.reason}' \
    2>/dev/null || true
}

write_bootstrap_evidence() {
  local suffix="$1"
  kubectl -n "${NS}" get job phase11-data-bootstrap -o wide \
    > "${REPORT_DIR}/phase11_bootstrap_job_${suffix}.txt" 2>&1 || true
  kubectl -n "${NS}" describe job/phase11-data-bootstrap \
    > "${REPORT_DIR}/phase11_bootstrap_describe_${suffix}.txt" 2>&1 || true
  kubectl -n "${NS}" get pods -l job-name=phase11-data-bootstrap -o wide \
    > "${REPORT_DIR}/phase11_bootstrap_pods_${suffix}.txt" 2>&1 || true
  kubectl -n "${NS}" get events --sort-by=.lastTimestamp \
    > "${REPORT_DIR}/phase11_bootstrap_events_${suffix}.txt" 2>&1 || true
  if [ -n "${bootstrap_pod:-}" ]; then
    kubectl -n "${NS}" describe "pod/${bootstrap_pod}" \
      > "${REPORT_DIR}/phase11_bootstrap_pod_describe_${suffix}.txt" 2>&1 || true
    kubectl -n "${NS}" logs "pod/${bootstrap_pod}" --all-containers=true \
      --pod-running-timeout=300s \
      > "${REPORT_DIR}/phase11_bootstrap_${suffix}.log" 2>&1 || true
  else
    kubectl -n "${NS}" logs job/phase11-data-bootstrap --all-containers=true \
      --pod-running-timeout=300s \
      > "${REPORT_DIR}/phase11_bootstrap_${suffix}.log" 2>&1 || true
  fi
}

echo "Building Phase 11 bootstrap image from ${RAG_RUNTIME_IMAGE}"
docker build \
  -f src/rag_service/Dockerfile.bootstrap \
  --build-arg "RAG_RUNTIME_IMAGE=${RAG_RUNTIME_IMAGE}" \
  -t "${BOOTSTRAP_IMAGE}" \
  .

docker push "${BOOTSTRAP_IMAGE}"

cp infra/kubernetes/04-phase11-bootstrap-job.yaml "${RENDER_DIR}/04-phase11-bootstrap-job.yaml"
sed -i "s/namespace: industrial-copilot/namespace: ${NS}/g" \
  "${RENDER_DIR}/04-phase11-bootstrap-job.yaml"
sed -i "s|image: .*rag-service:.*-bootstrap|image: ${BOOTSTRAP_IMAGE}|g" \
  "${RENDER_DIR}/04-phase11-bootstrap-job.yaml"
sed -i "s/activeDeadlineSeconds: .*/activeDeadlineSeconds: ${BOOTSTRAP_TIMEOUT_SECONDS}/g" \
  "${RENDER_DIR}/04-phase11-bootstrap-job.yaml"

kubectl -n "${NS}" delete job phase11-data-bootstrap --ignore-not-found --wait=true
kubectl apply -f "${RENDER_DIR}/04-phase11-bootstrap-job.yaml"

bootstrap_pod=""
for _ in $(seq 1 120); do
  bootstrap_pod="$(
    kubectl -n "${NS}" get pods \
      -l job-name=phase11-data-bootstrap \
      -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null \
      | sed -n '1p' || true
  )"
  if [ -n "${bootstrap_pod}" ]; then
    break
  fi
  sleep 5
done

if [ -n "${bootstrap_pod}" ]; then
  echo "Waiting for pod/${bootstrap_pod} container to become loggable"
  attach_start_epoch="$(date +%s)"
  last_status_epoch=0
  while true; do
    state="$(pod_state "${bootstrap_pod}")"
    IFS="|" read -r phase waiting_reason running_started terminated_reason <<< "${state}"
    now_epoch="$(date +%s)"
    elapsed=$((now_epoch - attach_start_epoch))

    if [ -n "${running_started}" ] || [ -n "${terminated_reason}" ] || \
       [ "${phase:-}" = "Succeeded" ] || [ "${phase:-}" = "Failed" ]; then
      break
    fi

    if [ "${waiting_reason:-}" = "ImagePullBackOff" ] || \
       [ "${waiting_reason:-}" = "ErrImagePull" ] || \
       [ "${waiting_reason:-}" = "CreateContainerConfigError" ] || \
       [ "${waiting_reason:-}" = "CreateContainerError" ]; then
      write_bootstrap_evidence "startup_failure"
      echo "Phase 11 bootstrap pod failed during startup: ${waiting_reason}" >&2
      echo "See ${REPORT_DIR}/phase11_bootstrap_pod_describe_startup_failure.txt" >&2
      exit 1
    fi

    if [ "$((now_epoch - last_status_epoch))" -ge 30 ]; then
      echo "Bootstrap pod status: phase=${phase:-Unknown}, waiting=${waiting_reason:-none}, elapsed=${elapsed}s"
      kubectl -n "${NS}" get pods -l job-name=phase11-data-bootstrap -o wide \
        > "${REPORT_DIR}/phase11_bootstrap_pods_starting.txt" 2>&1 || true
      last_status_epoch="${now_epoch}"
    fi

    if [ "${elapsed}" -ge "${LOG_ATTACH_TIMEOUT_SECONDS}" ]; then
      write_bootstrap_evidence "startup_timeout"
      echo "Phase 11 bootstrap pod did not become loggable within ${LOG_ATTACH_TIMEOUT_SECONDS}s." >&2
      echo "See ${REPORT_DIR}/phase11_bootstrap_pods_starting.txt" >&2
      exit 1
    fi

    sleep "${BOOTSTRAP_POLL_SECONDS}"
  done

  echo "Streaming bootstrap logs from pod/${bootstrap_pod}"
  kubectl -n "${NS}" logs "pod/${bootstrap_pod}" -f \
    2> "${REPORT_DIR}/phase11_bootstrap_live_errors.log" \
    | tee "${REPORT_DIR}/phase11_bootstrap_live.log" &
  log_pid="$!"
else
  echo "Bootstrap pod was not created within 10 minutes; waiting for job condition anyway." >&2
fi

wait_start_epoch="$(date +%s)"
last_status_epoch=0
while true; do
  condition="$(job_condition)"
  now_epoch="$(date +%s)"
  elapsed=$((now_epoch - wait_start_epoch))

  if [ "${condition}" = "Complete" ]; then
    break
  fi

  if [ "${condition}" = "Failed" ]; then
    if [ -n "${log_pid}" ]; then
      kill "${log_pid}" >/dev/null 2>&1 || true
    fi
    write_bootstrap_evidence "failure"
    echo "Phase 11 bootstrap job failed. See ${REPORT_DIR}/phase11_bootstrap_failure.log" >&2
    exit 1
  fi

  if [ "$((now_epoch - last_status_epoch))" -ge 60 ]; then
    echo "Bootstrap job still running after ${elapsed}s"
    write_bootstrap_evidence "latest"
    last_status_epoch="${now_epoch}"
  fi

  if [ "${elapsed}" -ge "${BOOTSTRAP_TIMEOUT_SECONDS}" ]; then
    if [ -n "${log_pid}" ]; then
      kill "${log_pid}" >/dev/null 2>&1 || true
    fi
    write_bootstrap_evidence "timeout"
    echo "Phase 11 bootstrap job timed out after ${BOOTSTRAP_TIMEOUT_SECONDS}s." >&2
    echo "See ${REPORT_DIR}/phase11_bootstrap_timeout.log" >&2
    exit 1
  fi

  sleep "${BOOTSTRAP_POLL_SECONDS}"
done

if [ -n "${log_pid}" ]; then
  wait "${log_pid}" || true
  log_pid=""
fi
if [ -s "${REPORT_DIR}/phase11_bootstrap_live_errors.log" ]; then
  echo "Bootstrap live log stream had non-fatal interruptions; see ${REPORT_DIR}/phase11_bootstrap_live_errors.log"
fi

if [ -n "${bootstrap_pod}" ]; then
  kubectl -n "${NS}" logs "pod/${bootstrap_pod}" --all-containers=true \
    --pod-running-timeout=300s \
    > "${REPORT_DIR}/phase11_bootstrap.log"
else
  kubectl -n "${NS}" logs job/phase11-data-bootstrap --all-containers=true \
    --pod-running-timeout=300s \
    > "${REPORT_DIR}/phase11_bootstrap.log"
fi
kubectl -n "${NS}" get job phase11-data-bootstrap -o wide \
  > "${REPORT_DIR}/phase11_bootstrap_job.txt"
write_bootstrap_evidence "success"

echo "Phase 11 bootstrap completed. Evidence: ${REPORT_DIR}/phase11_bootstrap.log"
