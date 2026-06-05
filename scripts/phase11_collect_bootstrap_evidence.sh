#!/usr/bin/env bash
set -euo pipefail

NS="${NS:-staging}"
JOB_NAME="${JOB_NAME:-phase11-data-bootstrap}"
REPORT_DIR="${REPORT_DIR:-data/phase11/reports/eks}"
SUFFIX="${1:-latest}"

mkdir -p "${REPORT_DIR}"

SUMMARY_FILE="${REPORT_DIR}/phase11_bootstrap_evidence_${SUFFIX}.txt"
JOB_YAML_FILE="${REPORT_DIR}/phase11_bootstrap_job_${SUFFIX}.yaml"
JOB_DESCRIBE_FILE="${REPORT_DIR}/phase11_bootstrap_describe_${SUFFIX}.txt"
PODS_FILE="${REPORT_DIR}/phase11_bootstrap_pods_${SUFFIX}.txt"
EVENTS_FILE="${REPORT_DIR}/phase11_bootstrap_events_${SUFFIX}.txt"
JOB_EVENTS_FILE="${REPORT_DIR}/phase11_bootstrap_job_events_${SUFFIX}.txt"
JOB_STATUS_FILE="${REPORT_DIR}/phase11_bootstrap_status_${SUFFIX}.txt"
NO_POD_FILE="${REPORT_DIR}/phase11_bootstrap_no_pod_${SUFFIX}.txt"

echo "Collecting Phase 11 bootstrap evidence: namespace=${NS}, job=${JOB_NAME}, suffix=${SUFFIX}"

{
  date -u
  echo "namespace=${NS}"
  echo "job=${JOB_NAME}"
  echo
  kubectl -n "${NS}" get job "${JOB_NAME}" -o wide 2>&1 || true
  echo
  kubectl -n "${NS}" get pods -l "job-name=${JOB_NAME}" -o wide 2>&1 || true
} > "${SUMMARY_FILE}"

kubectl -n "${NS}" get job "${JOB_NAME}" -o yaml \
  > "${JOB_YAML_FILE}" 2>&1 || true
kubectl -n "${NS}" describe job/"${JOB_NAME}" \
  > "${JOB_DESCRIBE_FILE}" 2>&1 || true
kubectl -n "${NS}" get pods -l "job-name=${JOB_NAME}" -o wide \
  > "${PODS_FILE}" 2>&1 || true
kubectl -n "${NS}" get events --sort-by=.lastTimestamp \
  > "${EVENTS_FILE}" 2>&1 || true
kubectl -n "${NS}" get events \
  --field-selector "involvedObject.kind=Job,involvedObject.name=${JOB_NAME}" \
  --sort-by=.lastTimestamp \
  > "${JOB_EVENTS_FILE}" 2>&1 || true

job_uid="$(
  kubectl -n "${NS}" get job "${JOB_NAME}" \
    -o jsonpath='{.metadata.uid}' 2>/dev/null || true
)"
controller_uid="$(
  kubectl -n "${NS}" get job "${JOB_NAME}" \
    -o jsonpath='{.spec.selector.matchLabels.batch\.kubernetes\.io/controller-uid}' \
    2>/dev/null || true
)"
if [ -z "${controller_uid}" ]; then
  controller_uid="$(
    kubectl -n "${NS}" get job "${JOB_NAME}" \
      -o jsonpath='{.spec.selector.matchLabels.controller-uid}' 2>/dev/null || true
  )"
fi

{
  date -u
  echo "namespace=${NS}"
  echo "job=${JOB_NAME}"
  echo "job_uid=${job_uid:-missing}"
  echo "controller_uid=${controller_uid:-missing}"
  echo
  kubectl -n "${NS}" get job "${JOB_NAME}" \
    -o jsonpath='active={.status.active} succeeded={.status.succeeded} failed={.status.failed} startTime={.status.startTime} completionTime={.status.completionTime}{"\n"}' \
    2>&1 || true
  kubectl -n "${NS}" get job "${JOB_NAME}" \
    -o jsonpath='{range .status.conditions[*]}condition={.type} status={.status} reason={.reason} message={.message}{"\n"}{end}' \
    2>&1 || true
} > "${JOB_STATUS_FILE}"

pod_names="$(
  kubectl -n "${NS}" get pods -l "job-name=${JOB_NAME}" \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null || true
)"
if [ -z "${pod_names}" ] && [ -n "${controller_uid}" ]; then
  pod_names="$(
    kubectl -n "${NS}" get pods -l "batch.kubernetes.io/controller-uid=${controller_uid}" \
      -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null || true
  )"
fi
if [ -z "${pod_names}" ] && [ -n "${controller_uid}" ]; then
  pod_names="$(
    kubectl -n "${NS}" get pods -l "controller-uid=${controller_uid}" \
      -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null || true
  )"
fi

if [ -z "${pod_names}" ]; then
  {
    date -u
    echo "No pods found for selector job-name=${JOB_NAME} in namespace ${NS}."
    echo "job_uid=${job_uid:-missing}"
    echo "controller_uid=${controller_uid:-missing}"
    echo "Also checked controller UID labels when available."
    echo "This can happen if the job was deleted/recreated, the namespace is wrong, TTL cleanup removed the pod, or the pod was garbage-collected after the failed Job deadline."
    echo "Saved available job, pod-list, status, and event evidence in ${REPORT_DIR}."
  } > "${NO_POD_FILE}"
  echo "No bootstrap pods found. See ${NO_POD_FILE}"
  exit 0
fi

while IFS= read -r pod_name; do
  if [ -z "${pod_name}" ]; then
    continue
  fi

  safe_pod_name="${pod_name//[^A-Za-z0-9_.-]/_}"
  kubectl -n "${NS}" describe "pod/${pod_name}" \
    > "${REPORT_DIR}/phase11_bootstrap_pod_describe_${safe_pod_name}_${SUFFIX}.txt" 2>&1 || true
  kubectl -n "${NS}" logs "pod/${pod_name}" --all-containers=true \
    --tail=-1 --pod-running-timeout=300s \
    > "${REPORT_DIR}/phase11_bootstrap_pod_${safe_pod_name}_${SUFFIX}.log" 2>&1 || true
  kubectl -n "${NS}" logs "pod/${pod_name}" --all-containers=true --previous \
    --tail=-1 --pod-running-timeout=300s \
    > "${REPORT_DIR}/phase11_bootstrap_pod_${safe_pod_name}_${SUFFIX}_previous.log" 2>&1 || true
done <<< "${pod_names}"

echo "Phase 11 bootstrap evidence saved under ${REPORT_DIR}"
