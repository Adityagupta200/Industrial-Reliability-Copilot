# Phase 11 Launch Runbook

This runbook turns Phase 11 into reproducible evidence. Do not claim a live
metric publicly until the corresponding JSON report exists under
`data/phase11/reports/` and passed.

## Scope

Phase 11 covers:

- End-to-end correctness and grounding over 50 diverse queries.
- Sustained load at 50 QPS with 50 simulated clients.
- Failure drills for malformed input, rate limiting, Qdrant degradation, and LLM fallback.
- Security audit: secrets, dependency audit, container scan, and access-control evidence.
- Portfolio packaging: public repo readiness, README/docs polish, LinkedIn/resume/blog assets.

The demo video from Phase 10 is intentionally separate. Add a video link to the
README only after recording and uploading a real demo.

## Final Validated Run

The latest Phase 11 evidence set was captured from a short-lived AWS EKS staging
deployment on June 5, 2026, then the paid AWS resources were destroyed. The
public claims should stay aligned with these artifacts:

| Check | Result | Evidence |
| --- | ---: | --- |
| Public EKS smoke case | 1/1 passed; p95 69.16 ms | `data/phase11/reports/eks_smoke_e2e_report.json` |
| 50-query launch suite | 50/50 passed; p95 878.55 ms | `data/phase11/reports/e2e_validation_report.json` |
| 600-second load test | 30,000/30,000 completed; 50.0 observed QPS; 1.000 success rate; wall p95 635.79 ms | `data/phase11/reports/load_test_report.json` |
| Failure drill | Passed malformed-input and rate-limit checks | `data/phase11/reports/failure_drill_report.json` |
| Security audit | Passed with no failed checks | `data/phase11/reports/security_audit_report.json` |
| Container scan | Passed; no actionable HIGH/CRITICAL vulnerabilities across 4 service images | `data/phase11/reports/container_scan_report.json` |
| Final launch gate | Passed with no failed checks | `data/phase11/reports/launch_gate_final_report.json` |
| HPA behavior | `llm-orchestrator` scaled from 4 to 5 replicas; other services stayed at baseline | `data/phase11/reports/hpa_samples_during_load.txt` |

Do not turn the HPA result into a broader claim that every service autoscaled.
The defensible claim is that HPA metrics were captured and the orchestrator
scaled under the validated 50 QPS workload.

## Fresh Local Validation

Start from a clean local deployment:

```powershell
Copy-Item .env.example .env -ErrorAction SilentlyContinue
docker compose up -d postgres qdrant
docker compose run --rm rag-service python -m rag_service.db.init_db
docker compose run --rm rag-service python -m rag_service.db.ingest_incidents
docker compose run --rm rag-service python -m rag_service.ingestion.pipeline
docker compose up -d
```

Verify readiness:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/ready
Invoke-RestMethod http://127.0.0.1:8080/health/ready
Invoke-RestMethod http://127.0.0.1:8002/health/ready
Invoke-RestMethod http://127.0.0.1:8001/health/ready
```

Run static launch and security checks:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\phase11_security_audit.py
.\.venv\Scripts\python.exe scripts\phase11_launch_gate.py
```

Run the 50-query E2E validation:

```powershell
.\.venv\Scripts\python.exe scripts\phase11_e2e_validation.py `
  --base-url http://127.0.0.1:8000 `
  --require-count 50
```

Run the sustained load test. Use `--duration-seconds 600` for final evidence;
use a shorter duration only while debugging the harness:

```powershell
.\.venv\Scripts\python.exe scripts\phase11_load_test.py `
  --base-url http://127.0.0.1:8000 `
  --duration-seconds 600 `
  --target-qps 50 `
  --client-count 50
```

Run failure drills:

```powershell
.\.venv\Scripts\python.exe scripts\phase11_failure_drill.py `
  --base-url http://127.0.0.1:8000 `
  --rag-url http://127.0.0.1:8002 `
  --run-llm-fallback-unit
```

Run the optional Qdrant outage drill only when you are comfortable letting the
script stop and restart the local Qdrant container:

```powershell
.\.venv\Scripts\python.exe scripts\phase11_failure_drill.py `
  --base-url http://127.0.0.1:8000 `
  --rag-url http://127.0.0.1:8002 `
  --run-qdrant-failure `
  --qdrant-container irc-qdrant
```

After the live reports exist, enforce the final launch gate:

```powershell
.\.venv\Scripts\python.exe scripts\phase11_launch_gate.py --require-live-evidence
```

## Security Audit

The security audit checks:

- Secret-shaped values in tracked files.
- Raw Git history markers such as `sk-` for review evidence.
- Kubernetes Secret usage instead of literal runtime secrets.
- Obvious secret-bearing log statements.
- Access-control evidence for tenant filters, PII redaction, and rate limiting.
- Dependency audit findings, with fixable vulnerabilities blocked and any
  no-fix residual risk documented in `security/vulnerability_acceptances.json`.

Optional scanners are available when installed:

```powershell
.\.venv\Scripts\python.exe scripts\phase11_security_audit.py `
  --run-pip-audit `
  --requirement src/api_gateway/requirements.txt `
  --requirement src/llm_orchestrator/requirements.txt `
  --requirement requirements-rag.txt `
  --requirement src/anomaly_service/requirements.txt `
  --run-trivy-fs `
  --run-gitleaks `
  --strict-optional-tools
```

The current PyTorch line may include no-fix advisories in the public advisory
database. Those may pass only when they match the explicit, expiring acceptance
policy and remain mitigated by the anomaly service's safetensors-only default
model loading path. Fixable dependency vulnerabilities must not be accepted.

For local container image scans, use the Phase 11 wrapper instead of calling
`trivy` directly. It uses host Trivy when installed and otherwise falls back to
the official Trivy Docker image, so Windows/Git Bash machines do not need a
separate Trivy installation:

```bash
export PYTHONPATH=src
PY=${PY:-python}

$PY scripts/phase11_container_scan.py \
  --output data/phase11/reports/container_scan_report.json
```

The scan builds `anomaly_service`, `api_gateway`, `llm_orchestrator`, and
`rag_service`, then fails on actionable `HIGH` or `CRITICAL` vulnerabilities
after applying `--ignore-unfixed`. The first Dockerized run may pull
`aquasec/trivy`; set `PHASE11_TRIVY_IMAGE` to pin a scanner image for a
particular release. The wrapper sets a longer Trivy timeout by default because
PyTorch shared objects can exceed the scanner's short default timeout on local
Windows Docker hosts. On memory-constrained machines, add
`--trivy-parallel 1` to scan one analyzer worker at a time. Dockerized scans
use an isolated cache namespace per invocation so stale local Trivy cache locks
do not turn into false launch failures.

The anomaly-service audit is included in the GitHub Phase 11 workflow because
its service image targets Python 3.11 and a PyTorch wheel that is not resolvable
from the local Python 3.12 development environment.

The manual GitHub workflow `.github/workflows/phase11-launch.yml` also runs the
static gate, dependency audit, optional live checks, and container scans.

## Evidence Files

| Evidence | Path |
| --- | --- |
| EKS smoke E2E report | `data/phase11/reports/eks_smoke_e2e_report.json` |
| 50-query validation report | `data/phase11/reports/e2e_validation_report.json` |
| Sustained load report | `data/phase11/reports/load_test_report.json` |
| HPA before/during/after evidence | `data/phase11/reports/hpa_before_load.txt`, `data/phase11/reports/hpa_samples_during_load.txt`, `data/phase11/reports/hpa_after_load.txt` |
| Failure drill report | `data/phase11/reports/failure_drill_report.json` |
| Security audit report | `data/phase11/reports/security_audit_report.json` |
| Container scan report | `data/phase11/reports/container_scan_report.json` |
| Bootstrap, ECR, node, log, and destroy evidence | `data/phase11/reports/eks/` and `data/phase11/reports/eks_destroy_verification.txt` |
| Launch gate report | `data/phase11/reports/launch_gate_final_report.json` |

Commit only reports that came from a real run you are willing to defend in an
interview. Do not commit placeholder reports.

## Acceptance Mapping

| Phase 11 requirement | Implementation |
| --- | --- |
| 50 diverse queries | `data/phase11/e2e_queries.json` plus `scripts/phase11_e2e_validation.py` |
| Correct, grounded responses | Per-case expected terms, forbidden terms, source count, and raw-context checks |
| Latency checks | Per-case `max_latency_ms` plus p50/p95 summary in the E2E report |
| Load test at 50 QPS | `scripts/phase11_load_test.py --target-qps 50 --client-count 50 --duration-seconds 600`; latest report completed 30,000/30,000 requests at 50.0 observed QPS with wall p95 635.79 ms |
| Rate limit returns 429 | `scripts/phase11_failure_drill.py` single-client burst test |
| Malformed input handling | `scripts/phase11_failure_drill.py` invalid JSON test |
| Qdrant failure behavior | Optional Qdrant outage drill verifies RAG service stays live and hybrid retrieval degrades |
| LLM timeout/fallback | `tests/unit/test_llm_client_routing.py` and failure-drill unit hook |
| Secrets and dependency scan | `scripts/phase11_security_audit.py` and manual launch workflow |
| Container scan | `scripts/phase11_container_scan.py` builds each service image and scans with host or Dockerized Trivy |
| Portfolio packaging | `docs/portfolio.md` |
