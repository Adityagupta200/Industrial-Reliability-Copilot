# Incident Documentation

This document records realistic simulated incidents for the Industrial Reliability Copilot. They are written in production incident-review style so future maintainers can rehearse detection, investigation, mitigation, validation, and prevention without pretending these were customer outages.

## Incident 1: Faithfulness Drop After Prompt Change

| Field | Details |
| --- | --- |
| Date | February 3, 2026 |
| Severity | SEV-2 quality regression |
| Detection | Grafana alert `LowFaithfulness` fired after average `llm_faithfulness` dropped to 0.72, below the 0.75 alert threshold. |
| User impact | Answers for root-cause analysis included plausible maintenance claims that were not clearly supported by retrieved manuals or procedures. |
| Time to resolution | 2 hours |

### Symptoms

- Operators reported answers that sounded correct but did not cite the relevant Pump P-23 procedure.
- Prometheus showed a downward trend in `llm_faithfulness` beginning February 2.
- `query_logs` showed that roughly 15 percent of sampled root-cause queries had low groundedness or weak citations.
- LangSmith traces showed the model using general bearing-maintenance knowledge when the retrieved context was thin.

### Investigation

1. Checked Grafana LLM Quality dashboard for faithfulness trend and affected chain labels.
2. Compared `llm_faithfulness` with `orchestrator_query_latency_seconds` to rule out a provider latency incident.
3. Queried `query_logs` for recent low-scoring answers and inspected `retrieved_contexts`.
4. Reviewed LangSmith traces for prompt, retrieved docs, model response, and output-guardrail result.
5. Diffed prompt templates and found that a v1.1 prompt draft had weakened the instruction requiring source tags from the allowed list.

### Root Cause

A prompt change removed explicit wording that every hypothesis must cite one of the allowed `DOC_*` tags. The LLM still produced useful-looking diagnoses, but source grounding became inconsistent.

### Fix

- Rolled back the root-cause prompt to v1.0.
- Restored the instruction that every hypothesis must cite exactly one allowed source tag.
- Kept chain-level citation validation that rejects missing or hallucinated `DOC_*` citations.

### Validation

- Re-ran the offline evaluation suite.
- Faithfulness recovered to 0.86+ aggregate in the committed Ragas artifact.
- Response contract checks passed for the Pump P-23 root-cause case.
- No new guardrail failures appeared in the smoke run.

### Prevention

- Added a policy that prompt changes require a golden-set diff and LangSmith trace review.
- Added more grounding-focused cases to the planned evaluation backlog.
- Kept CI thresholds in `scripts/check_thresholds.py` at faithfulness `>= 0.85`.
- Required every future prompt to include source-tag constraints and output-schema examples.

## Incident 2: Latency Spike From Excessive Context

| Field | Details |
| --- | --- |
| Date | February 6, 2026 |
| Severity | SEV-2 latency regression |
| Detection | Alertmanager fired `HighLatency` after orchestrator p95 latency exceeded 4 seconds for more than 10 minutes. |
| User impact | Engineers experienced slow query completion and delayed status polling. |
| Time to resolution | 45 minutes |

### Symptoms

- Grafana showed normal RAG service request latency but high end-to-end orchestrator latency.
- OpenAI status was normal; the issue reproduced with local fallback as well.
- LangSmith traces showed large prompts with long retrieved chunks.
- Query logs showed repeated root-cause requests carrying 5,000+ token contexts.

### Investigation

1. Split latency by service: gateway, orchestrator, RAG service, anomaly service, and provider calls.
2. Verified Qdrant retrieval latency remained near normal while LLM latency increased.
3. Inspected retrieved contexts for affected queries.
4. Checked recent ingestion and chunking configuration.
5. Found that chunk overlap and context limits had increased, causing redundant context to reach the LLM.

### Root Cause

The context window was inflated by larger chunk overlap and too many retrieved chunks. LLM calls slowed because the prompt approached the practical context-size limit.

### Fix

- Reduced retrieval candidate defaults to bounded values: `semantic_k=15`, `keyword_k=15`, and `out_k=8`.
- Kept root-cause chain context limited to the top 5 deduplicated documents.
- Preserved `max_context_chars_per_chunk=6000` and ensured chunks are truncated before upsert.

### Validation

- Replayed affected root-cause queries and confirmed prompts were smaller.
- p95 query latency dropped back under the 2 second target in the local smoke scenario.
- Faithfulness did not regress in the Ragas gate.

### Prevention

- Added alert review for context size and retrieved document count.
- Required retrieval-tuning changes to include before/after latency and quality results.
- Kept RRF defaults conservative to avoid cross-encoder or LLM CPU spikes.

## Incident 3: EKS Deployment Blocked By Invalid Database DSN

| Field | Details |
| --- | --- |
| Date | February 10, 2026 |
| Severity | SEV-3 deployment incident |
| Detection | GitHub Actions CD preflight failed before production rollout. |
| User impact | No production impact. Deployment was blocked before rollout. |
| Time to resolution | 1 hour 10 minutes |

### Symptoms

- CD workflow reported that `POSTGRES_DSN` or `INCIDENTS_DB_DSN` still pointed to `localhost`, `127.0.0.1`, or the in-cluster host `postgres`.
- Staging manifests rendered correctly, but the orchestrator init container would not be able to reach RDS with those DSNs.
- Terraform output showed the correct RDS endpoint, but repository secrets had not been updated.

### Investigation

1. Reviewed the failing CD preflight logs.
2. Checked `docs/deployment_secrets.md` for expected DSN format.
3. Retrieved `db_instance_endpoint` from Terraform output.
4. Confirmed the orchestrator uses `postgresql+asyncpg` for `INCIDENTS_DB_DSN` while the RAG service uses `postgresql+psycopg` for `POSTGRES_DSN`.
5. Confirmed RDS security group allows PostgreSQL access only from the VPC CIDR.

### Root Cause

The production GitHub environment still contained local Docker Compose DSNs. Those values work for local development but are invalid for EKS pods that must connect to RDS through the private endpoint.

### Fix

- Updated GitHub environment secrets with RDS endpoint DSNs.
- Kept passwords URL-encoded in both DSNs.
- Re-ran CD preflight and staging deployment.

### Validation

- CD preflight passed required secret and DNS checks.
- The orchestrator init container reached the telemetry database.
- Staging smoke pod passed `api-gateway` and `llm-orchestrator` readiness checks.
- Production rollout monitor completed without triggering rollback.

### Prevention

- Kept DSN preflight checks in `.github/workflows/cd.yml`.
- Documented required repository secrets and Terraform outputs in `docs/deployment_secrets.md`.
- Kept Kubernetes init-container validation for `INCIDENTS_DB_DSN` so a bad secret fails before the app starts.
- Required deployment runbooks to verify `terraform output -raw db_instance_endpoint` before updating secrets.

## Common Debugging Playbook

Use this sequence for future incidents:

1. Confirm user impact and severity.
2. Check Grafana for latency, error rate, quality, guardrails, and feedback.
3. Query Prometheus directly when a dashboard panel is ambiguous.
4. Inspect LangSmith traces for the failing chain.
5. Query `query_logs` and `async_job_states` by trace or job ID.
6. Determine the layer: retrieval, prompt, model provider, parser, guardrail, database, Kubernetes, or CI/CD.
7. Apply the smallest safe mitigation: rollback prompt, reduce context, fail over provider, undo rollout, or block unsafe requests.
8. Validate with Ragas, deterministic contracts, smoke tests, and targeted replay.
9. Add a regression case or alert before closing the incident.
