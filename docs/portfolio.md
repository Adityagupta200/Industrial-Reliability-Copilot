# Portfolio Packaging

Use this file as the launch checklist and source material for public packaging.
Replace placeholders only with values from committed reports or live evidence.

## GitHub Repository

Recommended repository settings:

- Visibility: public, or private with explicit recruiter access.
- Profile: pin `Industrial-Reliability-Copilot`.
- Topics: `machine-learning`, `llmops`, `rag`, `kubernetes`, `production-ml`,
  `fastapi`, `qdrant`, `ragas`, `terraform`, `mlops`.
- README: keep architecture, metrics, quick start, screenshots, and evidence paths visible.
- Actions: keep CI, CD, and Phase 11 Launch Validation workflows enabled.

Do not add a demo video badge or link until a real uploaded demo exists.

## Current Evidence-Based Claims

These are backed by committed files at the time of writing:

| Claim | Evidence |
| --- | --- |
| Ragas faithfulness is `1.000` on the committed eval artifact | `ragas_results.json` |
| Ragas answer relevancy is `0.933` on the committed eval artifact | `data/evaluation_results/summary.json` |
| Context precision and recall are both approximately `1.000` | `ragas_results.json` |
| Safety and response-contract pass rates are `1.000` | `data/evaluation_results/evaluation_report.json` |
| The local corpus contains 10 raw maintenance PDFs | `data/raw/manuals/` |
| The incident corpus contains 180 records | `data/raw/incidents/synthetic_incidents.csv` |
| Docker, Kubernetes, Terraform, Prometheus, Grafana, CI/CD, and rollback paths exist | `docker-compose.yml`, `infra/`, `.github/workflows/` |
| 50-query Phase 11 E2E validation passed 50/50 with p95 latency 878.55 ms | `data/phase11/reports/e2e_validation_report.json` |
| 600-second load validation completed 30,000/30,000 requests at 50.0 observed QPS with 1.000 success rate and 635.79 ms wall p95 | `data/phase11/reports/load_test_report.json` |
| Failure drill, security audit, container scan, and final launch gate passed | `data/phase11/reports/failure_drill_report.json`, `data/phase11/reports/security_audit_report.json`, `data/phase11/reports/container_scan_report.json`, `data/phase11/reports/launch_gate_final_report.json` |
| EKS HPA evidence shows `llm-orchestrator` scaled from 4 to 5 replicas during load | `data/phase11/reports/hpa_samples_during_load.txt` |

Claims to phrase carefully:

- Do not claim long-term uptime, customer production traffic, regional failover,
  or multi-day SRE readiness; the live evidence is a short-lived EKS staging
  validation.
- Do not claim every service autoscaled. The captured HPA evidence shows the
  `llm-orchestrator` HPA scaled from 4 to 5 replicas; other services stayed at
  their configured baselines.
- Do not claim end-user JWT/OIDC authentication or complete RBAC enforcement as
  implemented. Those remain documented production hardening items.
- Do not add a demo video badge or link until a real uploaded demo exists.

## LinkedIn Post Draft

```text
I just finished the launch validation layer for Industrial Reliability Copilot,
a production-oriented RAG + MLOps platform for industrial maintenance triage.

The system combines:
- FastAPI microservices for anomaly/RUL inference, RAG retrieval, and LLM orchestration
- Qdrant + BM25 hybrid retrieval over maintenance manuals, procedures, and incident history
- Root-cause, remediation, and historical-search chains with citation/grounding guardrails
- Ragas evaluation, deterministic contract tests, online telemetry, Prometheus/Grafana, and LangSmith traces
- Docker, Kubernetes, Terraform, CI/CD, rollout monitoring, and rollback automation

Current measured quality and launch evidence:
- Faithfulness: 1.000
- Answer relevancy: 0.933
- Context precision: 1.000
- Context recall: 1.000
- Safety pass rate: 1.000
- Phase 11 E2E: 50/50 passed
- Load validation: 50.0 observed QPS for 600 seconds, 30,000/30,000 completed,
  wall p95 latency 635.79 ms

The hardest part was making the project defensible as an engineering system, not only a demo:
grounded answers, failure handling, rate limits, security checks, evaluation gates, and launch evidence.

GitHub: <repo link>
Demo: <add only after uploading a real demo>

#MachineLearning #LLMOps #GenAI #RAG #MLOps #Kubernetes
```

## Resume Bullets

Use bullets that match the evidence you can show. Safe current bullets:

- Built Industrial Reliability Copilot, a production-oriented RAG + MLOps platform for
  predictive-maintenance triage across 10 maintenance PDFs, procedure runbooks, and 180 incident records.
- Implemented hybrid retrieval with dense embeddings, BM25 keyword search, Reciprocal Rank Fusion,
  and reranking hooks across Qdrant-backed maintenance evidence.
- Implemented three LLM workflows for root-cause analysis, remediation guidance, and historical
  incident search with prompt versioning, provider fallback, citation checks, and groundedness guardrails.
- Added offline Ragas evaluation and deterministic safety/contract gates with committed scores:
  faithfulness `1.000`, answer relevancy `0.933`, context precision `1.000`, context recall `1.000`.
- Built cloud-native deployment assets with Docker, Kubernetes HPAs/probes, Terraform AWS infrastructure,
  GitHub Actions CI/CD, rollout monitoring, and rollback automation.
- Added observability with Prometheus metrics, Grafana dashboards, alert rules, query logs, feedback
  telemetry, and LangSmith trace capture.
- Validated 50 end-to-end launch queries with `1.000` pass rate across root-cause, remediation,
  historical, guardrail, and edge-case scenarios.
- Load-tested the EKS staging deployment for 600 seconds at `50.0` observed QPS with
  `30,000/30,000` completed requests, `1.000` success rate, and `635.79 ms` wall-clock p95 latency.
- Captured Kubernetes HPA evidence during load; `llm-orchestrator` scaled from 4 to 5 replicas while
  other services stayed within baseline capacity.
- Verified malformed-input handling, rate-limit enforcement, security audit, container scan, and final
  launch gate with no failed checks in Phase 11 artifacts.

## Blog Outline

Suggested title: `How I Built a Production RAG Evaluation and Launch Gate for an Industrial Reliability Copilot`

Outline:

1. Problem: maintenance triage needs grounded answers, not generic chatbot advice.
2. Architecture: ML inference, RAG, LLM chains, telemetry, and deployment layers.
3. Retrieval: why hybrid search beats semantic-only for asset IDs and maintenance terminology.
4. Guardrails: prompt injection, PII redaction, safety checks, citations, groundedness.
5. Evaluation: golden set, Ragas metrics, deterministic contracts, CI quality gate.
6. Launch validation: 50-query suite, sustained load, failure drills, security audit.
7. Lessons learned: where latency, grounding, and deployment reliability actually broke.
8. Next improvements: auth/OIDC, stronger tenant isolation, managed vector DB, public demo.
