# Industrial Reliability Copilot

Production-grade Reliability Copilot: a microservice-based system that combines:
- Classical anomaly/RUL inference (Anomaly Service)
- Retrieval-Augmented Generation (RAG Service + Vector DB)
- LLM orchestration & safety guardrails (LLM Orchestrator)
- API Gateway
- Evaluation (offline/online)

## Repository layout

- data/
  - raw/               # source docs, incidents, runbooks
  - processed/         # cleaned text/chunks/derived artifacts
  - golden_test_set/   # evaluation datasets (later phases)
- src/
  - anomaly_service/
  - rag_service/
  - llm_orchestrator/
  - api_gateway/
  - evaluation/
- infra/
  - terraform/
  - kubernetes/
  - monitoring/
- tests/
  - unit/
  - integration/
  - regression/
- docs/
  - architecture/
  - evaluation/
  - incidents/

## Quick start (dev)

1) Create and activate a Python 3.11+ virtualenv
2) Install dependencies
3) Run linters/tests

See `make help` after setup.
