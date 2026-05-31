# Screenshots

Store real screenshots from a live local, staging, or production run in this directory.

Do not commit placeholders or mock images as production evidence. The expected
files and naming convention are listed here so README links always point to real
committed assets.

Use `slug.png` when one screenshot is enough. Use `slug-1.png`, `slug-2.png`,
and so on when one screen cannot show the evidence legibly. Keep README links
pointing at the actual committed filenames.

Current evidence set:

| Evidence | Files | Notes |
| --- | --- | --- |
| Query response | `query-interface-1.png`, `query-interface-2.png` | Pump P-23 response, citations, latency, guardrails, raw context, and evidence summary. |
| GitHub Actions Ragas | `github-actions-ragas-1.png`, `github-actions-ragas-2.png`, `github-actions-ragas-3.png` | Passing workflow, threshold logs, and artifacts. |
| Grafana RAG quality | `grafana-rag-quality-1.png`, `grafana-rag-quality-2.png` | Online quality, feedback, retrieval, and guardrail panels. |
| System health | `system-health-1.png` | Live service health and runtime metrics. |
| Prometheus targets | `prometheus-targets.png` | Healthy scrape targets for the local stack. |
| LangSmith trace | `langsmith-trace.png` | Retrieval, root-cause fast path, and output groundedness trace. |

For Grafana screenshots, use dashboards after they have been refreshed from the
current provisioning JSON. Capture cost/routing as `grafana-cost-usage.png` or
`grafana-cost-usage-*.png` only after real token/cost metrics are present; do
not create a mock cost screenshot.
