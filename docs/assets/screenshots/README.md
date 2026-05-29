# Screenshots

Store real screenshots from a live local, staging, or production run in this directory.

Do not commit placeholders or mock images as production evidence. The expected files are listed in `docs/demo_video.md`.

For Grafana screenshots, use the dashboards after they have been refreshed from the current provisioning JSON:

- `grafana-rag-quality.png`: online groundedness/relevancy proxies, retrieved evidence count, user feedback, and guardrail blocks.
- `grafana-system-health.png`: completed async query latency from `orchestrator_query_latency_seconds`, query submission rate, query error rate, and active pods.
- `grafana-cost-usage.png`: actual LLM token usage only, estimated OpenAI cost, fast-path share, and cache hit rate.
