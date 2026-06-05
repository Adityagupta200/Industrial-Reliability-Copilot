#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = Path("data/phase11/reports/launch_gate_report.json")


@dataclass
class GateCheck:
    name: str
    passed: bool
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _count_panels(value: Any) -> int:
    if isinstance(value, dict):
        total = 1 if value.get("type") and value.get("title") else 0
        return total + sum(_count_panels(item) for item in value.values())
    if isinstance(value, list):
        return sum(_count_panels(item) for item in value)
    return 0


def check_data_volume(repo: Path) -> list[GateCheck]:
    manual_count = len(list((repo / "data" / "raw" / "manuals").glob("*.pdf")))
    incident_file = repo / "data" / "raw" / "incidents" / "synthetic_incidents.csv"
    incident_count = 0
    if incident_file.exists():
        with incident_file.open(newline="", encoding="utf-8") as handle:
            incident_count = sum(1 for _ in csv.DictReader(handle))

    return [
        GateCheck(
            "manual_pdf_count",
            manual_count >= 10,
            f"{manual_count} raw manual PDFs available; target >= 10.",
            {"manual_count": manual_count},
        ),
        GateCheck(
            "incident_record_count",
            incident_count >= 100,
            f"{incident_count} incident records available; target >= 100.",
            {"incident_count": incident_count},
        ),
    ]


def check_validation_cases(repo: Path) -> GateCheck:
    path = repo / "data" / "phase11" / "e2e_queries.json"
    cases = _load_json(path) if path.exists() else []
    chain_counts: dict[str, int] = {}
    for case in cases:
        chain = str((case.get("payload") or {}).get("chain") or "unknown")
        chain_counts[chain] = chain_counts.get(chain, 0) + 1
    return GateCheck(
        "phase11_50_query_set",
        len(cases) >= 50
        and all(
            chain_counts.get(name, 0) > 0 for name in ["root_cause", "remediation", "historical"]
        ),
        f"{len(cases)} Phase 11 query cases with chain distribution {chain_counts}.",
        {"case_count": len(cases), "chain_counts": chain_counts},
    )


def check_architecture(repo: Path) -> list[GateCheck]:
    dockerfiles = list((repo / "src").glob("*/Dockerfile"))
    compose = (repo / "docker-compose.yml").read_text(encoding="utf-8")
    app_services = ["api-gateway", "llm-orchestrator", "rag-service", "anomaly-service"]
    k8s = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((repo / "infra" / "kubernetes").glob("*.yaml"))
    )
    terraform_files = [
        repo / "infra" / "terraform" / name
        for name in ["main.tf", "variables.tf", "outputs.tf", "provider.tf"]
    ]

    return [
        GateCheck(
            "dockerized_app_services",
            len(dockerfiles) >= 4 and all(service in compose for service in app_services),
            f"{len(dockerfiles)} service Dockerfiles and compose entries for {app_services}.",
            {"dockerfiles": [str(path.relative_to(repo)) for path in dockerfiles]},
        ),
        GateCheck(
            "kubernetes_manifests",
            all(
                token in k8s
                for token in [
                    "readinessProbe:",
                    "livenessProbe:",
                    "HorizontalPodAutoscaler",
                    "LoadBalancer",
                ]
            ),
            "Kubernetes manifests include probes, HPAs, and public gateway service.",
        ),
        GateCheck(
            "terraform_iac",
            all(path.exists() for path in terraform_files),
            "Terraform stack contains provider, variables, main resources, and outputs.",
            {"files": [str(path.relative_to(repo)) for path in terraform_files]},
        ),
    ]


def check_rag_and_guardrails(repo: Path) -> list[GateCheck]:
    files = {
        "hybrid_retriever": repo / "src" / "rag_service" / "retrieval" / "hybrid_retriever.py",
        "semantic_retriever": repo / "src" / "rag_service" / "retrieval" / "semantic_retriever.py",
        "keyword_retriever": repo / "src" / "rag_service" / "retrieval" / "keyword_retriever.py",
        "reranker": repo / "src" / "rag_service" / "retrieval" / "reranker.py",
        "input_guardrails": repo / "src" / "llm_orchestrator" / "guardrails" / "input_filters.py",
        "output_guardrails": repo / "src" / "llm_orchestrator" / "guardrails" / "output_filters.py",
    }
    chains = ["root_cause_chain.py", "remediation_chain.py", "historical_chain.py"]
    chain_files = [repo / "src" / "llm_orchestrator" / "chains" / name for name in chains]
    return [
        GateCheck(
            "hybrid_retrieval_stack",
            all(path.exists() for path in files.values()),
            "Semantic, keyword, hybrid, reranker, and guardrail modules exist.",
            {"files": {name: str(path.relative_to(repo)) for name, path in files.items()}},
        ),
        GateCheck(
            "three_llm_chains",
            all(path.exists() for path in chain_files),
            "Root-cause, remediation, and historical-search chains are implemented.",
            {"files": [str(path.relative_to(repo)) for path in chain_files]},
        ),
    ]


def check_evaluation_artifacts(repo: Path) -> GateCheck:
    path = repo / "ragas_results.json"
    if not path.exists():
        return GateCheck("ragas_artifact", False, "ragas_results.json is missing.")
    results = _load_json(path)
    ragas = results.get("ragas") or {}
    safety = results.get("safety") or {}
    contracts = results.get("response_contracts") or {}
    thresholds = {
        "faithfulness": 0.85,
        "answer_relevancy": 0.85,
        "context_precision": 0.80,
        "context_recall": 0.80,
    }
    failures = [
        name for name, threshold in thresholds.items() if float(ragas.get(name, 0.0)) < threshold
    ]
    if float(safety.get("pass_rate", 0.0)) < 1.0:
        failures.append("safety_pass_rate")
    if float(contracts.get("pass_rate", 0.0)) < 1.0:
        failures.append("response_contract_pass_rate")

    return GateCheck(
        "ragas_quality_gate_artifact",
        not failures,
        (
            "Committed Ragas and deterministic checks meet launch thresholds."
            if not failures
            else f"Evaluation failures: {failures}."
        ),
        {"ragas": ragas, "safety": safety, "response_contracts": contracts},
    )


def check_observability(repo: Path) -> list[GateCheck]:
    dashboard_paths = sorted((repo / "infra" / "monitoring" / "grafana-dashboards").glob("*.json"))
    panel_count = 0
    for path in dashboard_paths:
        panel_count += _count_panels(_load_json(path))
    alerts = repo / "infra" / "monitoring" / "prometheus-alerts.yaml"
    alerts_text = alerts.read_text(encoding="utf-8") if alerts.exists() else ""
    return [
        GateCheck(
            "grafana_dashboards",
            len(dashboard_paths) >= 3 and panel_count >= 10,
            f"{len(dashboard_paths)} dashboards with approximately {panel_count} panels.",
            {"dashboards": [path.name for path in dashboard_paths], "panel_count": panel_count},
        ),
        GateCheck(
            "prometheus_alerts",
            all(
                token in alerts_text
                for token in ["HighLatency", "LowFaithfulness", "HighErrorRate"]
            ),
            "Latency, quality, and error alerts are configured.",
        ),
    ]


def check_docs_and_portfolio(repo: Path) -> list[GateCheck]:
    required_docs = [
        "README.md",
        "docs/architecture.md",
        "docs/evaluation.md",
        "docs/incidents.md",
        "docs/deployment_secrets.md",
        "docs/launch.md",
        "docs/portfolio.md",
    ]
    missing = [path for path in required_docs if not (repo / path).exists()]
    screenshots = list((repo / "docs" / "assets" / "screenshots").glob("*.png"))
    readme = (repo / "README.md").read_text(encoding="utf-8")
    return [
        GateCheck(
            "launch_documentation",
            not missing,
            (
                "README, architecture, evaluation, incidents, deployment, launch, and portfolio docs exist."
                if not missing
                else f"Missing docs: {missing}."
            ),
            {"missing": missing},
        ),
        GateCheck(
            "screenshot_evidence",
            len(screenshots) >= 8 and "docs/assets/screenshots" in readme,
            f"{len(screenshots)} committed screenshot PNGs are linked from README.",
            {"screenshot_count": len(screenshots)},
        ),
    ]


def check_live_evidence(repo: Path, *, required: bool) -> list[GateCheck]:
    report_specs = {
        "phase11_e2e_report": repo / "data" / "phase11" / "reports" / "e2e_validation_report.json",
        "phase11_load_report": repo / "data" / "phase11" / "reports" / "load_test_report.json",
        "phase11_failure_report": repo
        / "data"
        / "phase11"
        / "reports"
        / "failure_drill_report.json",
        "phase11_security_report": repo
        / "data"
        / "phase11"
        / "reports"
        / "security_audit_report.json",
        "phase11_container_scan_report": repo
        / "data"
        / "phase11"
        / "reports"
        / "container_scan_report.json",
    }
    checks: list[GateCheck] = []
    for name, path in report_specs.items():
        if not path.exists():
            checks.append(
                GateCheck(
                    name,
                    not required,
                    (
                        f"{path} is missing."
                        if required
                        else f"{path} not present; run live Phase 11 checks before public launch."
                    ),
                )
            )
            continue
        payload = _load_json(path)
        passed = bool(payload.get("passed", payload.get("failed", 1) == 0))
        if name == "phase11_e2e_report":
            passed = int(payload.get("failed", 1)) == 0 and int(payload.get("total", 0)) >= 50
        if name == "phase11_load_report":
            passed = bool(payload.get("passed_thresholds", False))
        if name == "phase11_container_scan_report":
            passed = bool(payload.get("passed", False)) and not payload.get("failed_images")
        checks.append(
            GateCheck(
                name,
                passed,
                f"{path} present; passed={passed}.",
                {"path": str(path.relative_to(repo))},
            )
        )
    return checks


def run_gate(repo: Path, *, require_live_evidence: bool) -> dict[str, Any]:
    checks: list[GateCheck] = []
    checks.extend(check_data_volume(repo))
    checks.append(check_validation_cases(repo))
    checks.extend(check_architecture(repo))
    checks.extend(check_rag_and_guardrails(repo))
    checks.append(check_evaluation_artifacts(repo))
    checks.extend(check_observability(repo))
    checks.extend(check_docs_and_portfolio(repo))
    checks.extend(check_live_evidence(repo, required=require_live_evidence))

    return {
        "passed": all(check.passed for check in checks),
        "failed_checks": [check.name for check in checks if not check.passed],
        "checks": [check.__dict__ for check in checks],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 11 launch-readiness gate.")
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-live-evidence", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    report = run_gate(repo, require_live_evidence=args.require_live_evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps({"passed": report["passed"], "failed_checks": report["failed_checks"]}, indent=2)
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
