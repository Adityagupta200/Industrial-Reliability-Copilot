from __future__ import annotations

import importlib
from typing import Set

from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from anomaly_service.core.metrics import REQUESTS


def _load_app() -> FastAPI:
    candidates = (
        "anomaly_service.main",
        "main",
    )
    last_err: Exception | None = None

    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            last_err = e
            continue

        app = getattr(mod, "app", None)
        if isinstance(app, FastAPI):
            return app

    raise RuntimeError(
        f"Could not import a FastAPI `app` from {candidates}. Last error: {last_err!r}"
    )


def _counter_value(metrics_text: str, metric_name: str, required_label_values: Set[str]) -> float:
    candidate_sample_names = {metric_name, f"{metric_name}_total"}

    for family in text_string_to_metric_families(metrics_text):
        for sample in family.samples:
            if sample.name not in candidate_sample_names:
                continue
            if sample.name.endswith("_created"):
                continue

            if required_label_values.issubset(set(sample.labels.values())):
                return float(sample.value)

    raise AssertionError(
        f"Could not find counter sample for metric '{metric_name}' with label values "
        f"including {sorted(required_label_values)}"
    )


def _has_histogram_bucket(metrics_text: str, required_label_values: Set[str]) -> bool:
    for family in text_string_to_metric_families(metrics_text):
        for sample in family.samples:
            if sample.name.endswith("_bucket") and required_label_values.issubset(
                set(sample.labels.values())
            ):
                return True
    return False


def test_metrics_names_exist_and_request_counts_increment() -> None:
    app = _load_app()

    with TestClient(app) as client:
        r1 = client.get("/health")
        assert r1.status_code == 200

        m1 = client.get("/metrics")
        assert m1.status_code == 200
        metrics_1 = m1.text

        requests_metric_name = getattr(REQUESTS, "_name", None)
        assert (
            isinstance(requests_metric_name, str) and requests_metric_name
        ), "REQUESTS._name not found"

        health_labels = {"/health", "GET", "200"}
        c1 = _counter_value(metrics_1, requests_metric_name, health_labels)

        assert _has_histogram_bucket(
            metrics_1, {"/health", "GET"}
        ), "Expected a histogram bucket metric for /health GET (Timer latency histogram missing?)"

        r2 = client.get("/health")
        assert r2.status_code == 200

        m2 = client.get("/metrics")
        assert m2.status_code == 200
        metrics_2 = m2.text

        c2 = _counter_value(metrics_2, requests_metric_name, health_labels)
        assert c2 == c1 + 1, f"Expected /health counter to increment by 1 (before={c1}, after={c2})"
