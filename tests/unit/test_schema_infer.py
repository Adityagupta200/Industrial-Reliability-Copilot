import json
from pathlib import Path

import joblib
import pytest


def test_artifacts_present(repo_root: Path):
    anom_dir = repo_root / "src" / "anomaly_service" / "artifacts" / "anomaly"
    assert anom_dir.exists(), f"Missing artifacts dir: {anom_dir}"

    assert (anom_dir / "feature_schema.json").exists()
    assert (anom_dir / "preprocess.joblib").exists()
    assert (anom_dir / "anomaly_model.pth").exists()


def test_schema_matches_preprocess_bundle(repo_root: Path):
    anom_dir = repo_root / "src" / "anomaly_service" / "artifacts" / "anomaly"
    schema_path = anom_dir / "feature_schema.json"
    bundle_path = anom_dir / "preprocess.joblib"

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    bundle = joblib.load(bundle_path)

    assert "schemas" in schema and isinstance(schema["schemas"], dict)
    assert (
        "scalers" in bundle
        and "pcas" in bundle
        and "ga_mask" in bundle
        and "domain_order" in bundle
    )

    domain_order = list(bundle["domain_order"])
    scalers = bundle["scalers"]
    pcas = bundle["pcas"]

    for domain in domain_order:
        assert domain in schema["schemas"], f"feature_schema.json missing domain: {domain}"
        cols = schema["schemas"][domain].get("feature_order")
        assert isinstance(cols, list) and len(cols) > 0

        scaler = scalers.get(domain)
        pca = pcas.get(domain)
        assert scaler is not None, f"preprocess bundle missing scaler for {domain}"
        assert pca is not None, f"preprocess bundle missing pca for {domain}"

        # sklearn objects typically store n_features_in_ after fit
        n_scaler = getattr(scaler, "n_features_in_", None)
        n_pca = getattr(pca, "n_features_in_", None)

        if n_scaler is not None:
            assert int(n_scaler) == len(
                cols
            ), f"{domain}: schema cols={len(cols)} scaler n_features_in_={n_scaler}"
        if n_pca is not None:
            assert int(n_pca) == len(
                cols
            ), f"{domain}: schema cols={len(cols)} pca n_features_in_={n_pca}"


@pytest.mark.parametrize("key", ["ga_mask"])
def test_ga_mask_shape_valid(repo_root: Path, key: str):
    anom_dir = repo_root / "src" / "anomaly_service" / "artifacts" / "anomaly"
    bundle = joblib.load(anom_dir / "preprocess.joblib")
    ga_mask = bundle[key]

    # Must be 1D mask
    import numpy as np

    ga_mask = np.asarray(ga_mask)
    assert ga_mask.ndim == 1
    assert ga_mask.size > 0
