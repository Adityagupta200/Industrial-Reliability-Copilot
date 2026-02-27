import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

for p in (REPO_ROOT, SRC_DIR):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return _repo_root()


@pytest.fixture(scope="session")
def sample_anom_request_path(repo_root: Path) -> Path:
    return repo_root / "tmp" / "anom_req.json"


@pytest.fixture(scope="session")
def sample_anom_request(sample_anom_request_path: Path) -> Dict[str, Any]:
    if not sample_anom_request_path.exists():
        pytest.skip(f"Missing sample request JSON at {sample_anom_request_path}")
    return json.loads(sample_anom_request_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def sample_rul_request_path(repo_root: Path) -> Path:
    """
    Sample request payload for the RUL endpoint.
    """
    return repo_root / "tmp" / "rul_req.json"


@pytest.fixture(scope="session")
def sample_rul_request(sample_rul_request_path: Path) -> Dict[str, Any]:
    if not sample_rul_request_path.exists():
        pytest.skip(f"Missing sample request JSON at {sample_rul_request_path}")

    return json.loads(sample_rul_request_path.read_text(encoding="utf-8"))


def _import_app():
    candidates = [
        ("anomaly_service.main", "app"),
    ]

    last_err: Optional[Exception] = None
    for mod_name, attr in candidates:
        try:
            mod = __import__(mod_name, fromlist=[attr])
            return getattr(mod, attr)
        except Exception as e:
            last_err = e

    raise RuntimeError(
        "Could not import FastAPI app. Tried:\n"
        + "\n".join([f"- {m}:{a}" for m, a in candidates])
        + "\n\nFix by updating candidates in tests/conftest.py to match your app module."
    ) from last_err


@pytest.fixture(scope="session")
def app():
    return _import_app()


@pytest.fixture(scope="session")
def client(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def base_url() -> str:
    return os.getenv("ANOMALY_BASE_URL", "http://localhost:8001")


@pytest.fixture(scope="session")
def api_prefix() -> str:
    return os.getenv("ANOMALY_API_PREFIX", "/v1")
