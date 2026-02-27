from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Keep your existing env behavior (ANOM_* variables)
    model_config = SettingsConfigDict(env_prefix="ANOM_", case_sensitive=False)

    service_name: str = "anomaly-service"
    api_prefix: str = "/v1"
    log_level: str = "INFO"

    # Absolute path to: <repo>/src/anomaly_service
    service_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1])

    # Paths (can still be overridden by ANOM_* env vars)
    # Defaults remain relative, then we resolve them against service_root in model_post_init.
    anomaly_ckpt_path: Path = Path("artifacts/anomaly/anomaly_model.pth")
    anomaly_preprocess_path: Path = Path("artifacts/anomaly/preprocess.joblib")
    anomaly_schema_path: Path = Path("artifacts/anomaly/feature_schema.json")

    rul_model_path: Path = Path("artifacts/rul/rul_model.joblib")
    rul_schema_path: Path = Path("artifacts/rul/feature_schema.json")

    # MLflow
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "anomaly-service"
    mlflow_log_inference: bool = False
    mlflow_inference_sample_rate: float = 0.01  # 1% sampling to reduce overhead

    # Runtime
    torch_device: str = "cpu"  # keep CPU for portability
    request_timeout_seconds: float = 5.0

    def model_post_init(self, __context) -> None:
        for name in (
            "anomaly_ckpt_path",
            "anomaly_preprocess_path",
            "anomaly_schema_path",
            "rul_model_path",
            "rul_schema_path",
        ):
            p: Path = getattr(self, name)
            if not p.is_absolute():
                setattr(self, name, (self.service_root / p))


settings = Settings()
