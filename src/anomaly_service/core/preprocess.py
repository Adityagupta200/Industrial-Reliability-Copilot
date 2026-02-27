from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np


@dataclass(frozen=True)
class AnomalyArtifacts:
    # Stored in preprocess.joblib
    scalers: Dict[str, Any]  # sklearn StandardScaler per domain
    pcas: Dict[str, Any]  # sklearn PCA per domain
    ga_mask: np.ndarray  # boolean mask or index list after PCA
    domain_order: List[str]  # e.g. ["edgeiiot","hai","modbus","swat","uah"]


def load_anomaly_artifacts(path: str) -> AnomalyArtifacts:
    obj = joblib.load(path)
    return AnomalyArtifacts(
        scalers=obj["scalers"],
        pcas=obj["pcas"],
        ga_mask=np.array(obj["ga_mask"]),
        domain_order=list(obj["domain_order"]),
    )


def load_schema(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_schema_id(sensor_values: Dict[str, Any]) -> str:
    keys = set(sensor_values.keys())

    # HAI has tags like P1_B2004 etc; the technical doc also describes a timestamp column + Attack labeling [file:8]
    if any(
        k.startswith("P1_") or k.startswith("P2_") or k.startswith("P3_") or k.startswith("P4_")
        for k in keys
    ):
        return "hai"

    # SWaT commonly includes tags like FIT101/LIT101 and many plant instruments [file:9]
    if "FIT101" in keys or "LIT101" in keys or "MV101" in keys:
        return "swat"

    # Modbus features in your schema are FC1..FC4
    if "FC1_Read_Input_Register" in keys or "FC3_Read_Holding_Register" in keys:
        return "modbus"

    # UAH has keys like 'R1-PA1:VH' etc plus marker
    if any(
        k.startswith("R1-") or k.startswith("R2-") or k.startswith("R3-") or k.startswith("R4-")
        for k in keys
    ):
        return "uah"

    # Edge-IIoT has packet-like fields (tcp.*, mqtt.*, mbtcp.*)
    if any(k.startswith("tcp.") or k.startswith("mqtt.") or k.startswith("mbtcp.") for k in keys):
        return "edgeiiot"

    raise ValueError("Unable to infer schema_id from keys; please pass schema_id explicitly.")


def to_float32_vector(sensor_values: Dict[str, Any], feature_order: List[str]) -> np.ndarray:
    missing = [c for c in feature_order if c not in sensor_values]
    if missing:
        raise ValueError(
            f"Missing required features: {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
    x = np.array([sensor_values[c] for c in feature_order], dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite values found in inputs (NaN/inf).")
    return x


def preprocess_anomaly(
    schema_id: str,
    sensor_values: Dict[str, Any],
    schema: Dict[str, Any],
    artifacts: AnomalyArtifacts,
) -> Tuple[np.ndarray, int]:
    """
    Replicates the notebook’s inference path:
    raw vector -> scaler(domain) -> PCA(domain) -> GA mask -> final x for torch [file:7]
    """
    dom = schema_id
    feature_order = schema["schemas"][dom]["feature_order"]
    x_raw = to_float32_vector(sensor_values, feature_order).reshape(1, -1)

    scaler = artifacts.scalers[dom]
    pca = artifacts.pcas[dom]

    x_scaled = scaler.transform(x_raw)
    x_pca = pca.transform(x_scaled)

    mask = artifacts.ga_mask
    if mask.dtype == bool:
        x_final = x_pca[:, mask]
    else:
        x_final = x_pca[:, mask.astype(int)]

    domain_index = artifacts.domain_order.index(dom)
    return x_final.astype(np.float32), domain_index


def preprocess_rul(sensor_values: Dict[str, Any], schema: Dict[str, Any]) -> np.ndarray:
    feature_order = schema["feature_cols"]
    x = to_float32_vector(sensor_values, feature_order).reshape(1, -1)
    return x
