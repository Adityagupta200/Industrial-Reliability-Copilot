from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class AnomalyClient:
    base_url: str
    predict_anomaly_path: str
    predict_rul_path: str
    timeout_s: float = 2.0

    async def predict(self, sensor_data: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_s) as client:
            anomaly_req = client.post(self.predict_anomaly_path, json={"sensor_data": sensor_data})
            rul_req = client.post(self.predict_rul_path, json={"sensor_data": sensor_data})
            r1, r2 = await httpx.AsyncClient.gather(anomaly_req, rul_req)  # type: ignore[attr-defined]
            # Fallback for older httpx versions (no gather):
            # r1, r2 = await asyncio.gather(anomaly_req, rul_req)

            r1.raise_for_status()
            r2.raise_for_status()
            return {"anomaly": r1.json(), "rul": r2.json()}
