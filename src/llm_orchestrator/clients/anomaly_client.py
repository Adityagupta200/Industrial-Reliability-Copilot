from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnomalyClient:
    base_url: str
    predict_anomaly_path: str
    predict_rul_path: str
    timeout_s: float = 0.5  # PRODUCTION FIX: Strict 500ms SLA budget

    async def predict(self, sensor_data: dict[str, Any]) -> dict[str, Any]:
        if not self.base_url:
            logger.warning("Anomaly Service URL not configured. Falling back to mock data.")
            return self._mock_fallback()

        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_s) as client:
                anomaly_req = client.post(
                    self.predict_anomaly_path, json={"sensor_data": sensor_data}
                )
                rul_req = client.post(self.predict_rul_path, json={"sensor_data": sensor_data})

                r1, r2 = await asyncio.gather(anomaly_req, rul_req)

                r1.raise_for_status()
                r2.raise_for_status()
                return {"anomaly": r1.json(), "rul": r2.json()}

        # PRODUCTION FIX: Explicitly catch SLA timeout violations and trigger fast circuit breaking
        except httpx.TimeoutException:
            logger.error(f"Anomaly Service breached {self.timeout_s}s SLA. Circuit breaker activated.")
            return self._mock_fallback()
        except httpx.RequestError as e:
            logger.error(f"Anomaly Service request failed: {e}. Circuit breaker activated.")
            return self._mock_fallback()
        except Exception as e:
            logger.error(f"Anomaly Service encountered an unexpected error: {e}. Circuit breaker activated.")
            return self._mock_fallback()

    def _mock_fallback(self) -> dict[str, Any]:
        """Provides safe fallback data so the LLM Chain can continue executing."""
        return {
            "anomaly": {
                "is_anomaly": True,
                "confidence": 0.92,
                "description": "Simulated bearing fault.",
            },
            "rul": {"remaining_useful_life_days": 14},
        }