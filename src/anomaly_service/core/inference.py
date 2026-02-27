from __future__ import annotations

import math
import random
import numpy as np
import torch
import torch.nn.functional as F

from .config import settings


def anomaly_infer(model, x_np: np.ndarray, domain_id: int) -> tuple[float, float]:
    x = torch.from_numpy(x_np).to(settings.torch_device)

    # Ensure [B, D]
    if x.dim() == 1:
        x = x.unsqueeze(0)

    with torch.no_grad():
        domain_tensor = torch.tensor(domain_id, device=settings.torch_device)
        logits, _, _ = model(x, domainid=domain_tensor)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]

    anomaly_score = float(probs[1])
    confidence = float(min(1.0, abs(anomaly_score - 0.5) * 2.0))
    return anomaly_score, confidence


def rul_infer(model, x_np: np.ndarray) -> tuple[float, float]:
    if x_np.ndim == 1:
        x_np = x_np.reshape(1, -1)

    y = float(model.predict(x_np)[0])
    confidence = float(1.0 / (1.0 + math.exp(-0.01 * abs(y))))
    return y, confidence


def should_log_to_mlflow() -> bool:
    if not settings.mlflow_log_inference:
        return False
    return random.random() < settings.mlflow_inference_sample_rate
