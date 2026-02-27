from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import settings


def _sha256_of_file(path: str) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _strip_state_dict_prefix(
    state_dict: Dict[str, torch.Tensor], prefix: str
) -> Dict[str, torch.Tensor]:
    """Strip a prefix like 'module.' from DataParallel checkpoints."""
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix) :]: v for k, v in state_dict.items()}


def _extract_state_dict_and_meta(ckpt: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Supports:
      - checkpoint dict with 'model_state_dict'
      - raw state_dict dict[str, Tensor]
    """
    if (
        isinstance(ckpt, dict)
        and "model_state_dict" in ckpt
        and isinstance(ckpt["model_state_dict"], dict)
    ):
        return ckpt["model_state_dict"], ckpt

    if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        tensor_vals = [v for v in ckpt.values() if torch.is_tensor(v)]
        if tensor_vals and len(tensor_vals) == len(ckpt):
            return ckpt, {}

    raise ValueError(
        "Unsupported anomaly checkpoint format. Expected dict with 'model_state_dict' or a raw state_dict."
    )


class MultiScaleFeatureExtraction(nn.Module):
    """
    Exposes keys:
      feature_extractor.input_projection.*
      feature_extractor.conv1d_1.*
      feature_extractor.conv1d_2.*
      feature_extractor.pathway_1.*
      feature_extractor.pathway_2.*
      feature_extractor.fusion_layer.*
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.conv1d_1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv1d_2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)

        self.pathway_1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.pathway_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.fusion_layer = nn.Linear(hidden_dim + 64, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.input_projection(x)  # [B, hidden_dim]

        x_conv = x_proj.unsqueeze(1)  # [B, 1, hidden_dim]
        conv_out = F.relu(self.conv1d_1(x_conv))
        conv_out = F.relu(self.conv1d_2(conv_out))
        conv_out = F.adaptive_avg_pool1d(conv_out, 1).squeeze(-1)  # [B, 64]

        p1 = self.pathway_1(x_proj)  # [B, hidden_dim//2]
        p2 = self.pathway_2(x_proj)  # [B, hidden_dim//2]
        parallel = torch.cat([p1, p2], dim=-1)  # [B, hidden_dim]

        combined = torch.cat([parallel, conv_out], dim=-1)  # [B, hidden_dim + 64]
        fused = F.relu(self.fusion_layer(combined))  # [B, hidden_dim]
        return fused


class DomainAdaptiveNormalization(nn.Module):
    """
    Exposes keys:
      domain_norm.domain_gamma
      domain_norm.domain_beta
      domain_norm.layer_norm.*
      domain_norm.domain_classifier.*
    """

    def __init__(self, num_features: int, num_domains: int = 5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(num_features)

        self.domain_gamma = nn.Parameter(torch.ones(num_domains, num_features))
        self.domain_beta = nn.Parameter(torch.zeros(num_domains, num_features))

        self.domain_classifier = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_domains),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor, domainid: Optional[torch.Tensor] = None) -> torch.Tensor:
        normalized = self.layer_norm(x)

        if domainid is None:
            probs = self.domain_classifier(x)  # [B, num_domains]
            domainid = torch.argmax(probs, dim=-1)  # [B]
        else:
            if not torch.is_tensor(domainid):
                domainid = torch.tensor(domainid, device=x.device)
            domainid = domainid.to(x.device)
            if domainid.dim() == 0:
                domainid = domainid.unsqueeze(0).expand(x.size(0))
            elif domainid.dim() == 1 and domainid.numel() == 1 and x.size(0) > 1:
                domainid = domainid.expand(x.size(0))

        gamma = self.domain_gamma[domainid]
        beta = self.domain_beta[domainid]
        return gamma * normalized + beta


class TemporalSpatialFusion(nn.Module):
    """
    Exposes keys:
      temporal_spatial_fusion.temporal_conv.*
      temporal_spatial_fusion.spatial_attention.*
      temporal_spatial_fusion.temporal_proj.*
      temporal_spatial_fusion.spatial_proj.*
      temporal_spatial_fusion.output_proj.*
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=4, batch_first=True
        )

        self.temporal_proj = nn.Linear(64, hidden_dim // 2)
        self.spatial_proj = nn.Linear(input_dim, hidden_dim // 2)

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_temp = x.unsqueeze(1)  # [B, 1, input_dim]
        temp_feat = self.temporal_conv(x_temp).squeeze(-1)  # [B, 64]
        temp_feat = self.temporal_proj(temp_feat)  # [B, hidden_dim//2]

        x_spatial = x.unsqueeze(1)  # [B, 1, input_dim]
        spatial_out, _ = self.spatial_attention(
            x_spatial, x_spatial, x_spatial
        )  # [B, 1, input_dim]
        spatial_feat = self.spatial_proj(spatial_out.squeeze(1))  # [B, hidden_dim//2]

        combined = torch.cat([temp_feat, spatial_feat], dim=-1)  # [B, hidden_dim]
        return self.output_proj(combined)  # [B, hidden_dim]


class HybridMultiDomainArchitecture(nn.Module):
    """
    Matches checkpoint module naming AND allows the additional_processing widths to be configured
    so we can load artifacts trained with different head sizes.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        hidden_dim: int = 128,
        num_domains: int = 5,
        ap_hidden1: int = 64,
        ap_hidden2: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_domains = num_domains

        self.feature_extractor = MultiScaleFeatureExtraction(
            input_dim=input_dim, hidden_dim=hidden_dim
        )
        self.domain_norm = DomainAdaptiveNormalization(
            num_features=hidden_dim, num_domains=num_domains
        )
        self.temporal_spatial_fusion = TemporalSpatialFusion(
            input_dim=hidden_dim, hidden_dim=hidden_dim
        )

        # IMPORTANT: this must match checkpoint shapes (e.g., 128->64->32 in your error)
        self.additional_processing = nn.Sequential(
            nn.Linear(hidden_dim, ap_hidden1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(ap_hidden1, ap_hidden2),
            nn.ReLU(),
        )

        # Heads must take ap_hidden2 as input (e.g., 32)
        self.classifier = nn.Sequential(
            nn.Linear(ap_hidden2, num_classes),
        )
        self.domain_predictor = nn.Sequential(
            nn.Linear(ap_hidden2, num_domains),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor, domainid: Optional[torch.Tensor] = None):
        features = self.feature_extractor(x)
        normed = self.domain_norm(features, domainid=domainid)
        fused = self.temporal_spatial_fusion(normed)
        processed = self.additional_processing(fused)
        logits = self.classifier(processed)
        domainpred = self.domain_predictor(processed)
        return logits, domainpred, processed


@dataclass(frozen=True)
class LoadedModels:
    anomaly_model: nn.Module | None
    anomaly_version: str | None
    rul_model: Any | None
    rul_version: str | None


def load_rul_model():
    model = joblib.load(settings.rul_model_path)
    version = _sha256_of_file(settings.rul_model_path)
    with open(settings.rul_schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    return model, version, schema


def load_anomaly_model():
    ckpt = torch.load(settings.anomaly_ckpt_path, map_location=settings.torch_device)
    version = _sha256_of_file(settings.anomaly_ckpt_path)

    state_dict, meta = _extract_state_dict_and_meta(ckpt)
    state_dict = _strip_state_dict_prefix(state_dict, "module.")

    # Infer core dims
    input_dim = meta.get("input_dim", None)
    hidden_dim = meta.get("hidden_dim", None)
    num_classes = meta.get("num_classes", None)
    num_domains = meta.get("num_domains", None)

    # input_dim/hidden_dim from input projection
    w_in = state_dict.get("feature_extractor.input_projection.weight", None)
    if w_in is None:
        raise ValueError(
            "Missing key 'feature_extractor.input_projection.weight' in anomaly state_dict."
        )

    if hidden_dim is None:
        hidden_dim = int(w_in.shape[0])
    if input_dim is None:
        input_dim = int(w_in.shape[1])

    # num_domains from domain gamma
    if num_domains is None:
        dg = state_dict.get("domain_norm.domain_gamma", None)
        if dg is None:
            raise ValueError("Missing key 'domain_norm.domain_gamma' in anomaly state_dict.")
        num_domains = int(dg.shape[0])

    # additional_processing dims directly from checkpoint (fixes your current mismatch)
    ap0 = state_dict.get("additional_processing.0.weight", None)
    ap3 = state_dict.get("additional_processing.3.weight", None)
    if ap0 is None or ap3 is None:
        raise ValueError(
            "Missing additional_processing weights in anomaly state_dict (expected .0.weight and .3.weight)."
        )

    # ap0: [ap_hidden1, hidden_dim], ap3: [ap_hidden2, ap_hidden1]
    ap_hidden1 = int(ap0.shape[0])
    ap_hidden2 = int(ap3.shape[0])

    # num_classes from classifier weight if not provided
    if num_classes is None:
        cw = state_dict.get("classifier.0.weight", None)
        if cw is None:
            raise ValueError("Missing key 'classifier.0.weight' in anomaly state_dict.")
        num_classes = int(cw.shape[0])

    model = HybridMultiDomainArchitecture(
        input_dim=int(input_dim),
        num_classes=int(num_classes),
        hidden_dim=int(hidden_dim),
        num_domains=int(num_domains),
        ap_hidden1=ap_hidden1,
        ap_hidden2=ap_hidden2,
    ).to(settings.torch_device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, version
