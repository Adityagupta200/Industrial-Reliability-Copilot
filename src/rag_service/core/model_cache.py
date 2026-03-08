from __future__ import annotations

from threading import Lock
from typing import Optional, Tuple, Dict, Any

_lock = Lock()
_sentence_models: Dict[Tuple[str, str], Any] = {}
_cross_models: Dict[Tuple[str, Optional[str]], Any] = {}


def _norm(s: str) -> str:
    return (s or "").strip()


def get_sentence_transformer(model_name: str, *, device: str = "cpu"):
    """
    Load-once SentenceTransformer registry (per Python process). [web:41]
    """
    from sentence_transformers import SentenceTransformer

    key = (_norm(model_name), _norm(device).lower())
    with _lock:
        m = _sentence_models.get(key)
        if m is None:
            m = SentenceTransformer(key[0], device=key[1])
            _sentence_models[key] = m
        return m


def get_cross_encoder(model_name: str, *, device: Optional[str] = None):
    """
    Load-once CrossEncoder registry (per Python process). [web:56]
    """
    from sentence_transformers import CrossEncoder

    dev = _norm(device).lower() if device else None
    key = (_norm(model_name), dev)
    with _lock:
        m = _cross_models.get(key)
        if m is None:
            m = CrossEncoder(key[0], device=dev)
            _cross_models[key] = m
        return m
