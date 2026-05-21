from __future__ import annotations

import os
import threading
import time
import json
from dataclasses import dataclass
from typing import Optional

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0


class QueryEmbeddingCache:
    """
    Caches query -> embedding vector.
    - Memory mode: TTL dict, thread-safe
    - Redis mode: enabled if REDIS_URL is set and redis is installed
    """

    def __init__(self, *, ttl_seconds: int = 3600, namespace: str = "rag:qemb"):
        self.ttl_seconds = ttl_seconds
        self.namespace = namespace
        self._lock = threading.RLock()
        self._mem: dict[str, tuple[float, list[float]]] = {}
        self.stats = CacheStats()

        self._redis = None
        redis_url = os.getenv("REDIS_URL")
        if redis_url and redis is not None:
            self._redis = redis.Redis.from_url(redis_url, decode_responses=False)

    def _mk_key(self, query: str) -> str:
        return f"{self.namespace}:{query.strip().lower()}"

    def get(self, query: str) -> Optional[list[float]]:
        key = self._mk_key(query)

        if self._redis is not None:
            raw = self._redis.get(key)
            if raw is None:
                self.stats.misses += 1
                return None
            try:
                # PRODUCTION FIX: Replaced pickle with secure json deserialization
                self.stats.hits += 1
                return json.loads(raw)
            except Exception:
                self.stats.misses += 1
                return None

        now = time.time()
        with self._lock:
            item = self._mem.get(key)
            if not item:
                self.stats.misses += 1
                return None
            expires_at, vec = item
            if now >= expires_at:
                self._mem.pop(key, None)
                self.stats.misses += 1
                return None
            self.stats.hits += 1
            return vec

    def set(self, query: str, vec: list[float]) -> None:
        key = self._mk_key(query)

        if self._redis is not None:
            # PRODUCTION FIX: Replaced pickle with secure json serialization
            self._redis.setex(key, self.ttl_seconds, json.dumps(vec))
            return

        expires_at = time.time() + self.ttl_seconds
        with self._lock:
            self._mem[key] = (expires_at, vec)
            if len(self._mem) > 5000:
                # simple bounded cache
                for k in list(self._mem.keys())[:1000]:
                    self._mem.pop(k, None)
