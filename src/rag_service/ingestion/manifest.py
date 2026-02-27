from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

from rag_service.core.config import settings


@dataclass
class Manifest:
    entries: dict[str, dict[str, Any]]

    @classmethod
    def load(cls, path: str | None = None) -> "Manifest":
        p = Path(path or settings.processed_manifest_path)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            return cls(entries={})
        return cls(entries=json.loads(p.read_text(encoding="utf-8")))

    def save(self, path: str | None = None) -> None:
        p = Path(path or settings.processed_manifest_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.entries, indent=2, sort_keys=True), encoding="utf-8")

    def is_unchanged(self, key: str, sha256: str) -> bool:
        e = self.entries.get(key)
        return bool(e and e.get("sha256") == sha256 and e.get("status") == "ok")

    def mark(self, key: str, sha256: str, *, status: str, detail: str | None = None) -> None:
        self.entries[key] = {
            "sha256": sha256,
            "status": status,
            "detail": detail,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
