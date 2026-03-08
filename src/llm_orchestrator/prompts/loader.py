from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromptBundle:
    name: str
    version: str
    template: str
    metadata: dict[str, Any]


class PromptNotFoundError(FileNotFoundError):
    pass


class PromptLoader:
    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or (Path(__file__).resolve().parent)

    def load(self, prompt_name: str, version: str) -> PromptBundle:
        prompt_dir = self._base_dir / prompt_name
        template_path = prompt_dir / f"v{version}.txt"
        metadata_path = prompt_dir / "metadata.json"

        if not template_path.exists():
            raise PromptNotFoundError(f"Prompt template not found: {template_path}")
        if not metadata_path.exists():
            raise PromptNotFoundError(f"Prompt metadata not found: {metadata_path}")

        template = template_path.read_text(encoding="utf-8")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return PromptBundle(name=prompt_name, version=version, template=template, metadata=metadata)
