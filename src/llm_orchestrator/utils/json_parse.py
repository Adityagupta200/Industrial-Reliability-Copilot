from __future__ import annotations

import json
import re
from typing import TypeVar

import orjson
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class LLMOutputParseError(RuntimeError):
    pass


def _extract_json_object(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        raise LLMOutputParseError("No JSON object found in LLM output.")
    return m.group(0)


def parse_llm_json(text: str, model: type[T]) -> T:
    candidate = _extract_json_object(text)
    try:
        data = orjson.loads(candidate)
    except Exception:
        try:
            data = json.loads(candidate)
        except Exception as e:
            raise LLMOutputParseError(f"Invalid JSON from LLM: {e}") from e

    try:
        return model.model_validate(data)
    except ValidationError as e:
        raise LLMOutputParseError(f"JSON schema validation failed: {e}") from e
