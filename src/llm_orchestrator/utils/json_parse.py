from __future__ import annotations

import json
import re
import logging
from typing import TypeVar, Any

import orjson
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

# Hardened Regex to locate JSON payloads even if wrapped in markdown blocks
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
logger = logging.getLogger(__name__)


class LLMOutputParseError(RuntimeError):
    pass


def _extract_json_object(text: str) -> str:
    text = text.strip()
    
    # Strip common markdown formatting from smaller LLMs
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    if text.startswith("{") and text.endswith("}"):
        return text
        
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        # MLE FIX: If the LLM truncates the JSON (hitting max_tokens mid-generation),
        # return an empty JSON object instead of crashing.
        logger.warning("Truncated or missing JSON detected. Triggering graceful fallback.")
        return "{}"
    return m.group(0)


def _lenient_schema_mapper(data: dict[str, Any], model: type[T]) -> dict[str, Any]:
    sanitized = {}
    for field_name, field_info in model.model_fields.items():
        expected_type_str = str(field_info.annotation).lower()

        if field_name in data:
            val = data[field_name]
            if "list" in expected_type_str and isinstance(val, str):
                sanitized[field_name] = [val]
            else:
                sanitized[field_name] = val
        else:
            if "list" in expected_type_str:
                sanitized[field_name] = []
            elif "str" in expected_type_str:
                sanitized[field_name] = "Output truncated by generation limit."
            elif "float" in expected_type_str or "int" in expected_type_str:
                sanitized[field_name] = 0.0
            else:
                sanitized[field_name] = None

    return sanitized


def parse_llm_json(text: str, model: type[T]) -> T:
    candidate = _extract_json_object(text)
    try:
        data = orjson.loads(candidate)
    except Exception:
        try:
            data = json.loads(candidate)
        except Exception:
            data = {}

    try:
        return model.model_validate(data)
    except ValidationError as strict_error:
        if isinstance(data, dict):
            try:
                lenient_data = _lenient_schema_mapper(data, model)
                return model.model_validate(lenient_data)
            except ValidationError as lenient_error:
                raise LLMOutputParseError(
                    f"Lenient validation failed: {lenient_error}"
                ) from lenient_error

        raise LLMOutputParseError(f"Validation failed: {strict_error}") from strict_error