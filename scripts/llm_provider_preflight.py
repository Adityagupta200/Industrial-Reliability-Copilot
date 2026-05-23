from __future__ import annotations

import argparse
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _is_placeholder(value: str) -> bool:
    lowered = value.lower()
    return not value or "<" in value or ">" in value or "replace" in lowered


def _post_openai_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout_seconds: float,
) -> dict:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Reply with the single word OK.",
            }
        ],
        "max_tokens": 4,
        "temperature": 0,
    }
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI preflight failed with HTTP {exc.code}: {error_body}") from exc
    except URLError as exc:
        raise RuntimeError(f"OpenAI preflight could not connect to {endpoint}: {exc}") from exc

    return json.loads(body)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail fast when the configured LLM is unusable.")
    parser.add_argument("--provider", default=os.getenv("LLM_PRIMARY_PROVIDER", "openai"))
    parser.add_argument("--model", default=os.getenv("LLM_OPENAI_MODEL", ""))
    parser.add_argument("--api-key-env", default="LLM_OPENAI_API_KEY")
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1",
    )
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()

    if args.provider != "openai":
        print(f"LLM preflight skipped for provider={args.provider!r}.", flush=True)
        return 0

    api_key = os.getenv(args.api_key_env) or os.getenv("OPENAI_API_KEY") or ""
    if _is_placeholder(api_key):
        print(
            f"::error::{args.api_key_env} or OPENAI_API_KEY must be configured for OpenAI LLM evaluation.",
            file=sys.stderr,
            flush=True,
        )
        return 1

    if _is_placeholder(args.model):
        print("::error::LLM_OPENAI_MODEL must be configured to a deployable model.", file=sys.stderr)
        return 1

    try:
        response = _post_openai_chat_completion(
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as exc:
        print(f"::error::{exc}", file=sys.stderr, flush=True)
        return 1

    choice_count = len(response.get("choices") or [])
    if choice_count < 1:
        print(
            f"::error::OpenAI preflight returned no choices for model {args.model!r}.",
            file=sys.stderr,
            flush=True,
        )
        return 1

    print(f"OpenAI LLM preflight passed for model {args.model}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
