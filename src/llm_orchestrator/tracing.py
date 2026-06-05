from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

try:
    from langsmith import traceable as _langsmith_traceable
except Exception:  # pragma: no cover - exercised when optional tracing is absent

    def traceable(*decorator_args: Any, **decorator_kwargs: Any) -> Callable[[F], F] | F:
        if decorator_args and callable(decorator_args[0]) and len(decorator_args) == 1:
            return decorator_args[0]

        def decorator(func: F) -> F:
            return func

        return decorator

else:
    traceable = _langsmith_traceable
