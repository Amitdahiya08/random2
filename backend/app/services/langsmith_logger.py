import os
from typing import Any, Callable
from shared.config import settings

# Simple no-op decorator if tracing disabled
def traceable(name: str) -> Callable:
    if not settings.langsmith_tracing:
        def _wrap(func):
            return func
        return _wrap

    # Lazy import to avoid hard dependency if disabled
    from langsmith import traceable as _traceable  # type: ignore
    return _traceable(name=name, project_name=settings.langsmith_project)
