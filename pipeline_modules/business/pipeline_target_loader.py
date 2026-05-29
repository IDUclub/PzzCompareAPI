from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

DEFAULT_PIPELINE_CALLABLE = "pipeline_modules.pipeline_impl:run_pipeline"


def load_pipeline_callable(target: str | None = None) -> Callable[..., Any]:
    """
    Load the low-level pipeline callable from a module path.

    Expected format:
        "some.module.path:callable_name"
    """
    target_value = (target or DEFAULT_PIPELINE_CALLABLE).strip()
    if ":" not in target_value:
        raise ValueError(
            "PIPELINE_CALLABLE must be in 'module.path:callable_name' format"
        )

    module_name, function_name = target_value.split(":", 1)
    module = import_module(module_name)

    pipeline_callable = getattr(module, function_name, None)
    if pipeline_callable is None:
        raise AttributeError(
            f"Callable '{function_name}' was not found in module '{module_name}'"
        )
    if not callable(pipeline_callable):
        raise TypeError(
            f"Resolved object '{function_name}' from module '{module_name}' is not callable"
        )

    return pipeline_callable