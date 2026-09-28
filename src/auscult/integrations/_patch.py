"""Patch an SDK ``create`` method so each call is recorded as a step."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from ..capture import AuscultTracer
from ..context import require_tracer

_WRAPPED_MARKER = "_auscult_wrapped"


def patch_create(
    resource: Any,
    *,
    tracer: AuscultTracer | None,
    format_command: Callable[[dict[str, Any]], str],
    extract_output: Callable[[dict[str, Any], Any], str | None],
) -> None:
    """Replace ``resource.create`` with a recording wrapper (idempotent).

    SDK exceptions are recorded as failed steps and re-raised. When ``tracer``
    is None, the ambient tracer is resolved at call time.
    """
    original = resource.create
    if getattr(original, _WRAPPED_MARKER, False):
        return

    def resolve_tracer() -> AuscultTracer:
        return tracer if tracer is not None else require_tracer()

    if inspect.iscoroutinefunction(original):

        async def async_create(*args: Any, **kwargs: Any) -> Any:
            active = resolve_tracer()
            command = format_command(kwargs)
            try:
                response = await original(*args, **kwargs)
            except Exception as exc:
                active.record_step(command, output=None, error_message=str(exc))
                raise
            active.record_step(command, output=extract_output(kwargs, response))
            return response

        setattr(async_create, _WRAPPED_MARKER, True)
        resource.create = async_create
        return

    def create(*args: Any, **kwargs: Any) -> Any:
        active = resolve_tracer()
        command = format_command(kwargs)
        try:
            response = original(*args, **kwargs)
        except Exception as exc:
            active.record_step(command, output=None, error_message=str(exc))
            raise
        active.record_step(command, output=extract_output(kwargs, response))
        return response

    setattr(create, _WRAPPED_MARKER, True)
    resource.create = create
