"""Ambient run context so integrations can record steps without plumbing.

A :class:`contextvars.ContextVar` holds the tracer for the current execution
context. Wrapped clients, callback handlers, and decorated functions look the
tracer up here, so host applications do not have to thread an
:class:`~auscult.capture.AuscultTracer` through their call stack to adopt
Auscult.

The context propagates into ``asyncio`` tasks automatically. It does *not*
propagate into ``ThreadPoolExecutor`` / ``ProcessPoolExecutor`` workers
(a contextvars limitation); pass an explicit tracer to integrations there.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, cast, overload

from .capture import AuscultTracer

_ACTIVE_TRACER: ContextVar[AuscultTracer | None] = ContextVar(
    "auscult_active_tracer", default=None
)


class NoActiveRunError(RuntimeError):
    """Raised when a step is recorded outside any active Auscult run."""

    def __init__(self) -> None:
        super().__init__(
            "No active Auscult run in this context. Wrap the agent entry point "
            "with auscult.start_run(...) or @auscult.observe_run(...), or pass "
            "an explicit tracer to the integration."
        )


def current_tracer() -> AuscultTracer | None:
    """Return the tracer active in this context, or None.

    A tracer that has already been finished is treated as inactive.
    """
    tracer = _ACTIVE_TRACER.get()
    if tracer is not None and tracer.finished:
        return None
    return tracer


def require_tracer() -> AuscultTracer:
    """Return the active tracer or raise :class:`NoActiveRunError`."""
    tracer = current_tracer()
    if tracer is None:
        raise NoActiveRunError()
    return tracer


def record_step(
    llm_command: str,
    output: str | None,
    error_message: str | None = None,
) -> None:
    """Record a step on the active run (raises if no run is active)."""
    require_tracer().record_step(
        llm_command=llm_command, output=output, error_message=error_message
    )


@contextmanager
def use_tracer(tracer: AuscultTracer) -> Generator[AuscultTracer]:
    """Activate an existing tracer for the duration of the block.

    Does not finish the tracer on exit; the caller owns its lifecycle.
    """
    token = _ACTIVE_TRACER.set(tracer)
    try:
        yield tracer
    finally:
        _ACTIVE_TRACER.reset(token)


@contextmanager
def start_run(
    agent_type: str,
    initial_prompt: str,
    **tracer_kwargs: Any,
) -> Generator[AuscultTracer]:
    """Create a tracer, activate it, and finish it when the block exits.

    An exception escaping the block marks the run as crashed and re-raises.
    Extra keyword arguments are forwarded to :class:`AuscultTracer`
    (e.g. ``background=True``, ``sanitizer=...``, ``run_id=...``).
    """
    tracer = AuscultTracer(
        agent_type=agent_type, initial_prompt=initial_prompt, **tracer_kwargs
    )
    token = _ACTIVE_TRACER.set(tracer)
    try:
        yield tracer
    except BaseException:
        if not tracer.finished:
            tracer.finish(crashed=True)
        raise
    finally:
        _ACTIVE_TRACER.reset(token)
        if not tracer.finished:
            tracer.finish()


def _default_prompt(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """First string argument, else a repr of the call arguments."""
    for value in (*args, *kwargs.values()):
        if isinstance(value, str):
            return value
    parts = [repr(a) for a in args]
    parts += [f"{k}={v!r}" for k, v in kwargs.items()]
    return f"({', '.join(parts)})"


@overload
def observe_run[**P, R](func: Callable[P, R]) -> Callable[P, R]: ...


@overload
def observe_run[**P, R](
    func: None = None,
    *,
    agent_type: str | None = None,
    prompt_from: Callable[[tuple[Any, ...], dict[str, Any]], str] | None = None,
    **tracer_kwargs: Any,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def observe_run[**P, R](
    func: Callable[P, R] | None = None,
    *,
    agent_type: str | None = None,
    prompt_from: Callable[[tuple[Any, ...], dict[str, Any]], str] | None = None,
    **tracer_kwargs: Any,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Wrap an agent entry point so each call becomes a captured run.

    The decorated function runs inside :func:`start_run`: a tracer is created,
    activated in the ambient context (so wrapped clients / callback handlers
    record into it), and finished when the function returns or raises.

    ``agent_type`` defaults to the function's qualified name. The initial
    prompt is the first string argument unless ``prompt_from`` is given.
    Works on both sync and async functions.
    """

    def decorate(target: Callable[P, R]) -> Callable[P, R]:
        run_agent_type = agent_type or target.__qualname__
        derive_prompt = prompt_from or _default_prompt

        if inspect.iscoroutinefunction(target):
            async_target = cast(Callable[P, Awaitable[Any]], target)

            @wraps(target)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                prompt = derive_prompt(args, kwargs)
                with start_run(run_agent_type, prompt, **tracer_kwargs):
                    return await async_target(*args, **kwargs)

            return cast(Callable[P, R], async_wrapper)

        @wraps(target)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            prompt = derive_prompt(args, kwargs)
            with start_run(run_agent_type, prompt, **tracer_kwargs):
                return target(*args, **kwargs)

        return wrapper

    if func is not None:
        return decorate(func)
    return decorate
