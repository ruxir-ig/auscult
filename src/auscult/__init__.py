"""Auscult: passive observability and safe replay for healthcare AI agents.

Public API (lazily imported so ``import auscult`` stays cheap):

- :class:`auscult.AuscultTracer` — explicit capture of a run and its steps.
- :func:`auscult.start_run` / :func:`auscult.observe_run` — ambient run
  context so integrations record steps without plumbing a tracer through.
- :func:`auscult.record_step` / :func:`auscult.current_tracer` /
  :func:`auscult.use_tracer` — work with the ambient run context.
- :class:`auscult.Sanitizer` — PHI detection and synthetic replacement.

Framework adapters live in :mod:`auscult.integrations`.
"""

from typing import Any

__all__ = [
    "AuscultTracer",
    "Sanitizer",
    "current_tracer",
    "observe_run",
    "record_step",
    "start_run",
    "use_tracer",
]

_EXPORTS = {
    "AuscultTracer": ("auscult.capture", "AuscultTracer"),
    "Sanitizer": ("auscult.sanitizer", "Sanitizer"),
    "current_tracer": ("auscult.context", "current_tracer"),
    "observe_run": ("auscult.context", "observe_run"),
    "record_step": ("auscult.context", "record_step"),
    "start_run": ("auscult.context", "start_run"),
    "use_tracer": ("auscult.context", "use_tracer"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    from importlib import import_module

    return getattr(import_module(module_name), attr)
