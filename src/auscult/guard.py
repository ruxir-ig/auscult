"""Keep PHI out of agent context by sanitizing content before the agent reads it.

Tracing sanitizes what Auscult *stores*. This module sanitizes what the agent
*sees*: file contents and tool results pass through the same detector and
synthetic replacement before they reach the model.

- :func:`read_sanitized` reads a file and returns sanitized text.
- :func:`guard_tool` wraps a tool function so its result is sanitized.
- :func:`check_hook_payload` backs ``auscult guard-hook``, a coding-agent
  pre-tool hook that blocks reads of files that contain PHI.

Detection can miss PHI. These helpers reduce exposure; they do not make
unreviewed data safe to give to a model.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, cast, overload

from .integrations._serialize import to_text
from .sanitizer import Sanitizer, SanitizeResult

# Keys coding agents use for the target file in tool-call payloads.
_PATH_KEYS = ("file_path", "path", "notebook_path")


def read_sanitized(
    path: str | Path, *, sanitizer: Sanitizer | None = None, encoding: str = "utf-8"
) -> str:
    """Return the file's text with detected PHI replaced by synthetic values."""
    return scan_file(path, sanitizer=sanitizer, encoding=encoding).text or ""


def scan_file(
    path: str | Path, *, sanitizer: Sanitizer | None = None, encoding: str = "utf-8"
) -> SanitizeResult:
    """Sanitize a file's text and report what was replaced."""
    text = Path(path).read_text(encoding=encoding)
    return (sanitizer or Sanitizer()).sanitize_with_stats(text)


@overload
def guard_tool[**P, R](func: Callable[P, R]) -> Callable[P, R]: ...


@overload
def guard_tool[**P, R](
    func: None = None, *, sanitizer: Sanitizer | None = None
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def guard_tool[**P, R](
    func: Callable[P, R] | None = None, *, sanitizer: Sanitizer | None = None
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Sanitize a tool's result before it is returned to the agent.

    String results are sanitized as text. ``dict`` / ``list`` results are
    sanitized leaf by leaf and returned with the same structure. Any other
    result type raises ``TypeError`` (fail-closed). One sanitizer is shared
    across calls, so the same real value maps to the same synthetic value.
    Works on sync and async functions.
    """

    def decorate(target: Callable[P, R]) -> Callable[P, R]:
        shared = sanitizer or Sanitizer()

        if inspect.iscoroutinefunction(target):
            async_target = cast(Callable[P, Awaitable[Any]], target)

            @wraps(target)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                return _sanitize_result(shared, await async_target(*args, **kwargs))

            return cast(Callable[P, R], async_wrapper)

        @wraps(target)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return cast(R, _sanitize_result(shared, target(*args, **kwargs)))

        return wrapper

    if func is not None:
        return decorate(func)
    return decorate


def _sanitize_result(sanitizer: Sanitizer, result: Any) -> Any:
    if result is None or isinstance(result, str):
        return sanitizer.sanitize(result)
    if isinstance(result, dict | list):
        return json.loads(sanitizer.sanitize(to_text(result)) or "null")
    raise TypeError(
        f"guard_tool cannot sanitize a {type(result).__name__} result; return str, dict, or list"
    )


@dataclass(frozen=True)
class HookDecision:
    """Outcome of checking one pre-tool hook payload."""

    allow: bool
    path: str | None = None
    reason: str | None = None
    entity_counts: dict[str, int] = field(default_factory=dict)


def check_hook_payload(
    payload: Mapping[str, Any], *, sanitizer: Sanitizer | None = None
) -> HookDecision:
    """Decide whether a tool call may read the file it targets.

    ``payload`` is the JSON a coding agent sends to a pre-tool hook (Claude
    Code's ``PreToolUse`` format: ``{"tool_name": ..., "tool_input": {...}}``).
    Calls without a file path, or for a path that does not exist, are allowed.
    A file that contains detected PHI, or that cannot be scanned, is blocked.
    """
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, Mapping):
        return HookDecision(allow=True)
    path = next(
        (value for key in _PATH_KEYS if isinstance(value := tool_input.get(key), str)),
        None,
    )
    if path is None or not Path(path).is_file():
        return HookDecision(allow=True, path=path)
    try:
        result = scan_file(path, sanitizer=sanitizer)
    except Exception as exc:  # noqa: BLE001 — fail closed on any scan error
        return HookDecision(allow=False, path=path, reason=f"could not scan file for PHI: {exc}")
    if result.redaction_count == 0:
        return HookDecision(allow=True, path=path)
    entities = ", ".join(f"{k}={v}" for k, v in sorted(result.entity_counts.items()))
    return HookDecision(
        allow=False,
        path=path,
        reason=(
            f"{path} appears to contain PHI ({entities}). "
            f"Read a sanitized copy instead: auscult sanitize {path}"
        ),
        entity_counts=dict(result.entity_counts),
    )
