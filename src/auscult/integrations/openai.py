"""Auto-instrumentation for OpenAI (and OpenAI-compatible) clients.

Usage — no changes to the agent loop, wrap the client once::

    from openai import OpenAI
    from auscult.context import start_run
    from auscult.integrations.openai import wrap_openai

    client = wrap_openai(OpenAI())

    with start_run("triage-agent", initial_prompt=prompt):
        # every chat.completions.create / responses.create call in here
        # is captured (and sanitized) as a step automatically
        run_existing_agent(client, prompt)

The wrapper is duck-typed: anything exposing ``chat.completions.create``
and/or ``responses.create`` works, including Azure OpenAI and local
OpenAI-compatible servers. ``AsyncOpenAI`` clients are supported.

Steps record the request (model + messages, JSON) as ``llm_command`` and the
response text as ``output``; SDK exceptions are recorded as failed steps and
re-raised. Streaming responses are recorded with ``output=None`` (the wrapper
never consumes the caller's stream).
"""

from __future__ import annotations

from functools import partial
from typing import Any

from ..capture import AuscultTracer
from ._patch import patch_create
from ._serialize import to_text


def wrap_openai[ClientT](client: ClientT, *, tracer: AuscultTracer | None = None) -> ClientT:
    """Patch an OpenAI-style client in place so completion calls become steps.

    When ``tracer`` is None (the default) each call records to the tracer
    active in the ambient run context at call time, so one wrapped client can
    be shared across many runs.
    """
    chat = getattr(client, "chat", None)
    resources = {
        "chat.completions": getattr(chat, "completions", None),
        "responses": getattr(client, "responses", None),
    }
    patchable = {
        kind: resource
        for kind, resource in resources.items()
        if resource is not None and hasattr(resource, "create")
    }
    if not patchable:
        raise TypeError(
            "wrap_openai: client has neither chat.completions.create nor "
            "responses.create; is this an OpenAI-style client?"
        )
    for kind, resource in patchable.items():
        patch_create(
            resource,
            tracer=tracer,
            format_command=partial(_format_command, kind),
            extract_output=partial(_extract_output, kind),
        )
    return client


def _format_command(kind: str, kwargs: dict[str, Any]) -> str:
    payload: dict[str, Any] = {"api": kind}
    for key in ("model", "messages", "input"):
        if key in kwargs:
            payload[key] = kwargs[key]
    return to_text(payload)


def _extract_output(kind: str, kwargs: dict[str, Any], response: Any) -> str | None:
    if kwargs.get("stream"):
        return None
    if kind == "responses":
        text = getattr(response, "output_text", None)
        return text if isinstance(text, str) else None
    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            return to_text([_tool_call_dict(tc) for tc in tool_calls])
    return None


def _tool_call_dict(tool_call: Any) -> Any:
    function = getattr(tool_call, "function", None)
    if function is not None:
        return {
            "tool_call": getattr(function, "name", None),
            "arguments": getattr(function, "arguments", None),
        }
    return to_text(tool_call)
