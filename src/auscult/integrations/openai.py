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

import inspect
from typing import Any

from ..capture import AuscultTracer
from ..context import require_tracer
from ._serialize import to_text

_WRAPPED_MARKER = "_auscult_wrapped"


def wrap_openai[ClientT](client: ClientT, *, tracer: AuscultTracer | None = None) -> ClientT:
    """Patch an OpenAI-style client in place so completion calls become steps.

    When ``tracer`` is None (the default) each call records to the tracer
    active in the ambient run context at call time, so one wrapped client can
    be shared across many runs.
    """
    wrapped_any = False

    chat = getattr(client, "chat", None)
    completions = getattr(chat, "completions", None)
    if completions is not None and hasattr(completions, "create"):
        _patch_create(completions, kind="chat.completions", tracer=tracer)
        wrapped_any = True

    responses = getattr(client, "responses", None)
    if responses is not None and hasattr(responses, "create"):
        _patch_create(responses, kind="responses", tracer=tracer)
        wrapped_any = True

    if not wrapped_any:
        raise TypeError(
            "wrap_openai: client has neither chat.completions.create nor "
            "responses.create; is this an OpenAI-style client?"
        )
    return client


def _patch_create(resource: Any, *, kind: str, tracer: AuscultTracer | None) -> None:
    original = resource.create
    if getattr(original, _WRAPPED_MARKER, False):
        return

    def resolve_tracer() -> AuscultTracer:
        return tracer if tracer is not None else require_tracer()

    if inspect.iscoroutinefunction(original):

        async def async_create(*args: Any, **kwargs: Any) -> Any:
            active = resolve_tracer()
            command = _format_command(kind, kwargs)
            try:
                response = await original(*args, **kwargs)
            except Exception as exc:
                active.record_step(command, output=None, error_message=str(exc))
                raise
            active.record_step(command, output=_extract_output(kind, kwargs, response))
            return response

        setattr(async_create, _WRAPPED_MARKER, True)
        resource.create = async_create
        return

    def create(*args: Any, **kwargs: Any) -> Any:
        active = resolve_tracer()
        command = _format_command(kind, kwargs)
        try:
            response = original(*args, **kwargs)
        except Exception as exc:
            active.record_step(command, output=None, error_message=str(exc))
            raise
        active.record_step(command, output=_extract_output(kind, kwargs, response))
        return response

    setattr(create, _WRAPPED_MARKER, True)
    resource.create = create


def _format_command(kind: str, kwargs: dict[str, Any]) -> str:
    payload: dict[str, Any] = {"api": kind}
    if "model" in kwargs:
        payload["model"] = kwargs["model"]
    if "messages" in kwargs:
        payload["messages"] = kwargs["messages"]
    if "input" in kwargs:
        payload["input"] = kwargs["input"]
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
