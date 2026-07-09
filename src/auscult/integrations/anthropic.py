"""Auto-instrumentation for Anthropic clients.

Usage — wrap the client once, keep the agent loop unchanged::

    from anthropic import Anthropic
    from auscult.context import start_run
    from auscult.integrations.anthropic import wrap_anthropic

    client = wrap_anthropic(Anthropic())

    with start_run("triage-agent", initial_prompt=prompt):
        run_existing_agent(client, prompt)

Duck-typed on ``messages.create``; async clients are supported. Steps record
the request (model + messages, JSON) as ``llm_command`` and the concatenated
text blocks of the response as ``output``. SDK exceptions are recorded as
failed steps and re-raised. Streaming calls record ``output=None``.
"""

from __future__ import annotations

import inspect
from typing import Any

from ..capture import AuscultTracer
from ..context import require_tracer
from ._serialize import to_text

_WRAPPED_MARKER = "_auscult_wrapped"


def wrap_anthropic[ClientT](client: ClientT, *, tracer: AuscultTracer | None = None) -> ClientT:
    """Patch an Anthropic-style client in place so message calls become steps."""
    messages = getattr(client, "messages", None)
    if messages is None or not hasattr(messages, "create"):
        raise TypeError(
            "wrap_anthropic: client has no messages.create; "
            "is this an Anthropic-style client?"
        )

    original = messages.create
    if getattr(original, _WRAPPED_MARKER, False):
        return client

    def resolve_tracer() -> AuscultTracer:
        return tracer if tracer is not None else require_tracer()

    if inspect.iscoroutinefunction(original):

        async def async_create(*args: Any, **kwargs: Any) -> Any:
            active = resolve_tracer()
            command = _format_command(kwargs)
            try:
                response = await original(*args, **kwargs)
            except Exception as exc:
                active.record_step(command, output=None, error_message=str(exc))
                raise
            active.record_step(command, output=_extract_output(kwargs, response))
            return response

        setattr(async_create, _WRAPPED_MARKER, True)
        messages.create = async_create
        return client

    def create(*args: Any, **kwargs: Any) -> Any:
        active = resolve_tracer()
        command = _format_command(kwargs)
        try:
            response = original(*args, **kwargs)
        except Exception as exc:
            active.record_step(command, output=None, error_message=str(exc))
            raise
        active.record_step(command, output=_extract_output(kwargs, response))
        return response

    setattr(create, _WRAPPED_MARKER, True)
    messages.create = create
    return client


def _format_command(kwargs: dict[str, Any]) -> str:
    payload: dict[str, Any] = {"api": "messages"}
    if "model" in kwargs:
        payload["model"] = kwargs["model"]
    if "system" in kwargs:
        payload["system"] = kwargs["system"]
    if "messages" in kwargs:
        payload["messages"] = kwargs["messages"]
    return to_text(payload)


def _extract_output(kwargs: dict[str, Any], response: Any) -> str | None:
    if kwargs.get("stream"):
        return None
    content = getattr(response, "content", None)
    if not content:
        return None
    texts = [
        block.text
        for block in content
        if getattr(block, "type", None) == "text" and isinstance(getattr(block, "text", None), str)
    ]
    if texts:
        return "".join(texts)
    return to_text(content)
