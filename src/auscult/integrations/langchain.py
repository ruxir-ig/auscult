"""LangChain / LangGraph callback handler.

Usage — pass the handler at invocation, no agent-loop changes::

    from auscult.integrations.langchain import AuscultCallbackHandler

    handler = AuscultCallbackHandler(agent_type="triage-agent")
    result = graph.invoke(inputs, config={"callbacks": [handler]})
    print(handler.last_run_id)  # captured Auscult run

The handler follows LangChain's standard callback protocol, so it works with
LCEL chains, agents, and LangGraph graphs alike. Each LLM call and tool call
becomes one sanitized step. The run boundary is the root callback run
(``parent_run_id is None``): the handler opens an ``AuscultTracer`` there and
finishes it when the root ends (crashed on root error).

Tracer resolution order per root run:

1. an explicit ``tracer=`` passed to the handler (never finished by us),
2. the ambient run context from :func:`auscult.context.start_run` /
   :func:`auscult.context.observe_run` (never finished by us),
3. a tracer the handler creates and owns (finished at root end).

``raise_error`` is enabled: sanitizer or database failures propagate into the
host application instead of being swallowed (fail-closed, nothing unsanitized
is ever stored).

Requires ``langchain-core`` (install with ``pip install 'auscult[langchain]'``
or ``uv add auscult --extra langchain``).
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any
from uuid import UUID

from ..capture import AuscultTracer
from ..context import current_tracer
from ._serialize import to_text

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "auscult.integrations.langchain requires langchain-core; "
        "install it with `pip install 'auscult[langchain]'` "
        "or `uv add auscult --extra langchain`."
    ) from exc

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult


class AuscultCallbackHandler(BaseCallbackHandler):
    """Capture LangChain / LangGraph executions as sanitized Auscult runs."""

    # Fail-closed: let sanitizer/DB errors propagate instead of being logged.
    raise_error = True
    # Preserve step ordering and ambient contextvars in async execution.
    run_inline = True

    def __init__(
        self,
        agent_type: str = "langchain-agent",
        *,
        tracer: AuscultTracer | None = None,
        **tracer_kwargs: Any,
    ) -> None:
        self._agent_type = agent_type
        self._explicit_tracer = tracer
        self._tracer_kwargs = tracer_kwargs
        self._lock = threading.Lock()
        self._active: AuscultTracer | None = None
        self._owns_active = False
        self._root_run_id: UUID | None = None
        self._pending: dict[UUID, str] = {}
        #: Auscult run id of the most recently captured run.
        self.last_run_id: str | None = None

    # -- run boundary -----------------------------------------------------

    def _begin_root(self, run_id: UUID, initial_prompt: str) -> None:
        """Bind a tracer for this root callback run (see resolution order)."""
        if self._root_run_id is not None:
            return
        self._root_run_id = run_id
        tracer = self._explicit_tracer or current_tracer()
        if tracer is not None:
            self._active = tracer
            self._owns_active = False
        else:
            self._active = AuscultTracer(
                agent_type=self._agent_type,
                initial_prompt=initial_prompt,
                **self._tracer_kwargs,
            )
            self._owns_active = True
        self.last_run_id = self._active.run_id

    def _end_root(self, run_id: UUID, *, crashed: bool) -> None:
        if run_id != self._root_run_id:
            return
        tracer = self._active
        self._root_run_id = None
        self._active = None
        self._pending.clear()
        if tracer is not None and self._owns_active and not tracer.finished:
            tracer.finish(crashed=crashed)
        self._owns_active = False

    def _record(
        self, llm_command: str, output: str | None, error_message: str | None = None
    ) -> None:
        if self._active is None or self._active.finished:
            return
        self._active.record_step(
            llm_command=llm_command, output=output, error_message=error_message
        )

    # -- chains / graphs ---------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            if parent_run_id is None:
                self._begin_root(run_id, to_text(inputs))

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            self._end_root(run_id, crashed=False)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            self._end_root(run_id, crashed=True)

    # -- LLM calls ----------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            if parent_run_id is None:
                self._begin_root(run_id, "\n\n".join(prompts))
            self._pending[run_id] = to_text(
                {"llm": _component_name(serialized), "prompts": prompts}
            )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        payload = [
            [{"role": m.type, "content": _message_content(m)} for m in batch]
            for batch in messages
        ]
        with self._lock:
            if parent_run_id is None:
                self._begin_root(run_id, to_text(payload))
            self._pending[run_id] = to_text(
                {"llm": _component_name(serialized), "messages": payload}
            )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        text = "\n\n".join(
            generation.text
            for batch in response.generations
            for generation in batch
        )
        with self._lock:
            command = self._pending.pop(run_id, "llm_call")
            self._record(command, output=text or None)
            self._end_root(run_id, crashed=False)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            command = self._pending.pop(run_id, "llm_call")
            self._record(command, output=None, error_message=str(error))
            self._end_root(run_id, crashed=True)

    # -- tool calls ----------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            if parent_run_id is None:
                self._begin_root(run_id, input_str)
            self._pending[run_id] = to_text(
                {"tool": _component_name(serialized), "input": input_str}
            )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            command = self._pending.pop(run_id, "tool_call")
            self._record(command, output=to_text(output))
            self._end_root(run_id, crashed=False)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            command = self._pending.pop(run_id, "tool_call")
            self._record(command, output=None, error_message=str(error))
            self._end_root(run_id, crashed=True)


def _component_name(serialized: dict[str, Any] | None) -> str:
    if not serialized:
        return "unknown"
    name = serialized.get("name")
    if isinstance(name, str) and name:
        return name
    identifier = serialized.get("id")
    if isinstance(identifier, list) and identifier:
        return str(identifier[-1])
    return "unknown"


def _message_content(message: BaseMessage) -> Any:
    content = message.content
    return content if isinstance(content, str | list) else str(content)
