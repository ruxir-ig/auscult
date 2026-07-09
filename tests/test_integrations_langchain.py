"""Tests for the LangChain / LangGraph callback handler.

Uses langchain-core fake models so real callback plumbing (chains, LLMs,
tools) drives the handler end-to-end without network access.
"""

import json

import pytest
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from sqlalchemy import select

from auscult.context import start_run
from auscult.db import get_session
from auscult.integrations.langchain import AuscultCallbackHandler
from auscult.models import Run, Step


def _load_run(run_id: str) -> tuple[Run, list[Step]]:
    with get_session() as session:
        run = session.get(Run, run_id)
        steps = list(
            session.execute(
                select(Step).where(Step.run_id == run_id).order_by(Step.step_index)
            )
            .scalars()
            .all()
        )
    return run, steps


def test_llm_invoke_creates_run_with_step(db) -> None:
    handler = AuscultCallbackHandler(agent_type="lc-agent")
    llm = FakeListLLM(responses=["Patient is stable."])

    result = llm.invoke("Assess the patient.", config={"callbacks": [handler]})
    assert result == "Patient is stable."
    assert handler.last_run_id is not None

    run, steps = _load_run(handler.last_run_id)
    assert run.agent_type == "lc-agent"
    assert run.status == "completed"
    assert run.total_steps == 1
    assert len(steps) == 1
    assert steps[0].output == "Patient is stable."
    assert "prompts" in json.loads(steps[0].llm_command)


def test_chain_invoke_records_llm_steps(db) -> None:
    handler = AuscultCallbackHandler(agent_type="lc-chain-agent")
    chain = PromptTemplate.from_template("Q: {question}") | FakeListLLM(
        responses=["Order labs."]
    )

    chain.invoke({"question": "Next step?"}, config={"callbacks": [handler]})

    run, steps = _load_run(handler.last_run_id)
    assert run.status == "completed"
    assert len(steps) == 1
    assert steps[0].output == "Order labs."
    assert "Next step?" in run.initial_prompt


def test_chat_model_records_messages(db) -> None:
    handler = AuscultCallbackHandler(agent_type="lc-chat-agent")
    model = GenericFakeChatModel(messages=iter([AIMessage(content="Vitals look fine.")]))

    model.invoke("Check vitals.", config={"callbacks": [handler]})

    run, steps = _load_run(handler.last_run_id)
    assert run.status == "completed"
    assert len(steps) == 1
    command = json.loads(steps[0].llm_command)
    assert "messages" in command
    assert steps[0].output == "Vitals look fine."


def test_tool_calls_are_recorded(db) -> None:
    @tool
    def lookup_vitals(patient: str) -> str:
        """Look up vitals for a patient."""
        return "BP 120/80"

    handler = AuscultCallbackHandler(agent_type="lc-tool-agent")
    lookup_vitals.invoke("bed-12", config={"callbacks": [handler]})

    run, steps = _load_run(handler.last_run_id)
    assert run.status == "completed"
    assert len(steps) == 1
    command = json.loads(steps[0].llm_command)
    assert command["tool"] == "lookup_vitals"
    assert steps[0].output == "BP 120/80"


def test_tool_error_marks_run_failed(db) -> None:
    @tool
    def broken_tool(x: str) -> str:
        """Always fails."""
        raise ValueError("tool exploded")

    handler = AuscultCallbackHandler(agent_type="lc-tool-agent")
    with pytest.raises(ValueError, match="tool exploded"):
        broken_tool.invoke("x", config={"callbacks": [handler]})

    run, steps = _load_run(handler.last_run_id)
    assert run.status == "failed"
    assert steps[0].error_message is not None
    assert "tool exploded" in steps[0].error_message


def test_phi_is_sanitized_in_captured_steps(db) -> None:
    handler = AuscultCallbackHandler(agent_type="lc-agent")
    llm = FakeListLLM(responses=["Contact at 212-555-0182 confirmed for John Smith."])

    llm.invoke("Call John Smith about his labs.", config={"callbacks": [handler]})

    run, steps = _load_run(handler.last_run_id)
    stored = run.initial_prompt + steps[0].llm_command + (steps[0].output or "")
    assert "John Smith" not in stored
    assert "212-555-0182" not in stored


def test_handler_reuse_creates_separate_runs(db) -> None:
    handler = AuscultCallbackHandler(agent_type="lc-agent")
    llm = FakeListLLM(responses=["First.", "Second."])

    llm.invoke("one", config={"callbacks": [handler]})
    first = handler.last_run_id
    llm.invoke("two", config={"callbacks": [handler]})
    second = handler.last_run_id

    assert first != second
    for run_id in (first, second):
        run, _ = _load_run(run_id)
        assert run.status == "completed"


def test_handler_uses_ambient_tracer_without_finishing_it(db) -> None:
    handler = AuscultCallbackHandler(agent_type="ignored")
    llm = FakeListLLM(responses=["From ambient run."])

    with start_run("host-agent", "Host prompt.") as tracer:
        llm.invoke("hello", config={"callbacks": [handler]})
        assert handler.last_run_id == tracer.run_id
        assert not tracer.finished  # handler must not finish a borrowed tracer

    run, steps = _load_run(tracer.run_id)
    assert run.agent_type == "host-agent"
    assert run.status == "completed"
    assert len(steps) == 1


def test_handler_uses_explicit_tracer(db) -> None:
    from auscult.capture import AuscultTracer

    tracer = AuscultTracer(agent_type="explicit-agent", initial_prompt="Prompt.")
    handler = AuscultCallbackHandler(tracer=tracer)
    llm = FakeListLLM(responses=["ok"])

    llm.invoke("hello", config={"callbacks": [handler]})
    assert not tracer.finished
    tracer.finish()

    _, steps = _load_run(tracer.run_id)
    assert len(steps) == 1


def test_handler_fail_closed_on_sanitizer_error(db) -> None:
    """Sanitizer failures must propagate to the caller, not be swallowed."""

    class BoomSanitizer:
        def sanitize_with_stats(self, text):
            from auscult.sanitizer import SanitizeResult

            if text and "boom" in text:
                raise RuntimeError("sanitizer exploded")
            return SanitizeResult(text=text, redaction_count=0, entity_counts={})

    handler = AuscultCallbackHandler(agent_type="lc-agent", sanitizer=BoomSanitizer())
    llm = FakeListLLM(responses=["boom output"])

    assert handler.raise_error  # fail-closed contract
    with pytest.raises(RuntimeError, match="sanitizer exploded"):
        llm.invoke("safe prompt", config={"callbacks": [handler]})
