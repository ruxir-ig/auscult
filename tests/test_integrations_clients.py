"""Tests for the OpenAI / Anthropic client wrappers (duck-typed fakes)."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from auscult.context import NoActiveRunError, start_run
from auscult.db import get_session
from auscult.integrations.anthropic import wrap_anthropic
from auscult.integrations.openai import wrap_openai
from auscult.models import Step


def _steps(run_id: str) -> list[Step]:
    with get_session() as session:
        return list(
            session.execute(
                select(Step).where(Step.run_id == run_id).order_by(Step.step_index)
            )
            .scalars()
            .all()
        )


def _chat_response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))]
    )


class FakeOpenAI:
    def __init__(self, reply: str = "Take ibuprofen.", error: Exception | None = None):
        self.calls: list[dict] = []

        def create(**kwargs):
            self.calls.append(kwargs)
            if error is not None:
                raise error
            return _chat_response(reply)

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
        self.responses = SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(output_text=reply)
        )


class FakeAnthropic:
    def __init__(self, reply: str = "Take ibuprofen.", error: Exception | None = None):
        def create(**kwargs):
            if error is not None:
                raise error
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=reply)])

        self.messages = SimpleNamespace(create=create)


def test_wrap_openai_records_chat_step(db) -> None:
    client = wrap_openai(FakeOpenAI(reply="Patient is stable."))
    with start_run("openai-agent", "Triage request.") as tracer:
        response = client.chat.completions.create(
            model="gpt-test",
            messages=[{"role": "user", "content": "How is John Smith?"}],
        )
    assert response.choices[0].message.content == "Patient is stable."

    steps = _steps(tracer.run_id)
    assert len(steps) == 1
    command = json.loads(steps[0].llm_command)
    assert command["api"] == "chat.completions"
    assert command["model"] == "gpt-test"
    assert "John Smith" not in steps[0].llm_command  # sanitized
    assert steps[0].output == "Patient is stable."


def test_wrap_openai_records_responses_step(db) -> None:
    client = wrap_openai(FakeOpenAI(reply="All clear."))
    with start_run("openai-agent", "Triage request.") as tracer:
        client.responses.create(model="gpt-test", input="Summarize the chart.")

    steps = _steps(tracer.run_id)
    assert len(steps) == 1
    assert json.loads(steps[0].llm_command)["api"] == "responses"
    assert steps[0].output == "All clear."


def test_wrap_openai_records_error_step_and_reraises(db) -> None:
    client = wrap_openai(FakeOpenAI(error=RuntimeError("rate limited")))
    with pytest.raises(RuntimeError, match="rate limited"):
        with start_run("openai-agent", "Triage request.") as tracer:
            client.chat.completions.create(model="gpt-test", messages=[])

    steps = _steps(tracer.run_id)
    assert len(steps) == 1
    assert steps[0].error_message == "rate limited"
    assert steps[0].output is None


def test_wrap_openai_stream_records_null_output(db) -> None:
    client = wrap_openai(FakeOpenAI())
    with start_run("openai-agent", "Triage request.") as tracer:
        client.chat.completions.create(model="gpt-test", messages=[], stream=True)

    steps = _steps(tracer.run_id)
    assert len(steps) == 1
    assert steps[0].output is None


def test_wrap_openai_records_tool_calls(db) -> None:
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name="lookup_patient", arguments='{"name":"X"}')
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tool_call]))]
    )
    client = FakeOpenAI()
    client.chat.completions.create = lambda **kwargs: response
    wrap_openai(client)

    with start_run("openai-agent", "Triage request.") as tracer:
        client.chat.completions.create(model="gpt-test", messages=[])

    steps = _steps(tracer.run_id)
    assert steps[0].output is not None
    assert "lookup_patient" in steps[0].output


def test_wrap_openai_requires_active_run(db) -> None:
    client = wrap_openai(FakeOpenAI())
    with pytest.raises(NoActiveRunError):
        client.chat.completions.create(model="gpt-test", messages=[])


def test_wrap_openai_explicit_tracer(db) -> None:
    from auscult.capture import AuscultTracer

    tracer = AuscultTracer(agent_type="openai-agent", initial_prompt="Triage.")
    client = wrap_openai(FakeOpenAI(), tracer=tracer)
    client.chat.completions.create(model="gpt-test", messages=[])
    tracer.finish()
    assert len(_steps(tracer.run_id)) == 1


def test_wrap_openai_idempotent(db) -> None:
    client = FakeOpenAI()
    wrap_openai(client)
    wrap_openai(client)  # second wrap must not double-record
    with start_run("openai-agent", "Triage request.") as tracer:
        client.chat.completions.create(model="gpt-test", messages=[])
    assert len(_steps(tracer.run_id)) == 1


def test_wrap_openai_rejects_non_client(db) -> None:
    with pytest.raises(TypeError, match="OpenAI-style client"):
        wrap_openai(object())


def test_wrap_openai_async_client(db) -> None:
    class FakeAsyncOpenAI:
        def __init__(self):
            async def create(**kwargs):
                return _chat_response("Async reply.")

            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    client = wrap_openai(FakeAsyncOpenAI())

    async def agent() -> str:
        with start_run("openai-agent", "Triage request.") as tracer:
            response = await client.chat.completions.create(model="gpt-test", messages=[])
            assert response.choices[0].message.content == "Async reply."
            return tracer.run_id

    run_id = asyncio.run(agent())
    steps = _steps(run_id)
    assert len(steps) == 1
    assert steps[0].output == "Async reply."


def test_wrap_anthropic_records_step(db) -> None:
    client = wrap_anthropic(FakeAnthropic(reply="Order a CBC."))
    with start_run("anthropic-agent", "Triage request.") as tracer:
        client.messages.create(
            model="claude-test",
            system="Be careful.",
            messages=[{"role": "user", "content": "Assess Jane Doe."}],
        )

    steps = _steps(tracer.run_id)
    assert len(steps) == 1
    command = json.loads(steps[0].llm_command)
    assert command["api"] == "messages"
    assert command["model"] == "claude-test"
    assert "Jane Doe" not in steps[0].llm_command  # sanitized
    assert steps[0].output == "Order a CBC."


def test_wrap_anthropic_error_and_reraise(db) -> None:
    client = wrap_anthropic(FakeAnthropic(error=RuntimeError("overloaded")))
    with pytest.raises(RuntimeError, match="overloaded"):
        with start_run("anthropic-agent", "Triage request.") as tracer:
            client.messages.create(model="claude-test", messages=[])

    steps = _steps(tracer.run_id)
    assert steps[0].error_message == "overloaded"


def test_wrap_anthropic_rejects_non_client(db) -> None:
    with pytest.raises(TypeError, match="Anthropic-style client"):
        wrap_anthropic(object())


def test_wrap_anthropic_idempotent(db) -> None:
    client = FakeAnthropic()
    wrap_anthropic(client)
    wrap_anthropic(client)
    with start_run("anthropic-agent", "Triage request.") as tracer:
        client.messages.create(model="claude-test", messages=[])
    assert len(_steps(tracer.run_id)) == 1
