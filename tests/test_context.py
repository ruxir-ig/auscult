import asyncio

import pytest
from sqlalchemy import select

from auscult.capture import AuscultTracer
from auscult.context import (
    NoActiveRunError,
    current_tracer,
    observe_run,
    record_step,
    require_tracer,
    start_run,
    use_tracer,
)
from auscult.db import get_session
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


def test_no_active_tracer_by_default(db) -> None:
    assert current_tracer() is None
    with pytest.raises(NoActiveRunError):
        require_tracer()
    with pytest.raises(NoActiveRunError):
        record_step("cmd", output="out")


def test_start_run_activates_and_finishes(db) -> None:
    with start_run("ctx-agent", "Check vitals.") as tracer:
        assert current_tracer() is tracer
        record_step("get_vitals()", output="BP 120/80.")
    assert current_tracer() is None

    run, steps = _load_run(tracer.run_id)
    assert run.status == "completed"
    assert run.total_steps == 1
    assert len(steps) == 1


def test_start_run_marks_crash_on_exception(db) -> None:
    with pytest.raises(ValueError, match="boom"):
        with start_run("ctx-agent", "Check vitals.") as tracer:
            record_step("get_vitals()", output="BP 120/80.")
            raise ValueError("boom")

    run, _ = _load_run(tracer.run_id)
    assert run.status == "failed"
    assert current_tracer() is None


def test_start_run_sanitizes_phi(db) -> None:
    with start_run("ctx-agent", "Triage John Smith.") as tracer:
        record_step("lookup('John Smith')", output="Phone: 212-555-0182.")

    run, steps = _load_run(tracer.run_id)
    stored = run.initial_prompt + steps[0].llm_command + (steps[0].output or "")
    assert "John Smith" not in stored
    assert "212-555-0182" not in stored


def test_use_tracer_does_not_finish(db) -> None:
    tracer = AuscultTracer(agent_type="ctx-agent", initial_prompt="Check vitals.")
    with use_tracer(tracer):
        assert current_tracer() is tracer
        record_step("get_vitals()", output="ok")
    assert current_tracer() is None
    assert not tracer.finished
    tracer.finish()


def test_finished_tracer_is_not_current(db) -> None:
    tracer = AuscultTracer(agent_type="ctx-agent", initial_prompt="Check vitals.")
    with use_tracer(tracer):
        tracer.finish()
        assert current_tracer() is None
        with pytest.raises(NoActiveRunError):
            record_step("cmd", output=None)


def test_observe_run_decorator(db) -> None:
    captured: dict[str, str] = {}

    @observe_run
    def triage(prompt: str) -> str:
        captured["run_id"] = require_tracer().run_id
        record_step("assess()", output="stable")
        return "done"

    assert triage("Assess patient vitals.") == "done"
    run, steps = _load_run(captured["run_id"])
    assert run.agent_type == "test_observe_run_decorator.<locals>.triage"
    assert run.status == "completed"
    assert len(steps) == 1


def test_observe_run_with_options(db) -> None:
    captured: dict[str, str] = {}

    @observe_run(agent_type="triage-agent", prompt_from=lambda args, kwargs: kwargs["question"])
    def triage(*, question: str) -> None:
        captured["run_id"] = require_tracer().run_id

    triage(question="Assess vitals.")
    run, _ = _load_run(captured["run_id"])
    assert run.agent_type == "triage-agent"
    assert run.initial_prompt == "Assess vitals."


def test_observe_run_marks_crash(db) -> None:
    captured: dict[str, str] = {}

    @observe_run(agent_type="crashy-agent")
    def crashy(prompt: str) -> None:
        captured["run_id"] = require_tracer().run_id
        raise RuntimeError("agent exploded")

    with pytest.raises(RuntimeError, match="agent exploded"):
        crashy("Do risky thing.")

    run, _ = _load_run(captured["run_id"])
    assert run.status == "failed"


def test_observe_run_async(db) -> None:
    captured: dict[str, str] = {}

    @observe_run(agent_type="async-agent")
    async def triage(prompt: str) -> str:
        captured["run_id"] = require_tracer().run_id
        record_step("assess()", output="stable")
        return "done"

    assert asyncio.run(triage("Assess vitals.")) == "done"
    run, steps = _load_run(captured["run_id"])
    assert run.agent_type == "async-agent"
    assert run.status == "completed"
    assert len(steps) == 1
