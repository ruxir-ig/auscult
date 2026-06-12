from auscult.capture import AuscultTracer
from auscult.db import get_session
from auscult.models import Step
from auscult.replay import (
    RunNotFoundError,
    RunReplayer,
    format_playback,
    load_run,
)
from sqlalchemy import select


def _record_sample_run() -> str:
    tracer = AuscultTracer(
        agent_type="test-agent",
        initial_prompt="Triage the patient.",
    )
    tracer.record_step(
        llm_command="ask_patient(symptoms)",
        output="Reports headache.",
    )
    tracer.record_step(
        llm_command="order_labs()",
        output=None,
        error_message="timeout",
    )
    tracer.finish()
    return tracer.run_id


def test_load_run_returns_recorded_steps(db) -> None:
    run_id = _record_sample_run()

    replay_run = load_run(run_id)

    assert replay_run.run_id == run_id
    assert replay_run.agent_type == "test-agent"
    assert replay_run.initial_prompt == "Triage the patient."
    assert replay_run.status == "failed"
    assert replay_run.failed_step == 1
    assert len(replay_run.steps) == 2
    assert replay_run.steps[0].llm_command == "ask_patient(symptoms)"
    assert replay_run.steps[1].error_message == "timeout"


def test_load_run_raises_for_missing_id(db) -> None:
    try:
        load_run("missing-run-id")
    except RunNotFoundError as exc:
        assert exc.run_id == "missing-run-id"
    else:
        raise AssertionError("expected RunNotFoundError")


def test_playback_yields_steps_in_order(db) -> None:
    run_id = _record_sample_run()
    replayer = RunReplayer.from_run_id(run_id)

    steps = list(replayer.playback())

    assert [step.step_index for step in steps] == [0, 1]
    assert steps[0].output == "Reports headache."


def test_replay_matches_when_handler_returns_recorded_values(db) -> None:
    run_id = _record_sample_run()
    replayer = RunReplayer.from_run_id(run_id)
    recorded = {step.llm_command: (step.output, step.error_message) for step in replayer.run.steps}

    def handler(command: str) -> tuple[str | None, str | None]:
        return recorded[command]

    result = replayer.replay(handler)

    assert result.all_matched
    assert len(result.steps) == 2


def test_replay_detects_output_mismatch(db) -> None:
    run_id = _record_sample_run()
    replayer = RunReplayer.from_run_id(run_id)

    def handler(command: str) -> tuple[str | None, str | None]:
        if command == "ask_patient(symptoms)":
            return ("Different output.", None)
        return (None, "timeout")

    result = replayer.replay(handler)

    assert not result.all_matched
    assert result.steps[0].output_match is False
    assert result.steps[1].matched is True


def test_format_playback_includes_prompt_and_steps(db) -> None:
    run_id = _record_sample_run()
    replay_run = load_run(run_id)

    text = format_playback(replay_run)

    assert f"Replaying run {run_id}" in text
    assert "initial_prompt: Triage the patient." in text
    assert "step 0: ask_patient(symptoms)" in text
    assert "-> Reports headache." in text
    assert "! timeout" in text


def test_replay_reads_sanitized_database_rows(db) -> None:
    run_id = _record_sample_run()

    with get_session() as session:
        step = (
            session.execute(select(Step).where(Step.run_id == run_id, Step.step_index == 0))
            .scalars()
            .one()
        )
        stored_output = step.output

    replay_run = load_run(run_id)
    assert replay_run.steps[0].output == stored_output
