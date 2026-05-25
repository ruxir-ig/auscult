from sqlalchemy import select

from auscult.capture import AuscultTracer
from auscult.db import SessionLocal
from auscult.models import Run, Step


def main() -> None:
    tracer = AuscultTracer(
        agent_type="fake-agent",
        initial_prompt="Diagnose the patient's symptoms.",
    )
    print(f"Created run: {tracer.run_id}")

    tracer.record_step(
        llm_command="ask_patient(symptoms)",
        output="Patient reports headache and fever.",
    )
    tracer.record_step(
        llm_command="lookup_conditions(['headache', 'fever'])",
        output="Possible: flu, migraine, sinus infection.",
    )
    tracer.record_step(
        llm_command="recommend_next_step()",
        output="Order CBC panel.",
    )

    tracer.finish()
    print("Run marked finished.\n")

    with SessionLocal() as session:
        print("=== runs ===")
        for run in session.execute(select(Run)).scalars():
            print(
                f"id={run.id} agent_type={run.agent_type} status={run.status} "
                f"total_steps={run.total_steps} failed_step={run.failed_step} "
                f"started_at={run.started_at} finished_at={run.finished_at} "
                f"initial_prompt={run.initial_prompt!r}"
            )

        print("\n=== steps ===")
        for step in session.execute(select(Step).order_by(Step.run_id, Step.step_index)).scalars():
            print(
                f"run_id={step.run_id} step_index={step.step_index} "
                f"llm_command={step.llm_command!r} output={step.output!r} "
                f"error_message={step.error_message!r}"
            )


if __name__ == "__main__":
    main()
