import uuid
from datetime import datetime

from .db import SessionLocal
from .models import Run, Step


class AuscultTracer:
    def __init__(self, agent_type: str, initial_prompt: str) -> None:
        self.run_id: str = str(uuid.uuid4())
        self._step_count: int = 0
        self._first_failed_step: int | None = None

        run = Run(
            id=self.run_id,
            agent_type=agent_type,
            initial_prompt=initial_prompt,
        )
        with SessionLocal() as session:
            session.add(run)
            session.commit()

    def record_step(
        self,
        llm_command: str,
        output: str | None,
        error_message: str | None = None,
    ) -> None:
        step_index = self._step_count
        self._step_count += 1
        if error_message is not None and self._first_failed_step is None:
            self._first_failed_step = step_index

        step = Step(
            run_id=self.run_id,
            step_index=step_index,
            llm_command=llm_command,
            output=output,
            error_message=error_message,
        )
        with SessionLocal() as session:
            session.add(step)
            session.commit()

    def finish(self) -> None:
        status = "failed" if self._first_failed_step is not None else "completed"
        with SessionLocal() as session:
            run = session.get(Run, self.run_id)
            run.status = status
            run.failed_step = self._first_failed_step
            run.total_steps = self._step_count
            run.finished_at = datetime.utcnow()
            session.commit()
