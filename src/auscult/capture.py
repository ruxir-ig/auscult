from __future__ import annotations

import time
import uuid
from types import TracebackType

from sqlalchemy.orm import Session

from .db import get_session
from .models import Run, Step, utcnow
from .sanitizer import Sanitizer


class AuscultTracer:
    """Records agent runs, sanitizing all free text before it is stored.

    Raw patient text is never written to the database: every text field
    passes through the per-run Sanitizer first.

    One database session is held for the lifetime of the run so high-frequency
    agents avoid opening a new connection per step. By default each step is
    committed immediately; pass ``commit_each_step=False`` to flush steps and
    commit once in ``finish()`` instead.
    """

    def __init__(
        self,
        agent_type: str,
        initial_prompt: str,
        *,
        commit_each_step: bool = True,
    ) -> None:
        self.run_id: str = str(uuid.uuid4())
        self._step_count: int = 0
        self._first_failed_step: int | None = None
        self._sanitizer = Sanitizer()
        self._commit_each_step = commit_each_step
        self._finished = False
        self._session: Session = get_session()

        self._run = Run(
            id=self.run_id,
            agent_type=agent_type,
            initial_prompt=self._sanitizer.sanitize(initial_prompt),
        )
        self._session.add(self._run)
        self._session.commit()

    def _ensure_active(self) -> None:
        if self._finished:
            raise RuntimeError("Cannot use AuscultTracer after finish()")

    def record_step(
        self,
        llm_command: str,
        output: str | None,
        error_message: str | None = None,
    ) -> None:
        self._ensure_active()
        started = time.perf_counter()

        step_index = self._step_count
        self._step_count += 1
        if error_message is not None and self._first_failed_step is None:
            self._first_failed_step = step_index

        step = Step(
            run_id=self.run_id,
            step_index=step_index,
            llm_command=self._sanitizer.sanitize(llm_command),
            output=self._sanitizer.sanitize(output),
            error_message=self._sanitizer.sanitize(error_message),
            time_for_completion=time.perf_counter() - started,
        )
        self._session.add(step)
        if self._commit_each_step:
            self._session.commit()
        else:
            self._session.flush()

    def finish(self, *, crashed: bool = False) -> None:
        self._ensure_active()

        failed = crashed or self._first_failed_step is not None
        self._run.status = "failed" if failed else "completed"
        self._run.failed_step = self._first_failed_step
        self._run.total_steps = self._step_count
        self._run.finished_at = utcnow()
        self._session.commit()
        self._session.close()
        self._finished = True

    def __enter__(self) -> AuscultTracer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if not self._finished:
            self.finish(crashed=exc_type is not None)
