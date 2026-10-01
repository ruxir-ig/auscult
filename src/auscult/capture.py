from __future__ import annotations

import time
import uuid
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import TracebackType

from sqlalchemy.orm import Session

from .db import get_session
from .models import Run, Step, utcnow
from .sanitizer import Sanitizer, SanitizeResult
from .writer import BackgroundWriter


@dataclass(frozen=True)
class PromptWriteRequest:
    initial_prompt: str


@dataclass(frozen=True)
class StepWriteRequest:
    step_index: int
    llm_command: str
    output: str | None
    error_message: str | None
    started_at: float


@dataclass(frozen=True)
class FinishWriteRequest:
    crashed: bool
    total_steps: int
    first_failed_step: int | None


type WriteRequest = PromptWriteRequest | StepWriteRequest | FinishWriteRequest


class AuscultTracer:
    """Records agent runs, sanitizing all free text before it is stored.

    Raw patient text is never written to the database: every text field
    passes through the per-run Sanitizer first.

    One database session is held for the lifetime of the run so high-frequency
    agents avoid opening a new connection per step. By default each step is
    committed immediately; pass ``commit_each_step=False`` to flush steps and
    commit once in ``finish()`` instead.

    Pass ``background=True`` to enqueue sanitize+DB work on a worker thread so
    the agent hot path is not blocked by spaCy / Postgres. Fail-closed: a full
    queue or worker error raises rather than dropping steps.

    Sanitizer failures are fail-closed: if ``sanitize`` raises, the step is not
    written and the exception propagates to the caller (sync mode) or via
    ``finish()`` / the next ``record_step`` (background mode).

    Faker replacements are seeded from ``run_id`` by default so the same raw
    input yields the same synthetic values across re-captures of that run.
    """

    def __init__(
        self,
        agent_type: str,
        initial_prompt: str,
        *,
        commit_each_step: bool = True,
        sanitizer: Sanitizer | None = None,
        background: bool = False,
        queue_maxsize: int = 256,
        run_id: str | None = None,
    ) -> None:
        self.run_id: str = run_id or str(uuid.uuid4())
        self._step_count: int = 0
        self._first_failed_step: int | None = None
        self._sanitizer = sanitizer or Sanitizer(seed=self.run_id)
        self._commit_each_step = commit_each_step
        self._finished = False
        self._session: Session = get_session()
        self._redaction_count: int = 0
        self._entity_counts: Counter[str] = Counter()
        self._writer: BackgroundWriter[WriteRequest] | None = None
        self._run = Run(
            id=self.run_id,
            agent_type=agent_type,
            initial_prompt="",
            redaction_count=0,
            entity_counts={},
        )

        if background:
            # Commit a placeholder row so the run id exists before the worker
            # fills in the sanitized prompt.
            self._session.add(self._run)
            self._session.commit()
            self._writer = BackgroundWriter(
                self._write,
                maxsize=queue_maxsize,
                name=f"auscult-writer-{self.run_id[:8]}",
            )
        self._dispatch(PromptWriteRequest(initial_prompt=initial_prompt))

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def step_count(self) -> int:
        """Steps recorded so far; also the index the next step will get."""
        return self._step_count

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
        step_index = self._step_count
        # If dispatch raises (sanitizer error, full queue), the step is not
        # counted (fail-closed).
        self._dispatch(
            StepWriteRequest(
                step_index=step_index,
                llm_command=llm_command,
                output=output,
                error_message=error_message,
                started_at=time.perf_counter(),
            )
        )
        self._step_count += 1
        if error_message is not None and self._first_failed_step is None:
            self._first_failed_step = step_index

    def finish(self, *, crashed: bool = False) -> None:
        self._ensure_active()
        self._dispatch(
            FinishWriteRequest(
                crashed=crashed,
                total_steps=self._step_count,
                first_failed_step=self._first_failed_step,
            )
        )
        if self._writer is not None:
            self._writer.close()
        self._finished = True

    def _dispatch(self, request: WriteRequest) -> None:
        if self._writer is not None:
            self._writer.submit(request)
        else:
            self._write(request)

    def _write(self, request: WriteRequest) -> None:
        # In background mode this runs on the single worker thread, which owns
        # the sanitizer, the counters, and the session.
        match request:
            case PromptWriteRequest():
                (prompt,) = self._sanitize(request.initial_prompt)
                self._run.initial_prompt = prompt.text or ""
                self._add_counts(prompt.redaction_count, prompt.entity_counts)
                self._session.add(self._run)
                self._session.commit()
            case StepWriteRequest():
                results = self._sanitize(
                    request.llm_command, request.output, request.error_message
                )
                command, output, error = results
                redactions = sum(r.redaction_count for r in results)
                entities: Counter[str] = Counter()
                for result in results:
                    entities.update(result.entity_counts)
                self._session.add(
                    Step(
                        run_id=self.run_id,
                        step_index=request.step_index,
                        llm_command=command.text or "",
                        output=output.text,
                        error_message=error.text,
                        time_for_completion=time.perf_counter() - request.started_at,
                        redaction_count=redactions,
                        entity_counts=dict(entities),
                    )
                )
                self._add_counts(redactions, entities)
                if self._commit_each_step:
                    self._session.commit()
                else:
                    self._session.flush()
            case FinishWriteRequest():
                failed = request.crashed or request.first_failed_step is not None
                self._run.status = "failed" if failed else "completed"
                self._run.failed_step = request.first_failed_step
                self._run.total_steps = request.total_steps
                self._run.finished_at = utcnow()
                self._session.commit()
                self._session.close()

    def _sanitize(self, *texts: str | None) -> tuple[SanitizeResult, ...]:
        return tuple(self._sanitizer.sanitize_with_stats(text) for text in texts)

    def _add_counts(self, redactions: int, entities: Mapping[str, int]) -> None:
        self._redaction_count += redactions
        self._entity_counts.update(entities)
        self._run.redaction_count = self._redaction_count
        self._run.entity_counts = dict(self._entity_counts)

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
