from __future__ import annotations

import time
import uuid
from collections import Counter
from dataclasses import dataclass
from types import TracebackType
from typing import Any

from sqlalchemy.orm import Session

from .db import get_session
from .models import Run, Step, utcnow
from .sanitizer import Sanitizer, SanitizeResult
from .writer import BackgroundWriter, FinishWriteRequest, StepWriteRequest


@dataclass(frozen=True)
class PromptWriteRequest:
    initial_prompt: str


def _merge_counts(*count_maps: dict[str, int]) -> dict[str, int]:
    merged: Counter[str] = Counter()
    for counts in count_maps:
        merged.update(counts)
    return dict(merged)


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
        self._background = background
        self._writer: BackgroundWriter | None = None

        if background:
            # Placeholder run row so the id exists; prompt is filled on the worker.
            self._run = Run(
                id=self.run_id,
                agent_type=agent_type,
                initial_prompt="",
                redaction_count=0,
                entity_counts={},
            )
            self._session.add(self._run)
            self._session.commit()
            self._writer = BackgroundWriter(
                self._handle_background_item,
                maxsize=queue_maxsize,
                name=f"auscult-writer-{self.run_id[:8]}",
            )
            self._writer.start()
            self._writer.submit(PromptWriteRequest(initial_prompt=initial_prompt))
        else:
            prompt_result = self._sanitizer.sanitize_with_stats(initial_prompt)
            self._redaction_count += prompt_result.redaction_count
            self._entity_counts.update(prompt_result.entity_counts)
            self._run = Run(
                id=self.run_id,
                agent_type=agent_type,
                initial_prompt=prompt_result.text or "",
                redaction_count=self._redaction_count,
                entity_counts=dict(self._entity_counts),
            )
            self._session.add(self._run)
            self._session.commit()

    @property
    def finished(self) -> bool:
        return self._finished

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

        if self._writer is not None:
            step_index = self._step_count
            self._step_count += 1
            if error_message is not None and self._first_failed_step is None:
                self._first_failed_step = step_index
            self._writer.submit(
                StepWriteRequest(
                    step_index=step_index,
                    llm_command=llm_command,
                    output=output,
                    error_message=error_message,
                    enqueued_at=time.perf_counter(),
                )
            )
            return

        started = time.perf_counter()

        # Sanitize before any counter bump or DB write. If sanitization fails,
        # nothing is stored for this step and step_count is unchanged (fail-closed).
        command_result = self._sanitizer.sanitize_with_stats(llm_command)
        output_result = self._sanitizer.sanitize_with_stats(output)
        error_result = self._sanitizer.sanitize_with_stats(error_message)
        self._persist_step(
            step_index=self._step_count,
            command_result=command_result,
            output_result=output_result,
            error_result=error_result,
            error_message=error_message,
            duration=time.perf_counter() - started,
        )
        self._step_count += 1

    def _persist_step(
        self,
        *,
        step_index: int,
        command_result: SanitizeResult,
        output_result: SanitizeResult,
        error_result: SanitizeResult,
        error_message: str | None,
        duration: float,
    ) -> None:
        step_redactions = (
            command_result.redaction_count
            + output_result.redaction_count
            + error_result.redaction_count
        )
        step_entities = _merge_counts(
            command_result.entity_counts,
            output_result.entity_counts,
            error_result.entity_counts,
        )
        if error_message is not None and self._first_failed_step is None:
            self._first_failed_step = step_index
        self._redaction_count += step_redactions
        self._entity_counts.update(step_entities)

        step = Step(
            run_id=self.run_id,
            step_index=step_index,
            llm_command=command_result.text or "",
            output=output_result.text,
            error_message=error_result.text,
            time_for_completion=duration,
            redaction_count=step_redactions,
            entity_counts=step_entities,
        )
        self._session.add(step)
        self._run.redaction_count = self._redaction_count
        self._run.entity_counts = dict(self._entity_counts)
        if self._commit_each_step:
            self._session.commit()
        else:
            self._session.flush()

    def _handle_background_item(self, item: Any) -> None:
        # All sanitizer + counter + session mutations for background mode happen
        # on this single worker thread (queue is FIFO).
        if isinstance(item, PromptWriteRequest):
            prompt_result = self._sanitizer.sanitize_with_stats(item.initial_prompt)
            self._redaction_count += prompt_result.redaction_count
            self._entity_counts.update(prompt_result.entity_counts)
            self._run.initial_prompt = prompt_result.text or ""
            self._run.redaction_count = self._redaction_count
            self._run.entity_counts = dict(self._entity_counts)
            self._session.commit()
            return

        if isinstance(item, StepWriteRequest):
            started = item.enqueued_at
            command_result = self._sanitizer.sanitize_with_stats(item.llm_command)
            output_result = self._sanitizer.sanitize_with_stats(item.output)
            error_result = self._sanitizer.sanitize_with_stats(item.error_message)
            step_redactions = (
                command_result.redaction_count
                + output_result.redaction_count
                + error_result.redaction_count
            )
            step_entities = _merge_counts(
                command_result.entity_counts,
                output_result.entity_counts,
                error_result.entity_counts,
            )
            self._redaction_count += step_redactions
            self._entity_counts.update(step_entities)
            step = Step(
                run_id=self.run_id,
                step_index=item.step_index,
                llm_command=command_result.text or "",
                output=output_result.text,
                error_message=error_result.text,
                time_for_completion=time.perf_counter() - started,
                redaction_count=step_redactions,
                entity_counts=step_entities,
            )
            self._session.add(step)
            self._run.redaction_count = self._redaction_count
            self._run.entity_counts = dict(self._entity_counts)
            if self._commit_each_step:
                self._session.commit()
            else:
                self._session.flush()
            return

        if isinstance(item, FinishWriteRequest):
            self._run.status = (
                "failed"
                if (item.crashed or item.first_failed_step is not None)
                else "completed"
            )
            self._run.failed_step = item.first_failed_step
            self._run.total_steps = item.total_steps
            self._run.redaction_count = self._redaction_count
            self._run.entity_counts = dict(self._entity_counts)
            self._run.finished_at = utcnow()
            self._session.commit()
            self._session.close()
            return

        raise TypeError(f"unknown background write item: {type(item)!r}")

    def finish(self, *, crashed: bool = False) -> None:
        self._ensure_active()

        if self._writer is not None:
            self._writer.submit(
                FinishWriteRequest(
                    crashed=crashed,
                    total_steps=self._step_count,
                    first_failed_step=self._first_failed_step,
                )
            )
            self._writer.close()
            self._finished = True
            return

        failed = crashed or self._first_failed_step is not None
        self._run.status = "failed" if failed else "completed"
        self._run.failed_step = self._first_failed_step
        self._run.total_steps = self._step_count
        self._run.redaction_count = self._redaction_count
        self._run.entity_counts = dict(self._entity_counts)
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
