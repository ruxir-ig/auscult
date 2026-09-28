"""Background writer so sanitization + DB I/O stay off the agent hot path.

When ``AuscultTracer(..., background=True)`` is used, ``record_step`` enqueues
raw text and returns immediately. A dedicated worker thread sanitizes and
persists each step.

Fail-closed semantics:
- Bounded queue: if full, ``record_step`` raises (never silently drops).
- Worker exceptions are stored and re-raised on the next ``record_step`` /
  ``finish`` call.
- ``finish()`` drains the queue and joins the worker before returning.

Note: the in-memory queue briefly holds unsanitized text until the worker
processes it. Keep ``maxsize`` small and treat process crashes before drain
as lost steps (same as an uncommitted sync write).
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable


class BackgroundWriter[T]:
    """Single-consumer queue that runs ``handler`` on a daemon thread.

    ``None`` is reserved as the stop sentinel, so items must not be ``None``.
    """

    def __init__(
        self,
        handler: Callable[[T], None],
        *,
        maxsize: int = 256,
        name: str = "auscult-writer",
    ) -> None:
        self._handler = handler
        self._queue: queue.Queue[T | None] = queue.Queue(maxsize=maxsize)
        self._error: BaseException | None = None
        self._error_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, item: T, *, timeout: float | None = 5.0) -> None:
        self._raise_if_failed()
        try:
            self._queue.put(item, timeout=timeout)
        except queue.Full as exc:
            raise RuntimeError(
                "Auscult background writer queue is full; refusing to drop a step "
                "(fail-closed). Increase maxsize or slow the agent, or disable "
                "background=True."
            ) from exc

    def close(self, *, timeout: float | None = 60.0) -> None:
        """Signal the worker to stop after draining, then join."""
        self._raise_if_failed()
        self._queue.put(None)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise RuntimeError("Auscult background writer did not finish in time")
        self._raise_if_failed()

    def _raise_if_failed(self) -> None:
        with self._error_lock:
            if self._error is not None:
                raise RuntimeError("Auscult background writer failed") from self._error

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                self._handler(item)
            except BaseException as exc:  # noqa: BLE001 — capture for fail-closed
                with self._error_lock:
                    self._error = exc
                # Drain remaining items so put() callers are not blocked forever.
                while True:
                    try:
                        leftover = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if leftover is None:
                        break
                return
