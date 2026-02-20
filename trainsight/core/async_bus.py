from __future__ import annotations

import asyncio
import logging
import threading
from queue import Empty, Queue
from typing import Callable

from .bus import EventBus
from .event import Event

logger = logging.getLogger("trainsight.async_bus")


class AsyncEventBus(EventBus):
    """
    Thread-safe, non-blocking EventBus with background processing queue.

    Integrations that run inside training loops should call ``emit_async``
    instead of ``emit`` so that handler execution never blocks the hot path.
    The background worker thread drains the queue and dispatches events to
    all registered subscribers.
    """

    def __init__(self, maxsize: int = 2000) -> None:
        super().__init__()
        self._queue: Queue[Event] = Queue(maxsize=maxsize)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(
            target=self._drain_loop,
            name="trainsight-event-worker",
            daemon=True,
        )
        self._worker.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def emit_async(self, event: Event) -> None:
        """Non-blocking emit – drops the event if the queue is full."""
        try:
            self._queue.put_nowait(event)
        except Exception:
            # Queue full – silently drop to avoid blocking training.
            pass

    def shutdown(self, timeout: float = 2.0) -> None:
        """Stop the background worker gracefully."""
        self._stop_event.set()
        self._worker.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _drain_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=0.05)
                try:
                    self.emit(event)
                except Exception as exc:
                    logger.debug("Event handler error: %s", exc)
                finally:
                    self._queue.task_done()
            except Empty:
                continue
