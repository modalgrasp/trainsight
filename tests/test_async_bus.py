"""
Tests for trainsight.core.async_bus.AsyncEventBus.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

from trainsight.core.async_bus import AsyncEventBus
from trainsight.core.event import Event


def _make_event(event_type: str = "test.event") -> Event:
    return Event(type=event_type, payload={"value": 42}, timestamp=datetime.now(timezone.utc))


def test_sync_emit_still_works():
    bus = AsyncEventBus()
    received = []
    bus.subscribe("test.event", lambda e: received.append(e.payload["value"]))
    bus.emit(_make_event())
    assert received == [42]
    bus.shutdown()


def test_async_emit_delivers_event():
    bus = AsyncEventBus()
    received = []
    bus.subscribe("test.event", lambda e: received.append(e.payload["value"]))
    bus.emit_async(_make_event())

    # Give the background worker time to process
    deadline = time.monotonic() + 2.0
    while not received and time.monotonic() < deadline:
        time.sleep(0.01)

    assert received == [42]
    bus.shutdown()


def test_async_emit_drops_when_queue_full():
    """Verify that emit_async never raises even when the queue is full."""
    bus = AsyncEventBus(maxsize=1)
    # Flood the queue
    for _ in range(100):
        bus.emit_async(_make_event())
    # No exception should have been raised
    bus.shutdown()


def test_multiple_subscribers():
    bus = AsyncEventBus()
    results: list[int] = []
    bus.subscribe("x", lambda e: results.append(1))
    bus.subscribe("x", lambda e: results.append(2))
    bus.emit(_make_event("x"))
    assert sorted(results) == [1, 2]
    bus.shutdown()


def test_shutdown_is_idempotent():
    bus = AsyncEventBus()
    bus.shutdown()
    bus.shutdown()  # should not raise
