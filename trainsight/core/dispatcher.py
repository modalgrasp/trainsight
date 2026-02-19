from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol

from .bus import EventBus
from .event import Event


class Collector(Protocol):
    event_type: str

    def collect(self) -> dict | None:
        ...


class Dispatcher:
    def __init__(self, bus: EventBus, collectors: list[Collector]) -> None:
        self.bus = bus
        self.collectors = collectors

    def collect_once(self) -> None:
        for collector in self.collectors:
            payload = collector.collect()
            if payload is None:
                continue
            self.bus.emit(Event(type=collector.event_type, payload=payload, timestamp=datetime.now(timezone.utc)))
