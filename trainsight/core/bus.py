from __future__ import annotations

from collections import defaultdict
from typing import Callable

from .event import Event


class EventBus:
    def __init__(self) -> None:
        self.subscribers: dict[str, list[Callable[[Event], None]]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: Callable[[Event], None]) -> None:
        self.subscribers[event_type].append(handler)

    def emit(self, event: Event) -> None:
        for handler in self.subscribers.get(event.type, []):
            handler(event)
