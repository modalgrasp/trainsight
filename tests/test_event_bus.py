from trainsight.core.bus import EventBus
from trainsight.core.event import Event


def test_event_bus_emit_subscribe():
    bus = EventBus()
    got = []

    def handler(evt):
        got.append(evt.payload["x"])

    bus.subscribe("gpu.stats", handler)
    bus.emit(Event(type="gpu.stats", payload={"x": 1}))
    assert got == [1]
