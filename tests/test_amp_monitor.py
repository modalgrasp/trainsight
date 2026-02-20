"""
Tests for trainsight.ml.amp_monitor.AMPStabilityMonitor.
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock

from trainsight.ml.amp_monitor import AMPStabilityMonitor


def make_monitor():
    m = MagicMock()
    m.event_bus = MagicMock()
    m.event_bus.emit = MagicMock()
    m.event_bus.emit_async = None
    return m


def test_stable_training_no_event():
    monitor = make_monitor()
    amp = AMPStabilityMonitor(monitor, window_size=50, cv_threshold=0.5)

    # Feed stable losses
    for i in range(100):
        amp.update(loss=1.0, scaler_scale=1024.0)

    calls = [str(c) for c in monitor.event_bus.emit.call_args_list]
    assert not any("amp_instability" in c for c in calls)


def test_high_variance_triggers_instability():
    monitor = make_monitor()
    amp = AMPStabilityMonitor(monitor, window_size=50, cv_threshold=0.1)

    # Feed highly variable losses
    for i in range(100):
        loss = 1.0 if i % 2 == 0 else 100.0
        amp.update(loss=loss)

    calls = [str(c) for c in monitor.event_bus.emit.call_args_list]
    assert any("amp_instability" in c for c in calls)


def test_high_overflow_rate_triggers_instability():
    monitor = make_monitor()
    amp = AMPStabilityMonitor(
        monitor,
        window_size=50,
        cv_threshold=10.0,  # high – won't trigger on variance
        overflow_rate_threshold=0.05,
    )

    # Feed 50% overflow rate
    for i in range(100):
        amp.update(loss=1.0, has_overflow=(i % 2 == 0))

    calls = [str(c) for c in monitor.event_bus.emit.call_args_list]
    assert any("amp_instability" in c for c in calls)


def test_nan_count_tracked():
    monitor = make_monitor()
    amp = AMPStabilityMonitor(monitor)

    for i in range(10):
        amp.update(loss=1.0, has_nan=(i < 3))

    assert amp._nan_count == 3


def test_history_bounded():
    monitor = make_monitor()
    amp = AMPStabilityMonitor(monitor, window_size=20)

    for i in range(200):
        amp.update(loss=1.0)

    assert len(amp._loss_history) <= 20
