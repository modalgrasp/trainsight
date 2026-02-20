"""
Tests for trainsight.ml.loss_monitor.LossTrendMonitor.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from trainsight.ml.loss_monitor import LossTrendMonitor


def make_monitor():
    m = MagicMock()
    m.event_bus = MagicMock()
    m.event_bus.emit = MagicMock()
    m.event_bus.emit_async = None
    return m


def test_plateau_detection():
    monitor = make_monitor()
    lm = LossTrendMonitor(
        monitor,
        window_size=20,
        plateau_threshold=0.01,
        divergence_threshold=1.0,
    )

    # Feed a flat loss curve
    for step in range(60):
        lm.update(step, 1.0)

    calls = [str(c) for c in monitor.event_bus.emit.call_args_list]
    assert any("loss_plateau" in c for c in calls)


def test_divergence_detection():
    monitor = make_monitor()
    lm = LossTrendMonitor(
        monitor,
        window_size=20,
        plateau_threshold=0.0001,
        divergence_threshold=0.05,
    )

    # Feed a strongly increasing loss curve
    for step in range(60):
        lm.update(step, float(step) * 0.5)

    calls = [str(c) for c in monitor.event_bus.emit.call_args_list]
    assert any("loss_divergence" in c for c in calls)


def test_overfitting_detection():
    monitor = make_monitor()
    lm = LossTrendMonitor(
        monitor,
        window_size=20,
        overfit_gap_pct=10.0,
    )

    # Train loss decreasing, val loss much higher
    for step in range(60):
        train_loss = 1.0 - step * 0.005
        val_loss = 2.0  # always much higher
        lm.update(step, train_loss, val_loss)

    calls = [str(c) for c in monitor.event_bus.emit.call_args_list]
    assert any("overfitting" in c for c in calls)


def test_no_events_with_insufficient_data():
    monitor = make_monitor()
    lm = LossTrendMonitor(monitor, window_size=50)

    for step in range(5):
        lm.update(step, 1.0)

    monitor.event_bus.emit.assert_not_called()


def test_history_is_bounded():
    monitor = make_monitor()
    lm = LossTrendMonitor(monitor, window_size=20)

    for step in range(500):
        lm.update(step, 1.0)

    # History should be capped at 2× window_size
    assert len(lm._train_losses) <= 40
    assert len(lm._steps) <= 40
