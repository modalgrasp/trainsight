"""
Tests for trainsight.integrations.pytorch.TrainSightHook.

Hardware-independent: uses a tiny CPU model and a mock monitor.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

# Skip entire module if torch is not installed
torch = pytest.importorskip("torch")
import torch.nn as nn

from trainsight.integrations.pytorch import TrainSightHook, monitor_pytorch_training


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def make_monitor():
    monitor = MagicMock()
    monitor.event_bus = MagicMock()
    monitor.event_bus.emit = MagicMock()
    monitor.event_bus.emit_async = None  # force sync emit
    return monitor


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_attach_and_detach():
    model = TinyModel()
    monitor = make_monitor()
    hook = TrainSightHook(model, monitor, sample_interval=1)

    hook.attach()
    assert len(hook._hooks) > 0

    hook.detach()
    assert len(hook._hooks) == 0


def test_parameter_snapshot_on_attach():
    model = TinyModel()
    monitor = make_monitor()
    hook = TrainSightHook(model, monitor, track_parameters=True, sample_interval=1)
    hook.attach()

    assert len(hook._param_snapshots) > 0
    hook.detach()


def test_parameter_drift_emits_event():
    model = TinyModel()
    monitor = make_monitor()
    hook = TrainSightHook(model, monitor, track_parameters=True, sample_interval=1)
    hook.attach()

    # Mutate parameters to create drift
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.ones_like(p) * 0.5)

    drift = hook.check_parameter_drift()
    assert len(drift) > 0
    monitor.event_bus.emit.assert_called()
    hook.detach()


def test_gradient_explosion_event():
    model = TinyModel()
    monitor = make_monitor()
    hook = TrainSightHook(
        model,
        monitor,
        sample_interval=1,
        grad_explosion_threshold=10.0,
    )
    hook.attach()

    # Manually inject a large gradient to trigger explosion
    with torch.no_grad():
        large_grad = torch.ones(16, 8) * 500.0  # fc1 weight shape

    # Reset step counter so the next call to _should_sample() returns True
    hook._step_counter = 0
    hook._last_sample_ts = 0.0

    grad_hook = hook._make_grad_hook("fc1")
    grad_hook(large_grad)

    # Check that an explosion event was emitted
    calls = [str(c) for c in monitor.event_bus.emit.call_args_list]
    assert any("gradient_explosion" in c for c in calls)
    hook.detach()


def test_vanishing_gradient_event():
    model = TinyModel()
    monitor = make_monitor()
    hook = TrainSightHook(
        model,
        monitor,
        sample_interval=1,
        grad_vanishing_threshold=1.0,  # high threshold for test
    )
    hook.attach()

    tiny_grad = torch.zeros_like(model.fc1.weight) + 1e-10
    grad_hook = hook._make_grad_hook("fc1")
    grad_hook(tiny_grad)

    calls = [str(c) for c in monitor.event_bus.emit.call_args_list]
    assert any("gradient_vanishing" in c for c in calls)
    hook.detach()


def test_layer_health_report():
    model = TinyModel()
    monitor = make_monitor()
    hook = TrainSightHook(model, monitor, sample_interval=1)
    hook.attach()

    x = torch.randn(4, 8)
    out = model(x)
    out.mean().backward()

    report = hook.get_layer_health_report()
    assert "gradient_norms" in report
    assert "activation_stats" in report
    assert "step_count" in report
    hook.detach()


def test_monitor_pytorch_training_convenience():
    model = TinyModel()
    monitor = make_monitor()
    hook = monitor_pytorch_training(model, monitor, sample_interval=5)
    assert isinstance(hook, TrainSightHook)
    assert len(hook._hooks) > 0
    hook.detach()


def test_no_hooks_when_disabled():
    model = TinyModel()
    monitor = make_monitor()
    hook = TrainSightHook(
        model,
        monitor,
        track_gradients=False,
        track_activations=False,
        track_parameters=False,
    )
    hook.attach()
    # No hooks should be registered (no activations, no grad hooks)
    assert len(hook._hooks) == 0
    hook.detach()
