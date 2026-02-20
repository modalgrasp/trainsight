"""
trainsight.integrations.pytorch
================================
Non-invasive PyTorch training monitor.

Attaches lightweight forward/backward hooks to a ``torch.nn.Module`` and
emits TrainSight events for:

* ``training.activation``          – per-layer activation statistics
* ``training.gradient_explosion``  – grad norm > threshold
* ``training.gradient_vanishing``  – grad norm < threshold
* ``training.parameter_drift``     – L2 distance from last snapshot

All analysis runs inside ``torch.no_grad()`` and is throttled by
``sample_interval`` (steps) and a minimum wall-clock gap so that the
training loop is never blocked.

Usage::

    from trainsight import Dashboard
    from trainsight.integrations.pytorch import TrainSightHook

    model = MyModel()
    dashboard = Dashboard()
    hook = TrainSightHook(model, dashboard, sample_interval=10)
    hook.attach()

    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()

    hook.detach()
"""
from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for the native integration: pip install torch"
    ) from exc

from trainsight.core.event import Event


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
_GRAD_EXPLOSION_THRESHOLD = 100.0
_GRAD_VANISHING_THRESHOLD = 1e-7
_MIN_SAMPLE_GAP_S = 0.1  # 100 ms minimum between samples


class TrainSightHook:
    """
    Attach gradient / activation / parameter-drift hooks to a PyTorch model.

    Parameters
    ----------
    model:
        The ``nn.Module`` to monitor.
    monitor:
        A TrainSight ``Dashboard`` instance (or any object with an
        ``event_bus`` attribute that is an ``EventBus``).
    sample_interval:
        Emit events every *N* forward/backward passes.  Default 10.
    track_gradients:
        Register backward hooks on leaf-parameter weights.
    track_activations:
        Register forward hooks on leaf modules.
    track_parameters:
        Snapshot parameters and compute drift periodically.
    max_layers_tracked:
        Maximum number of leaf layers to instrument.
    grad_explosion_threshold:
        Gradient L2 norm above this value triggers an explosion event.
    grad_vanishing_threshold:
        Gradient L2 norm below this value triggers a vanishing event.
    """

    def __init__(
        self,
        model: nn.Module,
        monitor: Any,
        sample_interval: int = 10,
        track_gradients: bool = True,
        track_activations: bool = True,
        track_parameters: bool = True,
        max_layers_tracked: int = 20,
        grad_explosion_threshold: float = _GRAD_EXPLOSION_THRESHOLD,
        grad_vanishing_threshold: float = _GRAD_VANISHING_THRESHOLD,
    ) -> None:
        self.model = model
        self.monitor = monitor
        self.sample_interval = max(1, int(sample_interval))
        self.track_gradients = track_gradients
        self.track_activations = track_activations
        self.track_parameters = track_parameters
        self.max_layers_tracked = max(1, int(max_layers_tracked))
        self.grad_explosion_threshold = float(grad_explosion_threshold)
        self.grad_vanishing_threshold = float(grad_vanishing_threshold)

        self._step_counter: int = 0
        self._hooks: list[Any] = []
        self._grad_norms: dict[str, float] = {}
        self._activation_stats: dict[str, dict[str, float]] = defaultdict(dict)
        self._param_snapshots: dict[str, torch.Tensor] = {}
        self._last_sample_ts: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attach(self) -> None:
        """Instrument the model with hooks."""
        layer_count = 0
        for name, module in self.model.named_modules():
            if layer_count >= self.max_layers_tracked:
                break
            # Skip container modules (Sequential, ModuleList, …)
            if len(list(module.children())) > 0:
                continue

            if self.track_activations:
                h = module.register_forward_hook(self._make_forward_hook(name))
                self._hooks.append(h)

            if self.track_gradients and hasattr(module, "weight") and module.weight is not None:
                h = module.weight.register_hook(self._make_grad_hook(name))
                self._hooks.append(h)

            layer_count += 1

        if self.track_parameters:
            self._snapshot_parameters()

    def detach(self) -> None:
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def check_parameter_drift(self) -> dict[str, float]:
        """
        Compute L2 drift for each tracked parameter since the last snapshot.

        Returns a mapping ``{param_name: drift_norm}``.  Also emits a
        ``training.parameter_drift`` event when called.
        """
        if not self.track_parameters:
            return {}

        drift_metrics: dict[str, float] = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self._param_snapshots:
                    old = self._param_snapshots[name]
                    drift = float((param.detach() - old).norm())
                    drift_metrics[name] = drift
                    self._param_snapshots[name] = param.detach().clone()

        if drift_metrics:
            self._emit(
                "training.parameter_drift",
                {"drift_metrics": drift_metrics, "step": self._step_counter},
            )
        return drift_metrics

    def get_layer_health_report(self) -> dict[str, Any]:
        """Return a snapshot of current layer health metrics."""
        return {
            "gradient_norms": dict(self._grad_norms),
            "activation_stats": dict(self._activation_stats),
            "total_layers_tracked": len(self._hooks),
            "step_count": self._step_counter,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_sample(self) -> bool:
        self._step_counter += 1
        now = time.monotonic()
        if (
            self._step_counter % self.sample_interval == 0
            and (now - self._last_sample_ts) >= _MIN_SAMPLE_GAP_S
        ):
            self._last_sample_ts = now
            return True
        return False

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        event = Event(
            type=event_type,
            payload=payload,
            timestamp=datetime.now(timezone.utc),
        )
        bus = getattr(self.monitor, "event_bus", None)
        if bus is None:
            return
        # Prefer non-blocking emit when available
        emit_fn = getattr(bus, "emit_async", None) or bus.emit
        try:
            emit_fn(event)
        except Exception:
            pass

    def _make_forward_hook(self, layer_name: str):
        def hook(module: nn.Module, inp: Any, output: Any) -> None:
            if not self._should_sample():
                return
            if not isinstance(output, torch.Tensor):
                return
            with torch.no_grad():
                stats = {
                    "mean": float(output.mean()),
                    "std": float(output.std()),
                    "max": float(output.max()),
                    "min": float(output.min()),
                    "has_nan": bool(torch.isnan(output).any()),
                    "has_inf": bool(torch.isinf(output).any()),
                }
            self._activation_stats[layer_name] = stats
            self._emit(
                "training.activation",
                {"layer": layer_name, "stats": stats, "step": self._step_counter},
            )

        return hook

    def _make_grad_hook(self, layer_name: str):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            if not self._should_sample():
                return grad
            with torch.no_grad():
                grad_norm = float(grad.norm())
            self._grad_norms[layer_name] = grad_norm

            if grad_norm > self.grad_explosion_threshold:
                self._emit(
                    "training.gradient_explosion",
                    {
                        "layer": layer_name,
                        "grad_norm": grad_norm,
                        "severity": "critical" if grad_norm > 1000.0 else "warning",
                        "step": self._step_counter,
                    },
                )
            elif grad_norm < self.grad_vanishing_threshold:
                self._emit(
                    "training.gradient_vanishing",
                    {
                        "layer": layer_name,
                        "grad_norm": grad_norm,
                        "step": self._step_counter,
                    },
                )
            return grad

        return hook

    def _snapshot_parameters(self) -> None:
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self._param_snapshots[name] = param.detach().clone()


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def monitor_pytorch_training(
    model: nn.Module,
    monitor: Any,
    **kwargs: Any,
) -> TrainSightHook:
    """
    Attach a :class:`TrainSightHook` to *model* and return it.

    Example::

        hook = monitor_pytorch_training(model, dashboard, sample_interval=20)
        # … training loop …
        hook.detach()
    """
    hook = TrainSightHook(model, monitor, **kwargs)
    hook.attach()
    return hook
