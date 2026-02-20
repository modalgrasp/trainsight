"""
trainsight.integrations.ray_train
====================================
Ray Train integration for TrainSight.

Emits events:
* ``ray.worker_metrics`` – per-worker training metrics

Usage::

    import ray
    from ray import train
    from ray.train.torch import TorchTrainer
    from trainsight import Dashboard
    from trainsight.integrations.ray_train import RayTrainSight

    dashboard = Dashboard()

    def train_func(config):
        monitor = RayTrainSight(dashboard)
        for epoch in range(config["epochs"]):
            metrics = run_epoch()
            monitor.report(metrics)

    trainer = TorchTrainer(train_func, ...)
    trainer.fit()
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

try:
    import ray  # type: ignore[import-unresolved]
    from ray import train as ray_train  # type: ignore[import-unresolved]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ray is required: pip install 'ray[train]'"
    ) from exc

from trainsight.core.event import Event

logger = logging.getLogger("trainsight.ray_train")


class RayTrainSight:
    """
    Emit TrainSight events from inside a Ray Train worker function.

    Parameters
    ----------
    monitor:
        A TrainSight ``Dashboard`` instance (must be accessible from the
        worker – typically passed via ``config`` or a shared actor).
    """

    def __init__(self, monitor: Any) -> None:
        self.monitor = monitor
        self._worker_id: str = ""
        self._node_id: str = ""

        try:
            ctx = ray.get_runtime_context()
            self._worker_id = str(ctx.get_worker_id())
            self._node_id = str(ctx.get_node_id())
        except Exception as exc:
            logger.debug("Could not get Ray runtime context: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def report(self, metrics: dict[str, Any]) -> None:
        """
        Report metrics from a Ray worker.

        Emits a ``ray.worker_metrics`` event and also calls
        ``ray.train.report`` so that Ray's own tracking still works.
        """
        payload: dict[str, Any] = {
            "worker_id": self._worker_id,
            "node_id": self._node_id,
            "metrics": {k: _safe_float(v) for k, v in metrics.items()},
        }
        self._emit("ray.worker_metrics", payload)

        # Forward to Ray Train
        try:
            ray_train.report(metrics)
        except Exception as exc:
            logger.debug("ray.train.report failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        event = Event(
            type=event_type,
            payload=payload,
            timestamp=datetime.now(timezone.utc),
        )
        bus = getattr(self.monitor, "event_bus", None)
        if bus is None:
            return
        emit_fn = getattr(bus, "emit_async", None) or bus.emit
        try:
            emit_fn(event)
        except Exception:
            pass


def _safe_float(value: Any) -> Any:
    try:
        return float(value)
    except Exception:
        return value
