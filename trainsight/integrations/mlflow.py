"""
trainsight.integrations.mlflow
================================
Automatic MLflow logging for TrainSight GPU and training metrics.

Subscribes to TrainSight events and pushes them into the active MLflow run.

Metrics logged:
* ``trainsight.gpu_util``        – GPU utilisation %
* ``trainsight.vram_percent``    – VRAM usage %
* ``trainsight.gpu_temp``        – GPU temperature °C
* ``trainsight.gpu_power``       – GPU power draw W
* ``trainsight.oom_probability`` – OOM risk score [0, 1]
* ``trainsight.efficiency``      – Training efficiency score

Usage::

    import mlflow
    from trainsight import Dashboard
    from trainsight.integrations.mlflow import enable_mlflow_logging

    dashboard = Dashboard()

    with mlflow.start_run():
        mlflow_monitor = enable_mlflow_logging(dashboard)
        # … training …
        mlflow_monitor.disable()
"""
from __future__ import annotations

import logging
from typing import Any

try:
    import mlflow
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mlflow is required: pip install mlflow"
    ) from exc

logger = logging.getLogger("trainsight.mlflow")


class MLflowTrainSight:
    """
    Bridge between TrainSight events and MLflow metric logging.

    Parameters
    ----------
    monitor:
        A TrainSight ``Dashboard`` instance.
    log_interval:
        Log to MLflow every *N* GPU events.
    """

    def __init__(self, monitor: Any, log_interval: int = 10) -> None:
        self.monitor = monitor
        self.log_interval = max(1, int(log_interval))
        self._enabled: bool = False
        self._step_counter: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enable(self) -> None:
        """Start forwarding TrainSight events to MLflow."""
        self._enabled = True
        bus = getattr(self.monitor, "event_bus", None)
        if bus is None:
            logger.warning("monitor has no event_bus – MLflow logging disabled")
            return
        bus.subscribe("gpu.stats", self._log_gpu_stats)
        bus.subscribe("training.oom_risk", self._log_oom_risk)
        bus.subscribe("training.efficiency", self._log_efficiency)
        bus.subscribe("training.gradient_explosion", self._log_gradient_explosion)
        bus.subscribe("training.loss_plateau", self._log_loss_event)
        bus.subscribe("training.loss_divergence", self._log_loss_event)
        bus.subscribe("training.overfitting_detected", self._log_loss_event)

    def disable(self) -> None:
        """Stop forwarding events."""
        self._enabled = False

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _log_gpu_stats(self, event: Any) -> None:
        if not self._enabled:
            return
        self._step_counter += 1
        if self._step_counter % self.log_interval != 0:
            return
        p = event.payload
        try:
            mlflow.log_metrics(
                {
                    "trainsight.gpu_util": float(p.get("gpu_util", 0)),
                    "trainsight.vram_percent": float(p.get("vram", 0)),
                    "trainsight.gpu_temp": float(p.get("temp", 0)),
                    "trainsight.gpu_power": float(p.get("power", 0)),
                },
                step=self._step_counter,
            )
        except Exception as exc:
            logger.debug("MLflow log_metrics failed: %s", exc)

    def _log_oom_risk(self, event: Any) -> None:
        if not self._enabled:
            return
        try:
            mlflow.log_metric(
                "trainsight.oom_probability",
                float(event.payload.get("probability", 0)),
                step=self._step_counter,
            )
        except Exception as exc:
            logger.debug("MLflow oom_risk log failed: %s", exc)

    def _log_efficiency(self, event: Any) -> None:
        if not self._enabled:
            return
        try:
            mlflow.log_metric(
                "trainsight.efficiency",
                float(event.payload.get("score", 0)),
                step=self._step_counter,
            )
        except Exception as exc:
            logger.debug("MLflow efficiency log failed: %s", exc)

    def _log_gradient_explosion(self, event: Any) -> None:
        if not self._enabled:
            return
        try:
            mlflow.log_metric(
                "trainsight.gradient_explosion",
                float(event.payload.get("grad_norm", 0)),
                step=self._step_counter,
            )
        except Exception as exc:
            logger.debug("MLflow gradient_explosion log failed: %s", exc)

    def _log_loss_event(self, event: Any) -> None:
        if not self._enabled:
            return
        try:
            key = f"trainsight.{event.type.replace('.', '_')}"
            mlflow.set_tag(key, event.payload.get("message", event.type))
        except Exception as exc:
            logger.debug("MLflow tag failed: %s", exc)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def enable_mlflow_logging(monitor: Any, **kwargs: Any) -> MLflowTrainSight:
    """
    Enable MLflow logging for a TrainSight monitor and return the bridge.

    Example::

        with mlflow.start_run():
            bridge = enable_mlflow_logging(dashboard)
            # … training …
            bridge.disable()
    """
    bridge = MLflowTrainSight(monitor, **kwargs)
    bridge.enable()
    return bridge
