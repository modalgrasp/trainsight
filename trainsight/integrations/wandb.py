"""
trainsight.integrations.wandb
================================
Weights & Biases integration for TrainSight.

Subscribes to TrainSight events and logs them to the active W&B run.

Metrics logged:
* ``trainsight/gpu_util``           – GPU utilisation %
* ``trainsight/vram_percent``       – VRAM usage %
* ``trainsight/gpu_temp``           – GPU temperature °C
* ``trainsight/gpu_power``          – GPU power draw W
* ``trainsight/anomaly_detected``   – 1 when anomaly fires
* ``trainsight/anomaly_z_score``    – z-score of the anomaly
* ``trainsight/gradient_explosion`` – 1 when explosion fires
* ``trainsight/grad_norm``          – gradient norm at explosion

Usage::

    import wandb
    from trainsight import Dashboard
    from trainsight.integrations.wandb import enable_wandb

    wandb.init(project="my-project")
    dashboard = Dashboard()
    bridge = enable_wandb(dashboard)

    # … training …

    bridge.disable()
    wandb.finish()
"""
from __future__ import annotations

import logging
from typing import Any

try:
    import wandb
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "wandb is required: pip install wandb"
    ) from exc

logger = logging.getLogger("trainsight.wandb")


class WandBTrainSight:
    """
    Bridge between TrainSight events and Weights & Biases logging.

    Parameters
    ----------
    monitor:
        A TrainSight ``Dashboard`` instance.
    log_interval:
        Log to W&B every *N* GPU events.
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
        """Start forwarding TrainSight events to W&B."""
        self._enabled = True
        bus = getattr(self.monitor, "event_bus", None)
        if bus is None:
            logger.warning("monitor has no event_bus – W&B logging disabled")
            return
        bus.subscribe("gpu.stats", self._log_gpu_stats)
        bus.subscribe("training.anomaly", self._log_anomaly)
        bus.subscribe("training.gradient_explosion", self._log_gradient_explosion)
        bus.subscribe("training.gradient_vanishing", self._log_gradient_vanishing)
        bus.subscribe("training.loss_plateau", self._log_loss_event)
        bus.subscribe("training.loss_divergence", self._log_loss_event)
        bus.subscribe("training.overfitting_detected", self._log_loss_event)
        bus.subscribe("training.amp_instability", self._log_amp_instability)
        bus.subscribe("training.validation_drift", self._log_validation_drift)

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
            wandb.log(
                {
                    "trainsight/gpu_util": float(p.get("gpu_util", 0)),
                    "trainsight/vram_percent": float(p.get("vram", 0)),
                    "trainsight/gpu_temp": float(p.get("temp", 0)),
                    "trainsight/gpu_power": float(p.get("power", 0)),
                },
                step=self._step_counter,
            )
        except Exception as exc:
            logger.debug("W&B log failed: %s", exc)

    def _log_anomaly(self, event: Any) -> None:
        if not self._enabled:
            return
        try:
            wandb.log(
                {
                    "trainsight/anomaly_detected": 1,
                    "trainsight/anomaly_z_score": float(event.payload.get("z_score", 0)),
                },
                step=self._step_counter,
            )
        except Exception as exc:
            logger.debug("W&B anomaly log failed: %s", exc)

    def _log_gradient_explosion(self, event: Any) -> None:
        if not self._enabled:
            return
        try:
            wandb.log(
                {
                    "trainsight/gradient_explosion": 1,
                    "trainsight/grad_norm": float(event.payload.get("grad_norm", 0)),
                    "trainsight/layer": event.payload.get("layer", "unknown"),
                },
                step=self._step_counter,
            )
        except Exception as exc:
            logger.debug("W&B gradient_explosion log failed: %s", exc)

    def _log_gradient_vanishing(self, event: Any) -> None:
        if not self._enabled:
            return
        try:
            wandb.log(
                {
                    "trainsight/gradient_vanishing": 1,
                    "trainsight/grad_norm": float(event.payload.get("grad_norm", 0)),
                },
                step=self._step_counter,
            )
        except Exception as exc:
            logger.debug("W&B gradient_vanishing log failed: %s", exc)

    def _log_loss_event(self, event: Any) -> None:
        if not self._enabled:
            return
        try:
            key = f"trainsight/{event.type.split('.')[-1]}"
            wandb.log({key: 1}, step=self._step_counter)
        except Exception as exc:
            logger.debug("W&B loss event log failed: %s", exc)

    def _log_amp_instability(self, event: Any) -> None:
        if not self._enabled:
            return
        try:
            wandb.log(
                {
                    "trainsight/amp_instability": 1,
                    "trainsight/amp_cv": float(
                        event.payload.get("coefficient_of_variation", 0)
                    ),
                },
                step=self._step_counter,
            )
        except Exception as exc:
            logger.debug("W&B AMP instability log failed: %s", exc)

    def _log_validation_drift(self, event: Any) -> None:
        if not self._enabled:
            return
        try:
            wandb.log(
                {"trainsight/validation_drift": 1},
                step=self._step_counter,
            )
        except Exception as exc:
            logger.debug("W&B validation_drift log failed: %s", exc)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def enable_wandb(monitor: Any, **kwargs: Any) -> WandBTrainSight:
    """
    Enable W&B logging for a TrainSight monitor and return the bridge.

    Example::

        wandb.init(project="my-project")
        bridge = enable_wandb(dashboard)
        # … training …
        bridge.disable()
        wandb.finish()
    """
    bridge = WandBTrainSight(monitor, **kwargs)
    bridge.enable()
    return bridge
