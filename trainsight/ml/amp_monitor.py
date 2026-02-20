"""
trainsight.ml.amp_monitor
===========================
Mixed Precision (AMP) training stability monitor.

Tracks:
* Loss variance (coefficient of variation)
* Gradient scaler overflow frequency
* NaN / Inf occurrence counts

Emits events:
* ``training.amp_instability`` – when AMP training appears unstable

Usage::

    from trainsight.ml.amp_monitor import AMPStabilityMonitor

    amp_monitor = AMPStabilityMonitor(dashboard)

    # Inside training loop (PyTorch AMP example):
    with torch.cuda.amp.autocast():
        loss = model(batch)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    amp_monitor.update(
        loss=loss.item(),
        scaler_scale=scaler.get_scale(),
        has_overflow=scaler._found_inf_per_device,
    )
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from trainsight.core.event import Event

logger = logging.getLogger("trainsight.ml.amp_monitor")

_ANALYSIS_INTERVAL = 20
_CV_INSTABILITY_THRESHOLD = 0.5   # coefficient of variation
_OVERFLOW_RATE_THRESHOLD = 0.1    # 10 % of steps with overflow


class AMPStabilityMonitor:
    """
    Monitor mixed precision training stability.

    Parameters
    ----------
    monitor:
        A TrainSight ``Dashboard`` instance.
    window_size:
        Number of recent steps used for stability analysis.
    cv_threshold:
        Coefficient of variation above which instability is flagged.
    overflow_rate_threshold:
        Fraction of steps with gradient overflow above which instability
        is flagged.
    """

    def __init__(
        self,
        monitor: Any,
        window_size: int = 100,
        cv_threshold: float = _CV_INSTABILITY_THRESHOLD,
        overflow_rate_threshold: float = _OVERFLOW_RATE_THRESHOLD,
    ) -> None:
        self.monitor = monitor
        self.window_size = max(10, int(window_size))
        self.cv_threshold = float(cv_threshold)
        self.overflow_rate_threshold = float(overflow_rate_threshold)

        self._loss_history: list[float] = []
        self._scaler_history: list[float] = []
        self._overflow_flags: list[bool] = []
        self._nan_count: int = 0
        self._update_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        loss: float,
        scaler_scale: Optional[float] = None,
        has_nan: bool = False,
        has_overflow: bool = False,
    ) -> None:
        """
        Record a training step.

        Parameters
        ----------
        loss:
            Scalar loss value for this step.
        scaler_scale:
            Current gradient scaler scale factor (from ``scaler.get_scale()``).
        has_nan:
            Whether NaN values were detected in this step.
        has_overflow:
            Whether gradient overflow occurred (scaler found inf).
        """
        self._loss_history.append(float(loss))
        self._overflow_flags.append(bool(has_overflow))
        if scaler_scale is not None:
            self._scaler_history.append(float(scaler_scale))
        if has_nan:
            self._nan_count += 1

        # Trim to window
        if len(self._loss_history) > self.window_size:
            self._loss_history.pop(0)
            self._overflow_flags.pop(0)
        if len(self._scaler_history) > self.window_size:
            self._scaler_history.pop(0)

        self._update_count += 1
        if (
            len(self._loss_history) >= 20
            and self._update_count % _ANALYSIS_INTERVAL == 0
        ):
            self._analyse()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _analyse(self) -> None:
        arr = np.array(self._loss_history)
        mean = float(arr.mean())
        std = float(arr.std())
        cv = std / (abs(mean) + 1e-8)

        overflow_rate = sum(self._overflow_flags) / len(self._overflow_flags)

        unstable = cv > self.cv_threshold or overflow_rate > self.overflow_rate_threshold

        if unstable:
            recommendation = (
                "Consider reducing learning rate, using fp32, "
                "or adjusting loss scaling strategy."
            )
            if overflow_rate > self.overflow_rate_threshold:
                recommendation = (
                    f"High overflow rate ({overflow_rate:.0%}). "
                    "Try lowering the initial loss scale or using dynamic scaling."
                )

            self._emit(
                "training.amp_instability",
                {
                    "coefficient_of_variation": cv,
                    "overflow_rate": overflow_rate,
                    "nan_count": self._nan_count,
                    "loss_mean": mean,
                    "loss_std": std,
                    "message": "Mixed precision training appears unstable",
                    "severity": "warning",
                    "recommendation": recommendation,
                },
            )

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
