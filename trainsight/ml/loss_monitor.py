"""
trainsight.ml.loss_monitor
============================
Training loss trend analysis using rolling linear regression.

Detects:
* **Plateau**     – loss slope near zero for ``window_size`` steps
* **Divergence**  – loss slope significantly positive
* **Overfitting** – validation loss > training loss by more than ``overfit_gap_pct``

Emits events:
* ``training.loss_plateau``          – loss has stopped decreasing
* ``training.loss_divergence``       – loss is increasing
* ``training.overfitting_detected``  – train/val gap is widening

Usage::

    from trainsight.ml.loss_monitor import LossTrendMonitor

    monitor = LossTrendMonitor(dashboard)

    for step, (train_loss, val_loss) in enumerate(training_loop()):
        monitor.update(step, train_loss, val_loss)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from trainsight.core.event import Event

logger = logging.getLogger("trainsight.ml.loss_monitor")

_ANALYSIS_INTERVAL = 10  # analyse every N updates


class LossTrendMonitor:
    """
    Monitor training loss trends and emit diagnostic events.

    Parameters
    ----------
    monitor:
        A TrainSight ``Dashboard`` instance.
    window_size:
        Number of recent steps used for regression analysis.
    plateau_threshold:
        Absolute slope below which a plateau is declared.
    divergence_threshold:
        Positive slope above which divergence is declared.
    overfit_gap_pct:
        Percentage by which val_loss must exceed train_loss to trigger
        an overfitting warning.
    """

    def __init__(
        self,
        monitor: Any,
        window_size: int = 50,
        plateau_threshold: float = 0.001,
        divergence_threshold: float = 0.01,
        overfit_gap_pct: float = 20.0,
    ) -> None:
        self.monitor = monitor
        self.window_size = max(10, int(window_size))
        self.plateau_threshold = float(plateau_threshold)
        self.divergence_threshold = float(divergence_threshold)
        self.overfit_gap_pct = float(overfit_gap_pct)

        self._train_losses: list[float] = []
        self._val_losses: list[float] = []
        self._steps: list[int] = []
        self._update_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        step: int,
        train_loss: float,
        val_loss: Optional[float] = None,
    ) -> None:
        """
        Record a new loss observation.

        Parameters
        ----------
        step:
            Global training step.
        train_loss:
            Training loss value.
        val_loss:
            Optional validation loss value.
        """
        self._steps.append(int(step))
        self._train_losses.append(float(train_loss))
        if val_loss is not None:
            self._val_losses.append(float(val_loss))

        # Trim to 2× window to keep memory bounded
        cap = self.window_size * 2
        if len(self._train_losses) > cap:
            self._steps = self._steps[-cap:]
            self._train_losses = self._train_losses[-cap:]
        if len(self._val_losses) > cap:
            self._val_losses = self._val_losses[-cap:]

        self._update_count += 1
        if (
            len(self._train_losses) >= self.window_size
            and self._update_count % _ANALYSIS_INTERVAL == 0
        ):
            self._analyse()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _analyse(self) -> None:
        recent_steps = np.array(self._steps[-self.window_size:], dtype=float)
        recent_losses = np.array(self._train_losses[-self.window_size:])

        # Normalise steps to [0, 1] for numerical stability
        step_range = recent_steps[-1] - recent_steps[0]
        if step_range > 0:
            x = (recent_steps - recent_steps[0]) / step_range
        else:
            x = np.linspace(0, 1, len(recent_steps))

        coeffs = np.polyfit(x, recent_losses, 1)
        slope = float(coeffs[0])

        if abs(slope) < self.plateau_threshold:
            self._emit(
                "training.loss_plateau",
                {
                    "slope": slope,
                    "message": "Training loss has plateaued",
                    "severity": "warning",
                    "step": self._steps[-1],
                },
            )
        elif slope > self.divergence_threshold:
            self._emit(
                "training.loss_divergence",
                {
                    "slope": slope,
                    "message": "Training loss is increasing – possible divergence",
                    "severity": "critical",
                    "step": self._steps[-1],
                },
            )

        if len(self._val_losses) >= 10:
            self._check_overfitting()

    def _check_overfitting(self) -> None:
        recent_train = float(np.mean(self._train_losses[-10:]))
        recent_val = float(np.mean(self._val_losses[-10:]))
        if recent_train <= 0:
            return
        gap_pct = ((recent_val - recent_train) / recent_train) * 100.0
        if gap_pct > self.overfit_gap_pct:
            self._emit(
                "training.overfitting_detected",
                {
                    "train_loss": recent_train,
                    "val_loss": recent_val,
                    "gap_percent": gap_pct,
                    "message": (
                        f"Validation loss is {gap_pct:.1f}% higher than training loss"
                    ),
                    "severity": "warning",
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
