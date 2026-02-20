"""
trainsight.optimization.batch_size
=====================================
Automatic batch size recommendation based on live VRAM usage patterns.

Subscribes to ``gpu.stats`` and ``training.oom_risk`` events and uses
rolling statistics to suggest a safe, efficient batch size.

Usage::

    from trainsight import Dashboard
    from trainsight.optimization import BatchSizeOptimizer

    dashboard = Dashboard()
    optimizer = BatchSizeOptimizer(
        monitor=dashboard,
        current_batch_size=32,
        total_vram_gb=24.0,
    )

    # After some training steps:
    rec = optimizer.get_recommendation()
    print(f"Suggested batch size: {rec.suggested_batch_size}")
    print(f"Confidence: {rec.confidence:.0%}")
    print(f"Reason: {rec.reasoning}")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from trainsight.core.event import Event

logger = logging.getLogger("trainsight.optimization.batch_size")

_MIN_SAMPLES = 20
_SAFE_VRAM_THRESHOLD = 85.0  # % – default upper bound
_LOW_VRAM_THRESHOLD = 60.0   # % – below this we can increase batch size


@dataclass
class BatchSizeRecommendation:
    suggested_batch_size: int
    confidence: float          # 0.0 – 1.0
    reasoning: str
    expected_vram_usage: float  # %
    current_avg_vram: float
    current_max_vram: float


class BatchSizeOptimizer:
    """
    Recommend an optimal batch size from live VRAM telemetry.

    Parameters
    ----------
    monitor:
        A TrainSight ``Dashboard`` instance.
    current_batch_size:
        The batch size currently in use.
    total_vram_gb:
        Total GPU VRAM in GB (used for absolute headroom calculation).
    safety_margin:
        Fraction of VRAM to keep free as a safety buffer (default 0.15 = 15 %).
    """

    def __init__(
        self,
        monitor: Any,
        current_batch_size: int,
        total_vram_gb: float = 0.0,
        safety_margin: float = 0.15,
    ) -> None:
        self.monitor = monitor
        self.current_batch_size = max(1, int(current_batch_size))
        self.total_vram_gb = float(total_vram_gb)
        self.safety_margin = max(0.0, min(0.5, float(safety_margin)))

        self._vram_samples: list[float] = []
        self._oom_risk_samples: list[float] = []

        bus = getattr(monitor, "event_bus", None)
        if bus:
            bus.subscribe("gpu.stats", self._collect_vram)
            bus.subscribe("training.oom_risk", self._collect_oom_risk)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_recommendation(self) -> BatchSizeRecommendation:
        """
        Return a batch size recommendation based on collected samples.

        Requires at least ``_MIN_SAMPLES`` GPU events before producing a
        meaningful recommendation.
        """
        if len(self._vram_samples) < _MIN_SAMPLES:
            return BatchSizeRecommendation(
                suggested_batch_size=self.current_batch_size,
                confidence=0.0,
                reasoning=f"Insufficient data – need {_MIN_SAMPLES}+ samples",
                expected_vram_usage=0.0,
                current_avg_vram=0.0,
                current_max_vram=0.0,
            )

        arr = np.array(self._vram_samples)
        avg_vram = float(arr.mean())
        max_vram = float(arr.max())
        std_vram = float(arr.std())
        avg_oom_risk = float(np.mean(self._oom_risk_samples)) if self._oom_risk_samples else 0.0

        safe_threshold = (1.0 - self.safety_margin) * 100.0

        if max_vram > safe_threshold or avg_oom_risk > 0.3:
            # Reduce batch size
            reduction = max_vram / safe_threshold
            suggested = max(1, int(self.current_batch_size / reduction))
            reasoning = (
                f"High VRAM usage (max {max_vram:.1f}%) or OOM risk "
                f"({avg_oom_risk:.2f}) – reducing batch size"
            )
        elif avg_vram < _LOW_VRAM_THRESHOLD and avg_oom_risk < 0.1:
            # Increase batch size conservatively
            headroom = (safe_threshold - avg_vram) / max(avg_vram, 1.0)
            factor = 1.0 + headroom * 0.5
            suggested = int(self.current_batch_size * factor)
            reasoning = (
                f"Low VRAM usage ({avg_vram:.1f}%) – "
                f"batch size can be increased safely"
            )
        else:
            suggested = self.current_batch_size
            reasoning = f"Current batch size is near-optimal (avg VRAM {avg_vram:.1f}%)"

        # Confidence: lower variance → higher confidence
        confidence = float(max(0.0, min(1.0, 1.0 - std_vram / 100.0)))

        # Estimate VRAM for suggested batch size
        vram_per_sample = avg_vram / self.current_batch_size
        expected_vram = vram_per_sample * suggested

        rec = BatchSizeRecommendation(
            suggested_batch_size=suggested,
            confidence=confidence,
            reasoning=reasoning,
            expected_vram_usage=expected_vram,
            current_avg_vram=avg_vram,
            current_max_vram=max_vram,
        )

        # Emit recommendation event
        self._emit_recommendation(rec)
        return rec

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _collect_vram(self, event: Any) -> None:
        vram = float(event.payload.get("vram", 0))
        self._vram_samples.append(vram)
        if len(self._vram_samples) > 500:
            self._vram_samples.pop(0)

    def _collect_oom_risk(self, event: Any) -> None:
        risk = float(event.payload.get("probability", 0))
        self._oom_risk_samples.append(risk)
        if len(self._oom_risk_samples) > 500:
            self._oom_risk_samples.pop(0)

    def _emit_recommendation(self, rec: BatchSizeRecommendation) -> None:
        event = Event(
            type="optimization.batch_size_recommendation",
            payload={
                "suggested_batch_size": rec.suggested_batch_size,
                "current_batch_size": self.current_batch_size,
                "confidence": rec.confidence,
                "reasoning": rec.reasoning,
                "expected_vram_usage": rec.expected_vram_usage,
                "current_avg_vram": rec.current_avg_vram,
                "current_max_vram": rec.current_max_vram,
            },
            timestamp=datetime.now(timezone.utc),
        )
        bus = getattr(self.monitor, "event_bus", None)
        if bus:
            emit_fn = getattr(bus, "emit_async", None) or bus.emit
            try:
                emit_fn(event)
            except Exception:
                pass
