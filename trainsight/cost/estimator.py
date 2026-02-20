"""
trainsight.cost.estimator
===========================
Cloud GPU cost estimation for training sessions.

Supports AWS, GCP, Azure, and Lambda Labs pricing.

Usage::

    from trainsight.cost import CloudCostEstimator, CloudProvider, GPUType

    estimator = CloudCostEstimator(
        provider=CloudProvider.AWS,
        gpu_type=GPUType.A100,
        num_gpus=8,
    )

    estimate = estimator.estimate_session_cost(
        duration_hours=12.5,
        avg_utilization=67.3,
    )

    print(f"Total cost:  ${estimate.total_cost_usd:.2f}")
    print(f"Wasted cost: ${estimate.wasted_cost_usd:.2f}")
    for rec in estimate.recommendations:
        print(" •", rec)

Real-time tracking::

    estimator.attach_realtime_tracking(dashboard)
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from trainsight.core.event import Event

logger = logging.getLogger("trainsight.cost")


class CloudProvider(str, Enum):
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    LAMBDA_LABS = "lambda"


class GPUType(str, Enum):
    V100 = "v100"
    A100 = "a100"
    A100_80GB = "a100_80gb"
    H100 = "h100"
    T4 = "t4"
    RTX_4090 = "rtx4090"
    A10G = "a10g"


@dataclass
class CostEstimate:
    total_cost_usd: float
    wasted_cost_usd: float
    efficiency_percent: float
    recommendations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pricing table (USD / GPU / hour, approximate 2026 on-demand rates)
# ---------------------------------------------------------------------------
_PRICING: dict[CloudProvider, dict[GPUType, float]] = {
    CloudProvider.AWS: {
        GPUType.T4: 0.526,
        GPUType.A10G: 1.006,
        GPUType.V100: 3.06,
        GPUType.A100: 4.10,
        GPUType.A100_80GB: 5.12,
        GPUType.H100: 8.00,
    },
    CloudProvider.GCP: {
        GPUType.T4: 0.35,
        GPUType.V100: 2.48,
        GPUType.A100: 3.67,
        GPUType.A100_80GB: 4.89,
        GPUType.H100: 7.50,
    },
    CloudProvider.AZURE: {
        GPUType.T4: 0.526,
        GPUType.V100: 3.06,
        GPUType.A100: 3.67,
        GPUType.A100_80GB: 4.60,
        GPUType.H100: 7.80,
    },
    CloudProvider.LAMBDA_LABS: {
        GPUType.RTX_4090: 0.50,
        GPUType.A100: 1.10,
        GPUType.A100_80GB: 1.29,
        GPUType.H100: 2.49,
    },
}


class CloudCostEstimator:
    """
    Estimate cloud GPU training costs.

    Parameters
    ----------
    provider:
        Cloud provider enum value.
    gpu_type:
        GPU model enum value.
    num_gpus:
        Number of GPUs in the training job.
    """

    def __init__(
        self,
        provider: CloudProvider,
        gpu_type: GPUType,
        num_gpus: int = 1,
    ) -> None:
        self.provider = provider
        self.gpu_type = gpu_type
        self.num_gpus = max(1, int(num_gpus))

        provider_pricing = _PRICING.get(provider, {})
        self.hourly_rate: float = provider_pricing.get(gpu_type, 0.0)
        if self.hourly_rate == 0.0:
            logger.warning(
                "No pricing data for %s / %s – cost estimates will be $0",
                provider.value,
                gpu_type.value,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_session_cost(
        self,
        duration_hours: float,
        avg_utilization: float,
    ) -> CostEstimate:
        """
        Estimate cost for a completed or in-progress training session.

        Parameters
        ----------
        duration_hours:
            Total elapsed training time in hours.
        avg_utilization:
            Average GPU utilisation percentage (0–100).
        """
        duration_hours = max(0.0, float(duration_hours))
        avg_utilization = max(0.0, min(100.0, float(avg_utilization)))

        total_cost = self.hourly_rate * self.num_gpus * duration_hours
        efficiency = avg_utilization / 100.0
        wasted_cost = total_cost * (1.0 - efficiency)

        recommendations: list[str] = []

        if avg_utilization < 50:
            recommendations.append(
                f"GPU utilisation is low ({avg_utilization:.1f}%). "
                "Consider increasing batch size or using a smaller GPU instance."
            )

        if avg_utilization < 30:
            cheaper = self._find_cheaper_gpus()
            if cheaper:
                savings = self._calculate_savings(cheaper[0], duration_hours)
                recommendations.append(
                    f"Switching to {cheaper[0].value} could save "
                    f"~${savings:.2f} for this session."
                )

        if wasted_cost > 20:
            recommendations.append(
                f"${wasted_cost:.2f} wasted due to underutilisation. "
                "Profile your data pipeline for bottlenecks."
            )

        return CostEstimate(
            total_cost_usd=total_cost,
            wasted_cost_usd=wasted_cost,
            efficiency_percent=avg_utilization,
            recommendations=recommendations,
        )

    def attach_realtime_tracking(self, monitor: Any, report_interval: int = 100) -> None:
        """
        Subscribe to GPU events and emit ``cost.estimate`` events periodically.

        Parameters
        ----------
        monitor:
            A TrainSight ``Dashboard`` instance.
        report_interval:
            Emit a cost estimate every *N* GPU events.
        """
        start_time = time.time()
        total_util = 0.0
        sample_count = 0

        def _tracker(event: Any) -> None:
            nonlocal total_util, sample_count
            total_util += float(event.payload.get("gpu_util", 0))
            sample_count += 1

            if sample_count % report_interval != 0:
                return

            elapsed_hours = (time.time() - start_time) / 3600.0
            avg_util = total_util / sample_count
            estimate = self.estimate_session_cost(elapsed_hours, avg_util)

            cost_event = Event(
                type="cost.estimate",
                payload={
                    "elapsed_hours": elapsed_hours,
                    "total_cost_usd": estimate.total_cost_usd,
                    "wasted_cost_usd": estimate.wasted_cost_usd,
                    "efficiency_percent": estimate.efficiency_percent,
                    "recommendations": estimate.recommendations,
                    "provider": self.provider.value,
                    "gpu_type": self.gpu_type.value,
                    "num_gpus": self.num_gpus,
                },
                timestamp=datetime.now(timezone.utc),
            )
            bus = getattr(monitor, "event_bus", None)
            if bus:
                emit_fn = getattr(bus, "emit_async", None) or bus.emit
                try:
                    emit_fn(cost_event)
                except Exception:
                    pass

        bus = getattr(monitor, "event_bus", None)
        if bus:
            bus.subscribe("gpu.stats", _tracker)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_cheaper_gpus(self) -> list[GPUType]:
        provider_pricing = _PRICING.get(self.provider, {})
        return sorted(
            [g for g, r in provider_pricing.items() if r < self.hourly_rate],
            key=lambda g: provider_pricing[g],
            reverse=True,
        )

    def _calculate_savings(self, gpu_type: GPUType, duration_hours: float) -> float:
        provider_pricing = _PRICING.get(self.provider, {})
        new_rate = provider_pricing.get(gpu_type, self.hourly_rate)
        current = self.hourly_rate * self.num_gpus * duration_hours
        new_cost = new_rate * self.num_gpus * duration_hours
        return max(0.0, current - new_cost)
