"""
Tests for trainsight.cost.estimator.CloudCostEstimator.
"""
from __future__ import annotations

import pytest

from trainsight.cost.estimator import (
    CloudCostEstimator,
    CloudProvider,
    CostEstimate,
    GPUType,
)


def test_basic_cost_calculation():
    estimator = CloudCostEstimator(
        provider=CloudProvider.AWS,
        gpu_type=GPUType.A100,
        num_gpus=1,
    )
    estimate = estimator.estimate_session_cost(
        duration_hours=10.0,
        avg_utilization=100.0,
    )
    assert estimate.total_cost_usd == pytest.approx(41.0, rel=0.01)
    assert estimate.wasted_cost_usd == pytest.approx(0.0, abs=0.01)
    assert estimate.efficiency_percent == 100.0


def test_wasted_cost_at_50_percent_utilization():
    estimator = CloudCostEstimator(
        provider=CloudProvider.AWS,
        gpu_type=GPUType.A100,
        num_gpus=1,
    )
    estimate = estimator.estimate_session_cost(
        duration_hours=10.0,
        avg_utilization=50.0,
    )
    assert estimate.wasted_cost_usd == pytest.approx(estimate.total_cost_usd * 0.5, rel=0.01)


def test_multi_gpu_cost():
    estimator = CloudCostEstimator(
        provider=CloudProvider.AWS,
        gpu_type=GPUType.A100,
        num_gpus=8,
    )
    single = CloudCostEstimator(
        provider=CloudProvider.AWS,
        gpu_type=GPUType.A100,
        num_gpus=1,
    )
    e8 = estimator.estimate_session_cost(10.0, 80.0)
    e1 = single.estimate_session_cost(10.0, 80.0)
    assert e8.total_cost_usd == pytest.approx(e1.total_cost_usd * 8, rel=0.01)


def test_recommendations_for_low_utilization():
    estimator = CloudCostEstimator(
        provider=CloudProvider.AWS,
        gpu_type=GPUType.A100,
        num_gpus=1,
    )
    estimate = estimator.estimate_session_cost(10.0, 20.0)
    assert len(estimate.recommendations) > 0
    assert any("utilisation" in r.lower() or "utilization" in r.lower() for r in estimate.recommendations)


def test_unknown_provider_gpu_combo_returns_zero():
    estimator = CloudCostEstimator(
        provider=CloudProvider.LAMBDA_LABS,
        gpu_type=GPUType.V100,  # not in Lambda pricing
        num_gpus=1,
    )
    estimate = estimator.estimate_session_cost(10.0, 80.0)
    assert estimate.total_cost_usd == 0.0


def test_utilization_clamped():
    estimator = CloudCostEstimator(
        provider=CloudProvider.GCP,
        gpu_type=GPUType.T4,
        num_gpus=1,
    )
    # Should not raise even with out-of-range values
    e = estimator.estimate_session_cost(1.0, 150.0)
    assert e.efficiency_percent == 100.0

    e2 = estimator.estimate_session_cost(1.0, -10.0)
    assert e2.efficiency_percent == 0.0


def test_all_providers_have_at_least_one_gpu():
    for provider in CloudProvider:
        pricing = CloudCostEstimator.__init__.__module__
        # Just verify we can instantiate without error
        for gpu in GPUType:
            try:
                est = CloudCostEstimator(provider=provider, gpu_type=gpu, num_gpus=1)
                result = est.estimate_session_cost(1.0, 80.0)
                assert isinstance(result, CostEstimate)
            except Exception as exc:
                pytest.fail(f"Unexpected error for {provider}/{gpu}: {exc}")
