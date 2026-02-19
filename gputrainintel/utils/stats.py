from __future__ import annotations


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))
