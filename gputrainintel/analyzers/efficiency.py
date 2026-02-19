from __future__ import annotations


def efficiency_score(tokens_per_sec: float, power: float) -> float:
    if power <= 0:
        return 0.0
    return float(tokens_per_sec) / float(power)
