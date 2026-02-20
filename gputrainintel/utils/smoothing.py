from __future__ import annotations


def ema(prev: float, current: float, alpha: float) -> float:
    alpha = max(0.0, min(1.0, float(alpha)))
    return alpha * float(current) + (1.0 - alpha) * float(prev)
