from __future__ import annotations


def thermal_headroom(temp_c: float) -> float:
    return max(0.0, 100.0 - float(temp_c))
