from __future__ import annotations


def thermal_state(temp_c: float, thermal_limit: float = 85.0) -> str:
    if temp_c >= thermal_limit + 5:
        return "critical"
    if temp_c >= thermal_limit:
        return "hot"
    return "ok"
