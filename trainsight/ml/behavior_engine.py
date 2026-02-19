from __future__ import annotations

import numpy as np


class BehaviorEngine:
    def __init__(self) -> None:
        self.vram_slope_history: list[float] = []

    def update(self, vram_series: list[float]) -> None:
        if len(vram_series) < 10:
            return
        slope = float(np.polyfit(range(len(vram_series)), vram_series, 1)[0])
        self.vram_slope_history.append(slope)

    def predict_oom_risk(self, current_slope: float) -> float:
        if len(self.vram_slope_history) < 5:
            return 0.0

        mean = float(np.mean(self.vram_slope_history))
        std = float(np.std(self.vram_slope_history))
        if current_slope > mean + 2 * std:
            return 0.8
        return 0.2
