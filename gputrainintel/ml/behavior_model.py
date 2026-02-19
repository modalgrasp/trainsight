from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


class BehaviorModel:
    def __init__(self) -> None:
        self.vram_model = LinearRegression()
        self.trained = False

    def train(self, time_series: list[float], vram_series: list[float]) -> None:
        if len(time_series) < 10 or len(vram_series) < 10:
            return
        n = min(len(time_series), len(vram_series))
        x = np.array(time_series[:n]).reshape(-1, 1)
        y = np.array(vram_series[:n])
        self.vram_model.fit(x, y)
        self.trained = True

    def predict_vram(self, future_t: float) -> float | None:
        if not self.trained:
            return None
        return float(self.vram_model.predict([[future_t]])[0])
