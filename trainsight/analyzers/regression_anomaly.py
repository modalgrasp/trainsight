from __future__ import annotations

import numpy as np


class RegressionAnomalyDetector:
    def __init__(self) -> None:
        self.history: list[float] = []

    def update(self, value: float) -> dict | None:
        self.history.append(float(value))
        if len(self.history) < 20:
            return None

        data = np.array(self.history[-50:])
        mean = float(data.mean())
        std = float(data.std())
        z_score = (float(value) - mean) / (std + 1e-6)

        if abs(z_score) > 2.5:
            return {"anomaly": True, "z_score": float(z_score)}
        return None
