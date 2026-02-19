from __future__ import annotations

import numpy as np


def oom_probability(current_vram: float, predicted_vram: float | None, total_vram: float) -> float:
    if predicted_vram is None:
        return 0.0
    margin = float(total_vram) - float(predicted_vram)
    prob = 1.0 / (1.0 + np.exp(margin / 500.0))
    return float(max(0.0, min(1.0, prob)))
