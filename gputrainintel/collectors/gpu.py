from __future__ import annotations

from typing import Any


def collect_gpu_snapshot(app: Any) -> dict[str, Any] | None:
    """Adapter to current app collector for easier future decoupling."""
    return app.get_gpu_stats()
