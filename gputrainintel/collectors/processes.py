from __future__ import annotations

from typing import Any


def collect_training_processes(stats: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not stats:
        return []
    return list(stats.get("training_processes", []))
