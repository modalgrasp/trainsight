from __future__ import annotations

from typing import Any

from .legacy_app import GPUDashboard

from .ml.behavior_model import BehaviorModel
from .ml.session_store import save_session


class Dashboard(GPUDashboard):
    """Thin wrapper over the current dashboard with config-driven behavior."""

    def __init__(self, config: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config or {}
        self.behavior_model = BehaviorModel()
        self._time_axis: list[float] = []

    async def on_mount(self) -> None:
        await super().on_mount()
        refresh_hz = float(self.config.get("refresh_rate", 30))
        refresh_hz = max(1.0, min(60.0, refresh_hz))
        self._set_refresh(1.0 / refresh_hz)

    def update_dashboard(self) -> None:
        super().update_dashboard()
        if hasattr(self, "mem_history") and hasattr(self, "current_refresh_interval"):
            t = len(self.mem_history) * float(self.current_refresh_interval)
            self._time_axis.append(t)

    async def on_unmount(self) -> None:
        if self.config.get("enable_behavior_learning", True):
            try:
                times = self._time_axis[-len(self.mem_history):]
                vram = list(self.mem_history)
                self.behavior_model.train(times, vram)
                save_session(
                    {
                        "duration_s": times[-1] if times else 0,
                        "peak_gpu": float(getattr(self, "peak_gpu", 0.0)),
                        "peak_vram": float(getattr(self, "peak_vram", 0.0)),
                        "peak_temp": float(getattr(self, "peak_temp", 0.0)),
                        "samples": len(vram),
                    }
                )
            except Exception:
                pass

        await super().on_unmount()

    def _sync_graph_heights(self) -> None:
        super()._sync_graph_heights()

        width = self.size.width
        if width < 120:
            self.visible_graphs = ["gpu"]
        elif width < 160:
            self.visible_graphs = ["gpu", "vram"]
        else:
            self.visible_graphs = ["gpu", "vram", "temp"]
