from __future__ import annotations

import csv
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, Static

from .analyzers.regression_anomaly import RegressionAnomalyDetector
from .branding import assert_watermark_integrity
from .collectors.gpu_collector import GPUCollector
from .core.bus import EventBus
from .core.dispatcher import Dispatcher
from .core.event import Event
from .observability.prometheus import TrainSightPrometheusExporter
from .plugins.loader import load_plugins

logger = logging.getLogger("trainsight")


def row_from_payload(payload: dict[str, Any], power_cap_watts: float = 200.0) -> dict[str, Any]:
    """Convert GPU payload to display row."""
    mem_total = max(1.0, float(payload.get("mem_total", 1.0)))
    mem_used = max(0.0, float(payload.get("mem_used", 0.0)))
    gpu_util = max(0.0, min(100.0, float(payload.get("gpu_util", 0.0))))
    mem_percent = max(0.0, min(100.0, float(payload.get("vram", (mem_used / mem_total) * 100.0))))
    temp = max(0.0, float(payload.get("temp", 0.0)))
    power = max(0.0, float(payload.get("power", 0.0)))
    power_limit = max(1.0, float(payload.get("power_limit", power_cap_watts)))
    name = str(payload.get("name", "NVIDIA GPU"))
    idx = int(payload.get("index", 0))

    row = {
        "index": idx,
        "name": name,
        "gpu_util": gpu_util,
        "mem_percent": mem_percent,
        "mem_used": mem_used,
        "mem_total": mem_total,
        "temp": temp,
        "power": power,
        "power_limit": power_limit,
        "clock": 0,
        "max_clock": 0,
        "throttling": False,
    }
    row["gpu_rows"] = [row.copy()]
    row["imbalance"] = 0.0
    row["training_processes"] = []
    row["active_training"] = None
    row["mem_categories"] = {"Torch": 0.0, "Xformers": 0.0, "Other": 0.0}
    row["process_visibility_limited"] = bool(gpu_util > 50 and mem_percent > 20)
    row["torch_mem"] = None
    row["torch_mem_estimated"] = None
    return row


class GPUStatsWidget(Static):
    """Widget to display GPU statistics."""

    gpu_util = reactive(0.0)
    mem_percent = reactive(0.0)
    temp = reactive(0.0)
    power = reactive(0.0)
    gpu_name = reactive("NVIDIA GPU")

    def render(self) -> Text:
        """Render GPU stats."""
        lines = [
            Text(f"GPU: {self.gpu_name}", style="bold cyan"),
            Text(f"Utilization: {self.gpu_util:.1f}%", style="green" if self.gpu_util < 80 else "yellow" if self.gpu_util < 95 else "red"),
            Text(f"Memory: {self.mem_percent:.1f}%", style="green" if self.mem_percent < 80 else "yellow" if self.mem_percent < 95 else "red"),
            Text(f"Temperature: {self.temp:.1f}°C", style="green" if self.temp < 80 else "yellow" if self.temp < 90 else "red"),
            Text(f"Power: {self.power:.1f}W", style="dim"),
        ]
        return Text("\n").join(lines)


class Dashboard(App):
    """TrainSight GPU Monitoring Dashboard."""

    CSS = """
    Screen {
        background: $surface;
    }
    
    .header {
        text-align: center;
        padding: 1;
    }
    
    .stats-container {
        padding: 1;
    }
    
    GPUStatsWidget {
        padding: 1;
        border: solid $primary;
        margin: 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("s", "toggle_simulation", "Simulation"),
    ]

    simulation = reactive(False)

    def __init__(self, config: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config or {}
        self.event_bus = EventBus()
        self.dispatcher = Dispatcher(self.event_bus, [GPUCollector()])
        self.gpu_anomaly = RegressionAnomalyDetector()
        self.last_event_payload: dict[str, Any] | None = None
        self.prometheus = None
        self.power_cap_watts = 200.0

        self.simulation = bool(self.config.get("simulation", False))
        self._rng = random.Random(7)
        self.replay_file = str(self.config.get("replay_file", "") or "")
        self._replay_rows: list[dict[str, Any]] = []
        self._replay_idx = 0

        self.event_bus.subscribe("gpu.stats", self._on_gpu_stats)
        load_plugins(self.event_bus)

    def _on_gpu_stats(self, event) -> None:
        self.last_event_payload = event.payload
        self.gpu_anomaly.update(float(event.payload.get("gpu_util", 0.0)))

    def compose(self) -> ComposeResult:
        """Compose the dashboard layout."""
        yield Header()
        with Container(classes="stats-container"):
            yield GPUStatsWidget(id="gpu-stats")
        yield Footer()

    async def on_mount(self) -> None:
        """Mount handler."""
        assert_watermark_integrity()
        hz = float(self.config.get("refresh_rate", 30))
        hz = max(30.0, min(60.0, hz))
        self.set_interval(1.0 / hz, self._update_stats)

        if bool(self.config.get("enable_prometheus", False)):
            port = int(self.config.get("prometheus_port", 9108))
            self.prometheus = TrainSightPrometheusExporter(port=port)
            self.prometheus.start()
            self.event_bus.subscribe("gpu.stats", self.prometheus.handle_gpu_event)
            logger.info("Prometheus exporter started on :%s", port)

        if self.replay_file:
            self._replay_rows = self._load_replay_rows(self.replay_file)
            logger.info("Replay mode: %s rows loaded from %s", len(self._replay_rows), self.replay_file)

        if self.simulation:
            logger.info("Simulation mode enabled")

    def _emit_payload(self, payload: dict[str, Any]) -> None:
        self.event_bus.emit(Event(type="gpu.stats", payload=payload, timestamp=datetime.now(timezone.utc)))

    def _simulated_payload(self) -> dict[str, Any]:
        mem_total = 12227.0
        mem_used = self._rng.uniform(0.4, 0.95) * mem_total
        return {
            "index": 0,
            "name": "Simulated GPU",
            "gpu_util": self._rng.uniform(60, 100),
            "vram": (mem_used / mem_total) * 100.0,
            "mem_used": mem_used,
            "mem_total": mem_total,
            "temp": self._rng.uniform(60, 85),
            "power": self._rng.uniform(80, 160),
            "power_limit": 200.0,
        }

    def _load_replay_rows(self, csv_path: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        p = Path(csv_path)
        if not p.exists():
            logger.warning("Replay file missing: %s", csv_path)
            return rows
        try:
            with p.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                for raw in reader:
                    if len(raw) < 5:
                        continue
                    try:
                        gpu_util = float(raw[1])
                        mem_percent = float(raw[2])
                        temp = float(raw[3])
                        power = float(raw[4])
                    except Exception:
                        continue
                    mem_total = 100.0
                    mem_used = max(0.0, min(100.0, mem_percent))
                    rows.append(
                        {
                            "index": 0,
                            "name": "Replay GPU",
                            "gpu_util": gpu_util,
                            "vram": mem_percent,
                            "mem_used": mem_used,
                            "mem_total": mem_total,
                            "temp": temp,
                            "power": power,
                            "power_limit": 200.0,
                        }
                    )
        except Exception as exc:
            logger.warning("Replay read failed (%s): %s", csv_path, exc)
        return rows

    def _next_replay_payload(self) -> dict[str, Any] | None:
        if not self._replay_rows:
            return None
        payload = self._replay_rows[self._replay_idx]
        self._replay_idx = (self._replay_idx + 1) % len(self._replay_rows)
        return payload

    def get_gpu_stats(self) -> dict[str, Any]:
        """Get GPU statistics."""
        if self.simulation:
            payload = self._simulated_payload()
            self._emit_payload(payload)
            return row_from_payload(payload, self.power_cap_watts)

        replay_payload = self._next_replay_payload()
        if replay_payload is not None:
            self._emit_payload(replay_payload)
            return row_from_payload(replay_payload, self.power_cap_watts)

        try:
            self.dispatcher.collect_once()
        except Exception as exc:
            logger.debug("Dispatcher collect failed: %s", exc)

        payload = self.last_event_payload
        if payload:
            return row_from_payload(payload, self.power_cap_watts)

        # Return default row if no data available
        return row_from_payload({}, self.power_cap_watts)

    def _update_stats(self) -> None:
        """Update GPU stats display."""
        try:
            stats = self.get_gpu_stats()
            widget = self.query_one("#gpu-stats", GPUStatsWidget)
            widget.gpu_name = stats.get("name", "NVIDIA GPU")
            widget.gpu_util = stats.get("gpu_util", 0.0)
            widget.mem_percent = stats.get("mem_percent", 0.0)
            widget.temp = stats.get("temp", 0.0)
            widget.power = stats.get("power", 0.0)
        except Exception as exc:
            logger.debug("Stats update failed: %s", exc)

    def action_toggle_simulation(self) -> None:
        """Toggle simulation mode."""
        self.simulation = not self.simulation
        logger.info("Simulation mode: %s", self.simulation)

    def action_refresh(self) -> None:
        """Force refresh."""
        self._update_stats()
