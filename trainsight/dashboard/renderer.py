from __future__ import annotations

from rich.panel import Panel
from rich.table import Table

from trainsight.branding import WATERMARK


class Renderer:
    def render_snapshot(self, payload: dict) -> Panel:
        table = Table.grid(padding=1)
        table.add_row("GPU", f"{payload.get('gpu_util', 0):.1f}%")
        table.add_row("VRAM", f"{payload.get('vram', 0):.1f}%")
        table.add_row("Temp", f"{payload.get('temp', 0):.0f}C")
        table.add_row("Power", f"{payload.get('power', 0):.1f}W")
        table.add_row("", WATERMARK)
        return Panel(table, title="TrainSight")
