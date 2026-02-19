from __future__ import annotations

import importlib.util
from pathlib import Path

from trainsight.core.bus import EventBus


def load_plugins(bus: EventBus) -> None:
    plugin_dir = Path.home() / ".trainsight" / "plugins"
    if not plugin_dir.exists():
        return

    for file in plugin_dir.glob("*.py"):
        try:
            spec = importlib.util.spec_from_file_location(f"trainsight_user_plugin_{file.stem}", file)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            register = getattr(module, "register", None)
            if callable(register):
                register(bus)
        except Exception:
            continue
