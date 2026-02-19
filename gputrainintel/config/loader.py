from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | None = None) -> dict[str, Any]:
    if path:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    packaged_default = files("gputrainintel.config").joinpath("default.yaml")
    return yaml.safe_load(packaged_default.read_text(encoding="utf-8")) or {}
