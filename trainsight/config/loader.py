from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | None = None) -> dict[str, Any]:
    if path:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    default = files("trainsight.config").joinpath("default.yaml")
    return yaml.safe_load(default.read_text(encoding="utf-8")) or {}
