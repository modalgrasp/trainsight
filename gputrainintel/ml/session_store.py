from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STORE = Path.home() / ".gputrainintel_sessions.json"


def load_sessions() -> list[dict[str, Any]]:
    if STORE.exists():
        return json.loads(STORE.read_text(encoding="utf-8"))
    return []


def save_session(data: dict[str, Any]) -> None:
    sessions = load_sessions()
    sessions.append(data)
    STORE.write_text(json.dumps(sessions, indent=2), encoding="utf-8")
