"""
trainsight.security.hardening
================================
Security hardening utilities for TrainSight.

Provides:
* Command sanitisation for subprocess calls
* Plugin path validation
* Plugin file hash computation
* Config schema validation (requires pydantic)

These utilities are used internally by TrainSight but can also be called
directly by security-conscious deployments.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("trainsight.security.hardening")

# ---------------------------------------------------------------------------
# Allowed subprocess commands (whitelist)
# ---------------------------------------------------------------------------
_ALLOWED_COMMANDS: frozenset[str] = frozenset({"nvidia-smi", "ps", "top"})

# Shell metacharacters that must never appear in command arguments
_SHELL_METACHARACTERS: frozenset[str] = frozenset(
    {";", "&", "|", "`", "$", "(", ")", "<", ">", "\\", "\n", "\r"}
)

# Allowed plugin directories
_ALLOWED_PLUGIN_DIRS: tuple[Path, ...] = (
    Path.home() / ".trainsight" / "plugins",
)


class SecurityError(Exception):
    """Raised when a security constraint is violated."""


# ---------------------------------------------------------------------------
# Command sanitisation
# ---------------------------------------------------------------------------

def sanitize_command(cmd: list[str]) -> list[str]:
    """
    Validate and sanitise a subprocess command list.

    Raises
    ------
    SecurityError
        If the command is not whitelisted or contains shell metacharacters.
    """
    if not cmd:
        raise SecurityError("Empty command")

    executable = Path(cmd[0]).name  # strip path prefix
    if executable not in _ALLOWED_COMMANDS:
        raise SecurityError(
            f"Command '{executable}' is not in the allowed list: "
            f"{sorted(_ALLOWED_COMMANDS)}"
        )

    sanitised: list[str] = []
    for arg in cmd:
        for ch in _SHELL_METACHARACTERS:
            if ch in arg:
                raise SecurityError(
                    f"Argument contains forbidden character {ch!r}: {arg!r}"
                )
        sanitised.append(arg)

    return sanitised


# ---------------------------------------------------------------------------
# Plugin path validation
# ---------------------------------------------------------------------------

def validate_plugin_path(plugin_path: Path) -> bool:
    """
    Return ``True`` if *plugin_path* is inside an allowed plugin directory.

    Resolves symlinks before checking to prevent path traversal attacks.
    """
    try:
        resolved = plugin_path.resolve()
    except Exception:
        return False

    for allowed in _ALLOWED_PLUGIN_DIRS:
        try:
            resolved.relative_to(allowed.resolve())
            return True
        except ValueError:
            continue
    return False


def compute_plugin_hash(plugin_path: Path) -> str:
    """
    Compute the SHA-256 hash of a plugin file.

    Returns the hex digest string.
    """
    sha256 = hashlib.sha256()
    try:
        with plugin_path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                sha256.update(chunk)
    except OSError as exc:
        raise SecurityError(f"Cannot read plugin file: {exc}") from exc
    return sha256.hexdigest()


# ---------------------------------------------------------------------------
# Config schema validation
# ---------------------------------------------------------------------------

def validate_config_schema(config: dict[str, Any]) -> tuple[bool, str]:
    """
    Validate a TrainSight config dictionary.

    Uses pydantic if available; falls back to manual type checks.

    Returns
    -------
    (ok, message)
        ``ok`` is ``True`` when the config is valid.
    """
    try:
        from pydantic import BaseModel, Field, ValidationError

        class _Config(BaseModel):
            mode: str = "full"
            refresh_rate: int = Field(default=30, ge=1, le=120)
            enable_prometheus: bool = False
            prometheus_port: int = Field(default=9108, ge=1024, le=65535)
            strict_official_build: bool = False
            enable_behavior_learning: bool = True
            oom_model: str = "statistical"
            thermal_limit: float = Field(default=85.0, ge=0.0, le=150.0)
            simulation: bool = False
            debug: bool = False

        _Config(**{k: v for k, v in config.items() if k in _Config.model_fields})
        return True, "Config is valid"

    except ImportError:
        # pydantic not installed – do basic checks
        return _manual_validate(config)
    except Exception as exc:
        return False, f"Config validation failed: {exc}"


def _manual_validate(config: dict[str, Any]) -> tuple[bool, str]:
    """Minimal config validation without pydantic."""
    refresh_rate = config.get("refresh_rate", 30)
    if not isinstance(refresh_rate, (int, float)) or not (1 <= refresh_rate <= 120):
        return False, f"refresh_rate must be 1–120, got {refresh_rate!r}"

    prometheus_port = config.get("prometheus_port", 9108)
    if not isinstance(prometheus_port, int) or not (1024 <= prometheus_port <= 65535):
        return False, f"prometheus_port must be 1024–65535, got {prometheus_port!r}"

    thermal_limit = config.get("thermal_limit", 85.0)
    if not isinstance(thermal_limit, (int, float)) or not (0 <= thermal_limit <= 150):
        return False, f"thermal_limit must be 0–150, got {thermal_limit!r}"

    return True, "Config is valid"
