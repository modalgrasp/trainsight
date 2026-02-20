"""
Tests for trainsight.security.hardening.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from trainsight.security.hardening import (
    SecurityError,
    compute_plugin_hash,
    sanitize_command,
    validate_config_schema,
    validate_plugin_path,
)


# ---------------------------------------------------------------------------
# sanitize_command
# ---------------------------------------------------------------------------

def test_allowed_command_passes():
    result = sanitize_command(["nvidia-smi", "--query-gpu=name", "--format=csv"])
    assert result[0] == "nvidia-smi"


def test_disallowed_command_raises():
    with pytest.raises(SecurityError, match="not in the allowed list"):
        sanitize_command(["rm", "-rf", "/"])


def test_shell_metacharacter_raises():
    with pytest.raises(SecurityError, match="forbidden character"):
        sanitize_command(["nvidia-smi", "--format=csv;rm -rf /"])


def test_empty_command_raises():
    with pytest.raises(SecurityError):
        sanitize_command([])


def test_path_prefix_stripped():
    # /usr/bin/nvidia-smi should still be allowed (basename is nvidia-smi)
    result = sanitize_command(["/usr/bin/nvidia-smi", "--help"])
    assert result[0] == "/usr/bin/nvidia-smi"


# ---------------------------------------------------------------------------
# validate_plugin_path
# ---------------------------------------------------------------------------

def test_valid_plugin_path():
    allowed_dir = Path.home() / ".trainsight" / "plugins"
    plugin = allowed_dir / "my_plugin.py"
    # We don't need the file to exist for path validation
    # But resolve() may fail on non-existent paths on some systems
    # so we patch the allowed dirs
    from trainsight.security import hardening as h
    original = h._ALLOWED_PLUGIN_DIRS
    h._ALLOWED_PLUGIN_DIRS = (allowed_dir,)
    try:
        # Create a temp file inside a temp dir to simulate
        with tempfile.TemporaryDirectory() as tmpdir:
            h._ALLOWED_PLUGIN_DIRS = (Path(tmpdir),)
            plugin_file = Path(tmpdir) / "test_plugin.py"
            plugin_file.write_text("# test")
            assert validate_plugin_path(plugin_file) is True
    finally:
        h._ALLOWED_PLUGIN_DIRS = original


def test_invalid_plugin_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        outside_file = Path(tmpdir) / "evil_plugin.py"
        outside_file.write_text("# evil")
        assert validate_plugin_path(outside_file) is False


# ---------------------------------------------------------------------------
# compute_plugin_hash
# ---------------------------------------------------------------------------

def test_plugin_hash_is_deterministic():
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        f.write("def register(bus): pass\n")
        path = Path(f.name)

    try:
        h1 = compute_plugin_hash(path)
        h2 = compute_plugin_hash(path)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest
    finally:
        path.unlink()


def test_plugin_hash_differs_for_different_content():
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f1:
        f1.write("def register(bus): pass\n")
        path1 = Path(f1.name)

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f2:
        f2.write("def register(bus): bus.subscribe('x', lambda e: None)\n")
        path2 = Path(f2.name)

    try:
        assert compute_plugin_hash(path1) != compute_plugin_hash(path2)
    finally:
        path1.unlink()
        path2.unlink()


# ---------------------------------------------------------------------------
# validate_config_schema
# ---------------------------------------------------------------------------

def test_valid_config():
    ok, msg = validate_config_schema({"refresh_rate": 30, "prometheus_port": 9108})
    assert ok is True


def test_invalid_refresh_rate():
    ok, msg = validate_config_schema({"refresh_rate": 0})
    assert ok is False


def test_invalid_prometheus_port():
    ok, msg = validate_config_schema({"prometheus_port": 80})
    assert ok is False


def test_empty_config_is_valid():
    ok, msg = validate_config_schema({})
    assert ok is True
