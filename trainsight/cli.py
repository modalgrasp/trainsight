from __future__ import annotations

import argparse
import logging
import os

from .app import Dashboard
from .branding import assert_watermark_integrity
from .config.loader import load_config
from .security.release import verify_release

logger = logging.getLogger("trainsight")


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trainsight", description="TrainSight GPU training monitor")
    p.add_argument("--config", help="Path to YAML config", default=None)
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    p.add_argument("--simulate", action="store_true", help="Run with simulated GPU metrics")
    p.add_argument("--replay", help="Replay telemetry from csv log")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    assert_watermark_integrity()
    config = load_config(args.config)

    if args.debug:
        config["debug"] = True
    if args.simulate:
        config["simulation"] = True
    if args.replay:
        config["replay_file"] = args.replay

    strict = bool(config.get("strict_official_build", False)) or _truthy(os.getenv("TRAINSIGHT_OFFICIAL_ONLY"))
    full_verify = bool(config.get("full_release_verify", False)) or _truthy(os.getenv("TRAINSIGHT_FULL_VERIFY"))
    result = verify_release(strict=strict, full_check=(strict or full_verify))
    if strict and not result.ok:
        raise SystemExit(f"TrainSight refused to start (official mode): {result.message}")
    if not result.ok:
        logger.debug("Release verification (non-strict): %s", result.message)

    logger.debug("TrainSight config: %s", config)
    app = Dashboard(config=config)
    app.run()
