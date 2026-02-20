"""
trainsight.integrations.deepspeed
===================================
DeepSpeed training monitor for TrainSight.

Emits events:
* ``deepspeed.zero_stats``            – ZeRO optimizer statistics
* ``deepspeed.gradient_accumulation`` – gradient accumulation state
* ``deepspeed.communication``         – communication data type info
* ``deepspeed.memory_imbalance``      – memory imbalance across ranks (if detectable)

Usage::

    import deepspeed
    from trainsight import Dashboard
    from trainsight.integrations.deepspeed import DeepSpeedMonitor

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, config=ds_config
    )
    monitor = DeepSpeedMonitor(model_engine, dashboard)

    for batch in dataloader:
        loss = model_engine(batch)
        model_engine.backward(loss)
        model_engine.step()
        monitor.log_step()
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

try:
    import deepspeed  # noqa: F401 – just verify it is installed
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "deepspeed is required: pip install deepspeed"
    ) from exc

from trainsight.core.event import Event

logger = logging.getLogger("trainsight.deepspeed")


class DeepSpeedMonitor:
    """
    Monitor a DeepSpeed ``ModelEngine`` and emit telemetry events.

    Parameters
    ----------
    model_engine:
        The ``deepspeed.DeepSpeedEngine`` returned by ``deepspeed.initialize``.
    monitor:
        A TrainSight ``Dashboard`` instance.
    log_interval:
        Emit events every *N* steps.
    """

    def __init__(
        self,
        model_engine: Any,
        monitor: Any,
        log_interval: int = 10,
    ) -> None:
        self.model_engine = model_engine
        self.monitor = monitor
        self.log_interval = max(1, int(log_interval))
        self._step_counter: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_step(self) -> None:
        """Call once per optimizer step to emit DeepSpeed telemetry."""
        self._step_counter += 1
        if self._step_counter % self.log_interval != 0:
            return

        self._emit_zero_stats()
        self._emit_gradient_accumulation()
        self._emit_communication_info()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        event = Event(
            type=event_type,
            payload=payload,
            timestamp=datetime.now(timezone.utc),
        )
        bus = getattr(self.monitor, "event_bus", None)
        if bus is None:
            return
        emit_fn = getattr(bus, "emit_async", None) or bus.emit
        try:
            emit_fn(event)
        except Exception:
            pass

    def _emit_zero_stats(self) -> None:
        stats: dict[str, Any] = {"step": self._step_counter}
        try:
            opt = self.model_engine.optimizer
            if hasattr(opt, "zero_optimization_stage"):
                stats["zero_stage"] = opt.zero_optimization_stage()
            if hasattr(opt, "partition_count"):
                stats["partition_count"] = opt.partition_count()
            if hasattr(opt, "get_memory_stats"):
                stats["memory"] = opt.get_memory_stats()
        except Exception as exc:
            logger.debug("ZeRO stats unavailable: %s", exc)
            return
        self._emit("deepspeed.zero_stats", stats)

    def _emit_gradient_accumulation(self) -> None:
        try:
            payload: dict[str, Any] = {"step": self._step_counter}
            if hasattr(self.model_engine, "gradient_accumulation_steps"):
                payload["gradient_accumulation_steps"] = (
                    self.model_engine.gradient_accumulation_steps()
                )
            if hasattr(self.model_engine, "train_micro_batch_size_per_gpu"):
                payload["micro_batch_size"] = (
                    self.model_engine.train_micro_batch_size_per_gpu()
                )
            self._emit("deepspeed.gradient_accumulation", payload)
        except Exception as exc:
            logger.debug("Gradient accumulation info unavailable: %s", exc)

    def _emit_communication_info(self) -> None:
        try:
            payload: dict[str, Any] = {"step": self._step_counter}
            if hasattr(self.model_engine, "communication_data_type"):
                payload["data_type"] = str(self.model_engine.communication_data_type)
            self._emit("deepspeed.communication", payload)
        except Exception as exc:
            logger.debug("Communication info unavailable: %s", exc)
