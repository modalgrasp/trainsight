"""
trainsight.integrations.accelerate
=====================================
HuggingFace Accelerate monitor for TrainSight.

Emits events:
* ``accelerate.config`` – Accelerator configuration at startup
* ``accelerate.step``   – per-step metrics (loss, grad scaler scale)

Usage::

    from accelerate import Accelerator
    from trainsight import Dashboard
    from trainsight.integrations.accelerate import AccelerateMonitor

    accelerator = Accelerator()
    dashboard = Dashboard()
    monitor = AccelerateMonitor(accelerator, dashboard)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    for batch in dataloader:
        with accelerator.accumulate(model):
            loss = model(batch)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            monitor.log_step(loss=loss.item())
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

try:
    from accelerate import Accelerator
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "accelerate is required: pip install accelerate"
    ) from exc

from trainsight.core.event import Event

logger = logging.getLogger("trainsight.accelerate")


class AccelerateMonitor:
    """
    Monitor a HuggingFace ``Accelerator`` training run.

    Parameters
    ----------
    accelerator:
        The ``Accelerator`` instance used for training.
    monitor:
        A TrainSight ``Dashboard`` instance.
    log_interval:
        Emit ``accelerate.step`` events every *N* steps.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        monitor: Any,
        log_interval: int = 10,
    ) -> None:
        self.accelerator = accelerator
        self.monitor = monitor
        self.log_interval = max(1, int(log_interval))
        self._step_counter: int = 0

        # Emit initial configuration
        self._emit_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_step(self, loss: Optional[float] = None) -> None:
        """Call once per optimizer step."""
        self._step_counter += 1
        if self._step_counter % self.log_interval != 0:
            return

        payload: dict[str, Any] = {
            "step": self._step_counter,
            "is_main_process": self.accelerator.is_main_process,
        }

        if loss is not None:
            payload["loss"] = float(loss)

        # Mixed precision grad scaler
        scaler = getattr(self.accelerator, "scaler", None)
        if scaler is not None:
            try:
                payload["grad_scaler_scale"] = float(scaler.get_scale())
            except Exception:
                pass

        self._emit("accelerate.step", payload)

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

    def _emit_config(self) -> None:
        try:
            payload: dict[str, Any] = {
                "distributed_type": str(self.accelerator.distributed_type),
                "num_processes": self.accelerator.num_processes,
                "process_index": self.accelerator.process_index,
                "local_process_index": self.accelerator.local_process_index,
                "device": str(self.accelerator.device),
                "mixed_precision": str(self.accelerator.mixed_precision),
                "gradient_accumulation_steps": self.accelerator.gradient_accumulation_steps,
            }
            self._emit("accelerate.config", payload)
        except Exception as exc:
            logger.debug("Could not emit Accelerate config: %s", exc)
