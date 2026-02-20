"""
trainsight.integrations.lightning
===================================
PyTorch Lightning callback for TrainSight monitoring.

Emits events:
* ``training.started``           – training session begins
* ``training.dataloader_timing`` – per-batch wall-clock time
* ``training.metrics``           – trainer callback_metrics snapshot
* ``training.validation_drift``  – val_loss increasing 3+ consecutive validations
* ``training.completed``         – training session ends

Usage::

    from trainsight import Dashboard
    from trainsight.integrations.lightning import TrainSightCallback

    dashboard = Dashboard()
    trainer = pl.Trainer(callbacks=[TrainSightCallback(dashboard)])
    trainer.fit(model, datamodule)
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

try:
    from pytorch_lightning import Callback, LightningModule, Trainer
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pytorch-lightning is required: pip install pytorch-lightning"
    ) from exc

from trainsight.core.event import Event


class TrainSightCallback(Callback):
    """
    PyTorch Lightning callback that feeds training telemetry into TrainSight.

    Parameters
    ----------
    monitor:
        A TrainSight ``Dashboard`` instance.
    log_interval:
        Emit ``training.metrics`` every *N* batches.
    track_dataloader_time:
        Measure wall-clock time per batch.
    track_validation_drift:
        Warn when validation loss increases for 3 consecutive validations.
    """

    def __init__(
        self,
        monitor: Any,
        log_interval: int = 10,
        track_dataloader_time: bool = True,
        track_validation_drift: bool = True,
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.log_interval = max(1, int(log_interval))
        self.track_dataloader_time = track_dataloader_time
        self.track_validation_drift = track_validation_drift

        self._batch_start_ts: float = 0.0
        self._dataloader_times: list[float] = []
        self._val_loss_history: list[float] = []

    # ------------------------------------------------------------------
    # Helpers
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

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    @rank_zero_only
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._emit(
            "training.started",
            {
                "max_epochs": trainer.max_epochs,
                "accelerator": type(trainer.accelerator).__name__,
                "num_devices": trainer.num_devices,
                "precision": str(trainer.precision),
            },
        )

    @rank_zero_only
    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.track_dataloader_time:
            self._batch_start_ts = time.monotonic()

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        # Dataloader timing
        if self.track_dataloader_time and self._batch_start_ts > 0:
            elapsed = time.monotonic() - self._batch_start_ts
            self._dataloader_times.append(elapsed)
            if len(self._dataloader_times) > 200:
                self._dataloader_times.pop(0)

        if batch_idx % self.log_interval != 0:
            return

        payload: dict[str, Any] = {
            "global_step": trainer.global_step,
            "epoch": trainer.current_epoch,
        }

        if self.track_dataloader_time and self._dataloader_times:
            avg_ms = (sum(self._dataloader_times) / len(self._dataloader_times)) * 1000
            payload["avg_batch_time_ms"] = avg_ms

        if trainer.callback_metrics:
            payload["metrics"] = {
                k: float(v) for k, v in trainer.callback_metrics.items()
            }

        self._emit("training.metrics", payload)

    @rank_zero_only
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.track_validation_drift:
            return
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is None:
            return
        self._val_loss_history.append(float(val_loss))
        if len(self._val_loss_history) >= 3:
            recent = self._val_loss_history[-3:]
            if all(recent[i] > recent[i - 1] for i in range(1, len(recent))):
                self._emit(
                    "training.validation_drift",
                    {
                        "message": "Validation loss increasing for 3 consecutive validations",
                        "recent_losses": recent,
                        "severity": "warning",
                        "epoch": trainer.current_epoch,
                    },
                )

    @rank_zero_only
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._emit(
            "training.completed",
            {
                "total_steps": trainer.global_step,
                "final_epoch": trainer.current_epoch,
            },
        )
