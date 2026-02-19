from __future__ import annotations

try:
    from transformers import TrainerCallback
except Exception as exc:  # pragma: no cover
    raise ImportError("transformers is required for HuggingFace integration") from exc


class GPUTrainIntelCallback(TrainerCallback):
    def __init__(self, monitor):
        self.monitor = monitor

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.monitor.update_training_stats(
                epoch=state.epoch,
                steps=state.global_step,
                loss=logs.get("loss"),
                lr=logs.get("learning_rate"),
            )
