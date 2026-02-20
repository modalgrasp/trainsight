from __future__ import annotations


def classify_bottleneck(gpu_util: float, cpu_util: float, dataloader_wait: bool) -> str:
    if gpu_util < 60 and cpu_util > 80:
        return "CPU Bottleneck"
    if gpu_util < 60 and dataloader_wait:
        return "DataLoader Bottleneck"
    if gpu_util > 90:
        return "Model Bound"
    return "Balanced"
