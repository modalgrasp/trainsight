# TrainSight

AI Training Intelligence Dashboard for live GPU telemetry, anomaly detection, and training-oriented diagnostics.

## Features

- **Live GPU Monitoring**: Real-time GPU utilization, memory, temperature, and power metrics
- **Anomaly Detection**: Automatic detection of thermal throttling, memory leaks, and training anomalies
- **OOM Prediction**: Predict Out-of-Memory errors before they crash your training
- **Framework Integrations**: Native support for PyTorch, Lightning, HuggingFace, DeepSpeed, Accelerate, Ray Train
- **Experiment Tracking**: Bridge to MLflow and Weights & Biases
- **Production Ready**: Prometheus exporter, Kubernetes GPU pod monitoring, cloud cost estimation
- **Batch Size Optimization**: Automatic batch size recommendation based on GPU memory

## Installation

### From PyPI (Recommended)

```bash
pip install trainsight
```

### From GitHub

```bash
pip install git+https://github.com/modalgrasp/trainsight.git
```

### Optional Dependencies

Install only what you need:

```bash
# PyTorch integration
pip install "trainsight[pytorch]"

# PyTorch Lightning integration
pip install "trainsight[lightning]"

# DeepSpeed integration
pip install "trainsight[deepspeed]"

# HuggingFace Accelerate integration
pip install "trainsight[accelerate]"

# Ray Train integration
pip install "trainsight[ray]"

# Experiment tracking
pip install "trainsight[mlflow]"
pip install "trainsight[wandb]"

# Kubernetes monitoring
pip install "trainsight[kubernetes]"

# Install everything
pip install "trainsight[all]"
```

## Quick Start

### CLI Dashboard

```bash
trainsight
```

### Programmatic Usage

```python
from trainsight import Dashboard
from trainsight.core.bus import EventBus
from trainsight.collectors.gpu_collector import GPUCollector

# Create event bus and collector
bus = EventBus()
collector = GPUCollector()

# Subscribe to GPU events
def on_gpu_stats(event):
    print(f"GPU Util: {event.payload['utilization']}%")

bus.subscribe("gpu.stats", on_gpu_stats)

# Start dashboard
# dashboard = Dashboard(bus, collector)
# dashboard.run()
```

### PyTorch Integration

```python
import torch
from trainsight.integrations.pytorch import TrainSightHook

model = torch.nn.Linear(100, 10)
hook = TrainSightHook(model)

# Hook automatically monitors gradients and activations
output = model(torch.randn(32, 100))
output.backward()
```

### PyTorch Lightning Integration

```python
import pytorch_lightning as pl
from trainsight.integrations.lightning import TrainSightCallback

trainer = pl.Trainer(
    callbacks=[TrainSightCallback()],
    max_epochs=10,
)
trainer.fit(model)
```

### HuggingFace Transformers Integration

```python
from transformers import Trainer, TrainingArguments
from trainsight.integrations.huggingface import TrainSightCallback

training_args = TrainingArguments(
    output_dir="./output",
    callbacks=[TrainSightCallback],
)
trainer = Trainer(model=model, args=training_args)
```

## Architecture

```text
Collectors -> EventBus -> Analyzers -> Predictors -> Dashboard / CLI / Logger
```

Core modules:

- `trainsight/core/event.py` - Event types and payloads
- `trainsight/core/bus.py` - Synchronous event bus
- `trainsight/core/async_bus.py` - Non-blocking async event bus
- `trainsight/core/dispatcher.py` - Collector orchestration
- `trainsight/collectors/gpu_collector.py` - NVIDIA GPU metrics
- `trainsight/analyzers/` - Anomaly and bottleneck detection
- `trainsight/predictors/oom_predictor.py` - OOM prediction
- `trainsight/integrations/` - Framework integrations

## Configuration

Default config: `trainsight/config/default.yaml`

```yaml
mode: full
enable_behavior_learning: true
oom_model: statistical
thermal_limit: 85
refresh_rate: 30
```

## Prometheus Exporter

Enable in config:

```yaml
enable_prometheus: true
prometheus_port: 9108
```

Metrics endpoint: `http://127.0.0.1:9108/metrics`

## Official Build Verification

- Soft check (default): warns on signature/hash mismatch.
- Strict mode: refuses startup when verification fails.

Enable strict mode:

```bash
export TRAINSIGHT_OFFICIAL_ONLY=1
```

Or in config:

```yaml
strict_official_build: true
```

## Plugin System

Create `~/.trainsight/plugins/my_plugin.py`:

```python
def register(bus):
    bus.subscribe("gpu.stats", custom_handler)

def custom_handler(event):
    print("Custom plugin:", event.payload)
```

## Debug / Simulation / Replay

```bash
trainsight --debug
trainsight --simulate
trainsight --replay gpu_usage_log.csv
```

Textual inspector:

```bash
TEXTUAL_DEVTOOLS=1 trainsight --debug
```

## Testing

Install dev dependencies and run tests:

```bash
pip install -e ".[dev]"
pytest -q
```

Tests are hardware-independent and use pure logic paths.

## License

TrainSight Community License v1.0 - See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
