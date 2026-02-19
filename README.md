# TrainSight

AI Training Intelligence Dashboard for live GPU telemetry, anomaly detection, and training-oriented diagnostics.

## Architecture

```text
Collectors -> EventBus -> Analyzers -> Predictors -> Dashboard / CLI / Logger
```

Core modules:

- `trainsight/core/event.py`
- `trainsight/core/bus.py`
- `trainsight/core/dispatcher.py`
- `trainsight/collectors/gpu_collector.py`
- `trainsight/analyzers/regression_anomaly.py`
- `trainsight/predictors/oom_predictor.py`
- `trainsight/plugins/loader.py`

## Installation

```bash
pip install .
```

## CLI Usage

```bash
trainsight
```

## HuggingFace Integration

```python
from trainsight import Dashboard
from trainsight.integrations.huggingface import TrainSightCallback
```

See `examples/huggingface_example.py`.

## Plugin Example

Create `~/.trainsight/plugins/my_plugin.py`:

```python
def register(bus):
    bus.subscribe("gpu.stats", custom_handler)


def custom_handler(event):
    print("Custom plugin:", event.payload)
```

## Config Example

Default config: `trainsight/config/default.yaml`

```yaml
mode: full
enable_behavior_learning: true
oom_model: statistical
thermal_limit: 85
refresh_rate: 30
```


## Official Build Verification

- Soft check (default): warns on signature/hash mismatch.
- Strict mode: refuses startup when verification fails.

Enable strict mode with either:

```bash
export TRAINSIGHT_OFFICIAL_ONLY=1
```

or in config:

```yaml
strict_official_build: true
```

## Prometheus Exporter

Enable in config:

```yaml
enable_prometheus: true
prometheus_port: 9108
```

Metrics endpoint:

```text
http://127.0.0.1:9108/metrics
```


## Test Framework

Install dev deps and run tests:

```bash
pip install -e .[dev]
pytest -q
```

Tests are hardware-independent and use pure logic paths.

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
