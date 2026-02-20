# TrainSight Testing Guide

This guide covers all testing procedures for the TrainSight library before publishing to PyPI.

## Prerequisites

- Python 3.10+ installed
- Git installed
- NVIDIA GPU with drivers (optional, for real GPU testing)

## 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/modalgrasp/trainsight.git
cd trainsight

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install in development mode with test dependencies
pip install -e ".[dev]"
```

## 2. Run Unit Tests

```bash
# Run all tests with verbose output
pytest -v

# Run with coverage report
pytest --cov=trainsight --cov-report=term-missing

# Run specific test file
pytest tests/test_pytorch_hook.py -v

# Run tests matching a pattern
pytest -k "async_bus" -v
```

### Expected Output

All 48 tests should pass:

```
============================= test session starts =============================
collected 48 items

tests/test_amp_monitor.py::test_stable_training_no_event PASSED
tests/test_amp_monitor.py::test_high_variance_triggers_instability PASSED
...
tests/test_stats.py::test_peak_tracking_logic_like_dashboard PASSED

============================= 48 passed in 1.66s ==============================
```

## 3. Test Categories

### Core Tests
| Test File | What It Tests |
|-----------|---------------|
| `test_event_bus.py` | EventBus subscribe/emit functionality |
| `test_async_bus.py` | AsyncEventBus non-blocking delivery, queue overflow |
| `test_anomaly.py` | RegressionAnomalyDetector outlier detection |
| `test_oom.py` | OOM probability prediction |
| `test_stats.py` | Payload transformation and clamping |

### Integration Tests
| Test File | What It Tests |
|-----------|---------------|
| `test_pytorch_hook.py` | TrainSightHook gradient/activation monitoring |
| `test_loss_monitor.py` | LossTrendMonitor plateau/divergence detection |
| `test_amp_monitor.py` | AMPStabilityMonitor mixed precision monitoring |
| `test_cost_estimator.py` | Cloud cost calculation for all providers |

### Security Tests
| Test File | What It Tests |
|-----------|---------------|
| `test_security_hardening.py` | Command sanitization, plugin path validation, config validation |

## 4. Build Verification

```bash
# Install build tool
pip install build

# Build source distribution and wheel
python -m build

# Verify the dist directory contains:
# - trainsight-0.2.0.tar.gz (source distribution)
# - trainsight-0.2.0-py3-none-any.whl (wheel)
ls dist/
```

## 5. Installation Testing

### Test from local wheel

```bash
# Create a fresh virtual environment
python -m venv test_env
test_env\Scripts\activate  # Windows
# source test_env/bin/activate  # Linux/macOS

# Install from local wheel
pip install dist/trainsight-0.2.0-py3-none-any.whl

# Test import
python -c "from trainsight import Dashboard; print('Import successful!')"

# Test CLI
trainsight --help
```

### Test optional dependencies

```bash
# Test PyTorch integration
pip install "trainsight[pytorch]"
python -c "from trainsight.integrations.pytorch import TrainSightHook; print('PyTorch integration OK')"

# Test Lightning integration
pip install "trainsight[lightning]"
python -c "from trainsight.integrations.lightning import TrainSightCallback; print('Lightning integration OK')"
```

## 6. Pre-Publish Checklist

- [ ] All 48 tests pass (`pytest -v`)
- [ ] No import errors (`python -c "import trainsight"`)
- [ ] CLI works (`trainsight --help`)
- [ ] Build succeeds (`python -m build`)
- [ ] Wheel installs in fresh environment
- [ ] Version number updated in `pyproject.toml`
- [ ] CHANGELOG updated (if exists)
- [ ] Git tag created for version

## 7. Publishing to PyPI

### Test PyPI (Recommended First)

```bash
# Install twine
pip install twine

# Upload to Test PyPI
twine upload --repository testpypi dist/*

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ trainsight
```

### Production PyPI

```bash
# Upload to PyPI
twine upload dist/*
```

## 8. Post-Publish Verification

```bash
# Install from PyPI
pip install trainsight

# Verify version
python -c "import trainsight; print(trainsight.__version__)"

# Test CLI
trainsight --version
```

## 9. Continuous Integration

The project uses GitHub Actions for CI. The workflow should:

1. Run on every push to `main` and on pull requests
2. Test on Python 3.10, 3.11, 3.12
3. Run all tests with coverage
4. Build the package

Example `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    
    - name: Run tests
      run: pytest --cov=trainsight --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v4
```

## 10. Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'trainsight'`:
- Ensure you're in the correct virtual environment
- Reinstall: `pip install -e .`

### Test Failures

If tests fail:
1. Check Python version: `python --version` (must be 3.10+)
2. Reinstall dependencies: `pip install -e ".[dev]" --force-reinstall`
3. Clear pytest cache: `pytest --cache-clear`

### Build Warnings

The license classifier warning is expected for proprietary licenses. The build will still succeed.

### GPU-Related Errors

Tests are designed to run without GPU hardware. If you see GPU-related errors:
- Tests should mock GPU access
- Real GPU testing is optional and requires NVIDIA drivers
