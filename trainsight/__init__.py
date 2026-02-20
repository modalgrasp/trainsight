from .app import Dashboard

__all__ = ["Dashboard"]

# ---------------------------------------------------------------------------
# Convenience re-exports for commonly used sub-packages.
# All framework integrations are optional – they raise ImportError with a
# helpful message if the required third-party library is not installed.
# ---------------------------------------------------------------------------

# trainsight.integrations.*  – import on demand
# trainsight.cost.*          – import on demand
# trainsight.optimization.*  – import on demand
# trainsight.ml.*            – import on demand
