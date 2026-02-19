# TrainSight Architecture

Pipeline:

Collectors -> EventBus -> Analyzers -> Predictors -> Dashboard/CLI/Logger

Event contract:

- `type`: event channel, e.g. `gpu.stats`
- `payload`: metric dictionary
- `timestamp`: UTC timestamp

Plugin path:

- `~/.trainsight/plugins/*.py`
- module must export `register(bus)`
