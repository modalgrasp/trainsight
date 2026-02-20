from .config.loader import load_config
from .dashboard import Dashboard


def main() -> None:
    config = load_config()
    app = Dashboard(config=config)
    app.run()
