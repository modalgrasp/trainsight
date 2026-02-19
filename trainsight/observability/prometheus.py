from __future__ import annotations

from prometheus_client import Gauge, start_http_server


class TrainSightPrometheusExporter:
    def __init__(self, port: int = 9108) -> None:
        self.port = int(port)
        self._started = False

        self.gpu_util = Gauge("trainsight_gpu_util_percent", "GPU utilization percent")
        self.vram_util = Gauge("trainsight_vram_util_percent", "VRAM utilization percent")
        self.temp_c = Gauge("trainsight_temp_celsius", "GPU temperature in Celsius")
        self.power_w = Gauge("trainsight_power_watts", "GPU power draw in Watts")

    def start(self) -> None:
        if self._started:
            return
        start_http_server(self.port)
        self._started = True

    def handle_gpu_event(self, event) -> None:
        payload = event.payload
        self.gpu_util.set(float(payload.get("gpu_util", 0.0)))
        self.vram_util.set(float(payload.get("vram", 0.0)))
        self.temp_c.set(float(payload.get("temp", 0.0)))
        self.power_w.set(float(payload.get("power", 0.0)))
