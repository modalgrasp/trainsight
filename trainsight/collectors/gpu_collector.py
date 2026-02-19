from __future__ import annotations

import subprocess
import time


class GPUCollector:
    event_type = "gpu.stats"

    def __init__(self, cache_ttl: float = 0.25) -> None:
        self.cache_ttl = float(cache_ttl)
        self._last_collect_ts = 0.0
        self._last_payload: dict | None = None

    def collect(self) -> dict | None:
        now = time.monotonic()
        if self._last_payload is not None and (now - self._last_collect_ts) < self.cache_ttl:
            return dict(self._last_payload)
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=0.35,
                check=False,
            )
            if proc.returncode != 0:
                return None
            line = next((ln for ln in proc.stdout.splitlines() if ln.strip()), None)
            if not line:
                return None
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 8:
                return None

            def _f(value: str, default: float) -> float:
                s = str(value).strip().lower()
                if s in {"", "n/a", "[n/a]", "na", "nan", "none"}:
                    return default
                try:
                    return float(value)
                except Exception:
                    return default

            mem_used = _f(parts[3], 0.0)
            mem_total = max(1.0, _f(parts[4], 1.0))
            payload = {
                "index": int(_f(parts[0], 0.0)),
                "name": parts[1],
                "gpu_util": _f(parts[2], 0.0),
                "vram": (mem_used / mem_total) * 100.0,
                "mem_used": mem_used,
                "mem_total": mem_total,
                "temp": _f(parts[5], 0.0),
                "power": _f(parts[6], 0.0),
                "power_limit": _f(parts[7], 200.0),
            }
            self._last_payload = payload
            self._last_collect_ts = now
            return dict(payload)
        except Exception:
            return None
