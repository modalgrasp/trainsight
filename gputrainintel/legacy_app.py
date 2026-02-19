import asyncio
import csv
import math
import json
import subprocess
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from pynvml import *
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import Header, Static

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

try:
    import torch  # type: ignore
except Exception:
    torch = None


LOG_FILE = "gpu_usage_log.csv"
SESSION_SUMMARY_FILE = "gpu_session_summary.txt"
HISTORY_LENGTH = 60
TRAINING_KEYWORDS = (
    "python",
    "torchrun",
    "deepspeed",
    "accelerate",
    "trainer",
    "jax",
)


class BtopWaveGraph:
    def __init__(self, height=10, max_buffer=800):
        self.height = height
        self.buffer = deque(maxlen=max_buffer)
        self.show_axes = True

    def push(self, value):
        self.buffer.append(max(0.0, min(100.0, float(value))))

    def _interp_color(self, value):
        # Neon gradient stops: cyan -> green -> yellow -> orange -> red
        stops = [
            (0, (0, 220, 255)),
            (40, (60, 255, 100)),
            (70, (255, 245, 80)),
            (90, (255, 150, 60)),
            (100, (255, 60, 60)),
        ]

        for i in range(len(stops) - 1):
            v0, c0 = stops[i]
            v1, c1 = stops[i + 1]
            if value <= v1:
                t = 0.0 if v1 == v0 else (value - v0) / (v1 - v0)
                r = int(c0[0] + (c1[0] - c0[0]) * t)
                g = int(c0[1] + (c1[1] - c0[1]) * t)
                b = int(c0[2] + (c1[2] - c0[2]) * t)
                return f"rgb({r},{g},{b})"

        r, g, b = stops[-1][1]
        return f"rgb({r},{g},{b})"

    def render(self, width):
        width = max(8, int(width))

        # Reserve columns for y-axis labels/separator only in non-compact mode.
        y_label_width = 3 if self.show_axes else 0
        axis_sep = 1 if self.show_axes else 0
        plot_width = max(4, width - y_label_width - axis_sep)

        visible = list(self.buffer)[-plot_width:]
        if len(visible) < plot_width:
            visible = [0.0] * (plot_width - len(visible)) + visible

        output = Text(end="", no_wrap=True, overflow="crop")

        for row in range(self.height, 0, -1):
            if self.show_axes:
                y_val = int(round((row / self.height) * 100))
                if row == self.height or row == max(1, self.height // 2) or row == 1:
                    y_text = f"{y_val:>3}"
                else:
                    y_text = "   "
                output.append(y_text, style="bright_black")
                output.append("|", style="bright_black")

            for value in visible:
                normalized = int((value / 100) * self.height)
                if normalized >= row:
                    output.append("\u2807", style=self._interp_color(value))
                else:
                    output.append(" ")

            output.append("\n")

        if self.show_axes:
            # X-axis baseline
            output.append(" " * y_label_width, style="bright_black")
            output.append("+", style="bright_black")
            output.append("-" * plot_width, style="bright_black")
            output.append("\n")

            # X-axis labels: left / middle / right
            label_line = [" "] * (y_label_width + axis_sep + plot_width)
            for pos, label in [(0, "-30s"), (plot_width // 2 - 2, "-15s"), (plot_width - 3, "now")]:
                start_idx = y_label_width + axis_sep + max(0, pos)
                for i, ch in enumerate(label):
                    idx = start_idx + i
                    if 0 <= idx < len(label_line):
                        label_line[idx] = ch

            output.append("".join(label_line), style="bright_black")

        return output


class GPUDashboard(App):
    CSS = """
    Screen { background: #0a0a0a; }
    #main_container {
        layout: horizontal;
        height: 1fr;
    }
    #stats {
        width: 30%;
        overflow-y: auto;
    }
    #graph {
        width: 70%;
        overflow-y: auto;
    }
    """

    refresh_rate = reactive(1 / 60)

    async def on_mount(self):
        try:
            nvmlInit()
            self.nvml_initialized = True
        except Exception:
            self.nvml_initialized = False

        self.session_started = datetime.now()
        self.peak_gpu = 0.0
        self.peak_vram = 0.0
        self.peak_temp = 0.0
        self.temp_sum = 0.0
        self.temp_count = 0

        self.gpu_history = deque(maxlen=HISTORY_LENGTH)
        self.mem_history = deque(maxlen=HISTORY_LENGTH)
        self.temp_history = deque(maxlen=HISTORY_LENGTH)
        self.power_history = deque(maxlen=HISTORY_LENGTH)

        graph_height = 10
        self.gpu_wave = BtopWaveGraph(height=graph_height)
        self.vram_wave = BtopWaveGraph(height=graph_height)
        self.temp_wave = BtopWaveGraph(height=graph_height)

        self.smoothed_gpu = 0.0
        self.smoothed_vram = 0.0
        self.smoothed_temp = 0.0
        self.smoothed_power = 0.0
        self.smoothing_factor = 0.30
        self.power_cap_watts = 200.0

        self.show_training_panel = True
        self.memory_deep_mode = False
        self.debug_mode = False
        self.performance_mode = True
        self.smoothing_enabled = True
        self.compact_ml_mode = False

        self.training_stats = {
            "epoch": None,
            "steps": None,
            "loss": None,
            "lr": None,
            "grad_norm": None,
            "tokens_per_sec": None,
            "samples_per_sec": None,
            "steps_per_sec": None,
        }
        self.loss_history = deque(maxlen=50)
        self.grad_history = deque(maxlen=50)

        self._last_external_fan_probe = 0.0
        self._cached_external_fans = []
        self.current_refresh_interval = 1 / 30
        self.last_error = None
        self.last_nvml_error = None
        self.error_count = 0
        self.no_data_count = 0

        await self.boot_sequence()
        self._apply_responsive_layout()
        self.refresh_timer = self.set_interval(1 / 30, self.safe_update)
        self.call_later(self.safe_update)

    async def on_unmount(self):
        started = getattr(self, "session_started", datetime.now())
        duration = datetime.now() - started
        temp_sum = getattr(self, "temp_sum", 0.0)
        temp_count = getattr(self, "temp_count", 0)
        avg_temp = (temp_sum / temp_count) if temp_count else 0.0

        summary = (
            "Session Summary\n"
            f"Start: {started.isoformat(timespec='seconds')}\n"
            f"Duration: {duration}\n"
            f"Peak GPU: {getattr(self, 'peak_gpu', 0.0):.1f}%\n"
            f"Peak VRAM: {getattr(self, 'peak_vram', 0.0):.1f}%\n"
            f"Peak Temp: {getattr(self, 'peak_temp', 0.0):.1f}C\n"
            f"Avg Temp: {avg_temp:.1f}C\n"
        )
        Path(SESSION_SUMMARY_FILE).write_text(summary, encoding="utf-8")

    def _set_refresh(self, seconds):
        try:
            self.refresh_timer.stop()
        except Exception:
            pass
        self.current_refresh_interval = seconds
        self.refresh_timer = self.set_interval(seconds, self.safe_update)

    def update_training_stats(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.training_stats:
                self.training_stats[k] = v

        if "loss" in kwargs and kwargs["loss"] is not None:
            try:
                self.loss_history.append(float(kwargs["loss"]))
            except Exception:
                pass

        if "grad_norm" in kwargs and kwargs["grad_norm"] is not None:
            try:
                self.grad_history.append(float(kwargs["grad_norm"]))
            except Exception:
                pass

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Container(
            Static(id="stats"),
            Static(id="graph"),
            id="main_container",
        )

    def _process_meta(self, pid):
        if psutil is None:
            return {"name": f"pid:{pid}", "cmdline": ""}
        try:
            proc = psutil.Process(pid)
            name = (proc.name() or "").lower()
            cmdline = " ".join(proc.cmdline()).lower()
            return {"name": name, "cmdline": cmdline}
        except Exception:
            return {"name": f"pid:{pid}", "cmdline": ""}

    def _running_processes_for_handle(self, handle):
        procs = []
        for fn_name in (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses_v2",
            "nvmlDeviceGetComputeRunningProcesses",
            "nvmlDeviceGetGraphicsRunningProcesses_v3",
            "nvmlDeviceGetGraphicsRunningProcesses_v2",
            "nvmlDeviceGetGraphicsRunningProcesses",
        ):
            fn = globals().get(fn_name)
            if fn is None:
                continue
            try:
                got = fn(handle)
                if got:
                    procs.extend(got)
            except Exception:
                continue

        uniq = {}
        for p in procs:
            pid = int(getattr(p, "pid", 0))
            used = int(getattr(p, "usedGpuMemory", 0))
            key = (pid, used)
            uniq[key] = p
        return list(uniq.values())

    def _torch_memory_stats(self):
        if torch is None:
            return None
        try:
            if not torch.cuda.is_available():
                return None
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            fragment = max(0.0, reserved - allocated)
            return {
                "allocated": allocated,
                "reserved": reserved,
                "fragment": fragment,
            }
        except Exception:
            return None


    def _gpu_fans_for_handle(self, handle, gpu_index):
        fans = []

        # Multi-fan capable path (desktop cards / some laptops)
        num_fans_fn = globals().get("nvmlDeviceGetNumFans")
        fan_v2_fn = globals().get("nvmlDeviceGetFanSpeed_v2")
        if num_fans_fn and fan_v2_fn:
            try:
                num_fans = int(num_fans_fn(handle))
                for fan_idx in range(num_fans):
                    try:
                        pct = int(fan_v2_fn(handle, fan_idx))
                        fans.append({
                            "label": f"GPU{gpu_index} Fan{fan_idx + 1}",
                            "display": f"{pct}%",
                            "source": "nvml",
                        })
                    except Exception:
                        continue
            except Exception:
                pass

        # Fallback single-fan path
        if not fans:
            fan_fn = globals().get("nvmlDeviceGetFanSpeed")
            if fan_fn:
                try:
                    pct = int(fan_fn(handle))
                    fans.append({
                        "label": f"GPU{gpu_index} Fan",
                        "display": f"{pct}%",
                        "source": "nvml",
                    })
                except Exception:
                    pass

        return fans

    def _system_fans(self):
        fans = []
        if psutil is None:
            return fans

        try:
            fan_map = psutil.sensors_fans() or {}
        except Exception:
            fan_map = {}

        for group_name, entries in fan_map.items():
            for idx, entry in enumerate(entries):
                label = getattr(entry, "label", "") or f"{group_name}{idx + 1}"
                rpm = getattr(entry, "current", None)
                if rpm is None:
                    continue
                fans.append(
                    {
                        "label": f"{group_name}:{label}",
                        "display": f"{float(rpm):.0f} RPM",
                        "source": "psutil",
                    }
                )

        return fans


    def _nvidia_smi_fans(self):
        fans = []
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,fan.speed",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=1.5,
                check=False,
            )
            if proc.returncode != 0:
                return fans

            for line in proc.stdout.splitlines():
                if not line.strip():
                    continue
                parts = [x.strip() for x in line.split(",")]
                if len(parts) < 2:
                    continue
                idx, speed = parts[0], parts[1]
                if speed.lower() in ("n/a", "na", ""):
                    continue
                fans.append(
                    {
                        "label": f"GPU{idx} Fan",
                        "display": f"{speed}%",
                        "source": "nvidia-smi",
                    }
                )
        except Exception:
            pass
        return fans


    def _libre_hardware_monitor_fans(self):
        fans = []
        ps_cmd = (
            "Get-CimInstance -Namespace root\\LibreHardwareMonitor -ClassName Sensor -ErrorAction SilentlyContinue "
            "| Where-Object { $_.SensorType -eq 'Fan' } "
            "| Select-Object Name,Value | ConvertTo-Json -Compress"
        )
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                capture_output=True,
                text=True,
                timeout=1.8,
                check=False,
            )
            if proc.returncode != 0:
                return fans
            payload = proc.stdout.strip()
            if not payload:
                return fans
            data = json.loads(payload)
            if isinstance(data, dict):
                data = [data]
            for item in data:
                name = str(item.get("Name", "")).strip() or "LHM Fan"
                value = item.get("Value", None)
                if value is None:
                    continue
                try:
                    rpm = float(value)
                except Exception:
                    continue
                if rpm <= 0:
                    continue
                fans.append(
                    {
                        "label": name,
                        "display": f"{rpm:.0f} RPM",
                        "source": "LHM",
                    }
                )
        except Exception:
            pass
        return fans

    def _wmi_fans(self):
        fans = []

        def _query(ps_cmd, source):
            out = []
            try:
                proc = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps_cmd],
                    capture_output=True,
                    text=True,
                    timeout=1.8,
                    check=False,
                )
                if proc.returncode != 0:
                    return out
                payload = proc.stdout.strip()
                if not payload:
                    return out
                data = json.loads(payload)
                if isinstance(data, dict):
                    data = [data]
                for item in data:
                    name = str(item.get("Name", "")).strip()
                    value = item.get("Value", item.get("DesiredSpeed", None))
                    if value is None:
                        continue
                    try:
                        val_f = float(value)
                    except Exception:
                        continue
                    if val_f <= 0:
                        continue
                    out.append(
                        {
                            "label": name or source,
                            "display": f"{val_f:.0f} RPM",
                            "source": source,
                        }
                    )
            except Exception:
                return out
            return out

        fans.extend(
            _query(
                "Get-CimInstance -Namespace root\\\\LibreHardwareMonitor -ClassName Sensor -ErrorAction SilentlyContinue | Where-Object { $_.SensorType -eq 'Fan' } | Select-Object Name,Value | ConvertTo-Json -Compress",
                "LHM",
            )
        )
        fans.extend(
            _query(
                "Get-CimInstance -Namespace root\\\\OpenHardwareMonitor -ClassName Sensor -ErrorAction SilentlyContinue | Where-Object { $_.SensorType -eq 'Fan' } | Select-Object Name,Value | ConvertTo-Json -Compress",
                "OHM",
            )
        )
        fans.extend(
            _query(
                "Get-CimInstance -ClassName Win32_Fan -ErrorAction SilentlyContinue | Select-Object Name,DesiredSpeed | ConvertTo-Json -Compress",
                "WMI",
            )
        )

        return fans

    def _external_fans_cached(self):
        now = time.time()
        if now - self._last_external_fan_probe < 3.0:
            return list(self._cached_external_fans)

        fans = []
        # Priority order: LHM -> psutil -> nvidia-smi -> WMI/OHM fallback
        fans.extend(self._libre_hardware_monitor_fans())
        fans.extend(self._system_fans())
        fans.extend(self._nvidia_smi_fans())
        fans.extend(self._wmi_fans())

        seen = set()
        deduped = []
        for fan in fans:
            key = (fan.get("label", ""), fan.get("display", ""))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(fan)

        self._cached_external_fans = deduped
        self._last_external_fan_probe = now
        return list(self._cached_external_fans)

    def _gpu_stats_from_nvidia_smi(self):
        now = time.monotonic()
        cached = getattr(self, "_last_smi_stats", None)
        if cached is not None and (now - getattr(self, "_last_smi_ts", 0.0)) < 0.25:
            base = dict(cached)
            base["gpu_rows"] = [dict(r) for r in cached.get("gpu_rows", [])]
            base["training_processes"] = list(cached.get("training_processes", []))
            base["mem_categories"] = dict(cached.get("mem_categories", {}))
            return base

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
                self.last_nvml_error = f"nvidia-smi failed: {proc.stderr.strip() or proc.returncode}"
                return None

            lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
            if not lines:
                self.last_nvml_error = "nvidia-smi returned no rows"
                return None

            gpu_rows = []
            for line in lines:
                parts = [x.strip() for x in line.split(",")]
                if len(parts) < 8:
                    continue

                idx = int(float(parts[0]))
                name = parts[1]
                util = float(parts[2] or 0.0)
                mem_used = float(parts[3] or 0.0)
                mem_total = max(1.0, float(parts[4] or 1.0))
                temp = float(parts[5] or 0.0)
                def _safe_float(raw, default):
                    s = str(raw).strip()
                    if not s or s.lower() in {"n/a", "[n/a]", "na", "nan", "none"}:
                        return float(default)
                    try:
                        return float(s)
                    except Exception:
                        return float(default)

                power = _safe_float(parts[6], 0.0)
                power_limit = _safe_float(parts[7], self.power_cap_watts)

                gpu_rows.append(
                    {
                        "index": idx,
                        "name": name,
                        "gpu_util": util,
                        "mem_percent": (mem_used / mem_total) * 100.0,
                        "mem_used": mem_used,
                        "mem_total": mem_total,
                        "temp": temp,
                        "power": power,
                        "power_limit": power_limit,
                        "clock": 0,
                        "max_clock": 0,
                        "throttling": False,
                    }
                )

            if not gpu_rows:
                self.last_nvml_error = "nvidia-smi parsing failed"
                return None

            utils = [r["gpu_util"] for r in gpu_rows]
            primary = gpu_rows[0]
            primary["gpu_rows"] = gpu_rows
            primary["imbalance"] = (max(utils) - min(utils)) if len(utils) > 1 else 0.0
            primary["training_processes"] = []
            primary["active_training"] = None
            primary["mem_categories"] = {"Torch": 0.0, "Xformers": 0.0, "Other": 0.0}
            primary["process_visibility_limited"] = bool(primary.get("gpu_util", 0) > 50 and primary.get("mem_percent", 0) > 20)
            primary["torch_mem"] = None
            primary["torch_mem_estimated"] = None
            self.last_nvml_error = None
            self._last_smi_stats = primary
            self._last_smi_ts = now
            return primary
        except Exception as exc:
            self.last_nvml_error = f"nvidia-smi exception: {exc}"
            return None

    def get_gpu_stats(self):
        if not getattr(self, "nvml_initialized", False):
            self.last_nvml_error = "NVML initialization failed; using nvidia-smi fallback"
            return self._gpu_stats_from_nvidia_smi()

        try:
            count = nvmlDeviceGetCount()
            if count < 1:
                self.last_nvml_error = "No NVIDIA GPUs from NVML; using nvidia-smi fallback"
                return self._gpu_stats_from_nvidia_smi()
        except Exception as exc:
            self.last_nvml_error = f"NVML device query failed ({exc}); using nvidia-smi fallback"
            return self._gpu_stats_from_nvidia_smi()

        gpu_rows = []
        training_processes = []
        primary = None

        for idx in range(count):
            try:
                handle = nvmlDeviceGetHandleByIndex(idx)
                name = nvmlDeviceGetName(handle).decode()
                util = nvmlDeviceGetUtilizationRates(handle)
                mem = nvmlDeviceGetMemoryInfo(handle)
                temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
                power = nvmlDeviceGetPowerUsage(handle) / 1000
                try:
                    power_limit = nvmlDeviceGetPowerManagementLimit(handle) / 1000
                except Exception:
                    power_limit = self.power_cap_watts

                mem_percent = (mem.used / mem.total) * 100
                cur_clock = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_SM)
                max_clock = nvmlDeviceGetMaxClockInfo(handle, NVML_CLOCK_SM)
                throttling = bool(max_clock > 0 and cur_clock < (0.9 * max_clock))

                row = {
                    "index": idx,
                    "name": name,
                    "gpu_util": float(util.gpu),
                    "mem_percent": float(mem_percent),
                    "mem_used": mem.used / 1024**2,
                    "mem_total": mem.total / 1024**2,
                    "temp": float(temp),
                    "power": float(power),
                    "power_limit": float(power_limit),
                    "clock": int(cur_clock),
                    "max_clock": int(max_clock),
                    "throttling": throttling,
                }
                gpu_rows.append(row)

                if idx == 0:
                    primary = row

                for proc in self._running_processes_for_handle(handle):
                    pid = int(getattr(proc, "pid", 0))
                    used = int(getattr(proc, "usedGpuMemory", 0))
                    # NVML may return sentinel values when memory is unavailable.
                    if used < 0 or used > (1 << 62):
                        used = 0
                    mem_gb = max(0.0, float(used) / (1024 ** 3))
                    meta = self._process_meta(pid)
                    training_processes.append(
                        {
                            "pid": pid,
                            "name": meta["name"],
                            "cmdline": meta["cmdline"],
                            "gpu": idx,
                            "mem_gb": mem_gb,
                        }
                    )
            except Exception:
                continue

        if not gpu_rows:
            self.last_nvml_error = "NVML returned no readable GPU rows; using nvidia-smi fallback"
            return self._gpu_stats_from_nvidia_smi()

        utils = [r["gpu_util"] for r in gpu_rows]
        imbalance = (max(utils) - min(utils)) if len(utils) > 1 else 0.0

        # Deduplicate by PID while preserving max memory observed across GPUs.
        by_pid = {}
        for p in training_processes:
            pid = p["pid"]
            if pid not in by_pid or p["mem_gb"] > by_pid[pid]["mem_gb"]:
                by_pid[pid] = p
        training_processes = sorted(by_pid.values(), key=lambda x: x["mem_gb"], reverse=True)

        # Process memory categories (name/cmdline proxy)
        mem_categories = {"Torch": 0.0, "Xformers": 0.0, "Other": 0.0}
        for p in training_processes:
            blob = f"{p['name']} {p.get('cmdline', '')}"
            if "xformers" in blob:
                mem_categories["Xformers"] += p["mem_gb"]
            elif any(k in blob for k in ("python", "torch", "deepspeed", "accelerate", "trainer", "transformers")):
                mem_categories["Torch"] += p["mem_gb"]
            else:
                mem_categories["Other"] += p["mem_gb"]

        active_training = None
        for p in training_processes:
            blob = f"{p['name']} {p.get('cmdline', '')}"
            if any(k in blob for k in TRAINING_KEYWORDS):
                active_training = p
                break

        # Fallback: pick heaviest GPU-memory process if explicit keyword match fails.
        if active_training is None and training_processes:
            active_training = training_processes[0]

        process_visibility_limited = (not training_processes and primary.get("gpu_util", 0) > 70 and primary.get("mem_percent", 0) > 30)

        # If process telemetry is hidden (common on Windows laptop drivers),
        # estimate training memory from total GPU memory use.
        if process_visibility_limited and mem_categories["Torch"] == 0.0:
            mem_categories["Torch"] = max(0.0, primary.get("mem_used", 0.0) / 1024.0)

        torch_mem = self._torch_memory_stats()
        if torch_mem and (torch_mem["allocated"] <= 0 and torch_mem["reserved"] <= 0):
            torch_mem = None

        torch_mem_estimated = None
        if torch_mem is None and mem_categories["Torch"] > 0:
            alloc = mem_categories["Torch"]
            reserved = max(alloc, primary.get("mem_used", 0.0) / 1024.0)
            torch_mem_estimated = {
                "allocated": alloc,
                "reserved": reserved,
                "fragment": max(0.0, reserved - alloc),
            }

        self.last_nvml_error = None
        primary["gpu_rows"] = gpu_rows
        primary["imbalance"] = imbalance
        primary["training_processes"] = training_processes
        primary["active_training"] = active_training
        primary["mem_categories"] = mem_categories
        primary["process_visibility_limited"] = process_visibility_limited
        primary["torch_mem"] = torch_mem
        primary["torch_mem_estimated"] = torch_mem_estimated

        return primary

    def log_stats(self, stats):
        try:
            with open(LOG_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        datetime.now().isoformat(),
                        stats["gpu_util"],
                        stats["mem_percent"],
                        stats["temp"],
                        stats["power"],
                    ]
                )
        except Exception as exc:
            self.last_error = f"Failed to write log file: {exc}"

    def detect_anomalies(self):
        alerts = []

        if len(self.gpu_history) > 10:
            delta = self.gpu_history[-1] - self.gpu_history[-10]
            if delta < -40:
                alerts.append("GPU Utilization Drop")

        if len(self.temp_history) > 10:
            temp_delta = self.temp_history[-1] - self.temp_history[-10]
            if temp_delta > 8:
                alerts.append("Rapid Thermal Rise")

        if len(self.power_history) > 10:
            power_delta = self.power_history[-1] - self.power_history[-10]
            if power_delta > 25:
                alerts.append("Power Spike")

        if self.smoothed_temp > 88:
            alerts.append("Critical Temperature")

        return alerts

    def predict_temp(self, seconds_ahead=15):
        if len(self.temp_history) < 5:
            return None

        slope = (
            self.temp_history[-1] - self.temp_history[-5]
        ) / (5 * self.current_refresh_interval)

        prediction = self.smoothed_temp + slope * seconds_ahead
        return prediction

    def predict_oom(self):
        if len(self.mem_history) < 10:
            return None

        growth = self.mem_history[-1] - self.mem_history[-10]

        if self.smoothed_vram > 92 and growth > 3:
            return "High OOM Risk"

        if self.smoothed_vram > 96:
            return "Critical OOM Imminent"

        return None

    def detect_dataloader_bottleneck(self):
        if self.training_stats["tokens_per_sec"] is None:
            return None

        if (
            self.smoothed_gpu < 60
            and self.smoothed_vram > 50
            and self.training_stats["tokens_per_sec"] < 200
        ):
            return "Dataloader Bottleneck Suspected"

        return None

    def compute_efficiency_score(self):
        score = 0.0

        # GPU utilization
        score += min(40.0, self.smoothed_gpu * 0.4)

        # VRAM utilization
        score += min(20.0, self.smoothed_vram * 0.2)

        # Thermal headroom
        headroom = max(0.0, 100.0 - self.smoothed_temp)
        score += min(20.0, headroom * 0.2)

        # Power efficiency (if tokens available)
        if self.training_stats["tokens_per_sec"] and self.smoothed_power > 0:
            eff = self.training_stats["tokens_per_sec"] / self.smoothed_power
            score += min(20.0, eff * 0.5)

        return min(100.0, score)

    def detect_gradient_explosion(self):
        if len(self.loss_history) >= 10:
            recent = list(self.loss_history)[-10:]
            mean = sum(recent) / len(recent)
            variance = sum((x - mean) ** 2 for x in recent) / len(recent)
            if variance > max(0.01, mean * 0.5):
                return "High Loss Variance"

        if self.grad_history and max(self.grad_history) > 10:
            return "Gradient Explosion"

        return None

    def suggest_batch_adjustment(self):
        if self.smoothed_vram > 95:
            return "Decrease Batch Size"
        if self.smoothed_vram < 60 and self.smoothed_gpu < 80:
            return "Increase Batch Size"
        return None

    def power_efficiency_recommendation(self):
        tok = self.training_stats.get("tokens_per_sec")
        if tok is None:
            return None

        if self.smoothed_power > 0:
            eff = float(tok) / self.smoothed_power
            if eff < 1.5:
                return "Consider lowering power cap"
            if eff > 3:
                return "Power usage optimal"

        return None

    def analyze_nccl_imbalance(self, imbalance):
        if imbalance > 40:
            return "Severe GPU Imbalance"
        if imbalance > 20:
            return "Moderate Load Imbalance"
        return None

    def detect_mixed_precision_issue(self):
        if len(self.loss_history) < 5:
            return None

        if len(self.grad_history) >= 5 and max(list(self.grad_history)[-5:]) > 8:
            return "Possible FP16 Instability"

        return None

    def predict_next_loss(self):
        if len(self.loss_history) < 5:
            return None

        xs = list(range(len(self.loss_history)))
        ys = list(self.loss_history)

        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n

        denom = sum((x - mean_x) ** 2 for x in xs)
        if denom == 0:
            return None

        slope = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n)) / denom
        next_x = n
        predicted = mean_y + slope * (next_x - mean_x)
        return predicted

    def compute_oom_probability(self):
        if len(self.mem_history) < 20:
            return 0.0

        recent = list(self.mem_history)[-20:]
        growth = recent[-1] - recent[0]
        utilization = self.smoothed_vram / 100.0
        risk = 0.6 * utilization + 0.4 * (growth / 20.0)
        return max(0.0, min(1.0, risk))

    def classify_bottleneck(self):
        power_pct = (self.smoothed_power / max(1.0, self.power_cap_watts)) * 100.0

        if self.smoothed_gpu < 50 and self.smoothed_vram > 60:
            return "CPU/Dataloader Bound"
        if self.smoothed_gpu > 90 and power_pct > 80:
            return "Model Compute Bound"
        if self.smoothed_gpu < 40 and power_pct < 50:
            return "IO Bound"

        return None

    def _oom_yes_no(self):
        return (self.predict_oom() is not None) or (self.compute_oom_probability() > 0.7)

    def build_compact_ml_panel(self, stats):
        table = Table.grid(padding=1)

        table.add_row("GPU", f"{stats['gpu_util']:.1f}%")
        table.add_row("VRAM", f"{stats['mem_percent']:.1f}%")
        table.add_row("Temp", f"{stats['temp']:.0f}C")
        table.add_row("Power", f"{stats['power']:.1f}W")

        active = stats.get("active_training")
        if active:
            table.add_row("PID", str(active["pid"]))

        if self.training_stats["loss"] is not None:
            table.add_row("Loss", str(self.training_stats["loss"]))

        if self.training_stats["tokens_per_sec"] is not None:
            table.add_row("Tok/s", str(self.training_stats["tokens_per_sec"]))

            if stats["power"] > 0:
                eff = float(self.training_stats["tokens_per_sec"]) / stats["power"]
                table.add_row("Eff", f"{eff:.2f} tok/W")

        if self.training_stats["steps_per_sec"] is not None:
            table.add_row("Steps/s", str(self.training_stats["steps_per_sec"]))

        headroom = 100 - stats["temp"]
        table.add_row("Thermal Headroom", f"{headroom:.0f}%")

        oom_flag = "Yes" if self._oom_yes_no() else "No"
        bottleneck = self.classify_bottleneck()
        table.add_row("OOM Soon", oom_flag)
        table.add_row("Bottleneck Soon", "Yes" if bottleneck else "No")

        prediction = self.predict_temp()
        if prediction is not None:
            table.add_row("Temp +15s", f"{prediction:.1f}C")

        prob = self.compute_oom_probability()
        table.add_row("OOM Prob", f"{prob * 100:.0f}%")

        next_loss = self.predict_next_loss()
        if next_loss is not None:
            table.add_row("Next Loss", f"{next_loss:.4f}")

        table.add_row("[bright_black]Keys[/bright_black]", "q quit | c compact | t/g/m/p/d toggles")

        return Panel(
            table,
            title="[bold green]ML Compact Mode[/bold green]",
            border_style="green",
            subtitle=self.build_watermark_subtitle(),
            subtitle_align="center",
        )

    def build_watermark_subtitle(self):
        return "[bright_black]Developed by Pratham Patel[/bright_black]"

    def build_sparkline(self, data, color):
        blocks = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
        spark = ""
        for v in list(data)[-30:]:
            idx = min(7, max(0, math.floor((v / 100) * 7)))
            spark += blocks[idx]
        return Text(spark, style=color)

    def build_stats_panel(self, stats):
        if self.compact_ml_mode:
            return self.build_compact_ml_panel(stats)

        temp_style = "bold red" if stats["temp"] > 80 else "yellow"

        table = Table.grid(padding=1)
        table.add_row("[cyan]GPU Usage[/cyan]", f"[bright_green]{stats['gpu_util']:.1f}%[/bright_green]")
        table.add_row(
            "[magenta]VRAM[/magenta]",
            f"{stats['mem_used']:.0f}/{stats['mem_total']:.0f} MB ({stats['mem_percent']:.1f}%)",
        )
        table.add_row("[yellow]Temp[/yellow]", f"[{temp_style}]{stats['temp']:.0f} C[/{temp_style}]")
        table.add_row("[red]Power[/red]", f"{stats['power']:.1f}/{stats['power_limit']:.0f} W")

        if len(self.temp_history) >= 2:
            window = min(len(self.temp_history) - 1, 30)
            dt = max(0.001, window * float(getattr(self, "current_refresh_interval", 1 / 30)))
            slope = (self.temp_history[-1] - self.temp_history[-1 - window]) / dt * 10.0
            table.add_row("[cyan]Cooling Slope[/cyan]", f"{slope:+.2f} C/10s")

        if stats["throttling"]:
            table.add_row("[red]Throttle[/red]", "[bold red]THERMAL THROTTLING[/bold red]")

        if stats["imbalance"] > 20:
            table.add_row("[red]NCCL Imbalance[/red]", f"[bold red]{stats['imbalance']:.1f}%[/bold red]")

        nccl_analysis = self.analyze_nccl_imbalance(stats["imbalance"])
        if nccl_analysis:
            table.add_row("[yellow]NCCL Analysis[/yellow]", nccl_analysis)

        oom_alert = self.predict_oom()
        if oom_alert:
            table.add_row("[red]Memory Alert[/red]", f"[bold red]{oom_alert}[/bold red]")

        oom_prob = self.compute_oom_probability()
        table.add_row("[cyan]OOM Soon[/cyan]", "[bold red]Yes[/bold red]" if self._oom_yes_no() else "No")
        if oom_prob > 0.7:
            table.add_row("[red]OOM Risk[/red]", f"{oom_prob * 100:.0f}%")

        dl_alert = self.detect_dataloader_bottleneck()
        if dl_alert:
            table.add_row("[yellow]Pipeline[/yellow]", dl_alert)

        bottleneck = self.classify_bottleneck()
        table.add_row("[cyan]Bottleneck Soon[/cyan]", "[bold yellow]Yes[/bold yellow]" if bottleneck else "No")
        if bottleneck:
            table.add_row("[yellow]Bottleneck[/yellow]", bottleneck)

        score = self.compute_efficiency_score()
        color = "green"
        if score < 60:
            color = "red"
        elif score < 80:
            color = "yellow"
        table.add_row("[cyan]Training Score[/cyan]", f"[{color}]{score:.0f}/100[/{color}]")

        batch_suggestion = self.suggest_batch_adjustment()
        if batch_suggestion:
            table.add_row("[cyan]Batch Advice[/cyan]", batch_suggestion)

        power_tuning = self.power_efficiency_recommendation()
        if power_tuning:
            table.add_row("[cyan]Power Advice[/cyan]", power_tuning)

        explosion = self.detect_gradient_explosion()
        if explosion:
            table.add_row("[red]Stability[/red]", explosion)

        fp16_issue = self.detect_mixed_precision_issue()
        if fp16_issue:
            table.add_row("[yellow]Precision[/yellow]", fp16_issue)

        next_loss = self.predict_next_loss()
        if next_loss is not None:
            table.add_row("[cyan]Next Loss[/cyan]", f"{next_loss:.4f}")

        if self.show_training_panel:
            active = stats.get("active_training")
            if active:
                table.add_row("[cyan]Active Training[/cyan]", f"PID {active['pid']} GPU{active.get('gpu', 0)} ({active['mem_gb']:.1f} GB)")
            else:
                if stats.get("process_visibility_limited"):
                    table.add_row("[cyan]Active Training[/cyan]", "Likely active (driver visibility limited)")
                else:
                    table.add_row("[cyan]Active Training[/cyan]", "None")

            cats = stats.get("mem_categories", {})
            training_visible = bool(stats.get("training_processes")) or stats.get("process_visibility_limited")

            def _mem_cell(v):
                if v > 0:
                    return f"{v:.1f} GB"
                return "N/A" if not training_visible else "0.0 GB"

            table.add_row("[magenta]Torch[/magenta]", _mem_cell(cats.get('Torch', 0.0)))
            table.add_row("[magenta]Xformers[/magenta]", _mem_cell(cats.get('Xformers', 0.0)))
            table.add_row("[magenta]Other[/magenta]", _mem_cell(cats.get('Other', 0.0)))

            if self.memory_deep_mode:
                for proc in stats.get("training_processes", [])[:10]:
                    table.add_row("[bright_black]Proc[/bright_black]", f"{proc['name']} ({proc['pid']}) {proc['mem_gb']:.2f} GB")

            tmem = stats.get("torch_mem")
            tmem_est = stats.get("torch_mem_estimated")
            if tmem:
                table.add_row("[yellow]Allocated[/yellow]", f"{tmem['allocated']:.2f} GB")
                table.add_row("[yellow]Reserved[/yellow]", f"{tmem['reserved']:.2f} GB")
                table.add_row("[yellow]Fragment[/yellow]", f"{tmem['fragment']:.2f} GB")
            elif tmem_est:
                table.add_row("[yellow]Allocated[/yellow]", f"~{tmem_est['allocated']:.2f} GB")
                table.add_row("[yellow]Reserved[/yellow]", f"~{tmem_est['reserved']:.2f} GB")
                table.add_row("[yellow]Fragment[/yellow]", f"~{tmem_est['fragment']:.2f} GB")
            else:
                table.add_row("[yellow]Allocated[/yellow]", "N/A")
                table.add_row("[yellow]Reserved[/yellow]", "N/A")
                table.add_row("[yellow]Fragment[/yellow]", "N/A")

            if self.training_stats["epoch"] is not None:
                table.add_row("[green]Epoch[/green]", str(self.training_stats["epoch"]))
            if self.training_stats["steps"] is not None:
                table.add_row("[green]Step[/green]", str(self.training_stats["steps"]))
            if self.training_stats["loss"] is not None:
                table.add_row("[green]Loss[/green]", str(self.training_stats["loss"]))
            if self.training_stats["lr"] is not None:
                table.add_row("[green]LR[/green]", str(self.training_stats["lr"]))
            if self.training_stats["grad_norm"] is not None:
                table.add_row("[green]Grad Norm[/green]", str(self.training_stats["grad_norm"]))
            if self.training_stats["tokens_per_sec"] is not None:
                table.add_row("[green]Tok/s[/green]", str(self.training_stats["tokens_per_sec"]))
            if self.training_stats["samples_per_sec"] is not None:
                table.add_row("[green]Samples/s[/green]", str(self.training_stats["samples_per_sec"]))
            if self.training_stats["steps_per_sec"] is not None:
                table.add_row("[green]Steps/s[/green]", str(self.training_stats["steps_per_sec"]))

            if self.training_stats["tokens_per_sec"] is not None and stats["power"] > 0:
                eff = float(self.training_stats["tokens_per_sec"]) / float(stats["power"])
                table.add_row("[cyan]Eff[/cyan]", f"{eff:.2f} tok/s/W")

        mode_live = f"G:{'ON' if self.smoothing_enabled else 'OFF'} M:{'DEEP' if self.memory_deep_mode else 'BASIC'} P:{'FAST' if self.performance_mode else 'ECO'}"
        table.add_row("[bright_black]Modes[/bright_black]", mode_live)

        if self.debug_mode:
            mode_flags = f"T:{int(self.show_training_panel)} G:{int(self.smoothing_enabled)} M:{int(self.memory_deep_mode)} P:{int(self.performance_mode)} D:{int(self.debug_mode)}"
            table.add_row("[bright_black]Debug[/bright_black]", mode_flags)

        # Multi-GPU per-device quick status
        for row in stats.get("gpu_rows", []):
            table.add_row(
                f"[cyan]GPU{row['index']}[/cyan]",
                f"{row['gpu_util']:.0f}% {row['mem_used'] / 1024:.1f}GB {row['temp']:.0f}C",
            )

        if (
            stats["gpu_util"] >= 95
            and stats["mem_percent"] >= 95
            and ((stats["power"] / max(1.0, stats["power_limit"])) * 100.0) >= 95
        ):
            table.add_row("[red]Alert[/red]", "[bold red]SYSTEM MAXED OUT[/bold red]")

        alerts = self.detect_anomalies()
        for alert in alerts:
            table.add_row("[red]Alert[/red]", f"[bold red]{alert}[/bold red]")

        prediction = self.predict_temp()
        if prediction is not None:
            table.add_row("[cyan]Temp +15s[/cyan]", f"{prediction:.1f}C")

        spark_gpu = self.build_sparkline(self.gpu_history, "green")
        spark_vram = self.build_sparkline(self.mem_history, "magenta")

        content = Group(
            table,
            Text("GPU Trend", style="green"),
            spark_gpu,
            Text("VRAM Trend", style="magenta"),
            spark_vram,
        )

        return Panel(
            content,
            title=f"[bold bright_cyan]{stats['name']}[/bold bright_cyan]",
            border_style="bright_magenta",
            subtitle=self.build_watermark_subtitle(),
            subtitle_align="center",
        )

    def _graph_width(self):
        graph_widget = self.query_one("#graph", Static)
        return max(10, graph_widget.size.width - 8)


    def _apply_responsive_layout(self):
        container = self.query_one("#main_container", Container)
        stats = self.query_one("#stats", Static)
        graph = self.query_one("#graph", Static)

        # Horizontal responsiveness
        if self.size.width < 150:
            container.styles.layout = "vertical"
            stats.styles.width = "1fr"
            graph.styles.width = "1fr"
        else:
            container.styles.layout = "horizontal"
            stats.styles.width = "30%"
            graph.styles.width = "70%"

        # Vertical responsiveness
        if self.size.height < 34:
            stats.styles.height = "48%"
            graph.styles.height = "52%"
        elif self.size.height < 46:
            stats.styles.height = "42%"
            graph.styles.height = "58%"
        else:
            stats.styles.height = "1fr"
            graph.styles.height = "1fr"

    def _sync_graph_heights(self):
        graph_widget = self.query_one("#graph", Static)

        available = max(10, graph_widget.size.height - 2)  # leave safety margin
        panel_count = 3

        compact = available < 40
        self.gpu_wave.show_axes = not compact
        self.vram_wave.show_axes = not compact
        self.temp_wave.show_axes = not compact

        # real safe overhead calculation
        border_overhead = 3   # top border + title + bottom border
        axis_overhead = 2 if not compact else 0

        usable = available - panel_count * (border_overhead + axis_overhead)

        each = max(3, usable // panel_count)

        self.gpu_wave.height = each
        self.vram_wave.height = each
        self.temp_wave.height = each

        # Keep compatibility with adaptive panel selection path.
        self.visible_graphs = ["gpu", "vram", "temp"]


    def build_graph_panel(self):
        if self.compact_ml_mode:
            return ""

        width = self._graph_width()

        panels = []

        if not hasattr(self, "visible_graphs"):
            return ""

        if "gpu" in self.visible_graphs:
            panels.append(
                Panel(
                    self.gpu_wave.render(width),
                    title=f" GPU {self.smoothed_gpu:.1f}% ",
                    border_style="bright_cyan",
                )
            )

        if "vram" in self.visible_graphs:
            panels.append(
                Panel(
                    self.vram_wave.render(width),
                    title=f" VRAM {self.smoothed_vram:.1f}% ",
                    border_style="magenta",
                )
            )

        if "temp" in self.visible_graphs:
            panels.append(
                Panel(
                    self.temp_wave.render(width),
                    title=f" TEMP {self.smoothed_temp:.0f}C ",
                    border_style="yellow",
                )
            )

        if not panels:
            return Panel(
                Text("Graph hidden (terminal too small)", style="bright_black"),
                border_style="red",
            )

        return Group(*panels)


    async def boot_sequence(self):
        for i in range(0, 101, 10):
            self.query_one("#stats", Static).update(f"Initializing GPU Engine... {i}%")
            await asyncio.sleep(0.1)

    def _render_status(self, title, message, hint=""):
        body = Text()
        body.append(message, style="bold red")
        if hint:
            body.append("\n\n" + hint, style="bright_black")

        self.query_one("#stats", Static).update(
            Panel(body, title=title, border_style="red")
        )
        self.query_one("#graph", Static).update(
            Panel(Text("Waiting for telemetry...", style="bright_black"), border_style="bright_black")
        )

    def safe_update(self):
        try:
            self.update_dashboard()
        except Exception as exc:
            self.error_count += 1
            self.last_error = f"Update failed: {exc}"
            self._render_status(
                "Runtime Error",
                self.last_error,
                "Press q to quit. Dashboard will retry automatically.",
            )

    def update_dashboard(self):
        stats = self.get_gpu_stats()
        if not stats:
            self.no_data_count += 1
            if self.no_data_count >= 2:
                reason = self.last_nvml_error or "GPU telemetry unavailable"
                self._render_status(
                    "Telemetry Unavailable",
                    reason,
                    "If training is running, check NVIDIA driver/NVML visibility in this environment.",
                )
            return

        self.no_data_count = 0

        # EMA smoothing for subpixel-stable motion.
        a = self.smoothing_factor if self.smoothing_enabled else 1.0
        self.smoothed_gpu = a * stats["gpu_util"] + (1 - a) * self.smoothed_gpu
        self.smoothed_vram = a * stats["mem_percent"] + (1 - a) * self.smoothed_vram
        self.smoothed_temp = a * stats["temp"] + (1 - a) * self.smoothed_temp
        self.smoothed_power = a * stats["power"] + (1 - a) * self.smoothed_power

        stats["gpu_util"] = self.smoothed_gpu
        stats["mem_percent"] = self.smoothed_vram
        stats["temp"] = self.smoothed_temp
        stats["power"] = self.smoothed_power

        self.log_stats(stats)

        power_pct = min(100.0, (self.smoothed_power / self.power_cap_watts) * 100.0)

        self.gpu_wave.push(self.smoothed_gpu)
        self.vram_wave.push(self.smoothed_vram)
        self.temp_wave.push(min(100.0, self.smoothed_temp))

        self.gpu_history.append(self.smoothed_gpu)
        self.mem_history.append(self.smoothed_vram)
        self.temp_history.append(self.smoothed_temp)
        self.power_history.append(power_pct)

        self.peak_gpu = max(self.peak_gpu, self.smoothed_gpu)
        self.peak_vram = max(self.peak_vram, self.smoothed_vram)
        self.peak_temp = max(self.peak_temp, self.smoothed_temp)
        self.temp_sum += self.smoothed_temp
        self.temp_count += 1

        self._apply_responsive_layout()
        self._sync_graph_heights()

        stats_panel = self.build_stats_panel(stats)
        graph_panel = self.build_graph_panel()

        self.query_one("#stats", Static).update(stats_panel)
        self.query_one("#graph", Static).update(graph_panel)
        self.last_error = None

    async def on_key(self, event: events.Key):
        key = str(event.key).lower()

        if key == "q":
            await self.action_quit()
            return

        if key == "t":
            self.show_training_panel = not self.show_training_panel
        elif key == "g":
            self.smoothing_enabled = not self.smoothing_enabled
        elif key == "m":
            self.memory_deep_mode = not self.memory_deep_mode
        elif key == "p":
            # Force realtime mode across terminals/OSes.
            self.performance_mode = True
            self._set_refresh(1 / 30)
        elif key == "d":
            self.debug_mode = not self.debug_mode
        elif key == "c":
            self.compact_ml_mode = not self.compact_ml_mode

        self.safe_update()

    async def on_resize(self, event: events.Resize):
        self._apply_responsive_layout()
        self.call_later(self.safe_update)


if __name__ == "__main__":
    GPUDashboard().run()


