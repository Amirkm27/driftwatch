"""
Collects system health metrics using psutil (+ GPU via pynvml/pyadl) and
writes them to SQLite. Disk/network I/O are converted from psutil's
cumulative counters into per-second rates at collection time.
"""
import os
import time
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime, timezone

import psutil

from src.database import init_db, insert_metric, TOP_N_PROCESSES

# --- Logging setup ---
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("driftwatch.collector")
logger.setLevel(logging.INFO)

_console_handler = logging.StreamHandler()
_file_handler = RotatingFileHandler(LOG_DIR / "collector.log", maxBytes=5_000_000, backupCount=3)
_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_console_handler.setFormatter(_formatter)
_file_handler.setFormatter(_formatter)
_EXCLUDED_PROCESS_NAMES = {"System Idle Process", "Idle"}  # Windows/Linux idle placeholders

if not logger.handlers:
    logger.addHandler(_console_handler)
    logger.addHandler(_file_handler)


# =====================================================================
# GPU BACKEND DETECTION — NVIDIA (pynvml) primary, AMD (pyadl) fallback
# =====================================================================
_GPU_BACKEND = None
_NVML_HANDLE = None
_AMD_DEVICE = None

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _GPU_BACKEND = "nvidia"
    logger.info("GPU backend detected: NVIDIA (pynvml)")
except Exception:
    try:
        from pyadl import ADLManager
        devices = ADLManager.getInstance().getDevices()
        if devices:
            _AMD_DEVICE = devices[0]
            _GPU_BACKEND = "amd"
            logger.info("GPU backend detected: AMD (pyadl)")
    except Exception:
        logger.warning("No GPU backend available — gpu_* fields will be None.")


def get_gpu_metrics() -> tuple[float | None, float | None, float | None]:
    """
    Returns (gpu_percent, gpu_memory_percent, gpu_temp).
    Never raises — a flaky GPU read must not kill a collection cycle.
    """
    if _GPU_BACKEND == "nvidia":
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
            mem = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
            temp = pynvml.nvmlDeviceGetTemperature(_NVML_HANDLE, pynvml.NVML_TEMPERATURE_GPU)
            mem_percent = round((mem.used / mem.total) * 100, 2) if mem.total else None
            return float(util.gpu), mem_percent, float(temp)
        except Exception as e:
            logger.debug(f"NVIDIA GPU read failed: {e}")
            return None, None, None

    elif _GPU_BACKEND == "amd":
        try:
            usage = _AMD_DEVICE.getCurrentUsage()
            temp = _AMD_DEVICE.getCurrentTemperature()
            # pyadl has no standardized memory-usage call across devices
            mem_percent = None
            return float(usage), mem_percent, float(temp)
        except Exception as e:
            logger.debug(f"AMD GPU read failed: {e}")
            return None, None, None

    return None, None, None


def _get_cpu_temp_hardview() -> float | None:
    """
    Primary CPU temp source: HardView's PyTempCpu (wraps LibreHardwareMonitorLib
    on Windows). Uses get_avg_temp() specifically — averaged across cores,
    which behaves as a proper time-series signal (rises/falls with real
    thermal load) unlike get_max_temp() (ratchets upward, never resets
    until restart) or get_temp() (single-core snapshot, noisy).
    Requires Administrator privileges.
    """
    try:
        import HardView
        temp_cpu = HardView.PyTempCpu()
        temp_cpu.update()
        temp = temp_cpu.get_avg_temp()
        if temp is not None:
            return round(float(temp), 1)
    except Exception as e:
        logger.debug(f"HardView CPU temp read failed: {e}")
    return None


def get_cpu_temp() -> float | None:
    """
    Three-tier CPU temp fallback, best source first:
      1. HardView (bundles LibreHardwareMonitorLib on Windows / lm-sensors
         on Linux — no external process to manage, but requires admin)
      2. Windows ACPI thermal zone via WMI (built-in, OEM-dependent, often fails)
      3. psutil sensors_temperatures() (Linux only, no-op on Windows)
    Returns None if all tiers fail — a real hardware/driver limitation,
    not a bug, and worth documenting rather than chasing further.
    """
    temp = _get_cpu_temp_hardview()
    if temp is not None:
        return temp

    if os.name == "nt":
        try:
            import wmi
            w = wmi.WMI(namespace="root\\wmi")
            thermal_info = w.MSAcpi_ThermalZoneTemperature()[0]
            temp_celsius = (thermal_info.CurrentTemperature / 10.0) - 273.15
            return round(temp_celsius, 1)
        except Exception as e:
            logger.debug(f"Windows ACPI CPU temp unavailable: {e}")

    try:
        temps = psutil.sensors_temperatures()
        if temps:
            first_group = next(iter(temps.values()))
            if first_group:
                return first_group[0].current
    except (AttributeError, NotImplementedError):
        pass

    return None


def get_disk_usage_percent() -> float | None:
    """Returns % capacity used on the system drive."""
    try:
        root = os.path.abspath(os.sep)
        return psutil.disk_usage(root).percent
    except Exception as e:
        logger.debug(f"disk_usage read failed: {e}")
        return None


# =====================================================================
# RATE CALCULATION — converts psutil's cumulative counters into MB/s
# =====================================================================
_prev_io_sample = None  # {"time", "disk_read", "disk_write", "net_sent", "net_recv"} in bytes


def _safe_rate_mb_s(current_bytes, prev_bytes, elapsed_seconds) -> float | None:
    """
    Computes MB/s between two cumulative byte counts. Returns None if
    inputs are missing, elapsed time is non-positive/implausible, or the
    counter appears to have reset (delta negative).

    The upper bound on elapsed_seconds guards against sleep/suspend edge
    cases where the monotonic clock's behavior during suspend is platform-
    dependent — an elapsed window far larger than the collector's own
    interval means a real-world gap (sleep, missed cycles) occurred, and
    computing a rate across it would be misleading, not informative.
    """
    if current_bytes is None or prev_bytes is None or elapsed_seconds <= 0:
        return None
    if elapsed_seconds > 300:  # >5x a typical 30-60s interval = treat as a gap, not a rate
        return None
    delta_bytes = current_bytes - prev_bytes
    if delta_bytes < 0:
        return None
    return round((delta_bytes / (1024 ** 2)) / elapsed_seconds, 4)


def get_io_rates(disk_io, net_io) -> dict:
    """
    Converts cumulative disk/network counters into per-second rates by
    comparing against the previous snapshot's readings. First call ever
    (no previous sample) returns None for all four — expected and fine,
    that row just has no rate data yet.
    """
    global _prev_io_sample
    now = time.monotonic()

    current = {
        "time": now,
        "disk_read": disk_io.read_bytes if disk_io else None,
        "disk_write": disk_io.write_bytes if disk_io else None,
        "net_sent": net_io.bytes_sent if net_io else None,
        "net_recv": net_io.bytes_recv if net_io else None,
    }

    rates: dict[str, float | None] = {
        "disk_read_mb_s": None,
        "disk_write_mb_s": None,
        "net_sent_mb_s": None,
        "net_recv_mb_s": None,
    }

    if _prev_io_sample is not None:
        elapsed = current["time"] - _prev_io_sample["time"]
        rates["disk_read_mb_s"] = _safe_rate_mb_s(current["disk_read"], _prev_io_sample["disk_read"], elapsed)
        rates["disk_write_mb_s"] = _safe_rate_mb_s(current["disk_write"], _prev_io_sample["disk_write"], elapsed)
        rates["net_sent_mb_s"] = _safe_rate_mb_s(current["net_sent"], _prev_io_sample["net_sent"], elapsed)
        rates["net_recv_mb_s"] = _safe_rate_mb_s(current["net_recv"], _prev_io_sample["net_recv"], elapsed)

    _prev_io_sample = current
    return rates


def _prime_process_cpu_percent():
    """Primes psutil's per-process CPU delta tracking."""
    for proc in psutil.process_iter():
        try:
            proc.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue


def get_top_processes(n: int = TOP_N_PROCESSES) -> dict:
    """Returns top N processes by CPU% and top N by RAM%, flattened."""
    processes = []
    for proc in psutil.process_iter(attrs=["pid", "name"]):
        try:
            name = proc.info.get("name") or "unknown"
            if name in _EXCLUDED_PROCESS_NAMES:
                continue  # not a real workload — skip so it never occupies a top slot
            cpu_pct = proc.cpu_percent(interval=None)
            ram_pct = proc.memory_percent()
            processes.append({"name": name, "cpu": cpu_pct, "ram": ram_pct})
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    result = {}
    top_cpu = sorted(processes, key=lambda p: p["cpu"], reverse=True)[:n]
    for i in range(n):
        if i < len(top_cpu):
            result[f"top_cpu_{i+1}_name"] = top_cpu[i]["name"]
            result[f"top_cpu_{i+1}_percent"] = round(top_cpu[i]["cpu"], 2)
        else:
            result[f"top_cpu_{i+1}_name"] = None
            result[f"top_cpu_{i+1}_percent"] = None

    top_ram = sorted(processes, key=lambda p: p["ram"], reverse=True)[:n]
    for i in range(n):
        if i < len(top_ram):
            result[f"top_ram_{i+1}_name"] = top_ram[i]["name"]
            result[f"top_ram_{i+1}_percent"] = round(top_ram[i]["ram"], 2)
        else:
            result[f"top_ram_{i+1}_name"] = None
            result[f"top_ram_{i+1}_percent"] = None

    return result


def collect_once() -> dict:
    """Takes a single snapshot of system + process metrics."""
    _prime_process_cpu_percent()

    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory().percent
    disk_io = psutil.disk_io_counters()
    net_io = psutil.net_io_counters()
    battery = psutil.sensors_battery()
    gpu_percent, gpu_memory_percent, gpu_temp = get_gpu_metrics()

    io_rates = get_io_rates(disk_io, net_io)
    top_processes = get_top_processes(n=TOP_N_PROCESSES)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cpu_percent": cpu,
        "cpu_temp": get_cpu_temp(),
        "ram_percent": ram,
        "disk_read_mb_s": io_rates["disk_read_mb_s"],
        "disk_write_mb_s": io_rates["disk_write_mb_s"],
        "disk_usage_percent": get_disk_usage_percent(),
        "net_sent_mb_s": io_rates["net_sent_mb_s"],
        "net_recv_mb_s": io_rates["net_recv_mb_s"],
        "battery_percent": battery.percent if battery else None,
        "battery_plugged": int(battery.power_plugged) if battery else None,
        "gpu_percent": gpu_percent,
        "gpu_memory_percent": gpu_memory_percent,
        "gpu_temp": gpu_temp,
    }
    record.update(top_processes)
    return record


def collect_with_retry(max_retries: int = 3, backoff_seconds: float = 2.0) -> dict | None:
    """Wraps collect_once() with retry + linear backoff for transient failures."""
    for attempt in range(1, max_retries + 1):
        try:
            return collect_once()
        except Exception as e:
            logger.warning(f"Snapshot attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(backoff_seconds * attempt)
    logger.error("All retry attempts exhausted — skipping this cycle.")
    return None


def run_collector(interval_seconds: int = 30):
    """Main loop. Individual failures are logged/skipped; 5 consecutive stops it."""
    init_db()
    logger.info(f"Collector started (interval={interval_seconds}s, GPU backend={_GPU_BACKEND}). Ctrl+C to stop.")
    consecutive_failures = 0

    try:
        while True:
            record = collect_with_retry()

            if record is None:
                consecutive_failures += 1
                logger.warning(f"Cycle skipped. Consecutive failures: {consecutive_failures}")
                if consecutive_failures >= 5:
                    logger.critical("5 consecutive failures — stopping for investigation.")
                    break
                time.sleep(interval_seconds)
                continue

            consecutive_failures = 0

            try:
                insert_metric(record)
                logger.info(
                    f"Logged: CPU={record['cpu_percent']}% RAM={record['ram_percent']}% "
                    f"GPU={record['gpu_percent']}% GPU_MEM={record['gpu_memory_percent']}% "
                    f"DiskR={record['disk_read_mb_s']}MB/s NetSent={record['net_sent_mb_s']}MB/s"
                )
            except Exception as e:
                logger.error(f"DB write failed, snapshot lost: {e} | Record was: {record}")

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        logger.info("Collector stopped by user (Ctrl+C).")
    except Exception as e:
        logger.critical(f"Collector crashed with unhandled error: {e}")
        raise


if __name__ == "__main__":
    run_collector(interval_seconds=30)