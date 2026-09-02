"""
Pulls system metrics via psutil (+ GPU via pynvml/pyadl, + CPU temp via
HardView) and writes them to SQLite. Disk/network numbers come out of
psutil as running totals since boot, not rates — converted to MB/s here,
see get_io_rates().
"""
import os
import time
import platform
import socket
import logging
import requests
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime, timezone

import psutil

from src.database import init_db, insert_metric, insert_system_info, TOP_N_PROCESSES

# --- logging: console + rotating file. leaving at DEBUG for now while
# I'm still chasing the GPU null issue — bump back to INFO once that's
# actually confirmed fixed, DEBUG is way too noisy for a week-long run ---
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("driftwatch.collector")
logger.setLevel(logging.INFO)

_console_handler = logging.StreamHandler()
_file_handler = RotatingFileHandler(LOG_DIR / "collector.log", maxBytes=5_000_000, backupCount=3)
_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_console_handler.setFormatter(_formatter)
_file_handler.setFormatter(_formatter)

if not logger.handlers:
    logger.addHandler(_console_handler)
    logger.addHandler(_file_handler)

# "System Idle Process" reports something like 1400%+ CPU on a multicore
# box (psutil quirk, not normalized) and camps in the #1 slot every single
# snapshot — zero variance, zero signal, just noise. Cut it before it
# ever gets ranked.
_EXCLUDED_PROCESS_NAMES = {"System Idle Process", "Idle"}


# =====================================================================
# GPU backend — try NVIDIA first (pynvml, official + solid), fall back
# to AMD (pyadl, unofficial, Windows-only, never tested on real AMD hw)
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
    Three separate try/excepts on purpose — originally had util/mem/temp
    all in one try block, and one failed call was silently wiping out the
    other two even when they'd have worked fine. Split them apart, no
    reason a dead utilization read should also cost me a good temp reading.
    """
    if _GPU_BACKEND == "nvidia":
        gpu_percent = None
        mem_percent = None
        gpu_temp = None

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
            gpu_percent = float(util.gpu)
        except (pynvml.NVMLError_NotSupported, pynvml.NVMLError_Unknown):
            # GPU sitting in a low-power state (P8) with nothing touching
            # it — utilization query just refuses to answer. Docs/forums
            # say this throws NotSupported, but on my actual laptop it's
            # NVMLError_Unknown ("Unknown Error", code 999) — confirmed
            # by turning DEBUG logging on and reading collector.log.
            # Either way it means "idle," so call it 0%, not unknown.
            gpu_percent = 0.0
        except Exception as e:
            logger.debug(f"NVIDIA GPU utilization read failed: {e}")

        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
            mem_percent = round((mem.used / mem.total) * 100, 2) if mem.total else None
        except Exception as e:
            logger.debug(f"NVIDIA GPU memory read failed: {e}")

        try:
            gpu_temp = float(pynvml.nvmlDeviceGetTemperature(_NVML_HANDLE, pynvml.NVML_TEMPERATURE_GPU))
        except Exception as e:
            logger.debug(f"NVIDIA GPU temp read failed: {e}")

        return gpu_percent, mem_percent, gpu_temp

    elif _GPU_BACKEND == "amd":
        gpu_percent = None
        gpu_temp = None

        # pyadl just doesn't have a VRAM call — not "sometimes fails,"
        # it flat out doesn't exist, so not even trying.
        mem_percent = None

        try:
            gpu_percent = float(_AMD_DEVICE.getCurrentUsage())
        except Exception as e:
            logger.debug(f"AMD GPU usage read failed: {e}")

        try:
            gpu_temp = float(_AMD_DEVICE.getCurrentTemperature())
        except Exception as e:
            logger.debug(f"AMD GPU temp read failed: {e}")

        return gpu_percent, mem_percent, gpu_temp

    return None, None, None


def _get_gpu_inventory() -> tuple[str, bool]:
    """
    Just listing every video controller Windows knows about, not polling
    anything. Point of this: most laptops are hybrid (Intel iGPU +
    NVIDIA dGPU), and gpu_percent above only ever reflects the discrete
    card. Worth knowing whether that's even the case on a given machine.
    Cheap WMI call, runs once at startup, no admin needed for this one.
    """
    try:
        import wmi
        w = wmi.WMI()
        names = [c.Name for c in w.Win32_VideoController() if c.Name]
        return ", ".join(names), len(names) > 1
    except Exception as e:
        logger.debug(f"GPU inventory enumeration failed: {e}")
        return "unknown", False


def collect_system_info(interval_seconds: int) -> dict:
    """
    Runs once at collector startup, not per snapshot — CPU model/RAM/GPU
    list don't change mid-session. Wrapped in a broad try/except because
    I'd rather start the collector with a half-empty system_info row
    than have some WMI hiccup block the whole thing from starting.
    """
    try:
        cpu_model = None
        try:
            import wmi
            w = wmi.WMI()
            cpu_model = w.Win32_Processor()[0].Name.strip()
        except Exception as e:
            logger.debug(f"WMI CPU model read failed, falling back: {e}")
            cpu_model = platform.processor()

        gpu_names, is_hybrid = _get_gpu_inventory()

        return {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "os_name": platform.system(),
            "os_version": platform.platform(),
            "cpu_model": cpu_model,
            "cpu_cores_physical": psutil.cpu_count(logical=False),
            "cpu_cores_logical": psutil.cpu_count(logical=True),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
            "gpu_backend": _GPU_BACKEND,
            "gpu_names": gpu_names,
            "is_hybrid_gpu_system": int(is_hybrid),
            "python_version": platform.python_version(),
            "collector_interval_seconds": interval_seconds,
        }
    except Exception as e:
        logger.error(f"Failed to collect system_info: {e}")
        return {"recorded_at": datetime.now(timezone.utc).isoformat()}



def _get_cpu_temp_lhm_webserver() -> float | None:
    """
    Uses LibreHardwareMonitor's built-in Remote Web Server (Options ->
    Remote Web Server -> Run, default port 8085) instead of WMI.

    Switched to this after confirming a real regression: LHM's latest
    release reads sensors fine (PawnIO works, temps show correctly in
    its own UI) but stopped publishing WMI entirely — confirmed via a
    matching GitHub issue, not something specific to this machine.
    Downgrading to get WMI back (v0.9.4) broke sensor reading instead,
    since that version predates PawnIO and still expects the old,
    now-removed WinRing0 driver. The web server sidesteps this
    conflict — it's part of the same app process that already reads
    sensors correctly, no separate publishing mechanism to break.
    """
    try:
        import requests
        resp = requests.get("http://localhost:8085/data.json", timeout=2)
        resp.raise_for_status()
        data = resp.json()

        def find_cpu_package_temp(node):
            if node.get("Text", "").startswith("CPU Package") and "°C" in str(node.get("Value", "")):
                try:
                    return float(node["Value"].replace("°C", "").strip())
                except (ValueError, KeyError):
                    return None
            for child in node.get("Children", []):
                result = find_cpu_package_temp(child)
                if result is not None:
                    return result
            return None

        temp = find_cpu_package_temp(data)
        return round(temp, 1) if temp is not None else None

    except Exception as e:
        logger.debug(f"LibreHardwareMonitor web server read failed (is it running?): {e}")
        return None


def get_cpu_temp() -> float | None:
    """
    Three-tier fallback, best source first:
      1. LibreHardwareMonitor's local web server (localhost:8085) —
         requires the app running with Remote Web Server enabled.
         Chosen over WMI after confirming WMI publishing is broken in
         the current LHM release (documented upstream regression).
      2. Windows ACPI thermal zone via WMI — built-in, no extra app,
         but OEM-dependent and fails on a lot of consumer laptops.
      3. psutil sensors_temperatures() — Linux only, harmless no-op
         on Windows, kept for portability.
    Returns None if all three fail — a real hardware/software
    limitation at that point, not a bug worth chasing further.
    """
    temp = _get_cpu_temp_lhm_webserver()
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
    try:
        root = os.path.abspath(os.sep)
        return psutil.disk_usage(root).percent
    except Exception as e:
        logger.debug(f"disk_usage read failed: {e}")
        return None


# =====================================================================
# disk/net rates — psutil only gives cumulative bytes since boot, which
# on its own is basically "how long has this machine been on." need the
# delta between two samples to get anything meaningful.
# =====================================================================
_prev_io_sample = None


def _safe_rate_mb_s(current_bytes, prev_bytes, elapsed_seconds) -> float | None:
    """
    time.monotonic() instead of time.time() for elapsed — wall clock can
    jump (NTP sync, manual change) and would wreck this math. monotonic
    only ever goes forward.

    The 300s cap exists because of laptop sleep — if the machine was
    suspended for hours between two snapshots, dividing the accumulated
    bytes by that huge elapsed window gives a rate that looks tiny/wrong,
    not "no data." Better to just say None and let the gap show up as a
    gap, not a fake reading.
    """
    if current_bytes is None or prev_bytes is None or elapsed_seconds <= 0:
        return None
    if elapsed_seconds > 300:
        return None
    delta_bytes = current_bytes - prev_bytes
    if delta_bytes < 0:
        # network interface reset or similar — counter went backwards,
        # can't trust it
        return None
    return round((delta_bytes / (1024 ** 2)) / elapsed_seconds, 4)


def get_io_rates(disk_io, net_io) -> dict:
    """
    First call ever has nothing to diff against — all four come back
    None on row 1 of every session. Expected, not a bug, don't panic
    when you see it in the DB.
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
    """
    proc.cpu_percent() needs two calls to mean anything — first call is
    always garbage/zero because there's no prior sample yet. This is the
    "throwaway" first call; the real read happens later in
    get_top_processes(), after the 1s window from cpu_percent(interval=1)
    has actually passed.
    """
    for proc in psutil.process_iter():
        try:
            proc.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue


def get_top_processes(n: int = TOP_N_PROCESSES) -> dict:
    """
    Order matters here — this has to run after cpu_percent(interval=1)
    in collect_once(), not before, or the per-process deltas are still
    zero. Learned that the hard way with a whole run of null process data.
    """
    processes = []
    for proc in psutil.process_iter(attrs=["pid", "name"]):
        try:
            name = proc.info.get("name") or "unknown"
            if name in _EXCLUDED_PROCESS_NAMES:
                continue
            cpu_pct = proc.cpu_percent(interval=None)
            ram_pct = proc.memory_percent()
            processes.append({"name": name, "cpu": cpu_pct, "ram": ram_pct})
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # process can vanish between listing it and reading it —
            # totally normal, not worth logging every time
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
    """One full snapshot. Called every interval_seconds from run_collector()."""
    _prime_process_cpu_percent()

    # this blocks for 1s — doing double duty as both the system-wide CPU
    # sample window AND the wait needed for per-process deltas to be real
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
    """
    Backing off instead of hammering retries immediately — if something's
    momentarily busy, retrying instantly just hits the same wall again.
    Give it a second to sort itself out.
    """
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
    """
    Main loop. One bad cycle shouldn't kill a multi-day run — only bail
    out after 5 in a row, since that's a real problem (disk full, DB
    gone, etc.) and not just a one-off hiccup.
    """
    init_db()

    system_info = collect_system_info(interval_seconds)
    insert_system_info(system_info)
    logger.info(
        f"System: {system_info.get('hostname')} | "
        f"CPU: {system_info.get('cpu_model')} | "
        f"GPUs: {system_info.get('gpu_names')} | "
        f"Hybrid: {bool(system_info.get('is_hybrid_gpu_system'))}"
    )

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
                # data's already collected, just failed to save it —
                # log the whole record so it's not totally lost
                logger.error(f"DB write failed, snapshot lost: {e} | Record was: {record}")

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        logger.info("Collector stopped by user (Ctrl+C).")
    except Exception as e:
        logger.critical(f"Collector crashed with unhandled error: {e}")
        raise


if __name__ == "__main__":
    run_collector(interval_seconds=30)