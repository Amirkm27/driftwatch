# DriftWatch

Continuous system telemetry logging with ML-based anomaly detection —
z-score baseline, Isolation Forest, and CUSUM — for spotting abnormal
performance or health patterns on personal hardware, running across
both a Windows laptop (NVIDIA GPU) and a Windows desktop (AMD).

## What It Does

Collects a snapshot of system vitals roughly every 30 seconds and writes
it to a local SQLite database:

- CPU / RAM / GPU utilization and temperatures
- Disk and network I/O, as MB/s rates (not raw cumulative counters)
- Disk capacity usage
- Battery status
- Top-3 CPU-consuming and top-3 RAM-consuming processes per snapshot
- Machine/hardware context (CPU model, RAM, detected GPUs) recorded
  once per collector session — and only re-recorded if it actually
  changes, so restarting the collector doesn't spam duplicate rows

The goal is a labeled, ground-truth dataset of "normal" system behavior
across a wide range of real states — idle, everyday use, heavy CPU load,
heavy GPU load, mixed workloads — used later to train and evaluate
anomaly detectors against both synthetic and naturally occurring events.

## Stack

Python 3.14 · psutil · pynvml/pyadl (GPU) · LibreHardwareMonitor Remote
Web Server (CPU temp, polled via `requests`) · SQLite · pandas ·
scikit-learn · matplotlib

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> If you ever regenerate this file with `pip freeze > requirements.txt`
> in PowerShell, redirect with `pip freeze | Out-File -Encoding utf8
> requirements.txt` instead — PowerShell's default `>` redirection
> writes UTF-16, which `pip install -r` chokes on.

## Running the Collector

**CPU temperature depends on LibreHardwareMonitor's Remote Web Server,
not its WMI publishing** (WMI publishing is broken in the current LHM
release — see Known Limitations).

1. In LibreHardwareMonitor: **Options → Remote Web Server → Run**
   (default port `8085`). This is what `get_cpu_temp()` polls for the
   CPU package temperature; without it, `cpu_temp` silently falls back
   to the ACPI/WMI reading (often unavailable on consumer laptops) or
   `None`.
2. Both `LibreHardwareMonitor.exe` and the collector run unattended via
   **Windows Task Scheduler**, as two separate tasks, both with "Run
   with highest privileges" enabled:
   - **LHM task** — launches `LibreHardwareMonitor.exe` at login, with
     a ~30s start delay so sensor init (PawnIO) finishes before
     anything tries to hit port 8085.
   - **Collector task** — `-m src.collector` (not `src\collector.py`),
     "Start in" set to the project root, with a 1-minute start delay so
     it doesn't race the LHM task on login.
3. If `cpu_temp` comes back `None` in the data, check port 8085 first —
   LibreHardwareMonitor not running, or Remote Web Server not enabled,
   is the most common cause, well ahead of the ACPI/psutil fallbacks
   ever actually needing to fire.

## Project Structure
```
src/
  collector.py — metric collection loop (psutil, GPU, temp sensors)
  database.py — SQLite schema and data access
  config.py — shared constants (DB_PATH, LOG_DIR, TOP_N_PROCESSES, interval, LHM URL, excluded processes)
data/ — local telemetry database (gitignored)
logs/ — rotating collector logs (gitignored)
notebooks/ — EDA and analysis notebooks
tests/ — test suite (planned)
```

## Database Schema

Two tables:
- **`metrics`** — one row per snapshot (~every 30s): vitals, GPU, disk/network rates, top processes
- **`system_info`** — one row per collector session: static hardware/OS facts (CPU model, RAM, detected GPUs, hybrid-GPU flag). Used to tell which machine a dataset came from when combining laptop + desktop data during analysis. A new row is only inserted when hostname, OS version, CPU model, RAM, or GPU info actually changes — not on every restart.

## GPU Support

- **NVIDIA** — fully tested via `pynvml`. Handles a driver quirk where
  `nvmlDeviceGetUtilizationRates()` raises `NVMLError_Unknown` when the
  GPU sits in a deep power-saving state with no active workload —
  treated as a real `0%` reading instead of `null`.
- **AMD** — implemented via `pyadl` for architectural completeness, but
  **not validated against real AMD hardware yet**. VRAM usage is
  unavailable regardless — no equivalent API exists in `pyadl`.

## Known Limitations

- **Integrated GPU utilization is not tracked.** Most laptops are hybrid
  (an integrated GPU alongside a discrete one), routing everyday work
  to the integrated chip while the discrete GPU idles — so `gpu_percent`
  reflects discrete-GPU load specifically, not total graphics activity.
  A CPU spike coinciding with integrated-GPU work (e.g. hardware video
  decode) may show up with no clear GPU-side explanation. Tracking
  integrated GPU load would require parsing undocumented Windows
  performance counters rather than a stable API, and was scoped out.
  `system_info.gpu_names` / `is_hybrid_gpu_system` record whether a
  given machine has this hybrid setup, for context.
- **CPU temperature depends on LibreHardwareMonitor running separately**,
  polled over HTTP rather than embedded. Two earlier approaches were
  tried and dropped: `HardView` bundled the older WinRing0 driver,
  which Windows Defender quarantined as known-vulnerable (real CVEs,
  unsigned); embedding LibreHardwareMonitorLib directly via `pythonnet`
  worked but pulled in ~20 interdependent .NET assemblies for a
  fragile, hard-to-verify dependency chain. Polling LHM's own Remote
  Web Server sidesteps both — it's part of the same process that
  already reads sensors correctly — at the cost of one extra background
  process, which Task Scheduler now starts automatically.
- **`system_info` change-detection doesn't cover `collector_interval_seconds`.**
  If the poll interval is ever changed (e.g. 30s → 60s) without any
  hardware also changing, no new `system_info` row is written to flag
  it — so a mixed-interval dataset won't be self-documenting from that
  table alone. Worth checking snapshot-to-snapshot gaps directly in
  `metrics.timestamp` if this matters for a given analysis.
- **AMD GPU path is unverified** — see above.

## Future Work (Out of Scope for Now)

- Integrated GPU utilization tracking (via Windows performance counters)
- Adaptive drift thresholding (e.g. ADWIN) instead of a fixed baseline
- Distribution-shift detection (e.g. MMD) as an alternative to CUSUM
- Real-time alerting/notification, rather than post-hoc analysis
- Cross-machine dataset merging into a unified schema/store instead of
  two separate local SQLite files
