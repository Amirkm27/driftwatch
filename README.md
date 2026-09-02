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
  once per collector session

The goal is a labeled, ground-truth dataset of "normal" system behavior
across a wide range of real states — idle, everyday use, heavy CPU load,
heavy GPU load, mixed workloads — used later to train and evaluate
anomaly detectors against both synthetic and naturally occurring events.

## Stack

Python 3.14 · psutil · pynvml/pyadl (GPU) · HardView (Windows CPU temp)
· SQLite · pandas · scikit-learn · matplotlib

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Collector

```bash
python -m src.collector
```

**⚠️ Administrator privileges required (Windows).** CPU temperature is
read via HardView, which wraps LibreHardwareMonitorLib — this needs
elevated permissions to touch hardware sensors. Without admin rights,
`cpu_temp` silently falls back to a less reliable ACPI/WMI reading, or
`None`, rather than erroring — always launch elevated (run your
terminal/IDE "as Administrator") for real collection runs.

For unattended background collection, this project uses Windows Task
Scheduler with "Run with highest privileges" enabled. Task arguments
must use `-m src.collector` (not `src\collector.py`), with "Start in"
set to the project root — running the file path directly breaks the
`from src.database import ...` import outside PyCharm's environment.

## Project Structure
src/
collector.py — metric collection loop (psutil, GPU, temp sensors)
database.py — SQLite schema and data access
config.py — shared constants
data/ — local telemetry database (gitignored)
logs/ — rotating collector logs (gitignored)
notebooks/ — EDA and analysis notebooks
tests/ — test suite (planned)

## Database Schema

Two tables:
- **`metrics`** — one row per snapshot (~every 30s): vitals, GPU, disk/network rates, top processes
- **`system_info`** — one row per collector session: static hardware/OS facts (CPU model, RAM, detected GPUs, hybrid-GPU flag). Used to tell which machine a dataset came from when combining laptop + desktop data during analysis.

## GPU Support

- **NVIDIA** — fully tested via `pynvml`. Handles a driver quirk where
  `nvmlDeviceGetUtilizationRates()` raises `NVMLError_Unknown` when the
  GPU sits in a deep power-saving state with no active workload —
  treated as a real `0%` reading instead of `null`.
- **AMD** — implemented via `pyadl` for architectural completeness, but
  **not validated against real AMD hardware**. VRAM usage is unavailable
  regardless — no equivalent API exists in `pyadl`.

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
- **CPU temperature reliability is hardware-dependent.** Even with the
  HardView → ACPI/WMI fallback chain, some machines never expose an
  accurate reading regardless of software used — this is a firmware
  limitation, not something fixable from Python.
- **AMD GPU path is unverified** — see above.

## Future Work (Out of Scope for Now)

- Integrated GPU utilization tracking (via Windows performance counters)
- Adaptive drift thresholding (e.g. ADWIN) instead of a fixed baseline
- Distribution-shift detection (e.g. MMD) as an alternative to CUSUM
- Real-time alerting/notification, rather than post-hoc analysis
- Cross-machine dataset merging into a unified schema/store instead of
  two separate local SQLite files