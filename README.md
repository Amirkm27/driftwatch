# DriftWatch

Continuous system telemetry logging with ML-based anomaly detection for
spotting abnormal performance or health patterns on personal hardware,
running across both a Windows laptop (NVIDIA GPU) and a Windows desktop
(AMD).

**Currently implemented:** z-score and Mahalanobis distance baseline
detectors, evaluated against both real confirmed events and synthetically
injected anomalies. **Planned (Week 3):** Isolation Forest and CUSUM.

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

That raw telemetry then feeds an analysis pipeline: rolling-window
feature engineering, two anomaly detectors (with more planned), and an
evaluation harness that scores each detector against both hand-confirmed
real events (stress tests, gaming sessions) and synthetically injected
anomalies with known ground truth.

## Stack

Python 3.14 · psutil · pynvml/pyadl (GPU) · LibreHardwareMonitor Remote
Web Server (CPU temp, polled via `requests`) · SQLite · pandas · numpy ·
scipy (Mahalanobis / chi-square) · scikit-learn (evaluation metrics,
planned Isolation Forest) · matplotlib

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
  collector.py             — metric collection loop (psutil, GPU, temp sensors)
  database.py               — metrics/system_info schema and data access
  config.py                 — shared constants (paths, metric list, collection settings)
  db_schema.py               — analysis schema (features/labels/detections/etc.)
  migrate_db.py              — applies/updates the analysis schema, imports labels.csv
  features.py                — rolling-window feature engineering
  detectors.py                — z-score and Mahalanobis distance detectors
  build_label_bridge.py       — resolves label time-windows into row-level links
  check_label_overlap.py      — informal per-label flag-rate check
  evaluate.py                  — formal precision/recall/F1/ROC-AUC evaluation
  inject_anomalies.py          — builds a synthetic evaluation dataset with known-location anomalies
  find_gaming_sessions.py      — utility for identifying gaming sessions to label
data/
  driftwatch.db              — real telemetry + analysis results (gitignored)
  driftwatch_synthetic.db     — synthetic evaluation dataset (gitignored)
  labels.csv                  — hand-confirmed real event labels (gitignored — personal usage data)
  evaluation_results.csv       — running log of every evaluation run (tracked)
logs/ — rotating collector logs (gitignored)
notebooks/ — EDA and analysis notebooks
tests/ — test suite (planned)
```

## Database Schema

Normalized across seven tables, redesigned in Week 2 after an initial
denormalized version bloated to ~3.6MB from duplicated raw values stored
in every derived table — the current schema cut that by roughly 90% and
added real foreign-key constraints throughout:

- **`metrics`** — one row per snapshot (~every 30s): vitals, GPU,
  disk/network rates, top processes. Append-only, single source of
  truth, never derived from anything else.
- **`system_info`** — one row per distinct hardware/OS configuration.
  Only writes a new row when hardware actually changes. `metrics`
  carries a `system_info_id` foreign key back to this table, so combined
  laptop + desktop datasets can be told apart later.
- **`features`** — one row per `metrics` row: rolling mean/std/median/MAD
  and first-difference per metric. No raw metric values duplicated —
  join back to `metrics` via `metrics_id` when needed.
- **`labels`** — confirmed anomaly/event time windows (`start_ts`,
  `end_ts`, `label`, `source`, `confidence`), both real
  (`ram_spike`, `cpu_stress`, `gaming_gpu_load`) and synthetic.
- **`label_metrics`** — bridge table materializing exactly which
  `metrics` rows fall inside each label's time window, so evaluation
  queries are clean joins instead of `BETWEEN` scans.
- **`detection_runs`** — one row per detector execution (method,
  threshold, timestamp). Re-running a detector replaces its previous
  run rather than accumulating history.
- **`detections`** — one row per (metrics × run): score and anomaly
  flag. References `detection_runs` for method/threshold context
  instead of repeating it on every row.

## Analysis Pipeline

Run in this order whenever there's new data:

```bash
python -m src.migrate_db          # ensure schema is current
python -m src.features            # compute rolling features for new rows
python -m src.detectors           # run z-score + Mahalanobis
python -m src.build_label_bridge  # resolve any new/changed labels
python -m src.evaluate --save     # precision/recall/F1/ROC-AUC, logged to CSV
```

All five scripts accept `--db <path>` to target a different database —
used to run the identical pipeline against the synthetic evaluation
dataset without touching real data:

```bash
python -m src.inject_anomalies                              # builds driftwatch_synthetic.db
python -m src.features --db data\driftwatch_synthetic.db
python -m src.detectors --db data\driftwatch_synthetic.db
python -m src.build_label_bridge --db data\driftwatch_synthetic.db
python -m src.evaluate --db data\driftwatch_synthetic.db --save
```

## Detection Methods

- **Z-score (baseline)** — robust per-metric z-score using rolling
  median/MAD rather than mean/std, with a global-std floor to prevent
  runaway values on metrics that sit at zero most of the time (disk/net
  rates). Reduced to a composite score per row (max absolute z-score
  across metrics).
- **Mahalanobis distance** — multivariate distance against a
  mean/covariance fit on data *not* covered by any label, so known
  anomalies don't get baked into the "normal" model. Threshold set via
  the chi-square distribution at the target confidence level.

## Evaluation

Two parallel ground-truth sources: hand-confirmed real events in
`labels.csv` (stress tests, memory spikes, gaming sessions — the latter
a deliberate negative-class case, since it's legitimate heavy load that
should *not* be flagged), and a synthetic dataset (`inject_anomalies.py`)
with CPU spikes, memory leaks, and disk thrash injected at known,
labeled locations. `evaluate.py` computes precision, recall, F1, and
ROC-AUC for each detector against both, logged to
`data/evaluation_results.csv`.

## GPU Support

- **NVIDIA** — fully tested via `pynvml`. Handles a driver quirk where
  `nvmlDeviceGetUtilizationRates()` raises `NVMLError_Unknown` when the
  GPU sits in a deep power-saving state with no active workload —
  treated as a real `0%` reading instead of `null`.
- **AMD** — implemented via `pyadl` for architectural completeness, but
  **not validated against real AMD hardware yet**. VRAM usage is
  unavailable regardless — no equivalent API exists in `pyadl`.

## Known Limitations

- **Ground truth treats unlabeled data as normal.** Evaluation assumes
  any `metrics` row not covered by a label is a true negative — real
  anomalies that were never manually confirmed would be silently
  scored as correct rejections. Mitigated but not eliminated by the
  synthetic dataset, which has exhaustive known-location labels.
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
- **AMD GPU path is unverified** — see above. `system_info_id` groundwork
  is in place for combining laptop + desktop data, but no desktop data
  has been collected or merged yet.

## Future Work (Out of Scope for Now)

- Isolation Forest and CUSUM detectors (Week 3)
- Streamlit dashboard, lightweight feature importance (Week 4)
- Integrated GPU utilization tracking (via Windows performance counters)
- Adaptive drift thresholding (e.g. ADWIN) instead of a fixed baseline
- Distribution-shift detection (e.g. MMD) as an alternative to CUSUM
- Real-time alerting/notification, rather than post-hoc analysis
- Merging the Ryzen 5 desktop's independently-collected data into the
  laptop's database (schema supports it via `system_info_id`; the actual
  merge script, which needs to handle ID-collision remapping between two
  independently auto-incremented databases, hasn't been written)