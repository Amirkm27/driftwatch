"""
Shared constants for DriftWatch. Single source of truth for anything
collector.py and database.py both need — or that's likely to get
tweaked (interval, process count) without either file's logic changing.
Splitting this out now, while there's only 300 rows in the DB, is cheap;
doing it after Week 2/3 code starts importing from collector.py and
database.py directly would mean chasing hardcoded values through more
files.
"""
from pathlib import Path

# --- Paths --------------------------------------------------------------
# Resolved from this file's location, not cwd — so it doesn't matter
# whether you run `python -m src.collector` from the project root or
# Task Scheduler launches it with a different working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "driftwatch.db"
LOG_DIR = PROJECT_ROOT / "logs"

# --- Collection loop ------------------------------------------------------
DEFAULT_INTERVAL_SECONDS = 30
TOP_N_PROCESSES = 3

# --- CPU temp source --------------------------------------------------
# LibreHardwareMonitor's Remote Web Server (Options -> Remote Web Server
# -> Run). See README "Running the Collector" for the Task Scheduler setup.
LHM_WEB_SERVER_URL = "http://localhost:8085/data.json"

# "System Idle Process" reports 1400%+ CPU on a multicore box (psutil
# quirk, not normalized) and camps in the #1 slot every snapshot — zero
# variance, zero signal. Cut before ranking, not after.
EXCLUDED_PROCESS_NAMES = {"System Idle Process", "Idle"}