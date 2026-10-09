"""Configuration constants for DriftWatch.

Centralizes shared settings used by collection and analysis components.
"""

from pathlib import Path

# Paths
# Resolved relative to the project location rather than the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "driftwatch.db"
SYNTHETIC_DB_PATH = PROJECT_ROOT / "data" / "driftwatch_synthetic.db"
LOG_DIR = PROJECT_ROOT / "logs"
LABELS_CSV_PATH = PROJECT_ROOT / "data" / "labels.csv"

# Collection settings
DEFAULT_INTERVAL_SECONDS = 30
TOP_N_PROCESSES = 3

# CPU temperature source
# LibreHardwareMonitor Remote Web Server endpoint.
LHM_WEB_SERVER_URL = "http://localhost:8085/data.json"

# Processes excluded from ranking because they do not provide useful workload information.
EXCLUDED_PROCESS_NAMES = {"System Idle Process", "Idle"}

# Metrics used by the feature and detection pipeline.
# Keeping this list centralized avoids updating multiple files when metrics change.
NUMERIC_METRICS = [
    "cpu_percent", "cpu_temp", "ram_percent",
    "disk_read_mb_s", "disk_write_mb_s", "disk_usage_percent",
    "net_sent_mb_s", "net_recv_mb_s",
    "battery_percent",
    "gpu_percent", "gpu_memory_percent", "gpu_temp",
]

ANOMALY_LABELS = {
    "ram_spike",
    "cpu_stress",
    "synthetic_cpu_spike",
    "synthetic_mem_leak",
    "synthetic_disk_thrash",
}

NORMAL_LABELS = {"gaming_gpu_load"}