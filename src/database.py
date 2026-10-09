"""
SQLite layer for DriftWatch.

Stores telemetry snapshots and static system information separately.
"""
import sqlite3
from contextlib import contextmanager

from src.config import DB_PATH, TOP_N_PROCESSES


def init_db():
    """
    Creates the required tables and keeps the existing database structure.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,

                -- CPU
                cpu_percent REAL,
                cpu_temp REAL,

                -- RAM
                ram_percent REAL,

                -- Disk (rates in MB/s, computed in collector.py — psutil
                -- only gives cumulative counters, useless on their own)
                disk_read_mb_s REAL,
                disk_write_mb_s REAL,
                disk_usage_percent REAL,

                -- Network (same deal, rates not raw counters)
                net_sent_mb_s REAL,
                net_recv_mb_s REAL,

                -- Battery
                battery_percent REAL,
                battery_plugged INTEGER,

                -- GPU (NVIDIA tested, AMD untested — no AMD box to test on)
                gpu_percent REAL,
                gpu_memory_percent REAL,
                gpu_temp REAL
            )
        """)
        conn.commit()

        _ensure_process_columns(conn)
        _create_system_info_table(conn)
        _ensure_system_info_fk_column(conn)


def _ensure_process_columns(conn: sqlite3.Connection):
    """
    Generate process columns based on the configured number of tracked processes.
    Missing columns are added without recreating the table.
    """
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(metrics)")}

    new_columns = []
    for i in range(1, TOP_N_PROCESSES + 1):
        new_columns.append((f"top_cpu_{i}_name", "TEXT"))
        new_columns.append((f"top_cpu_{i}_percent", "REAL"))
        new_columns.append((f"top_ram_{i}_name", "TEXT"))
        new_columns.append((f"top_ram_{i}_percent", "REAL"))

    for col_name, col_type in new_columns:
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE metrics ADD COLUMN {col_name} {col_type}")
    conn.commit()


def _ensure_system_info_fk_column(conn: sqlite3.Connection):
    """
    Links telemetry rows to the system information record that produced them.
    """
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(metrics)")}
    if "system_info_id" not in existing_cols:
        conn.execute(
            "ALTER TABLE metrics ADD COLUMN system_info_id INTEGER REFERENCES system_info (id)"
        )
    conn.commit()


def _create_system_info_table(conn: sqlite3.Connection):
    """
    Stores static hardware information separately from telemetry snapshots.
    The hybrid GPU flag records systems with multiple GPU devices.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS system_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recorded_at TEXT NOT NULL,
            hostname TEXT,
            os_name TEXT,
            os_version TEXT,
            cpu_model TEXT,
            cpu_cores_physical INTEGER,
            cpu_cores_logical INTEGER,
            ram_total_gb REAL,
            gpu_backend TEXT,
            gpu_names TEXT,
            is_hybrid_gpu_system INTEGER,
            python_version TEXT,
            collector_interval_seconds INTEGER
        )
    """)
    conn.commit()


@contextmanager
def get_connection():
    """
    Provides a managed SQLite connection with cleanup and rollback handling.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    except sqlite3.Error as e:
        conn.rollback()
        raise RuntimeError(f"Database error: {e}") from e
    finally:
        conn.close()


def _build_process_columns_and_placeholders():
    """Generate SQL columns and placeholders for process fields."""
    cols = []
    placeholders = []
    for i in range(1, TOP_N_PROCESSES + 1):
        for prefix in ("top_cpu", "top_ram"):
            cols.append(f"{prefix}_{i}_name")
            cols.append(f"{prefix}_{i}_percent")
            placeholders.append(f":{prefix}_{i}_name")
            placeholders.append(f":{prefix}_{i}_percent")
    return cols, placeholders


def _get_current_system_info_id(conn: sqlite3.Connection):
    """Return the latest system information record."""
    row = conn.execute("SELECT id FROM system_info ORDER BY recorded_at DESC LIMIT 1").fetchone()
    return row[0] if row else None


def insert_metric(record: dict):
    """
    Fill missing optional process fields with NULL before insertion.
    """
    process_cols, process_placeholders = _build_process_columns_and_placeholders()

    safe_record = dict(record)
    for col in process_cols:
        safe_record.setdefault(col, None)

    all_cols = [
        "timestamp",
        "cpu_percent", "cpu_temp",
        "ram_percent",
        "disk_read_mb_s", "disk_write_mb_s", "disk_usage_percent",
        "net_sent_mb_s", "net_recv_mb_s",
        "battery_percent", "battery_plugged",
        "gpu_percent", "gpu_memory_percent", "gpu_temp",
        "system_info_id",
    ] + process_cols

    all_placeholders = [
        ":timestamp",
        ":cpu_percent", ":cpu_temp",
        ":ram_percent",
        ":disk_read_mb_s", ":disk_write_mb_s", ":disk_usage_percent",
        ":net_sent_mb_s", ":net_recv_mb_s",
        ":battery_percent", ":battery_plugged",
        ":gpu_percent", ":gpu_memory_percent", ":gpu_temp",
        ":system_info_id",
    ] + process_placeholders

    query = f"""
        INSERT INTO metrics ({", ".join(all_cols)})
        VALUES ({", ".join(all_placeholders)})
    """

    with get_connection() as conn:
        safe_record["system_info_id"] = _get_current_system_info_id(conn)
        conn.execute(query, safe_record)
        conn.commit()


def insert_system_info(record: dict):
    """
    Insert a new system record only when the hardware or environment changes.
    """
    compare_cols = ["hostname", "os_version", "cpu_model", "ram_total_gb",
                     "gpu_backend", "gpu_names", "is_hybrid_gpu_system"]

    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        last = conn.execute(
            "SELECT * FROM system_info ORDER BY recorded_at DESC LIMIT 1"
        ).fetchone()

        if last and all(last[c] == record.get(c) for c in compare_cols):
            return  # nothing meaningful changed, skip the insert

        cols = [
            "recorded_at", "hostname", "os_name", "os_version", "cpu_model",
            "cpu_cores_physical", "cpu_cores_logical", "ram_total_gb",
            "gpu_backend", "gpu_names", "is_hybrid_gpu_system",
            "python_version", "collector_interval_seconds",
        ]
        safe_record = dict(record)
        for col in cols:
            safe_record.setdefault(col, None)

        placeholders = [f":{c}" for c in cols]
        query = f"INSERT INTO system_info ({', '.join(cols)}) VALUES ({', '.join(placeholders)})"
        conn.execute(query, safe_record)
        conn.commit()


def get_system_info():
    """Return stored system information records."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM system_info ORDER BY recorded_at DESC")
        return cursor.fetchall()


def get_metrics(limit: int = None):
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        query = "SELECT * FROM metrics ORDER BY timestamp DESC"
        if limit:
            query += f" LIMIT {limit}"
        cursor = conn.execute(query)
        return cursor.fetchall()


def get_metrics_as_dataframe():
    """Return metrics as a pandas DataFrame for analysis."""
    import pandas as pd
    with get_connection() as conn:
        return pd.read_sql_query("SELECT * FROM metrics ORDER BY timestamp ASC", conn)