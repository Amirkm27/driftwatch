"""
SQLite layer for DriftWatch. Two tables:
- metrics: one row per snapshot, ~every 30s
- system_info: one row per collector session — hardware/OS facts that
  don't change between snapshots, so no point repeating them 1000s of times
"""
import sqlite3
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "driftwatch.db"

TOP_N_PROCESSES = 3


def init_db():
    """
    Idempotent — safe to call on every startup. Columns grouped by
    category below just so I can actually find things when scrolling.
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


def _ensure_process_columns(conn: sqlite3.Connection):
    """
    Top-N process columns are generated, not hardcoded, because
    TOP_N_PROCESSES might change later and I don't want to hand-edit
    the schema if it does. ALTER TABLE only adds what's missing so
    existing data doesn't get wiped.
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


def _create_system_info_table(conn: sqlite3.Connection):
    """
    Separate table on purpose — CPU model / RAM size / which GPUs exist
    don't change between snapshots, so shoving them into `metrics` would
    just mean the same string repeated a few thousand times for nothing.
    One row per collector run is enough.

    Also this is where the hybrid-GPU thing lives (is_hybrid_gpu_system) —
    most laptops have both an Intel iGPU and a discrete NVIDIA card, and
    DriftWatch only tracks the discrete one. This column at least records
    that the other GPU exists, even though we're not polling it.
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
    Wrap every DB touch in this — learned the hard way that leaving a
    connection open across a whole run loop eventually locks the file
    if anything else tries to read it (e.g. inspecting the DB mid-run).
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
    """Same generation trick as _ensure_process_columns, just for the INSERT side."""
    cols = []
    placeholders = []
    for i in range(1, TOP_N_PROCESSES + 1):
        for prefix in ("top_cpu", "top_ram"):
            cols.append(f"{prefix}_{i}_name")
            cols.append(f"{prefix}_{i}_percent")
            placeholders.append(f":{prefix}_{i}_name")
            placeholders.append(f":{prefix}_{i}_percent")
    return cols, placeholders


def insert_metric(record: dict):
    """
    setdefault(None) on every process column because some snapshots
    genuinely have fewer than TOP_N_PROCESSES worth of data (or GPU
    read failed that cycle) — without this, a missing key throws
    instead of just storing NULL, and I'd rather lose one field than
    lose the whole row.
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
    ] + process_cols

    all_placeholders = [
        ":timestamp",
        ":cpu_percent", ":cpu_temp",
        ":ram_percent",
        ":disk_read_mb_s", ":disk_write_mb_s", ":disk_usage_percent",
        ":net_sent_mb_s", ":net_recv_mb_s",
        ":battery_percent", ":battery_plugged",
        ":gpu_percent", ":gpu_memory_percent", ":gpu_temp",
    ] + process_placeholders

    query = f"""
        INSERT INTO metrics ({", ".join(all_cols)})
        VALUES ({", ".join(all_placeholders)})
    """

    with get_connection() as conn:
        conn.execute(query, safe_record)
        conn.commit()


def insert_system_info(record: dict):
    """One call per collector startup — see collect_system_info() in collector.py."""
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
    query = f"""
        INSERT INTO system_info ({", ".join(cols)})
        VALUES ({", ".join(placeholders)})
    """
    with get_connection() as conn:
        conn.execute(query, safe_record)
        conn.commit()


def get_system_info():
    """Most recent session first — mostly just for a sanity check in the notebook."""
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
    """This is basically the only function I actually use once EDA starts."""
    import pandas as pd
    with get_connection() as conn:
        return pd.read_sql_query("SELECT * FROM metrics ORDER BY timestamp ASC", conn)