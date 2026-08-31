"""
Handles all SQLite database interactions for DriftWatch.
"""
import sqlite3
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "driftwatch.db"

TOP_N_PROCESSES = 3


def init_db():
    """
    Creates the metrics table if it doesn't exist. Called once at startup;
    safe to call repeatedly (idempotent). Columns grouped by category.
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

                -- Disk (rates, MB/s — not cumulative; see collector.py)
                disk_read_mb_s REAL,
                disk_write_mb_s REAL,
                disk_usage_percent REAL,

                -- Network (rates, MB/s — not cumulative)
                net_sent_mb_s REAL,
                net_recv_mb_s REAL,

                -- Battery
                battery_percent REAL,
                battery_plugged INTEGER,

                -- GPU
                gpu_percent REAL,
                gpu_memory_percent REAL,
                gpu_temp REAL
            )
        """)
        conn.commit()

        _ensure_process_columns(conn)


def _ensure_process_columns(conn: sqlite3.Connection):
    """
    ALTER TABLEs in top-process columns if missing — kept as migration
    logic since TOP_N_PROCESSES is a variable that may change later.
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


@contextmanager
def get_connection():
    """Context manager for a SQLite connection. Always closes, even on error."""
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    except sqlite3.Error as e:
        conn.rollback()
        raise RuntimeError(f"Database error: {e}") from e
    finally:
        conn.close()


def _build_process_columns_and_placeholders():
    """Generates column names + named-placeholders for top-process fields."""
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
    Inserts one row of collected metrics. Missing keys (GPU unavailable,
    first snapshot with no rate data yet, fewer than TOP_N_PROCESSES
    running) are treated as None via setdefault.
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


def get_metrics(limit: int = None):
    """Retrieves metrics, most recent first."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        query = "SELECT * FROM metrics ORDER BY timestamp DESC"
        if limit:
            query += f" LIMIT {limit}"
        cursor = conn.execute(query)
        return cursor.fetchall()


def get_metrics_as_dataframe():
    """Returns all metrics as a pandas DataFrame, oldest first."""
    import pandas as pd
    with get_connection() as conn:
        return pd.read_sql_query("SELECT * FROM metrics ORDER BY timestamp ASC", conn)