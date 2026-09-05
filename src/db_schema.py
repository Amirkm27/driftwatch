import sqlite3

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS labels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        start_ts TEXT NOT NULL,
        end_ts TEXT NOT NULL,
        label TEXT NOT NULL,
        source TEXT,
        confidence TEXT,
        notes TEXT,
        UNIQUE (start_ts, end_ts, label)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS detection_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        method TEXT NOT NULL,
        threshold_used REAL,
        computed_at TEXT NOT NULL,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL,
        metrics_id INTEGER NOT NULL,
        score REAL NOT NULL,
        is_anomaly INTEGER NOT NULL,
        FOREIGN KEY (run_id) REFERENCES detection_runs (id),
        FOREIGN KEY (metrics_id) REFERENCES metrics (id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS label_metrics (
        label_id INTEGER NOT NULL,
        metrics_id INTEGER NOT NULL,
        PRIMARY KEY (label_id, metrics_id),
        FOREIGN KEY (label_id) REFERENCES labels (id),
        FOREIGN KEY (metrics_id) REFERENCES metrics (id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_detections_run_id ON detections (run_id)",
    "CREATE INDEX IF NOT EXISTS idx_detections_metrics_id ON detections (metrics_id)",
    "CREATE INDEX IF NOT EXISTS idx_labels_start_end ON labels (start_ts, end_ts)",
]

# features table columns are generated dynamically since they depend on
# NUMERIC_METRICS, see build_features_table_sql() below

LEGACY_TABLES_UNAMBIGUOUS = ["baseline_flags", "mahalanobis_flags"]


def build_features_table_sql(numeric_metrics: list) -> str:
    """
    Features table is one row per metrics row (metrics_id is both the
    primary key and the foreign key, a 1:1 relationship). Only computed
    columns live here, no raw metric values, since those already exist
    in metrics and duplicating them was most of the old bloat.
    """
    columns = ["metrics_id INTEGER PRIMARY KEY"]
    for metric in numeric_metrics:
        columns.append(f"{metric}_roll_mean REAL")
        columns.append(f"{metric}_roll_std REAL")
        columns.append(f"{metric}_roll_median REAL")
        columns.append(f"{metric}_roll_mad REAL")
        columns.append(f"{metric}_delta REAL")

    columns_sql = ",\n        ".join(columns)
    return f"""
    CREATE TABLE IF NOT EXISTS features (
        {columns_sql},
        FOREIGN KEY (metrics_id) REFERENCES metrics (id)
    )
    """


def create_schema(conn: sqlite3.Connection, numeric_metrics: list) -> None:
    """Creates every table in the new schema. Safe to run multiple times."""
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()

    cur.execute(build_features_table_sql(numeric_metrics))
    for statement in SCHEMA_STATEMENTS:
        cur.execute(statement)

    conn.commit()


def drop_legacy_tables(conn: sqlite3.Connection) -> None:
    """
    Drops the old bloated tables from before the redesign. Must run
    BEFORE create_schema(), since the old 'features' table shares its
    name with the new one — CREATE TABLE IF NOT EXISTS would otherwise
    silently keep the old bloated version in place. Only call this
    after confirming you have a backup, since it's destructive.

    baseline_flags and mahalanobis_flags are unambiguous legacy names,
    safe to drop outright. 'features' is NOT unconditionally dropped —
    that name is reused by the new thin table, so it's only dropped if
    it still has the old bloated signature (a bare 'cpu_percent' column,
    which the new schema never has). Dropping it unconditionally would
    wipe the current, correct features table on every re-run.
    """
    cur = conn.cursor()
    for table in LEGACY_TABLES_UNAMBIGUOUS:
        cur.execute(f"DROP TABLE IF EXISTS {table}")

    existing = {row[0] for row in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    if "features" in existing:
        cols = {row[1] for row in cur.execute("PRAGMA table_info(features)").fetchall()}
        if "cpu_percent" in cols:
            cur.execute("DROP TABLE IF EXISTS features")

    conn.commit()