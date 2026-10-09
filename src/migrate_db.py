import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import DB_PATH, NUMERIC_METRICS, LABELS_CSV_PATH
from src.db_schema import create_schema, drop_legacy_tables


def backup_database(db_path: Path) -> Path:
    """
    Create a timestamped backup before migration.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.parent / f"driftwatch_pre_migration_{timestamp}.db"
    shutil.copy2(db_path, backup_path)
    print(f"backup written to {backup_path}")
    return backup_path


def import_labels(conn: sqlite3.Connection, csv_path: Path) -> None:
    """
    Import labels from CSV without creating duplicates.
    """
    if not csv_path.exists():
        print(f"no labels.csv found at {csv_path}, skipping label import")
        return

    labels_df = pd.read_csv(csv_path)
    rows = list(labels_df[["start_ts", "end_ts", "label", "source", "confidence", "notes"]]
                .itertuples(index=False, name=None))

    cur = conn.executemany(
        """
        INSERT OR IGNORE INTO labels (start_ts, end_ts, label, source, confidence, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    print(f"processed {len(rows)} label rows from csv, {cur.rowcount if cur.rowcount != -1 else 'some'} newly inserted")


def vacuum(conn: sqlite3.Connection) -> None:
    """
    Reclaim unused SQLite space.
    """
    conn.execute("VACUUM")


def has_legacy_tables(conn: sqlite3.Connection) -> bool:
    """
    Check if the database uses the old schema.
    """
    existing = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}

    if "baseline_flags" in existing or "mahalanobis_flags" in existing:
        return True

    if "features" in existing:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(features)").fetchall()}
        if "cpu_percent" in cols:
            return True

    return False


def run_migration(db_path: Path = DB_PATH, labels_path: Path = LABELS_CSV_PATH) -> None:
    if not db_path.exists():
        raise FileNotFoundError(f"no database found at {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")

        # Backup only when legacy tables exist.
        needs_backup = has_legacy_tables(conn)
        if needs_backup:
            conn.close()
            backup_database(db_path)
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA foreign_keys = ON")
        else:
            print("no legacy tables found, skipping backup (nothing destructive to do)")

        size_before = db_path.stat().st_size
        drop_legacy_tables(conn)
        create_schema(conn, NUMERIC_METRICS)
        import_labels(conn, labels_path)
        vacuum(conn)
        size_after = db_path.stat().st_size

        print(f"db size: {size_before / 1024:.1f} KB -> {size_after / 1024:.1f} KB")
    finally:
        conn.close()

    print("migration complete. run features.py then detectors.py to repopulate.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Migrate the database to the current schema")
    parser.add_argument("--db", type=Path, default=None,
                         help="Path to target database (defaults to config.DB_PATH)")
    parser.add_argument("--labels-csv", type=Path, default=None,
                         help="Path to labels.csv (defaults to config.LABELS_CSV_PATH)")
    args = parser.parse_args()

    run_migration(
        db_path=args.db or DB_PATH,
        labels_path=args.labels_csv or LABELS_CSV_PATH,
    )