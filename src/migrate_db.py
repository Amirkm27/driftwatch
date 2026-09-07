import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import DB_PATH, NUMERIC_METRICS, LABELS_CSV_PATH
from src.db_schema import create_schema, drop_legacy_tables


def backup_database(db_path: Path) -> Path:
    """
    Copies the current db file before any destructive changes. Named
    with a timestamp so repeated migration attempts don't overwrite an
    earlier backup.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.parent / f"driftwatch_pre_migration_{timestamp}.db"
    shutil.copy2(db_path, backup_path)
    print(f"backup written to {backup_path}")
    return backup_path


def import_labels(conn: sqlite3.Connection, csv_path: Path) -> None:
    """
    Loads labels.csv into the labels table. Uses INSERT OR IGNORE against
    the (start_ts, end_ts, label) uniqueness constraint, so running the
    migration again after already importing labels doesn't duplicate
    every row — safe to re-run whenever labels.csv changes.
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
    Reclaims disk space after dropping tables. SQLite doesn't shrink the
    file automatically when data is deleted, VACUUM rebuilds the file
    without the freed space.
    """
    conn.execute("VACUUM")


def has_legacy_tables(conn: sqlite3.Connection) -> bool:
    """
    Checks whether the old, pre-migration versions of these tables still
    exist. baseline_flags/mahalanobis_flags are unambiguous — they only
    ever existed in the old bloated schema. features needs a closer look
    since that name is reused by the new thin table: the old version
    stored raw metric values directly (e.g. a bare 'cpu_percent' column),
    the new one only stores derived columns keyed by metrics_id, so a
    bare 'cpu_percent' column is what marks it as the legacy version.
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

        # only back up when something destructive is actually about to
        # happen. schema creation and label import are both safe to
        # re-run (IF NOT EXISTS / INSERT OR IGNORE), so running this
        # repeatedly after the first successful migration shouldn't
        # pile up backup files for no reason
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