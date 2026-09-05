import sqlite3

from src.config import DB_PATH


def dedupe_labels(conn: sqlite3.Connection) -> None:
    """
    Keeps the lowest id for each (start_ts, end_ts, label) combination
    and deletes the rest, along with their now-orphaned label_metrics
    entries. Needed once for data imported before the UNIQUE constraint
    existed — new imports can't duplicate anymore.
    """
    dupes = conn.execute(
        """
        SELECT id FROM labels
        WHERE id NOT IN (
            SELECT MIN(id) FROM labels GROUP BY start_ts, end_ts, label
        )
        """
    ).fetchall()
    dupe_ids = [row[0] for row in dupes]

    if not dupe_ids:
        print("no duplicate labels found")
        return

    placeholders = ", ".join("?" * len(dupe_ids))
    conn.execute(f"DELETE FROM label_metrics WHERE label_id IN ({placeholders})", dupe_ids)
    conn.execute(f"DELETE FROM labels WHERE id IN ({placeholders})", dupe_ids)
    conn.commit()
    print(f"removed {len(dupe_ids)} duplicate label rows: {dupe_ids}")


def dedupe_detection_runs(conn: sqlite3.Connection) -> None:
    """Keeps only the most recent detection_runs row per method, deletes the rest and their detections."""
    methods = [row[0] for row in conn.execute("SELECT DISTINCT method FROM detection_runs").fetchall()]

    for method in methods:
        runs = conn.execute(
            "SELECT id FROM detection_runs WHERE method = ? ORDER BY computed_at DESC",
            (method,),
        ).fetchall()
        run_ids = [row[0] for row in runs]

        if len(run_ids) <= 1:
            continue

        stale_ids = run_ids[1:]  # keep the newest, drop the rest
        placeholders = ", ".join("?" * len(stale_ids))
        conn.execute(f"DELETE FROM detections WHERE run_id IN ({placeholders})", stale_ids)
        conn.execute(f"DELETE FROM detection_runs WHERE id IN ({placeholders})", stale_ids)
        print(f"{method}: removed {len(stale_ids)} stale run(s), kept run_id={run_ids[0]}")

    conn.commit()


def run_cleanup(db_path=DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        dedupe_labels(conn)
        dedupe_detection_runs(conn)
    finally:
        conn.close()
    print("cleanup complete")


if __name__ == "__main__":
    run_cleanup()