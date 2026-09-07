import sqlite3

from src.config import DB_PATH


def build_label_bridge(db_path=DB_PATH) -> None:
    """
    For every row in labels, finds the metrics rows whose timestamp falls
    inside that label's [start_ts, end_ts] window and records the pairing
    in label_metrics. Uses INSERT OR IGNORE so re-running after adding a
    new label doesn't duplicate pairs already resolved for existing ones.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        labels = conn.execute("SELECT id, start_ts, end_ts, label FROM labels").fetchall()

        total_pairs = 0
        for label_id, start_ts, end_ts, label_name in labels:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO label_metrics (label_id, metrics_id)
                SELECT ?, id FROM metrics
                WHERE timestamp >= ? AND timestamp <= ?
                """,
                (label_id, start_ts, end_ts),
            )
            print(f"{label_name} (label_id={label_id}): {cur.rowcount} metrics rows linked")
            total_pairs += cur.rowcount

        conn.commit()
        print(f"total: {total_pairs} label-metrics pairs")
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Populate the label_metrics bridge table")
    parser.add_argument("--db", type=Path, default=None,
                         help="Path to target database (defaults to config.DB_PATH)")
    args = parser.parse_args()

    build_label_bridge(db_path=args.db or DB_PATH)