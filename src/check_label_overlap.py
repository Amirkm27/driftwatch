import sqlite3
from pathlib import Path

import pandas as pd

from src.config import DB_PATH, ANOMALY_LABELS, NORMAL_LABELS


def load_latest_run_detections(db_path: Path, method: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            '''
            SELECT metrics.id AS metrics_id,
                   metrics.timestamp,
                   detections.score,
                   detections.is_anomaly
            FROM detections
            JOIN metrics ON metrics.id = detections.metrics_id
            WHERE detections.run_id = (
                SELECT id
                FROM detection_runs
                WHERE method = ?
                ORDER BY computed_at DESC
                LIMIT 1
            )
            ORDER BY metrics.timestamp ASC
            ''',
            conn,
            params=(method,),
        )
    finally:
        conn.close()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["is_anomaly"] = df["is_anomaly"].astype(bool)
    return df


def load_labels(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        labels = pd.read_sql_query("SELECT * FROM labels", conn)
    finally:
        conn.close()

    labels["start_ts"] = pd.to_datetime(labels["start_ts"])
    labels["end_ts"] = pd.to_datetime(labels["end_ts"])
    return labels


def load_label_metrics_ids(db_path: Path, label_id: int) -> set:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT metrics_id FROM label_metrics WHERE label_id = ?",
            (label_id,),
        )
        return {row[0] for row in cur.fetchall()}
    finally:
        conn.close()


def check_overlap(db_path: Path, detections: pd.DataFrame, labels: pd.DataFrame, method_name: str) -> None:
    print(f"=== {method_name} ===")

    for _, row in labels.iterrows():
        metrics_ids = load_label_metrics_ids(db_path, row["id"])
        window = detections[detections["metrics_id"].isin(metrics_ids)]

        if window.empty:
            print(f"{row['label']} ({row['start_ts']} to {row['end_ts']}): no matching rows found")
            continue

        n_flagged = window["is_anomaly"].sum()
        n_total = len(window)
        max_score = window["score"].max()

        if row["label"] in ANOMALY_LABELS:
            expectation = "SHOULD be flagged"
        elif row["label"] in NORMAL_LABELS:
            expectation = "should NOT be flagged"
        else:
            expectation = "expectation unknown"

        print(f"{row['label']} [{expectation}]")
        print(f"  {n_flagged}/{n_total} rows flagged, max score = {max_score:.2f}")

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check detection results against known labels")
    parser.add_argument("--db", type=Path, default=None)
    args = parser.parse_args()

    db_path = args.db or DB_PATH
    labels = load_labels(db_path)

    for method in ["zscore", "mahalanobis", "cusum", "ocsvm"]:
        detections = load_latest_run_detections(db_path, method)
        check_overlap(db_path, detections, labels, method)
