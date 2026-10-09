import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
)

from src.config import DB_PATH, ANOMALY_LABELS, NORMAL_LABELS

# Known anomaly labels are treated as the positive class.
# Normal workloads (for example gaming_gpu_load) and unlabeled rows
# are treated as negative examples.


def load_ground_truth(db_path: Path) -> set:
    """Return metrics ids that belong to confirmed anomaly labels."""
    conn = sqlite3.connect(db_path)
    try:
        labels = pd.read_sql_query("SELECT id, label FROM labels", conn)
        bridge = pd.read_sql_query("SELECT label_id, metrics_id FROM label_metrics", conn)
    finally:
        conn.close()

    merged = bridge.merge(labels, left_on="label_id", right_on="id")
    return set(
        merged.loc[merged["label"].isin(ANOMALY_LABELS), "metrics_id"]
    )


def load_latest_run_detections(db_path: Path, method: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT metrics_id, score, is_anomaly
            FROM detections
            WHERE run_id = (
                SELECT id
                FROM detection_runs
                WHERE method = ?
                ORDER BY computed_at DESC
                LIMIT 1
            )
            """,
            conn,
            params=(method,),
        )
    finally:
        conn.close()

    df["is_anomaly"] = df["is_anomaly"].astype(bool)
    return df


def evaluate_method(db_path: Path, method: str) -> dict:
    detections = load_latest_run_detections(db_path, method)
    anomaly_ids = load_ground_truth(db_path)

    y_true = detections["metrics_id"].isin(anomaly_ids).astype(int)
    y_pred = detections["is_anomaly"].astype(int)
    y_score = detections["score"]

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")

    return {
        "method": method,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
        "n_rows": len(detections),
        "n_true_anomalies": int(y_true.sum()),
    }


def print_report(result: dict) -> None:
    print(f"=== {result['method']} ===")
    print(
        f"  TP={result['tp']}  FP={result['fp']}  "
        f"TN={result['tn']}  FN={result['fn']}"
    )
    print(
        f"  precision={result['precision']:.3f}  "
        f"recall={result['recall']:.3f}  "
        f"f1={result['f1']:.3f}  "
        f"roc_auc={result['roc_auc']:.3f}"
    )
    print(f"  ({result['n_true_anomalies']}/{result['n_rows']} rows are true anomalies)")
    print()


def save_results(results: list, db_path: Path, output_path: Path) -> None:
    """Append evaluation results to CSV."""
    rows = []
    computed_at = datetime.now(timezone.utc).isoformat()

    for result in results:
        rows.append(
            {
                **result,
                "db_path": str(db_path),
                "computed_at": computed_at,
            }
        )

    df = pd.DataFrame(rows)
    header_needed = not output_path.exists()

    df.to_csv(
        output_path,
        mode="a",
        header=header_needed,
        index=False,
    )

    print(f"results appended to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate detectors against labeled ground truth"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to target database (defaults to config.DB_PATH)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Append results to data/evaluation_results.csv",
    )

    args = parser.parse_args()
    db_path = (args.db or DB_PATH).resolve()

    all_results = []

    for method in ["zscore", "mahalanobis", "cusum", "ocsvm"]:
        result = evaluate_method(db_path, method)
        print_report(result)
        all_results.append(result)

    if args.save:
        output_path = db_path.parent / "evaluation_results.csv"
        save_results(all_results, db_path, output_path)
