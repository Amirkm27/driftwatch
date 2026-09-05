import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from src.config import DB_PATH, NUMERIC_METRICS

Z_THRESHOLD = 3.0
Z_CLIP = 20.0  # hard ceiling on any single metric's z-score, well past significance
MAD_SCALE = 1.4826  # scales MAD to be consistent with std under a normal distribution
EPSILON = 1e-6
MAHALANOBIS_CONFIDENCE = 0.997
COV_REGULARIZATION = 1e-6


def load_joined_data(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Joins metrics with their features, one row per timestamp with everything needed to score it."""
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT metrics.*, features.*
            FROM metrics
            JOIN features ON metrics.id = features.metrics_id
            ORDER BY metrics.timestamp ASC
            """,
            conn,
        )
    finally:
        conn.close()
    return df


def start_run(conn: sqlite3.Connection, method: str, threshold: float, notes: str = "") -> int:
    """
    Creates a new detection_runs row for this method, after removing any
    previous run(s) for the same method. Keeping only the latest run per
    method (rather than a full history) matches how this pipeline is
    actually used — re-running a detector means "redo this," not "add
    another version to compare later."
    """
    old_run_ids = [row[0] for row in conn.execute(
        "SELECT id FROM detection_runs WHERE method = ?", (method,)
    ).fetchall()]

    if old_run_ids:
        placeholders = ", ".join("?" * len(old_run_ids))
        conn.execute(f"DELETE FROM detections WHERE run_id IN ({placeholders})", old_run_ids)
        conn.execute(f"DELETE FROM detection_runs WHERE id IN ({placeholders})", old_run_ids)

    computed_at = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "INSERT INTO detection_runs (method, threshold_used, computed_at, notes) VALUES (?, ?, ?, ?)",
        (method, threshold, computed_at, notes),
    )
    conn.commit()
    return cur.lastrowid


def save_detections(conn: sqlite3.Connection, run_id: int, metrics_ids, scores, flags) -> None:
    rows = list(zip([run_id] * len(metrics_ids), metrics_ids, scores, [int(f) for f in flags]))
    conn.executemany(
        "INSERT INTO detections (run_id, metrics_id, score, is_anomaly) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Z-score detector
# ---------------------------------------------------------------------------

def compute_zscore_composite(df: pd.DataFrame, cols: list) -> pd.Series:
    """
    Robust z-score per metric using rolling median/MAD, floored by global
    std to prevent runaway values on metrics that sit at zero most of
    the time (disk/net rates), then reduced to one composite score per
    row as the max absolute z-score across all metrics.
    """
    z_scores = pd.DataFrame(index=df.index)

    for col in cols:
        median_col = f"{col}_roll_median"
        mad_col = f"{col}_roll_mad"
        if median_col not in df.columns or mad_col not in df.columns:
            continue

        global_std = df[col].std()
        floor = max(global_std * 0.1, EPSILON)

        scaled_mad = (df[mad_col] * MAD_SCALE).clip(lower=floor)
        z = (df[col] - df[median_col]) / scaled_mad
        z_scores[col] = z.clip(-Z_CLIP, Z_CLIP)

    return z_scores.abs().max(axis=1)


def run_zscore_detector(db_path: Path = DB_PATH, threshold: float = Z_THRESHOLD) -> None:
    df = load_joined_data(db_path)
    composite = compute_zscore_composite(df, NUMERIC_METRICS)
    is_anomaly = composite > threshold

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        run_id = start_run(conn, method="zscore", threshold=threshold)
        save_detections(conn, run_id, df["metrics_id"].tolist(), composite.tolist(), is_anomaly.tolist())
    finally:
        conn.close()

    print(f"zscore run {run_id}: {is_anomaly.sum()}/{len(df)} rows flagged ({is_anomaly.mean():.1%})")


# ---------------------------------------------------------------------------
# Mahalanobis detector
# ---------------------------------------------------------------------------

def compute_mahalanobis(df: pd.DataFrame, cols: list, regularization: float = COV_REGULARIZATION):
    """
    Squared Mahalanobis distance against the dataset's own mean vector
    and covariance matrix. Fit on whatever data is passed in, so if
    you're evaluating against injected anomalies later, fit this on a
    clean split first rather than the full mixed dataset.
    """
    available_cols = [c for c in cols if c in df.columns]
    data = df[available_cols].copy()

    complete_mask = data.notna().all(axis=1)
    fit_data = data[complete_mask].to_numpy()

    mean_vec = fit_data.mean(axis=0)
    cov = np.cov(fit_data, rowvar=False)
    cov_reg = cov + np.eye(cov.shape[0]) * regularization
    inv_cov = np.linalg.pinv(cov_reg)

    diffs = data.to_numpy() - mean_vec
    dist_sq = np.einsum("ij,jk,ik->i", diffs, inv_cov, diffs)
    dist_sq[~complete_mask.to_numpy()] = np.nan

    return pd.Series(dist_sq, index=df.index), len(available_cols)


def run_mahalanobis_detector(db_path: Path = DB_PATH, confidence: float = MAHALANOBIS_CONFIDENCE) -> None:
    df = load_joined_data(db_path)
    dist_sq, n_features = compute_mahalanobis(df, NUMERIC_METRICS)
    threshold = chi2.ppf(confidence, df=n_features)
    is_anomaly = dist_sq > threshold

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        run_id = start_run(conn, method="mahalanobis", threshold=threshold,
                            notes=f"chi2 confidence={confidence}")
        # rows with missing metric values didn't get scored, don't insert nulls
        scored_mask = dist_sq.notna()
        save_detections(
            conn, run_id,
            df.loc[scored_mask, "metrics_id"].tolist(),
            dist_sq[scored_mask].tolist(),
            is_anomaly[scored_mask].tolist(),
        )
    finally:
        conn.close()

    n_scored = dist_sq.notna().sum()
    print(f"mahalanobis run {run_id}: threshold={threshold:.2f}, "
          f"{is_anomaly.sum()}/{n_scored} scored rows flagged ({is_anomaly.sum() / n_scored:.1%})")


if __name__ == "__main__":
    run_zscore_detector()
    run_mahalanobis_detector()