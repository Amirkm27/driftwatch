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

def get_labeled_metrics_ids(db_path: Path) -> set:
    """
    Every metrics_id already covered by a label (real or synthetic), via
    the label_metrics bridge. Used to exclude known-anomalous rows from
    the Mahalanobis "normal" fit — otherwise the anomalies themselves get
    baked into the mean/covariance, which is exactly backwards.
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute("SELECT DISTINCT metrics_id FROM label_metrics")
        return {row[0] for row in cur.fetchall()}
    except sqlite3.OperationalError:
        return set()
    finally:
        conn.close()


def compute_mahalanobis(df: pd.DataFrame, cols: list, fit_mask: pd.Series,
                         regularization: float = COV_REGULARIZATION):
    """
    Squared Mahalanobis distance for every row, but the mean vector and
    covariance matrix are fit ONLY on rows where fit_mask is True — the
    known-clean subset. Fitting on the full mixed dataset (clean +
    anomalies) lets anomalies pull the mean/covariance toward themselves,
    which mutes exactly the deviations you're trying to detect. Scoring
    still happens for every row, clean or not, against this fixed fit.
    """
    available_cols = [c for c in cols if c in df.columns]
    data = df[available_cols].copy()

    complete_mask = data.notna().all(axis=1)
    fit_rows_mask = complete_mask & fit_mask.reindex(df.index, fill_value=False)
    fit_data = data[fit_rows_mask].to_numpy()

    if len(fit_data) < len(available_cols) + 1:
        raise ValueError(
            f"not enough clean rows to estimate a stable covariance matrix, "
            f"got {len(fit_data)} clean rows for {len(available_cols)} features"
        )

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

    labeled_ids = get_labeled_metrics_ids(db_path)
    fit_mask = ~df["metrics_id"].isin(labeled_ids)
    print(f"fitting on {fit_mask.sum()}/{len(df)} rows not covered by any label "
          f"({len(labeled_ids)} labeled rows excluded from fit)")

    dist_sq, n_features = compute_mahalanobis(df, NUMERIC_METRICS, fit_mask)
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
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run z-score and Mahalanobis detectors")
    parser.add_argument("--db", type=Path, default=None,
                         help="Path to target database (defaults to config.DB_PATH)")
    args = parser.parse_args()
    db_path = args.db or DB_PATH

    run_zscore_detector(db_path=db_path)
    run_mahalanobis_detector(db_path=db_path)