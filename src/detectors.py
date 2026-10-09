import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from src.config import DB_PATH, NUMERIC_METRICS, ANOMALY_LABELS, NORMAL_LABELS

# Detection thresholds
Z_THRESHOLD = 3.0
Z_CLIP = 20.0
MAD_SCALE = 1.4826
EPSILON = 1e-6
MAHALANOBIS_CONFIDENCE = 0.997
COV_REGULARIZATION = 1e-6

# CUSUM currently runs on CPU only
CUSUM_METRICS = ["cpu_percent"]
CUSUM_K_MULT = 2.5
CUSUM_H_MULT = 10.0
CUSUM_MIN_BASELINE_ROWS = 10

# OCSVM parameter from bakeoff tuning
OCSVM_NU = 0.02

TRIMMED_FEATURE_SUFFIXES = ("_roll_mean", "_delta")


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


def get_labeled_metrics_ids(db_path: Path) -> set:
    """
    Return every metrics_id covered by any label.

    Kept for CUSUM. It returns all labeled rows because CUSUM was
    originally tuned with all labels excluded.
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute("SELECT DISTINCT metrics_id FROM label_metrics")
        return {row[0] for row in cur.fetchall()}
    except sqlite3.OperationalError:
        return set()
    finally:
        conn.close()

def get_metrics_ids_for_labels(db_path: Path, label_names: set) -> set:
    """
    Return metric ids belonging to selected labels.

    Used to remove known bad events from baseline fitting while keeping
    valid workloads in the normal profile.
    """
    if not label_names:
        return set()

    conn = sqlite3.connect(db_path)
    try:
        placeholders = ", ".join("?" * len(label_names))
        rows = conn.execute(
            f"""
            SELECT DISTINCT lm.metrics_id
            FROM label_metrics lm
            JOIN labels l ON l.id = lm.label_id
            WHERE l.label IN ({placeholders})
            """,
            tuple(label_names),
        ).fetchall()

        return {row[0] for row in rows}

    except sqlite3.OperationalError:
        return set()

    finally:
        conn.close()




def get_baseline_exclusions(db_path: Path, labels: set) -> set:
    """
    Return metric ids excluded from baseline fitting.

    Warn when expected labels are missing or nothing is excluded.
    """
    excluded_ids = get_metrics_ids_for_labels(db_path, labels)

    conn = sqlite3.connect(db_path)
    try:
        existing_labels = {
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT label FROM labels"
            ).fetchall()
        }
    except sqlite3.OperationalError:
        existing_labels = set()
    finally:
        conn.close()

    unclassified = existing_labels - ANOMALY_LABELS - NORMAL_LABELS

    if unclassified:
        print(
            "WARNING: labels not classified as anomaly or normal: "
            f"{sorted(unclassified)}"
        )

    if not excluded_ids and existing_labels & labels:
        print(
            "WARNING: matching anomaly labels exist but no rows were "
            "excluded; check label_metrics"
        )

    return excluded_ids


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

    bad_metric_ids = get_baseline_exclusions(db_path, ANOMALY_LABELS)

    fit_mask = ~df["metrics_id"].isin(bad_metric_ids)

    print(f"fitting on {fit_mask.sum()}/{len(df)} baseline rows excluding "
          f"{len(bad_metric_ids)} known anomaly events")

    dist_sq, n_features = compute_mahalanobis(df, NUMERIC_METRICS, fit_mask)
    threshold = chi2.ppf(confidence, df=n_features)
    is_anomaly = dist_sq > threshold

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        run_id = start_run(conn, method="mahalanobis", threshold=threshold,
                           notes=f"chi2 confidence={confidence}")

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


def compute_cusum(values: np.ndarray, mu: float, sigma: float,
                  k_mult: float = CUSUM_K_MULT, h_mult: float = CUSUM_H_MULT):
    """
    Two-sided CUSUM for detecting sustained shifts in a single metric.
    It accumulates deviations from the baseline instead of judging rows
    individually, allowing it to detect gradual changes that point-wise
    detectors may miss.

    Returns anomaly scores and flags. The score is the accumulated deviation
    normalized by the detection threshold.
    """
    k = k_mult * sigma
    h = h_mult * sigma

    pos = np.zeros(len(values))
    neg = np.zeros(len(values))
    for i in range(1, len(values)):
        pos[i] = max(0.0, pos[i - 1] + values[i] - mu - k)
        neg[i] = min(0.0, neg[i - 1] + values[i] - mu + k)

    flag = (pos > h) | (neg < -h)
    score = np.maximum(pos, -neg) / h
    return score, flag


def run_cusum_detector(db_path: Path = DB_PATH, k_mult: float = CUSUM_K_MULT,
                       h_mult: float = CUSUM_H_MULT) -> None:
    """
    Run CUSUM on configured raw metrics.

    CUSUM excludes known anomaly events from the baseline.
    Legitimate workloads such as gaming stay in the baseline.
    """
    conn = sqlite3.connect(db_path)
    try:
        cols = ", ".join(["id AS metrics_id", "timestamp"] + CUSUM_METRICS)
        df = pd.read_sql_query(f"SELECT {cols} FROM metrics ORDER BY timestamp ASC", conn)
    finally:
        conn.close()

    excluded_ids = get_baseline_exclusions(
        db_path,
        ANOMALY_LABELS
    )
    fit_mask = ~df["metrics_id"].isin(excluded_ids)

    composite_score = np.zeros(len(df))
    composite_flag = np.zeros(len(df), dtype=bool)
    skipped_metrics = []

    for col in CUSUM_METRICS:
        if col not in df.columns:
            continue

        baseline = df.loc[fit_mask, col].dropna()
        if len(baseline) < CUSUM_MIN_BASELINE_ROWS:
            skipped_metrics.append(col)
            continue

        mu, sigma = baseline.mean(), baseline.std()
        if sigma < EPSILON:
            skipped_metrics.append(col)
            continue

        values = df[col].ffill().bfill().to_numpy()
        score, flag = compute_cusum(values, mu, sigma, k_mult, h_mult)
        composite_score = np.maximum(composite_score, score)
        composite_flag = composite_flag | flag

    if skipped_metrics:
        print(f"cusum: skipped {skipped_metrics} (too few clean baseline rows or ~zero variance)")

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        run_id = start_run(conn, method="cusum", threshold=h_mult,
                           notes=f"k_mult={k_mult}, h_mult={h_mult}")
        save_detections(conn, run_id, df["metrics_id"].tolist(),
                        composite_score.tolist(), composite_flag.tolist())
    finally:
        conn.close()

    n_flagged = composite_flag.sum()
    print(f"cusum run {run_id}: {n_flagged}/{len(df)} rows flagged ({n_flagged / len(df):.1%})")


def get_trimmed_feature_cols(df: pd.DataFrame, metrics: list) -> list:
    """roll_mean + delta per metric only — see TRIMMED_FEATURE_SUFFIXES."""
    cols = []
    for metric in metrics:
        for suffix in TRIMMED_FEATURE_SUFFIXES:
            col = f"{metric}{suffix}"
            if col in df.columns:
                cols.append(col)
    return cols


def run_ocsvm_detector(db_path: Path = DB_PATH, nu: float = OCSVM_NU) -> None:
    """
    One-Class SVM on the trimmed roll_mean + delta features. Fit on all rows
    except known anomaly events (ANOMALY_LABELS); legitimate heavy workloads
    such as gaming_gpu_load stay in the baseline. Fit-set history: fitting on
    every row let anomalies be absorbed into "normal"; excluding all labeled
    rows (gaming too) scored poorly; excluding anomaly labels only gave real
    F1 around 0.43. Any further fit-strategy change needs a nu re-sweep
    against gaming / cpu_stress / ram_spike.
    """
    df = load_joined_data(db_path)
    feature_cols = get_trimmed_feature_cols(df, NUMERIC_METRICS)
    if not feature_cols:
        raise ValueError("no roll_mean/delta feature columns found — run features.py first")

    data = df[feature_cols]
    complete_mask = data.notna().all(axis=1)

    bad_metric_ids = get_baseline_exclusions(db_path, ANOMALY_LABELS)

    fit_mask = complete_mask & ~df["metrics_id"].isin(bad_metric_ids)

    scored_df = df[complete_mask]

    if fit_mask.sum() < len(feature_cols) + 1:
        raise ValueError(
            f"not enough clean rows to fit OCSVM, "
            f"got {fit_mask.sum()} for {len(feature_cols)} features"
        )

    scaler = StandardScaler()
    scaler.fit(data[fit_mask])

    X_scaled = scaler.transform(data[complete_mask])

    model = OneClassSVM(nu=nu)
    model.fit(scaler.transform(data[fit_mask]))

    pred = model.predict(X_scaled)

    decision = model.decision_function(X_scaled)
    score = -decision
    is_anomaly = pred == -1

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        run_id = start_run(conn, method="ocsvm", threshold=0.0, notes=f"nu={nu}")
        save_detections(conn, run_id, scored_df["metrics_id"].tolist(),
                        score.tolist(), is_anomaly.tolist())
    finally:
        conn.close()

    n_fit = fit_mask.sum()
    n_scored = complete_mask.sum()
    n_flagged = is_anomaly.sum()

    print(
        f"ocsvm run {run_id}: fit on {n_fit} baseline rows, "
        f"scored {n_scored} rows, "
        f"{n_flagged}/{n_scored} flagged ({n_flagged / n_scored:.1%})"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run DriftWatch core anomaly detectors"
    )
    parser.add_argument("--db", type=Path, default=None,
                        help="Path to target database (defaults to config.DB_PATH)")
    args = parser.parse_args()
    db_path = args.db or DB_PATH

    run_zscore_detector(db_path=db_path)
    run_mahalanobis_detector(db_path=db_path)
    run_cusum_detector(db_path=db_path)
    run_ocsvm_detector(db_path=db_path)