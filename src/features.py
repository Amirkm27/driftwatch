import sqlite3
from pathlib import Path

import pandas as pd

from src.config import DB_PATH, NUMERIC_METRICS

ROLLING_WINDOW = 20  # ~10 min at 30s sampling


def load_all_metrics(db_path: Path = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM metrics ORDER BY timestamp ASC", conn)
    finally:
        conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def get_processed_metrics_ids(db_path: Path = DB_PATH) -> set:
    """Returns the set of metrics_id values that already have a features row."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute("SELECT metrics_id FROM features")
        return {row[0] for row in cur.fetchall()}
    except sqlite3.OperationalError:
        return set()
    finally:
        conn.close()


def compute_rolling_features(df: pd.DataFrame, cols: list, window: int) -> pd.DataFrame:
    """
    Adds rolling mean/std/median/mad and first-difference per metric.
    Rolling stats need enough trailing history to be meaningful, so this
    always computes over the FULL metrics history, not just the new
    rows — otherwise the first `window` new rows would have artificially
    thin context right after a batch boundary.
    """
    out = pd.DataFrame({"metrics_id": df["id"]})

    for col in cols:
        if col not in df.columns:
            print(f"skipping '{col}', not in metrics table")
            continue

        rolling = df[col].rolling(window=window, min_periods=1)
        out[f"{col}_roll_mean"] = rolling.mean()
        out[f"{col}_roll_std"] = rolling.std().fillna(0)
        out[f"{col}_roll_median"] = rolling.median()
        out[f"{col}_roll_mad"] = (
            rolling.apply(lambda x: (x - x.median()).abs().median(), raw=False).fillna(0)
        )
        out[f"{col}_delta"] = df[col].diff().fillna(0)

    return out


def build_features(db_path: Path = DB_PATH) -> int:
    """
    Computes features for any metrics rows not already in the features
    table, and appends just those. Returns the number of new rows written.
    """
    metrics_df = load_all_metrics(db_path)
    already_processed = get_processed_metrics_ids(db_path)

    features_df = compute_rolling_features(metrics_df, NUMERIC_METRICS, ROLLING_WINDOW)
    new_rows = features_df[~features_df["metrics_id"].isin(already_processed)]

    if new_rows.empty:
        print("no new metrics rows to process, features table is up to date")
        return 0

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        new_rows.to_sql("features", conn, if_exists="append", index=False)
    finally:
        conn.close()

    print(f"added {len(new_rows)} new rows to features")
    return len(new_rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build/update the features table")
    parser.add_argument("--db", type=Path, default=None,
                         help="Path to target database (defaults to config.DB_PATH)")
    args = parser.parse_args()

    build_features(db_path=args.db or DB_PATH)