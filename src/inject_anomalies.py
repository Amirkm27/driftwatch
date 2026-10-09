import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DB_PATH, NUMERIC_METRICS, SYNTHETIC_DB_PATH
from src.db_schema import create_schema

SEED = 42
MIN_GAP_ROWS = 5  # gap between injected events

# number and duration of injected events
INJECTION_SPECS = {
    "synthetic_cpu_spike": {"count": 8, "duration_range": (4, 8)},
    "synthetic_mem_leak": {"count": 5, "duration_range": (20, 40)},
    "synthetic_disk_thrash": {"count": 8, "duration_range": (4, 8)},
}


def load_real_metrics(db_path: Path = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM metrics ORDER BY timestamp ASC", conn)
    finally:
        conn.close()
    return df.reset_index(drop=True)


def load_real_labeled_row_positions(db_path: Path, df: pd.DataFrame) -> set:
    """Return rows already covered by real labels."""
    conn = sqlite3.connect(db_path)
    try:
        labeled_ids = {row[0] for row in conn.execute(
            "SELECT DISTINCT metrics_id FROM label_metrics"
        ).fetchall()}
    except sqlite3.OperationalError:
        labeled_ids = set()
    finally:
        conn.close()

    id_to_pos = {mid: pos for pos, mid in enumerate(df["id"])}
    return {id_to_pos[mid] for mid in labeled_ids if mid in id_to_pos}


def pick_injection_windows(n_rows: int, count: int, duration_range: tuple,
                            excluded_positions: set, rng: np.random.Generator) -> list:
    """Pick injection windows without overlap."""
    windows = []
    attempts = 0
    max_attempts = count * 200

    while len(windows) < count and attempts < max_attempts:
        attempts += 1
        duration = int(rng.integers(duration_range[0], duration_range[1] + 1))
        start = int(rng.integers(MIN_GAP_ROWS, n_rows - duration - MIN_GAP_ROWS))
        end = start + duration

        buffered = set(range(start - MIN_GAP_ROWS, end + MIN_GAP_ROWS))
        if buffered & excluded_positions:
            continue

        windows.append((start, end))
        excluded_positions |= buffered

    if len(windows) < count:
        print(f"warning: only placed {len(windows)}/{count} windows after {max_attempts} attempts, "
              f"dataset may be too small or too much space already excluded")

    return windows


def inject_cpu_spike(df: pd.DataFrame, start: int, end: int, rng: np.random.Generator) -> None:
    df.loc[start:end - 1, "cpu_percent"] = rng.uniform(92, 100, end - start)


def inject_mem_leak(df: pd.DataFrame, start: int, end: int, rng: np.random.Generator) -> None:
    baseline = df.loc[start, "ram_percent"]
    target = rng.uniform(92, 98)
    df.loc[start:end - 1, "ram_percent"] = np.linspace(baseline, target, end - start)


def inject_disk_thrash(df: pd.DataFrame, start: int, end: int, rng: np.random.Generator) -> None:
    n = end - start
    df.loc[start:end - 1, "disk_read_mb_s"] = rng.uniform(50, 150, n)
    df.loc[start:end - 1, "disk_write_mb_s"] = rng.uniform(50, 150, n)


INJECTORS = {
    "synthetic_cpu_spike": inject_cpu_spike,
    "synthetic_mem_leak": inject_mem_leak,
    "synthetic_disk_thrash": inject_disk_thrash,
}


def build_synthetic_dataset(source_db_path: Path = DB_PATH,
                             target_db_path: Path = SYNTHETIC_DB_PATH,
                             seed: int = SEED) -> None:
    rng = np.random.default_rng(seed)

    df = load_real_metrics(source_db_path)
    excluded = load_real_labeled_row_positions(source_db_path, df)

    synthetic_labels = []
    total_injected_rows = 0
    for label_name, spec in INJECTION_SPECS.items():
        windows = pick_injection_windows(
            len(df), spec["count"], spec["duration_range"], excluded, rng
        )
        injector = INJECTORS[label_name]

        for start, end in windows:
            injector(df, start, end, rng)
            total_injected_rows += end - start
            synthetic_labels.append({
                "start_ts": df.loc[start, "timestamp"],
                "end_ts": df.loc[end - 1, "timestamp"],
                "label": label_name,
                "source": "synthetic",
                "confidence": "ground_truth",
                "notes": f"injected, {end - start} rows",
            })

        print(f"{label_name}: placed {len(windows)}/{spec['count']} windows")

    # create fresh synthetic database
    conn_src = sqlite3.connect(source_db_path)
    system_info_df = pd.read_sql_query("SELECT * FROM system_info", conn_src)
    # preserve original table schema
    create_statements = conn_src.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' AND name IN ('metrics', 'system_info')"
    ).fetchall()
    conn_src.close()

    conn_tgt = sqlite3.connect(target_db_path)
    try:
        conn_tgt.execute("PRAGMA foreign_keys = ON")

        for table_name, stmt in create_statements:
            conn_tgt.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn_tgt.execute(stmt)

        # insert referenced table first
        system_info_df.to_sql("system_info", conn_tgt, if_exists="append", index=False)
        df.to_sql("metrics", conn_tgt, if_exists="append", index=False)
        create_schema(conn_tgt, NUMERIC_METRICS)

        conn_tgt.executemany(
            """
            INSERT INTO labels (start_ts, end_ts, label, source, confidence, notes)
            VALUES (:start_ts, :end_ts, :label, :source, :confidence, :notes)
            """,
            synthetic_labels,
        )
        conn_tgt.commit()
    finally:
        conn_tgt.close()

    n_anomalous_rows = total_injected_rows
    print(f"\ntotal injected labels: {len(synthetic_labels)}")
    print(f"total injected (actually anomalous) rows: {n_anomalous_rows}/{len(df)} "
          f"({n_anomalous_rows / len(df):.1%})")
    print(f"synthetic dataset written to {target_db_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the synthetic anomaly DB from a real DB")
    parser.add_argument("--source", type=Path, default=DB_PATH,
                         help="real DB to copy and inject into (defaults to config.DB_PATH)")
    parser.add_argument("--target", type=Path, default=SYNTHETIC_DB_PATH,
                         help="synthetic DB to create (defaults to config.SYNTHETIC_DB_PATH)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    build_synthetic_dataset(args.source, args.target, args.seed)