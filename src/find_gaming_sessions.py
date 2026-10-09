import sqlite3
import pandas as pd

from src.config import DB_PATH

GPU_SUSTAINED_MIN = 90
GPU_SUSTAINED_MAX = 100
GPU_LAUNCH_MIN = 40
GAP_THRESHOLD_SECONDS = 120
SESSION_MERGE_GAP_SECONDS = 300

MIN_SESSION_SECONDS = 60
MIN_SUSTAINED_ROWS = 3
PROCESS_KEYWORD_DEFAULT = "rainbow"  # case-insensitive


def load_raw_metrics(db_path: str = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM metrics ORDER BY timestamp ASC", conn)
    finally:
        conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def matches_process(row, keyword: str) -> bool:
    name_cols = [
        "top_cpu_1_name", "top_cpu_2_name", "top_cpu_3_name",
        "top_ram_1_name", "top_ram_2_name", "top_ram_3_name",
    ]
    for col in name_cols:
        val = row.get(col)
        if isinstance(val, str) and keyword.lower() in val.lower():
            return True
    return False


def find_gaming_sessions(df: pd.DataFrame, process_keyword: str = PROCESS_KEYWORD_DEFAULT) -> pd.DataFrame:
    sustained_mask = df["gpu_percent"].between(GPU_SUSTAINED_MIN, GPU_SUSTAINED_MAX)
    process_mask = df.apply(lambda r: matches_process(r, process_keyword), axis=1)
    launch_mask = df["gpu_percent"] >= GPU_LAUNCH_MIN

    candidate_mask = sustained_mask | process_mask | launch_mask
    candidates = df[candidate_mask].copy().sort_values("timestamp").reset_index(drop=True)
    candidates["is_sustained"] = sustained_mask[candidate_mask].values
    candidates["process_match"] = process_mask[candidate_mask].values

    if candidates.empty:
        print("no candidate rows found")
        return pd.DataFrame(columns=[
            "start_ts", "end_ts", "duration_s", "process_confirmed", "includes_launch_activity"
        ])

    gaps = candidates["timestamp"].diff().dt.total_seconds().fillna(0)
    tight_cluster_id = (gaps > GAP_THRESHOLD_SECONDS).cumsum()
    candidates["tight_cluster_id"] = tight_cluster_id

    clusters = []
    for cid, group in candidates.groupby("tight_cluster_id"):
        sustained_rows = group.loc[group["is_sustained"], "timestamp"]
        clusters.append({
            "start": group["timestamp"].min(),
            "end": group["timestamp"].max(),
            "n_sustained": group["is_sustained"].sum(),
            "process_match": group["process_match"].any(),
            "first_sustained_ts": sustained_rows.min() if not sustained_rows.empty else pd.NaT,
        })

    sessions = []
    i = 0
    while i < len(clusters):
        c = clusters[i]
        if c["n_sustained"] >= MIN_SUSTAINED_ROWS:
            session_start = c["start"]
            first_sustained_ts = c["first_sustained_ts"]

            j = i - 1
            while j >= 0 and clusters[j]["n_sustained"] < MIN_SUSTAINED_ROWS:
                gap = (clusters[j + 1]["start"] - clusters[j]["end"]).total_seconds()
                if gap > SESSION_MERGE_GAP_SECONDS:
                    break
                session_start = clusters[j]["start"]
                j -= 1

            duration = (c["end"] - session_start).total_seconds()
            if duration >= MIN_SESSION_SECONDS:
                includes_launch = (
                        pd.notna(first_sustained_ts) and session_start < first_sustained_ts
                )
                sessions.append({
                    "start_ts": session_start,
                    "end_ts": c["end"],
                    "duration_s": duration,
                    "process_confirmed": bool(c["process_match"]),
                    "includes_launch_activity": includes_launch,
                })
        i += 1

    return pd.DataFrame(sessions)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Find gaming sessions")
    parser.add_argument("--process", type=str, default=PROCESS_KEYWORD_DEFAULT,
                        help="Process name substring to match")
    args = parser.parse_args()

    df = load_raw_metrics()
    sessions = find_gaming_sessions(df, process_keyword=args.process)
    if sessions.empty:
        print("no gaming sessions found matching criteria")
    else:
        print(sessions.to_string(index=False))