import sqlite3
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "driftwatch.db"

GPU_SUSTAINED_MIN = 90       # gpu load that counts as "actually playing"
GPU_SUSTAINED_MAX = 100
GPU_LAUNCH_MIN = 40          # looser gpu floor for pre-game loading/menu screens
GAP_THRESHOLD_SECONDS = 120       # gap larger than this splits into a new candidate cluster
SESSION_MERGE_GAP_SECONDS = 300   # clusters within this gap get merged into one session
                                   # (covers loading screens / menus where gpu drops to 0
                                   # and the process briefly isn't top of any list)
MIN_SESSION_SECONDS = 60     # drop short blips that aren't real sessions
MIN_SUSTAINED_ROWS = 3       # cluster must contain at least this many high-gpu rows to count as real
PROCESS_KEYWORD = "rainbow"  # case-insensitive match against top process columns


def load_raw_metrics(db_path: str = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM metrics ORDER BY timestamp ASC", conn)
    finally:
        conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def matches_process(row, keyword: str) -> bool:
    # checks all top_cpu / top_ram name slots, since the game might show
    # up in either depending on what it's heaviest on at that moment
    name_cols = [
        "top_cpu_1_name", "top_cpu_2_name", "top_cpu_3_name",
        "top_ram_1_name", "top_ram_2_name", "top_ram_3_name",
    ]
    for col in name_cols:
        val = row.get(col)
        if isinstance(val, str) and keyword.lower() in val.lower():
            return True
    return False


def find_gaming_sessions(df: pd.DataFrame) -> pd.DataFrame:
    sustained_mask = df["gpu_percent"].between(GPU_SUSTAINED_MIN, GPU_SUSTAINED_MAX)
    process_mask = df.apply(lambda r: matches_process(r, PROCESS_KEYWORD), axis=1)
    launch_mask = df["gpu_percent"] >= GPU_LAUNCH_MIN

    # candidate rows are anything that could plausibly be part of a gaming
    # session: sustained high gpu, OR the process shows up in the top
    # lists at all, OR gpu is elevated but not yet at "actually playing"
    # levels (covers loading screens and launch spikes that happen before
    # gpu load settles into the sustained range)
    candidate_mask = sustained_mask | process_mask | launch_mask
    candidates = df[candidate_mask].copy().sort_values("timestamp").reset_index(drop=True)
    candidates["is_sustained"] = sustained_mask[candidate_mask].values
    candidates["process_match"] = process_mask[candidate_mask].values

    if candidates.empty:
        print("no candidate rows found")
        return pd.DataFrame(columns=[
            "start_ts", "end_ts", "duration_s", "process_confirmed", "includes_launch_activity"
        ])

    # stage 1: tight clustering. a menu screen or loading pause can drop
    # gpu to 0 and briefly push the game out of the top process slots,
    # which would otherwise fracture one real session into several
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

    # stage 2: attach launch-only lead-ins to the session they belong to.
    # a cluster that never reaches MIN_SUSTAINED_ROWS on its own is
    # treated as a launch/loading lead-in and pulled into the *next*
    # cluster if it qualifies as a real session. two clusters that BOTH
    # independently qualify are never merged together, since that would
    # collapse two genuinely separate sessions (e.g. quit and relaunched
    # later) into one
    sessions = []
    i = 0
    while i < len(clusters):
        c = clusters[i]
        if c["n_sustained"] >= MIN_SUSTAINED_ROWS:
            session_start = c["start"]
            first_sustained_ts = c["first_sustained_ts"]

            # walk backward through preceding non-qualifying clusters and
            # pull them in as lead-in, as long as the gap to the next
            # cluster in the chain is short enough to plausibly be a
            # loading pause rather than a real break
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
    df = load_raw_metrics()
    sessions = find_gaming_sessions(df)
    if sessions.empty:
        print("no gaming sessions found matching criteria")
    else:
        print(sessions.to_string(index=False))


window = df[(df["timestamp"] >= "2026-09-02T19:47:00") & (df["timestamp"] <= "2026-09-02T19:53:00")]
print(window[["timestamp", "gpu_percent", "cpu_percent", "top_cpu_1_name", "top_ram_1_name"]].to_string())