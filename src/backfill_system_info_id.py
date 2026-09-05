import sqlite3

from src.config import DB_PATH


def backfill_system_info_id(db_path=DB_PATH) -> None:
    """
    One-time fix for rows collected before system_info_id existed. Only
    safe to run this way because every row so far came from a single
    machine — if you ever have metrics from more than one system_info
    entry with unknown boundaries, this needs a real per-row decision
    instead of a blanket assignment.
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT id, recorded_at, hostname FROM system_info").fetchall()

        if len(rows) == 0:
            raise RuntimeError("no system_info rows found, nothing to backfill against")
        if len(rows) > 1:
            raise RuntimeError(
                f"found {len(rows)} system_info rows, blanket backfill isn't safe here — "
                f"assign metrics.system_info_id manually per row instead"
            )

        system_info_id = rows[0][0]
        cur = conn.execute(
            "UPDATE metrics SET system_info_id = ? WHERE system_info_id IS NULL",
            (system_info_id,),
        )
        conn.commit()
        print(f"backfilled {cur.rowcount} metrics rows to system_info_id={system_info_id} "
              f"(hostname={rows[0][2]})")
    finally:
        conn.close()


if __name__ == "__main__":
    backfill_system_info_id()