"""
Run the repeatable DriftWatch analysis pipeline on the real and synthetic DBs.

  python run_pipeline.py                        re-run analysis on both DBs
  python run_pipeline.py --save                 also append results to evaluation_results.csv
  python run_pipeline.py --fresh                back up the old CSV, start a new one (implies --save)
  python run_pipeline.py --rebuild-synthetic    delete + regenerate the synthetic DB (implies --fresh)

Collection, migration and the notebooks stay manual.
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from src.config import DB_PATH, SYNTHETIC_DB_PATH

ROOT = Path(__file__).resolve().parent
RESULTS_CSV = DB_PATH.parent / "evaluation_results.csv"
STEPS = ["src.features", "src.build_label_bridge", "src.detectors",
         "src.evaluate", "src.check_label_overlap"]


def run(module, db=None, extra=()):
    cmd = [sys.executable, "-m", module]
    if db:
        cmd += ["--db", str(db)]
    cmd += list(extra)
    print(f"\n>>> {' '.join(cmd[2:])}")
    subprocess.run(cmd, cwd=ROOT, check=True)  # stop at the first failure


def analyze(db, save):
    for step in STEPS:
        run(step, db, ["--save"] if save and step == "src.evaluate" else [])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--save", action="store_true", help="append results to evaluation_results.csv")
    p.add_argument("--fresh", action="store_true", help="back up the old CSV, start a new one (implies --save)")
    p.add_argument("--rebuild-synthetic", action="store_true",
                   help="delete and regenerate the synthetic DB (implies --fresh)")
    a = p.parse_args()

    if a.rebuild_synthetic:
        a.fresh = True  # new synthetic DB puts anomalies on different rows; old synthetic rows are incomparable
    save = a.save or a.fresh

    if a.fresh and RESULTS_CSV.exists():
        backup = RESULTS_CSV.with_name(f"evaluation_results_{datetime.now():%Y%m%d_%H%M%S}.csv")
        RESULTS_CSV.rename(backup)
        print(f"old results backed up to {backup.name}")

    print("\n===== real DB =====")
    analyze(DB_PATH, save)  # real first: inject_anomalies needs its label bridge

    if a.rebuild_synthetic:
        if SYNTHETIC_DB_PATH.exists():
            SYNTHETIC_DB_PATH.unlink()  # close dashboard/notebooks first or Windows will refuse
            print(f"deleted {SYNTHETIC_DB_PATH.name}")
        run("src.inject_anomalies",
            extra=["--source", str(DB_PATH), "--target", str(SYNTHETIC_DB_PATH)])

    if not SYNTHETIC_DB_PATH.exists():
        sys.exit("synthetic DB missing: re-run with --rebuild-synthetic")

    print("\n===== synthetic DB =====")
    analyze(SYNTHETIC_DB_PATH, save)


if __name__ == "__main__":
    main()