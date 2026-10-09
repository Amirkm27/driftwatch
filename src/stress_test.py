import multiprocessing
import time
from datetime import datetime, timezone

def burn_cpu(duration_seconds):
    end_time = time.time() + duration_seconds
    while time.time() < end_time:
        pass

if __name__ == "__main__":
    DURATION_SECONDS = 300  # 5 minutes
    n_cores = max(1, multiprocessing.cpu_count() - 2)

    print(f"start: {datetime.now(timezone.utc).isoformat()}")
    processes = [multiprocessing.Process(target=burn_cpu, args=(DURATION_SECONDS,)) for _ in range(n_cores)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print(f"end: {datetime.now(timezone.utc).isoformat()}")