#this is a script that runs the plume_preprocessor.py based on a list of dates 
import subprocess
import multiprocessing
from pathlib import Path
from datetime import datetime, timedelta
import os

def make_date_list(start_str='250326', end_str='250531'):
    start = datetime.strptime(start_str, '%y%m%d').date()
    end = datetime.strptime(end_str, '%y%m%d').date()
    dates = []
    cur = start
    while cur <= end:
        dates.append(cur.strftime('%y%m%d'))
        cur += timedelta(days=1)
    return dates

date_list_orig = make_date_list('250326', '260611')

def run_for_date(date):
    print(f"[PID {os.getpid()}] Running plume_preprocessor for date: {date}")
    try:
        cp = subprocess.run(
            ['python3', str(Path(__file__).resolve().with_name('plume_preprocessor.py')), date],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        print(cp.stdout or f"Finished plume_preprocessor for date: {date} (rc={cp.returncode})")
        if cp.returncode == 0 and "SKIPPED:" in (cp.stdout or ""):
            return date, "skipped"
        return date, cp.returncode
    except Exception as e:
        print(f"Error running date {date}: {e}")
        return date, 1

if __name__ == '__main__':
    processes = 20
    with multiprocessing.Pool(processes=processes) as pool:
        try:
            results = pool.map(run_for_date, date_list_orig)
        except KeyboardInterrupt:
            pool.terminate()
            raise

    skipped = [d for d, rc in results if rc == "skipped"]
    if skipped:
        print(f"Skipped {len(skipped)} dates without SD measurements:", skipped)
    failed = [d for d, rc in results if rc not in (0, "skipped")]
    if failed:
        print("The following dates failed:", failed)
        raise SystemExit(1)
    print("All dates processed successfully.")
