#this is a script that runs the plume_preprocessor.py based on a list of dates 
import subprocess
import multiprocessing
from pathlib import Path

date_list_orig = ['250417', '250418', '250421', '250423', '250430', '250520', '250522', '250524', '250527', '250604', '250712', '250812', '250824', '250905', ]

def run_for_date(date):
    print(f"[PID {Path('.').absolute()}] Running plume_preprocessor for date: {date}")
    try:
        cp = subprocess.run(
            ['python3', 'scripts/SEICOR/plume_preprocessor.py', date],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        print(cp.stdout or f"Finished plume_preprocessor for date: {date} (rc={cp.returncode})")
        return date, cp.returncode
    except Exception as e:
        print(f"Error running date {date}: {e}")
        return date, 1

if __name__ == '__main__':
    processes = len(date_list_orig)  # one worker per date
    with multiprocessing.Pool(processes=processes) as pool:
        try:
            results = pool.map(run_for_date, date_list_orig)
        except KeyboardInterrupt:
            pool.terminate()
            raise

    # summary
    failed = [d for d, rc in results if rc != 0]
    if failed:
        print("The following dates failed:", failed)
        raise SystemExit(1)
    print("All dates processed successfully.")