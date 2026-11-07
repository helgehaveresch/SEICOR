"""Run plume_preprocessor.py for several days, one after the other, in the active Python environment.

Usage:
    python run_plume_preprocessor_days.py                   all days in DAYS
    python run_plume_preprocessor_days.py 250412 250514     only these days
"""
import os
import subprocess
import sys
import time
from pathlib import Path

DAYS = ["250412", "250514", "250520", "250530", "250605", "250619"]
PREPROCESSOR = Path(__file__).resolve().parent / "plume_preprocessor.py"

days = sys.argv[1:] or DAYS
env = {**os.environ, "MPLBACKEND": "Agg"}  # save the plots without opening windows

results = {}
for day in days:
    print(f"=== {day} ===", flush=True)
    start = time.time()
    exit_code = subprocess.run([sys.executable, str(PREPROCESSOR), day], env=env).returncode  # a failed day does not stop the others
    results[day] = (exit_code, (time.time() - start) / 60)

print("\n=== Summary ===")
for day, (exit_code, minutes) in results.items():
    print(f"{day}: {'ok' if exit_code == 0 else f'FAILED (exit code {exit_code})'}, {minutes:.1f} min")

sys.exit(any(exit_code != 0 for exit_code, _ in results.values()))
