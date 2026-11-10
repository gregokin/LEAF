"""Execute a bounded stable-snapshot test/fit with separate streams and identity."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parent
OLD = TASK.parent / "paired_growth_et_survey_20261101/public_evidence/selected"
parser = argparse.ArgumentParser()
parser.add_argument("job", choices=["tests_02", "fit_02"])
args = parser.parse_args()
job = args.job
if (TASK / (job + ".exit.json")).exists():
    raise FileExistsError("Preserve completed checks")
if job.startswith("tests"):
    argv = [sys.executable, "-m", "pytest", str(TASK / "review_snapshot_02/test_study_parameters.py"), "-q", "-p", "no:cacheprovider"]
else:
    argv = [sys.executable, str(TASK / "review_snapshot_02/study_parameters.py"),
            "--images", str(TASK / "public_evidence/setaria_image_scalars.csv"),
            "--weights", str(OLD / "setaria_snapshots.csv"), "--harvest", str(OLD / "setaria_harvest.csv"),
            "--zoom", str(TASK / "public_evidence/zoom_calibration_data.txt"),
            "--peterson", str(TASK / "public_evidence/peterson_figure_readings.json"),
            "--out", str(TASK / "fit_02"), "--n-boot", "100", "--qc-day-label", "end"]
env = os.environ.copy()
env.update(PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", MPLCONFIGDIR="/tmp/leaf-study-data/mpl",
           OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
started = datetime.now(timezone.utc).isoformat()
clock = time.perf_counter()
with (TASK / (job + ".stdout.txt")).open("x") as out, (TASK / (job + ".stderr.txt")).open("x") as err:
    child = subprocess.Popen(argv, env=env, stdout=out, stderr=err, text=True)
    stat = Path(f"/proc/{child.pid}/stat").read_text()
    (TASK / (job + ".launch.json")).write_text(json.dumps({"argv": argv, "cwd": str(Path.cwd()), "pid": child.pid,
        "start_ticks": stat.rsplit(")", 1)[1].split()[19], "started_utc": started,
        "scope": "Synthetic analysis tests or 17-33 DAP public-data parameter fit; no native hydrology/transport scenario."}, indent=2) + "\n")
    print("Started", job, "PID", child.pid, flush=True)
    code = child.wait()
record = {"returncode": code, "child_reaped": True, "ended_utc": datetime.now(timezone.utc).isoformat(),
          "elapsed_seconds": time.perf_counter() - clock}
(TASK / (job + ".exit.json")).write_text(json.dumps(record, indent=2) + "\n")
print(job, json.dumps(record), flush=True)
print((TASK / (job + ".stdout.txt")).read_text()[-2000:])
print((TASK / (job + ".stderr.txt")).read_text()[-2000:])
raise SystemExit(code)
