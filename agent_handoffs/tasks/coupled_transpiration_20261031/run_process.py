"""Capture one independent verification/run process; no model calculations."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
TASK = Path(__file__).resolve().parent


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def identity(pid):
    raw = Path(f"/proc/{pid}/stat").read_text()
    return raw, raw.rsplit(")", 1)[1].split()[19]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--cwd", type=Path, default=ROOT)
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    if not command or Path(args.name).name != args.name or args.timeout <= 0:
        parser.error("name must be a basename, timeout positive, and command nonempty")
    prefix = TASK / args.name
    if prefix.with_suffix(".launch.json").exists():
        parser.error("run name already exists; preserve previous evidence")
    source_root = Path(os.environ.get("LEAF_SOURCE_ROOT", str(ROOT)))
    paths = [source_root / "benchmarks/daily_water_growth.py", source_root / "benchmarks/spatial_mapping.py",
             source_root / "leaf/coupling.py", source_root / "leaf/__init__.py",
             source_root / "native/maple_syrup/subsurface_transpiration.py",
             *(Path("/home/okin/SYRUP/src/maple_syrup") / n for n in ("subsurface.py", "subsurface_numba.py", "subsurface_cuda.py", "subsurface_transpiration.py"))]
    paths.extend(Path(a).resolve() for a in command if a.endswith(".json") and Path(a).is_file())
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    start = time.perf_counter()
    started_at = utc_now()
    timed_out = False
    with prefix.with_suffix(".stdout.log").open("w") as stdout, prefix.with_suffix(".stderr.log").open("w") as stderr:
        child = subprocess.Popen(command, cwd=args.cwd, env=os.environ.copy(), stdout=stdout, stderr=stderr,
                                 start_new_session=True)
        stat, ticks = identity(child.pid)
        launch = {"argv": command, "cwd": str(args.cwd.resolve()), "pid": child.pid, "proc_stat_at_launch": stat,
                  "start_ticks": ticks, "started_at_utc": started_at, "timeout_seconds": args.timeout,
                  "source_and_case_sha256": hashes,
                  "environment": {k: os.environ.get(k) for k in (
                      "PYTHONPATH", "LEAF_SOURCE_ROOT", "CUDA_VISIBLE_DEVICES", "NUMBA_CACHE_DIR", "CUPY_CACHE_DIR", "CUDA_CACHE_PATH",
                      "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")}}
        prefix.with_suffix(".launch.json").write_text(json.dumps(launch, indent=2) + "\n")
        try:
            code = child.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            if identity(child.pid)[1] != ticks:
                raise RuntimeError("process identity changed; refusing signal")
            child.terminate()
            try:
                code = child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if identity(child.pid)[1] != ticks:
                    raise RuntimeError("process identity changed; refusing kill")
                child.kill()
                code = child.wait(timeout=10)
    result = {"returncode": code, "timed_out": timed_out, "child_reaped": child.poll() is not None,
              "ended_at_utc": utc_now(), "whole_child_wall_seconds": time.perf_counter() - start}
    prefix.with_suffix(".exit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"run": args.name, **result}), flush=True)
    return code if code else int(timed_out)


if __name__ == "__main__":
    raise SystemExit(main())
