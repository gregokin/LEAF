"""Run independent per-case processes and preserve each run's receipts."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
TASK = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--select", nargs="+")
    args = parser.parse_args()
    cases = json.loads((TASK / "benchmark_cases.json").read_text())
    chosen = [c for c in cases if args.select is None or c["name"] in args.select]
    if not chosen or (args.select and len(chosen) != len(args.select)):
        parser.error("select must name existing, unique cases")
    for case in chosen:
        name = f'{args.backend}_{case["name"]}_{args.tag}'
        command = [sys.executable, str(TASK / "run_process.py"), "--name", name, "--timeout", "1800", "--",
                   sys.executable, str(TASK / "profile_daily_bounded.py"), "--cases-json",
                   str(TASK / "case_sets" / (case["name"] + ".json")), "--output",
                   str(ROOT / "benchmarks/results/species_root_profiles_20261031" / (name + ".json")),
                   "--backend", args.backend, "--years", "30", "--report-every", "365"]
        print(json.dumps({"event": "suite_case_start", "name": name}), flush=True)
        code = subprocess.run(command, cwd=ROOT).returncode
        if code:
            print(json.dumps({"event": "suite_stopped", "name": name, "returncode": code}), flush=True)
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
