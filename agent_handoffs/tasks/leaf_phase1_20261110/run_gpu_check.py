"""Abort CUDA checks on failed/non-idle inventory; launch only the selected idle GPU."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import subprocess
import sys

TASK = Path(__file__).resolve().parent


def main():
    name, cwd, *command = sys.argv[1:]
    if not name.replace('_', '').isalnum() or not command:
        raise ValueError('Expected unique check name, cwd and argv')
    index = 1
    query = ['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu',
             '--format=csv,noheader']
    capture = subprocess.run(query, text=True, capture_output=True)
    record = {'at_utc': datetime.now(timezone.utc).isoformat(), 'query': query,
              'query_returncode': capture.returncode, 'query_stdout': capture.stdout,
              'query_stderr': capture.stderr, 'physical_index': index,
              'purpose': 'bounded independent CUDA qualification; no transport scenario',
              'authority': 'User-authorized implementation and GPU qualification using an idle device',
              'launch_allowed': False}
    path = TASK / f'{name}.gpu_selection.json'
    if path.exists():
        raise RuntimeError('Preserve previous selection record')
    if capture.returncode:
        path.write_text(json.dumps(record, indent=2) + '\n')
        raise SystemExit('GPU inventory failed; no CUDA check launched')
    rows = [[c.strip() for c in line.split(',')] for line in capture.stdout.splitlines() if line.strip()]
    chosen = [r for r in rows if r[0] == str(index)]
    if len(chosen) != 1 or len(chosen[0]) != 6:
        path.write_text(json.dumps(record, indent=2) + '\n')
        raise SystemExit('Selected GPU absent or malformed inventory; no CUDA check launched')
    row = chosen[0]
    used_mib = int(row[4].split()[0])
    util_percent = int(row[5].split()[0])
    record.update(uuid=row[1], gpu_name=row[2], used_mib=used_mib, utilization_percent=util_percent)
    if used_mib != 0 or util_percent != 0:
        path.write_text(json.dumps(record, indent=2) + '\n')
        raise SystemExit('Selected GPU is no longer idle; no CUDA check launched')
    env = dict(os.environ)
    env.update(LEAF_PHASE0_GPU=str(index), CUPY_GPU_MEMORY_LIMIT='1500000000', PYTHONDONTWRITEBYTECODE='1')
    record.update(launch_allowed=True, pool_cap_bytes=1500000000,
                  cuda_visible_devices=str(index), requested_command=command, cwd=cwd)
    path.write_text(json.dumps(record, indent=2) + '\n')
    # No device import or allocation precedes the successful inventory above.
    return subprocess.call([sys.executable, str(TASK / 'run_check.py'), name, cwd, *command], env=env)


if __name__ == '__main__':
    raise SystemExit(main())
