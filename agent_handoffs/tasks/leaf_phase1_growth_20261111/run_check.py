"""Capture independent checks without merging streams or hiding failures."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parent


def main():
    name, cwd, *command = sys.argv[1:]
    if not name.replace('_', '').isalnum() or not command:
        raise ValueError('Expected unique name, cwd, argv')
    env = dict(os.environ)
    device = env.get('LEAF_PHASE0_GPU', '')
    if device and not device.isdigit():
        raise ValueError('LEAF_PHASE0_GPU must be one explicit physical GPU index')
    env.update(PYTHONPATH=f'{cwd}:/home/okin/SYRUP/src',
               NUMBA_CACHE_DIR='/tmp/leaf-phase0-numba',
               CUPY_CACHE_DIR='/tmp/leaf-phase0-cupy',
               PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES=device)
    start = datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    with (TASK / f'{name}.stdout.log').open('x') as out, \
            (TASK / f'{name}.stderr.log').open('x') as err:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=out, stderr=err)
        ticks = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
        (TASK / f'{name}.launch.json').write_text(json.dumps({
            'command': command, 'cwd': cwd, 'pid': child.pid, 'start_ticks': ticks,
            'started_at_utc': start, 'overrides': {k: env[k] for k in
                ('PYTHONPATH', 'NUMBA_CACHE_DIR', 'CUPY_CACHE_DIR',
                 'PYTHONDONTWRITEBYTECODE', 'CUDA_VISIBLE_DEVICES')},
        }, indent=2) + '\n')
        print(f'{name} started, PID {child.pid}', flush=True)
        code = child.wait()
    record = {'returncode': code, 'child_reaped': child.poll() is not None,
              'ended_at_utc': datetime.now(timezone.utc).isoformat(),
              'elapsed_seconds': time.perf_counter() - clock}
    (TASK / f'{name}.exit.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'name': name, **record}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
