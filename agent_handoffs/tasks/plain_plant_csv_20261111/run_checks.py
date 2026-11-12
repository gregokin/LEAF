"""Codex independent CPU verification; no case evolution or benchmarks."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import subprocess
import time

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[2]

def now():
    return datetime.now(timezone.utc).isoformat()

def main():
    label = os.environ.get('LEAF_CHECK_LABEL', 'cpu_suite_01')
    argv = ['/home/okin/MAPLE/.venv-cupy/bin/python', '-m', 'unittest', 'discover', '-s', 'tests', '-v']
    env = dict(os.environ, PYTHONPATH=str(ROOT) + ':/home/okin/SYRUP/src',
               NUMBA_CACHE_DIR='/tmp/leaf-plain-csv-numba', CUDA_VISIBLE_DEVICES='',
               PYTHONDONTWRITEBYTECODE='1')
    start = time.perf_counter()
    with (TASK / (label + '.stdout.log')).open('x') as out, (TASK / (label + '.stderr.log')).open('x') as err:
        proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=out, stderr=err)
        ticks = Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
        (TASK / (label + '.launch.json')).write_text(json.dumps(dict(argv=argv,cwd=str(ROOT),pid=proc.pid,
            start_ticks=ticks,started_at_utc=now(),environment={k:env[k] for k in ['PYTHONPATH','NUMBA_CACHE_DIR','CUDA_VISIBLE_DEVICES','PYTHONDONTWRITEBYTECODE']}),indent=2)+'\n')
        print('Independent CPU suite started: PID',proc.pid,flush=True)
        code=proc.wait()
    result=dict(returncode=code,child_reaped=True,ended_at_utc=now(),elapsed_seconds=time.perf_counter()-start)
    (TASK / (label + '.exit.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return code

if __name__=='__main__':
    raise SystemExit(main())
