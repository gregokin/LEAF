"""Run one explicit verification command with separate streams and a reaped exit record."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


task = Path(__file__).resolve().parent
name = sys.argv[1]
cfg = json.loads((task / f'{name}.command.json').read_text())
if (task / f'{name}.launch.json').exists():
    raise RuntimeError('Preserve prior verification records')
started = now()
tick = time.perf_counter()
env = os.environ.copy()
env.update(cfg.get('env', {}))
with (task / f'{name}.stdout.log').open('x') as out, (task / f'{name}.stderr.log').open('x') as err:
    child = subprocess.Popen(cfg['argv'], cwd=cfg['cwd'], env=env, stdout=out, stderr=err)
    identity = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    (task / f'{name}.launch.json').write_text(json.dumps({
        'argv': cfg['argv'], 'cwd': cfg['cwd'], 'env': cfg.get('env', {}),
        'pid': child.pid, 'start_ticks': identity, 'started_at_utc': started,
    }, indent=2) + '\n')
    print(f'{name}: PID {child.pid}', flush=True)
    code = child.wait(timeout=cfg.get('timeout_seconds', 1800))
record = {'returncode': code, 'child_reaped': child.poll() is not None,
          'elapsed_seconds': time.perf_counter() - tick, 'ended_at_utc': now()}
(task / f'{name}.exit.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
raise SystemExit(code)
