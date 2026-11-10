"""Stop only the identity-verified task writer to deliver concrete review feedback."""
from pathlib import Path
import datetime
import json
import os
import signal

TASK = Path(__file__).resolve().parent
record = json.loads((TASK / 'launch_02.json').read_text())
proc = Path('/proc') / str(record['pid'])
result = {'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'session_id': record['session_id'], 'pid': record['pid'],
          'start_ticks': record['start_ticks'],
          'reason': 'All six candidate files delivered; independent review found a concrete failing test and '
                    'two validation/report corrections. Interrupt the unfinished final-report generation '
                    'and resume this exact session with bounded review feedback.',
          'changes_preserved': True}
try:
    stat = (proc / 'stat').read_text()
    args = (proc / 'cmdline').read_bytes().split(b'\0')
    assert stat.rsplit(')', 1)[1].split()[19] == record['start_ticks'], 'start identity changed; no signal'
    assert record['session_id'].encode() in args, 'session UUID absent; no signal'
    os.kill(record['pid'], signal.SIGTERM)
    result['action'] = 'SIGTERM to exact recorded writer; original launcher must reap it'
except (FileNotFoundError, ProcessLookupError):
    result['action'] = 'writer already exited; no signal'
(TASK / 'review_interruption_02.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
