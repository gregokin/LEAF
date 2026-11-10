"""Launch/resume only the bounded generic candidate with ordinary file permissions."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parent


def main():
    # Look up the old command by its unique session UUID across host-visible processes.
    # PID namespaces can differ between managed executions; UUID AND start ticks must match.
    if int(sys.argv[1]) == 2:
        old = json.loads((TASK / 'launch_01.json').read_text())
        observations = []
        for proc in Path('/proc').glob('[0-9]*'):
            try:
                args = (proc / 'cmdline').read_bytes().split(b'\0')
                if old['session_id'].encode() not in args:
                    continue
                stat = (proc / 'stat').read_text()
                ticks = stat.rsplit(')', 1)[1].split()[19]
                if ticks != old['start_ticks']:
                    raise RuntimeError('UUID matched but start identity differs; refusing signal')
                pid = int(proc.name)
                os.kill(pid, signal.SIGTERM)
                gone = False
                for _ in range(100):
                    if not proc.exists():
                        gone = True
                        break
                    try:
                        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
                    except FileNotFoundError:
                        gone = True
                        break
                    if fields[0] == 'Z' or fields[19] != ticks:
                        gone = True
                        break
                    time.sleep(0.1)
                if not gone:
                    raise RuntimeError('original exact session remains live; refusing second writer')
                observations.append({'host_pid': pid, 'start_ticks': ticks, 'signalled': 'SIGTERM', 'ceased_execution': gone})
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
        (TASK / 'previous_session_preflight_02.json').write_text(json.dumps({
            'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'session_id': old['session_id'], 'observations': observations,
            'result': 'no executing original session remains among host-visible processes',
            'reaped_by_this_process': False}, indent=2) + '\n')
    revision = int(sys.argv[1])
    stem = f'{revision:02d}'
    assert not (TASK / f'launch_{stem}.json').exists(), 'preserve existing launch evidence'
    cfg = json.loads((TASK / 'session_config_02.json').read_text())
    candidate = Path(cfg['cwd'])
    prompt_path = TASK / f'prompt_{stem}.md'
    prompt = prompt_path.read_text()
    write_paths = ['benchmarks/daily_water_growth.py', 'leaf/__init__.py', 'leaf/coupling.py', 'leaf/stepwat.py', 'tests/test_stepwat.py', 'docs/stepwat-growth.md']
    rules = [f'Read(/{candidate}/**)']
    for rel in write_paths:
        rules += [f'Edit(/{candidate}/{rel})', f'Write(/{candidate}/{rel})']
    argv = ['claude', '--safe-mode', '--print', '--verbose', '--output-format', 'stream-json',
            '--session-id' if revision == 2 else '--resume', cfg['session_id'], '--effort', 'high',
            '--tools', 'Read,Edit,Write', '--allowedTools', *rules, '--permission-mode', 'dontAsk',
            '--permission-prompts', 'none', '--strict-mcp-config', '--system-prompt',
            'Implement the supplied bounded growth adapter in the isolated candidate only. Read only candidate files. No shell, outside files, private context or execution claims. Follow the supplied numerical contract and preserve legacy defaults; report limitations.']
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t = time.perf_counter()
    timeout = False
    with (TASK / f'claude_{stem}.stdout.jsonl').open('w') as out, (TASK / f'claude_{stem}.stderr.log').open('w') as err:
        child = subprocess.Popen(argv, cwd=candidate, stdin=subprocess.PIPE, stdout=out, stderr=err,
                                 text=True, start_new_session=True)
        stat = Path(f'/proc/{child.pid}/stat').read_text()
        ticks = stat.rsplit(')', 1)[1].split()[19]
        launch = {'argv': argv, 'cwd': str(candidate), 'session_id': cfg['session_id'],
                  'pid': child.pid, 'start_ticks': ticks, 'proc_stat_at_launch': stat,
                  'started_at_utc': started, 'prompt': str(prompt_path),
                  'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                  'safe_mode': True, 'permission_bypass': False, 'tools': ['Read','Edit','Write'],
                  'approved_candidate_writes': write_paths, 'timeout_seconds': cfg['timeout_seconds'],
                  'submitted_context': 'User-approved copied candidate code, bounded growth specification and corrections; no upstream source, full project brief, datasets or credentials.'}
        (TASK / f'launch_{stem}.json').write_text(json.dumps(launch, indent=2)+'\n')
        try:
            child.communicate(prompt, timeout=cfg['timeout_seconds'])
        except subprocess.TimeoutExpired:
            timeout = True
            if Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19] != ticks:
                raise RuntimeError('process identity changed; refusing signal')
            child.terminate()
            try:
                child.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                if Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19] != ticks:
                    raise RuntimeError('process identity changed; refusing kill')
                child.kill()
                child.communicate(timeout=10)
    end = {'returncode': child.returncode, 'timed_out': timeout, 'child_reaped': child.poll() is not None,
           'ended_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'elapsed_seconds': time.perf_counter()-t, 'session_id': cfg['session_id']}
    (TASK / f'exit_{stem}.json').write_text(json.dumps(end,indent=2)+'\n')
    print(json.dumps(end),flush=True)
    return child.returncode or (1 if timeout else 0)


if __name__=='__main__':
    raise SystemExit(main())
