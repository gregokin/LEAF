"""Launch/resume only the bounded generic candidate with ordinary file permissions."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parent


def main():
    revision = int(sys.argv[1])
    stem = f'{revision:02d}'
    assert not (TASK / f'launch_{stem}.json').exists(), 'preserve existing launch evidence'
    cfg = json.loads((TASK / 'session_config.json').read_text())
    candidate = Path(cfg['cwd'])
    prompt_path = TASK / f'prompt_{stem}.md'
    prompt = prompt_path.read_text()
    write_paths = ['benchmarks/daily_water_growth.py', 'leaf/__init__.py', 'leaf/coupling.py', 'leaf/stepwat.py', 'tests/test_stepwat.py', 'docs/stepwat-growth.md']
    rules = [f'Read(/{candidate}/**)']
    for rel in write_paths:
        rules += [f'Edit(/{candidate}/{rel})', f'Write(/{candidate}/{rel})']
    argv = ['claude', '--safe-mode', '--print', '--verbose', '--output-format', 'stream-json',
            '--session-id' if revision == 1 else '--resume', cfg['session_id'], '--effort', 'high',
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
                  'submitted_context': 'Original generic benchmark files and bounded numerical API facts only; no upstream source, project briefs, datasets or private ecology.'}
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
