"""Launch the bounded Phase 3 writer with exact session/process evidence."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parent
WRITE_FILES = json.loads((TASK/'session.json').read_text())['write_files']


def now():
    return datetime.now(timezone.utc).isoformat()


def ticks(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]


def main():
    rev = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    cfg = json.loads((TASK/'session.json').read_text())
    cwd = Path(cfg['cwd']); prompt_path = TASK/f'prompt_{rev:02d}.md'
    prompt = prompt_path.read_text()
    if (TASK/f'launch_{rev:02d}.json').exists():
        raise RuntimeError('Preserve prior launch records')
    if rev > 1:
        previous = json.loads((TASK/f'exit_{rev-1:02d}.json').read_text())
        assert previous['child_reaped'], 'Prior writer is not reaped'
    rules = [f'Read(/{cwd}/**)', f'Read(/{prompt_path})']
    for rel in WRITE_FILES:
        rules += [f'Edit(/{cwd/rel})', f'Write(/{cwd/rel})']
    session = ['--session-id' if rev == 1 else '--resume', cfg['session_id']]
    argv = ['claude', '--safe-mode', '--print', '--verbose', '--output-format',
            'stream-json', *session, '--effort', cfg.get('effort', 'high'), '--tools', 'Read,Edit,Write',
            '--allowedTools', *rules, '--permission-mode', 'dontAsk',
            '--permission-prompts', 'none', '--strict-mcp-config', '--system-prompt',
            'You implement the bounded LEAF Phase 3 task. Follow the embedded '
            'Codex-Claude instructions and exact candidate scope. Only ordinary Read/Edit/Write; '
            'no execution, agents, network tools, settings, installs or Git. Report checks honestly.']
    if cfg.get('model'):
        argv += ['--model', cfg['model']]
    started = now(); start = time.perf_counter(); timed_out = False
    with (TASK/f'claude_{rev:02d}.stdout.jsonl').open('x') as out, (TASK/f'claude_{rev:02d}.stderr.log').open('x') as err:
        child = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.PIPE, stdout=out, stderr=err,
                                 text=True, start_new_session=True)
        identity = ticks(child.pid)
        (TASK/f'launch_{rev:02d}.json').write_text(json.dumps({
            'argv': argv, 'cwd': str(cwd), 'session_id': cfg['session_id'],
            'pid': child.pid, 'start_ticks': identity, 'started_at_utc': started,
            'prompt_file': prompt_path.name, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'allowed_tools': ['Read','Edit','Write'], 'write_files': WRITE_FILES,
            'permission_bypass': False,
            'sharing_authority': 'User explicit automatic Codex-Claude Phases1-3 implementation and source sharing.'
        }, indent=2)+'\n')
        print(f'Claude revision {rev} started: PID {child.pid}, session {cfg["session_id"]}', flush=True)
        try:
            child.communicate(prompt, timeout=cfg['timeout_seconds'])
        except subprocess.TimeoutExpired:
            timed_out = True
            assert ticks(child.pid) == identity
            child.terminate()
            try:
                child.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                assert ticks(child.pid) == identity
                child.kill(); child.communicate(timeout=10)
    record = {'session_id': cfg['session_id'], 'returncode': child.returncode,
              'timed_out': timed_out, 'child_reaped': child.poll() is not None,
              'ended_at_utc': now(), 'elapsed_seconds': time.perf_counter()-start}
    (TASK/f'exit_{rev:02d}.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record), flush=True)
    return child.returncode or (1 if timed_out else 0)


if __name__ == '__main__':
    raise SystemExit(main())
