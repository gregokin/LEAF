"""Run the approved Phase 0 candidate writer with bounded ordinary file tools."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parent
WRITE_FILES = [
    'leaf/__init__.py', 'leaf/traits.py', 'leaf/state.py', 'leaf/canopy.py',
    'leaf/initialization.py', 'leaf/recording.py', 'leaf/events.py',
    'leaf/runtime.py', 'leaf/__main__.py', 'tests/test_phase0.py',
    'docs/phase0-vegetation.md', 'docs/maple-leaf-integration.md',
    'examples/phase0/species.json', 'examples/phase0/plants.csv',
    'examples/phase0/generation.json', 'pyproject.toml', 'report.md',
]


def now():
    return datetime.now(timezone.utc).isoformat()


def start_ticks(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]


def main():
    revision = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    cfg = json.loads((TASK / 'session.json').read_text())
    cwd = Path(cfg['cwd'])
    prompt_path = TASK / f'prompt_{revision:02d}.md'
    prompt = prompt_path.read_text()
    launch_path = TASK / f'launch_{revision:02d}.json'
    if launch_path.exists():
        raise FileExistsError('Preserve earlier launch records')
    rules = [f'Read(/{cwd}/**)', f'Read(/{prompt_path})']
    if revision > 1:
        previous = json.loads((TASK / f'exit_{revision - 1:02d}.json').read_text())
        if not previous.get('child_reaped'):
            raise RuntimeError('Previous writer not reaped; refusing a second writer')
        for name in ('review_initial.md', 'review_probes_01.py',
                     'review_geometry_recording_01.py', 'independent_reference.py',
                     'generation_oracles.py', 'early_state_probes.stdout.log',
                     'early_geometry_recording.stdout.log',
                     'diagnostic_generation.stderr.log', 'candidate_phase0_01.stderr.log'):
            rules.append(f'Read(/{TASK / name})')
    if revision > 2:
        for name in ('candidate_phase0_02.stderr.log', 'checkpoint_state_02.stderr.log',
                     'checkpoint_memory_replay_02.stdout.log',
                     'cover_memory_replay_oracles.py', 'transfer_preflight_oracle.py',
                     'transfer_preflight_02.stdout.log'):
            rules.append(f'Read(/{TASK / name})')
    for rel in WRITE_FILES:
        rules.extend((f'Write(/{cwd / rel})', f'Edit(/{cwd / rel})'))
    session_args = (['--session-id', cfg['session_id']] if revision == 1
                    else ['--resume', cfg['session_id']])
    argv = ['claude', '--safe-mode', '--print', '--verbose', '--output-format',
            'stream-json', *session_args, '--effort', 'high', '--tools',
            'Read,Edit,Write', '--allowedTools', *rules, '--permission-mode',
            'dontAsk', '--permission-prompts', 'none', '--strict-mcp-config',
            '--system-prompt',
            'You are the bounded LEAF Phase 0 implementer. Follow the embedded '
            'Codex-Claude instructions and exact candidate file scope. Only '
            'ordinary Read/Edit/Write tools; no shell, execution, external '
            'context, agents, network tools, installs or Git. Report checks '
            'honestly as not executed.']
    started = now()
    clock = time.perf_counter()
    timed_out = False
    with (TASK / f'claude_{revision:02d}.stdout.jsonl').open('x') as out, \
            (TASK / f'claude_{revision:02d}.stderr.log').open('x') as err:
        child = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.PIPE, stdout=out,
                                 stderr=err, text=True, start_new_session=True)
        ticks = start_ticks(child.pid)
        launch_path.write_text(json.dumps({
            'argv': argv, 'cwd': str(cwd), 'session_id': cfg['session_id'],
            'pid': child.pid, 'start_ticks': ticks, 'started_at_utc': started,
            'prompt_file': prompt_path.name,
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'allowed_tools': ['Read', 'Edit', 'Write'],
            'write_files': WRITE_FILES, 'permission_bypass': False,
            'sharing_authority': 'User Phase 0 proceed; bounded specification '
                                 'and listed LEAF source files only.',
        }, indent=2) + '\n')
        print(f'Claude revision {revision} launched: PID {child.pid}, '
              f'session {cfg["session_id"]}', flush=True)
        try:
            child.communicate(prompt, timeout=cfg['timeout_seconds'])
        except subprocess.TimeoutExpired:
            timed_out = True
            if start_ticks(child.pid) != ticks:
                raise RuntimeError('Process identity changed; refusing signal')
            child.terminate()
            try:
                child.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                if start_ticks(child.pid) != ticks:
                    raise RuntimeError('Process identity changed; refusing kill')
                child.kill()
                child.communicate(timeout=10)
    record = {
        'session_id': cfg['session_id'], 'returncode': child.returncode,
        'timed_out': timed_out, 'child_reaped': child.poll() is not None,
        'ended_at_utc': now(), 'elapsed_seconds': time.perf_counter() - clock,
    }
    (TASK / f'exit_{revision:02d}.json').write_text(
        json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
    return child.returncode or (1 if timed_out else 0)


if __name__ == '__main__':
    raise SystemExit(main())
