"""Audit completed writer scope and freeze an immutable-by-convention review tree."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys

TASK = Path(__file__).resolve().parent


def files(root):
    result = {}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        raw = path.read_bytes()
        result[path.relative_to(root).as_posix()] = {
            'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}
    return result


def main():
    rev = int(sys.argv[1])
    cfg = json.loads((TASK / 'session.json').read_text())
    done = json.loads((TASK / f'exit_{rev:02d}.json').read_text())
    assert done['child_reaped'], 'Writer has not been reaped'
    candidate = Path(cfg['cwd']).resolve()
    write_files = set(cfg['write_files'])
    baseline = json.loads((TASK / cfg.get('baseline_file', 'baseline.json')).read_text())['candidate_initial']
    current = files(candidate)
    changed = sorted(k for k in set(baseline) | set(current) if baseline.get(k) != current.get(k))
    violations = [f'Unallowed source change: {k}' for k in changed if k not in write_files]
    calls, errors, results = {}, [], []
    for line in (TASK / f'claude_{rev:02d}.stdout.jsonl').open():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get('type') == 'result':
            results.append({k: row.get(k) for k in ('subtype', 'is_error', 'result', 'session_id')})
        for block in row.get('message', {}).get('content', []):
            if block.get('type') == 'tool_use':
                name, arg = block['name'], block.get('input', {})
                path = arg.get('file_path')
                entry = {'name': name, 'file_path': path}
                calls[block['id']] = entry
                if name not in ('Read', 'Edit', 'Write'):
                    violations.append(f'Unallowed tool {name}')
                if path:
                    resolved = (candidate / path).resolve()
                    if name == 'Read':
                        if (resolved != candidate and candidate not in resolved.parents
                                and resolved != (TASK / f'prompt_{rev:02d}.md').resolve()):
                            violations.append(f'Unallowed read: {path}')
                    elif candidate not in resolved.parents or resolved.relative_to(candidate).as_posix() not in write_files:
                        violations.append(f'Unallowed write: {path}')
            if block.get('type') == 'tool_result' and block.get('is_error'):
                errors.append({'classification': 'tool_error',
                               'call': calls.get(block.get('tool_use_id')),
                               'literal_error': block.get('content')})
    final = results[-1] if results else None
    successful = (done['returncode'] == 0 and not done['timed_out'] and final
                  and final['subtype'] == 'success' and final['is_error'] is False
                  and bool(final.get('result')))
    audit = {'revision': rev, 'at_utc': datetime.now(timezone.utc).isoformat(),
             'session_id': cfg['session_id'], 'changed_files': changed,
             'scope_violations': violations, 'intermediate_tool_errors': errors,
             'process': done, 'final_result': final, 'structured_completion_success': bool(successful),
             'report_present': 'report.md' in current,
             'note': 'Intermediate read failures remain failures requiring explicit disposition; '
                     'structured completion alone is not task or phase acceptance.'}
    (TASK / f'scope_{rev:02d}.json').write_text(json.dumps(audit, indent=2) + '\n')
    if violations or not successful or not audit['report_present']:
        raise SystemExit('Writer scope/completion audit failed; preserve partial work')
    destination = TASK / f'snapshot_{rev:02d}'
    if destination.exists():
        raise RuntimeError('Preserve prior snapshot')
    destination.mkdir()
    for rel in current:
        path = destination / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(candidate / rel, path)
    if files(candidate) != current or files(destination) != current:
        raise RuntimeError('Candidate changed during freeze')
    (TASK / f'snapshot_{rev:02d}.manifest.json').write_text(json.dumps({
        'at_utc': datetime.now(timezone.utc).isoformat(), 'source': str(candidate),
        'snapshot': str(destination), 'files': current, 'write_files': sorted(write_files),
        'scope_audit': f'scope_{rev:02d}.json'}, indent=2) + '\n')
    print(json.dumps({'revision': rev, 'snapshot': str(destination), 'files': len(current),
                      'changed_files': changed, 'intermediate_tool_errors': len(errors),
                      'structured_completion_success': True, 'scope_violations': []}))


if __name__ == '__main__':
    main()
