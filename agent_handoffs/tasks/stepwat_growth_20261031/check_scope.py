"""Inspect captured Claude tool scope and preserve pre-task source identities."""
from pathlib import Path
import argparse
import hashlib
import json

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[2]
ALLOWED = {
    'benchmarks/daily_water_growth.py', 'leaf/__init__.py', 'leaf/coupling.py',
    'leaf/stepwat.py', 'tests/test_stepwat.py', 'docs/stepwat-growth.md',
}
CONTROL = {
    'agent_handoffs/current_prompt.md', 'agent_handoffs/orchestration_state.md',
    'agent_handoffs/codex_review.md', 'agent_handoffs/verification.md',
    'agent_handoffs/claude_report.md',
}


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--installed', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    candidate = TASK / 'candidate'
    changes, violations, tool_reads, tool_writes, read_errors = [], [], [], [], []
    baseline = json.loads((TASK / 'baseline.json').read_text())['files']
    doc_updates = {
        'README.md', 'agent_handoffs/project_brief.md', 'agent_handoffs/decisions.md',
        'docs/water-to-biomass-survey.md',
        'docs/coupled-transpiration.md', 'agent_handoffs/README.md',
    } if args.installed else set()
    permitted = CONTROL | (ALLOWED if args.installed else set()) | doc_updates
    for name, record in baseline.items():
        p = ROOT / name
        if not p.is_file() or digest(p) != record['sha256']:
            changes.append(name)
            if name not in permitted:
                violations.append(f'protected root file changed: {name}')
    for p in (TASK / 'baseline_sources').rglob('*.py'):
        relative = str(p.relative_to(TASK / 'baseline_sources'))
        if not (candidate / relative).is_file():
            violations.append(f'candidate baseline file missing: {relative}')
        elif digest(p) != digest(candidate / relative) and relative not in ALLOWED:
            violations.append(f'read-only candidate file changed: {relative}')
        if digest(p) != baseline[relative]['sha256']:
            violations.append(f'baseline snapshot changed: {relative}')
    for p in candidate.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts:
            continue
        relative = str(p.relative_to(candidate))
        if relative not in ALLOWED and relative not in baseline:
            violations.append(f'unassigned candidate output: {relative}')
    for log in sorted(TASK.glob('claude_*.stdout.jsonl')):
        for line in log.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            for item in event.get('message', {}).get('content', []):
                if item.get('type') == 'tool_result' and item.get('is_error'):
                    read_errors.append({'log': log.name, 'message': str(item.get('content'))[:1000]})
                if item.get('type') != 'tool_use':
                    continue
                tool, path = item['name'], item.get('input', {}).get('file_path')
                if tool not in ('Read', 'Write', 'Edit') or not path:
                    violations.append(f'unassigned tool or missing path: {tool}')
                    continue
                p = Path(path)
                p = p.resolve() if p.is_absolute() else (candidate / p).resolve()
                try:
                    relative = str(p.relative_to(candidate))
                except ValueError:
                    violations.append(f'outside source scope: {p}')
                    continue
                if tool == 'Read':
                    tool_reads.append(relative)
                else:
                    tool_writes.append(relative)
                    if relative not in ALLOWED:
                        violations.append(f'outside write scope: {relative}')
    result = {'ok': not violations, 'installed': args.installed,
              'root_changes': changes, 'violations': violations,
              'claude_read_paths': sorted(set(tool_reads)),
              'claude_write_paths': sorted(set(tool_writes)),
              'captured_tool_errors': read_errors}
    (TASK / args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'ok': result['ok'], 'root_changes': len(changes),
                      'violations': violations, 'read_paths': len(set(tool_reads)),
                      'write_paths': len(set(tool_writes))}))
    if violations:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
