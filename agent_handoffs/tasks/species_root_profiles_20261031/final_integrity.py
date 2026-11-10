"""Check final task provenance, preservation and local documentation links."""
from pathlib import Path
import ast
import datetime
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[3]
TASK = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = json.loads((TASK / 'baseline.json').read_text())
    protected = ['AGENTS.md', 'CLAUDE.md', 'agent_handoffs/AGENTS.md', 'claude.md',
                 'benchmarks/spatial_mapping.py', 'tests/test_spatial_mapping.py', 'agent_handoffs/upstreams.md',
                 'agent_handoffs/tasks/daily_water_30year_20261030/final_manifest.json']
    for rel in protected:
        assert digest(ROOT / rel) == base['files'][rel]['sha256'], rel
    hist = json.loads((ROOT / 'agent_handoffs/tasks/daily_water_30year_20261030/final_manifest.json').read_text())
    preserved = []
    for rel, item in hist['files'].items():
        if rel.startswith(('agent_handoffs/tasks/', 'benchmarks/results/')):
            assert digest(ROOT / rel) == item['sha256'], rel
            preserved.append(rel)
    frozen = json.loads((TASK / 'measurement_source_manifest.json').read_text())
    for rel, h in frozen['files'].items():
        assert digest(ROOT / rel) == h, rel
    after = json.loads((TASK / 'source_after_03.json').read_text())['files']
    for rel in ['benchmarks/daily_water_growth.py', 'tests/test_daily_water_growth.py']:
        assert digest(ROOT / rel) == after[rel]['sha256'], rel
    after = json.loads((TASK / 'source_after_04.json').read_text())['files']
    assert digest(ROOT / 'benchmarks/daily-water-growth.md') == after['benchmarks/daily-water-growth.md']['sha256']
    snapshot = json.loads((TASK / 'final_source_snapshot.json').read_text())['files']
    for rel, h in snapshot.items():
        assert digest(ROOT / rel) == digest(TASK / 'final_sources' / rel) == h, rel
    claude = []
    session = json.loads((TASK / 'session_config.json').read_text())['session_id']
    for i in range(1, 5):
        k = f'{i:02d}'
        launch = json.loads((TASK / f'launch_{k}.json').read_text())
        end = json.loads((TASK / f'exit_{k}.json').read_text())
        assert launch['session_id'] == end['session_id'] == session
        assert launch['tools'] == [] and launch['safe_mode'] and not launch['permission_bypass']
        assert end['returncode'] == 0 and end['child_reaped'] and not end['timed_out']
        assert launch['pid'] > 0 and len(launch['proc_stat_at_launch'].rsplit(')', 1)[1].split()) >= 20
        assert digest(Path(launch['prompt'])) == launch['prompt_sha256']
        events = [json.loads(s) for s in (TASK / f'claude_{k}.stdout.jsonl').read_text().splitlines()]
        result = events[-1]
        assert result['type'] == 'result' and result['subtype'] == 'success' and not result['is_error']
        data = json.loads(result['result'])
        allowed = {'benchmarks/daily-water-growth.md'} if i == 4 else {
            'benchmarks/daily_water_growth.py', 'tests/test_daily_water_growth.py', 'benchmarks/daily-water-growth.md'}
        assert all(e['path'] in allowed for e in data['edits'])
        claude.append({'revision': i, 'structured_success': True, 'child_reaped': True, 'edit_count': len(data['edits'])})
    for name, skips in [('tests_cpu_03', 3), ('tests_gpu_03', 1)]:
        end = json.loads((TASK / f'{name}.exit.json').read_text())
        assert end['returncode'] == 0 and end['child_reaped']
        s = (TASK / f'{name}.stderr.log').read_text()
        assert 'Ran 47 tests' in s and f'OK (skipped={skips})' in s
    oracles = json.loads((TASK / 'oracles_03.stdout.log').read_text())['independent_analytical_checks']
    assert [r['backend'] for r in oracles] == ['cpu', 'gpu']
    assert all(all(x['passed'] for x in r['heterogeneous_supply_cases']) and r['explicit_third_profile']
               and r['uniform_analytical_control'] for r in oracles)
    runs = json.loads((TASK / 'result_verification.json').read_text())
    assert runs['status'] == 'pass' and len(runs['records']) == 8
    receipts = []
    for f in TASK.glob('*.launch.json'):
        launch = json.loads(f.read_text())
        end = json.loads(f.with_name(f.name.replace('.launch.', '.exit.')).read_text())
        assert end['child_reaped'] and not end['timed_out']
        assert launch['start_ticks'] and launch['pid'] > 0
        for rel, h in launch['source_and_case_sha256'].items():
            if str(rel).endswith('.json'):
                assert digest(Path(rel)) == h, rel
        receipts.append({'run': f.name.removesuffix('.launch.json'), 'returncode': end['returncode'], 'reaped': True})
    syntax = ['benchmarks/daily_water_growth.py', 'tests/test_daily_water_growth.py']
    syntax += [str(p.relative_to(ROOT)) for p in TASK.glob('*.py')]
    for rel in syntax:
        ast.parse((ROOT / rel).read_text(), filename=rel)
    md_files = ['README.md', 'docs/species-root-profiles.md', 'docs/daily-water-30year.md',
                'agent_handoffs/README.md', 'agent_handoffs/decisions.md', 'agent_handoffs/project_brief.md',
                'agent_handoffs/claude_report.md', 'agent_handoffs/verification.md', 'agent_handoffs/codex_review.md',
                'agent_handoffs/orchestration_state.md', 'agent_handoffs/current_prompt.md']
    link_count = 0
    for rel in md_files:
        p = ROOT / rel
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', p.read_text()):
            target = target.strip('<>').split('#')[0]
            if not target or re.match(r'[a-zA-Z]+:', target):
                continue
            assert (p.parent / target).exists(), (rel, target)
            link_count += 1
    result = {'status': 'pass', 'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'protected_files': protected, 'historical_artifacts_preserved': len(preserved),
              'frozen_sources_unchanged': True, 'claude_revisions': claude,
              'process_receipts': receipts, 'syntax_files': len(syntax), 'valid_local_links': link_count,
              'tests_cpu': '47, OK (3 expected skips)', 'tests_gpu': '47, OK (1 expected skip)',
              'independent_analytical_cpu_gpu': True, 'full_30_year_runs': 8}
    (TASK / 'integrity_checks.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
