"""Pin the installed source and check preservation; never write upstream trees."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json
import subprocess
import sys

ROOT = Path('/home/okin/leaf')
TASK = Path(__file__).resolve().parent


def identity(path):
    data = path.read_bytes()
    return {'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}


def git(root, *arguments):
    run = subprocess.run(['git', '-C', str(root), *arguments], capture_output=True, text=True)
    return {'returncode': run.returncode, 'stdout': run.stdout, 'stderr': run.stderr}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    baseline = json.loads((TASK/'baseline.json').read_text())['files']
    allowed = {'README.md', 'leaf/__init__.py', 'agent_handoffs/project_brief.md',
               'agent_handoffs/decisions.md', 'agent_handoffs/current_prompt.md',
               'agent_handoffs/orchestration_state.md', 'agent_handoffs/codex_review.md',
               'agent_handoffs/verification.md', 'agent_handoffs/claude_report.md'}
    changed = [rel for rel, old in baseline.items()
               if not (ROOT/rel).is_file() or identity(ROOT/rel)['sha256'] != old['sha256']]
    assert not set(changed) - allowed, changed
    installation = json.loads((TASK/'installation.json').read_text())
    for rel, record in installation['installed_files'].items():
        assert identity(ROOT/rel) == record, rel
    historical = ROOT/'dependencies/stepwat-growth.lock.json'
    old = json.loads(historical.read_text())
    for rel, record in old['syrup']['runtime_files'].items():
        assert identity(Path(old['syrup']['root'])/rel) == record, rel
    interface = json.loads((TASK/'upstream_interface.json').read_text())
    for absolute, record in interface['inspected_read_only'].items():
        assert identity(Path(absolute)) == record, absolute
    upstream = {}
    for name in ['MAPLE', 'SYRUP']:
        root = Path('/home/okin')/name
        upstream[name] = {'root': str(root), 'head': git(root, 'rev-parse', 'HEAD'),
                          'branch': git(root, 'branch', '--show-current'),
                          'status': git(root, 'status', '--short', '--untracked-files=normal')}
    save(TASK/'final_upstream_git.json', upstream)
    maple_files = {str(p.relative_to('/home/okin/MAPLE')): identity(p)
                   for p in sorted(Path('/home/okin/MAPLE/src/maple').rglob('*.py'))}
    save(TASK/'maple_runtime_identity_current.json', {
        'scope': 'Current read-only source identities, including unrelated concurrent upstream work; '
                 'not a claim to qualify all MAPLE physics.',
        'head': upstream['MAPLE']['head']['stdout'].strip(), 'files': maple_files})
    runtime = sorted(str(p.relative_to(ROOT)) for p in (ROOT/'leaf').glob('*.py'))
    runtime += ['benchmarks/spatial_mapping.py', 'benchmarks/daily_water_growth.py']
    fixtures = ['pyproject.toml', 'examples/phase0/species.json', 'examples/phase0/plants.csv',
                'examples/phase0/generation.json']
    record = {
        'schema': 'leaf-phase0-source-lock-v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'task': 'phase0_vegetation_20261110', 'revision': 3,
        'scope': 'Installed Phase 0 state/placement/geometry plus unchanged legacy numerical drivers. '
                 'Fixtures are uncalibrated; no complete coupled restart or native scenario is qualified.',
        'historical_growth_lock': {'path': 'stepwat-growth.lock.json', **identity(historical),
             'note': 'Preserved historical qualification. LEAF hashes below supersede it for the current '
                     'runtime; legacy numerical files are identical. Upstream current identities are explicit.'},
        'leaf': {'root': str(ROOT), 'runtime_files': {rel: identity(ROOT/rel) for rel in runtime},
                 'tests': {str(p.relative_to(ROOT)): identity(p) for p in sorted((ROOT/'tests').glob('test_*.py'))},
                 'fixtures_and_packaging': {rel: identity(ROOT/rel) for rel in fixtures}},
        'syrup': {**old['syrup'], 'head': upstream['SYRUP']['head']['stdout'].strip(),
                  'branch': upstream['SYRUP']['branch']['stdout'].strip(),
                  'note': 'All 71 runtime source hashes rechecked against the historical growth lock.'},
        'maple': {'root': '/home/okin/MAPLE', 'head': upstream['MAPLE']['head']['stdout'].strip(),
                  'branch': upstream['MAPLE']['branch']['stdout'].strip(),
                  'runtime_manifest': '../agent_handoffs/tasks/phase0_vegetation_20261110/maple_runtime_identity_current.json',
                  'runtime_manifest_sha256': identity(TASK/'maple_runtime_identity_current.json')['sha256'],
                  'file_count': len(maple_files),
                  'vegetation_interface': interface['inspected_read_only']['/home/okin/MAPLE/src/maple/core/types/vegetation.py'],
                  'note': 'Actual dirty checkout pinned; only native canopy container ABI independently exercised. '
                          'Other agents may continue upstream work.'},
        'upstream_git': {'path': '../agent_handoffs/tasks/phase0_vegetation_20261110/final_upstream_git.json',
                         **identity(TASK/'final_upstream_git.json')},
        'environment': {'python_executable': sys.executable, 'python_version': sys.version.split()[0],
                        **{name: importlib.metadata.version(name) for name in ['numpy', 'numba', 'cupy-cuda12x']},
                        'note': 'Existing environment; no installation or new environment.'},
        'deferred': ['calibrated species allometry and ecological traits', 'MAPLE adoption',
                     'recruitment and mortality laws', 'event injury laws',
                     'evolving soil/root/water remapping', 'full MASY event calendar',
                     'complete coupled restart', 'wheel build']}
    save(ROOT/'dependencies/phase0-vegetation.lock.json', record)
    save(TASK/'scope_final.json', {
        'at_utc': datetime.now(timezone.utc).isoformat(), 'baseline_file_count': len(baseline),
        'expected_baseline_changes': sorted(changed), 'unexpected_changes': [],
        'installed_files_match_frozen_revision_03': True,
        'root_AGENTS_preserved': True, 'legacy_numerical_sources_preserved': True,
        'historical_locks_and_accepted_archives_preserved': True,
        'syrup_runtime_files_rechecked': len(old['syrup']['runtime_files']),
        'maple_interface_preserved': True})
    print(json.dumps({'pass': True, 'baseline_files': len(baseline),
                      'installed_files': len(installation['installed_files']),
                      'syrup_files': len(old['syrup']['runtime_files']), 'maple_files_pinned': len(maple_files)}))


if __name__ == '__main__':
    main()
