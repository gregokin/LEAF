"""Install an independently reviewed snapshot and append its source qualification."""
from datetime import datetime, timezone
from pathlib import Path
import ast
import hashlib
import json
import shutil
import subprocess
import sys

ROOT=Path('/home/okin/leaf')
TASK=Path(__file__).resolve().parent
INSTALL=['leaf/canopy.py','leaf/__init__.py','tests/test_phase0.py',
         'docs/phase0-vegetation.md','docs/maple-leaf-integration.md','docs/shared-canopy.md']
CONTROLS={'README.md','.gitignore','agent_handoffs/project_brief.md',
          'agent_handoffs/decisions.md','agent_handoffs/orchestration_state.md',
          'agent_handoffs/current_prompt.md','agent_handoffs/codex_review.md',
          'agent_handoffs/verification.md','agent_handoffs/claude_report.md'}


def identity(path):
    b=path.read_bytes()
    return {'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}


def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()


def save(path,data):
    path.write_text(json.dumps(data,indent=2)+'\n')


def node_map(text):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body
            if isinstance(n,(ast.FunctionDef,ast.ClassDef))}


def main():
    revision=int(sys.argv[1]);frozen=TASK/f'revision_{revision:02d}'
    baseline=json.loads((TASK/'baseline.json').read_text())
    manifest=json.loads((TASK/f'revision_{revision:02d}_manifest.json').read_text())
    changed=[rel for rel,old in baseline['files'].items()
             if not (ROOT/rel).is_file() or identity(ROOT/rel)!=old]
    assert not set(changed)-CONTROLS,changed
    for rel in INSTALL:
        target=ROOT/rel
        if rel in baseline['files']:assert identity(target)==baseline['files'][rel],rel
        else:assert not target.exists(),rel
        assert identity(frozen/rel)==manifest[rel],rel
    old_nodes=node_map((ROOT/'leaf/canopy.py').read_text())
    new_nodes=node_map((frozen/'leaf/canopy.py').read_text())
    for name in ['_check_catalog','derive_geometry','cover_admission','measure_cover']:
        assert old_nodes[name]==new_nodes[name],name
    old_init=ast.parse((ROOT/'leaf/__init__.py').read_text()).body[1:]
    new_init=ast.parse((frozen/'leaf/__init__.py').read_text()).body[1:]
    assert [ast.dump(n) for n in old_init]==[ast.dump(n) for n in new_init]
    old_tests=node_map((ROOT/'tests/test_phase0.py').read_text())
    new_tests=node_map((frozen/'tests/test_phase0.py').read_text())
    for name,node in old_tests.items():assert node==new_tests[name],name
    previous=ROOT/'dependencies/phase0-vegetation.lock.json'
    old_lock=json.loads(previous.read_text());syrup=old_lock['syrup']
    for rel,old in syrup['runtime_files'].items():assert identity(Path(syrup['root'])/rel)==old,rel
    native=Path('/home/okin/MAPLE/src/maple/core/types/vegetation.py')
    assert identity(native)==old_lock['maple']['vegetation_interface']
    for rel in INSTALL:
        target=ROOT/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(frozen/rel,target)
    now=datetime.now(timezone.utc).isoformat()
    upstream={name:{'head':git(Path('/home/okin')/name,'rev-parse','HEAD'),
                    'branch':git(Path('/home/okin')/name,'branch','--show-current')}
              for name in ['MAPLE','SYRUP']}
    record={
        'schema':'leaf-shared-canopy-source-lock-v1','created_at_utc':now,
        'task':'shared_canopy_view_20261110','revision':revision,
        'scope':'Consumer-neutral canopy_view and legacy MAPLE projection; numerical algorithms unchanged.',
        'interfaces':{'canopy_view':'leaf.canopy_view v1','maple_vegetation_view':'leaf.maple_vegetation_view v2'},
        'previous_qualification':{'path':'phase0-vegetation.lock.json',**identity(previous),
              'note':'Historical upstream/environment, scientific limits and physical driver qualification retained. '
                     'LEAF hashes below supersede its LEAF file identities.'},
        'leaf':{'root':str(ROOT),
                'runtime_files':{str(p.relative_to(ROOT)):identity(p) for p in sorted((ROOT/'leaf').glob('*.py'))},
                'tests':{str(p.relative_to(ROOT)):identity(p) for p in sorted((ROOT/'tests').glob('test_*.py'))}},
        'upstream_recheck':{'current_git_heads':upstream,
                           'syrup_runtime_hashes_match_previous':len(syrup['runtime_files']),
                           'maple_native_container_hash_matches_previous':identity(native),
                           'note':'No upstream writes; native MAPLE ABI rechecked. This does not requalify unrelated '
                                  'upstream physics or all dirty source beyond the named qualification.'},
        'numerics':'derive_geometry, cover bounds/measurement, root/growth/water kernels, traits and fixtures unchanged.',
        'qualification':'Full CPU suite plus five-case exact legacy equivalence and shared-array/ID/purity/import '
                        'checks; native MAPLE container validation. No new GPU performance or integrated scenario claim.',
        'remaining_scope':['upstream adoption','hydrological canopy physics','coupled event scheduling',
                           'ecological injury laws','calibrated allometry']}
    save(ROOT/'dependencies/canopy-view.lock.json',record)
    save(TASK/'installation.json',{'at_utc':now,'source':f'revision_{revision:02d}',
         'files':{rel:identity(ROOT/rel) for rel in INSTALL},'exact_frozen_match':True,
         'canopy_numerical_functions_AST_unchanged':True,'existing_test_functions_AST_unchanged':True})
    save(TASK/'scope_final.json',{'at_utc':now,'baseline_head':baseline['head'],
         'baseline_tracked_files':len(baseline['files']),
         'allowed_changes':sorted(CONTROLS|set(INSTALL)), 'unexpected_changes':[],
         'root_AGENTS_preserved':True,'prior_archives_locks_traits_numerical_sources_preserved':True,
         'source_lock':'dependencies/canopy-view.lock.json'})
    print(json.dumps({'pass':True,'installed_files':len(INSTALL),'baseline_files':len(baseline['files']),
                      'preserved_numeric_functions':4,'old_tests_unchanged':True,
                      'syrup_runtime_files_rechecked':len(syrup['runtime_files'])}))


if __name__=='__main__':
    main()
