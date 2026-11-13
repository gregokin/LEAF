"""Codex: materialize explicit post-writer check commands for one frozen revision."""
from pathlib import Path
import json,sys
P=Path(__file__).resolve().parent;rev=int(sys.argv[1]);tag=f'review{rev:02d}';snap=P/f'public_evidence/review_{rev:02d}'
assert snap.is_dir()
PREV=P.parent/'masy_daily_plant_interface_20261112';legacy=PREV/'public_evidence/syrup_legacy/src';n15=PREV/'public_evidence/syrup_phase15/src'
py='/home/okin/MAPLE/.venv/bin/python';cuda_py='/home/okin/MAPLE/.venv-cupy/bin/python';gpu='GPU-b02858c1-29c3-9ea9-33b1-831c33f8cf75'
base={'PYTHONDONTWRITEBYTECODE':'1','NUMBA_CACHE_DIR':str(P/'public_evidence/numba_cache'),'CUPY_CACHE_DIR':str(P/'public_evidence/cupy_cache')}
def put(suffix,argv,env,cwd=snap):
 f=P/f'{tag}_{suffix}.command.json';assert not f.exists();f.write_text(json.dumps({'argv':argv,'cwd':str(cwd),'env':{**env,**base}},indent=2)+'\n')
for be in ['cpu','cuda']:
 env={'PYTHONPATH':f'{snap}:{legacy}','CUDA_VISIBLE_DEVICES':gpu if be=='cuda' else '-1'}
 put(be,[cuda_py if be=='cuda' else py,'-B','-m','unittest','discover','-s','tests','-v'],env)
for be in ['numpy','numba','cuda']:
 env={'PYTHONPATH':str(snap),'CUDA_VISIBLE_DEVICES':gpu if be=='cuda' else '-1','LEAF_CHECK_ROOT':str(snap),'LEAF_CHECK_BACKEND':be}
 put('geometry_'+be,[cuda_py if be=='cuda' else py,'-B',str(P/'verify_periodic_geometry.py')],env)
for suffix,script,args in [('golden','nonperiodic_golden.py',[str(P/'baseline_golden.stdout.log')]),('aliases','verify_periodic_aliases_v2.py',[]),('canopy_union','verify_periodic_canopy_union_v2.py',[]),('mixed_endpoints','verify_periodic_mixed_endpoints.py',[]),('recruitment_memory','verify_periodic_recruitment_memory.py',[]),('cover_memory','verify_periodic_cover_memory.py',[])]:
 put(suffix,[py,'-B',str(P/script)]+args,{'PYTHONPATH':str(snap),'CUDA_VISIBLE_DEVICES':'-1','LEAF_CHECK_ROOT':str(snap)})
for be in ['cpu','cuda']:
 prior=json.loads((PREV/f'qualify04_{be}.command.json').read_text());env=prior['env']
 env['PYTHONPATH']=env['PYTHONPATH'].replace(str(PREV/'public_evidence/review_04'),str(snap))+f':{snap}/tests'
 env.update(LEAF_CHECK_ROOT=str(snap),LEAF_CHECK_WORK=str(P/f'public_evidence/{tag}_native_pipeline_{be}'))
 put('native_pipeline_'+be,[cuda_py if be=='cuda' else py,'-B',str(P/'qualify_periodic_native.py')],env)
 put('native_tests_'+be,[cuda_py if be=='cuda' else py,'-B','-m','unittest','test_periodic_boundaries.NativePeriodicN15Tests','test_lifecycle_bridge.NativeDailyTests','-v'],env)
out=P/f'public_evidence/{tag}_example';out.mkdir()
put('example',[py,'-B','-m','leaf','generate','--species','examples/phase0/species.json','--config','examples/periodic/generation.json','--surface-constant','0','--out',str(out/'plants.csv'),'--report',str(out/'report.json')],{'PYTHONPATH':str(snap),'CUDA_VISIBLE_DEVICES':'-1'})
print(f'{tag}: commands written for frozen {snap}')
