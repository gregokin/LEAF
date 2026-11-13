"""Real source-guarded c76 standalone periodic checkpoint continuity (synthetic plants)."""
from pathlib import Path
import json,os,sys,unittest
import numpy as np
root=Path(os.environ['LEAF_CHECK_ROOT']);sys.path.insert(0,str(root/'tests'))
import test_interevent as ti
from leaf import state,checkpoint
assert ti.NATIVE and ti.VERIFY_NATIVE, 'configured legacy source must be genuine and source-qualified'
be=os.environ.get('LEAF_CHECK_BACKEND','numpy')
if be=='cuda':
 import cupy as cp
 cp.cuda.Device(0).use()
work=Path(os.environ['LEAF_CHECK_WORK']);work.mkdir(exist_ok=False)
env=ti.Env(be);cat=ti.load_catalog();dom=state.Domain(4,4,1.,periodic=(True,True))
records=[dict(plant_id=i+1,species=sp,x_m=x,y_m=y,establishment_ground_m=100.,agb_dry_kg=B,seasonal_agb_dry_kg=0.,age_days=100.,status='live') for i,(sp,x,y,B) in enumerate([('grass',.03,.03,.3),('grass',3.97,3.97,.3),('shrub',2.,2.,5.)])]
pop=state.PlantState.from_records(cat,dom,records)
cont=ti.make_session(env,cat,pop,builder='compiled');split=ti.make_session(env,cat,pop,builder='compiled')
ti.run_days(cont,env,8);ti.run_days(split,env,3)
path=split.save(work/'checkpoint');loaded=ti.IntereventSession.load(path,cat,operator=env.operator,backend=be,max_edges=10000000,max_bytes=1<<30,verify_native_sources=True)
assert loaded.snapshot().domain.periodic==(True,True)
ti.run_days(loaded,env,5,first=3)
tc=unittest.TestCase();ti.assert_same(tc,ti.fp(cont),ti.fp(loaded))
a=loaded.root_map.host_arrays()
for p in range(pop.n):
 rr=a['resources'][a['offsets'][p]:a['offsets'][p+1]];assert len(rr)==len(np.unique(rr))
meta=json.loads((Path(path)/'plants.metadata.json').read_text());assert meta['domain']['periodic']==[True,True]
assert (Path(path)/'plants.csv').read_text().splitlines()[0]==','.join(state.CSV_COLUMNS)
f=Path(path)/'restart.json';man=json.loads(f.read_text());man['config']['domain']=state.Domain(4,4,1.).to_dict();body=dict(man);body.pop('content_hash');man['content_hash']=checkpoint._sha(checkpoint._canonical(body));f.write_text(json.dumps(man))
try:ti.IntereventSession.load(path,cat,operator=env.operator,backend=be,max_edges=10000000,max_bytes=1<<30,verify_native_sources=True)
except checkpoint.CheckpointError as exc:
 assert 'boundary mismatch' in str(exc);refusal=str(exc)
else:raise AssertionError('wrong boundary choice was silently restored')
print(json.dumps({'backend':be,'legacy_source_guard_verified':ti.VERIFY_NATIVE,'model_days':8,'continuous_equals_day3_checkpoint_resume':True,'unique_plant_cell_layer_edges':True,'checkpoint_plain_csv_and_periodic_companion':True,'wrong_boundary_refusal':refusal},indent=2))
