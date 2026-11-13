import pathlib,sys,json,dataclasses
TASK=pathlib.Path(__file__).resolve().parent;SNAP=TASK/'public_evidence/review_02'
sys.path.insert(0,str(SNAP/'tests'))
import native_leaf_fixture as nf
from leaf import plant_session as ps,plant_checkpoint as pc
cat=nf.make_catalog(fixture='phase3');cfg=nf.make_config(cat,rng_seed=10,patch_size_m=20.)
s=ps.PlantSession.start(nf.make_state(cat,species=cat.species_ids[0]),cfg,anchor=nf.anchor())
work=TASK/'public_evidence/rev02_rng';work.mkdir(exist_ok=False);ext=nf.identity();pc.save_biology(work/'ck',s,external=ext,limits=pc.CheckpointLimits(1<<26))
out={}
for label,other in [('rng_seed',dataclasses.replace(cfg,rng_seed=20)),('patch_size_m',dataclasses.replace(cfg,patch_size_m=4.))]:
 try:r=pc.load_biology(work/'ck',other,bindings=pc.ExternalBindings(ext),limits=pc.CheckpointLimits(1<<26));out[label]={'changed_configuration_accepted':True,'same_token':r.session.token==s.token}
 except pc.PlantCheckpointError as exc:out[label]={'changed_configuration_accepted':False,'error':str(exc)}
print(json.dumps(out,indent=2))
assert not any(v['changed_configuration_accepted'] for v in out.values()),'Biological restart failed to bind RNG seed or ecological patch config'
