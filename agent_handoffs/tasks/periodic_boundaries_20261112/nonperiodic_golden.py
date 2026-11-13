"""Independent small default-behavior identity check, never the user trial case."""
import hashlib,json,os,pathlib,sys
import numpy as np
from leaf import state,traits,initialization,roots,plant_pet
root=pathlib.Path(os.environ.get('LEAF_CHECK_ROOT',os.getcwd()))
cat=traits.SpeciesCatalog.from_file(root/'examples/phase0/species.json')
def sha(x):return hashlib.sha256(x).hexdigest()
def ah(arrays):return {k:{'dtype':str(a.dtype),'shape':list(a.shape),'sha256':sha(np.ascontiguousarray(a).tobytes())} for k,a in sorted(arrays.items())}
results={}
for mode in ['allowed','disallowed']:
 cfg={'schema':'leaf.generation_config','schema_version':1,'domain':{'nx':16,'ny':16,'dx_m':1.},'seed':11,'overlap':mode,'cover_tolerance':.01,'cover_subsamples':8,'max_iterations':12,'limits':{'max_plants':5000,'max_attempts':100000},'species':{'grass':{'target_cover':.1,'diameter_mean_m':.4,'diameter_std_m':.1,'initial_age_days':730},'shrub':{'target_cover':.05,'diameter_mean_m':1.,'diameter_std_m':.2,'initial_age_days':{'distribution':'uniform','min_days':365,'max_days':3650}}}}
 pop,report=initialization.generate_population(cat,cfg,surface=state.constant_surface(state.Domain(16,16,1.),4.))
 rm=roots.build_root_map(pop,cat,max_edges=100000,max_bytes=1<<27,backend='numpy',builder='compiled')
 cm=plant_pet.canopy_map_from_state(pop,cat,max_edges=100000,max_bytes=1<<27,backend='numpy',builder='compiled')
 results[mode]={'n':pop.n,'plants':ah({k:getattr(pop,k) for k in pop.ARRAY_NAMES}),'domain':pop.domain.to_dict(),'roots':ah(rm.host_arrays()),'canopy':ah(cm.host_arrays()),'generation_config':report['generation_config'],'measured_cover':[s['measured_union_cover'] for s in report['species']]}
print(json.dumps(results,indent=2))
if len(sys.argv)>1:
 expected=json.loads(pathlib.Path(sys.argv[1]).read_text());assert results==expected,'Default nonperiodic numeric output or metadata changed'
