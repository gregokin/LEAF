"""Bounded synthetic daughter-placement allocation check, not a vegetation scenario run."""
from pathlib import Path
import gc,hashlib,json,os,tracemalloc
import numpy as np
from leaf import state,traits,recruitment as rec
p=Path(os.environ['LEAF_CHECK_ROOT']);doc=json.loads((p/'examples/phase3/species.json').read_text())
for sp in doc['species'].values():
 tr=sp['reproduction'].get('traits',{})
 if tr.get('seed') is not None:tr['seed']['establishment_probability']=0.
 if tr.get('clonal') is not None:tr['clonal']['probability']=0.
c=doc['species']['grass']['reproduction']['traits']['clonal'];c.update(probability=1.,max_daughters_per_parent=50,max_attempts_per_daughter=1,dispersal_radius_m=.6,max_donation_fraction=.5)
cat=traits.SpeciesCatalog(doc);table=rec.compile_species(cat);n=1000;gi=cat.species_ids.index('grass')
la=dict(live_ids=np.arange(1,n+1,dtype=np.int64),live_species=np.full(n,gi,np.int32),live_x=np.full(n,4.),live_y=np.full(n,4.),live_B=np.full(n,100.),live_E=np.full(n,1.),live_age_days=np.full(n,1000.))
out=[];digests=[]
for flags in [(False,False),(True,True)]:
 dom=state.Domain(8,8,1.,periodic=flags);surface=np.full(dom.shape,10.)
 gc.collect();tracemalloc.start();tracemalloc.reset_peak()
 result=rec.evaluate(table,rec.PatchConfig(4.),dom,surface,max_existing_id=n,n_total_records=n,seed=73,counter=0,max_bytes=1<<26,max_new_plants=100000,max_total_records=1000000,max_attempts=1000000,**la)
 _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
 est=result['diagnostics']['memory_estimate'];births=result['births'];estimate=int(est['total_bytes'])
 assert result['diagnostics']['ledger']['closed'];assert result['diagnostics']['n_daughters_rejected']==0
 digest=hashlib.sha256()
 for name,a in sorted(births.items()):digest.update(name.encode());digest.update(np.ascontiguousarray(a).tobytes())
 digests.append(digest.hexdigest())
 out.append({'periodic':flags,'live_parents':n,'requested_daughters':int(result['diagnostics']['n_daughters_requested']),'placed_daughters':int(result['diagnostics']['n_daughters_placed']),'traced_peak_bytes':peak,'admitted_array_estimate_bytes':estimate,'max_bytes':1<<26,'ledger_closed':True,'birth_array_sha256':digests[-1]})
 assert peak<estimate, (flags,peak,estimate)
 del result,births
assert digests[0]==digests[1], 'interior daughter positions/biomass/IDs must match unchanged draws'
print(json.dumps({'runs':out,'scope':'synthetic fixed interior parents; existing reproduction law and tables, no model/case advancement; tracemalloc includes Python plus NumPy arrays during evaluation'},indent=2))
