"""Mixed-boundary closed upper endpoints with decimal origins (no trial case)."""
import itertools,json,os
from pathlib import Path
import numpy as np
from leaf import state,traits,roots,plant_pet
cat=traits.SpeciesCatalog.from_file(Path(os.environ['LEAF_CHECK_ROOT'])/'examples/phase0/species.json')
errors=[];cases=0
for flags,dx,origin in itertools.product([(True,False),(False,True)],[.1,.2,.3,.7],[.1,.3,.4,100000.1]):
 d=state.Domain(3,7,dx,origin_x_m=origin,origin_y_m=-origin,periodic=flags)
 upper=(d.x_max_m,d.y_max_m)
 equivalent=(d.x_min_m if flags[0] else upper[0],d.y_min_m if flags[1] else upper[1])
 def pop(x,y):return state.PlantState.from_records(cat,d,[dict(plant_id=3,species='grass',x_m=x,y_m=y,establishment_ground_m=0.,agb_dry_kg=.1,seasonal_agb_dry_kg=0.,age_days=100.,status='live')])
 try:
  ps=[pop(*upper),pop(*equivalent)]
  for module,key,vals in [(roots,'resources','weights'),(plant_pet,'cells','areas_m2')]:
   build=module.build_root_map if module is roots else module.canopy_map_from_state
   arrays=[build(p,cat,max_edges=10000,max_bytes=1<<26,builder='compiled').host_arrays() for p in ps]
   kv=[dict(zip(a[key].tolist(),a[vals].tolist())) for a in arrays]
   assert kv[0]==kv[1], (module.__name__,kv)
 except Exception as exc:errors.append({'flags':flags,'dx':dx,'origin':origin,'error':repr(exc)})
 cases+=1
print(json.dumps({'cases':cases,'failure_count':len(errors),'failures':errors[:8]},indent=2))
assert not errors,'Legal mixed-boundary upper positions must build and respect only the periodic alias'
