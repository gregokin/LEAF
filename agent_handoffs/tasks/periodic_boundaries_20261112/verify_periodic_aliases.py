"""Decimal-grid upper seam aliases and canonical public canopy centers."""
import itertools,json,math,os
from pathlib import Path
import numpy as np
from leaf import state,traits,canopy
root=Path(os.environ['LEAF_CHECK_ROOT']);cat=traits.SpeciesCatalog.from_file(root/'examples/phase0/species.json')
bad=[];cases=0
for nx,dx,ox in itertools.product([3,7],[.1,.2,.3,.7],[.1,.3,.4,.6,1.1,100000.1]):
 d=state.Domain(nx,2,dx,origin_x_m=ox,origin_y_m=-ox,periodic=(True,True))
 xx=np.array([d.x_max_m]);yy=np.array([d.y_max_m]);row,col=d.cell_indices(xx,yy)
 if int(col[0])!=0 or int(row[0])!=0:bad.append({'kind':'sample','nx':nx,'dx':dx,'origin':ox,'row':int(row[0]),'col':int(col[0])})
 wx,wy=d.wrap_positions(xx,yy)
 if wx[0]!=d.x_min_m or wy[0]!=d.y_min_m:bad.append({'kind':'wrap','nx':nx,'dx':dx,'origin':ox,'x':float(wx[0]),'y':float(wy[0])})
 st=state.PlantState.from_records(cat,d,[dict(plant_id=9,species='grass',x_m=float(xx[0]),y_m=float(yy[0]),establishment_ground_m=0.,agb_dry_kg=.1,seasonal_agb_dry_kg=0.,age_days=100.,status='live')])
 out=canopy.canopy_view(st,cat,np.zeros(d.shape))
 if out['fields']['center_x_m'][0]!=0. or out['fields']['center_y_m'][0]!=0.:bad.append({'kind':'public_center','nx':nx,'dx':dx,'origin':ox})
 assert st.x_m[0]==xx[0] and st.y_m[0]==yy[0], 'export must not mutate authoritative records'
 cases+=1
print(json.dumps({'cases':cases,'failures':bad[:12],'failure_count':len(bad),'scope':'exact upper endpoint only; ordinary just-inside coordinates remain distinct'},indent=2))
assert not bad, 'periodic upper endpoint is the lower endpoint, including decimal grids and origins'
