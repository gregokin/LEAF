"""Independent geometric and uptake oracles. Small synthetic traits, no trial run."""
from pathlib import Path
import itertools,json,math,os,sys
import numpy as np
from leaf import state,traits,canopy,roots,plant_pet,uptake
ROOT=Path(os.environ['LEAF_CHECK_ROOT']);be=os.environ.get('LEAF_CHECK_BACKEND','numpy');cuda=be=='cuda'
if cuda:
 import cupy as cp
 cp.cuda.Device(0).use()
else: cp=None
catdoc=traits.SpeciesCatalog.from_file(ROOT/'examples/phase1/species.json').document
for sp in catdoc['species'].values():
 sp['allometry']['diameter']={'a':2.,'b':.5};sp['roots']['lateral_radius_multiplier']=1.;sp['roots']['layer_fractions']=[.75,.25]
cat=traits.SpeciesCatalog(catdoc)
CAP=1<<27;ME=100000
host=lambda a:np.array(a.get() if hasattr(a,'get') else a,copy=True)
def mk(dom,x,y,r):
 return state.PlantState.from_records(cat,dom,[dict(plant_id=31,species='grass',x_m=x,y_m=y,establishment_ground_m=0.,agb_dry_kg=r*r,seasonal_agb_dry_kg=0.,age_days=100.,status='live')])
def reached(dom,x,y,r):
 out=[]
 for j in range(dom.ny):
  for i in range(dom.nx):
   xx0=dom.origin_x_m+i*dom.dx_m;xx1=xx0+dom.dx_m;yy0=dom.origin_y_m+j*dom.dy_m;yy1=yy0+dom.dy_m
   yes=False
   for sx in ([-dom.width_m,0.,dom.width_m] if dom.periodic[0] else [0.]):
    for sy in ([-dom.height_m,0.,dom.height_m] if dom.periodic[1] else [0.]):
     px=x+sx;py=y+sy
     dxx=max(xx0-px,0.,px-xx1);dyy=max(yy0-py,0.,py-yy1)
     if dxx*dxx+dyy*dyy<r*r:yes=True
   if yes:out.append(j*dom.nx+i)
 return out
root_cases=0
for nx,ny in [(1,1),(1,4),(4,1),(4,4),(5,3)]:
 for flags in itertools.product([False,True],repeat=2):
  dom=state.Domain(nx,ny,.5,origin_x_m=-3.25,origin_y_m=8.,periodic=flags)
  for fx,fy in [(0.,0.),(.013,.91),(.5,.5),(1.,1.)]:
   x=dom.origin_x_m+fx*dom.width_m;y=dom.origin_y_m+fy*dom.height_m
   for r in [.03,.4,.91,4.7]:
    pop=mk(dom,x,y,r);rr=float(canopy.derive_geometry(pop,cat)['root_radius_m'][0]);want=reached(dom,x,y,rr)
    for builder in ['numpy','compiled']:
     rm=roots.build_root_map(pop,cat,max_edges=ME,max_bytes=CAP,backend=be,builder=builder);a=rm.host_arrays()
     ids=a['resources'];ww=a['weights'];assert len(ids)==2*len(want)
     assert len(np.unique(ids))==len(ids)
     for ell,f in [(0,.75),(1,.25)]:
      vals={int(c-ell*dom.ncells):float(w) for c,w in zip(ids,ww) if ell*dom.ncells<=c<(ell+1)*dom.ncells}
      assert sorted(vals)==want,(nx,ny,flags,fx,fy,r,builder,sorted(vals),want)
      np.testing.assert_allclose(list(vals.values()),[f/len(want)]*len(want),rtol=3e-14,atol=0.)
     np.testing.assert_allclose(ww.sum(),1.,rtol=3e-14,atol=0.)
     root_cases+=1
pet_cases=0
for flags in itertools.product([False,True],repeat=2):
 dom=state.Domain(4,4,1.,periodic=flags)
 for x,y in [(0.,0.),(0.,1.5),(4.,4.),(.15,.2)]:
  r=.4
  ids=np.array([8],np.int64);xx=np.array([x]);yy=np.array([y]);rad=np.array([r]);want=reached(dom,x,y,r)
  for builder in ['numpy','compiled']:
   cm=plant_pet.build_canopy_map(dom,ids,xx,yy,rad,max_edges=1000,max_bytes=CAP,backend=be,builder=builder,device=0 if cuda else None)
   a=cm.host_arrays();assert sorted(a['cells'].tolist())==want;assert len(np.unique(a['cells']))==len(a['cells'])
   if x in [0.,4.] and y in [0.,4.]:
    total=math.pi*r*r*(1. if flags[0] else .5)*(1. if flags[1] else .5)
    np.testing.assert_allclose(a['area_total_m2'],[total],rtol=1e-12,atol=1e-15)
    np.testing.assert_allclose(a['areas_m2'],[total/len(want)]*len(want),rtol=1e-12,atol=1e-15)
   pet=np.arange(16,dtype=np.float64)
   field=cp.asarray(pet) if cuda else pet
   v=host(plant_pet.canopy_pet(cm,field,max_bytes=CAP)['pet_mm_per_day'])[0]
   expect=float(np.sum(pet[a['cells']]*a['areas_m2'])/np.sum(a['areas_m2']))
   np.testing.assert_allclose(v,expect,rtol=1e-12,atol=1e-15)
   pet_cases+=1
# Independent midpoint cover union for multiple circles and oversized images.
cover_cases=0
for flags in itertools.product([False,True],repeat=2):
 dom=state.Domain(7,5,.3,origin_x_m=2.5,origin_y_m=-1.25,periodic=flags)
 xx=np.array([2.5,4.45,3.8]);yy=np.array([-1.25,.23,-.8]);rad=np.array([.35,.42,.1]);s=32
 xc=dom.origin_x_m+(np.arange(dom.nx*s)+.5)*dom.dx_m/s;yc=dom.origin_y_m+(np.arange(dom.ny*s)+.5)*dom.dy_m/s
 mask=np.zeros((len(yc),len(xc)),bool)
 for x,y,r in zip(xx,yy,rad):
  vx=xc-x;vy=yc-y
  if flags[0]:vx=(vx+dom.width_m/2)%dom.width_m-dom.width_m/2
  if flags[1]:vy=(vy+dom.height_m/2)%dom.height_m-dom.height_m/2
  mask|=(vy[:,None]**2+vx[None,:]**2)<r*r
 got=canopy.measure_cover(dom,xx,yy,rad,subsamples=s)
 assert got['total_union_fraction']==float(mask.mean()),(flags,got['total_union_fraction'],mask.mean())
 cover_cases+=1
# Full demand only comes from opposite-seam available cells; no compensation or double debit.
dom=state.Domain(4,4,1.,periodic=(True,True));pop=mk(dom,3.9,1.5,.5)
rm=roots.build_root_map(pop,cat,max_edges=ME,max_bytes=CAP,backend=be,builder='compiled')
supply=np.zeros((2,dom.ncells));supply[:,::4]=.1
D=np.array([.2]);Ad=cp.asarray(supply) if cuda else supply;Dd=cp.asarray(D) if cuda else D
pr=uptake.propose(rm,Dd,Ad);actual=pr.proposed.copy();fb=uptake.accept(pr,actual)
a=rm.host_arrays();on_opposite=(a['resources']%dom.ncells)%4==0;expected=.2*float(a['weights'][on_opposite].sum())
np.testing.assert_allclose(host(fb.delivered),[expected],rtol=1e-12,atol=1e-15)
assert float(host(actual).sum())==float(host(fb.delivered).sum())
assert expected>0 and expected<.2
print(json.dumps({'backend':be,'root_cases':root_cases,'canopy_pet_cases':pet_cases,'sampled_cover_cases':cover_cases,'opposite_seam_delivered_m3':float(host(fb.delivered)[0]),'normalized_unique_layered_roots':True,'independent_oracles':'nine-image closest-point roots, analytic circle quadrants, minimum-image midpoint cover'},indent=2))
