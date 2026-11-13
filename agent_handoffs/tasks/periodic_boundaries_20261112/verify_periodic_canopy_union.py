"""Independent adaptive-quadrature oracle for unique toroidal canopy-cell areas."""
import itertools,json,math,os
import numpy as np
from leaf import state,plant_pet
be=os.environ.get('LEAF_CHECK_BACKEND','numpy')
if be=='cuda':
 import cupy as cp
 cp.cuda.Device(0).use()

def integral(f,a,b,eps=2e-11,depth=24):
 def rec(a,b,fa,fm,fb,s,eps,d):
  c=(a+b)*.5;l=(a+c)*.5;r=(c+b)*.5;fl=f(l);fr=f(r)
  sl=(c-a)*(fa+4*fl+fm)/6;sr=(b-c)*(fm+4*fr+fb)/6;err=sl+sr-s
  if d==0 or abs(err)<=15*eps:return sl+sr+err/15
  return rec(a,c,fa,fl,fm,sl,eps*.5,d-1)+rec(c,b,fm,fr,fb,sr,eps*.5,d-1)
 fa=f(a);fm=f((a+b)*.5);fb=f(b);s=(b-a)*(fa+4*fm+fb)/6
 return rec(a,b,fa,fm,fb,s,eps,depth)

def cell_area(dom,px,py,r,i,j):
 h=dom.dx_m;Lx=dom.width_m;Ly=dom.height_m;px-=dom.origin_x_m;py-=dom.origin_y_m
 x0=i*h;x1=x0+h;y0=j*h;y1=y0+h
 def f(x):
  dx=abs(x-px)
  if dom.periodic[0]:dx=min(abs(x-px-k*Lx) for k in [-1,0,1])
  if dx>=r:return 0.
  half=math.sqrt(max(0.,r*r-dx*dx));intervals=[]
  for k in ([-1,0,1] if dom.periodic[1] else [0]):
   a=max(y0,py+k*Ly-half);b=min(y1,py+k*Ly+half)
   if b>a:intervals.append((a,b))
  intervals.sort();total=0.;last=y0
  for a,b in intervals:
   total+=max(0.,b-max(a,last));last=max(last,b)
  return total
 cuts={x0,x1}
 for k in ([-1,0,1] if dom.periodic[0] else [0]):
  center=px+k*Lx
  for x in [center-r,center,center+r,center-Lx*.5,center+Lx*.5]:
   if x0<x<x1:cuts.add(x)
  for m in ([-1,0,1] if dom.periodic[1] else [0]):
   for yy in [y0,y1]:
    dy=yy-(py+m*Ly)
    if abs(dy)<r:
     span=math.sqrt(r*r-dy*dy)
     for x in [center-span,center+span]:
      if x0<x<x1:cuts.add(x)
 xs=sorted(cuts)
 return sum(integral(f,a,b) for a,b in zip(xs[:-1],xs[1:]))

cases=0;maxerr=0.
for nx,ny in [(1,1),(1,3),(3,2)]:
 for flags in [(True,True),(True,False),(False,True)]:
  dom=state.Domain(nx,ny,.5,origin_x_m=-.25,origin_y_m=2.,periodic=flags)
  for fx,fy,r in [(.13,.31,.1),(.5,.5,.4),(.99,.98,.7),(1.,1.,2.7)]:
   x=dom.origin_x_m+fx*dom.width_m;y=dom.origin_y_m+fy*dom.height_m
   expected=np.array([cell_area(dom,x,y,r,i,j) for j in range(ny) for i in range(nx)])
   for builder in ['numpy','compiled']:
    cm=plant_pet.build_canopy_map(dom,np.array([1],np.int64),np.array([x]),np.array([y]),np.array([r]),backend=be,builder=builder,max_edges=1000,max_bytes=1<<26,device=0 if be=='cuda' else None)
    a=cm.host_arrays();got=np.zeros(dom.ncells);assert len(a['cells'])==len(set(a['cells']))
    got[a['cells']]=a['areas_m2'];np.testing.assert_allclose(got,expected,atol=4e-10,rtol=1e-9)
    assert np.all(got>=0.) and np.all(got<=dom.dx_m**2+1e-14)
    np.testing.assert_allclose(a['area_total_m2'],[expected.sum()],atol=1e-9,rtol=1e-9)
    maxerr=max(maxerr,float(np.max(abs(got-expected))));cases+=1
print(json.dumps({'backend':be,'canopy_cell_union_cases':cases,'max_absolute_cell_area_error_m2':maxerr,'oracle':'adaptive integration of independently merged vertical image intervals, per physical cell'},indent=2))
