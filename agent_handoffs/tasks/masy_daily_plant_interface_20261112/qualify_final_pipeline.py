"""Independent small native qualification; synthetic traits, real N15 water. No trial case."""
from pathlib import Path
import datetime as dt
import gc,json,os,resource,sys,time,tracemalloc
from unittest import mock
import numpy as np
TASK=Path(__file__).resolve().parent
ROOT=Path(os.environ['LEAF_CHECK_ROOT'])
sys.path.insert(0,str(ROOT/'tests'))
import native_leaf_fixture as nf
from leaf import plant_session as ps,lifecycle_bridge as lb,plant_checkpoint as pc,biocalendar as bio
from maple_syrup import daily_water as dw,water_lifecycle as wl
backend=nf.native_backend(); cuda=backend=='cuda'
if cuda:
 import cupy as cp
 cp.cuda.Device(0).use()
 pool=cp.get_default_memory_pool()
 class PeakHook(cp.cuda.MemoryHook):
  name='IndependentLeafPeak'
  def __init__(self): self.peak=pool.used_bytes();self.allocs=0;self.frees=0
  def malloc_postprocess(self,**kw): self.peak=max(self.peak,pool.used_bytes());self.allocs+=1
  def free_postprocess(self,**kw): self.peak=max(self.peak,pool.used_bytes());self.frees+=1
 sync=lambda:cp.cuda.get_current_stream().synchronize()
else: sync=lambda:None
host=nf.to_host
work=Path(os.environ['LEAF_CHECK_WORK']);work.mkdir(exist_ok=False)
started=time.perf_counter();lop,boundary,profile=nf.native_daily_harness(work/'native')
run='da'*32;a=nf.anchor();w=dw.start_daily_lineage(lop,boundary,expected_parent=wl.expect_parent(boundary),run_digest=run)
cat=nf.make_catalog(n_layers=lop.n_layers);cfg=nf.make_config(cat,backend=backend,device=0 if cuda else None)
plants=nf.native_plants_for(lop,cat)
s=ps.PlantSession.start(plants,cfg,anchor=a,day_binding=ps.DayBinding(run,a.date().isoformat()),initial_resource=lb.resource_identity(lop,w))
source=lb.bind_source(profile.syrup_package_digest)
pet=lop.xp.linspace(0.5e-8,1.5e-8,lop.n_cells,dtype=np.float64);petbio=lb.pet_m_s_to_mm_per_day(pet)
startup_s=time.perf_counter()-started;rows=[];failed_rollback=False;pending_resume=False;closed_resume=False
tracemalloc.start();tracemalloc.reset_peak()
hook=PeakHook() if cuda else None
if hook:hook.__enter__()
try:
 for i in range(8):
  sync();tick=time.perf_counter();before_h=w.hydro_count;before_t=w.t_s
  dr=dw.daily_water_action(lop,w,expected_parent=wl.expect_parent(w),plan=dw.next_plan(w,2.,10.,5.),pet_terrain_m_s=pet)
  assert dr.status=='complete' and dr.fallback is None
  day=dr.state
  assert day.hydro_count==before_h+1 and day.t_s==before_t+86400.0
  rd=lb.read_availability(source,lop,day,max_bytes=cfg.max_bytes,reserve_bytes=s.resident_bytes())
  plan=s.prepare_daily(ps.DayContext(a+dt.timedelta(days=i),20.,True,i!=7),terrain_pet=petbio)
  prop=s.propose_withdrawal(plan,rd)
  depth=lb.request_depth_m(rd,prop.proposed_m3,max_bytes=cfg.max_bytes,reserve_bytes=s.resident_bytes())
  wn,cand=wl.request_withdrawal(lop,day,**lb.request_args(rd,lop,day),depth_m=depth,identity=prop.identity,retained_feedback_bytes=prop.retained_feedback_bytes)
  ac=lb.accepted_withdrawal(rd,prop.identity,wn,cand,lop=lop,max_bytes=cfg.max_bytes,reserve_bytes=s.resident_bytes())
  assert wn.t_s==day.t_s and wn.hydro_count==day.hydro_count and wn.n_requests==day.n_requests+1
  if i==6:
   tok=s.token;state0=s.snapshot_state();credit0=s.pending();counter0=s.counters()
   try:
    with mock.patch.object(bio,'rebuild_geometry',side_effect=RuntimeError('injected late geometry rebuild failure')):
     s.stage_day(prop,ac)
   except RuntimeError as exc:
    assert str(exc)=='injected late geometry rebuild failure'
   else:raise AssertionError('weekly geometry failure was not exercised')
   assert s.token==tok and s.counters()==counter0
   np.testing.assert_array_equal(s.snapshot_state().agb_dry_kg,state0.agb_dry_kg)
   for k in credit0:np.testing.assert_array_equal(s.pending()[k],credit0[k])
   failed_rollback=True
   ck=work/'pending_after_debit'
   man=pc.save_biology(ck,s,external=ac.after,limits=pc.CheckpointLimits(1<<26),pending=pc.PendingStage(plan,prop,ac),syrup_source=source.package_sha256)
   restored=pc.load_biology(ck,cfg,bindings=pc.ExternalBindings(ac.after),limits=pc.CheckpointLimits(1<<26))
   s=restored.session
   plan,prop,ac=pc.rebuild_pending(s,restored.pending)
   pending_resume=True  # no native request is made by the replay path
  st=s.stage_day(prop,ac)
  assert abs(float(st.feedback.delivered_m3.sum())-ac.actual_total_m3)<=float(cand.budget.bound_m3)+1e-20
  token=s.token
  st.discard()
  assert s.token==token
  st=s.stage_day(prop,ac)
  s=s.publish(st)
  try:s.publish(st)
  except ps.PlantSessionStateError:pass
  else:raise AssertionError('duplicate publication accepted')
  w=wl.close_interevent(lop,wn,expected_parent=wl.expect_parent(wn))
  audit=wl.audit_lifecycle(lop,w)
  assert abs(audit['residual_m3'])<=audit['bound_m3']
  if i==2:
   ext=lb.resource_identity(lop,w);ck=work/'closed_boundary'
   pc.save_biology(ck,s,external=ext,limits=pc.CheckpointLimits(1<<26),syrup_source=source.package_sha256)
   r=pc.load_biology(ck,cfg,bindings=pc.ExternalBindings(ext),limits=pc.CheckpointLimits(1<<26))
   assert r.session.token==s.token;s=r.session;closed_resume=True
  sync();elapsed=time.perf_counter()-tick
  rows.append({'day':i,'hydro_count':w.hydro_count,'n_requests':w.n_requests,'t_s':w.t_s,'T_actual_m3':ac.actual_total_m3,'closure_residual_m3':audit['residual_m3'],'closure_bound_m3':audit['bound_m3'],'growth_updates':s.counters()['growth_updates'],'wall_s':elapsed,'resident_array_bytes':s.resident_bytes()})
  del plan,prop,ac,st,rd,depth,cand,day,wn,dr
  gc.collect()
finally:
 sync()
 if hook:hook.__exit__(None,None,None)
cur,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
assert w.hydro_count==8 and w.n_requests==8 and w.t_s==8*86400.0
assert s.counters()['growth_updates']==1 and s.counters()['withdrawals_accepted']==8
assert s.geometry_sha256==s.export_roots()['geometry_sha256']
assert np.all(s.pending()['D']==0) and np.all(s.pending()['U']==0) # day8 T-on/G-off has no deferred credit
age=host(s.snapshot_state().age_days);np.testing.assert_array_equal(age,host(plants.age_days)+8.)
assert failed_rollback and pending_resume and closed_resume
out={'backend':backend,'source':source.package_sha256,'native_maple_package':profile.package_digest,'startup_wall_s':startup_s,'rows':rows,'model_days':8,'physical_seconds':w.t_s,'one_hydro_and_one_T_each_day':True,'late_geometry_rollback':failed_rollback,'pending_after_debit_resume_without_water_replay':pending_resume,'closed_boundary_restart':closed_resume,'final_counters':s.counters(),'growth_disabled_T_has_no_credit':True,'final_dry_agb_kg':host(s.snapshot_state().agb_dry_kg).tolist(),'final_seasonal_dry_agb_kg':host(s.snapshot_state().seasonal_agb_dry_kg).tolist(),'final_age_days':age.tolist(),'byte_ledger':s.memory_ledger(),'max_bytes':cfg.max_bytes,'tracemalloc_current_bytes':cur,'tracemalloc_peak_bytes':peak,'process_maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'memory_scope':'tracemalloc includes Python and CPU allocations after initialization; RSS includes native/JIT/context peak. CUDA pool live peak includes native+LEAF and pool alignment, excludes context and cached free blocks. Synthetic three-plant 4x6 harness, no domain extrapolation.'}
if hook:out.update(cuda_pool_live_peak_bytes=hook.peak,cuda_pool_retained_bytes=pool.total_bytes(),cuda_allocation_calls=hook.allocs,cuda_free_calls=hook.frees)
print(json.dumps(out,indent=2))
(work/'qualification.json').write_text(json.dumps(out,indent=2)+'\n')
