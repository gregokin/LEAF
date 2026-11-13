import pathlib,sys,json,datetime as dt,tempfile
import numpy as np
TASK=pathlib.Path(__file__).resolve().parent;SNAP=TASK/'public_evidence/review_04'
sys.path.insert(0,str(SNAP/'tests'))
import native_leaf_fixture as nf
from leaf import plant_session as ps, lifecycle_bridge as lb,plant_checkpoint as pc
from maple_syrup import daily_water as dw,water_lifecycle as wl
work=TASK/'public_evidence/final_unpaired_native';work.mkdir(exist_ok=False)
lop,boundary,profile=nf.native_daily_harness(work)
a=nf.anchor();run='ac'*32
w0=dw.start_daily_lineage(lop,boundary,expected_parent=wl.expect_parent(boundary),run_digest=run)
pet=np.full(lop.n_cells,1e-8)
w=dw.daily_water_action(lop,w0,expected_parent=wl.expect_parent(w0),plan=dw.next_plan(w0,2.,10.,5.),pet_terrain_m_s=pet).state
cat=nf.make_catalog(n_layers=lop.n_layers);cfg=nf.make_config(cat)
s=ps.PlantSession.start(nf.native_plants_for(lop,cat),cfg,anchor=a,day_binding=ps.DayBinding(run,a.date().isoformat()),initial_resource=lb.resource_identity(lop,w0))
source=lb.bind_source(profile.syrup_package_digest)
rd=lb.read_availability(source,lop,w,max_bytes=cfg.max_bytes)
p=s.prepare_daily(ps.DayContext(a,20.,True,True,duration=dt.timedelta(hours=12)),terrain_pet=lb.pet_m_s_to_mm_per_day(pet))
pr=s.propose_withdrawal(p,rd);depth=lb.request_depth_m(rd,pr.proposed_m3,max_bytes=cfg.max_bytes)
w1,n1=wl.request_withdrawal(lop,w,**lb.request_args(rd,lop,w),depth_m=depth,identity=pr.identity,retained_feedback_bytes=pr.retained_feedback_bytes)
ac=lb.accepted_withdrawal(rd,pr.identity,w1,n1,lop=lop,max_bytes=cfg.max_bytes)
s1=s.publish(s.stage_day(pr,ac))
w2,n2=wl.request_withdrawal(lop,w1,expected_parent=wl.expect_parent(w1),depth_m=np.full((lop.n_layers,lop.n_cells),1e-6),identity='unpaired-extra-debit')
rd2=lb.read_availability(source,lop,w2,max_bytes=cfg.max_bytes)
p2=s1.prepare_daily(ps.DayContext(a+dt.timedelta(hours=12),20.,True,True),terrain_pet=lb.pet_m_s_to_mm_per_day(pet))
try:
 s1.propose_withdrawal(p2,rd2);next_ok=True;next_err=None
except ps.PlantSessionError as exc:
 next_ok=False;next_err=str(exc)
try:
 pc.save_biology(work/'ck',s1,external=lb.resource_identity(lop,w2),limits=pc.CheckpointLimits(1<<26));save_ok=True;save_err=None
except pc.PlantCheckpointError as exc:
 save_ok=False;save_err=str(exc)
print(json.dumps({'unpaired_extra_transpiration_accepted_for_next_fragment':next_ok,'unpaired_extra_transpiration_accepted_as_checkpoint_pair':save_ok,'next_error':next_err,'save_error':save_err,'consumed_T':s1.resource_cursor.n_requests,'physical_T':w2.n_requests},indent=2))
assert not next_ok and not save_ok,'Unpaired physical debit passed biological identity/checkpoint guards'
