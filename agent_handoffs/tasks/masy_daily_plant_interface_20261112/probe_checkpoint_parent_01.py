import json,pathlib,sys,tempfile
import numpy as np
TASK=pathlib.Path(__file__).resolve().parent
SNAP=TASK/"public_evidence/review_01"
sys.path.insert(0,str(SNAP/"tests"))
import native_leaf_fixture as nf
from leaf import lifecycle_bridge as lb,plant_session as ps,plant_checkpoint as ck
from maple_syrup import daily_water as dw,water_lifecycle as wl
work=TASK/"public_evidence/checkpoint_parent_probe_01";work.mkdir(exist_ok=False)
lop,boundary,profile=nf.native_daily_harness(work)
s0=dw.start_daily_lineage(lop,boundary,expected_parent=wl.expect_parent(boundary),run_digest="ef"*32)
pet=lop.xp.full(lop.n_cells,1e-8)
res=dw.daily_water_action(lop,s0,expected_parent=wl.expect_parent(s0),plan=dw.next_plan(s0,2.,10.,5.),pet_terrain_m_s=pet)
water=res.state;cat=nf.make_catalog(n_layers=lop.n_layers);cfg=nf.make_config(cat)
s=ps.PlantSession.start(nf.native_plants_for(lop,cat),cfg,anchor=nf.anchor(),session_id="ckparent")
plan=s.prepare_daily(nf.day_ctx(nf.anchor(),0),terrain_pet=lb.pet_m_s_to_mm_per_day(pet))
rd=lb.read_availability(lb.bind_source(profile.syrup_package_digest),lop,water)
pr=s.propose_withdrawal(plan,rd)
after,cand=wl.request_withdrawal(lop,water,expected_parent=wl.expect_parent(water),depth_m=lb.request_depth_m(rd,pr.proposed_m3),identity=pr.identity)
ac=lb.accepted_withdrawal(rd,pr.identity,after,cand,lop=lop)
st=s.stage_day(pr,ac);child=s.publish(st)
ck.save_biology(work/"bio",child)
extra=np.zeros((lop.n_layers,lop.n_cells));extra[0]=1e-6
wrong,_=wl.request_withdrawal(lop,after,expected_parent=wl.expect_parent(after),depth_m=extra,identity="unpaired-extra-T")
try:
 restored=ck.load_biology(work/"bio",cfg,bindings=ck.ExternalBindings(lb.resource_identity(lop,wrong)))
 accepted=True;error=None
except ck.PlantCheckpointError as exc:
 accepted=False;error=str(exc)
print(json.dumps({"wrong_external_parent_accepted":accepted,"saved_request_sequence":after.request_sequence,"supplied_request_sequence":wrong.request_sequence,"saved_T_m3":wl.audit_lifecycle(lop,after)["transpiration_m3"],"supplied_T_m3":wl.audit_lifecycle(lop,wrong)["transpiration_m3"],"error":error},indent=2))
assert not accepted,"biological checkpoint accepted a different physical state containing unpaired T"
