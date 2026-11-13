import pathlib,sys,json
import numpy as np
TASK=pathlib.Path(__file__).resolve().parent;SNAP=TASK/"public_evidence/review_02"
sys.path.insert(0,str(SNAP/"tests"))
import native_leaf_fixture as nf
from leaf import plant_session as ps,lifecycle_bridge as lb
from maple_syrup import daily_water as dw,water_lifecycle as wl
cat=nf.make_catalog();cfg=nf.make_config(cat);s=ps.PlantSession.start(nf.make_state(cat),cfg,anchor=nf.anchor())
try:
 p=s.prepare_daily(ps.DayContext(nf.anchor(),20.,True,True,process_kind="wind"),terrain_pet=np.full(s.base.domain.ncells,5.))
 wind_accepted=True;wind_demand=float(p.D.sum());wind_error=None
except ps.PlantSessionError as exc:
 wind_accepted=False;wind_demand=None;wind_error=str(exc)
work=TASK/"public_evidence/rev02_contract_native";work.mkdir(exist_ok=False)
lop,boundary,profile=nf.native_daily_harness(work)
w0=dw.start_daily_lineage(lop,boundary,expected_parent=wl.expect_parent(boundary),run_digest="ab"*32)
pet=np.full(lop.n_cells,1e-8)
w=dw.daily_water_action(lop,w0,expected_parent=wl.expect_parent(w0),plan=dw.next_plan(w0,2.,10.,5.),pet_terrain_m_s=pet).state
cat=nf.make_catalog(n_layers=lop.n_layers);cfg=nf.make_config(cat)
s=ps.PlantSession.start(nf.native_plants_for(lop,cat),cfg,anchor=nf.anchor(year=2037))
p=s.prepare_daily(nf.day_ctx(nf.anchor(year=2037),0),terrain_pet=lb.pet_m_s_to_mm_per_day(pet))
rd=lb.read_availability(lb.bind_source(profile.syrup_package_digest),lop,w,max_bytes=cfg.max_bytes)
try:
 s.propose_withdrawal(p,rd);unbound=True;unbound_error=None
except ps.PlantSessionError as exc:
 unbound=False;unbound_error=str(exc)
print(json.dumps({"wind_context_accepted_as_interevent":wind_accepted,"wind_context_demand_m3":wind_demand,"wind_error":wind_error,"missing_day_mapping_accepts_unrelated_nominal_year":unbound,"nominal_date":p.day.nominal_date.isoformat(),"native_daily_ordinal":rd.identity.daily_ordinal,"unbound_error":unbound_error},indent=2))
assert not wind_accepted and not unbound,"transport gate or mandatory nominal-day binding is missing"
