import datetime as dt,json,pathlib,tempfile
import numpy as np
from leaf import growth,traits,state,plant_session as ps,plant_checkpoint as ck,lifecycle_bridge as lb
TASK=pathlib.Path(__file__).resolve().parent
cat=traits.load_catalog(TASK.parents[2]/"examples/phase1/species.json")
table=growth.compile_species(cat,"numpy",100000000)
cfg=ps.SessionConfig(cat,table,"numpy",100000,100000000,"declared_none",7,builder="numpy",canopy={"max_edges":100000,"max_bytes":100000000,"builder":"numpy"})
def plants(pid=1,b=.1):
 return state.PlantState.from_records(cat,state.Domain(2,2,1.),[dict(plant_id=pid,species="grass",x_m=.5,y_m=.5,establishment_ground_m=0.,agb_dry_kg=b,seasonal_agb_dry_kg=0.,age_days=5.,status="live")])
anchor=dt.datetime(2026,5,1,tzinfo=dt.timezone.utc)
a=ps.PlantSession.start(plants(),cfg,anchor=anchor,session_id="same")
c=ps.PlantSession.start(plants(pid=2,b=.2),cfg,anchor=anchor,session_id="same")
plan=a.prepare_daily(ps.DayContext(anchor,22.,True,True),terrain_pet=np.full(4,5.))
rid=lb.ResourceIdentity("11"*32,"22"*32,0,0,1,0,86400.,daily_ordinal=1)
read=lb.AvailabilityRead(rid,np.full((2,4),1.),1.,2,4,"numpy")
try:c.propose_withdrawal(plan,read);wrong=True
except ps.PlantSessionError:wrong=False
base=plants();s=ps.PlantSession.start(base,cfg,anchor=anchor,session_id="alias")
before=s.snapshot_state().x_m.copy()
try:base.x_m[0]=1.5;alias_changed=not np.array_equal(before,s.snapshot_state().x_m)
except ValueError:alias_changed=False
d=ps.PlantSession.start(plants(),cfg,anchor=anchor,session_id="restart")
p=d.prepare_daily(ps.DayContext(anchor,22.,False,False),terrain_pet=0.)
d=d.publish(d.stage_dry_day(p))
with tempfile.TemporaryDirectory(dir=TASK/"public_evidence") as temp:
 directory=pathlib.Path(temp)/"bio"
 try:
  ck.save_biology(directory,d)
  restored=ck.load_biology(directory,cfg,bindings=ck.ExternalBindings(rid))
  restart_ok=True;restart_error=None
 except Exception as exc:
  restart_ok=False;restart_error=f"{type(exc).__name__}: {exc}"
print(json.dumps({"different_plants_same_parent_token":a.token==c.token,"foreign_plant_plan_accepted":wrong,"caller_mutation_changed_accepted_locations":alias_changed,"ordinary_one_day_restart_ok":restart_ok,"restart_error":restart_error},indent=2))
