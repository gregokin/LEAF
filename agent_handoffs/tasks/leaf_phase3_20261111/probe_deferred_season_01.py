"""Independent chronology probe: recruits born after year-end must not lose tissue to an earlier season."""
import datetime as dt,json
import numpy as np
from leaf import forcing
from review_tools import make,step_day,host,DAY
from probe_recruitment_01 import raw_catalog

raw=raw_catalog()
for item in raw['species'].values():item['mortality']=None
session,cat,op,xp=make(raw,[],backend='numba')
# A closed interevent gate passes a year end with no plants and no water or plant process.
end=dt.datetime(2025,1,1,tzinfo=dt.timezone.utc)
session.step_interevent(forcing.TimestepRecord(session.cursor,end,0.,None,'interevent','closed'))
assert session.season_due
surface=np.zeros((4,4))
receipt=session.apply_reproduction(2024,surface,max_new_plants=100,max_total_records=100,max_attempts=100)
born=session.snapshot();water=host(session.native_state.water_m).copy()
day=step_day(session,xp)[0];after=session.snapshot()
loss=day.get('deferred_season',{}).get('ledger',{})
report={'born_after_yearend':True,'birth_B_kg':born.agb_dry_kg.tolist(),'birth_E_kg':born.seasonal_agb_dry_kg.tolist(),
        'after_first_day_B_kg':after.agb_dry_kg.tolist(),'after_first_day_E_kg':after.seasonal_agb_dry_kg.tolist(),
        'prior_season_applied_to_newborns':bool(day.get('deferred_season_applied')),'deferred_season_ledger':loss}
print(json.dumps(report,indent=2))
assert not report['prior_season_applied_to_newborns'],'deferred season from before birth applied to new recruits'
