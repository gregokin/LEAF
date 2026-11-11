"""Independent deferred-season ordering, rollback and no-prebirth tissue-loss checks."""
import json
import numpy as np
from unittest import mock
from leaf import forcing
from leaf.interevent import IntereventSession
from probe_recruitment_01 import raw_catalog
from review_tools import make,record,host,fingerprint,step_day,DAY

reports=[]
for case in ['empty','live_zero_E','live_positive_E']:
    raw=raw_catalog()
    for item in raw['species'].values():item['mortality']=None
    records=[] if case=='empty' else [record(-3,biomass=.8,seasonal=.2 if case=='live_positive_E' else 0.),
        record(0,'shrub',x=2.5,biomass=1.,seasonal=.3 if case=='live_positive_E' else 0.)]
    session,cat,op,xp=make(raw,records,backend='numba')
    session.step_interevent(forcing.TimestepRecord(session.cursor,session.cursor+DAY,0.,None,'interevent','closed'))
    assert session.season_due
    before=fingerprint(session);old=session.snapshot();stock=host(session.native_state.water_m).copy()
    limits=dict(max_new_plants=100,max_total_records=100,max_attempts=100)
    try:session.apply_reproduction(2024,np.zeros((4,4)),**limits)
    except ValueError as exc:assert 'season' in str(exc)
    else:raise AssertionError('birth accepted while old season deferred')
    assert fingerprint(session)==before,'refused chronology mutated accepted state'
    if case=='live_positive_E':
        with mock.patch.object(IntereventSession,'_build_geometry',side_effect=RuntimeError('season root failure')):
            try:session.flush_deferred_season()
            except RuntimeError:pass
            else:raise AssertionError('injected seasonal root failure did not propagate')
        assert fingerprint(session)==before,'failed season finalization changed state'
    flush=session.flush_deferred_season();after_flush=session.snapshot()
    assert not session.season_due
    np.testing.assert_array_equal(host(session.native_state.water_m),stock)
    assert session.native_state.t_s==before['native_t']
    assert session.boundaries()['next_growth_boundary'].isoformat()==before['describe']['next_growth_boundary']
    np.testing.assert_allclose(after_flush.agb_dry_kg,old.agb_dry_kg-old.seasonal_agb_dry_kg,rtol=2e-14,atol=1e-15)
    np.testing.assert_array_equal(after_flush.seasonal_agb_dry_kg,0.)
    finished=fingerprint(session);noop=session.flush_deferred_season()
    assert not noop['published'];assert fingerprint(session)==finished
    session.apply_reproduction(2024,np.zeros((4,4)),**limits)
    born=session.snapshot();n0=old.n
    day=step_day(session,xp)[0];following=session.snapshot()
    assert not day.get('deferred_season_applied',False)
    np.testing.assert_array_equal(following.agb_dry_kg[n0:],born.agb_dry_kg[n0:])
    np.testing.assert_array_equal(following.seasonal_agb_dry_kg[n0:],born.seasonal_agb_dry_kg[n0:])
    reports.append({'case':case,'old_season_finalized_once':True,'rollback_exact':True,
                    'native_water_and_clock_untouched_by_flush':True,'no_prebirth_tissue_loss':True})
print(json.dumps(reports,indent=2))
