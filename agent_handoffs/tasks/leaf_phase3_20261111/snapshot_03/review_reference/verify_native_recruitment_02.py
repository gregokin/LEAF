"""Independent actual-native reproduction, rollback, root/age/AGB ledgers and exact restart."""
import datetime as dt
import json
from pathlib import Path
import sys,tempfile
from unittest import mock
import numpy as np
from leaf import state,traits
from leaf.interevent import IntereventSession
from probe_recruitment_01 import raw_catalog
from review_tools import make,record,host,step_day,fingerprint,same_population,DAY

LIMITS=dict(max_new_plants=100,max_total_records=1000,max_attempts=1000)

def raw():
    doc=raw_catalog()
    for item in doc['species'].values():
        item['growth']['traits']['monthly_activity']=[1.]*12  # explicit all-year test fixture, not field phenology
        item['mortality']={'law':'age_background_stress_v1','traits':{
            'age_hazard':None,'background_hazard_per_year':0.,'stress':{
                'metric':'support_ratio','low_threshold':.1,'exempt_until_age_days':365.,
                'limit_years':2,'counter_cap_years':10,'mortality_probability':0.,'biomass_loss_fraction':.1}}}
    return doc

def opportunity(session,xp):
    step_day(session,xp)
    if session.annual_due_year is not None:session.apply_annual_decision(2024)
    assert session.cursor==dt.datetime(2025,1,1,tzinfo=dt.timezone.utc)

def main():
    backend=sys.argv[1] if len(sys.argv)>1 else 'numba';reports=[]
    surface=np.arange(16,dtype=np.float64).reshape(4,4)*.01
    for empty in [False,True]:
        records=[] if empty else [record(-7,biomass=1.,seasonal=.2),
            record(0,'shrub',x=2.5,biomass=2.,seasonal=.2),
            record(9,biomass=.02,seasonal=0.,status='dead_standing',age=500.)]
        session,cat,op,xp=make(raw(),records,backend=backend,overrides=None if empty else {-7:[.9,.1]})
        opportunity(session,xp)
        for _ in range(6):step_day(session,xp)
        session.flush_growth()
        old_metrics=session.annual_accumulators()
        old_stress=session.stress_counters()
        before=fingerprint(session);old=session.snapshot();water=host(session.native_state.water_m).copy()
        # admission refusal and root-builder failure must publish nothing and consume no random opportunity
        if not empty:
            try:session.apply_reproduction(2024,surface,**{**LIMITS,'max_new_plants':0})
            except ValueError:pass
            else:raise AssertionError('birth-count limit accepted')
            assert fingerprint(session)==before,'count refusal mutated accepted state'
        with mock.patch.object(IntereventSession,'_build_geometry',side_effect=RuntimeError('independent root failure')):
            try:session.apply_reproduction(2024,surface,**LIMITS)
            except RuntimeError as exc:assert str(exc)=='independent root failure'
            else:raise AssertionError('injected root failure did not propagate')
        assert fingerprint(session)==before,'root failure mutated accepted state'
        with tempfile.TemporaryDirectory(prefix='leaf-phase3-independent-') as temp:
            saved=session.save(Path(temp)/'before_birth')
            restored=IntereventSession.load(saved,cat,operator=op,backend=backend,max_edges=100_000,
                                            max_bytes=150_000_000,root_builder='compiled')
            receipts=[]
            for run in [session,restored]:
                receipts.append(run.apply_reproduction(2024,surface,**LIMITS))
                after=run.snapshot();n0=old.n
                np.testing.assert_array_equal(host(run.native_state.water_m),water)
                assert run.native_state.t_s==before['native_t']
                assert after.n>n0
                np.testing.assert_array_equal(after.ids[:n0],old.ids)
                np.testing.assert_array_equal(after.x_m[:n0],old.x_m)
                np.testing.assert_array_equal(after.y_m[:n0],old.y_m)
                np.testing.assert_array_equal(after.age_days[:n0],old.age_days)
                np.testing.assert_array_equal(after.establishment_ground_m[:n0],old.establishment_ground_m)
                np.testing.assert_array_equal(after.age_days[n0:],0.)
                np.testing.assert_array_equal(after.establishment_ground_m[n0:],
                    after.domain.sample(surface,after.x_m[n0:],after.y_m[n0:]))
                assert len(np.unique(after.ids))==after.n
                np.testing.assert_array_equal(np.sort(host(run.root_map.ids)),np.sort(after.ids[after.status==state.LIVE]))
                credit=receipts[-1]['reproduction']['ledger']['external_seed_agb_kg']
                np.testing.assert_allclose(np.sum(after.agb_dry_kg),np.sum(old.agb_dry_kg)+credit,rtol=2e-14)
                if not empty:np.testing.assert_array_equal(run._overrides[-7],[.9,.1])
                metrics=run.annual_accumulators()
                n_old_live=int(np.count_nonzero(old.status==state.LIVE))
                for name,value in old_metrics.items():
                    np.testing.assert_array_equal(metrics[name][:n_old_live],value)
                    np.testing.assert_array_equal(metrics[name][n_old_live:],0.)
                np.testing.assert_array_equal(run.stress_counters()[:n_old_live],old_stress)
                np.testing.assert_array_equal(run.stress_counters()[n_old_live:],0)
                root=run.root_map
                offsets,resources,weights=map(host,(root.offsets,root.resources,root.weights))
                for j,pid in enumerate(host(root.ids)):
                    index=int(np.flatnonzero(after.ids==pid)[0])
                    sid=cat.species_ids[int(after.species_index[index])]
                    expected=[.9,.1] if int(pid)==-7 else cat.section(sid,'roots')['layer_fractions']
                    measured=np.bincount(resources[offsets[j]:offsets[j+1]]//run._N,
                                         weights=weights[offsets[j]:offsets[j+1]],minlength=2)
                    np.testing.assert_allclose(measured,expected,rtol=2e-14,atol=1e-15)
                again=fingerprint(run)
                try:run.apply_reproduction(2024,surface,**LIMITS)
                except ValueError:pass
                else:raise AssertionError('opportunity replay accepted')
                assert fingerprint(run)==again
            same_population(session.snapshot(),restored.snapshot())
            # next real day: new roots take water, all current live rows accrue water metrics once
            for run in [session,restored]:
                step_day(run,xp)
                now=run.snapshot();newmask=~np.isin(host(run.root_map.ids),old.ids)
                assert bool(np.all(host(run._b.U)[newmask]>0.)), 'new plants received no actual native water'
                assert np.any(host(run.native_state.water_m)<water)
                path=run.save(Path(temp)/('pending_'+str(run is restored)))
                loaded=IntereventSession.load(path,cat,operator=op,backend=backend,max_edges=100_000,
                                              max_bytes=150_000_000,root_builder='compiled')
                same_population(run.snapshot(),loaded.snapshot())
                np.testing.assert_array_equal(host(run._b.D),host(loaded._b.D))
                np.testing.assert_array_equal(host(run._b.U),host(loaded._b.U))
                np.testing.assert_array_equal(host(run.native_state.water_m),host(loaded.native_state.water_m))
            same_population(session.snapshot(),restored.snapshot())
            np.testing.assert_array_equal(host(session.native_state.water_m),host(restored.native_state.water_m))
        reports.append({'initially_empty':empty,'rollback_exact':True,'restart_exact':True,'water_untouched_by_birth':True,
                        'new_roots_actual_next_day_uptake':True,'dry_AGB_ledger_closed':True})
    print(json.dumps({'backend':backend,'cases':reports},indent=2))

if __name__=='__main__':main()
