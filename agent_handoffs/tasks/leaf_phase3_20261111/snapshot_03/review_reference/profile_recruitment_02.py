"""Independent bounded actual-native CUDA annual-birth and following-day array/memory/timing profile."""
import datetime as dt,gc,json,sys,time,tracemalloc
from pathlib import Path
import numpy as np
from leaf import forcing,state,traits
from leaf.interevent import IntereventSession
from maple_syrup import subsurface as ss
from maple_syrup import subsurface_transpiration as st
from probe_recruitment_01 import raw_catalog
from memory_tools import host_array_bytes
from unittest import mock
from leaf import recruitment as rp
sys.path.append(str(Path(__file__).resolve().parent.parent/'leaf_phase1_20261110'))
from profile_uptake_memory import make_hook


def main():
    import cupy as cp
    started=time.perf_counter()
    count=int(sys.argv[1]) if len(sys.argv)>1 else 100_000
    reject=True
    sampled=[]
    raw=raw_catalog()
    for item in raw['species'].values():item['mortality']=None
    raw['species']['grass']['reproduction']['traits']['clonal']['dispersal_radius_m']=1e9 if reject else .1
    if reject:
        raw['species']['shrub']['reproduction']=None
        raw['species']['grass']['reproduction']['traits']['clonal'].update(
            max_daughters_per_parent=4,initial_agb_dry_kg=.01)
    cat=traits.SpeciesCatalog(raw);nx=64;N=nx*nx
    domain=state.Domain(nx,nx,1.)
    plants=state.PlantState(domain,cat.species_ids,cat.hash,
        ids=np.arange(1,count+1,dtype=np.int64),species_index=np.full(count,cat.index('grass'),dtype=np.int32),
        x_m=np.full(count,.5),y_m=np.full(count,.5),establishment_ground_m=np.zeros(count),
        agb_dry_kg=np.ones(count),seasonal_agb_dry_kg=np.zeros(count),age_days=np.full(count,1000.),
        status=np.where(np.arange(count)%4==0,state.LIVE,state.DEAD_STANDING).astype(np.int8))
    with make_hook(cp) as hook:
        layers=ss.soil_layers([0.,.25,1.],theta_residual=.05,theta_field_capacity=.3,
                              theta_saturation=.5,diffusivity_m2_s=0.,theta_evaporation_stress=.1)
        col=ss.prepare_column(layers,cp.ones(N,dtype=bool),1.,backend='cuda');op=st.prepare_transpiration(col)
        water=ss.column_state(col,cp.asarray(np.broadcast_to([[.05],[.15]],(2,N)).copy()))
        session=IntereventSession(plants,cat,operator=op,native_state=water,
            start=dt.datetime(2024,12,31,tzinfo=dt.timezone.utc),backend='cuda',rng_seed=123,
            growth_cadence=dt.timedelta(days=7),max_edges=1_000_000,max_bytes=1<<29,root_builder='compiled',
            reproduction_patch_size_m=8.)
        del water,plants
        added,evap=cp.zeros((2,N)),cp.zeros(N)
        record=forcing.TimestepRecord(session.cursor,session.cursor+dt.timedelta(days=1),25.,pet_mm_per_day=3.)
        session.step_interevent(record,added_water_m=added,potential_evaporation_m_s=evap)
        surface=np.zeros((nx,nx))
        gc.collect();cp.cuda.Device().synchronize()
        host_before=host_array_bytes(session,added,evap,surface)
        device_before=hook.raw;hook.raw_peak=hook.raw;hook.rounded_peak=hook.rounded
        startup_seconds=time.perf_counter()-started
        tracemalloc.start();timer=time.perf_counter()
        real_contains=state.Domain.contains
        real_u=rp._u
        def sample():
            snap=tracemalloc.take_snapshot()
            new_numpy=sum(trace.size for trace in snap.traces if trace.domain==np.lib.tracemalloc_domain)
            sampled.append({'new_numpy_array_bytes':new_numpy,'device_current_bytes':hook.raw,
                            'global_array_lower_bound_bytes':host_before+new_numpy+hook.raw})
        def observed_contains(dom,x,y):
            result=real_contains(dom,x,y);sample();return result
        def observed_u(*args):
            result=real_u(*args);sample();return result
        with mock.patch.object(state.Domain,'contains',observed_contains),mock.patch.object(rp,'_u',observed_u):
            rep=session.apply_reproduction(2024,surface,max_new_plants=count,max_total_records=2*count,
                                           max_attempts=2*count)
        cp.cuda.Device().synchronize();birth_seconds=time.perf_counter()-timer
        traced_live,traced_peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        birth_declared=rep['memory']['peak_estimate_bytes'];birth_device_peak=hook.raw_peak
        birth_host_after=host_array_bytes(session,added,evap,surface)
        gc.collect();cp.cuda.Device().synchronize()
        host_after=host_array_bytes(session,added,evap,surface)
        hook.raw_peak=hook.raw;hook.rounded_peak=hook.rounded
        record=forcing.TimestepRecord(session.cursor,session.cursor+dt.timedelta(days=1),25.,pet_mm_per_day=3.)
        timer=time.perf_counter();day=session.step_interevent(record,added_water_m=added,potential_evaporation_m_s=evap)
        cp.cuda.Device().synchronize();daily_seconds=time.perf_counter()-timer
        observed=host_after+hook.raw_peak;declared=day['memory']['peak_estimate_bytes']
        diag=rep['reproduction']
        report={'scope':'one annual population boundary and one actual native day, synthetic clustered population, no transport scenario',
            'records_before':count,'live_before':count//4,'records_after':session.n_total_records,'live_after':session.n_live,
            'requested_daughters':diag['n_daughters_requested'],'placed_daughters':diag['n_daughters_placed'],
            'new_seedlings':diag['n_seed_recruits'],'all_placements_rejected_case':reject,
            'root_edges_after':session.root_map.n_edges,'cells':N,'layers':2,
            'startup_and_warmup_seconds':startup_seconds,'annual_birth_seconds':birth_seconds,'following_day_seconds':daily_seconds,
            'birth_host_resident_before_bytes':host_before,'birth_device_before_bytes':device_before,
            'birth_device_requested_peak_bytes':birth_device_peak,'birth_host_resident_after_bytes':birth_host_after,
            'birth_host_tracemalloc_new_peak_bytes':traced_peak,'birth_host_tracemalloc_new_live_bytes':traced_live,
            'birth_observer_samples':sampled,'birth_sampled_global_array_lower_bound_bytes':max(x['global_array_lower_bound_bytes'] for x in sampled),
            'birth_declared_global_peak_bytes':birth_declared,'birth_stage_estimates':rep['memory']['stage_totals'],
            'pure_recruitment_scratch_estimate_bytes':diag['memory_estimate']['total_bytes'],
            'daily_host_resident_array_bytes':host_after,'daily_device_requested_peak_bytes':hook.raw_peak,
            'daily_global_array_peak_lower_bound_bytes':observed,'daily_declared_global_peak_bytes':declared,
            'daily_underestimate_confirmed':observed>declared,
            'pool_used_bytes':cp.get_default_memory_pool().used_bytes(),'pool_reserved_bytes':cp.get_default_memory_pool().total_bytes(),
            'pool_cap_bytes':cp.get_default_memory_pool().get_limit(),
            'limitations':'resident host arrays + hooked device peak is an array-data lower bound; host tracemalloc includes Python overhead and is reported separately. No complete process/context peak or 30-year extrapolation.'}
        report['birth_underestimate_confirmed']=report['birth_sampled_global_array_lower_bound_bytes']>birth_declared
        report['limitations']+=' Birth timing includes allocation observation instrumentation; exact concurrent samples count NumPy allocation-domain bytes plus resident host arrays and current hooked device bytes.'
        del session,col,op,added,evap
        gc.collect();cp.cuda.Device().synchronize();report.update(unknown_free_events=hook.unknown_frees,requested_live_at_end_bytes=hook.raw)
    print(json.dumps(report,indent=2))
    return int(report['daily_underestimate_confirmed'] or report['birth_underestimate_confirmed'] or report['unknown_free_events'])

if __name__=='__main__':raise SystemExit(main())
