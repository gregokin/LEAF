"""Public-API analytical oracles independent of the contributed tests."""
from pathlib import Path
import json
import math
import sys
from unittest import mock
import numpy as np
from maple_syrup import subsurface as ss
from maple_syrup import subsurface_transpiration as st
from leaf.coupling import CoupledSimulation, CoupledPhaseError
from benchmarks import daily_water_growth as dw
from benchmarks import spatial_mapping as sm


def host(a):
    return a.get() if hasattr(a,'get') else np.asarray(a)


def fixture(backend, area=1., masked=False, available=None):
    if backend=='cuda':
        from maple.core.backend import cupy_module
        xp=cupy_module()
        xp.cuda.Device().synchronize()  # explicit CUDA request must execute, never skip
    else:
        xp=np
    dx=math.sqrt(area)
    case={'name':'independent','nx':2 if masked else 1,'ny':1,'dx':dx,
          'grass_count':1 if masked else 2,'shrub_count':0,'grass_diameter':.1,'shrub_diameter':1.,
          'root_multiplier':3.,'pattern':'uniform','seed':1,'layers':2,
          'max_edges':1000,'max_bytes':100_000_000,
          'root_layer_fractions':{'grass':[.8,.2],'deep':[.2,.8]}}
    layers=ss.soil_layers([0,.1,.3],theta_residual=.05,theta_field_capacity=.25,theta_saturation=.45,
                          diffusivity_m2_s=0,theta_evaporation_stress=.20,gravity_drainage=False)
    N=case['nx']*case['ny'];P=case['grass_count']
    mask=np.array([True,False]) if masked else np.ones(1,bool)
    col=ss.prepare_column(layers,xp.asarray(mask),area,backend=backend,
                          available_bytes_fn=(lambda:available['n']) if available is not None else None)
    radius=.4 if masked else .05
    D=.003*math.pi*radius**2
    w=np.array([[.02,.03],[.02,.03]]) if masked else np.array([[.005],[.01+.5*D/area]])
    state=ss.column_state(col,xp.asarray(w),t_s=3600.)
    inp={'x':np.full(P,.9*dx if masked else .5*dx),'y':np.full(P,.5*dx),
         'canopy_radius':np.full(P,radius),'root_radius':np.full(P,3*radius),
         'ids':np.arange(P,dtype=np.int64),'root_profile_id':np.array([0] if masked else [0,1],np.int32)}
    return case,col,state,inp,D


def zero(sim):
    return sim.xp.zeros((sim.L,sim.N),dtype=np.float64),sim.xp.zeros(sim.N,dtype=np.float64)


def finite_week(backend,area):
    case,col,initial,inp,D=fixture(backend,area)
    initial_array=np.array(host(initial.water_m),copy=True)
    sim=CoupledSimulation(case,col,initial,inputs=inp)
    assert sim.state is initial
    for name in ['W','scale','removed_cum','overflow_cum']:
        assert getattr(sim.vegetation,name,None) is None,name
    result=sim.step_day(*zero(sim))
    expected=np.array([.1*D,.4*D])
    np.testing.assert_allclose(host(sim.vegetation.uptake_cum),expected,rtol=1e-12,atol=1e-16)
    np.testing.assert_allclose(host(sim.state.water_m),np.array([[.005],[.01]]),rtol=1e-14,atol=1e-17)
    assert math.isclose(result.budget.transpiration_m3,.5*D,rel_tol=1e-12,abs_tol=1e-16)
    for _ in range(6):
        sim.step_day(*zero(sim))
    expected_B=math.pi*.05**2*math.exp(-.003*7)+1000*.002*expected
    np.testing.assert_allclose(host(sim.vegetation.B),expected_B,rtol=1e-12,atol=1e-15)
    np.testing.assert_array_equal(host(initial.water_m),initial_array)
    np.testing.assert_array_equal(host(sim.vegetation.root_profile_ids),[0,1])
    assert sim.days==7 and sim.vegetation.day==7 and sim.vegetation.weeks_done==1
    assert sim.vegetation.maps_built==sim.vegetation.demand_map_builds==2
    assert sim.state.t_s==3600+7*86400
    summary=sim.summary();assert summary['ok'],summary['checks']
    old=sim.state
    try:sim.step_day(*zero(sim),phase='wind')
    except CoupledPhaseError:pass
    else:raise AssertionError('Wind phase accepted')
    assert sim.state is old and sim.days==7
    try:
        sim.vegetation.gather_uptake_candidate(sim.xp.ones((2,1),dtype=np.float32),sim.xp.zeros(2,dtype=np.float64))
    except (TypeError,ValueError):pass
    else:raise AssertionError('Wrong-dtype gather accepted')
    return {'backend':backend,'area_m2':area,'expected_total_T_m3':.5*D,
            'actual_T_m3':summary['water']['transpiration'],'water_residual_m3':summary['water']['residual'],
            'biomass_analytical':True,'original_state_immutable':True,'time_and_weekly_calendar':True,
            'phase_refusal':True,'strict_gather_dtype':True}


def masked(backend):
    case,col,initial,inp,D=fixture(backend,masked=True)
    sim=CoupledSimulation(case,col,initial,inputs=inp)
    result=sim.step_day(*zero(sim));s=sim.summary();assert s['ok'],s['checks']
    np.testing.assert_array_equal(host(sim.state.water_m)[:,1],host(initial.water_m)[:,1])
    assert math.isclose(s['water']['initial_storage'],.04,rel_tol=1e-14)
    assert math.isclose(result.transpiration_budget.storage_before_m3,.04,rel_tol=1e-14)
    for k in ['mapped_requested','unmapped_inactive','plant_actual','native_actual']:
        assert math.isclose(s['demand'][k],.5*D,rel_tol=1e-12,abs_tol=1e-16),(k,s['demand'])
    assert math.isclose(s['water']['current_storage'],.04-.5*D,rel_tol=1e-13)
    return {'backend':backend,'inactive_storage_excluded':True,'inactive_water_unchanged':True,
            'unmapped_potential_explicit':True,'no_renormalization':True}


def split(backend):
    case,col,initial,inp,D=fixture(backend)
    a=CoupledSimulation(case,col,initial,inputs=inp)
    b=CoupledSimulation(case,col,initial,inputs=inp)
    a.step_days([zero(a) for _ in range(9)])
    b.step_days([zero(b) for _ in range(4)])
    b.step_days([zero(b) for _ in range(5)])
    np.testing.assert_array_equal(host(a.state.water_m),host(b.state.water_m))
    for name in ['B','canopy','root','uptake_cum','uptake_week','root_profile_ids','R']:
        np.testing.assert_array_equal(host(getattr(a.vegetation,name)),host(getattr(b.vegetation,name)))
    assert a.vegetation.counters()==b.vegetation.counters()
    assert a.cum==b.cum and a.summary()['ok'] and b.summary()['ok']
    return {'backend':backend,'split_4_plus_5_equals_9':True}


def guards():
    available={'n':10**9}
    case,col,state,inp,D=fixture('numpy',available=available)
    op=st.prepare_transpiration(col)
    q=np.zeros((2,1))
    try:st.withdraw_transpiration(op,state,q,extra_reserve_bytes=-10**12)
    except (TypeError,ValueError):pass
    else:raise AssertionError('Negative reserve admitted')
    available['n']=0
    with mock.patch.object(dw,'Simulation',side_effect=AssertionError('allocation before admission')):
        try:CoupledSimulation(case,col,state,inputs=inp)
        except (ss.SubsurfaceMemoryError,sm.AdmissionRejected):pass
        else:raise AssertionError('Zero memory accepted')
    return {'negative_reserve_rejected':True,'native_zero_memory_precedes_vegetation_allocation':True}


def main():
    selected=sys.argv[1:]
    assert selected and set(selected)<= {'cpu','gpu'}
    backends=(['numpy','numba'] if 'cpu' in selected else [])+(['cuda'] if 'gpu' in selected else [])
    results={'analytical_weeks':[finite_week(b,area) for b in backends for area in [1.,.0625]],
             'masked':[masked(b) for b in backends],'split_runs':[split(b) for b in backends],
             'guard_regressions':guards(),'status':'pass'}
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
