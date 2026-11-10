"""Analytical native-water and calendar/retry oracles, independent of Claude's tests."""
from pathlib import Path
import argparse
import json
import math
import numpy as np

from leaf.coupling import CoupledSimulation
from benchmarks import spatial_mapping as sm
from test_coupling import make_case, make_inputs, make_column, host, RES_M, WET
from maple_syrup import subsurface as ss


def traits():
    return {'grass': {'mature_biomass_kg': 1., 'intrinsic_rate_per_year': .4,
                      'biomass_support_kg_per_m3': 10., 'live_fraction': .8,
                      'canopy_biomass_density_kg_m2': 2.,
                      'potential_transpiration_m_per_day': .004,
                      'extra_growth_fraction': .2, 'monthly_activity': [1.] * 12}}


def build(backend, dx):
    case = make_case(dx=dx, grass_diameter=.2, shrub_diameter=2., root_multiplier=3.,
                     root_layer_fractions={'grass': [.8, .2]})
    inp = make_inputs([dx / 2], [dx / 2], [.1], mult=3., pid=[0])
    col = make_column(case, backend)
    water = col.xp.asarray([[WET], [RES_M]], dtype=col.xp.float64)
    state = ss.column_state(col, water)
    return CoupledSimulation(case, col, state, inp, stepwat_traits=traits())


def forcing(sim, pet=0.):
    return (sim.xp.zeros((sim.L, sim.N), dtype=sim.xp.float64),
            sim.xp.full(sim.N, pet, dtype=sim.xp.float64))


def scalar_update(b, e, uptake, days, tf, year_end=False):
    # Scalar reference from the frozen task contract, not the production kernel.
    required = b * .8 * days / 365
    available = 10 * uptake
    water = min(1., available / required) if required else 0.
    normal = min(.4 * .95 * tf * days / 365 * water * (b - e) * (1 - (b - e)),
                 max(1 - b, 0.))
    extra = min(.2 * max(available - required, 0.) * tf, max(1 - b - normal, 0.))
    loss = e + extra if year_end else 0.
    return np.array([b + normal + extra - loss,
                     0. if year_end else e + extra, normal + extra, loss])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    backend = 'cuda' if args.gpu else 'numpy'
    for dx in (.25, 1.):
        sim = build(backend, dx)
        added, pet = forcing(sim, pet=1e-9)
        b0 = 2 * math.pi * .1 ** 2
        demand = math.pi * .1 ** 2 * .004 * .8
        actual = .8 * demand  # top roots wet; bottom at residual, no compensation
        for _ in range(7):
            result = sim.step_day(added, pet, temperature_factor=1.)
            np.testing.assert_allclose(result.transpiration_budget.actual_m3, actual, rtol=2e-12)
        expected = scalar_update(b0, 0., actual * 7, 7, 1.)
        veg, sw = sim.vegetation, sim.vegetation.stepwat
        np.testing.assert_allclose([host(veg.B)[0], host(sw.E)[0],
                                   host(veg.gain_cum)[0], host(veg.loss_cum)[0]],
                                  expected, rtol=3e-12, atol=1e-14)
        summary = sim.summary()
        assert summary['ok'], summary['checks']
        np.testing.assert_allclose(summary['water']['evaporation'], 7 * dx * dx * 86400e-9,
                                   rtol=2e-12)
        np.testing.assert_allclose(summary['water']['transpiration'], actual * 7, rtol=2e-12)
    # Real day364→365 partial period, failure then retry, followed by day371.
    sim = build(backend, 1.)
    added, pet = forcing(sim)
    for _ in range(364):
        sim.step_day(added, pet, temperature_factor=.7)
    veg, sw = sim.vegetation, sim.vegetation.stepwat
    b, e = float(host(veg.B)[0]), float(host(sw.E)[0])
    before_gain, before_loss = float(host(veg.gain_cum)[0]), float(host(veg.loss_cum)[0])
    original = veg._build_graph
    def refuse(*_):
        raise sm.AdmissionRejected('independent.year_end', 'intentional graph refusal', {})
    veg._build_graph = refuse
    try:
        sim.step_day(added, pet, temperature_factor=.7)
    except sm.AdmissionRejected:
        pass
    else:
        raise AssertionError('year-end rebuild did not refuse')
    assert sim.days == veg.day == 365 and veg.weekly_pending
    assert sw.period_start_day == 364
    assert float(host(veg.B)[0]) == b and float(host(sw.E)[0]) == e
    water, time_s = host(sim.state.water_m).copy(), sim.state.t_s
    uptake = float(host(veg.uptake_week)[0])
    expected = scalar_update(b, e, uptake, 1, .7, year_end=True)
    np.testing.assert_allclose(host(sw.temp_integral), [.7 / 365], rtol=3e-14)
    veg._build_graph = original
    sim.retry_weekly_update()
    assert not veg.weekly_pending and sim.days == 365 and sw.period_start_day == 365
    assert sim.state.t_s == time_s
    np.testing.assert_array_equal(host(sim.state.water_m), water)
    np.testing.assert_allclose([host(veg.B)[0], host(sw.E)[0],
                               host(veg.gain_cum)[0] - before_gain,
                               host(veg.loss_cum)[0] - before_loss], expected, rtol=3e-12, atol=2e-14)
    assert sw.growth_updates == 53 and sw.year_end_updates == 1 and veg.weeks_done == 52
    assert float(host(veg.uptake_week)[0]) == 0 and float(host(sw.temp_integral)[0]) == 0
    sim.step_days(((*forcing(sim), .7) for _ in range(6)))
    assert sim.days == 371 and sw.growth_updates == 54 and veg.maps_built == 55
    assert sw.period_start_day == 371 and sim.summary()['ok']
    # Invalid forcing must leave the accepted water day, mass and climate state untouched.
    before = (sim.days, sim.state.t_s, host(sim.state.water_m).copy(), host(sw.temp_integral).copy())
    for bad in (None, float('nan'), -1., 1.1, [1., 1.]):
        try:
            sim.step_day(*forcing(sim), temperature_factor=bad)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f'invalid temperature accepted: {bad}')
        assert (sim.days, sim.state.t_s) == before[:2]
        np.testing.assert_array_equal(host(sim.state.water_m), before[2])
        np.testing.assert_array_equal(host(sw.temp_integral), before[3])
    print(json.dumps({'ok': True, 'backend': backend,
                      'oracles': ['native two-area E/T accounting', 'actual uptake to analytical biomass',
                                  '365-day partial period', 'year-end refusal and single retry',
                                  'following partial week', 'invalid forcing before water mutation']}))


if __name__ == '__main__':
    main()
