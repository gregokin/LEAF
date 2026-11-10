'''Coupled native soil-water / vegetation driver tests (leaf.coupling) against the actual native API.'''
import math
import unittest
from unittest import mock

import numpy as np

from benchmarks import daily_water_growth as dwg
from benchmarks import spatial_mapping as sm
from leaf import coupling
from leaf.coupling import CoupledSimulation, CoupledPhaseError

try:
    from maple_syrup import subsurface as ss
    from maple_syrup import subsurface_transpiration as st
    NATIVE = True
except ImportError:  # pragma: no cover
    NATIVE = False

F = dwg.FIXTURE
DP1 = F['demand_m_per_day'] * math.pi  # m3/day potential demand of a radius-1 plant
THETA_R, THICK = 0.05, 0.1
RES_M = THETA_R * THICK
WET = 0.02  # theta 0.2 (< field capacity 0.25): 0.015 m available per layer/cell


def make_case(**over):
    case = {'name': 'coupled', 'nx': 1, 'ny': 1, 'dx': 10.0, 'grass_count': 1, 'shrub_count': 0,
            'grass_diameter': 2.0, 'shrub_diameter': 4.0, 'root_multiplier': 1.0, 'pattern': 'uniform',
            'seed': 1, 'layers': 2, 'max_edges': 100000, 'max_bytes': 50_000_000}
    case.update(over)
    return case


def make_inputs(xs, ys, radii, mult=1.0, pid=None):
    cr = np.asarray(radii, dtype=np.float64)
    inp = {'x': np.asarray(xs, np.float64), 'y': np.asarray(ys, np.float64), 'canopy_radius': cr,
           'root_radius': mult * cr, 'ids': np.arange(len(cr), dtype=np.int64)}
    if pid is not None:
        inp['root_profile_id'] = pid
    return inp


def xp_for(backend):
    return sm.cp if backend == 'cuda' else np


def make_column(case, backend='numpy', active=None, diffusivity=0.0, available=None, reserve_bytes=0):
    '''diffusivity=0 for isolated uptake oracles (no redistribution); pass a positive value to exercise hydrology.'''
    L, N = case['layers'], case['nx'] * case['ny']
    layers = ss.soil_layers(np.linspace(0.0, THICK * L, L + 1), theta_residual=THETA_R, theta_field_capacity=0.25,
                            theta_saturation=0.40, diffusivity_m2_s=diffusivity, theta_evaporation_stress=0.10)
    mask = np.ones(N, bool) if active is None else np.asarray(active, bool)
    mask = xp_for(backend).asarray(mask)  # explicit setup mask in the selected namespace
    fn = None if available is None else (lambda: available['bytes'])
    return ss.prepare_column(layers, mask, case['dx'] ** 2, backend=backend, reserve_bytes=reserve_bytes,
                             available_bytes_fn=fn)


def build(case, inputs, water=None, backend='numpy', active=None, t_s=0.0, diffusivity=0.0):
    col = make_column(case, backend, active, diffusivity)
    xp = col.xp
    L, N = case['layers'], case['nx'] * case['ny']
    w = np.full((L, N), WET) if water is None else np.asarray(water, dtype=np.float64)
    state = ss.column_state(col, xp.asarray(w), t_s=t_s)
    sim = CoupledSimulation(case, col, state, inputs=inputs)
    return sim, col


def forcing(sim):
    xp = sim.xp
    return xp.zeros((sim.L, sim.N), dtype=xp.float64), xp.zeros(sim.N, dtype=xp.float64)


def host(a):
    return np.asarray(a.get() if hasattr(a, 'get') else a)


@unittest.skipUnless(NATIVE, 'maple_syrup native package / transpiration branch not importable')
class CoupledBasicsTests(unittest.TestCase):
    def test_two_competing_individuals_with_different_fractions(self):
        frac = {'top': [0.8, 0.2], 'bottom': [0.2, 0.8]}
        case = make_case(grass_count=2, root_layer_fractions=frac)
        X = 0.5 * DP1  # m3 available in layer 0 (< requested R0 = DP1); layer 1 plentiful (1.5 m3)
        water = [[RES_M + X / 100.0], [WET]]
        sim, _ = build(case, make_inputs([5.0, 5.0], [5.0, 5.0], [1.0, 1.0], pid=[0, 1]), water)
        self.assertIsNone(sim.vegetation.W)
        added, pet = forcing(sim)
        res = sim.step_day(added, pet)
        s0 = X / DP1  # layer-0 acceptance scale
        np.testing.assert_allclose(host(sim.vegetation.uptake_cum), [DP1 * (0.8 * s0 + 0.2), DP1 * (0.2 * s0 + 0.8)], rtol=1e-9)
        np.testing.assert_allclose(res.transpiration_budget.actual_m3, host(sim.vegetation.uptake_cum).sum(), rtol=1e-9)
        np.testing.assert_allclose(host(sim.state.water_m)[0, 0], RES_M, rtol=1e-9)  # drained to residual, not below
        self.assertEqual(sim.state.t_s, 86400.0)
        s = sim.summary()
        self.assertTrue(s['ok'], s['checks'])
        self.assertAlmostEqual(s['demand']['unmapped_inactive'], 0.0, places=15)

    def test_dry_top_vs_dry_bottom(self):
        frac = {'top': [0.8, 0.2], 'bottom': [0.2, 0.8]}
        for dry, expect in ((0, (0.2, 0.8)), (1, (0.8, 0.2))):
            case = make_case(grass_count=2, root_layer_fractions=frac)
            water = np.full((2, 1), 0.001)  # below residual everywhere ...
            X = 0.5 * DP1
            water[1 - dry, 0] = RES_M + X / 100.0  # ... except X m3 in the wet layer
            sim, _ = build(case, make_inputs([5.0, 5.0], [5.0, 5.0], [1.0, 1.0], pid=[0, 1]), water)
            sim.step_day(*forcing(sim))
            np.testing.assert_allclose(host(sim.vegetation.uptake_cum), [expect[0] * X, expect[1] * X], rtol=1e-9)
            np.testing.assert_allclose(host(sim.state.water_m)[dry, 0], 0.001)  # untouched below residual
            self.assertTrue(sim.summary()['ok'])

    def test_depth_volume_conversion_two_areas(self):
        for dx in (0.25, 1.0):  # 0.0625 m2 and 1 m2 cells
            case = make_case(dx=dx, layers=1, root_layer_fractions={'grass': [1.0]})
            r = 0.1
            sim, _ = build(case, make_inputs([dx / 2], [dx / 2], [r]))
            Dp = F['demand_m_per_day'] * math.pi * r * r
            res = sim.step_day(*forcing(sim))
            np.testing.assert_allclose(host(sim.vegetation.uptake_cum)[0], Dp, rtol=1e-12)
            np.testing.assert_allclose(res.transpiration_budget.actual_m3, Dp, rtol=1e-12)
            np.testing.assert_allclose(WET - host(sim.state.water_m)[0, 0], Dp / (dx * dx), rtol=1e-9)
            np.testing.assert_allclose(host(sim.request_m)[0, 0], Dp / (dx * dx), rtol=1e-12)

    def test_actual_uptake_drives_weekly_growth(self):
        case = make_case(layers=1, root_layer_fractions={'grass': [1.0]})
        avail = 0.3 * DP1  # finite supply: 7 days of demand cannot be met
        sim, _ = build(case, make_inputs([5.0], [5.0], [1.0]), [[RES_M + avail / 100.0]])
        B0 = float(host(sim.vegetation.B)[0])
        for _ in range(7):
            sim.step_day(*forcing(sim))
        v = sim.vegetation
        u = float(host(v.uptake_cum)[0])
        np.testing.assert_allclose(u, avail, rtol=1e-9)
        self.assertLess(u, 7 * DP1 * 0.5)
        np.testing.assert_allclose(host(v.gain_cum)[0], F['water_density_kg_m3'] * F['wue_grass_kg_per_kg'] * u, rtol=1e-12)
        np.testing.assert_allclose(host(v.B)[0], B0 - host(v.loss_cum)[0] + host(v.gain_cum)[0], rtol=1e-12)
        self.assertEqual((v.weeks_done, v.demand_map_builds), (1, 2))
        s = sim.summary()
        self.assertTrue(s['ok'], s['checks'])
        self.assertEqual(s['calendar']['accepted_days'], 7)
        self.assertEqual(s['calendar']['native_t_s'], 7 * 86400.0)

    def test_vegetation_only_has_no_water_and_refuses_synthetic_paths(self):
        case = make_case(layers=1, root_layer_fractions={'grass': [1.0]})
        sim, _ = build(case, make_inputs([5.0], [5.0], [1.0]))
        v = sim.vegetation
        self.assertTrue(v.vegetation_only)
        for n in ('W', 'scale', 'removed_cum', 'overflow_cum', 'W0', 'cap', 'rech', 'period'):
            self.assertIsNone(getattr(v, n), n)
        self.assertNotIn('W', v.live_names())
        self.assertEqual(v.live_bytes(), dwg.state_bytes_estimate(1, 1, 1, n_profiles=1, soil_maps=1))
        self.assertEqual(v.last_ledger['soil_maps'], 1)
        self.assertGreater(v.last_ledger['external_reserve_bytes'], 0)
        with self.assertRaises(ValueError):
            v.step_days(1)
        with self.assertRaises(ValueError):
            v.summary()
        c = v.counters()
        self.assertIsNone(c['recharge_events'])
        self.assertTrue(c['vegetation_only'])
        self.assertTrue(v.plant_summary()['ok'])
        with self.assertRaises(ValueError):  # no inferred recharge: explicit forcing required
            sim.step_day(None, None)
        # standalone defaults unchanged (no new counter key, synthetic store present)
        std = dwg.Simulation(make_case(layers=1, root_layer_fractions={'grass': [1.0]}), 'cpu', inputs=make_inputs([5.0], [5.0], [1.0]))
        self.assertFalse(std.vegetation_only)
        self.assertEqual(std.W.shape, (1, 1))
        self.assertEqual(std.last_ledger['external_reserve_bytes'], 0)
        self.assertNotIn('vegetation_only', std.counters())
        with self.assertRaises(ValueError):  # external-water commit is refused in standalone mode
            std.commit_daily_uptake(np.zeros(1))

    def test_gather_and_commit_validate_arrays_before_launch(self):
        case = make_case(layers=1, root_layer_fractions={'grass': [1.0]})
        sim, _ = build(case, make_inputs([5.0], [5.0], [1.0]))
        v = sim.vegetation
        out = np.zeros(1)
        good = np.ones((1, 1))
        v.gather_uptake_candidate(good, out)
        for bad in (np.ones((1, 1), np.float32), np.ones((2, 2))[:, :1], np.ones((1, 2)), [[1.0]], np.matrix([[1.0]])):
            with self.assertRaises((TypeError, ValueError), msg=repr(type(bad))):
                v.gather_uptake_candidate(bad, out)
        for bad in (np.zeros(1, np.float32), np.zeros(2), np.zeros(2)[::2], [0.0]):
            with self.assertRaises((TypeError, ValueError)):
                v.gather_uptake_candidate(good, bad)
            with self.assertRaises((TypeError, ValueError)):
                v.commit_daily_uptake(bad)
        self.assertEqual(v.day, 0)
        if sm.cp is not None and dwg.gpu_available():
            with self.assertRaises(TypeError):  # wrong namespace on a CPU simulation
                v.gather_uptake_candidate(sm.cp.ones((1, 1)), out)

    def test_phase_exclusion_before_work(self):
        case = make_case(layers=1, root_layer_fractions={'grass': [1.0]})
        sim, _ = build(case, make_inputs([5.0], [5.0], [1.0]))
        added, pet = forcing(sim)
        for ph in ('rain', 'wind', 'water_event', 'storm', 'unknown'):
            with self.assertRaises(CoupledPhaseError):
                sim.step_day(added, pet, phase=ph)
            with self.assertRaises(CoupledPhaseError):
                sim.step_days([(added, pet)], phase=ph)
        self.assertEqual((sim.days, sim.state.t_s), (0, 0.0))
        np.testing.assert_array_equal(host(sim.state.water_m), WET)

    def test_forcing_records_daily_only(self):
        case = make_case(layers=1, root_layer_fractions={'grass': [1.0]})
        sim, _ = build(case, make_inputs([5.0], [5.0], [1.0]))
        added, pet = forcing(sim)

        class Rec:
            def __init__(self, dt):
                self.added_water_m, self.potential_evaporation_m_s, self.dt_s = added, pet, dt

        self.assertEqual(sim.step_days([Rec(86400.0), (added, pet)]), 2)
        with self.assertRaises(ValueError):
            sim.step_days([Rec(3600.0)])
        self.assertEqual(sim.days, 2)


@unittest.skipUnless(NATIVE, 'maple_syrup native package / transpiration branch not importable')
class AdmissionTests(unittest.TestCase):
    def test_native_memory_checked_before_vegetation_allocation(self):
        case = make_case(layers=1, root_layer_fractions={'grass': [1.0]})
        avail = {'bytes': 0}
        col = make_column(case, available=avail)
        state = ss.column_state(col, np.full((1, 1), WET))
        with mock.patch.object(dwg, 'Simulation') as veg_cls:
            with self.assertRaises(sm.AdmissionRejected):
                CoupledSimulation(case, col, state, inputs=make_inputs([5.0], [5.0], [1.0]))
        veg_cls.assert_not_called()
        avail['bytes'] = 1 << 30
        sim = CoupledSimulation(case, col, state, inputs=make_inputs([5.0], [5.0], [1.0]))
        self.assertEqual(int(col.reserve_bytes), 0)  # column never mutated
        for bad in (-1, True, 1.5, -10 ** 12):
            with self.assertRaises((ValueError, TypeError), msg=repr(bad)):
                CoupledSimulation(case, col, state, inputs=make_inputs([5.0], [5.0], [1.0]), extra_reserve_bytes=bad)
        avail['bytes'] = 0  # a later daily step is refused with nothing published
        snap = (sim.days, host(sim.state.water_m).copy(), host(sim.vegetation.uptake_cum).copy())
        with self.assertRaises(ss.SubsurfaceMemoryError):
            sim.step_day(*forcing(sim))
        self.assertEqual(sim.days, snap[0])
        np.testing.assert_array_equal(host(sim.state.water_m), snap[1])
        np.testing.assert_array_equal(host(sim.vegetation.uptake_cum), snap[2])

    def test_case_max_bytes_scalar_guard_before_allocation(self):
        case = make_case(layers=1, root_layer_fractions={'grass': [1.0]}, max_bytes=1000)
        col = make_column(case)
        state = ss.column_state(col, np.full((1, 1), WET))
        with mock.patch.object(dwg, 'Simulation') as veg_cls:
            with self.assertRaises(sm.AdmissionRejected):
                CoupledSimulation(case, col, state, inputs=make_inputs([5.0], [5.0], [1.0]))
        veg_cls.assert_not_called()

    def test_memory_report_scopes(self):
        case = make_case(layers=1, root_layer_fractions={'grass': [1.0]})
        sim, _ = build(case, make_inputs([5.0], [5.0], [1.0]))
        m0 = sim.memory_report()
        self.assertIsNone(m0['last_native_combined_memory'])
        self.assertEqual(m0['retained_diagnostics_bytes'], 0)
        sim.step_day(*forcing(sim))
        m1 = sim.memory_report()
        self.assertGreater(m1['retained_diagnostics_bytes'], 0)
        self.assertEqual(m1['last_native_combined_memory']['extra_reserve_bytes'],
                         sim.vegetation.live_bytes() + sim.vegetation.graph_bytes + sim.buffer_bytes)
        self.assertEqual(sim.vegetation.external_reserve_bytes, sim._native_reserve_bytes())
        self.assertGreaterEqual(m1['peak_required_bytes'], m0['peak_required_bytes'])


@unittest.skipUnless(NATIVE, 'maple_syrup native package / transpiration branch not importable')
class RollbackAndLatchTests(unittest.TestCase):
    def _sim(self):
        case = make_case(layers=1, root_layer_fractions={'grass': [1.0]})
        return build(case, make_inputs([5.0], [5.0], [1.0]))[0]

    def _snapshot(self, sim):
        v = sim.vegetation
        return (sim.days, sim.state, float(sim.state.t_s), host(sim.state.water_m).copy(), host(v.uptake_cum).copy(),
                host(v.uptake_week).copy(), v.day, dict(sim.cum))

    def test_native_failure_rolls_back_whole_day(self):
        sim = self._sim()
        sim.step_day(*forcing(sim))
        snap = self._snapshot(sim)
        with mock.patch.object(sim._st, 'step_with_transpiration', side_effect=ss.SubsurfaceMemoryError('forced')):
            with self.assertRaises(ss.SubsurfaceMemoryError):
                sim.step_day(*forcing(sim))
        self.assertEqual(self._snapshot(sim)[:3], snap[:3])
        np.testing.assert_array_equal(self._snapshot(sim)[3], snap[3])
        np.testing.assert_array_equal(host(sim.vegetation.uptake_cum), snap[4])
        self.assertEqual(sim.vegetation.day, snap[6])
        self.assertEqual(sim.cum, snap[7])
        sim.step_day(*forcing(sim))  # recovers
        self.assertEqual(sim.days, 2)

    def test_gather_failure_rolls_back_whole_day(self):
        sim = self._sim()
        snap = self._snapshot(sim)
        real = type(sim.vegetation).gather_uptake_candidate

        def bad(self_, scale, out):
            real(self_, scale, out)
            out[...] = np.nan
            return out

        with mock.patch.object(type(sim.vegetation), 'gather_uptake_candidate', bad):
            with self.assertRaises(RuntimeError):
                sim.step_day(*forcing(sim))
        self.assertIs(sim.state, snap[1])
        np.testing.assert_array_equal(host(sim.vegetation.uptake_cum), snap[4])
        self.assertEqual((sim.days, sim.vegetation.day), (0, 0))

    def test_weekly_refusal_latches_and_retries(self):
        sim = self._sim()
        for _ in range(6):
            sim.step_day(*forcing(sim))
        v = sim.vegetation
        B, cells = host(v.B).copy(), host(v.graph['cells']).copy()
        forced = sm.AdmissionRejected('forced', 'forced refusal', {})
        with mock.patch.object(dwg.sm, 'build_root_graph_cpu', side_effect=forced):
            with self.assertRaises(sm.AdmissionRejected):
                sim.step_day(*forcing(sim))
        # the physical day was accepted; the weekly rebuild is pending/latched
        self.assertEqual((sim.days, v.day, v.weeks_done), (7, 7, 0))
        self.assertEqual(sim.state.t_s, 7 * 86400.0)
        self.assertTrue(sim.weekly_pending)
        self.assertGreater(host(v.uptake_week)[0], 0.0)
        np.testing.assert_array_equal(host(v.B), B)
        np.testing.assert_array_equal(host(v.graph['cells']), cells)
        with self.assertRaises(sm.AdmissionRejected):  # no further daily stepping until resolved
            sim.step_day(*forcing(sim))
        self.assertEqual((sim.days, sim.state.t_s), (7, 7 * 86400.0))
        sim.retry_weekly_update()
        self.assertFalse(sim.weekly_pending)
        self.assertEqual((v.weeks_done, v.demand_map_builds), (1, 2))
        sim.step_day(*forcing(sim))
        self.assertEqual(sim.days, 8)
        self.assertTrue(sim.summary()['ok'])


@unittest.skipUnless(NATIVE, 'maple_syrup native package / transpiration branch not importable')
class MaskGeometryTests(unittest.TestCase):
    def test_inactive_cells_unmapped_demand(self):
        case = make_case(nx=2, ny=2, dx=1.0, layers=1, root_layer_fractions={'grass': [1.0]})
        inp = make_inputs([1.0], [1.0], [0.6])  # root circle covers all four cells equally
        sim, col = build(case, inp, active=[True, True, True, False])
        R = host(sim.vegetation.R)
        self.assertTrue((R > 0).all())
        res = sim.step_day(*forcing(sim))
        req = host(sim.request_m)
        self.assertEqual(req[0, 3], 0.0)
        np.testing.assert_allclose(req[0, :3], R[0, :3] / 1.0, rtol=1e-12)  # no renormalization onto active cells
        Dp = float(host(sim.vegetation.Dp)[0])
        np.testing.assert_allclose(host(sim.vegetation.uptake_cum)[0], 0.75 * Dp, rtol=1e-9)
        np.testing.assert_allclose(res.transpiration_budget.actual_m3, 0.75 * Dp, rtol=1e-9)
        np.testing.assert_array_equal(host(sim.state.water_m)[:, 3], WET)
        s = sim.summary()
        self.assertTrue(s['ok'], s['checks'])
        np.testing.assert_allclose(s['demand']['potential'], Dp, rtol=1e-12)
        np.testing.assert_allclose(s['demand']['mapped_requested'], 0.75 * Dp, rtol=1e-9)
        np.testing.assert_allclose(s['demand']['unmapped_inactive'], 0.25 * Dp, rtol=1e-9)
        np.testing.assert_allclose(s['water']['initial_storage'], 3 * WET, rtol=1e-12)  # active cells only
        self.assertEqual(s['mask']['n_active'], 3)

    def test_hydrology_budget_split_run_matches_single_run(self):
        # nonzero diffusion, intake, evaporation and drainage: real hydrology plus coupled closure
        case = make_case(nx=4, ny=4, dx=0.5, grass_count=2, layers=2, max_edges=10000)
        inp = make_inputs([0.8, 1.3], [1.0, 1.1], [0.5, 0.4])
        water = np.vstack([np.full(16, 0.024), np.full(16, RES_M + 0.0004)])  # top near field capacity
        a, _ = build(case, inp, water, diffusivity=1e-9, active=[True] * 15 + [False])
        b, _ = build(case, inp, water, diffusivity=1e-9, active=[True] * 15 + [False])
        rng = np.random.default_rng(2)
        days = []
        for _ in range(10):
            add = rng.uniform(0, 0.004, (2, 16)) * np.array([[1.0], [0.0]])
            add[:, 15] = 0.0
            pet = rng.uniform(0, 2e-8, 16)
            pet[15] = 0.0
            days.append((add, pet))
        a.step_days(days)
        b.step_days(days[:6])
        b.step_days(days[6:])
        np.testing.assert_allclose(host(b.state.water_m), host(a.state.water_m), rtol=1e-12)
        for n in ('B', 'uptake_cum', 'uptake_week', 'canopy'):
            np.testing.assert_allclose(host(getattr(b.vegetation, n)), host(getattr(a.vegetation, n)), rtol=1e-12)
        sa = a.summary()
        self.assertEqual(sa['calendar'], b.summary()['calendar'])
        self.assertTrue(sa['ok'], sa['checks'])
        w = sa['water']
        self.assertGreater(w['added'], 0.0)
        self.assertGreater(w['evaporation'], 0.0)
        self.assertGreater(w['transpiration'], 0.0)
        self.assertGreater(w['drainage'] + w['surface_return'], 0.0)
        np.testing.assert_array_equal(host(a.state.water_m)[:, 15], water[:, 15])  # inactive preserved

    def test_nonzero_initial_time_and_fixed_profiles(self):
        case = make_case(layers=2, grass_count=1, shrub_count=1, grass_diameter=2.0, shrub_diameter=2.0)
        sim, _ = build(case, make_inputs([5.0, 5.0], [5.0, 5.0], [1.0, 1.0]), t_s=12345.0)
        tab = host(sim.vegetation.root_layer_fractions).copy()
        for _ in range(8):
            sim.step_day(*forcing(sim))
        self.assertEqual(sim.state.t_s, 12345.0 + 8 * 86400.0)
        np.testing.assert_array_equal(host(sim.vegetation.root_layer_fractions), tab)
        self.assertEqual(sim.roots['graph']['counts'].shape[0], 2)
        self.assertEqual(sim.summary()['calendar']['vegetation']['weeks_done'], 1)

    def test_mismatches_refused(self):
        case = make_case(layers=1, root_layer_fractions={'grass': [1.0]})
        inp = make_inputs([5.0], [5.0], [1.0])
        col = make_column(make_case(nx=2, layers=1))
        with self.assertRaises(ValueError):  # grid N
            CoupledSimulation(case, col, ss.uniform_state(col, [0.2]), inputs=inp)
        col = make_column(make_case(layers=2))
        with self.assertRaises(ValueError):  # layers
            CoupledSimulation(case, col, ss.uniform_state(col, [0.2, 0.2]), inputs=inp)
        col = make_column(make_case(dx=5.0, layers=1))
        with self.assertRaises(ValueError):  # cell area != dx^2
            CoupledSimulation(case, col, ss.uniform_state(col, [0.2]), inputs=inp)
        with self.assertRaises(ValueError):
            coupling.check_phase('rain')


@unittest.skipUnless(NATIVE and dwg.gpu_available(), 'CuPy device with CC >= 6.0 not available')
class GpuTests(unittest.TestCase):
    def test_cpu_gpu_parity(self):
        case = make_case(nx=6, ny=6, dx=0.5, grass_count=3, shrub_count=2, grass_diameter=0.6, shrub_diameter=1.2,
                         root_multiplier=2.0, layers=3, max_edges=100000)
        rng = np.random.default_rng(7)
        inp = make_inputs(rng.uniform(0.5, 2.5, 5), rng.uniform(0.5, 2.5, 5), [0.3, 0.3, 0.3, 0.6, 0.6], mult=2.0)
        water = rng.uniform(RES_M, 0.03, (3, 36))
        active = [True] * 34 + [False, True]
        days = []
        for _ in range(9):
            add = rng.uniform(0, 0.002, (3, 36)) * np.array([[1.0], [0.0], [0.0]])
            add[:, 34] = 0.0
            pet = rng.uniform(0, 2e-8, 36)
            pet[34] = 0.0
            days.append((add, pet))
        g, _ = build(case, inp, water, backend='cuda', active=active, diffusivity=1e-9)  # errors fail, never skip
        c, _ = build(case, inp, water, backend='numba', active=active, diffusivity=1e-9)
        cp = sm.cp
        c.step_days(days)
        g.step_days([(cp.asarray(a), cp.asarray(p)) for a, p in days])
        np.testing.assert_allclose(host(g.state.water_m), host(c.state.water_m), rtol=1e-9, atol=1e-15)
        for n in ('B', 'uptake_cum', 'uptake_week', 'canopy', 'root'):
            np.testing.assert_allclose(host(getattr(g.vegetation, n)), host(getattr(c.vegetation, n)), rtol=1e-8, atol=1e-12)
        sg, sc = g.summary(), c.summary()
        self.assertTrue(sg['ok'], sg['checks'])
        self.assertEqual(sg['calendar']['vegetation']['weeks_done'], sc['calendar']['vegetation']['weeks_done'])
        self.assertEqual(sg['memory']['where'], 'device')
        self.assertIs(type(g.state.water_m), cp.ndarray)

    def test_gpu_gather_rejects_before_launch(self):
        case = make_case(layers=1, root_layer_fractions={'grass': [1.0]})
        sim, _ = build(case, make_inputs([5.0], [5.0], [1.0]), backend='cuda')
        v, cp = sim.vegetation, sm.cp
        out = cp.zeros(1)
        v.gather_uptake_candidate(cp.ones((1, 1)), out)
        with mock.patch.dict(dwg._GPU_CACHE, {}, clear=False):
            launched = dwg._gpu_kernels()['gather_candidate']
            spy = mock.MagicMock(wraps=launched)
            with mock.patch.dict(dwg._GPU_CACHE, {'gather_candidate': spy}):
                for bad in (cp.ones((1, 1), cp.float32), np.ones((1, 1)), cp.ones((2, 2))[:, :1], cp.ones((1, 2))):
                    with self.assertRaises((TypeError, ValueError)):
                        v.gather_uptake_candidate(bad, out)
                with self.assertRaises((TypeError, ValueError)):
                    v.gather_uptake_candidate(cp.ones((1, 1)), np.zeros(1))
            spy.assert_not_called()


if __name__ == '__main__':
    unittest.main()
