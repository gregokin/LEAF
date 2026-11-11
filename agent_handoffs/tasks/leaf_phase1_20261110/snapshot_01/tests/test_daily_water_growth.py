'''Small independent unittest checks for benchmarks/daily_water_growth.py (fixture rules, not ecology).'''
import json
import math
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from benchmarks import daily_water_growth as dwg
from benchmarks import spatial_mapping as sm

F = dwg.FIXTURE
CELL = 0.5 * 0.5  # dx^2 of the default test grid
CAP = F['storage_m'] * CELL
RECH = F['recharge_m'] * CELL


def make_case(**over):
    case = {'name': 'test', 'nx': 8, 'ny': 8, 'dx': 0.5, 'grass_count': 1, 'shrub_count': 0,
            'grass_diameter': 2.0, 'shrub_diameter': 4.0, 'root_multiplier': 1.0, 'pattern': 'uniform',
            'seed': 1, 'layers': 1, 'max_edges': 100000, 'max_bytes': 50_000_000}
    case.update(over)
    return case


def make_inputs(xs, ys, radii, mult=1.0, pid=None):
    cr = np.asarray(radii, dtype=np.float64)
    inp = {'x': np.asarray(xs, np.float64), 'y': np.asarray(ys, np.float64), 'canopy_radius': cr,
           'root_radius': mult * cr, 'ids': np.arange(len(cr), dtype=np.int64)}
    if pid is not None:
        inp['root_profile_id'] = pid  # raw (unconverted) so that invalid values reach the validator
    return inp


DP1 = F['demand_m_per_day'] * math.pi  # daily potential demand of a radius-1 plant
FRAC3 = {'grass': [0.7, 0.2, 0.1], 'shrub': [0.15, 0.25, 0.6], 'deep': [0.1, 0.3, 0.6]}


def assert_audit_with_external(tc, s, ext):
    '''Audit after a TEST-ONLY external overwrite of W: the water residual must equal exactly the tracked
    external net change (production accounting unchanged); every other invariant must hold.'''
    others = {k: v for k, v in s['checks'].items() if k != 'water_budget'}
    tc.assertTrue(all(others.values()), others)
    np.testing.assert_allclose(s['water']['residual'], ext, rtol=1e-9, atol=1e-12)
    tc.assertEqual(s['checks']['water_budget'], abs(ext) <= s['water']['tolerance'])


def run(case, inputs, days, backend='cpu'):
    return dwg.run_case(case, backend, days=days, inputs=inputs, report_every=days)


def uncovered_mask(res):
    cov = np.unique(res['arrays']['graph']['cells'])
    mask = np.ones(res['arrays']['W'].shape[1], bool)
    mask[cov] = False
    return cov, mask


class SoilWaterTests(unittest.TestCase):
    def test_single_plant_depletes_without_overdraw(self):
        # 1x1 grid, one layer, 3 days (< one week, no growth): supply 0.5*CAP + RECH = 0.02125 m3,
        # potential 3*0.003*pi > supply -> store exhausted, uptake = removal = supply, no overdraw.
        res = run(make_case(nx=1, ny=1), make_inputs([0.25], [0.25], [1.0]), 3)
        self.assertEqual(res['status'], 'ok', res.get('error'))
        self.assertTrue(res['audit']['ok'], res['audit']['checks'])
        a = res['arrays']
        supply = 0.5 * CAP + RECH
        self.assertEqual(a['W'].shape, (1, 1))
        self.assertEqual(float(a['W'][0, 0]), 0.0)
        np.testing.assert_allclose(a['uptake_cum'][0], supply, rtol=1e-12)
        np.testing.assert_allclose(a['removed_cum'][0, 0], supply, rtol=1e-12)
        self.assertLess(a['uptake_cum'][0], 3 * a['Dp'][0])  # supply-limited
        self.assertEqual(res['counters']['weeks_done'], 0)

    def test_two_plants_share_finite_store(self):
        inp = make_inputs([1.5, 2.5], [2.0, 2.0], [1.0, 1.0])
        res = run(make_case(grass_count=2), inp, 40)
        self.assertEqual(res['status'], 'ok', res.get('error'))
        a, w = res['arrays'], res['audit']['water']
        self.assertTrue((a['uptake_cum'] > 0).all())
        np.testing.assert_allclose(a['uptake_cum'][0], a['uptake_cum'][1], rtol=1e-9)  # mirror symmetry
        self.assertLessEqual(w['plant_uptake'], w['initial'] + w['input'] + 1e-12)
        self.assertLess(abs(w['flux_residual']), w['flux_tolerance'])
        self.assertTrue(res['audit']['ok'], res['audit']['checks'])

    def test_dry_cells_give_no_phantom_uptake(self):
        sim = dwg.Simulation(make_case(), 'cpu', inputs=make_inputs([2.0], [2.0], [1.0]))
        sim.step_days(1)  # day 0 includes recharge
        sim.W.fill(0.0)
        u0, r0 = sim.uptake_cum.copy(), sim.removed_cum.sum()
        sim.step_days(1)  # day 1: no recharge, empty store
        np.testing.assert_array_equal(sim.uptake_cum, u0)
        self.assertEqual(sim.removed_cum.sum(), r0)
        self.assertEqual(np.abs(sim.W).max(), 0.0)
        self.assertEqual(sim.day, 2)

    def test_recharge_overflow_budget(self):
        res = run(make_case(), make_inputs([0.5], [0.5], [0.2]), 99)  # 8 recharge events (days 0..98)
        self.assertEqual(res['status'], 'ok', res.get('error'))
        a, w = res['arrays'], res['audit']['water']
        _, mask = uncovered_mask(res)
        self.assertTrue(mask.any())
        self.assertEqual(w['recharge_events'], 8)
        np.testing.assert_allclose(a['W'][0, mask], CAP, rtol=1e-12)
        np.testing.assert_allclose(a['overflow_cum'][0, mask], 0.5 * CAP + 8 * RECH - CAP, rtol=1e-9)
        self.assertLessEqual(a['W'].max(), CAP * (1 + 1e-12))
        self.assertGreater(w['overflow'], 0.0)
        self.assertLess(abs(w['residual']), 1e-12)


class GrowthAndCalendarTests(unittest.TestCase):
    def test_weekly_growth_derives_radii_and_budgets(self):
        case = make_case(nx=16, ny=16, grass_count=1, shrub_count=1, root_multiplier=3.0)
        inp = make_inputs([2.0, 5.5], [2.0, 5.5], [1.0, 2.0], mult=3.0)
        res = run(case, inp, 7)
        self.assertEqual(res['status'], 'ok', res.get('error'))
        a = res['arrays']
        np.testing.assert_array_equal(a['is_shrub'], [False, True])
        wue = np.where(a['is_shrub'], F['wue_shrub_kg_per_kg'], F['wue_grass_kg_per_kg'])
        B0 = math.pi * np.array([1.0, 4.0])
        np.testing.assert_allclose(a['gain_cum'], F['water_density_kg_m3'] * wue * a['uptake_cum'], rtol=1e-12)
        np.testing.assert_allclose(a['loss_cum'], B0 * (1 - math.exp(-F['turnover_rate_per_day'] * 7)), rtol=1e-12)
        np.testing.assert_allclose(a['B'], B0 - a['loss_cum'] + a['gain_cum'], rtol=1e-12)
        np.testing.assert_allclose(a['canopy'], np.sqrt(a['B'] / math.pi), rtol=1e-12)
        np.testing.assert_allclose(a['root'], 3.0 * a['canopy'], rtol=1e-12)
        np.testing.assert_allclose(a['Dp'], F['demand_m_per_day'] * math.pi * a['canopy'] ** 2, rtol=1e-12)
        self.assertTrue((a['uptake_week'] == 0.0).all())
        self.assertEqual(res['counters']['maps_built'], 2)
        self.assertTrue(res['audit']['ok'], res['audit']['checks'])

    def test_counters_and_pending_uptake(self):
        for days, weeks, pend in ((7, 1, 0), (8, 1, 1), (14, 2, 0)):
            res = run(make_case(), make_inputs([2.0], [2.0], [1.0]), days)
            self.assertEqual(res['status'], 'ok', res.get('error'))
            c = res['counters']
            self.assertEqual((c['weeks_done'], c['pending_days'], c['maps_built'], c['demand_map_builds']),
                             (weeks, pend, weeks + 1, weeks + 1), days)
            pending = res['arrays']['uptake_week'].sum()
            self.assertEqual(pending > 0.0, pend > 0)  # pending days retained, never flushed or dropped
            np.testing.assert_allclose(res['audit']['water']['pending_week_uptake'], pending)
        cal = dwg.calendar(10950)
        self.assertEqual((cal['weekly_updates'], cal['pending_days'], cal['maps_built']), (1564, 2, 1565))

    def test_demand_map_cached_within_week(self):
        sim = dwg.Simulation(make_case(), 'cpu', inputs=make_inputs([2.0], [2.0], [1.0]))
        sim.step_days(3)
        R_ref, builds = sim.R.copy(), sim.demand_map_builds
        self.assertEqual(builds, 1)
        sim.step_days(3)
        self.assertEqual(sim.demand_map_builds, 1)
        np.testing.assert_array_equal(sim.R, R_ref)
        sim.step_days(1)  # day 7 -> growth + rebuild
        self.assertEqual(sim.demand_map_builds, 2)
        self.assertFalse(np.array_equal(sim.R, R_ref))

    def test_split_run_matches_single_run(self):
        inp = make_inputs([2.0], [2.0], [1.0])
        s1 = dwg.Simulation(make_case(), 'cpu', inputs=inp)
        s1.step_days(16)
        s2 = dwg.Simulation(make_case(), 'cpu', inputs=inp)
        s2.step_days(10)
        s2.step_days(6)
        for n in ('W', 'B', 'uptake_cum', 'uptake_week', 'overflow_cum'):
            np.testing.assert_allclose(getattr(s2, n), getattr(s1, n), rtol=1e-12)
        self.assertEqual(s1.counters(), s2.counters())


class AdmissionTests(unittest.TestCase):
    def test_memory_refusal_before_generation(self):
        case = make_case(nx=100000, ny=100000, max_bytes=10 ** 6)
        res = dwg.run_case(case, 'cpu', days=1)
        self.assertEqual(res['status'], 'rejected', res.get('error'))
        self.assertNotIn('arrays', res)
        with self.assertRaises(sm.AdmissionRejected):
            dwg.Simulation(case, 'cpu')

    def test_refusal_does_not_generate_or_transfer(self):
        case = make_case(nx=100000, ny=100000, max_bytes=10 ** 6)
        with mock.patch.object(dwg.sm, 'generate_inputs') as gen, mock.patch.object(dwg.sm, 'to_device') as tod:
            with self.assertRaises(sm.AdmissionRejected):
                dwg.Simulation(case, 'cpu')
        gen.assert_not_called()
        tod.assert_not_called()

    def test_initial_bound_counts_each_trait_separately(self):
        N, dx = 400 * 400, 0.25
        for mult in (3.0, 5.0):
            case = make_case(nx=400, ny=400, dx=dx, grass_count=547493, shrub_count=95, grass_diameter=0.1,
                             shrub_diameter=2.0, root_multiplier=mult, layers=3, max_edges=10_000_000,
                             max_bytes=1_500_000_000)
            case = dwg.validate_case(case)
            edges = dwg.initial_edge_bound(case, dx, N)
            self.assertLessEqual(edges, case['max_edges'])
            led = dwg.admission_ledger(547588, N, 3, 0, edges)
            self.assertLessEqual(led['total_bytes'], case['max_bytes'], led)
            self.assertGreater(led['scratch_allowance_bytes'], 0)
        # per-plant bound for supplied inputs equals the sum of bbox cells
        inp = make_inputs([1.0, 2.0], [1.0, 2.0], [0.2, 1.0])
        self.assertEqual(dwg.initial_edge_bound(make_case(), 0.5, 64, inp),
                         dwg._bbox_cells(0.2, 0.5, 64) + dwg._bbox_cells(1.0, 0.5, 64))

    def test_rejected_weekly_update_preserves_state_and_latches(self):
        inp = make_inputs([2.0], [2.0], [1.0])
        sim = dwg.Simulation(make_case(), 'cpu', inputs=inp)
        sim.step_days(5)
        snap = {n: getattr(sim, n).copy() for n in ('B', 'canopy', 'root', 'Dp', 'gain_cum', 'loss_cum', 'R')}
        cells = sim.graph['cells'].copy()
        u5 = sim.uptake_cum.copy()
        forced = sm.AdmissionRejected('forced', 'forced refusal', {})
        with mock.patch.object(dwg.sm, 'build_root_graph_cpu', side_effect=forced):
            with self.assertRaises(sm.AdmissionRejected):
                sim.step_days(2)
        self.assertEqual(sim.day, 7)
        self.assertEqual(sim.weeks_done, 0)
        self.assertIsNotNone(sim.failed)
        for n, v in snap.items():
            np.testing.assert_array_equal(getattr(sim, n), v)
        np.testing.assert_array_equal(sim.graph['cells'], cells)
        self.assertEqual(sim.demand_map_builds, 1)
        self.assertGreater(sim.uptake_week[0], 0.0)  # pending week retained, not flushed
        self.assertGreater(sim.uptake_cum[0], u5[0])
        u7 = sim.uptake_cum.copy()
        with self.assertRaises(sm.AdmissionRejected):
            sim.step_days(1)  # latched: no further day advances
        self.assertEqual(sim.day, 7)
        np.testing.assert_array_equal(sim.uptake_cum, u7)
        sim.retry_weekly_update()  # same pending boundary, builder unpatched
        self.assertEqual((sim.weeks_done, sim.maps_built, sim.demand_map_builds), (1, 2, 2))
        self.assertEqual(sim.uptake_week[0], 0.0)
        self.assertTrue(sim.summary()['ok'])

    def test_memory_hook_entered_before_initialization(self):
        events = []

        class Spy:
            active = False

            def __enter__(self):
                Spy.active = True
                events.append('enter')
                return self

            def __exit__(self, *exc):
                Spy.active = False
                events.append('exit')
                return False

        real_init = dwg.Simulation.__init__

        def spy_init(self, *a, **k):
            events.append('init_active' if Spy.active else 'init_inactive')
            real_init(self, *a, **k)

        with mock.patch.object(dwg.Simulation, '__init__', spy_init):
            res = dwg.run_case(make_case(), 'cpu', days=2, inputs=make_inputs([2.0], [2.0], [1.0]), memory_hook=Spy())
        self.assertEqual(res['status'], 'ok', res.get('error'))
        self.assertEqual(events, ['enter', 'init_active', 'exit'])

    def test_invalid_scalars_rejected(self):
        with self.assertRaises(ValueError):
            dwg.run_case(make_case(nx=0), 'cpu', days=1)
        with self.assertRaises(ValueError):
            dwg.run_case(make_case(), 'cpu', days=0)


class RootProfileTests(unittest.TestCase):
    def _one_cell_case(self, **over):
        # 1x1 domain with dx=10: capacity 0.15/L*100 m3 per layer cell, far above any demand here
        return make_case(nx=1, ny=1, dx=10.0, **over)

    def test_default_profiles_scatter_per_layer_demand(self):
        # grass + shrub of equal radius at the same position (equal diameters -> ties: index 0 grass, 1 shrub),
        # 3 days (< 1 week, no growth), well watered: per-layer R = weighted sum, uptake = potential demand.
        case = self._one_cell_case(grass_count=1, shrub_count=1, grass_diameter=2.0, shrub_diameter=2.0, layers=3)
        res = run(case, make_inputs([5.0, 5.0], [5.0, 5.0], [1.0, 1.0]), 3)
        self.assertEqual(res['status'], 'ok', res.get('error'))
        a = res['arrays']
        np.testing.assert_array_equal(a['is_shrub'], [False, True])
        np.testing.assert_array_equal(a['root_profile_ids'], [0, 1])
        self.assertEqual(a['root_profile_ids'].dtype, np.int32)
        self.assertEqual(res['root_profiles']['names'], ['grass', 'shrub'])
        fg, fs = [0.70, 0.20, 0.10], [0.15, 0.25, 0.60]
        np.testing.assert_allclose(a['root_layer_fractions'], [fg, fs], rtol=0, atol=1e-15)
        np.testing.assert_allclose(a['R'][:, 0], [DP1 * (fg[l] + fs[l]) for l in range(3)], rtol=1e-12)
        self.assertTrue((a['R'] > 0).all())  # every layer receives some roots
        np.testing.assert_allclose(a['R'].sum(), 2 * DP1, rtol=1e-12)
        np.testing.assert_allclose(a['uptake_cum'], [3 * DP1, 3 * DP1], rtol=1e-12)
        np.testing.assert_allclose(a['removed_cum'].sum(), 6 * DP1, rtol=1e-12)
        np.testing.assert_allclose(a['removed_cum'][:, 0], [3 * DP1 * (fg[l] + fs[l]) for l in range(3)], rtol=1e-12)
        self.assertTrue(res['audit']['ok'], res['audit']['checks'])
        self.assertEqual(res['counters']['weeks_done'], 0)

    def test_opposite_profiles_dry_layer_disadvantage(self):
        frac = {'top': [0.8, 0.2], 'bottom': [0.2, 0.8]}
        for dry, expect in ((0, (0.2, 0.8)), (1, (0.8, 0.2))):
            case = self._one_cell_case(grass_count=2, layers=2, root_layer_fractions=frac)
            sim = dwg.Simulation(case, 'cpu', inputs=make_inputs([5.0, 5.0], [5.0, 5.0], [1.0, 1.0], pid=[0, 1]))
            np.testing.assert_array_equal(sim.root_profile_ids, [0, 1])
            sim.step_days(1)  # day 0 (recharge), well watered: both take full demand
            np.testing.assert_allclose(sim.uptake_cum, [DP1, DP1], rtol=1e-12)
            X = 0.5 * DP1  # remaining water in the wet layer, below its requested R = DP1*(0.8+0.2)
            w_before = float(sim.W.sum())
            sim.W.fill(0.0)
            sim.W[1 - dry, 0] = X
            ext = float(sim.W.sum()) - w_before  # test-only external water change, tracked explicitly
            u, r = sim.uptake_cum.copy(), sim.removed_cum.copy()
            sim.step_days(1)  # day 1: no recharge
            np.testing.assert_allclose(sim.R[:, 0], [DP1, DP1], rtol=1e-12)
            np.testing.assert_allclose(sim.uptake_cum - u, [expect[0] * X, expect[1] * X], rtol=1e-12)
            np.testing.assert_allclose((sim.removed_cum - r).sum(), X, rtol=1e-12)  # soil removal = plant total
            self.assertEqual(float(np.abs(sim.W).max()), 0.0)  # no phantom water, no overdraw
            assert_audit_with_external(self, sim.summary(), ext)

    def test_all_dry_finite_supply_growth_and_fixed_third_profile(self):
        frac = {'grass': [0.5, 0.3, 0.2], 'shrub': [0.2, 0.3, 0.5], 'deep': [0.1, 0.1, 0.8]}
        tab = np.array([frac['grass'], frac['shrub'], frac['deep']])
        ids = [2, 0, 1]
        case = self._one_cell_case(grass_count=3, layers=3, root_layer_fractions=frac)
        sim = dwg.Simulation(case, 'cpu', inputs=make_inputs([5.0] * 3, [5.0] * 3, [1.0] * 3, pid=ids))
        np.testing.assert_array_equal(sim.root_profile_ids, ids)
        np.testing.assert_allclose(sim.root_layer_fractions, tab, atol=1e-15)
        np.testing.assert_allclose(sim.R[:, 0], DP1 * tab[ids].sum(axis=0), rtol=1e-12)
        sim.step_days(1)
        ext = -float(sim.W.sum())
        sim.W.fill(0.0)  # all dry (test-only external change, tracked in ext)
        u1 = sim.uptake_cum.copy()
        sim.step_days(1)
        np.testing.assert_array_equal(sim.uptake_cum, u1)  # no phantom uptake
        X = 0.3 * DP1  # finite supply in the bottom layer only; requested there = DP1*(0.8+0.2+0.5)
        sim.W[2, 0] = X
        ext += X
        sim.step_days(1)
        shares = tab[ids, 2] / tab[ids, 2].sum()
        np.testing.assert_allclose(sim.uptake_cum - u1, shares * X, rtol=1e-12)
        np.testing.assert_allclose(sim.uptake_cum.sum() - u1.sum(), X, rtol=1e-12)
        sim.step_days(4)  # day 7: weekly growth from actual uptake, geometry rebuilt, profiles fixed
        self.assertEqual((sim.weeks_done, sim.demand_map_builds), (1, 2))
        wue = F['wue_grass_kg_per_kg']  # all three plants are grass-sized; WUE fixture unchanged by profiles
        np.testing.assert_allclose(sim.gain_cum, F['water_density_kg_m3'] * wue * sim.uptake_cum, rtol=1e-12)
        np.testing.assert_array_equal(sim.root_profile_ids, ids)
        np.testing.assert_allclose(sim.root_layer_fractions, tab, atol=1e-15)
        np.testing.assert_allclose(sim.root_layer_fractions.sum(axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(sim.R[:, 0], (sim.Dp[:, None] * tab[ids]).sum(axis=0), rtol=1e-12)
        s = sim.summary()
        assert_audit_with_external(self, s, ext)
        self.assertEqual(s['root_profiles']['names'], ['grass', 'shrub', 'deep'])
        # fresh, unmodified 7-day run of the same case: full production budget/growth checks must pass
        res = run(case, make_inputs([5.0] * 3, [5.0] * 3, [1.0] * 3, pid=ids), 7)
        self.assertEqual(res['status'], 'ok', res.get('error'))
        self.assertTrue(res['audit']['ok'], res['audit']['checks'])
        a = res['arrays']
        np.testing.assert_array_equal(a['root_profile_ids'], ids)
        np.testing.assert_allclose(a['gain_cum'], F['water_density_kg_m3'] * wue * a['uptake_cum'], rtol=1e-12)
        self.assertEqual(res['root_profiles']['names'], ['grass', 'shrub', 'deep'])

    def test_explicit_uniform_profile_matches_equal_layer_reference(self):
        # 1x1 grid (dx 0.5), 3 equal layers, 3 days: supply per layer (0.5*CAP+RECH)/3 < 3*Dp/3 so every
        # layer is exhausted equally; uptake = removal = total supply (original equal-layer behaviour).
        u3 = [1.0 / 3.0] * 3
        case = make_case(nx=1, ny=1, layers=3, root_layer_fractions={'grass': u3, 'shrub': u3})
        res = run(case, make_inputs([0.25], [0.25], [1.0]), 3)
        self.assertEqual(res['status'], 'ok', res.get('error'))
        a = res['arrays']
        supply = 0.5 * CAP + RECH
        np.testing.assert_allclose(a['R'][:, 0], [DP1 / 3.0] * 3, rtol=1e-12)
        np.testing.assert_array_equal(a['W'], np.zeros((3, 1)))
        np.testing.assert_allclose(a['uptake_cum'][0], supply, rtol=1e-12)
        np.testing.assert_allclose(a['removed_cum'][:, 0], [supply / 3.0] * 3, rtol=1e-12)
        self.assertTrue(res['audit']['ok'], res['audit']['checks'])
        # two plants, uniform 2-layer profiles, water only in the top layer: equal shares X/2 each
        u2 = [0.5, 0.5]
        case = self._one_cell_case(grass_count=2, layers=2, root_layer_fractions={'a': u2, 'b': u2})
        sim = dwg.Simulation(case, 'cpu', inputs=make_inputs([5.0, 5.0], [5.0, 5.0], [1.0, 1.0], pid=[0, 1]))
        sim.step_days(1)
        X = 0.5 * DP1
        sim.W.fill(0.0)
        sim.W[0, 0] = X
        u = sim.uptake_cum.copy()
        sim.step_days(1)
        np.testing.assert_allclose(sim.uptake_cum - u, [0.5 * X, 0.5 * X], rtol=1e-12)

    def test_invalid_profiles_and_ids_refuse_before_generation(self):
        ok3 = [0.15, 0.25, 0.60]
        bad_rows = ({'grass': [0.5, 0.5, 0.0], 'shrub': ok3}, {'grass': [0.5, 0.6, -0.1], 'shrub': ok3},
                    {'grass': [0.5, float('nan'), 0.2], 'shrub': ok3}, {'grass': [0.5, float('inf'), 0.2], 'shrub': ok3},
                    {'grass': [True, 0.0, 0.0], 'shrub': ok3}, {'grass': [0.5, 0.5], 'shrub': ok3},
                    {'grass': [0.5, 0.5, 0.1], 'shrub': ok3}, {'grass': ['0.7', 0.2, 0.1], 'shrub': ok3},
                    {'shrub': ok3}, [[0.7, 0.2, 0.1]], {})
        for rlf in bad_rows:
            with mock.patch.object(dwg.sm, 'generate_inputs') as gen, mock.patch.object(dwg.sm, 'to_device') as tod:
                with self.assertRaises(ValueError, msg=repr(rlf)):
                    dwg.Simulation(make_case(layers=3, root_layer_fractions=rlf), 'cpu')
            gen.assert_not_called()
            tod.assert_not_called()
        case = make_case(grass_count=2, layers=3, root_layer_fractions={'grass': [0.7, 0.2, 0.1], 'shrub': ok3})
        for pid in ([True, False], [0.0, 1.0], np.array([0.0, 1.0]), np.array([True, False]), [0, 2], [-1, 0],
                    [0], [[0], [1]], np.zeros((2, 1), np.int64), 'ab'):
            with mock.patch.object(dwg.sm, 'to_device') as tod, mock.patch.object(dwg.sm, 'validate_inputs') as val:
                with self.assertRaises(ValueError, msg=repr(pid)):
                    dwg.Simulation(case, 'cpu', inputs=make_inputs([2.0, 3.0], [2.0, 3.0], [1.0, 1.0], pid=pid))
            tod.assert_not_called()
            val.assert_not_called()
        with self.assertRaises(ValueError):  # run_case refuses invalid rows before any allocation
            dwg.run_case(make_case(layers=2, root_layer_fractions={'grass': [1.0, 0.0], 'shrub': [0.5, 0.5]}), 'cpu', days=1)

    def test_generated_population_runs_with_configured_profiles(self):
        case = make_case(nx=16, ny=16, grass_count=3, shrub_count=2, grass_diameter=0.6, shrub_diameter=1.2,
                         root_multiplier=2.0, layers=3, root_layer_fractions=FRAC3, max_edges=200000)
        res = dwg.run_case(case, 'cpu', days=8)  # generated inputs: one weekly rebuild + 1 pending day
        self.assertEqual(res['status'], 'ok', res.get('error'))
        self.assertTrue(res['audit']['ok'], res['audit']['checks'])
        a = res['arrays']
        self.assertEqual(res['root_profiles']['names'], ['grass', 'shrub', 'deep'])
        np.testing.assert_array_equal(a['root_profile_ids'], a['is_shrub'].astype(np.int32))
        np.testing.assert_allclose(a['R'].sum(), a['Dp'].sum(), rtol=1e-8)
        self.assertEqual(res['counters']['weeks_done'], 1)
        self.assertIsNone(res['profile_transfer_record'])

    def test_cli_json_case_with_profiles(self):
        case = make_case(nx=12, ny=12, grass_count=2, shrub_count=1, grass_diameter=0.6, shrub_diameter=1.2,
                         root_multiplier=2.0, layers=2, root_layer_fractions={'grass': [0.8, 0.2], 'shrub': [0.25, 0.75]})
        with tempfile.TemporaryDirectory() as td:
            cin, cout = os.path.join(td, 'cases.json'), os.path.join(td, 'out.json')
            with open(cin, 'w') as fh:
                json.dump([case], fh)
            rc = dwg.main(['--cases-json', cin, '--output', cout, '--backend', 'cpu', '--days', '8',
                           '--report-every', '8'])
            self.assertEqual(rc, 0)
            with open(cout) as fh:
                out = json.load(fh)
        self.assertEqual(out['status'], 'ok')
        r = out['results'][0]
        self.assertEqual(r['status'], 'ok', r.get('error'))
        self.assertEqual(r['root_profiles']['names'], ['grass', 'shrub'])
        self.assertEqual(r['root_profiles']['table'], [[0.8, 0.2], [0.25, 0.75]])
        self.assertNotIn('arrays', r)

    def test_admission_preflight_precedes_profile_materialization(self):
        calls = []

        def rec(name, fn):
            def wrapped(*a, **k):
                calls.append(name)
                return fn(*a, **k)
            return wrapped

        patches = [mock.patch.object(dwg, 'admission_ledger', rec('ledger', dwg.admission_ledger)),
                   mock.patch.object(dwg, 'build_root_profile_table', rec('table', dwg.build_root_profile_table)),
                   mock.patch.object(dwg, 'materialize_profile_ids', rec('ids', dwg.materialize_profile_ids)),
                   mock.patch.object(dwg, 'initial_edge_bound', rec('edges', dwg.initial_edge_bound))]
        case = make_case(nx=10, ny=6, grass_count=3, layers=3, root_layer_fractions=FRAC3)
        inp = make_inputs([1.5, 2.5, 3.5], [1.5, 1.5, 1.5], [0.3] * 3, pid=[0, 1, 2])
        with patches[0], patches[1], patches[2], patches[3]:
            sim = dwg.Simulation(case, 'cpu', inputs=inp)
        self.assertEqual(calls[0], 'ledger')  # scalar zero-edge guard first, then edge bound, then materialization
        self.assertLess(calls.index('ledger'), calls.index('edges'))
        self.assertLess(calls.index('edges'), calls.index('table'))
        self.assertLess(calls.index('ledger'), calls.index('ids'))
        self.assertEqual(sim.last_ledger['n_profiles'], 3)
        # hopeless supplied-input case: state alone exceeds max_bytes -> no edge-bound temporaries at all
        calls.clear()
        hopeless = make_case(nx=1000, ny=1000, max_bytes=10 ** 6, layers=3, root_layer_fractions=FRAC3)
        with patches[0], patches[1], patches[2], patches[3], mock.patch.object(dwg.sm, 'generate_inputs') as gen, \
                mock.patch.object(dwg.sm, 'to_device') as tod:
            with self.assertRaises(sm.AdmissionRejected) as cm:
                dwg.Simulation(hopeless, 'cpu', inputs=make_inputs([50.0, 60.0], [50.0, 60.0], [1.0, 1.0], pid=[0, 2]))
        self.assertEqual(calls, ['ledger'])
        self.assertEqual(cm.exception.estimate['stage'] if hasattr(cm.exception, 'estimate') else 'init_scalar_bound',
                         'init_scalar_bound')
        gen.assert_not_called()
        tod.assert_not_called()
        # refusal before any materialization/generation/transfer (no large table or ID vector is built)
        calls.clear()
        big = make_case(nx=100000, ny=100000, max_bytes=10 ** 6, layers=3, root_layer_fractions=FRAC3)
        with patches[0], patches[1], patches[2], mock.patch.object(dwg.sm, 'generate_inputs') as gen, \
                mock.patch.object(dwg.sm, 'to_device') as tod:
            with self.assertRaises(sm.AdmissionRejected):
                dwg.Simulation(big, 'cpu')
            with self.assertRaises(sm.AdmissionRejected):
                dwg.Simulation(big, 'cpu', inputs=make_inputs([50.0, 60.0], [50.0, 60.0], [1.0, 1.0], pid=[0, 2]))
        self.assertEqual(set(calls), {'ledger'})
        gen.assert_not_called()
        tod.assert_not_called()
        res = dwg.run_case(big, 'cpu', days=1)
        self.assertEqual(res['status'], 'rejected', res.get('error'))

    def test_default_class_rows_required_only_when_used(self):
        grass_only = make_case(nx=12, ny=12, grass_count=2, shrub_count=0, grass_diameter=0.6, shrub_diameter=1.2,
                               layers=2, root_layer_fractions={'grass': [0.8, 0.2]})
        res = dwg.run_case(grass_only, 'cpu', days=2)
        self.assertEqual(res['status'], 'ok', res.get('error'))
        self.assertEqual(res['root_profiles']['names'], ['grass'])
        np.testing.assert_array_equal(res['arrays']['root_profile_ids'], [0, 0])
        shrub_only = make_case(nx=12, ny=12, grass_count=0, shrub_count=2, grass_diameter=0.6, shrub_diameter=1.2,
                               layers=2, root_layer_fractions={'shrub': [0.3, 0.7]})
        res = dwg.run_case(shrub_only, 'cpu', days=2)
        self.assertEqual(res['status'], 'ok', res.get('error'))
        self.assertEqual(res['root_profiles']['names'], ['shrub'])
        np.testing.assert_array_equal(res['arrays']['root_profile_ids'], [0, 0])
        np.testing.assert_array_equal(res['arrays']['is_shrub'], [True, True])
        with mock.patch.object(dwg.sm, 'generate_inputs') as gen:
            with self.assertRaises(ValueError):  # grass plants generated but no grass row
                dwg.Simulation(make_case(grass_count=2, layers=2, root_layer_fractions={'shrub': [0.3, 0.7]}), 'cpu')
        gen.assert_not_called()
        # supplied inputs without explicit IDs: grass-sized plants need only the grass row
        inp = make_inputs([2.0, 3.0], [2.0, 2.0], [1.0, 1.0])
        sim = dwg.Simulation(make_case(grass_count=2, layers=2, root_layer_fractions={'grass': [0.8, 0.2]}), 'cpu',
                             inputs=inp)
        np.testing.assert_array_equal(sim.root_profile_ids, [0, 0])
        with self.assertRaises(ValueError):
            dwg.Simulation(make_case(grass_count=2, layers=2, root_layer_fractions={'shrub': [0.3, 0.7]}), 'cpu',
                           inputs=inp)
        # explicit IDs remain authoritative with arbitrary profile names
        sim = dwg.Simulation(make_case(grass_count=2, layers=2, root_layer_fractions={'a': [0.8, 0.2], 'b': [0.3, 0.7]}),
                             'cpu', inputs=make_inputs([2.0, 3.0], [2.0, 2.0], [1.0, 1.0], pid=[1, 0]))
        np.testing.assert_array_equal(sim.root_profile_ids, [1, 0])

    def test_memory_formula_matches_live_ledger(self):
        frac = {'grass': [0.7, 0.2, 0.1], 'shrub': [0.15, 0.25, 0.6], 'deep': [0.1, 0.3, 0.6]}
        case = make_case(nx=10, ny=6, grass_count=4, layers=3, root_layer_fractions=frac)
        sim = dwg.Simulation(case, 'cpu', inputs=make_inputs([1.5, 2.5, 3.5, 1.5], [1.5, 1.5, 1.5, 2.0], [0.3] * 4,
                                                             pid=[0, 1, 2, 2]))
        P, N, L = 4, 60, 3
        expected = 8 * (5 * L * N + 21 * P) + 4 * P + 8 * 3 * L
        self.assertEqual(dwg.state_bytes_estimate(P, N, L, n_profiles=3), expected)
        self.assertEqual(sim.live_bytes(), expected)
        recs = {r['name']: r for r in sim.array_records()}
        self.assertEqual(sum(r['nbytes'] for n, r in recs.items() if not n.startswith('graph.')), expected)
        self.assertEqual((recs['root_profile_ids']['dtype'], recs['root_profile_ids']['shape']), ('int32', [P]))
        self.assertEqual((recs['root_layer_fractions']['dtype'], recs['root_layer_fractions']['shape']),
                         ('float64', [3, L]))
        self.assertNotIn('Rh', recs)
        led = sim.last_ledger
        self.assertEqual(led['state_bytes'], expected)
        self.assertEqual((led['n_profiles'], led['profile_table_bytes'], led['profile_ids_bytes']), (3, 72, 16))
        self.assertEqual(dwg.admission_ledger(P, N, L, 0, 10)['state_bytes'], dwg.state_bytes_estimate(P, N, L))
        self.assertEqual(led['scratch_allowance_bytes'], dwg.scratch_allowance(P, N, L, 3))
        self.assertEqual(dwg.scratch_allowance(P, N, L, 3) - dwg.scratch_allowance(P, N, L), 3 * L + 96 - (2 * L + 64))
        self.assertEqual(dwg.scratch_allowance(P, N, L), 64 * P + 8 * L * N + 2 * L + 64 + 65536)


class GpuTests(unittest.TestCase):
    @unittest.skipUnless(dwg.gpu_available(), 'CuPy device with compute capability >= 6.0 not available')
    def test_cpu_gpu_parity_short(self):
        case = make_case(nx=16, ny=16, grass_count=6, shrub_count=3, grass_diameter=0.6, shrub_diameter=1.2,
                         root_multiplier=3.0, max_edges=200000)
        rc = dwg.run_case(case, 'cpu', days=21)
        rg = dwg.run_case(case, 'gpu', days=21)
        self.assertEqual(rc['status'], 'ok', rc.get('error'))
        self.assertEqual(rg['status'], 'ok', rg.get('error'))
        for k in ('W', 'B', 'uptake_cum', 'canopy', 'root', 'uptake_week'):
            np.testing.assert_allclose(rg['arrays'][k], rc['arrays'][k], rtol=1e-8, atol=1e-12)
        self.assertEqual(rc['counters'], rg['counters'])
        self.assertTrue(rg['audit']['ok'], rg['audit']['checks'])

    @unittest.skipUnless(dwg.gpu_available(), 'CuPy device with compute capability >= 6.0 not available')
    def test_cpu_gpu_parity_profiles_heterogeneous_water(self):
        frac = {'grass': [0.6, 0.3, 0.1], 'shrub': [0.2, 0.3, 0.5], 'deep': [0.05, 0.15, 0.8]}
        case = make_case(nx=12, ny=12, grass_count=5, layers=3, root_multiplier=2.0, root_layer_fractions=frac,
                         max_edges=200000)
        rng = np.random.default_rng(3)
        xs, ys = rng.uniform(1.5, 4.5, 5), rng.uniform(1.5, 4.5, 5)
        inp = make_inputs(xs, ys, [0.5] * 5, mult=2.0, pid=[0, 1, 2, 2, 1])
        Wh = rng.uniform(0.0, CAP / 3.0, size=(3, 144))
        sims = {}
        for b in ('cpu', 'gpu'):
            s = dwg.Simulation(case, b, inputs=inp)
            s.W[...] = s.xp.asarray(Wh)
            s.W0 = float(Wh.sum())  # chosen initial state set before ANY time advances: budget starts here
            s.step_days(10)  # one weekly rebuild plus 3 pending days
            sims[b] = s
        conv = sm.cp.asnumpy
        for n in ('W', 'R', 'B', 'uptake_cum', 'uptake_week', 'canopy', 'root', 'root_layer_fractions'):
            np.testing.assert_allclose(conv(getattr(sims['gpu'], n)), getattr(sims['cpu'], n), rtol=1e-8, atol=1e-12)
        np.testing.assert_array_equal(conv(sims['gpu'].root_profile_ids), sims['cpu'].root_profile_ids)
        self.assertEqual(sims['cpu'].counters(), sims['gpu'].counters())
        sg = sims['gpu'].summary()
        self.assertTrue(sg['ok'], sg['checks'])
        self.assertEqual(sims['gpu'].live_bytes(), dwg.state_bytes_estimate(5, 144, 3, n_profiles=3))
        rec = sims['gpu'].profile_transfer_record
        self.assertEqual(rec['bytes'], 4 * 5 + 8 * 3 * 3)
        self.assertGreaterEqual(rec['wall_s'], 0.0)
        rg = dwg.run_case(case, 'gpu', days=2, inputs=inp)
        self.assertEqual(rg['status'], 'ok', rg.get('error'))
        ts = rg['transfer_summary']
        self.assertEqual(ts['trait_bytes'], 8 * 5)
        self.assertEqual(ts['total_setup_h2d_bytes'], 52 * 5 + 8 * 3 * 3)  # 5*8P inputs + 4P ids + 8SL table + 8P WUE

    def test_gpu_unavailable_is_explicit(self):
        if dwg.gpu_available():
            self.skipTest('GPU available; unavailability path not exercised')
        res = dwg.run_case(make_case(), 'gpu', days=1, inputs=make_inputs([2.0], [2.0], [1.0]))
        self.assertEqual(res['status'], 'gpu_unavailable')


if __name__ == '__main__':
    unittest.main()
