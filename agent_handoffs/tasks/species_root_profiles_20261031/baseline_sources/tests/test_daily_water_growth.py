'''Small independent unittest checks for benchmarks/daily_water_growth.py (fixture rules, not ecology).'''
import math
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


def make_inputs(xs, ys, radii, mult=1.0):
    cr = np.asarray(radii, dtype=np.float64)
    return {'x': np.asarray(xs, np.float64), 'y': np.asarray(ys, np.float64), 'canopy_radius': cr,
            'root_radius': mult * cr, 'ids': np.arange(len(cr), dtype=np.int64)}


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

    def test_gpu_unavailable_is_explicit(self):
        if dwg.gpu_available():
            self.skipTest('GPU available; unavailability path not exercised')
        res = dwg.run_case(make_case(), 'gpu', days=1, inputs=make_inputs([2.0], [2.0], [1.0]))
        self.assertEqual(res['status'], 'gpu_unavailable')


if __name__ == '__main__':
    unittest.main()
