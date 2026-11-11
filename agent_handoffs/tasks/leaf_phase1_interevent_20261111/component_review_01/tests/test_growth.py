'''Tests for leaf.growth (pure species demand and growth kernels). They build small catalogs from the shipped
example fixture, run only the pure kernels (no model/driver/benchmark state) and compare against hand
calculations written out literally in the tests. NumPy is the reference; Numba and real CUDA parity tests are
skipped when the package/device is unavailable; the CUDA device guard is also tested with a mock that never
touches another GPU.'''
import copy
import json
import os
import sys
import tracemalloc
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from leaf import growth  # noqa: E402
from leaf import traits  # noqa: E402

FIXTURE = os.path.join(ROOT, 'examples', 'phase1', 'species.json')
PHASE0 = os.path.join(ROOT, 'examples', 'phase0', 'species.json')
BIG = 1 << 30

try:
    import numba  # noqa: F401
    HAVE_NUMBA = True
except Exception:   # pragma: no cover
    HAVE_NUMBA = False


def _cuda_device():
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            return None
        return int(cp.cuda.runtime.getDevice())
    except Exception:
        return None


def fixture_doc():
    with open(FIXTURE, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def make_catalog(edit=None, doc=None):
    doc = copy.deepcopy(fixture_doc() if doc is None else doc)
    if edit is not None:
        edit(doc)
    return traits.SpeciesCatalog(doc)


def simple_edit(doc):
    '''Round numbers for hand calculation: grass mature 1, rate .5, support 10, live .5, extra .2, activity 1.'''
    g = doc['species']['grass']
    g['growth']['traits'].update({'mature_biomass_kg': 1.0, 'intrinsic_rate_per_year': 0.5,
                                  'biomass_support_kg_year_per_m3': 10.0, 'live_fraction': 0.5,
                                  'extra_growth_rate_per_year': 0.2, 'monthly_activity': [1.0] * 12,
                                  'temperature_min_c': 0.0, 'temperature_opt_c': 20.0, 'temperature_max_c': 40.0})
    g['water_use']['traits']['transpiration_coefficient'] = 0.5


def arrs(B, E, idx, live, D=None, U=None):
    B = np.array(B, dtype=np.float64)
    out = {'B_kg': B, 'E_kg': np.array(E, dtype=np.float64), 'species_index': np.array(idx, dtype=np.int32),
           'live': np.array(live, dtype=np.bool_)}
    out['requested_m3'] = np.array(np.zeros_like(B) if D is None else D, dtype=np.float64)
    out['delivered_m3'] = np.array(np.zeros_like(B) if U is None else U, dtype=np.float64)
    return out


def vec(*v):
    return np.array(v, dtype=np.float64)


def do_grow(table, a, A, T, year_end=False, **kw):
    return growth.grow(table, B_kg=a['B_kg'], E_kg=a['E_kg'], species_index=a['species_index'], live=a['live'],
                       requested_m3=a['requested_m3'], delivered_m3=a['delivered_m3'], activity_integral=A,
                       temperature_integral=T, year_end=year_end, max_bytes=kw.pop('max_bytes', BIG), **kw)


class SimpleTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = make_catalog(simple_edit)
        cls.table = growth.compile_species(cls.cat, 'numpy', BIG)
        cls.A = vec(0.015, 0.015)
        cls.T = vec(0.0075, 0.0075)

    def one(self, B=0.4, E=0.1, D=0.01, U=0.0075, year_end=False, idx=0, live=True, A=None, T=None):
        a = arrs([B], [E], [idx], [live], [D], [U])
        return do_grow(self.table, a, self.A if A is None else A, self.T if T is None else T, year_end)


class TestHandCalculation(SimpleTable):
    # grass: mature 1, rate .5, k 10, live .5, extra .2; B .4, E .1 (persistent .3), Aint .015, Tint .0075
    # required = .4*.5*.015 = .003 ; requested_support = 10*.01 = .1 -> Fplan = 1
    # normal_plan = .95*.5*.0075*1*.3*(1-.3/1) = .000748125 ; extra_plan = .2*(.1-.003)*(.0075/.015) = .0097
    def test_full_delivery_plan(self):
        r = self.one(U=0.01)
        self.assertAlmostEqual(r['required_support_kg_year'][0], 0.003, places=15)
        self.assertAlmostEqual(r['requested_support_kg_year'][0], 0.1, places=15)
        self.assertAlmostEqual(r['actual_support_kg_year'][0], 0.1, places=15)
        self.assertAlmostEqual(r['normal_plan_kg'][0], 0.000748125, places=15)
        self.assertAlmostEqual(r['extra_plan_kg'][0], 0.0097, places=15)
        self.assertEqual(r['satisfaction'][0], 1.0)
        self.assertAlmostEqual(r['normal_gain_kg'][0], 0.000748125, places=15)
        self.assertAlmostEqual(r['extra_gain_kg'][0], 0.0097, places=15)
        self.assertAlmostEqual(r['E_new_kg'][0], 0.1097, places=14)
        self.assertAlmostEqual(r['B_new_kg'][0], 0.3 + 0.000748125 + 0.1097, places=14)
        self.assertEqual(r['seasonal_loss_kg'][0], 0.0)
        self.assertEqual(r['maintenance_deficit_kg_year'][0], 0.0)

    def test_75_percent_delivery_gives_75_percent_of_full_growth(self):
        full = self.one(U=0.01)
        part = self.one(U=0.0075)
        self.assertAlmostEqual(part['satisfaction'][0], 0.75, places=15)
        self.assertAlmostEqual(part['normal_gain_kg'][0], 0.75 * 0.000748125, places=15)
        self.assertAlmostEqual(part['extra_gain_kg'][0], 0.75 * 0.0097, places=15)
        self.assertAlmostEqual(part['normal_gain_kg'][0] / full['normal_gain_kg'][0], 0.75, places=12)
        self.assertAlmostEqual(part['extra_gain_kg'][0] / full['extra_gain_kg'][0], 0.75, places=12)
        self.assertNotAlmostEqual(part['normal_gain_kg'][0] / full['normal_gain_kg'][0], 0.5625, places=3)
        # the plan itself (requested support) is unchanged: actual support is NOT used a second time
        self.assertEqual(part['normal_plan_kg'][0], full['normal_plan_kg'][0])
        self.assertAlmostEqual(part['actual_support_kg_year'][0], 0.075, places=15)
        self.assertAlmostEqual(part['B_new_kg'][0], 0.3 + 0.75 * 0.000748125 + 0.1 + 0.75 * 0.0097, places=14)

    def test_requested_support_below_requirement_scales_normal_and_blocks_extra(self):
        # D = 1e-4 -> requested_support .001 < required .003 -> Fplan = 1/3, no surplus -> no extra
        r = self.one(D=1e-4, U=1e-4)
        self.assertAlmostEqual(r['normal_plan_kg'][0], 0.000748125 / 3.0, places=15)
        self.assertEqual(r['extra_plan_kg'][0], 0.0)
        self.assertAlmostEqual(r['maintenance_deficit_kg_year'][0], 0.003 - 0.001, places=15)

    def test_dry_and_zero_demand_give_no_gain(self):
        dry = self.one(U=0.0)
        self.assertEqual(dry['satisfaction'][0], 0.0)
        self.assertEqual(dry['normal_gain_kg'][0], 0.0)
        self.assertEqual(dry['extra_gain_kg'][0], 0.0)
        self.assertAlmostEqual(dry['B_new_kg'][0], 0.4, places=15)
        self.assertAlmostEqual(dry['maintenance_deficit_kg_year'][0], 0.003, places=15)   # required - 0
        zero = self.one(D=0.0, U=0.0)
        self.assertEqual(zero['satisfaction'][0], 0.0)
        self.assertEqual(zero['normal_plan_kg'][0], 0.0)    # Fplan = 0
        self.assertEqual(zero['extra_plan_kg'][0], 0.0)
        self.assertEqual(zero['normal_gain_kg'][0] + zero['extra_gain_kg'][0], 0.0)

    def test_dormancy_and_zero_temperature_integral(self):
        z = vec(0.0, 0.0)
        r = self.one(A=z, T=z)                         # Aint = Tint = 0: required 0, Fplan 0, meanT 0
        self.assertEqual(r['required_support_kg_year'][0], 0.0)
        self.assertEqual(r['normal_gain_kg'][0] + r['extra_gain_kg'][0], 0.0)
        cold = self.one(T=z)                           # activity but temperature factor 0 throughout
        self.assertEqual(cold['normal_gain_kg'][0] + cold['extra_gain_kg'][0], 0.0)
        self.assertEqual(cold['normal_plan_kg'][0], 0.0)

    def test_year_end_seasonal_loss_and_ledger(self):
        r = self.one(year_end=True)
        self.assertAlmostEqual(r['seasonal_loss_kg'][0], 0.1 + 0.75 * 0.0097, places=14)
        self.assertEqual(r['E_new_kg'][0], 0.0)
        self.assertAlmostEqual(r['B_new_kg'][0], 0.3 + 0.75 * 0.000748125, places=14)
        gains = r['normal_gain_kg'][0] + r['extra_gain_kg'][0]
        self.assertAlmostEqual(r['B_new_kg'][0], 0.4 + gains - r['seasonal_loss_kg'][0], places=14)
        self.assertTrue(r['ledger']['population_closed'])
        self.assertEqual(r['packet']['ledger_violations'], 0)
        o = self.one(year_end=False)
        self.assertAlmostEqual(o['B_new_kg'][0], 0.4 + o['normal_gain_kg'][0] + o['extra_gain_kg'][0], places=14)
        self.assertEqual(o['seasonal_loss_kg'][0], 0.0)

    def test_above_mature_never_shrinks_and_gains_nothing(self):
        r = self.one(B=1.5, E=0.2, U=0.01)
        self.assertEqual(r['normal_plan_kg'][0], 0.0)
        self.assertEqual(r['extra_plan_kg'][0], 0.0)
        self.assertAlmostEqual(r['B_new_kg'][0], 1.5, places=14)
        self.assertAlmostEqual(r['E_new_kg'][0], 0.2, places=14)

    def test_normal_plan_capped_to_room_to_mature(self):
        r = self.one(B=0.9999, E=0.5, U=0.01)
        self.assertAlmostEqual(r['normal_plan_kg'][0], 1.0 - 0.9999, places=12)   # uncapped ~ 8.9e-4
        self.assertAlmostEqual(r['extra_plan_kg'][0], 0.0, places=12)
        self.assertLessEqual(r['B_new_kg'][0], 1.0 + 1e-12)

    def test_species_differ(self):
        # shrub: mature 12, rate .2, k 3.5, live .4, extra .2 ; required = .4*.4*.015 = .0024 ; reqsup = .035
        a = arrs([0.4, 0.4], [0.1, 0.1], [0, 1], [True, True], [0.01, 0.01], [0.01, 0.01])
        r = do_grow(self.table, a, self.A, self.T)
        self.assertAlmostEqual(r['normal_plan_kg'][0], 0.000748125, places=15)
        self.assertAlmostEqual(r['normal_plan_kg'][1], 0.95 * 0.2 * 0.0075 * 1.0 * 0.3 * (1 - 0.3 / 12.0), places=15)
        self.assertAlmostEqual(r['required_support_kg_year'][1], 0.0024, places=15)
        self.assertAlmostEqual(r['extra_plan_kg'][1], 0.2 * (0.035 - 0.0024) * 0.5, places=15)
        self.assertNotAlmostEqual(r['normal_plan_kg'][0], r['normal_plan_kg'][1], places=6)

    def test_inactive_plants_unchanged_and_ignored(self):
        a = arrs([0.4, 0.7, 0.0], [0.1, 0.2, 0.0], [0, 1, 0], [True, False, False], [0.01, 0, 0], [0.01, 0, 0])
        r = do_grow(self.table, a, self.A, self.T)
        for j in (1, 2):
            self.assertEqual(r['B_new_kg'][j], a['B_kg'][j])
            self.assertEqual(r['E_new_kg'][j], a['E_kg'][j])
            for name in ('normal_gain_kg', 'extra_gain_kg', 'seasonal_loss_kg', 'satisfaction', 'required_support_kg_year',
                         'maintenance_deficit_kg_year'):
                self.assertEqual(r[name][j], 0.0)
        ye = do_grow(self.table, a, self.A, self.T, year_end=True)
        self.assertEqual(ye['E_new_kg'][1], 0.2)       # no invented seasonal loss for a non-live plant
        self.assertEqual(ye['seasonal_loss_kg'][1], 0.0)
        self.assertEqual(r['packet']['n_live'], 1)

    def test_overdelivery_within_tolerance_clamps_and_reports(self):
        r = self.one(D=0.01, U=0.01 * (1.0 + 5e-10))
        self.assertEqual(r['satisfaction'][0], 1.0)
        self.assertEqual(r['packet']['clamped_overdelivery'], 1)
        self.assertAlmostEqual(r['actual_support_kg_year'][0], 10.0 * 0.01 * (1.0 + 5e-10), places=14)   # U not clipped
        with self.assertRaises(growth.GrowthDataError):
            self.one(D=0.01, U=0.01 * (1.0 + 1e-7))

    def test_aggregate_packet_matches_plants(self):
        a = arrs([0.4, 0.3, 0.2], [0.1, 0.0, 0.0], [0, 1, 0], [True, True, True], [0.01, 0.02, 0.005],
                 [0.01, 0.01, 0.005])
        r = do_grow(self.table, a, self.A, self.T)
        pk = r['packet']
        self.assertAlmostEqual(pk['sum_B_new_kg'], float(r['B_new_kg'].sum()), places=13)
        self.assertAlmostEqual(pk['sum_normal_gain_kg'], float(r['normal_gain_kg'].sum()), places=15)
        self.assertAlmostEqual(pk['sum_requested_m3'], 0.035, places=15)
        self.assertAlmostEqual(pk['sum_delivered_m3'], 0.025, places=15)
        self.assertTrue(r['ledger']['population_closed'])


class TestTemperatureAndForcing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = growth.compile_species(make_catalog(), 'numpy', BIG)   # fixture: grass 2/22/40, shrub 0/25/45

    def test_cardinal_points(self):
        for t, g, s in ((-5.0, 0.0, 0.0), (2.0, 0.0, 0.2 / 25 * 0 + 2.0 / 25.0), (12.0, 0.5, 12.0 / 25.0),
                        (22.0, 1.0, 22.0 / 25.0), (25.0, (40 - 25) / 18.0, 1.0), (31.0, 0.5, (45 - 31) / 20.0),
                        (40.0, 0.0, (45 - 40) / 20.0), (45.0, 0.0, 0.0), (60.0, 0.0, 0.0)):
            r = growth.species_response(self.table, month=5, temperature_c=t, max_bytes=BIG)
            self.assertAlmostEqual(r['temperature_factor'][0], g, places=14, msg=f'grass T={t}')
            self.assertAlmostEqual(r['temperature_factor'][1], s, places=14, msg=f'shrub T={t}')
        self.assertEqual(growth.temperature_factor(22.0, 2.0, 22.0, 40.0), 1.0)
        self.assertEqual(growth.temperature_factor(2.0, 2.0, 22.0, 40.0), 0.0)
        self.assertEqual(growth.temperature_factor(40.0, 2.0, 22.0, 40.0), 0.0)

    def test_monthly_activity_by_actual_month(self):
        acts = [growth.species_response(self.table, month=m, temperature_c=22.0, max_bytes=BIG)['activity'][0]
                for m in range(1, 13)]
        self.assertEqual(acts, [0.0, 0.0, 0.2, 0.6, 1.0, 1.0, 0.7, 0.5, 0.8, 0.5, 0.1, 0.0])
        self.assertEqual(growth.species_response(self.table, month=1, temperature_c=22.0,
                                                 max_bytes=BIG)['effective'][0], 0.0)

    def test_invalid_month_and_temperature_refused(self):
        for m in (0, 13, -1, 1.0, True, None, '5'):
            with self.assertRaises(growth.GrowthError, msg=repr(m)):
                growth.species_response(self.table, month=m, temperature_c=20.0, max_bytes=BIG)
        for t in (float('nan'), float('inf'), -273.16, None, True, '20'):
            with self.assertRaises(growth.GrowthError, msg=repr(t)):
                growth.species_response(self.table, month=5, temperature_c=t, max_bytes=BIG)
        self.assertEqual(growth.species_response(self.table, month=5, temperature_c=-273.15, max_bytes=BIG)
                         ['temperature_factor'][0], 0.0)

    def test_integral_increments(self):
        a, t = growth.integral_increments(self.table, month=5, temperature_c=12.0, dt_days=3.0, year_days=366,
                                          max_bytes=BIG)
        self.assertAlmostEqual(a[0], 3.0 * 1.0 / 366.0, places=15)
        self.assertAlmostEqual(t[0], 0.5 * 3.0 * 1.0 / 366.0, places=15)
        a, t = growth.integral_increments(self.table, month=4, temperature_c=22.0, dt_days=7.0, year_days=365,
                                          max_bytes=BIG)
        self.assertAlmostEqual(a[0], 0.6 * 7.0 / 365.0, places=15)
        with self.assertRaises(growth.GrowthError):
            growth.integral_increments(self.table, month=5, temperature_c=12.0, dt_days=3.0, year_days=364,
                                       max_bytes=BIG)
        with self.assertRaises(growth.GrowthError):
            growth.integral_increments(self.table, month=5, temperature_c=12.0, dt_days=7.5, year_days=365,
                                       max_bytes=BIG)

    def test_helper_admission_and_scalar_only_temperature_factor(self):
        S = self.table.n_species
        for op, fn in (('response', lambda **k: growth.species_response(self.table, month=5, temperature_c=12.0, **k)),
                       ('integrals', lambda **k: growth.integral_increments(
                           self.table, month=5, temperature_c=12.0, dt_days=3.0, year_days=365, **k))):
            need = growth.memory_estimate(op, 0, S, 'numpy')['total_bytes']
            names = ' '.join(i['name'] for i in growth.memory_estimate(op, 0, S, 'numpy')['items'])
            for word in ('table.host', 'outputs', 'scratch', 'extra_reserve'):
                self.assertIn(word, names)
            with mock.patch.object(growth, '_response', side_effect=AssertionError('allocated before admission')):
                with self.assertRaises(growth.GrowthBudgetError):
                    fn(max_bytes=need - 1)
                with self.assertRaises(growth.GrowthBudgetError):
                    fn(max_bytes=need, extra_reserve_bytes=1)
                with self.assertRaises(TypeError):
                    fn()                                         # max_bytes is a required keyword
            fn(max_bytes=need)
        for bad in (np.array([1.0, 2.0]), [1.0], np.float64(5.0) * np.ones(3)):
            with self.assertRaises(growth.GrowthError):
                growth.temperature_factor(bad, 2.0, 22.0, 40.0)
            with self.assertRaises(growth.GrowthError):
                growth.temperature_factor(12.0, bad, 22.0, 40.0)
        with self.assertRaises(growth.GrowthError):
            growth.temperature_factor(12.0, 22.0, 22.0, 40.0)
        self.assertIsInstance(growth.temperature_factor(12.0, 2.0, 22.0, 40.0), float)


class TestDemand(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = make_catalog()
        cls.table = growth.compile_species(cls.cat, 'numpy', BIG)

    def dem(self, B, area, idx, live, **kw):
        args = dict(month=5, temperature_c=22.0, pet_mm_per_day=5.0, dt_days=1.0, max_bytes=BIG)
        args.update(kw)
        return growth.demand(self.table, B_kg=np.array(B, float), canopy_area_m2=None if area is None else
                             np.array(area, float), species_index=np.array(idx, np.int32),
                             live=np.array(live, np.bool_), **args)

    def test_canopy_law_units_and_arbitrary_area(self):
        # D = area*PET*0.001*dt*tc*lf*act*tf = 0.2 * 5 * 0.001 * 1 * 0.6 * 0.6 * 1 * 1
        r = self.dem([0.3, 0.3], [0.2, 0.8], [0, 0], [True, True])
        D = r['requested_m3']
        self.assertAlmostEqual(D[0], 0.2 * 5.0 * 0.001 * 0.6 * 0.6, places=15)
        self.assertAlmostEqual(D[1] / D[0], 4.0, places=12)        # same B, different supplied area
        r2 = self.dem([0.3, 3.0], [0.2, 0.2], [0, 0], [True, True])
        self.assertEqual(r2['requested_m3'][0], r2['requested_m3'][1])   # B is not used by the canopy law

    def test_biomass_specific_law_units_no_second_live_fraction(self):
        # D = B * 0.0004 * (PET/5) * dt * act * tf = 2 * 0.0004 * (10/5) * 3 * 1 * 1
        r = self.dem([2.0], [0.1], [1], [True], pet_mm_per_day=10.0, dt_days=3.0, temperature_c=25.0)
        self.assertAlmostEqual(r['requested_m3'][0], 2.0 * 0.0004 * 2.0 * 3.0, places=15)
        r_area = self.dem([2.0], [5.0], [1], [True], pet_mm_per_day=10.0, dt_days=3.0, temperature_c=25.0)
        self.assertEqual(r['requested_m3'][0], r_area['requested_m3'][0])     # area irrelevant for this law
        doc = fixture_doc()
        doc['species']['shrub']['growth']['traits']['live_fraction'] = 0.05
        t2 = growth.compile_species(traits.SpeciesCatalog(doc), 'numpy', BIG)
        r3 = growth.demand(t2, B_kg=vec(2.0), canopy_area_m2=vec(0.1), species_index=np.array([1], np.int32),
                           live=np.array([True]), month=5, temperature_c=25.0, pet_mm_per_day=10.0, dt_days=3.0,
                           max_bytes=BIG)
        self.assertEqual(r3['requested_m3'][0], r['requested_m3'][0])         # live_fraction NOT applied again

    def test_area_optional_only_without_canopy_law(self):
        with self.assertRaises(growth.GrowthError):
            self.dem([1.0], None, [0], [True])
        doc = fixture_doc()
        doc['species']['grass']['water_use'] = {'law': 'biomass_specific_pet_v1',
                                                'traits': {'demand_m3_per_kg_day': 0.001, 'reference_pet_mm_per_day': 4.0}}
        t = growth.compile_species(traits.SpeciesCatalog(doc), 'numpy', BIG)
        r = growth.demand(t, B_kg=vec(0.5, 2.0), canopy_area_m2=None, species_index=np.array([0, 1], np.int32),
                          live=np.array([True, True]), month=5, temperature_c=22.0, pet_mm_per_day=8.0,
                          dt_days=1.0, max_bytes=BIG)
        self.assertAlmostEqual(r['requested_m3'][0], 0.5 * 0.001 * 2.0, places=15)

    def test_zero_pet_dormancy_zero_temperature_and_dead_plants(self):
        self.assertEqual(float(self.dem([1.0], [1.0], [0], [True], pet_mm_per_day=0.0)['requested_m3'][0]), 0.0)
        self.assertEqual(float(self.dem([1.0], [1.0], [0], [True], month=1)['requested_m3'][0]), 0.0)    # dormant
        self.assertEqual(float(self.dem([1.0], [1.0], [0], [True], temperature_c=2.0)['requested_m3'][0]), 0.0)
        self.assertEqual(float(self.dem([1.0], [1.0], [0], [True], temperature_c=40.0)['requested_m3'][0]), 0.0)
        r = self.dem([1.0, 0.0], [1.0, 1.0], [0, 1], [False, False])
        self.assertEqual(list(r['requested_m3']), [0.0, 0.0])
        self.assertEqual(r['packet']['n_live'], 0)

    def test_half_temperature_factor_scales_linearly(self):
        full = self.dem([1.0], [1.0], [0], [True], temperature_c=22.0)['requested_m3'][0]
        half = self.dem([1.0], [1.0], [0], [True], temperature_c=12.0)['requested_m3'][0]
        self.assertAlmostEqual(half / full, 0.5, places=13)

    def test_invalid_forcing_refused(self):
        for kw in ({'pet_mm_per_day': -1.0}, {'pet_mm_per_day': float('nan')}, {'pet_mm_per_day': None},
                   {'dt_days': 0.0}, {'dt_days': 7.5}, {'dt_days': float('inf')}, {'month': 13},
                   {'temperature_c': -300.0}, {'temperature_c': float('nan')}):
            with self.assertRaises(growth.GrowthError, msg=str(kw)):
                self.dem([1.0], [1.0], [0], [True], **kw)

    def test_invalid_content_and_overflow(self):
        for B, area, idx in (([float('nan')], [1.0], [0]), ([0.0], [1.0], [0]), ([1.0], [float('nan')], [0]),
                             ([1.0], [-1.0], [0]), ([1.0], [1.0], [2]), ([1.0], [1.0], [-1])):
            with self.assertRaises(growth.GrowthDataError, msg=str((B, area, idx))):
                self.dem(B, area, idx, [True])
        with self.assertRaises(growth.GrowthDataError):       # biomass law overflow: 1e308 * 80
            self.dem([1e308], [1.0], [1], [True], pet_mm_per_day=1e6, temperature_c=25.0)
        # non-live plants may carry B = 0 (removed) without error
        self.dem([0.0], [0.0], [0], [False])

    def test_dtype_shape_and_type_refused_without_conversion(self):
        good = dict(B_kg=vec(1.0, 2.0), canopy_area_m2=vec(1.0, 1.0), species_index=np.array([0, 1], np.int32),
                    live=np.array([True, True]))
        args = dict(month=5, temperature_c=22.0, pet_mm_per_day=5.0, dt_days=1.0, max_bytes=BIG)
        bads = {'B_kg': [np.array([1.0, 2.0], np.float32), [1.0, 2.0], vec(1.0), np.ones((2, 1)), vec(1, 2, 3, 4)[::2]],
                'species_index': [np.array([0, 1], np.int64), np.array([0.0, 1.0])],
                'live': [np.array([1, 1], np.uint8), [True, True]],
                'canopy_area_m2': [np.array([1.0, 1.0], np.float32)]}
        for name, values in bads.items():
            for v in values:
                kw = dict(good)
                kw[name] = v
                with self.assertRaises(growth.GrowthError, msg=f'{name} {type(v)}'):
                    growth.demand(self.table, **kw, **args)

    def test_failure_leaves_inputs_unchanged(self):
        B, area = vec(1.0, float('nan')), vec(1.0, 1.0)
        idx, live = np.array([0, 1], np.int32), np.array([True, True])
        snap = [x.copy() for x in (B, area, idx, live)]
        with self.assertRaises(growth.GrowthDataError):
            growth.demand(self.table, B_kg=B, canopy_area_m2=area, species_index=idx, live=live, month=5,
                          temperature_c=22.0, pet_mm_per_day=5.0, dt_days=1.0, max_bytes=BIG)
        for x, y in zip(snap, (B, area, idx, live)):
            np.testing.assert_array_equal(x, y)


class TestCompile(unittest.TestCase):
    def doc(self):
        return fixture_doc()

    def bad(self, edit, text=None):
        doc = self.doc()
        edit(doc)
        with self.assertRaises(growth.GrowthError) as cm:
            growth.compile_species(traits.SpeciesCatalog(doc), 'numpy', BIG)
        if text is not None:
            self.assertIn(text, str(cm.exception))

    def test_fixture_compiles_with_sorted_ids_and_hash(self):
        cat = traits.SpeciesCatalog.from_file(FIXTURE)
        t = growth.compile_species(cat, 'numpy', BIG)
        self.assertEqual(t.species_ids, ('grass', 'shrub'))
        self.assertEqual(t.catalog_hash, cat.hash)
        self.assertEqual(t.demand_laws, ('canopy_pet_v1', 'biomass_specific_pet_v1'))
        self.assertTrue(growth.check_catalog(t, cat))
        self.assertFalse(t.host['mature'].flags.writeable)
        rep = t.report()
        self.assertEqual(rep['per_species']['shrub']['demand_reference_pet_mm_per_day'], 5.0)
        other = make_catalog(simple_edit)
        with self.assertRaises(growth.GrowthError):
            growth.check_catalog(t, other)
        self.assertEqual(growth.table_memory(2, 'numpy')['total_bytes'], t.memory()['total_bytes'])

    def test_fixture_traits_in_documented_ranges(self):
        for sid, rec in fixture_doc()['species'].items():
            tr = rec['growth']['traits']
            self.assertGreaterEqual(rec['roots']['lateral_radius_multiplier'], 3.0)
            self.assertLessEqual(rec['roots']['lateral_radius_multiplier'], 5.0)
            self.assertTrue(all(f > 0 for f in rec['roots']['layer_fractions']))     # roots in both layers
            self.assertLessEqual(tr['intrinsic_rate_per_year'], 1.0)

    def test_rate_above_one_rejected_not_rebounded(self):
        self.bad(lambda d: d['species']['grass']['growth']['traits'].update({'intrinsic_rate_per_year': 295.0}), '295')
        self.bad(lambda d: d['species']['grass']['growth']['traits'].update({'intrinsic_rate_per_year': 1.0000001}))
        self.bad(lambda d: d['species']['grass']['growth']['traits'].update({'intrinsic_rate_per_year': -0.1}))
        doc = self.doc()
        doc['species']['grass']['growth']['traits']['intrinsic_rate_per_year'] = 1.0
        doc['species']['shrub']['growth']['traits']['intrinsic_rate_per_year'] = 0.0
        growth.compile_species(traits.SpeciesCatalog(doc), 'numpy', BIG)

    def test_missing_unknown_or_unimplemented_laws_fail(self):
        self.bad(lambda d: d['species']['grass']['growth'].update({'law': 'unimplemented'}), 'unimplemented')
        self.bad(lambda d: d['species']['grass'].update({'growth': None}), 'grass')
        self.bad(lambda d: d['species']['grass']['growth'].update({'law': traits.STEPWAT_GROWTH_LAW}), 'older')
        self.bad(lambda d: d['species']['grass']['growth'].update({'law': 'mystery_v9'}), 'mystery_v9')
        self.bad(lambda d: d['species']['shrub']['water_use'].update({'law': 'unimplemented'}), 'water_use')
        self.bad(lambda d: d['species']['shrub']['water_use'].update({'law': 'transpiration_magic'}))
        self.bad(lambda d: d['species']['shrub'].update({'water_use': None}))

    def test_missing_unknown_and_invalid_traits_fail(self):
        for key in growth.GROWTH_TRAITS:
            self.bad(lambda d, k=key: d['species']['grass']['growth']['traits'].pop(k), key)
        self.bad(lambda d: d['species']['grass']['growth']['traits'].update({'surprise': 1.0}), 'surprise')
        self.bad(lambda d: d['species']['grass']['growth'].update({'fitted': 1.0}), 'unknown keys')
        for k, v in (('mature_biomass_kg', 0.0), ('mature_biomass_kg', -1.0), ('biomass_support_kg_year_per_m3', 0.0),
                     ('live_fraction', 1.5), ('live_fraction', -0.1), ('extra_growth_rate_per_year', 1.1),
                     ('extra_growth_rate_per_year', -0.1), ('mature_biomass_kg', True), ('mature_biomass_kg', 'x'),
                     ('temperature_min_c', -300.0), ('temperature_min_c', 22.0), ('temperature_opt_c', 40.0),
                     ('temperature_max_c', 22.0), ('monthly_activity', [1.0] * 11), ('monthly_activity', [1.0] * 13),
                     ('monthly_activity', [1.0] * 11 + [1.5]), ('monthly_activity', [1.0] * 11 + [-0.1]),
                     ('monthly_activity', 'x')):
            self.bad(lambda d, k=k, v=v: d['species']['grass']['growth']['traits'].update({k: v}), None)
        self.bad(lambda d: d['species']['grass']['water_use']['traits'].pop('transpiration_coefficient'))
        self.bad(lambda d: d['species']['grass']['water_use']['traits'].update({'transpiration_coefficient': -1.0}))
        self.bad(lambda d: d['species']['shrub']['water_use']['traits'].pop('reference_pet_mm_per_day'))
        self.bad(lambda d: d['species']['shrub']['water_use']['traits'].update({'reference_pet_mm_per_day': 0.0}))
        self.bad(lambda d: d['species']['shrub']['water_use']['traits'].update({'demand_m3_per_kg_day': -0.1}))
        self.bad(lambda d: d['species']['shrub']['water_use']['traits'].update({'canopy_coefficient': 1.0}))

    def test_phase0_catalog_still_loads_but_is_not_usable_here(self):
        if not os.path.exists(PHASE0):
            self.skipTest('phase0 example not present')
        cat = traits.SpeciesCatalog.from_file(PHASE0)       # Phase 0 reading is untouched
        self.assertEqual(cat.law('grass', 'growth'), 'unimplemented')
        with self.assertRaises(growth.GrowthError):
            growth.compile_species(cat, 'numpy', BIG)

    def test_backend_and_budget_arguments(self):
        cat = make_catalog()
        for kw in ({'backend': None}, {'backend': 'auto'}, {'backend': 'cuda'}):
            with self.assertRaises(growth.GrowthError):
                growth.compile_species(cat, kw['backend'], BIG)
        with self.assertRaises(growth.GrowthError):
            growth.compile_species(cat, 'numpy', BIG, device=0)
        with self.assertRaises(growth.GrowthError):
            growth.compile_species(cat, 'cuda', BIG)                   # no implicit device
        with self.assertRaises(growth.GrowthError):
            growth.compile_species(cat, 'numpy', 0)
        with self.assertRaises(growth.GrowthError):
            growth.compile_species(cat, 'numpy', True)

    def test_budget_refused_before_parsing_or_allocation(self):
        cat = make_catalog()
        need = growth.table_memory(cat.n_species, 'numpy')['total_bytes']
        with mock.patch.object(growth, '_parse_species', side_effect=AssertionError('parsed before admission')):
            with self.assertRaises(growth.GrowthBudgetError):
                growth.compile_species(cat, 'numpy', need - 1)
            with self.assertRaises(growth.GrowthBudgetError):
                growth.compile_species(cat, 'numpy', need, extra_reserve_bytes=1)
        growth.compile_species(cat, 'numpy', need)


class TestAdmissionAndMemory(SimpleTable):
    def test_validation_refused_before_work(self):
        a = arrs([0.4] * 1000, [0.1] * 1000, [0] * 1000, [True] * 1000, [0.01] * 1000, [0.01] * 1000)
        with mock.patch.object(growth, '_grow_numpy', side_effect=AssertionError('ran before admission')):
            with self.assertRaises(growth.GrowthBudgetError):
                do_grow(self.table, a, self.A, self.T, max_bytes=100)
            with self.assertRaises(growth.GrowthBudgetError):
                do_grow(self.table, a, self.A, self.T, max_bytes=10 ** 9, extra_reserve_bytes=10 ** 9)
            bad = dict(a)
            bad['B_kg'] = a['B_kg'].astype(np.float32)
            with self.assertRaises(growth.GrowthError):
                do_grow(self.table, bad, self.A, self.T)
        with mock.patch.object(growth, '_demand_numpy', side_effect=AssertionError('ran before admission')):
            with self.assertRaises(growth.GrowthBudgetError):
                growth.demand(self.table, B_kg=a['B_kg'], canopy_area_m2=a['B_kg'], species_index=a['species_index'],
                              live=a['live'], month=5, temperature_c=20.0, pet_mm_per_day=5.0, dt_days=1.0,
                              max_bytes=100)

    def test_exact_threshold(self):
        n = 50
        a = arrs([0.4] * n, [0.1] * n, [0] * n, [True] * n, [0.01] * n, [0.01] * n)
        est = growth.memory_estimate('grow', n, 2, 'numpy')['total_bytes']
        do_grow(self.table, a, self.A, self.T, max_bytes=est)
        with self.assertRaises(growth.GrowthBudgetError):
            do_grow(self.table, a, self.A, self.T, max_bytes=est - 1)
        with self.assertRaises(growth.GrowthBudgetError):
            do_grow(self.table, a, self.A, self.T, max_bytes=est, extra_reserve_bytes=1)

    def test_estimate_is_itemized_and_scales(self):
        m = growth.memory_estimate('grow', 1000, 2, 'numpy', extra_reserve_bytes=7)
        self.assertEqual(m['total_bytes'], sum(i['bytes'] for i in m['items']))
        names = ' '.join(i['name'] for i in m['items'])
        for word in ('table.host', 'input B_kg', 'outputs', 'numpy chunk', 'packet', 'extra_reserve'):
            self.assertIn(word, names)
        small, big = growth.memory_estimate('grow', 1000, 2, 'numpy'), growth.memory_estimate('grow', 10 ** 9, 2, 'numpy')
        chunk = lambda m_: [i['bytes'] for i in m_['items'] if 'numpy chunk' in i['name']][0]
        self.assertEqual(chunk(big), growth.NUMPY_GROW_CHUNK_BYTES_PER_PLANT * growth.NUMPY_CHUNK)   # bounded by chunk
        self.assertGreater(chunk(big), chunk(small))
        self.assertEqual(growth.memory_estimate('grow', 10, 2, 'numpy')['device_bytes'], 0)
        g = growth.memory_estimate('demand', 10 ** 6, 2, 'cuda')
        self.assertGreater(g['device_bytes'], 21 * 10 ** 6)
        self.assertTrue(any(i['where'] == 'host' and i['kind'] == 'packet' for i in g['items']))
        self.assertTrue(any('device' == i['where'] and i['kind'] == 'table' for i in g['items']))
        self.assertIn('exclusions', g)

    def test_real_allocation_stays_inside_estimate(self):
        n = 20000
        rng = np.random.RandomState(3)
        a = arrs(rng.uniform(0.1, 0.9, n), np.zeros(n), np.zeros(n), np.ones(n), np.full(n, 0.01), np.full(n, 0.008))
        est = growth.memory_estimate('grow', n, 2, 'numpy')
        allocating = est['allocating_bytes']
        tracemalloc.start()
        try:
            do_grow(self.table, a, self.A, self.T)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLessEqual(peak, allocating)


class TestInvalidGrowthInputs(SimpleTable):
    def assert_refused(self, exc=growth.GrowthDataError, **over):
        a = arrs([0.4, 0.5], [0.1, 0.0], [0, 1], [True, True], [0.01, 0.01], [0.01, 0.01])
        A, T = self.A.copy(), self.T.copy()
        for k, v in over.items():
            if k == 'A':
                A = v
            elif k == 'T':
                T = v
            else:
                a[k] = v
        snap = {k: v.copy() for k, v in a.items()}
        with self.assertRaises(exc, msg=str(list(over))):
            do_grow(self.table, a, A, T)
        for k in snap:
            np.testing.assert_array_equal(snap[k], a[k])

    def test_bad_population_content(self):
        nan = float('nan')
        inf = float('inf')
        self.assert_refused(B_kg=vec(nan, 0.5))
        self.assert_refused(B_kg=vec(0.0, 0.5))                       # live B must be > 0
        self.assert_refused(B_kg=vec(-0.4, 0.5))
        self.assert_refused(E_kg=vec(0.5, 0.0))                       # E > B
        self.assert_refused(E_kg=vec(-0.1, 0.0))
        self.assert_refused(E_kg=vec(nan, 0.0))
        self.assert_refused(species_index=np.array([0, 2], np.int32))
        self.assert_refused(species_index=np.array([-1, 0], np.int32))
        self.assert_refused(requested_m3=vec(nan, 0.01))
        self.assert_refused(requested_m3=vec(-0.01, 0.01))
        self.assert_refused(delivered_m3=vec(0.01, -1.0))
        self.assert_refused(delivered_m3=vec(inf, 0.01))
        self.assert_refused(delivered_m3=vec(0.0100001, 0.01))        # U > D beyond roundoff
        self.assert_refused(requested_m3=vec(0.0, 0.01), delivered_m3=vec(1e-12, 0.01))   # D == 0 requires U == 0
        self.assert_refused(live=np.array([False, True]))              # water delivered to a non-live plant

    def test_bad_integrals(self):
        self.assert_refused(A=vec(0.015, float('nan')))
        self.assert_refused(A=vec(0.02, 0.015))                         # > 7/365
        self.assert_refused(T=vec(0.02, 0.0075), A=vec(0.015, 0.015))   # Tint > Aint
        self.assert_refused(A=vec(-0.001, 0.015), T=vec(-0.002, 0.0))
        self.assert_refused(growth.GrowthError, A=np.array([0.015, 0.015], np.float32))
        self.assert_refused(growth.GrowthError, A=np.array([0.015]))
        self.assert_refused(growth.GrowthError, T=[0.0075, 0.0075])
        ok = vec(7.0 / 365.0, 7.0 / 365.0)
        self.assertEqual(do_grow(self.table, arrs([0.4], [0.1], [0], [True], [0.01], [0.01]), ok, ok * 0.5)
                         ['packet']['bad_input'], 0)

    def test_bad_types_shapes_flags(self):
        good = arrs([0.4, 0.5], [0.1, 0.0], [0, 1], [True, True], [0.01, 0.01], [0.01, 0.01])
        for name, values in {'B_kg': [good['B_kg'].astype(np.float32), list(good['B_kg']), np.ones((2, 1)),
                                      vec(1, 2, 3, 4)[::2], vec(0.4)],
                             'species_index': [good['species_index'].astype(np.int64)],
                             'live': [good['live'].astype(np.uint8)]}.items():
            for v in values:
                a = dict(good)
                a[name] = v
                with self.assertRaises(growth.GrowthError, msg=f'{name} {type(v)}'):
                    do_grow(self.table, a, self.A, self.T)
        for kw in ({'year_end': 1}, {'year_end': None}):
            with self.assertRaises(growth.GrowthError):
                growth.grow(self.table, B_kg=good['B_kg'], E_kg=good['E_kg'], species_index=good['species_index'],
                            live=good['live'], requested_m3=good['requested_m3'], delivered_m3=good['delivered_m3'],
                            activity_integral=self.A, temperature_integral=self.T, max_bytes=BIG, **kw)
        for rt in (-1e-9, 1e-3, float('nan'), None):
            with self.assertRaises(growth.GrowthError):
                do_grow(self.table, good, self.A, self.T, overdelivery_rtol=rt)

    def test_overflow_refused(self):
        # D = 1e308 -> requested_support = 10 * 1e308 overflows to inf; refused, nothing returned
        a = arrs([0.4], [0.1], [0], [True], [1e308], [1e308])
        with self.assertRaises(growth.GrowthDataError):
            do_grow(self.table, a, self.A, self.T)
        a = arrs([1e308], [0.0], [0], [True], [0.01], [0.01])         # ledger tolerance 16*eps*(2e308) overflows
        with self.assertRaises(growth.GrowthDataError):               # never certified with an inf bound
            do_grow(self.table, a, self.A, self.T)
        big = arrs([1e308, 1e308], [0.0, 0.0], [0, 0], [True, True], [0.0, 0.0], [0.0, 0.0])
        a2 = dict(big)
        a2['B_kg'] = vec(1.7e308, 1.7e308)
        with self.assertRaises(growth.GrowthDataError):               # aggregate sum_B overflows
            do_grow(self.table, a2, self.A, self.T)

    def test_empty_population(self):
        a = arrs([], [], [], [], [], [])
        r = do_grow(self.table, a, self.A, self.T)
        self.assertEqual(r['B_new_kg'].shape, (0,))
        self.assertEqual(r['packet']['n_live'], 0)
        d = growth.demand(self.table, B_kg=a['B_kg'], canopy_area_m2=a['B_kg'], species_index=a['species_index'],
                          live=a['live'], month=5, temperature_c=20.0, pet_mm_per_day=5.0, dt_days=1.0,
                          max_bytes=BIG)
        self.assertEqual(d['requested_m3'].shape, (0,))


def random_population(table, n, seed):
    rng = np.random.RandomState(seed)
    S = table.n_species
    idx = rng.randint(0, S, n).astype(np.int32)
    live = rng.uniform(size=n) < 0.85
    mature = table.host['mature'][idx]
    B = np.where(live, rng.uniform(0.02, 1.2, n) * mature, rng.uniform(0.0, 1.0, n) * mature)
    B = np.maximum(B, 1e-6)
    E = B * rng.uniform(0.0, 0.5, n)
    area = rng.uniform(0.0, 3.0, n)
    frac = np.where(rng.uniform(size=n) < 0.2, 1.0, rng.uniform(0.0, 1.0, n))
    return B, E, idx, live, area, frac


def run_chain(table, xp_array, to_host, n=3000, seed=11, year_end=False):
    B, E, idx, live, area, frac = random_population(table, n, seed)
    dB, dE, didx, dlive, darea = (xp_array(v) for v in (B, E, idx, live, area))
    dem = growth.demand(table, B_kg=dB, canopy_area_m2=darea, species_index=didx, live=dlive, month=6,
                        temperature_c=17.0, pet_mm_per_day=6.5, dt_days=5.0, max_bytes=BIG)
    D = to_host(dem['requested_m3'])
    U = D * frac
    A, T = growth.integral_increments(table, month=6, temperature_c=17.0, dt_days=5.0, year_days=365,
                                         max_bytes=BIG)
    A, T = np.ascontiguousarray(A), np.ascontiguousarray(T)
    gr = growth.grow(table, B_kg=dB, E_kg=dE, species_index=didx, live=dlive, requested_m3=dem['requested_m3'],
                     delivered_m3=xp_array(np.ascontiguousarray(U)), activity_integral=A, temperature_integral=T,
                     year_end=year_end, max_bytes=BIG)
    return dem, gr, to_host


class TestBackendParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = make_catalog()
        cls.ref_table = growth.compile_species(cls.cat, 'numpy', BIG)
        cls.ref = {ye: run_chain(cls.ref_table, np.ascontiguousarray, lambda x: x, year_end=ye)
                   for ye in (False, True)}

    def compare(self, table, xp_array, to_host, tol=1e-12):
        for ye in (False, True):
            rd, rg, _ = self.ref[ye]
            d, g, _ = run_chain(table, xp_array, to_host, year_end=ye)
            np.testing.assert_allclose(to_host(d['requested_m3']), rd['requested_m3'], rtol=tol, atol=1e-300)
            for name in growth.GROW_OUTPUTS:
                np.testing.assert_allclose(to_host(g[name]), rg[name], rtol=tol, atol=1e-14, err_msg=name)
            for k, v in rg['packet'].items():
                if isinstance(v, int):
                    self.assertEqual(g['packet'][k], v, k)
                else:
                    self.assertAlmostEqual(g['packet'][k], v, delta=1e-9 * max(1.0, abs(v)), msg=k)
            self.assertEqual(g['packet']['ledger_violations'], 0)

    def test_reference_is_sane(self):
        rd, rg, _ = self.ref[False]
        self.assertGreater(rg['packet']['n_growing'], 0)
        self.assertGreater(rg['packet']['n_live'], 2000)
        self.assertGreater(rd['packet']['n_demanding'], 0)
        ye = self.ref[True][1]
        self.assertGreater(ye['packet']['sum_seasonal_loss_kg'], 0.0)
        self.assertTrue(ye['ledger']['population_closed'])

    @unittest.skipUnless(HAVE_NUMBA, 'numba not available')
    def test_numba_matches_numpy(self):
        table = growth.compile_species(self.cat, 'numba', BIG)
        self.compare(table, np.ascontiguousarray, lambda x: x)

    def test_cuda_matches_numpy(self):
        dev = _cuda_device()
        if dev is None:
            self.skipTest('no CUDA device / CuPy')
        import cupy as cp
        table = growth.compile_species(self.cat, 'cuda', BIG, device=dev)
        self.compare(table, cp.asarray, lambda x: cp.asnumpy(x) if hasattr(x, 'device') else x, tol=1e-11)

    def test_cuda_rejects_host_arrays_and_other_device(self):
        dev = _cuda_device()
        if dev is None:
            self.skipTest('no CUDA device / CuPy')
        table = growth.compile_species(self.cat, 'cuda', BIG, device=dev)
        with self.assertRaises(growth.GrowthError):
            growth.demand(table, B_kg=vec(1.0), canopy_area_m2=vec(1.0), species_index=np.array([0], np.int32),
                          live=np.array([True]), month=5, temperature_c=20.0, pet_mm_per_day=5.0, dt_days=1.0,
                          max_bytes=BIG)


class TestNonFiniteLedgerAllBackends(unittest.TestCase):
    '''Revision-2 regression: mature = B = 1e308 gave an infinite ledger tolerance that was certified closed.'''

    def tables(self):
        def edit(doc):
            tr = doc['species']['grass']['growth']['traits']
            tr.update({'mature_biomass_kg': 1e308, 'intrinsic_rate_per_year': 0.5, 'monthly_activity': [1.0] * 12,
                       'temperature_min_c': 0.0, 'temperature_opt_c': 20.0, 'temperature_max_c': 40.0})
        cat = make_catalog(edit)
        out = [(growth.compile_species(cat, 'numpy', BIG), np.ascontiguousarray)]
        if HAVE_NUMBA:
            out.append((growth.compile_species(cat, 'numba', BIG), np.ascontiguousarray))
        dev = _cuda_device()
        if dev is not None:
            import cupy as cp
            out.append((growth.compile_species(cat, 'cuda', BIG, device=dev), cp.asarray))
        return out

    def test_infinite_bound_refused_everywhere(self):
        A = vec(0.01, 0.01)
        for table, put in self.tables():
            a = arrs([1e308], [0.0], [0], [True], [0.0], [0.0])
            dev = {k: put(v) for k, v in a.items()}
            with self.assertRaises(growth.GrowthDataError, msg=table.backend):
                growth.grow(table, B_kg=dev['B_kg'], E_kg=dev['E_kg'], species_index=dev['species_index'],
                            live=dev['live'], requested_m3=dev['requested_m3'], delivered_m3=dev['delivered_m3'],
                            activity_integral=A, temperature_integral=A.copy(), year_end=False, max_bytes=BIG)

    def test_ordinary_finite_values_still_close_everywhere(self):
        A, T = vec(0.015, 0.015), vec(0.0075, 0.0075)
        for table, put in self.tables():
            a = arrs([3.0, 5.0], [0.5, 0.0], [0, 1], [True, True], [0.01, 0.01], [0.01, 0.01])
            dev = {k: put(v) for k, v in a.items()}
            r = growth.grow(table, B_kg=dev['B_kg'], E_kg=dev['E_kg'], species_index=dev['species_index'],
                            live=dev['live'], requested_m3=dev['requested_m3'], delivered_m3=dev['delivered_m3'],
                            activity_integral=A, temperature_integral=T, year_end=False, max_bytes=BIG)
            self.assertTrue(r['ledger']['population_closed'])
            self.assertEqual(r['packet']['ledger_violations'], 0)


class TestBackendNaming(unittest.TestCase):
    def test_canonical_cuda_and_alias(self):
        cat = make_catalog()
        self.assertEqual(growth.BACKENDS, ('numpy', 'numba', 'cuda'))
        a = growth.memory_estimate('grow', 10, 2, 'cuda')
        b = growth.memory_estimate('grow', 10, 2, 'cupy')            # documented alias
        self.assertEqual(a['total_bytes'], b['total_bytes'])
        self.assertEqual(b['backend'], 'cuda')
        with mock.patch.object(growth, '_current_device_id', return_value=0), \
                mock.patch.object(growth, '_cupy', side_effect=AssertionError('touched cupy')):
            for name in ('cuda', 'cupy'):
                with self.assertRaises(growth.GrowthDeviceError):
                    growth.compile_species(cat, name, BIG, device=1)
        with self.assertRaises(growth.GrowthError):
            growth.compile_species(cat, 'gpu', BIG)
        self.assertEqual(growth.compile_species(cat, 'numpy', BIG).backend, 'numpy')


class TestDeviceGuardMocked(SimpleTable):
    def test_current_device_mismatch_refused_before_any_cupy_work(self):
        fake = growth.CompiledSpecies('cuda', 1, self.table.catalog_hash, None, self.table.species_ids,
                                      self.table.host, self.table.host, self.table.demand_laws)
        a = arrs([0.4], [0.1], [0], [True], [0.01], [0.01])
        with mock.patch.object(growth, '_current_device_id', return_value=0) as cur, \
                mock.patch.object(growth, '_cupy', side_effect=AssertionError('touched cupy')) as cp:
            with self.assertRaises(growth.GrowthDeviceError):
                do_grow(fake, a, self.A, self.T)
            with self.assertRaises(growth.GrowthDeviceError):
                growth.demand(fake, B_kg=a['B_kg'], canopy_area_m2=a['B_kg'], species_index=a['species_index'],
                              live=a['live'], month=5, temperature_c=20.0, pet_mm_per_day=5.0, dt_days=1.0,
                              max_bytes=BIG)
            cur.assert_called()
            cp.assert_not_called()

    def test_array_on_other_device_refused(self):
        fake = growth.CompiledSpecies('cuda', 0, self.table.catalog_hash, None, self.table.species_ids,
                                      self.table.host, self.table.host, self.table.demand_laws)

        class FakeArray:
            pass

        class FakeCupy:
            ndarray = FakeArray

        arr = FakeArray()
        with mock.patch.object(growth, '_current_device_id', return_value=0), \
                mock.patch.object(growth, '_cupy', return_value=FakeCupy), \
                mock.patch.object(growth, '_array_device_id', return_value=3):
            with self.assertRaises(growth.GrowthDeviceError):
                growth._check_vector(fake, 'B_kg', arr, np.float64)

    def test_compile_cupy_requires_exact_current_device(self):
        with mock.patch.object(growth, '_current_device_id', return_value=0), \
                mock.patch.object(growth, '_cupy', side_effect=AssertionError('touched cupy')):
            with self.assertRaises(growth.GrowthDeviceError):
                growth.compile_species(make_catalog(), 'cuda', BIG, device=2)


if __name__ == '__main__':
    unittest.main()
