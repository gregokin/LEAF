'''LEAF Phase 2 age / background hazard and persistent stress tests (unittest): leaf.demography, pure kernel level.

Run from the candidate root:  python -m unittest tests.test_demography -v

* No native SYRUP is needed: `demography.decide` is a pure function of explicit arrays + (seed, counter). The session
  level (annual accumulators, transactional rebind, checkpoint equality of the random draws) is covered in
  tests/test_interevent.py.
* Coefficients come from the UNCALIBRATED fixture catalog examples/phase2/species.json, edited per test through
  explicit catalog modifications (there are no module defaults). Expected values are recomputed from the documented
  formulas here, never taken from module constants.
* numba / CUDA cases SKIP when the backend is unavailable. Written WITHOUT the ability to execute them (file tools
  only): they are NOT RUN by the author; Codex records the real runs.
'''
from __future__ import annotations

import json
import math
import pathlib
import unittest

import numpy as np

from leaf import demography as dm
from leaf import traits

ROOT = pathlib.Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / 'examples' / 'phase2' / 'species.json'
BIG = 1 << 30
try:
    import numba  # noqa: F401
    NUMBA = True
except Exception:  # pragma: no cover - environment dependent
    NUMBA = False
try:
    import cupy as _cp
    CUDA = bool(_cp.cuda.runtime.getDeviceCount() > 0)
except Exception:  # pragma: no cover - environment dependent
    CUDA = False


def load_cat(modify=None):
    with open(CATALOG_PATH, 'r', encoding='utf-8') as fh:
        doc = json.load(fh)
    if modify is not None:
        modify(doc)
    return traits.SpeciesCatalog(doc)


def only_stress(p_stress=0.0, loss=0.25, species=('grass',)):
    '''Catalog: no age / background hazard anywhere; stress law kept (with the given response) on `species`.'''
    def mod(doc):
        for sid, rec in doc['species'].items():
            tr = rec['mortality']['traits']
            tr['age_hazard'] = None
            tr['background_hazard_per_year'] = 0.0
            if sid in species:
                tr['stress'].update(mortality_probability=p_stress, biomass_loss_fraction=loss)
            else:
                tr['stress'] = None
    return mod


def only_age(bg_grass=0.0, weibull=True):
    def mod(doc):
        for sid, rec in doc['species'].items():
            tr = rec['mortality']['traits']
            tr['stress'] = None
            if not weibull:
                tr['age_hazard'] = None
            tr['background_hazard_per_year'] = bg_grass if sid == 'grass' else 0.0
    return mod


class Call:
    '''Build the explicit arrays of one `decide` call (backend numpy / numba arrays).'''

    def __init__(self, cat, names, ids, age, stress=None, req=None, act=None, plan=None, gain=None):
        n = len(names)
        self.cat = cat
        self.ids = np.array(ids, dtype=np.int64)
        self.sidx = np.array([cat.index(s) for s in names], dtype=np.int32)
        self.age = np.array(age, dtype=np.float64)
        z = np.zeros(n)
        self.stress = np.zeros(n, np.int32) if stress is None else np.array(stress, dtype=np.int32)
        self.req = z.copy() if req is None else np.array(req, dtype=np.float64)
        self.act = z.copy() if act is None else np.array(act, dtype=np.float64)
        self.plan = z.copy() if plan is None else np.array(plan, dtype=np.float64)
        self.gain = z.copy() if gain is None else np.array(gain, dtype=np.float64)

    def run(self, table, period=365.0, seed=7, counter=0, max_bytes=BIG):
        return dm.decide(table, species_index=self.sidx, plant_ids=self.ids, age_end_days=self.age, stress=self.stress,
                         required_support=self.req, actual_support=self.act, planned_growth=self.plan,
                         actual_growth=self.gain, period_days=period, seed=seed, counter=counter, max_bytes=max_bytes)


class KeyedRandomness(unittest.TestCase):
    def test_vector_matches_independent_python_reference_and_range(self):
        ids = np.array([1, 2, 3, 10 ** 12, 2 ** 40 + 5], dtype=np.int64)
        for stream in (dm.STREAM_AGE, dm.STREAM_STRESS):
            got = dm.keyed_uniform(123456789, 4, stream, ids)
            ref = np.array([dm.keyed_uniform_py(123456789, 4, stream, int(i)) for i in ids])
            np.testing.assert_array_equal(got, ref)
            self.assertTrue(np.all((got >= 0.0) & (got < 1.0)))
        u = dm.keyed_uniform(7, 0, dm.STREAM_AGE, np.arange(100000))
        self.assertLess(abs(float(u.mean()) - 0.5), 0.01)

    def test_independent_of_row_order_and_changes_with_seed_counter_stream(self):
        ids = np.arange(1, 201, dtype=np.int64)
        a = dm.keyed_uniform(11, 0, dm.STREAM_AGE, ids)
        perm = np.random.default_rng(3).permutation(ids.shape[0])
        b = dm.keyed_uniform(11, 0, dm.STREAM_AGE, ids[perm])
        np.testing.assert_array_equal(a[perm], b)                     # draw is a function of the ID only
        for other in (dm.keyed_uniform(12, 0, dm.STREAM_AGE, ids), dm.keyed_uniform(11, 1, dm.STREAM_AGE, ids),
                      dm.keyed_uniform(11, 0, dm.STREAM_STRESS, ids)):
            self.assertFalse(np.array_equal(a, other))
        self.assertEqual(len(set(a.tolist())), 200)                   # no collisions among 200 draws

    def test_strict_seed_counter_validation(self):
        for bad in (True, -1, 1.5, 2 ** 64, None, '7'):
            with self.assertRaises(dm.DemographyError):
                dm.check_seed(bad)
        for bad in (True, -1, 1.5):
            with self.assertRaises(dm.DemographyError):
                dm.check_counter(bad)
        self.assertEqual(dm.check_seed(2 ** 64 - 1), 2 ** 64 - 1)
        with self.assertRaises(dm.DemographyError):
            dm.keyed_uniform(1, 0, dm.STREAM_AGE, np.array([1.5]))


class LawParsing(unittest.TestCase):
    def test_inactive_catalogs_and_explicit_unimplemented(self):
        p1 = traits.load_catalog(ROOT / 'examples' / 'phase1' / 'species.json')
        t = dm.compile_species(p1, 'numpy')
        self.assertFalse(t.any_active)
        self.assertEqual(t.resident_bytes(), 0)
        cat = load_cat(lambda d: d['species']['grass'].update(mortality={'law': 'unimplemented'}))
        t = dm.compile_species(cat, 'numpy')
        self.assertEqual(t.active, (False, True))
        self.assertTrue(t.any_stress)

    def test_unknown_and_malformed_laws_refuse(self):
        def path(d):
            return d['species']['grass']['mortality']['traits']

        mods = {
            'law': lambda d: d['species']['grass']['mortality'].update(law='weibull_v2'),
            'section_key': lambda d: d['species']['grass']['mortality'].update(surprise=1),
            'trait_missing': lambda d: path(d).pop('background_hazard_per_year'),
            'trait_unknown': lambda d: path(d).update(extra=1),
            'bg_negative': lambda d: path(d).update(background_hazard_per_year=-0.1),
            'shape_zero': lambda d: path(d)['age_hazard'].update(weibull_shape=0.0),
            'scale_negative': lambda d: path(d)['age_hazard'].update(weibull_scale_years=-1.0),
            'metric': lambda d: path(d)['stress'].update(metric='yield'),
            'low_zero': lambda d: path(d)['stress'].update(low_threshold=0.0),
            'low_above_one': lambda d: path(d)['stress'].update(low_threshold=1.5),
            'exempt_negative': lambda d: path(d)['stress'].update(exempt_until_age_days=-1.0),
            'limit_zero': lambda d: path(d)['stress'].update(limit_years=0),
            'limit_float': lambda d: path(d)['stress'].update(limit_years=2.0),
            'cap_below_limit': lambda d: path(d)['stress'].update(counter_cap_years=1),
            'p_above_one': lambda d: path(d)['stress'].update(mortality_probability=1.5),
            'loss_one': lambda d: path(d)['stress'].update(biomass_loss_fraction=1.0),
            'stress_key': lambda d: path(d)['stress'].pop('metric'),
        }
        for name, m in mods.items():
            with self.assertRaises(dm.DemographyError, msg=name):
                dm.compile_species(load_cat(m), 'numpy')

    def test_life_expectancy_metadata_is_not_read(self):
        a = dm.compile_species(load_cat(), 'numpy')
        b = dm.compile_species(load_cat(lambda d: d['species']['grass'].update(
            life_expectancy={'mean': 1, 'unit': 'days'})), 'numpy')
        for sid in ('grass', 'shrub'):
            self.assertEqual(a.coefficients(sid), b.coefficients(sid))

    def test_table_holds_catalog_values_and_caps(self):
        t = dm.compile_species(load_cat(), 'numpy')
        c = t.coefficients('shrub')
        self.assertEqual((c['limit'], c['cap'], c['metric']), (3, 6, 1))
        self.assertEqual(c['scale'], 60.0)
        np.testing.assert_array_equal(t.counter_cap(), [4, 6])


class AgeAndBackgroundHazard(unittest.TestCase):
    def test_weibull_plus_background_probability_matches_closed_form(self):
        cat = load_cat(only_age(bg_grass=0.01))
        t = dm.compile_species(cat, 'numpy')
        age_end, period = 2000.0, 365.0
        call = Call(cat, ['grass', 'shrub'], [1, 2], [age_end, age_end])
        out = call.run(t, period=period)
        k, lam, bg = 3.0, 8.0, 0.01
        H = lambda a: (a / 365.0 / lam) ** k                              # noqa: E731
        dh = H(age_end) - H(age_end - period) + bg * period / 365.0
        self.assertAlmostEqual(float(out['p_age'][0]), -math.expm1(-dh), places=13)
        k2, lam2 = 4.0, 60.0
        dh2 = (age_end / 365.0 / lam2) ** k2 - ((age_end - period) / 365.0 / lam2) ** k2
        self.assertAlmostEqual(float(out['p_age'][1]), -math.expm1(-dh2), places=13)

    def test_probability_zero_never_kills_and_no_life_mean_cutoff(self):
        cat = load_cat(only_age(bg_grass=0.0, weibull=False))
        t = dm.compile_species(cat, 'numpy')
        mean_days = cat.life_expectancy_days('grass')
        n = 500
        call = Call(cat, ['grass'] * n, np.arange(1, n + 1), np.full(n, 5.0 * mean_days))   # five times the MEAN life
        out = call.run(t)
        np.testing.assert_array_equal(out['p_age'], np.zeros(n))
        self.assertEqual(int(out['cause'].sum()), 0)
        self.assertEqual(float(out['loss_fraction'].max()), 0.0)

    def test_probability_one_always_kills_and_half_is_half(self):
        cat = load_cat(only_age(bg_grass=1e9, weibull=False))
        t = dm.compile_species(cat, 'numpy')
        n = 300
        out = Call(cat, ['grass'] * n, np.arange(1, n + 1), np.full(n, 1000.0)).run(t)
        np.testing.assert_array_equal(out['p_age'], np.ones(n))
        self.assertTrue(np.all(out['cause'] == dm.CAUSE_AGE))
        cat = load_cat(only_age(bg_grass=math.log(2.0), weibull=False))     # one 365-day year: dH = ln 2 -> p = 1/2
        t = dm.compile_species(cat, 'numpy')
        n = 20000
        out = Call(cat, ['grass'] * n, np.arange(1, n + 1), np.full(n, 1000.0)).run(t)
        self.assertAlmostEqual(float(out['p_age'][0]), 0.5, places=14)
        self.assertLess(abs(float((out['cause'] == dm.CAUSE_AGE).mean()) - 0.5), 0.02)

    def test_period_scales_the_background_hazard(self):
        cat = load_cat(only_age(bg_grass=0.2, weibull=False))
        t = dm.compile_species(cat, 'numpy')
        call = Call(cat, ['grass'], [1], [3000.0])
        p1 = float(call.run(t, period=365.0)['p_age'][0])
        p2 = float(call.run(t, period=730.0)['p_age'][0])
        self.assertAlmostEqual(p1, -math.expm1(-0.2), places=14)
        self.assertAlmostEqual(p2, -math.expm1(-0.4), places=14)

    def test_overflowing_hazard_is_refused_without_a_result(self):
        cat = load_cat(lambda d: d['species']['grass']['mortality']['traits']['age_hazard'].update(
            weibull_shape=400.0, weibull_scale_years=1e-3))
        t = dm.compile_species(cat, 'numpy')
        with self.assertRaises(dm.DemographyError):
            Call(cat, ['grass'], [1], [3000.0]).run(t)


class StressCounter(unittest.TestCase):
    def setUp(self):
        self.cat = load_cat(only_stress(p_stress=0.0, loss=0.25))
        self.tab = dm.compile_species(self.cat, 'numpy')

    def scenario(self):
        #           A exempt   B dormant  C low->stressed  D healthy  E healthy@0  F low@cap  G low first year of stress
        age = [300.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
        req = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        act = [0.0, 0.0, 0.2, 0.9, 0.9, 0.1, 0.2]
        st = [1, 1, 1, 3, 0, 4, 0]
        return Call(self.cat, ['grass'] * 7, range(1, 8), age, stress=st, req=req, act=act)

    def test_first_year_dormant_low_healthy_cap_and_recovery_rules(self):
        out = self.scenario().run(self.tab)
        np.testing.assert_array_equal(out['flag'], [dm.FLAG_EXEMPT, dm.FLAG_NOT_EVALUATED, dm.FLAG_LOW, dm.FLAG_HEALTHY,
                                                    dm.FLAG_HEALTHY, dm.FLAG_LOW, dm.FLAG_LOW])
        # exempt and dormant years leave the counter UNCHANGED; low +1 (cap 4); healthy -1 (decrement, never reset)
        np.testing.assert_array_equal(out['stress_new'], [1, 1, 2, 2, 0, 4, 1])
        # response only in a LOW year with counter >= limit (2): C and F; G (counter 1) and the healthy plants are not
        np.testing.assert_array_equal(out['loss_fraction'], [0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0])
        np.testing.assert_array_equal(out['cause'], np.zeros(7, np.int8))
        s = out['summary']
        self.assertEqual(int(s['flag_counts'][self.cat.index('grass'), dm.FLAG_LOW]), 3)
        self.assertEqual(int(s['n_dieback'][self.cat.index('grass')]), 2)

    def test_stressed_year_mortality_probability_one_kills_without_dieback(self):
        cat = load_cat(only_stress(p_stress=1.0, loss=0.25))
        t = dm.compile_species(cat, 'numpy')
        call = self.scenario()
        call.cat = cat
        out = call.run(t)
        np.testing.assert_array_equal(out['cause'], [0, 0, dm.CAUSE_STRESS, 0, 0, dm.CAUSE_STRESS, 0])
        np.testing.assert_array_equal(out['loss_fraction'], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])   # dead: B / E kept
        cat0 = load_cat(only_stress(p_stress=0.0, loss=0.25))
        out0 = Call(cat0, ['grass'] * 7, range(1, 8), call.age, stress=call.stress, req=call.req, act=call.act).run(
            dm.compile_species(cat0, 'numpy'))
        self.assertEqual(int((out0['cause'] != 0).sum()), 0)                  # probability 0: nobody dies

    def test_repeated_bad_years_then_recovery_with_persistent_counter(self):
        n = 1
        stress = np.zeros(n, np.int32)
        history = []
        plan = [(1.0, 0.1)] * 5 + [(1.0, 0.9)] * 6          # five bad years, then six healthy years
        for yr, (req, act) in enumerate(plan):
            out = Call(self.cat, ['grass'], [1], [2000.0 + 365.0 * yr], stress=stress, req=[req], act=[act]).run(
                self.tab, counter=yr)
            stress = out['stress_new']
            history.append((int(stress[0]), float(out['loss_fraction'][0])))
        self.assertEqual([h[0] for h in history], [1, 2, 3, 4, 4, 3, 2, 1, 0, 0, 0])      # cap 4, then decrement
        self.assertEqual([h[1] > 0.0 for h in history], [False, True, True, True, True] + [False] * 6)

    def test_growth_ratio_metric_does_not_mistake_maturity_for_stress(self):
        cat = load_cat(only_stress(p_stress=0.0, loss=0.1, species=('shrub',)))
        t = dm.compile_species(cat, 'numpy')
        names = ['shrub'] * 3
        # mature (planned growth 0) with poor support, planned growth with small actual growth, healthy growth
        out = Call(cat, names, [1, 2, 3], [4000.0] * 3, stress=[2, 2, 2], req=[1.0] * 3, act=[0.0] * 3,
                   plan=[0.0, 1.0, 1.0], gain=[0.0, 0.1, 0.9]).run(t)
        np.testing.assert_array_equal(out['flag'], [dm.FLAG_NOT_EVALUATED, dm.FLAG_LOW, dm.FLAG_HEALTHY])
        np.testing.assert_array_equal(out['stress_new'], [2, 3, 1])
        np.testing.assert_array_equal(out['loss_fraction'], [0.0, 0.1, 0.0])       # limit 3 reached only by the low plant

    def test_species_without_stress_law_is_not_evaluated(self):
        cat = load_cat(only_stress(p_stress=0.0, loss=0.25, species=('grass',)))      # shrub stress = null
        t = dm.compile_species(cat, 'numpy')
        out = Call(cat, ['shrub'], [1], [4000.0], stress=[0], req=[1.0], act=[0.0]).run(t)
        self.assertEqual(int(out['flag'][0]), dm.FLAG_NOT_EVALUATED)
        self.assertEqual(int(out['stress_new'][0]), 0)


class Invariance(unittest.TestCase):
    def setUp(self):
        self.cat = load_cat(lambda d: (only_age(bg_grass=math.log(2.0), weibull=True)(d)))
        self.tab = dm.compile_species(self.cat, 'numpy')

    def test_decisions_are_independent_of_row_order_and_repeatable(self):
        n = 400
        names = ['grass' if i % 2 else 'shrub' for i in range(n)]
        ids = np.arange(100, 100 + n)
        age = 800.0 + 3.0 * np.arange(n)
        base = Call(self.cat, names, ids, age).run(self.tab, counter=3)
        again = Call(self.cat, names, ids, age).run(self.tab, counter=3)
        np.testing.assert_array_equal(base['cause'], again['cause'])         # same (seed, counter): same draws
        perm = np.random.default_rng(9).permutation(n)
        got = Call(self.cat, [names[i] for i in perm], ids[perm], age[perm]).run(self.tab, counter=3)
        np.testing.assert_array_equal(got['cause'], base['cause'][perm])
        np.testing.assert_array_equal(got['p_age'], base['p_age'][perm])
        nxt = Call(self.cat, names, ids, age).run(self.tab, counter=4)
        self.assertFalse(np.array_equal(base['cause'], nxt['cause']))        # a new decision uses new draws
        self.assertGreater(int((base['cause'] != 0).sum()), 0)

    def test_independent_of_species_declaration_order(self):
        def reverse(d):
            only_age(bg_grass=math.log(2.0), weibull=True)(d)
            d['species'] = dict(reversed(list(d['species'].items())))
        cat2 = load_cat(reverse)
        self.assertEqual(cat2.hash, self.cat.hash)
        t2 = dm.compile_species(cat2, 'numpy')
        n = 200
        names = ['grass' if i % 3 else 'shrub' for i in range(n)]
        ids = np.arange(1, n + 1)
        age = np.full(n, 900.0)
        a = Call(self.cat, names, ids, age).run(self.tab).get('cause')
        b = Call(cat2, names, ids, age).run(t2).get('cause')
        np.testing.assert_array_equal(a, b)

    def test_budget_refusal_before_allocation_and_linear_memory(self):
        call = Call(self.cat, ['grass'] * 10, np.arange(1, 11), np.full(10, 900.0))
        with self.assertRaises(dm.DemographyBudgetError):
            call.run(self.tab, max_bytes=100)
        e1, e2 = dm.memory_estimate(1000, 2, 'numpy'), dm.memory_estimate(2000, 2, 'numpy')
        self.assertEqual(e2['decision_bytes'], 2 * e1['decision_bytes'])
        self.assertEqual(e2['persistent_state_bytes'], 2 * e1['persistent_state_bytes'])
        self.assertEqual(e1['persistent_state_bytes'], dm.PERSISTENT_BYTES_PER_LIVE * 1000)
        self.assertGreater(dm.memory_estimate(1000, 2, 'cuda')['decision_bytes'], e1['decision_bytes'])
        with self.assertRaises(dm.DemographyError):
            call.run(self.tab, period=0.0)
        with self.assertRaises(dm.DemographyError):
            call.run(self.tab, seed=-1)
        with self.assertRaises(dm.DemographyError):
            call.run(self.tab, max_bytes=True)

    def test_scientific_content_is_validated_before_any_kernel(self):
        def call(**kw):
            c = Call(self.cat, ['grass'] * 4, np.arange(1, 5), np.full(4, 900.0), req=np.ones(4), act=np.ones(4),
                     plan=np.ones(4), gain=np.ones(4))
            for k, v in kw.items():
                setattr(c, k, v)
            return c
        call().run(self.tab)                                  # baseline valid; actual support MAY exceed required
        call(act=np.full(4, 5.0)).run(self.tab)
        bad = {
            'nan_required': dict(req=np.array([1.0, np.nan, 1.0, 1.0])),
            'inf_actual': dict(act=np.array([1.0, np.inf, 1.0, 1.0])),
            'negative_actual': dict(act=np.array([1.0, -0.1, 1.0, 1.0])),
            'negative_plan': dict(plan=np.array([1.0, 1.0, -1.0, 1.0])),
            'gain_above_plan': dict(gain=np.array([2.0, 1.0, 1.0, 1.0]), plan=np.array([1.0, 1.0, 1.0, 1.0])),
            'negative_counter': dict(stress=np.array([0, -1, 0, 0], dtype=np.int32)),
            'counter_above_cap': dict(stress=np.array([0, 5, 0, 0], dtype=np.int32)),          # grass cap is 4
            'negative_species': dict(sidx=np.array([0, -1, 0, 0], dtype=np.int32)),
            'species_out_of_range': dict(sidx=np.array([0, 2, 0, 0], dtype=np.int32)),
            'duplicate_ids': dict(ids=np.array([1, 1, 2, 3], dtype=np.int64)),
            'float_ids': dict(ids=np.array([0.0, 1.0, 2.0, 3.0])),
            'bool_ids': dict(ids=np.array([True, False, True, False])),
            'nan_age': dict(age=np.array([1.0, np.nan, 1.0, 1.0])),
        }
        for name, kw in bad.items():
            with self.assertRaises(dm.DemographyError, msg=name):
                call(**kw).run(self.tab)
        # tolerance: a sum-rounding sized excess of actual over planned growth is accepted (no repair, no clamp)
        out = call(gain=np.full(4, 1.0 + 1e-12)).run(self.tab)
        self.assertEqual(out['flag'].shape, (4,))

    SIGNED = [-2 ** 63, -1, 0, 2 ** 63 - 1]

    def _signed_call(self, ids):
        c = Call(self.cat, ['grass'] * 4, np.arange(1, 5), np.full(4, 900.0))
        c.ids = ids                           # keep the supplied dtype (Call would coerce to int64)
        return c

    def test_signed_int64_ids_are_valid_and_match_the_scalar_hash_reference(self):
        ids = np.array(self.SIGNED, dtype=np.int64)
        out = self._signed_call(ids).run(self.tab, counter=2)          # zero / negative / endpoint IDs are NOT refused
        self.assertEqual(out['flag'].shape, (4,))
        u = dm.keyed_uniform(7, 2, dm.STREAM_AGE, ids)
        ref = np.array([dm.keyed_uniform_py(7, 2, dm.STREAM_AGE, int(i)) for i in self.SIGNED])
        np.testing.assert_array_equal(u, ref)                           # modulo 2**64, no abs / truncation
        self.assertEqual(len(set(u.tolist())), 4)
        # decisions follow the keyed draw of each ID: p = 1/2 background makes the draw observable
        cat = load_cat(only_age(bg_grass=math.log(2.0), weibull=False))
        t = dm.compile_species(cat, 'numpy')
        out = Call(cat, ['grass'] * 4, ids, np.full(4, 900.0)).run(t, counter=2)
        np.testing.assert_array_equal(out['cause'] == dm.CAUSE_AGE, ref < out['p_age'])
        perm = np.array([2, 0, 3, 1])
        got = Call(cat, ['grass'] * 4, ids[perm], np.full(4, 900.0)).run(t, counter=2)
        np.testing.assert_array_equal(got['cause'], out['cause'][perm])   # row permutation invariance

    def test_non_integer_or_duplicate_signed_ids_still_refuse_before_a_kernel(self):
        for ids in (np.array([True, False, True, False]), np.array([0.0, 1.0, 2.0, 3.0]),
                    np.array([-1, -1, 0, 5], dtype=np.int64), np.array([0, 0, 1, 2], dtype=np.int64),
                    np.array([-2 ** 63, 5, 5, 2 ** 63 - 1], dtype=np.int64)):
            with self.assertRaises(dm.DemographyError):
                self._signed_call(ids).run(self.tab)

    def test_compile_admission_precedes_allocation_and_reserve_is_counted(self):
        with self.assertRaises(dm.DemographyBudgetError):
            dm.compile_species(self.cat, 'numpy', max_bytes=10)
        with self.assertRaises(dm.DemographyBudgetError):
            dm.compile_species(self.cat, 'numpy', max_bytes=1000, extra_reserve_bytes=1000)
        with self.assertRaises(dm.DemographyError):
            dm.compile_species(self.cat, 'numpy', max_bytes=1.5)
        dm.compile_species(self.cat, 'numpy', max_bytes=10 ** 6, extra_reserve_bytes=5)

    def test_full_dieback_endpoint_is_refused_at_configuration(self):
        # biomass_loss_fraction must lie in [0, 1): complete death belongs to mortality_probability (no zero-biomass live plant)
        with self.assertRaises(dm.DemographyError):
            dm.compile_species(load_cat(only_stress(p_stress=0.0, loss=1.0)), 'numpy')
        dm.compile_species(load_cat(only_stress(p_stress=0.0, loss=0.0)), 'numpy')
        dm.compile_species(load_cat(only_stress(p_stress=1.0, loss=0.9999999999999999)), 'numpy')

    def test_array_contracts_are_strict(self):
        call = Call(self.cat, ['grass'] * 4, np.arange(1, 5), np.full(4, 900.0))
        call.stress = call.stress.astype(np.int64)
        with self.assertRaises(dm.DemographyError):
            call.run(self.tab)
        call = Call(self.cat, ['grass'] * 4, np.arange(1, 5), np.full(4, 900.0))
        call.ids = call.ids.astype(np.int32)
        with self.assertRaises(dm.DemographyError):
            call.run(self.tab)
        call = Call(self.cat, ['grass'] * 4, np.arange(1, 5), np.full(4, 900.0))
        call.age[0] = -1.0
        with self.assertRaises(dm.DemographyError):
            call.run(self.tab)


@unittest.skipUnless(NUMBA, 'numba backend unavailable')
class NumbaParity(unittest.TestCase):
    def test_numba_matches_numpy_reference_including_keyed_draws(self):
        cat = load_cat(lambda d: d['species']['grass']['mortality']['traits']['stress'].update(
            mortality_probability=0.5, biomass_loss_fraction=0.2))
        t_np, t_nb = dm.compile_species(cat, 'numpy'), dm.compile_species(cat, 'numba')
        rng = np.random.default_rng(21)
        n = 2000
        names = ['grass' if i % 2 else 'shrub' for i in range(n)]
        call = Call(cat, names, np.arange(5, 5 + n), rng.uniform(100.0, 20000.0, n), stress=rng.integers(0, 4, n),
                    req=rng.uniform(0.0, 2.0, n) * (rng.uniform(size=n) > 0.1), act=rng.uniform(0.0, 2.0, n),
                    plan=rng.uniform(0.0, 2.0, n) * (rng.uniform(size=n) > 0.2), gain=None)
        call.gain = call.plan * rng.uniform(0.0, 1.0, n)                   # actual growth <= planned growth
        a, b = call.run(t_np, counter=2), call.run(t_nb, counter=2)
        np.testing.assert_allclose(a['p_age'], b['p_age'], rtol=1e-12, atol=1e-300)
        # probabilities may differ in the last bit between pow implementations: a death decision can only differ for
        # a draw within ~1e-15 of p (practically never); compare everything else where the decisions agree
        same = a['cause'] == b['cause']
        self.assertGreaterEqual(int(same.sum()), n - 2)
        np.testing.assert_array_equal(a['flag'][same], b['flag'][same])
        np.testing.assert_array_equal(a['stress_new'][same], b['stress_new'][same])
        np.testing.assert_array_equal(a['loss_fraction'][same], b['loss_fraction'][same])
        self.assertGreater(int((a['cause'] == dm.CAUSE_STRESS).sum()), 0)       # the keyed stress draw is exercised
        self.assertGreater(int((a['cause'] == dm.CAUSE_AGE).sum()), 0)


SIGNED_IDS = [-2 ** 63, -1, 0, 2 ** 63 - 1]


def _signed_cat():
    return load_cat(only_age(bg_grass=math.log(2.0), weibull=False))


@unittest.skipUnless(NUMBA, 'numba backend unavailable')
class NumbaSignedIds(unittest.TestCase):
    def test_signed_ids_match_the_scalar_reference_and_permutations(self):
        cat = _signed_cat()
        t_np, t_nb = dm.compile_species(cat, 'numpy'), dm.compile_species(cat, 'numba')
        ids = np.array(SIGNED_IDS, dtype=np.int64)
        ref = np.array([dm.keyed_uniform_py(7, 3, dm.STREAM_AGE, i) for i in SIGNED_IDS])
        for perm in (np.arange(4), np.array([3, 1, 0, 2])):
            c = Call(cat, ['grass'] * 4, ids[perm], np.full(4, 900.0))
            a, b = c.run(t_np, counter=3), c.run(t_nb, counter=3)
            np.testing.assert_array_equal(a['cause'], b['cause'])
            np.testing.assert_array_equal(b['cause'] == dm.CAUSE_AGE, ref[perm] < b['p_age'])


@unittest.skipUnless(CUDA, 'CUDA backend unavailable')
class CudaSignedIds(unittest.TestCase):
    def test_signed_ids_match_the_scalar_reference_on_the_device(self):
        import cupy as cp
        dev = int(cp.cuda.Device().id)
        cat = _signed_cat()
        t_cu = dm.compile_species(cat, 'cuda', device=dev)
        ids = np.array(SIGNED_IDS, dtype=np.int64)
        ref = np.array([dm.keyed_uniform_py(7, 3, dm.STREAM_AGE, i) for i in SIGNED_IDS])
        for perm in (np.arange(4), np.array([2, 0, 3, 1])):
            c = Call(cat, ['grass'] * 4, ids[perm], np.full(4, 900.0))
            out = dm.decide(t_cu, species_index=cp.asarray(c.sidx), plant_ids=c.ids, age_end_days=c.age,
                            stress=cp.asarray(c.stress), required_support=cp.asarray(c.req),
                            actual_support=cp.asarray(c.act), planned_growth=cp.asarray(c.plan),
                            actual_growth=cp.asarray(c.gain), period_days=365.0, seed=7, counter=3, max_bytes=BIG)
            p = cp.asnumpy(out['p_age'])
            np.testing.assert_array_equal(cp.asnumpy(out['cause']) == dm.CAUSE_AGE, ref[perm] < p)


@unittest.skipUnless(CUDA, 'CUDA backend unavailable')
class CudaParity(unittest.TestCase):
    def test_cuda_matches_numpy_reference_on_the_explicit_device(self):
        import cupy as cp
        dev = int(cp.cuda.Device().id)
        cat = load_cat()
        t_np, t_cu = dm.compile_species(cat, 'numpy'), dm.compile_species(cat, 'cuda', device=dev)
        n = 1000
        rng = np.random.default_rng(22)
        names = ['grass' if i % 2 else 'shrub' for i in range(n)]
        c = Call(cat, names, np.arange(1, n + 1), rng.uniform(100.0, 9000.0, n), stress=rng.integers(0, 4, n),
                 req=rng.uniform(0.1, 2.0, n), act=rng.uniform(0.0, 2.0, n), plan=rng.uniform(0.0, 1.0, n))
        c.gain = c.plan * rng.uniform(0.0, 1.0, n)
        ref = c.run(t_np, counter=1)
        out = dm.decide(t_cu, species_index=cp.asarray(c.sidx), plant_ids=c.ids, age_end_days=c.age,
                        stress=cp.asarray(c.stress), required_support=cp.asarray(c.req), actual_support=cp.asarray(c.act),
                        planned_growth=cp.asarray(c.plan), actual_growth=cp.asarray(c.gain), period_days=365.0, seed=7,
                        counter=1, max_bytes=BIG)
        np.testing.assert_array_equal(cp.asnumpy(out['flag']), ref['flag'])
        np.testing.assert_array_equal(cp.asnumpy(out['stress_new']), ref['stress_new'])
        np.testing.assert_allclose(cp.asnumpy(out['p_age']), ref['p_age'], rtol=1e-12, atol=1e-300)
        with self.assertRaises(dm.DemographyBackendError):
            dm.compile_species(cat, 'cuda', device=dev + 1000)
        with self.assertRaises(dm.DemographyError):                    # host arrays handed to a cuda table: refused
            dm.decide(t_cu, species_index=c.sidx, plant_ids=c.ids, age_end_days=c.age, stress=c.stress,
                      required_support=c.req, actual_support=c.act, planned_growth=c.plan, actual_growth=c.gain,
                      period_days=365.0, seed=7, counter=1, max_bytes=BIG)


if __name__ == '__main__':
    unittest.main()
