'''LEAF Phase 2 species-specific mechanical response tests (unittest): leaf.disturbance, pure state level.

Run from the candidate root:  python -m unittest tests.test_disturbance -v

* No native SYRUP is needed here: the response law is evaluated on canonical `PlantState` objects through
  `disturbance.respond_state` (which never mutates its input). The transactional session boundary is covered in
  tests/test_interevent.py.
* All coefficients come from the UNCALIBRATED fixture catalog examples/phase2/species.json (species `grass`: burial
  sensitive, shallow anchorage; `shrub`: burial tolerant, firm anchorage). Expected values are recomputed in the tests
  from the documented formulas and the shared catalog allometry (never from magic constants of the module).
* numba / CUDA cases SKIP when the backend is unavailable. Written WITHOUT the ability to execute them (file tools
  only): they are NOT RUN by the author; Codex records the real runs.
'''
from __future__ import annotations

import copy
import json
import math
import pathlib
import unittest
import unittest.mock

import numpy as np

from leaf import disturbance, events, state, traits

ROOT = pathlib.Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / 'examples' / 'phase2' / 'species.json'
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
T0, T1 = 4.0, 5.0       # event interval (model days); the state clock is T1


def doc_p2():
    with open(CATALOG_PATH, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def load_cat(modify=None):
    doc = doc_p2()
    if modify is not None:
        modify(doc)
    return traits.SpeciesCatalog(doc)


def make_state(cat, spec, *, time_days=T1, nx=4):
    '''spec rows: (id, species, column, B, E, status); plants sit in row 0, cell centres, establishment ground 100.'''
    rec = [dict(plant_id=pid, species=sp, x_m=col + 0.5, y_m=0.5, establishment_ground_m=100.0, agb_dry_kg=B,
                seasonal_agb_dry_kg=E, age_days=300.0 + pid, status=status) for pid, sp, col, B, E, status in spec]
    return state.PlantState.from_records(cat, state.Domain(nx, nx, 1.0), rec, time_days=time_days)


def surface(deltas, nx=4):
    s = np.full((nx, nx), 100.0)
    for col, d in deltas.items():
        s[0, col] = 100.0 + d
    return s


def ev(ids, source='maple_wind', t0=T0, t1=T1, nodose=False, **kw):
    '''A nominal wind event under an applicable abrasion law must SUPPLY the dose explicitly: when a test exercises burial /
    erosion alone it passes an explicit ZERO dose in the declared unit. `nodose=True` builds the event without any dose
    (the dedicated missing-dose / flux-is-not-dose tests).'''
    if source == 'maple_wind' and not nodose and 'abrasion_dose' not in kw:
        kw['abrasion_dose'] = np.zeros(len(ids))
        kw['abrasion_unit'] = 'kg m-1'
    return events.SurfaceEventDiagnostics(ids, t0, t1, source, **kw)


def by_id(event, arr):
    return {int(i): arr[k] for k, i in enumerate(event.plant_id)}


class Base(unittest.TestCase):
    def setUp(self):
        self.cat = load_cat()
        self.tab = disturbance.compile_species(self.cat, 'numpy')

    def respond(self, st, event, surf, table=None):
        return disturbance.respond_state(st, self.cat, event, surf, table=table or self.tab, backend='numpy')

    def height(self, sp, B):
        return float(self.cat.height_from_biomass(sp, B))


class LawParsing(Base):
    def test_inactive_catalogs_hold_no_table_and_unimplemented_is_inactive(self):
        p1 = traits.load_catalog(ROOT / 'examples' / 'phase1' / 'species.json')
        t = disturbance.compile_species(p1, 'numpy')
        self.assertFalse(t.any_active)
        self.assertEqual(t.resident_bytes(), 0)
        cat = load_cat(lambda d: d['species']['grass'].update(disturbance={'law': 'unimplemented', 'fixture': {'x': 1}}))
        t = disturbance.compile_species(cat, 'numpy')
        self.assertEqual(t.active, (False, True))              # grass inactive, shrub active (sorted ids)
        cat = load_cat(lambda d: d['species']['grass'].update(disturbance=None))
        self.assertEqual(disturbance.compile_species(cat, 'numpy').active, (False, True))

    def test_unknown_and_malformed_active_laws_refuse_not_disable(self):
        def mod_law(d):
            d['species']['grass']['disturbance']['law'] = 'burial_v9'

        def mod_missing(d):
            del d['species']['grass']['disturbance']['traits']['burial']['curve']

        def mod_unknown_key(d):
            d['species']['grass']['disturbance']['traits']['burial']['surprise'] = 1.0

        def mod_f1(d):
            d['species']['grass']['disturbance']['traits']['burial']['max_injury_fraction'] = 1.0

        def mod_order(d):
            d['species']['grass']['disturbance']['traits']['burial']['injury_saturation'] = 0.01

        def mod_curve(d):
            d['species']['grass']['disturbance']['traits']['exposure']['curve'] = 'cubic'

        def mod_unit(d):
            d['species']['grass']['disturbance']['traits']['abrasion']['dose_unit'] = ''

        def mod_src(d):
            d['species']['grass']['disturbance']['traits']['abrasion']['applies_to_sources'] = ['wind']

        def mod_leth(d):
            d['species']['grass']['disturbance']['traits']['exposure']['lethal_depth_m'] = 0.0

        def mod_section(d):
            d['species']['grass']['disturbance']['extra'] = 1

        def mod_nan(d):
            d['species']['grass']['disturbance']['traits']['burial']['injury_threshold'] = True
        for m in (mod_law, mod_missing, mod_unknown_key, mod_f1, mod_order, mod_curve, mod_unit, mod_src, mod_leth,
                  mod_section, mod_nan):
            with self.assertRaises(disturbance.DisturbanceError, msg=m.__name__):
                disturbance.compile_species(load_cat(m), 'numpy')

    def test_species_table_holds_catalog_coefficients_only(self):
        c = self.tab.coefficients('grass')
        doc = doc_p2()['species']['grass']['disturbance']['traits']
        self.assertEqual(c['b_f'], doc['burial']['max_injury_fraction'])
        self.assertEqual(c['e_dm'], doc['exposure']['dislodgement_depth_m'])
        self.assertTrue(math.isinf(c['b_lm']))                 # null lethal depth = explicitly off
        self.assertEqual(self.tab.coefficients('shrub')['b_lm'], 1.5)
        self.assertEqual(self.tab.info[0]['dose_unit'], 'kg m-1')

    def test_backend_failures_are_explicit(self):
        with self.assertRaises(disturbance.DisturbanceError):
            disturbance.compile_species(self.cat, 'cupy')
        if not CUDA:
            with self.assertRaises(disturbance.DisturbanceError):
                disturbance.compile_species(self.cat, 'cuda')             # cuda needs the explicit device (and CuPy)
        if CUDA:
            with self.assertRaises(disturbance.DisturbanceBackendError):
                disturbance.compile_species(self.cat, 'cuda')             # no device given: refused, never guessed


class Contrasts(Base):
    def test_burial_species_contrast_zero_partial_and_lethal(self):
        st = make_state(self.cat, [(1, 'grass', 0, 0.3, 0.0, 'live'), (2, 'shrub', 1, 5.0, 0.0, 'live')])
        # +0.0625 m: grass injured (fraction of height above threshold), shrub below its 0.1 m threshold: EXACT zero
        cand, res = self.respond(st, ev([1, 2], deposition_depth_m=[0.0625, 0.0625]), surface({0: 0.0625, 1: 0.0625}))
        b = by_id(ev([1, 2]), res['B_new'])
        h = self.height('grass', 0.3)
        f = 0.6 * ((0.0625 / h - 0.05) / (0.5 - 0.05))
        self.assertAlmostEqual(b[1], 0.3 * (1.0 - f), places=13)
        self.assertEqual(b[2], 5.0)
        self.assertEqual(cand.status_names().tolist(), ['live', 'live'])
        self.assertEqual(float(cand.agb_dry_kg[1]), 5.0)
        # +0.5 m: grass post-event burial / height >= lethal fraction -> dead_standing, B unchanged; shrub partial injury
        cand, res = self.respond(st, ev([1, 2], deposition_depth_m=[0.5, 0.5]), surface({0: 0.5, 1: 0.5}))
        self.assertEqual(cand.status_names().tolist(), ['dead_standing', 'live'])
        self.assertEqual(float(cand.agb_dry_kg[0]), 0.3)                          # dead plants keep their B
        fs = 0.3 * ((0.5 - 0.1) / (0.8 - 0.1))
        self.assertAlmostEqual(float(cand.agb_dry_kg[1]), 5.0 * (1.0 - fs), places=12)
        self.assertEqual(res['outcome'].tolist(), [disturbance.OUT_DEAD, disturbance.OUT_INJURED])
        self.assertEqual(res['lethal_bits'][0] & disturbance.LETHAL_BURIAL, disturbance.LETHAL_BURIAL)

    def test_net_versus_gross_and_signed_net_change(self):
        st = make_state(self.cat, [(1, 'grass', 0, 0.3, 0.0, 'live'), (2, 'shrub', 1, 5.0, 0.0, 'live')])
        # equal erosion and deposition: signed NET change zero, surface unchanged
        e = ev([1, 2], erosion_depth_m=[0.125, 0.125], deposition_depth_m=[0.125, 0.125])
        np.testing.assert_array_equal(e.net_surface_delta_m, [0.0, 0.0])
        cand, res = self.respond(st, e, surface({}))
        b = by_id(e, res['B_new'])
        self.assertEqual(b[1], 0.3)                                  # grass uses NET basis: no injury
        g = (0.125 - 0.05) / (0.5 - 0.05)
        self.assertAlmostEqual(b[2], 5.0 * (1.0 - 0.3 * g * g * (3.0 - 2.0 * g)), places=12)   # shrub exposure: GROSS
        np.testing.assert_array_equal(res['burial_increment_m'], [0.0, 0.0])
        # a deposition that only refills an erosion hole: net burial increase is smaller than the gross deposition
        e2 = ev([1], deposition_depth_m=[0.25])
        cand2, res2 = self.respond(make_state(self.cat, [(1, 'grass', 0, 0.3, 0.0, 'live')]), e2, surface({0: 0.125}))
        self.assertEqual(float(res2['burial_increment_m'][0]), 0.125)      # delta_before = 0.125 - 0.25 < 0 -> burial 0 -> 0.125
        self.assertEqual(float(res2['burial_after_m'][0]), 0.125)

    def test_exposure_dislodgement_lethal_and_anchors_never_move(self):
        st = make_state(self.cat, [(1, 'grass', 0, 0.3, 0.1, 'live'), (2, 'shrub', 1, 5.0, 0.5, 'live'),
                                   (3, 'shrub', 2, 4.0, 0.0, 'live')])
        e = ev([1, 2, 3], erosion_depth_m=[0.3125, 0.75, 0.25])
        cand, res = self.respond(st, e, surface({0: -0.3125, 1: -0.75, 2: -0.25}))
        self.assertEqual(cand.status_names().tolist(), ['removed', 'dead_standing', 'live'])
        self.assertEqual((float(cand.agb_dry_kg[0]), float(cand.seasonal_agb_dry_kg[0])), (0.0, 0.0))   # exported, zeroed
        self.assertEqual(float(res['removed_kg']['dislodgement'][0]), 0.3)
        self.assertEqual(float(res['exported_E_kg'][0]), 0.1)
        self.assertEqual((float(cand.agb_dry_kg[1]), float(cand.seasonal_agb_dry_kg[1])), (5.0, 0.5))   # standing dead
        g = (0.25 - 0.05) / (0.5 - 0.05)
        self.assertAlmostEqual(float(cand.agb_dry_kg[2]), 4.0 * (1.0 - 0.3 * g * g * (3.0 - 2.0 * g)), places=12)
        # IDs, X / Y, FIXED establishment anchors and ages are untouched; dead / removed records remain
        for name in ('ids', 'x_m', 'y_m', 'establishment_ground_m', 'age_days', 'species_index'):
            np.testing.assert_array_equal(getattr(cand, name), getattr(st, name), err_msg=name)
        self.assertEqual(cand.n, st.n)
        d = res['diagnostics']
        self.assertEqual((d['n_removed'], d['n_dead_standing'], d['n_injured']), (1, 1, 1))

    def test_historical_burial_is_not_reapplied_as_new_damage(self):
        st = make_state(self.cat, [(1, 'grass', 0, 0.3, 0.0, 'live')])
        surf = surface({0: 0.125})                                        # already buried 0.125 m before this event
        quiet = ev([1])                                                   # no deposition / erosion in THIS event
        cand = st
        for _ in range(3):
            cand, res = self.respond(cand, quiet, surf)
            self.assertEqual(float(cand.agb_dry_kg[0]), 0.3)
            self.assertEqual(float(res['burial_increment_m'][0]), 0.0)
            self.assertEqual(float(res['burial_after_m'][0]), 0.125)      # the absolute value is reported, not damage

    def test_simultaneous_responses_are_bounded_and_scale_E(self):
        st = make_state(self.cat, [(1, 'grass', 0, 0.3, 0.1, 'live')])
        e = ev([1], deposition_depth_m=[0.25], abrasion_dose=[5.0], abrasion_unit='kg m-1')
        cand, res = self.respond(st, e, surface({0: 0.25}))
        retained = (1.0 - 0.6) * (1.0 - 0.0) * (1.0 - 0.4)               # burial saturated, abrasion saturated
        self.assertAlmostEqual(float(cand.agb_dry_kg[0]), 0.3 * retained, places=14)
        self.assertAlmostEqual(float(cand.seasonal_agb_dry_kg[0]), 0.1 * retained, places=14)
        self.assertLessEqual(float(cand.seasonal_agb_dry_kg[0]), float(cand.agb_dry_kg[0]))
        rem = res['removed_kg']
        total = sum(float(rem[c][0]) for c in disturbance.CAUSES)
        self.assertAlmostEqual(total, 0.3 - float(cand.agb_dry_kg[0]), places=15)
        self.assertGreater(float(rem['burial'][0]), float(rem['abrasion'][0]))
        self.assertEqual(float(rem['exposure'][0]), 0.0)
        self.assertLess(total, 0.3)                                       # never more than the existing dry AGB

    def test_zero_injury_event_changes_nothing(self):
        st = make_state(self.cat, [(1, 'grass', 0, 0.3, 0.1, 'live'), (2, 'shrub', 1, 5.0, 0.0, 'live')])
        cand, res = self.respond(st, ev([1, 2]), surface({}))
        for name in state.PlantState.ARRAY_NAMES:
            np.testing.assert_array_equal(getattr(cand, name), getattr(st, name))
        self.assertEqual(res['outcome'].tolist(), [disturbance.OUT_NONE, disturbance.OUT_NONE])
        self.assertEqual(res['diagnostics']['ledger']['removed_dry_agb_kg'], 0.0)


class DoseAndSources(Base):
    def setUp(self):
        super().setUp()
        self.st = make_state(self.cat, [(1, 'grass', 0, 0.3, 0.0, 'live')])

    def test_dose_units_must_match_exactly(self):
        f = 0.4 * ((2.0 - 0.5) / (5.0 - 0.5))
        cand, res = self.respond(self.st, ev([1], abrasion_dose=[2.0], abrasion_unit='kg m-1'), surface({}))
        self.assertAlmostEqual(float(cand.agb_dry_kg[0]), 0.3 * (1.0 - f), places=14)
        with self.assertRaises(disturbance.DisturbanceError):
            self.respond(self.st, ev([1], abrasion_dose=[2.0], abrasion_unit='g m-1'), surface({}))
        with self.assertRaises(disturbance.DisturbanceError):
            self.respond(self.st, ev([1], abrasion_dose=[2.0], abrasion_unit='kg m-1 '), surface({}))

    def test_flux_is_not_a_dose_and_is_never_converted(self):
        with self.assertRaises(disturbance.DisturbanceError):      # applicable abrasion but only a flux: refused
            self.respond(self.st, ev([1], nodose=True, transport_flux=[100.0], flux_unit='kg m-1 s-1'), surface({}))
        with self.assertRaises(disturbance.DisturbanceError):      # no dose at all under an applicable abrasion law
            self.respond(self.st, ev([1], nodose=True), surface({}))
        same_dose = ev([1], abrasion_dose=[2.0], abrasion_unit='kg m-1')
        with_flux = ev([1], abrasion_dose=[2.0], abrasion_unit='kg m-1', transport_flux=[1e9], flux_unit='kg m-1 s-1')
        a = self.respond(self.st, same_dose, surface({}))[0].agb_dry_kg
        b = self.respond(self.st, with_flux, surface({}))[0].agb_dry_kg
        np.testing.assert_array_equal(a, b)                         # the huge flux changed nothing

    def test_water_events_do_not_abrade(self):
        e = ev([1], source='syrup_water', transport_flux=[100.0], flux_unit='kg m-1 s-1')
        cand, res = self.respond(self.st, e, surface({}))           # abrasion applies_to_sources = maple_wind only
        self.assertEqual(float(cand.agb_dry_kg[0]), 0.3)
        e = ev([1], source='syrup_water', abrasion_dose=[50.0], abrasion_unit='kg m-1')
        cand, _ = self.respond(self.st, e, surface({}))
        self.assertEqual(float(cand.agb_dry_kg[0]), 0.3)            # a dose from a non-applicable source is ignored
        e = ev([1], source='syrup_water', deposition_depth_m=[0.5])
        cand, _ = self.respond(self.st, e, surface({0: 0.5}))        # but water deposition buries
        self.assertEqual(cand.status_names().tolist(), ['dead_standing'])


class Guards(Base):
    def test_clock_ids_inputs_and_non_live_plants(self):
        st = make_state(self.cat, [(1, 'grass', 0, 0.3, 0.0, 'live'), (2, 'shrub', 1, 5.0, 0.0, 'dead_standing'),
                                   (3, 'grass', 2, 0.0, 0.0, 'removed')])
        with self.assertRaises(disturbance.DisturbanceError):       # the event must END at the population clock
            self.respond(st, ev([1], t0=3.0, t1=4.0, deposition_depth_m=[0.5]), surface({0: 0.5}))
        with self.assertRaises(disturbance.DisturbanceError):       # unknown ID
            self.respond(st, ev([99], deposition_depth_m=[0.5]), surface({}))
        with self.assertRaises(events.EventError):                  # duplicate IDs are refused by the event itself
            ev([1, 1], deposition_depth_m=[0.5, 0.5])
        before = st.copy()
        e = ev([1, 2, 3], deposition_depth_m=[0.5, 0.5, 0.5], erosion_depth_m=[0.0, 0.0, 0.0])
        cand, res = self.respond(st, e, surface({0: 0.5, 1: 0.5, 2: 0.5}))
        for name in state.PlantState.ARRAY_NAMES:                   # the input is never mutated
            np.testing.assert_array_equal(getattr(st, name), getattr(before, name))
        self.assertEqual(cand.status_names().tolist(), ['dead_standing', 'dead_standing', 'removed'])
        self.assertEqual(float(cand.agb_dry_kg[1]), 5.0)             # dead_standing plants do not respond again
        self.assertEqual(res['outcome'].tolist(), [disturbance.OUT_DEAD, -1, -1])
        ps = res['diagnostics']['per_species']
        self.assertEqual(ps['shrub']['n_skipped_not_live'], 1)
        np.testing.assert_array_equal(cand.age_days, st.age_days)    # no age is changed by a response

    def test_budget_refusal_happens_before_allocation(self):
        st = make_state(self.cat, [(1, 'grass', 0, 0.3, 0.0, 'live')])
        with self.assertRaises(disturbance.DisturbanceBudgetError):
            disturbance.respond_state(st, self.cat, ev([1], deposition_depth_m=[0.5]), surface({0: 0.5}), table=self.tab,
                                      max_bytes=10)
        est = disturbance.estimate_event_bytes(1, 2, 'numpy')
        self.assertEqual(est['total_bytes'], (disturbance.BYTES_PER_EVENT_PLANT_HOST + 64))
        self.assertGreater(disturbance.estimate_event_bytes(1, 2, 'cuda')['total_bytes'], est['total_bytes'])
        for bad in (True, -1, 1.5):
            with self.assertRaises(disturbance.DisturbanceError):
                disturbance.estimate_event_bytes(bad, 2, 'numpy')
        with self.assertRaises(disturbance.DisturbanceError):
            disturbance.respond_state(st, self.cat, ev([1]), surface({}), table=self.tab, max_bytes=True)

    def test_pure_entry_points_admit_before_any_population_work_or_table_compile(self):
        st = make_state(self.cat, [(1, 'grass', 0, 0.3, 0.0, 'live'), (2, 'shrub', 1, 5.0, 0.0, 'live')])
        empty = ev(np.zeros(0, dtype=np.int64))
        calls = []
        real_copy = state.PlantState.copy
        with unittest.mock.patch.object(state.PlantState, 'copy', lambda s_: (calls.append(s_.n), real_copy(s_))[1]), \
                unittest.mock.patch.object(disturbance, 'compile_species',
                                           side_effect=AssertionError('compiled before admission')), \
                unittest.mock.patch.object(state.PlantState, 'indices_of',
                                           side_effect=AssertionError('resolved before admission')):
            with self.assertRaises(disturbance.DisturbanceBudgetError):
                disturbance.respond_state(st, self.cat, empty, surface({}), table=None, max_bytes=1)
            with self.assertRaises(disturbance.DisturbanceBudgetError):
                disturbance.respond_state(st, self.cat, empty, surface({}), table=self.tab, max_bytes=100,
                                          extra_reserve_bytes=0)
        self.assertEqual(calls, [])
        with self.assertRaises(disturbance.DisturbanceBudgetError):       # table admission before allocation / upload
            disturbance.compile_species(self.cat, 'numpy', max_bytes=10)
        with self.assertRaises(disturbance.DisturbanceBudgetError):       # the caller's reserve is counted
            disturbance.compile_species(self.cat, 'numpy', max_bytes=1000, extra_reserve_bytes=1000)
        with self.assertRaises(disturbance.DisturbanceError):
            disturbance.compile_species(self.cat, 'numpy', max_bytes=True)
        disturbance.compile_species(self.cat, 'numpy', max_bytes=10 ** 6, extra_reserve_bytes=10)

    def test_ledger_closes_per_cause(self):
        rng = np.random.default_rng(11)
        spec = [(i + 1, 'grass' if i % 2 else 'shrub', i % 8, float(rng.uniform(0.1, 0.5)), 0.0, 'live')
                for i in range(16)]
        st = make_state(self.cat, spec, nx=8)
        deltas = {c: float(c % 6) / 16.0 for c in range(8)}       # exact binary fractions, grass columns 1, 3, 5, 7 > 0
        # two plants per cell column: same cell, same surface
        e = ev(list(range(1, 17)), deposition_depth_m=[deltas[i % 8] for i in range(16)])
        _, res = self.respond(st, e, surface(deltas, nx=8))
        led = res['diagnostics']['ledger']
        self.assertLessEqual(abs(led['residual_kg']), led['bound_kg'] + 1e-300)
        self.assertLessEqual(abs(led['attribution_residual_kg']), 1e-14)
        self.assertAlmostEqual(led['dry_agb_before_kg'] - led['dry_agb_after_kg'], led['removed_dry_agb_kg'], places=14)
        self.assertGreater(led['removed_dry_agb_kg'], 0.0)

    def test_row_permutation_invariance(self):
        spec = [(i + 1, 'grass' if i % 3 else 'shrub', i, 0.2 + 0.05 * i, 0.0, 'live') for i in range(8)]
        deltas = {c: (c % 4) / 8.0 for c in range(8)}
        e = ev(list(range(1, 9)), deposition_depth_m=[deltas[i] for i in range(8)])
        base = self.respond(make_state(self.cat, spec, nx=8), e, surface(deltas, nx=8))[1]
        perm = [5, 2, 7, 0, 3, 6, 1, 4]
        spec_p = [spec[i] for i in perm]
        e_p = ev([spec[i][0] for i in perm], deposition_depth_m=[deltas[spec[i][2]] for i in perm])
        got = self.respond(make_state(self.cat, spec_p, nx=8), e_p, surface(deltas, nx=8))[1]
        a, b = by_id(e, base['B_new']), by_id(e_p, got['B_new'])
        for k in a:
            self.assertEqual(a[k], b[k])
        self.assertEqual(by_id(e, base['outcome']), by_id(e_p, got['outcome']))


@unittest.skipUnless(NUMBA, 'numba backend unavailable')
class NumbaParity(Base):
    def test_numba_matches_numpy_reference(self):
        rng = np.random.default_rng(5)
        n = 64
        spec = [(i + 1, 'grass' if i % 2 else 'shrub', i % 8, float(rng.uniform(0.1, 6.0)), 0.0, 'live')
                for i in range(n)]
        st = make_state(self.cat, spec, nx=8)
        deltas = {c: float(c - 4) / 4.0 for c in range(8)}
        dep = [max(deltas[i % 8], 0.0) for i in range(n)]
        ero = [max(-deltas[i % 8], 0.0) for i in range(n)]
        dose = [float(rng.uniform(0.0, 30.0)) for _ in range(n)]
        e = ev(list(range(1, n + 1)), deposition_depth_m=dep, erosion_depth_m=ero, abrasion_dose=dose,
               abrasion_unit='kg m-1')
        surf = surface(deltas, nx=8)
        ref = self.respond(st, e, surf)[1]
        got = self.respond(st, e, surf, table=disturbance.compile_species(self.cat, 'numba'))[1]
        for k in ('B_new', 'E_new'):
            np.testing.assert_allclose(got[k], ref[k], rtol=1e-14, atol=0.0)
        np.testing.assert_array_equal(got['outcome'], ref['outcome'])
        np.testing.assert_array_equal(got['status_new'], ref['status_new'])
        for c in disturbance.CAUSES:
            np.testing.assert_allclose(got['removed_kg'][c], ref['removed_kg'][c], rtol=1e-13, atol=1e-300)
        self.assertGreater(int(np.count_nonzero(ref['outcome'] == disturbance.OUT_REMOVED)), 0)   # branches exercised
        self.assertGreater(int(np.count_nonzero(ref['outcome'] == disturbance.OUT_INJURED)), 0)


@unittest.skipUnless(CUDA, 'CUDA backend unavailable')
class CudaParity(Base):
    def test_cuda_matches_numpy_reference_on_the_explicit_device(self):
        import cupy as cp
        dev = int(cp.cuda.Device().id)
        rng = np.random.default_rng(6)
        n = 64
        spec = [(i + 1, 'grass' if i % 2 else 'shrub', i % 8, float(rng.uniform(0.1, 6.0)), 0.0, 'live')
                for i in range(n)]
        st = make_state(self.cat, spec, nx=8)
        deltas = {c: float(c - 4) / 4.0 for c in range(8)}
        e = ev(list(range(1, n + 1)), deposition_depth_m=[max(deltas[i % 8], 0.0) for i in range(n)],
               erosion_depth_m=[max(-deltas[i % 8], 0.0) for i in range(n)])
        surf = surface(deltas, nx=8)
        ref = self.respond(st, e, surf)[1]
        tab = disturbance.compile_species(self.cat, 'cuda', device=dev)
        got = self.respond(st, e, surf, table=tab)[1]
        np.testing.assert_allclose(got['B_new'], ref['B_new'], rtol=1e-14, atol=0.0)
        np.testing.assert_array_equal(got['outcome'], ref['outcome'])
        with self.assertRaises(disturbance.DisturbanceBackendError):
            disturbance.compile_species(self.cat, 'cuda', device=dev + 1000)


if __name__ == '__main__':
    unittest.main()
