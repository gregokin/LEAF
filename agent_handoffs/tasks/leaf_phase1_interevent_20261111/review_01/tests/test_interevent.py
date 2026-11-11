'''LEAF Phase 1 dated interevent session + restart tests (unittest) against the REAL native SYRUP API.

Run from the candidate root with the pinned SYRUP package importable:  python -m unittest tests.test_interevent -v

* The native water authority is never faked: every case builds an actual `maple_syrup.subsurface` column, a
  prepared `prepare_transpiration` operator and a `SoilColumnState`; the session calls the ACTUAL `step` and
  `withdraw_transpiration`. Spies only WRAP real functions (to record arguments / results or to inject ONE failure
  before delegating); no fake water store exists.
* Classes SKIP only when the native package (or numba / CuPy) is unavailable. Everything else is a failure.
* The explicit root builder is the NumPy reference ('numpy') so no numba is required; numba combinations run only
  when numba is genuinely available. CUDA cases never allocate on another device.
* ALL numbers are UNCALIBRATED fixtures (examples/phase1/species.json). Written WITHOUT the ability to execute
  them (file tools only): Codex records the real runs.
'''
from __future__ import annotations

import copy
import datetime as dt
import json
import math
import os
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

from leaf import canopy, checkpoint, forcing, growth, interevent, roots, state, traits, uptake
from leaf import syrup_water as bridge
from leaf.interevent import (IntereventAdmissionError, IntereventBoundaryError, IntereventError, IntereventSession,
                             IntereventStateError)

ROOT = pathlib.Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / 'examples' / 'phase1' / 'species.json'
STATUS = bridge.native_status()
NATIVE = bool(STATUS['available'])
if NATIVE:
    from maple_syrup import subsurface as ss
    from maple_syrup import subsurface_transpiration as st
    VERIFY_NATIVE = bool(bridge.native_provenance(verify=True).get('all_hashes_match'))
else:
    VERIFY_NATIVE = False
needs_native = unittest.skipUnless(NATIVE, f'native SYRUP package unavailable: {STATUS["reason"]}')
NUMBA = bool(uptake.backend_status('numba')['available'])
CUDA = bool(uptake.backend_status('cuda')['available'])
BIG = 1 << 30
UTC = dt.timezone.utc
START = dt.datetime(2026, 5, 1, tzinfo=UTC)
DAY = dt.timedelta(days=1)
NX = NY = 4
L = 2


# ------------------------------------------------------------------ helpers
def host(a):
    return np.asarray(a.get() if hasattr(a, 'get') else a)


def load_catalog(modify=None):
    with open(CATALOG_PATH, 'r', encoding='utf-8') as fh:
        doc = json.load(fh)
    if modify is not None:
        modify(doc)
    return traits.SpeciesCatalog(doc)


def make_plants(catalog, *, shrub_e=0.0, time_days=0.0):
    '''3 grasses, 2 shrubs (one with seasonal biomass shrub_e), 1 dead standing shrub, 1 removed grass (B = 0).'''
    rec = []

    def add(pid, sp, x, y, b, e, age, status):
        rec.append(dict(plant_id=pid, species=sp, x_m=x, y_m=y, establishment_ground_m=100.0, agb_dry_kg=b,
                        seasonal_agb_dry_kg=e, age_days=age, status=status))

    add(1, 'grass', 0.8, 0.8, 0.3, 0.0, 300.0, 'live')
    add(2, 'grass', 2.5, 1.2, 0.25, 0.0, 200.0, 'live')
    add(3, 'grass', 3.2, 3.1, 0.35, 0.0, 400.0, 'live')
    add(4, 'shrub', 1.5, 2.5, 5.0, shrub_e, 1000.0, 'live')
    add(5, 'shrub', 3.0, 0.7, 4.0, 0.0, 900.0, 'live')
    add(6, 'shrub', 2.0, 2.0, 3.0, 0.0, 800.0, 'dead_standing')
    add(7, 'grass', 0.5, 3.5, 0.0, 0.0, 77.0, 'removed')
    return state.PlantState.from_records(catalog, state.Domain(NX, NY, 1.0), rec, time_days=time_days)


class Env:
    '''A REAL native column + prepared operator (+ state factory); 2 layers, 4 x 4 active cells of 1 m2.'''

    def __init__(self, backend='numpy', *, theta=(0.12, 0.15), reserve_bytes=0, edges=(0.0, 0.1, 0.5), area=1.0):
        self.layers = ss.soil_layers(list(edges), diffusivity_m2_s=2e-7, theta_evaporation_stress=0.2,
                                     theta_residual=0.05, theta_field_capacity=0.2, theta_saturation=0.45)
        self.backend, self.N, self.L, self.area = backend, NX * NY, self.layers.n_layers, float(area)
        mask = np.ones(self.N, dtype=bool)
        if backend == 'cuda':
            import cupy as cp
            self.xp, mask_xp = cp, cp.asarray(mask)
        else:
            self.xp, mask_xp = np, mask
        self.column = ss.prepare_column(self.layers, mask_xp, self.area, backend=backend, reserve_bytes=reserve_bytes)
        self.operator = st.prepare_transpiration(self.column)
        self.theta = theta

    def state(self, t_s=0.0):
        return ss.uniform_state(self.column, list(self.theta), t_s)

    def forcing(self, e0=0.0, added=0.0):
        a = self.xp.full((self.L, self.N), float(added), dtype=np.float64)
        e = self.xp.full((self.N,), float(e0), dtype=np.float64)
        return a, e


def make_session(env, catalog=None, plants=None, *, cadence=dt.timedelta(days=7), start=START, backend=None,
                 max_bytes=BIG, builder='numpy', overrides=None, native_state=None, **kw):
    catalog = catalog or load_catalog()
    plants = plants or make_plants(catalog)
    return IntereventSession(plants, catalog, operator=env.operator, native_state=native_state or env.state(),
                             start=start, backend=backend or env.backend, growth_cadence=cadence, max_edges=10_000_000,
                             max_bytes=max_bytes, root_builder=builder, layer_fraction_overrides=overrides,
                             verify_native_sources=VERIFY_NATIVE, **kw)


def day(n, *, start=START, temperature=22.0, pet=5.0, gate='open', hours=24):
    s = start + n * DAY
    return forcing.TimestepRecord(s, s + dt.timedelta(hours=hours), temperature, pet, 'interevent', gate)


def run_days(session, env, n, *, e0=0.0, first=0):
    out = []
    for i in range(first, first + n):
        a, e = env.forcing(e0)
        out.append(session.step_interevent(day(i, start=session_start(session)), added_water_m=a,
                                           potential_evaporation_m_s=e))
    return out


def session_start(session):
    return session._anchor


def fp(s):
    '''Full observable + internal fingerprint of the accepted boundary (host copies).'''
    snap = s.snapshot()
    rm = s.root_map
    b = s._b
    return {'cursor': s.cursor, 'nb': s.boundaries()['next_growth_boundary'], 'seq': s.sequence,
            'pending': s.pending_days, 'season': s.season_due, 'water': host(s.native_state.water_m).copy(),
            't': s.native_state.t_s, 'B': snap.agb_dry_kg.copy(), 'E': snap.seasonal_agb_dry_kg.copy(),
            'age': snap.age_days.copy(), 'time': snap.time_days, 'D': host(b.D).copy(), 'U': host(b.U).copy(),
            'A': b.A.copy(), 'T': b.T.copy(), 'area': None if b.area is None else host(b.area).copy(),
            'roots': None if rm is None else rm.host_arrays(), 'receipt': s.last_receipt}


def assert_same(tc, a, b):
    for k in a:
        if k in ('roots',):
            if a[k] is None or b[k] is None:
                tc.assertEqual(a[k] is None, b[k] is None)
                continue
            for kk in a[k]:
                np.testing.assert_array_equal(a[k][kk], b[k][kk], err_msg=f'roots.{kk}')
        elif isinstance(a[k], np.ndarray):
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)
        else:
            tc.assertEqual(a[k], b[k], msg=k)


def fail_once(real, exc):
    box = {'n': 0}

    def wrapper(*args, **kwargs):
        if box['n'] == 0:
            box['n'] += 1
            raise exc
        return real(*args, **kwargs)
    return wrapper, box


def spy(real, store):
    def wrapper(*args, **kwargs):
        res = real(*args, **kwargs)
        cap = {k: (v.copy() if isinstance(v, np.ndarray) or hasattr(v, 'get') and hasattr(v, 'copy') else v)
               for k, v in kwargs.items()}
        out = {k: (np.array(host(v)) if hasattr(v, 'shape') else v) for k, v in res.items()} \
            if isinstance(res, dict) else None
        store.append({'kwargs': cap, 'result': out})
        return res
    return wrapper


# ------------------------------------------------------------------ core session behaviour
@needs_native
class SessionCore(unittest.TestCase):
    backend = 'numpy'

    def setUp(self):
        self.env = Env(self.backend)
        self.catalog = load_catalog()
        self.s = make_session(self.env, self.catalog)

    # ---- construction / identity
    def test_initial_boundary_rows_and_recoverable_population(self):
        s = self.s
        self.assertEqual(s.n_live, 5)
        np.testing.assert_array_equal(s.live_ids, [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(s.live_rows, [0, 1, 2, 3, 4])
        snap = s.snapshot()
        np.testing.assert_array_equal(snap.ids, np.arange(1, 8))
        self.assertEqual(snap.counts(), {'live': 5, 'dead_standing': 1, 'removed': 1})
        self.assertEqual(snap.time_days, 0.0)
        self.assertEqual(s.root_map.n_plants, 5)
        np.testing.assert_array_equal(host(s.root_map.ids), [1, 2, 3, 4, 5])
        self.assertEqual(s.cursor, START)
        self.assertEqual(s.boundaries()['next_growth_boundary'], START + dt.timedelta(days=7))

    def test_construction_refusals(self):
        env, cat = self.env, self.catalog
        with self.assertRaises(IntereventError):          # external root map refused
            make_session(env, cat, root_map=object())
        with self.assertRaises(IntereventError):          # cadence > 7 days
            make_session(env, cat, cadence=dt.timedelta(days=8))
        with self.assertRaises(IntereventError):
            make_session(env, cat, cadence=dt.timedelta(0))
        with self.assertRaises(IntereventError):          # no silent backend switch
            make_session(env, cat, backend='numba' if self.backend == 'numpy' else 'numpy')
        with self.assertRaises(forcing.ForcingError):     # naive start refused
            make_session(env, cat, start=dt.datetime(2026, 5, 1))
        with self.assertRaises(IntereventAdmissionError):  # too small GLOBAL budget before any allocation
            make_session(env, cat, max_bytes=1000)
        other = load_catalog(lambda d: d.update(root_layers=2))
        bad = make_plants(other)
        cat3 = load_catalog(lambda d: d['species']['grass']['allometry']['diameter'].update(a=0.8))
        with self.assertRaises(IntereventError):          # catalog hash mismatch between plants and catalog
            IntereventSession(bad, cat3, operator=env.operator, native_state=env.state(), start=START,
                              backend=self.backend, growth_cadence=dt.timedelta(days=7), max_edges=10 ** 7,
                              max_bytes=BIG, root_builder='numpy', verify_native_sources=VERIFY_NATIVE)

    # ---- one daily interval: native + plant T + closure
    def test_daily_interval_closure_and_exact_accounting(self):
        s, env = self.s, self.env
        a, e = env.forcing(0.0)
        rec = s.step_interevent(day(0), added_water_m=a, potential_evaporation_m_s=e)
        self.assertEqual(s.cursor, START + DAY)
        self.assertEqual(s.sequence, 1)
        wc = rec['water_closure']
        self.assertTrue(wc['residual_within_tolerance'], wc)
        self.assertGreater(wc['transpiration_actual_m3'], 0.0)
        br = rec['transpiration']['bridge']
        self.assertTrue(br['checks_all_true'])
        self.assertAlmostEqual(br['delivered_m3'], br['demand_m3'], delta=1e-12)     # wet soil: fully supplied
        self.assertAlmostEqual(rec['native_time_advanced_s'], 86400.0, places=6)
        self.assertEqual(s.native_state.t_s, 86400.0)
        self.assertFalse(rec['growth']['performed'])
        self.assertTrue(s.has_pending_growth)
        self.assertEqual(s.pending_days, 1)
        snap = s.snapshot()
        self.assertAlmostEqual(snap.time_days, 1.0)
        self.assertEqual(rec['memory']['max_bytes'], BIG)
        self.assertLessEqual(rec['memory']['peak_estimate_bytes'], BIG)
        # snapshots never advance the clock
        s.snapshot()
        s.snapshot()
        self.assertEqual(s.cursor, START + DAY)
        self.assertAlmostEqual(s.snapshot().time_days, 1.0)

    def test_native_water_matches_independent_native_calls(self):
        '''The session result equals an independent hand composition of the real native calls.'''
        s, env = self.s, self.env
        a, e = env.forcing(1e-8)
        before = fp(s)
        rec = s.step_interevent(day(0), added_water_m=a, potential_evaporation_m_s=e)
        ref = ss.step(env.column, env.state(), 86400.0, added_water_m=a, potential_evaporation_m_s=e)
        total_t = rec['transpiration']['native_budget']['actual_m3']
        removed = ss.storage_m3(env.column, ref.state) - ss.storage_m3(env.column, s.native_state)
        self.assertAlmostEqual(removed, total_t, delta=1e-9 * max(1.0, abs(removed)))
        self.assertEqual(before['t'], 0.0)

    def test_week_growth_applies_water_once_and_rebuilds_roots(self):
        s, env = self.s, self.env
        store_g, store_d = [], []
        with mock.patch.object(growth, 'grow', spy(growth.grow, store_g)), \
                mock.patch.object(growth, 'demand', spy(growth.demand, store_d)):
            recs = run_days(s, env, 7)
        self.assertEqual(len(store_g), 1)                           # exactly ONE growth application per boundary
        self.assertEqual(len(store_d), 7)
        self.assertTrue(recs[-1]['growth']['performed'])
        self.assertFalse(recs[-1]['growth']['year_end'])
        self.assertEqual(s.cursor, START + 7 * DAY)
        self.assertEqual(s.boundaries()['next_growth_boundary'], START + 14 * DAY)
        self.assertFalse(s.has_pending_growth)
        cap = store_g[0]
        daily = sum(d['result']['requested_m3'] for d in store_d)           # sum of the 7 daily requests per live plant
        np.testing.assert_allclose(cap['kwargs']['requested_m3'], daily, rtol=1e-12)   # grow got the ACCUMULATED request
        self.assertTrue(np.all(cap['kwargs']['delivered_m3'] <= cap['kwargs']['requested_m3'] * (1 + 1e-9)))
        res = cap['result']
        snap = s.snapshot()
        np.testing.assert_array_equal(snap.agb_dry_kg[:5], res['B_new_kg'])            # copied out of the shared buffer
        np.testing.assert_array_equal(snap.seasonal_agb_dry_kg[:5], res['E_new_kg'])
        # actual satisfaction is applied ONCE after the plan was capped
        np.testing.assert_allclose(res['normal_gain_kg'], res['normal_plan_kg'] * res['satisfaction'], rtol=1e-12, atol=0)
        np.testing.assert_allclose(res['extra_gain_kg'], res['extra_plan_kg'] * res['satisfaction'], rtol=1e-12, atol=0)
        self.assertTrue(np.all(snap.agb_dry_kg[:5] > np.array([0.3, 0.25, 0.35, 5.0, 4.0])))
        np.testing.assert_array_equal(snap.agb_dry_kg[5:], [3.0, 0.0])                  # dead / removed untouched
        self.assertEqual(s._b.n_rebuilds, 1)
        # roots follow the NEW geometry (grown biomass => at least as many edges)
        self.assertGreaterEqual(s.root_map.n_edges, 1)
        # the period integrals handed to grow are the sums of the daily increments (7 whole May days)
        A = cap['kwargs']['activity_integral']
        np.testing.assert_allclose(A, np.array([1.0, 1.0]) * 7 / 365.0, rtol=1e-12)    # grass / shrub May activity 1.0

    def test_water_limited_growth_uses_actual_over_requested(self):
        env = Env(self.backend, theta=(0.0502, 0.0502))
        s = make_session(env, self.catalog)
        store = []
        with mock.patch.object(growth, 'grow', spy(growth.grow, store)):
            recs = run_days(s, env, 7)
        br = [r['transpiration']['bridge'] for r in recs]
        self.assertTrue(any(b['delivered_m3'] < b['demand_m3'] * (1 - 1e-6) for b in br))
        res = store[0]['result']
        self.assertTrue(np.any(res['satisfaction'] < 1.0))
        np.testing.assert_allclose(res['normal_gain_kg'], res['normal_plan_kg'] * res['satisfaction'], rtol=1e-12)
        pk = recs[-1]['growth']['packet']
        self.assertGreater(pk['sum_maintenance_deficit_kg_year'], 0.0)       # support diagnostics retained
        per = recs[-1]['growth']['per_species']
        self.assertLess(per['shrub']['delivered_m3'], per['shrub']['requested_m3'])

    def test_partial_day_leap_day_and_new_year_lengths(self):
        start = dt.datetime(2027, 12, 29, tzinfo=UTC)
        s = make_session(self.env, self.catalog, start=start, cadence=dt.timedelta(days=7),
                         plants=make_plants(self.catalog, shrub_e=0.2))
        incs = []
        real = growth.integral_increments

        def wrap(*a, **k):
            incs.append((k['year_days'], k['dt_days'], k['month']))
            return real(*a, **k)

        a, e = self.env.forcing(0.0)
        with mock.patch.object(growth, 'integral_increments', wrap):
            for i in range(3):                       # 29, 30, 31 Dec: 365-day year; a half day split into two halves
                half =forcing.TimestepRecord(start + i * DAY, start + i * DAY + dt.timedelta(hours=12), 5.0, 5.0)
                s.step_interevent(half, added_water_m=a, potential_evaporation_m_s=e)
                half2 = forcing.TimestepRecord(start + i * DAY + dt.timedelta(hours=12), start + (i + 1) * DAY, 5.0, 5.0)
                rec = s.step_interevent(half2, added_water_m=a, potential_evaporation_m_s=e)
            self.assertEqual(s.cursor, dt.datetime(2028, 1, 1, tzinfo=UTC))
            self.assertTrue(rec['growth']['performed'])
            self.assertTrue(rec['growth']['year_end'])                        # forced new-year boundary
            self.assertEqual(rec['next_growth_boundary'], (start + dt.timedelta(days=7)).isoformat())
            snap = s.snapshot()
            self.assertEqual(float(snap.seasonal_agb_dry_kg.max()), 0.0)     # seasonal component removed at year end
            for i in range(2):                       # 1, 2 Jan 2028 (leap year, 366 days)
                s.step_interevent(day(3 + i, start=start, temperature=5.0), added_water_m=a, potential_evaporation_m_s=e)
        self.assertEqual({y for y, _, _ in incs[:6]}, {365})
        self.assertEqual({y for y, _, _ in incs[6:]}, {366})
        self.assertEqual({m for _, _, m in incs[:6]}, {12})
        self.assertEqual({m for _, _, m in incs[6:]}, {1})
        self.assertEqual({round(d, 12) for _, d, _ in incs[:6]}, {0.5})
        # leap day itself
        s2 = make_session(self.env, self.catalog, start=dt.datetime(2028, 2, 28, tzinfo=UTC))
        a, e = self.env.forcing(0.0)
        for i in range(2):
            s2.step_interevent(day(i, start=dt.datetime(2028, 2, 28, tzinfo=UTC)), added_water_m=a,
                               potential_evaporation_m_s=e)
        self.assertEqual(s2.cursor, dt.datetime(2028, 3, 1, tzinfo=UTC))      # 28 Feb + 29 Feb exist
        self.assertAlmostEqual(s2.snapshot().time_days, 2.0)

    def test_boundaries_split_and_unsplit_refusal_before_mutation(self):
        s, env = self.s, self.env
        a, e = env.forcing(0.0)
        long_rec = forcing.TimestepRecord(START, START + 10 * DAY, 20.0, 5.0)
        before = fp(s)
        with self.assertRaises(IntereventBoundaryError):
            s.step_interevent(long_rec, added_water_m=a, potential_evaporation_m_s=e)
        straddle = forcing.TimestepRecord(START + dt.timedelta(hours=12), START + dt.timedelta(hours=36), 20.0, 5.0)
        s_off = make_session(env, self.catalog, start=START + dt.timedelta(hours=12))
        with self.assertRaises(IntereventBoundaryError):                    # crosses local midnight
            s_off.step_interevent(straddle, added_water_m=a, potential_evaporation_m_s=e)
        assert_same(self, before, fp(s))
        pieces = s.split_record(long_rec)
        self.assertEqual(pieces[0].start, START)
        self.assertEqual(pieces[-1].end, START + 10 * DAY)
        for p, q in zip(pieces, pieces[1:]):
            self.assertEqual(p.end, q.start)
        self.assertEqual(len(pieces), 10)                                    # every midnight; grid (day 7) is a midnight
        for p in pieces:
            self.assertFalse(p.crosses_local_midnight)
        closed = forcing.TimestepRecord(START, START + 10 * DAY, 0.0, None, 'interevent', 'closed')
        self.assertEqual(s.split_record(closed), [closed])
        # growth boundary at 12:00 splits as well
        s12 = make_session(env, self.catalog, cadence=dt.timedelta(hours=12))
        pieces = s12.split_record(forcing.TimestepRecord(START, START + DAY, 20.0, 5.0))
        self.assertEqual([p.end for p in pieces], [START + dt.timedelta(hours=12), START + DAY])

    def test_gap_overlap_replay_wrong_offset_and_forcing_refused(self):
        s, env = self.s, self.env
        a, e = env.forcing(0.0)
        s.step_interevent(day(0), added_water_m=a, potential_evaporation_m_s=e)
        ref = fp(s)
        with self.assertRaises(IntereventStateError):                        # replay
            s.step_interevent(day(0), added_water_m=a, potential_evaporation_m_s=e)
        with self.assertRaises(IntereventStateError):                        # gap
            s.step_interevent(day(2), added_water_m=a, potential_evaporation_m_s=e)
        overlap = forcing.TimestepRecord(START + dt.timedelta(hours=12), START + dt.timedelta(hours=36), 20.0, 5.0)
        with self.assertRaises(IntereventStateError):
            s.step_interevent(overlap, added_water_m=a, potential_evaporation_m_s=e)
        tz1 = dt.timezone(dt.timedelta(hours=1))
        wrong_off = forcing.TimestepRecord(dt.datetime(2026, 5, 2, 1, tzinfo=tz1), dt.datetime(2026, 5, 3, 1, tzinfo=tz1),
                                           20.0, 5.0)
        with self.assertRaises(IntereventError):
            s.step_interevent(wrong_off, added_water_m=a, potential_evaporation_m_s=e)
        with self.assertRaises(forcing.ForcingError):                        # wind is not an interevent record kind
            forcing.TimestepRecord(START + DAY, START + 2 * DAY, 20.0, 5.0, 'wind')
        with self.assertRaises(IntereventError):                             # PET required for plant demand
            s.step_interevent(day(1, pet=None), added_water_m=a, potential_evaporation_m_s=e)
        with self.assertRaises(ValueError):                                  # wrong forcing shape
            s.step_interevent(day(1), added_water_m=self.env.xp.zeros((L, self.env.N + 1)), potential_evaporation_m_s=e)
        with self.assertRaises(IntereventError):
            s.step_interevent(day(1), added_water_m=None, potential_evaporation_m_s=e)
        with self.assertRaises(IntereventStateError):                        # explicit equal interval id
            s.step_interevent(day(1), added_water_m=a, potential_evaporation_m_s=e,
                              interval_id=s.last_receipt['interval_id'])
        assert_same(self, ref, fp(s))
        s.step_interevent(day(1), added_water_m=a, potential_evaporation_m_s=e)   # still works afterwards
        self.assertEqual(s.sequence, 2)

    # ---- failure atomicity
    def _inject_and_retry(self, patch_target, attr, rounds_before=0, exc=RuntimeError('injected'), n_days=1):
        s, env = self.s, self.env
        a, e = env.forcing(0.0)
        for i in range(rounds_before):
            s.step_interevent(day(i), added_water_m=a, potential_evaporation_m_s=e)
        before = fp(s)
        wrapper, box = fail_once(getattr(patch_target, attr), exc)
        with mock.patch.object(patch_target, attr, wrapper):
            with self.assertRaises(type(exc)):
                s.step_interevent(day(rounds_before), added_water_m=a, potential_evaporation_m_s=e)
            assert_same(self, before, fp(s))             # the entire accepted boundary is unchanged
            rec = s.step_interevent(day(rounds_before), added_water_m=a, potential_evaporation_m_s=e)   # retry succeeds
        self.assertEqual(box['n'], 1)
        return before, rec

    def test_native_step_failure_leaves_boundary_and_retry_accounts_once(self):
        before, rec = self._inject_and_retry(ss, 'step')
        self.assertEqual(self.s.native_state.t_s, 86400.0)                  # advanced exactly one day, not two
        self.assertEqual(self.s.sequence, 1)
        self.assertEqual(self.s.pending_days, 1)

    def test_demand_failure_atomic(self):
        self._inject_and_retry(growth, 'demand')
        self.assertEqual(self.s.pending_days, 1)

    def test_bridge_failure_atomic_no_double_debit(self):
        before, rec = self._inject_and_retry(bridge, 'withdraw_plants', exc=bridge.SyrupWaterError('injected'))
        removed = ss.storage_m3(self.env.column, self.env.state()) - ss.storage_m3(self.env.column, self.s.native_state)
        self.assertAlmostEqual(removed, rec['water_closure']['transpiration_actual_m3'], delta=1e-9)

    def test_native_withdrawal_failure_atomic(self):
        self._inject_and_retry(st, 'withdraw_transpiration', exc=st.TranspirationError('injected')
                               if hasattr(st, 'TranspirationError') else RuntimeError('injected'))

    def test_grow_failure_keeps_pending_demand_and_retry_succeeds(self):
        s = self.s
        before, rec = self._inject_and_retry(growth, 'grow', rounds_before=6, exc=growth.GrowthDataError('injected'))
        self.assertTrue(rec['growth']['performed'])
        self.assertFalse(s.has_pending_growth)
        self.assertEqual(before['pending'], 6)                               # six days of pending demand survived the failure
        self.assertEqual(s.sequence, 7)

    def test_root_rebuild_failure_atomic(self):
        before, rec = self._inject_and_retry(roots, 'build_root_map', rounds_before=6, exc=roots.RootsError('injected'))
        self.assertTrue(rec['growth']['performed'])

    def test_admission_too_small_refuses_before_any_work(self):
        s, env = self.s, self.env
        a, e = env.forcing(0.0)
        before = fp(s)
        s.max_bytes = 1500
        wrapper = mock.Mock(side_effect=AssertionError('native step must not run'))
        with mock.patch.object(ss, 'step', wrapper):
            with self.assertRaises(IntereventAdmissionError) as cm:
                s.step_interevent(day(0), added_water_m=a, potential_evaporation_m_s=e)
        wrapper.assert_not_called()
        self.assertGreater(cm.exception.total, 1500)
        with self.assertRaises(IntereventAdmissionError):
            s.snapshot()
        s.max_bytes = BIG
        assert_same(self, before, fp(s))

    def test_growth_boundary_that_cannot_fit_refuses_and_keeps_old_boundary(self):
        '''Budget below the weekly growth / rebuild stage ledger: the whole transaction is refused, boundary intact.'''
        s, env = self.s, self.env
        a, e = env.forcing(0.0)
        run_days(s, env, 6)
        before = fp(s)
        stages = s._stages(s._b, fire=True, carry=0, forcing_bytes=8 * L * env.N + 8 * env.N, native_extra=0)
        s.max_bytes = min(stages['grow']['total'], stages['rebuild_preflight']['total']) - 1
        try:
            with self.assertRaises(IntereventAdmissionError):
                s.step_interevent(day(6), added_water_m=a, potential_evaporation_m_s=e)
        finally:
            s.max_bytes = BIG
        assert_same(self, before, fp(s))

    # ---- flush / transport / gates
    def test_flush_growth_partial_period_and_noop(self):
        s, env = self.s, self.env
        r0 = s.flush_growth()
        self.assertFalse(r0['growth']['performed'])
        self.assertFalse(r0['published'])
        self.assertEqual(s.sequence, 0)
        run_days(s, env, 3)
        B0 = s.snapshot().agb_dry_kg.copy()
        nb = s.boundaries()['next_growth_boundary']
        rec = s.flush_growth()
        self.assertTrue(rec['growth']['performed'])
        self.assertAlmostEqual(rec['growth']['period_days'], 3.0)
        self.assertFalse(s.has_pending_growth)
        self.assertEqual(s.cursor, START + 3 * DAY)                           # no clock advance
        self.assertEqual(s.boundaries()['next_growth_boundary'], nb)           # grid unchanged
        self.assertTrue(np.all(s.snapshot().agb_dry_kg[:5] >= B0[:5]))
        self.assertEqual(s.sequence, 4)
        # canopy export of the CURRENT biomass through the shared view
        view = s.export_canopy_view(np.full((NY, NX), 100.0))
        self.assertEqual(view['meta']['n_objects'], 6)                          # live + dead standing, positive diameter

    def test_transport_needs_flush_then_advances_ages_without_catch_up(self):
        s, env = self.s, self.env
        run_days(s, env, 3)
        w0 = host(s.native_state.water_m).copy()
        with self.assertRaises(IntereventStateError):
            s.advance_transport('wind', START + 3 * DAY, START + 5 * DAY)
        s.flush_growth()
        B0 = s.snapshot().agb_dry_kg.copy()
        age0 = s.snapshot().age_days.copy()
        w_before = host(s.native_state.water_m).copy()
        t0 = s.native_state.t_s
        with mock.patch.object(growth, 'demand', side_effect=AssertionError('no T during transport')), \
                mock.patch.object(ss, 'step', side_effect=AssertionError('no native step during wind')):
            rec = s.advance_transport('wind', START + 3 * DAY, START + 12 * DAY)
        self.assertEqual(rec['growth_boundaries_skipped'], 1)                   # the day-7 boundary was crossed, no catch-up
        self.assertEqual(s.boundaries()['next_growth_boundary'], START + 14 * DAY)   # fixed grid preserved
        self.assertEqual(s.cursor, START + 12 * DAY)
        np.testing.assert_array_equal(host(s.native_state.water_m), w_before)    # wind: water unchanged
        self.assertEqual(s.native_state.t_s, t0 + 9 * 86400.0)                   # native time advanced consistently
        np.testing.assert_array_equal(s.snapshot().agb_dry_kg, B0)
        snap = s.snapshot()
        live = snap.status == state.LIVE
        np.testing.assert_allclose(snap.age_days[live], age0[live] + 9.0, rtol=0, atol=1e-9)
        np.testing.assert_array_equal(snap.age_days[~live], age0[~live])          # dead / removed ages frozen
        self.assertAlmostEqual(snap.time_days, 12.0)
        self.assertEqual(s.pending_days, 0)
        with self.assertRaises(IntereventStateError):                            # stale replay refused
            s.advance_transport('wind', START + 3 * DAY, START + 12 * DAY)
        with self.assertRaises(IntereventError):
            s.advance_transport('interevent', START + 12 * DAY, START + 13 * DAY)
        with self.assertRaises(IntereventError):
            s.advance_transport('wind', START + 12 * DAY, START + 13 * DAY, native_state=s.native_state)
        # the next open interval accrues ONLY its own duration (no catch-up over the event)
        a, e = env.forcing(0.0)
        s.step_interevent(day(12), added_water_m=a, potential_evaporation_m_s=e)
        self.assertEqual(s.pending_days, 1)
        self.assertEqual(w0.shape, w_before.shape)

    def test_rain_needs_external_native_owner_result(self):
        s, env = self.s, self.env
        with self.assertRaises(IntereventError):
            s.advance_transport('rain', START, START + DAY)
        added = env.xp.zeros((L, env.N))
        added[0] += 0.004
        res = ss.step(env.column, s.native_state, 86400.0, added_water_m=added,
                      potential_evaporation_m_s=env.xp.zeros(env.N))
        before = fp(s)
        bad_time = ss.SoilColumnState(res.state.water_m, res.state.t_s + 10.0)
        with self.assertRaises(IntereventError):
            s.advance_transport('rain', START, START + DAY, native_state=bad_time)
        bad_water = ss.SoilColumnState(res.state.water_m * 0.0 - 1.0, res.state.t_s)
        with self.assertRaises(ValueError):
            s.advance_transport('recession', START, START + DAY, native_state=bad_water)
        assert_same(self, before, fp(s))
        rec = s.advance_transport('rain', START, START + DAY, native_state=res.state)
        self.assertTrue(rec['native_water_replaced_by_external_owner'])
        np.testing.assert_array_equal(host(s.native_state.water_m), host(res.state.water_m))
        self.assertIsNot(s.native_state.water_m, res.state.water_m)                 # the session owns its copy
        self.assertEqual(s.native_state.t_s, 86400.0)

    def test_closed_gate_is_a_pause_without_native_or_plant_activity(self):
        s, env = self.s, self.env
        a, e = env.forcing(0.0)
        closed = day(0, gate='closed')
        with self.assertRaises(IntereventError):                                 # forcing must not be passed
            s.step_interevent(closed, added_water_m=a, potential_evaporation_m_s=e)
        w0 = host(s.native_state.water_m).copy()
        with mock.patch.object(ss, 'step', side_effect=AssertionError('closed gate runs no native hydrology')), \
                mock.patch.object(growth, 'demand', side_effect=AssertionError('closed gate has no T')):
            rec = s.step_interevent(closed)
        self.assertEqual(rec['gate'], 'closed')
        np.testing.assert_array_equal(host(s.native_state.water_m), w0)
        self.assertEqual(s.native_state.t_s, 86400.0)
        self.assertEqual(s.cursor, START + DAY)
        self.assertEqual(s.pending_days, 0)
        run_days(s, env, 1, first=1)                                              # open interval pending, then...
        with self.assertRaises(IntereventStateError):
            s.step_interevent(day(2, gate='closed'))                              # pending growth must be flushed first

    def test_deferred_seasonal_turnover_inside_transport(self):
        start = dt.datetime(2027, 12, 30, tzinfo=UTC)
        plants = make_plants(self.catalog, shrub_e=0.2)
        s = make_session(self.env, self.catalog, plants=plants, start=start)
        snap0 = s.snapshot()
        rec = s.advance_transport('wind', start, dt.datetime(2028, 1, 3, tzinfo=UTC))
        self.assertTrue(rec['season_due'])
        self.assertTrue(s.season_due)
        snap1 = s.snapshot()
        np.testing.assert_array_equal(snap1.agb_dry_kg, snap0.agb_dry_kg)          # event duration accrues no growth
        self.assertEqual(float(snap1.seasonal_agb_dry_kg[3]), 0.2)                 # turnover is DEFERRED
        a, e = self.env.forcing(0.0)
        r = s.step_interevent(forcing.TimestepRecord(dt.datetime(2028, 1, 3, tzinfo=UTC),
                                                     dt.datetime(2028, 1, 4, tzinfo=UTC), 5.0, 5.0),
                              added_water_m=a, potential_evaporation_m_s=e)
        self.assertTrue(r['deferred_season_applied'])
        self.assertFalse(s.season_due)
        snap2 = s.snapshot()
        self.assertEqual(float(snap2.seasonal_agb_dry_kg.max()), 0.0)
        self.assertAlmostEqual(float(snap2.agb_dry_kg[3]), float(snap0.agb_dry_kg[3]) - 0.2, places=12)
        self.assertEqual(s.pending_days, 1)                                         # only the 1 open day accrued

    # ---- roots / allometry
    def test_species_and_individual_root_profiles_persist_through_rebuild(self):
        ov = {1: [0.1, 0.9]}
        s = make_session(self.env, self.catalog, overrides=ov)

        def layer_sums(sess, pid):
            h = sess.root_map.host_arrays()
            row = int(np.flatnonzero(h['ids'] == pid)[0])
            sl = slice(int(h['offsets'][row]), int(h['offsets'][row + 1]))
            lay = h['resources'][sl] // (NX * NY)
            w = h['weights'][sl]
            return [float(w[lay == k].sum()) for k in range(L)]

        np.testing.assert_allclose(layer_sums(s, 1), [0.1, 0.9], atol=1e-12)       # individual override
        np.testing.assert_allclose(layer_sums(s, 2), [0.85, 0.15], atol=1e-12)     # species default (grass)
        np.testing.assert_allclose(layer_sums(s, 4), [0.25, 0.75], atol=1e-12)     # species default (shrub)
        run_days(s, self.env, 7)                                                     # weekly rebuild
        np.testing.assert_allclose(layer_sums(s, 1), [0.1, 0.9], atol=1e-12)
        np.testing.assert_allclose(layer_sums(s, 2), [0.85, 0.15], atol=1e-12)

    def test_unmet_roots_are_not_redirected(self):
        '''Dry cell under a plant keeps its unmet request: weights are the same before and after a dry interval.'''
        theta = (0.0501, 0.0501)
        env = Env(self.backend, theta=theta)
        s = make_session(env, self.catalog)
        w0 = s.root_map.host_arrays()
        run_days(s, env, 2)
        w1 = s.root_map.host_arrays()
        for k in w0:
            np.testing.assert_array_equal(w0[k], w1[k])                              # fixed weights inside a period

    def test_arbitrary_allometry_exponents_and_canopy_area_demand(self):
        def mod(doc):
            doc['species']['grass']['allometry']['diameter'].update(a=0.7, b=0.85)
            doc['species']['grass']['allometry']['height'].update(a=0.4, b=0.15)
            doc['species']['shrub']['allometry']['diameter'].update(a=2.1, b=0.55)
        cat = load_catalog(mod)
        s = make_session(self.env, cat, plants=make_plants(cat))
        store = []
        a, e = self.env.forcing(0.0)
        with mock.patch.object(growth, 'demand', spy(growth.demand, store)):
            s.step_interevent(day(0), added_water_m=a, potential_evaporation_m_s=e)
        snap = s.snapshot()
        view = s.export_canopy_view(np.full((NY, NX), 100.0))
        sc, fl = view['sidecar'], view['fields']
        for i, pid in enumerate(sc['plant_id']):
            row = int(np.flatnonzero(snap.ids == pid)[0])
            spec = snap.species_names[snap.species_index[row]]
            al = cat.allometry_arrays()
            si = cat.index(spec)
            B = snap.agb_dry_kg[row]
            self.assertAlmostEqual(2 * fl['semi_major_m'][i], al['a_d'][si] * B ** al['b_d'][si], places=12)
        area = host(store[0]['kwargs']['canopy_area_m2'])
        geo = canopy.derive_geometry(make_plants(cat), cat)
        np.testing.assert_allclose(area, np.pi * geo['radius_m'][:5] ** 2, rtol=1e-12)

    # ---- diagnostics bound
    def test_receipts_are_compact_json_and_history_free(self):
        s, env = self.s, self.env
        recs = run_days(s, env, 8)
        json.dumps(recs[-1], allow_nan=False)
        self.assertIsNotNone(s.last_receipt)
        self.assertEqual(s.last_receipt['interval_id'], recs[-1]['interval_id'])
        for name in ('_history', 'history', 'receipts'):
            self.assertFalse(hasattr(s, name))
        d = s.describe()
        self.assertEqual(d['n_growth_updates'], 1)


# ------------------------------------------------------------------ restart
@needs_native
class RestartTests(unittest.TestCase):
    backend = 'numpy'

    def setUp(self):
        self.env = Env(self.backend)
        self.catalog = load_catalog()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = pathlib.Path(self.tmp.name)

    def load(self, path, **kw):
        args = dict(operator=self.env.operator, backend=self.backend, max_edges=10_000_000, max_bytes=BIG,
                    verify_native_sources=VERIFY_NATIVE)
        args.update(kw)
        return IntereventSession.load(path, self.catalog, **args)

    def test_split_run_equals_continuous_run_mid_week(self):
        cont = make_session(self.env, self.catalog, overrides={2: [0.3, 0.7]})
        run_days(cont, self.env, 10)
        split = make_session(self.env, self.catalog, overrides={2: [0.3, 0.7]})
        run_days(split, self.env, 3)                                                  # mid-week: pending D/U/integrals
        path = split.save(self.dir / 'cp')
        before = fp(split)
        loaded = self.load(path)
        a = fp(loaded)
        for k in ('cursor', 'nb', 'seq', 'pending', 'season', 't'):
            self.assertEqual(a[k], before[k], k)
        for k in ('water', 'B', 'E', 'D', 'U', 'A', 'T', 'area'):
            np.testing.assert_array_equal(a[k], before[k], err_msg=k)
        for k in before['roots']:
            np.testing.assert_array_equal(a['roots'][k], before['roots'][k])
        np.testing.assert_allclose(a['age'], before['age'], rtol=1e-14)
        self.assertEqual(loaded.last_receipt['interval_id'], split.last_receipt['interval_id'])
        a_, e_ = self.env.forcing(0.0)
        for i in range(3, 10):
            loaded.step_interevent(day(i), added_water_m=a_, potential_evaporation_m_s=e_)
        c, l = fp(cont), fp(loaded)
        for k in ('water', 'B', 'E', 'D', 'U', 'A', 'T', 'area'):
            np.testing.assert_array_equal(c[k], l[k], err_msg=k)
        np.testing.assert_allclose(c['age'], l['age'], rtol=1e-12)
        self.assertEqual((c['cursor'], c['nb'], c['pending'], c['t']), (l['cursor'], l['nb'], l['pending'], l['t']))
        for k in c['roots']:
            np.testing.assert_array_equal(c['roots'][k], l['roots'][k])
        # the original session is unchanged by saving
        assert_same(self, before, fp(split))

    def test_csv_is_the_canonical_plant_format_with_all_ids(self):
        s = make_session(self.env, self.catalog)
        run_days(s, self.env, 2)
        path = s.save(self.dir / 'cp')
        snap = state.read_plant_csv(os.path.join(path, 'plants.csv'), self.catalog)
        np.testing.assert_array_equal(snap.ids, np.arange(1, 8))
        np.testing.assert_array_equal(snap.agb_dry_kg, s.snapshot().agb_dry_kg)
        self.assertEqual(snap.time_days, s.snapshot().time_days)
        self.assertEqual(sorted(os.listdir(path)), ['plants.csv', 'restart.json', 'restart_arrays.npz'])

    def test_deferred_season_is_restored(self):
        start = dt.datetime(2027, 12, 30, tzinfo=UTC)
        s = make_session(self.env, self.catalog, plants=make_plants(self.catalog, shrub_e=0.2), start=start)
        s.advance_transport('wind', start, dt.datetime(2028, 1, 2, tzinfo=UTC))
        path = s.save(self.dir / 'cp')
        loaded = self.load(path)
        self.assertTrue(loaded.season_due)
        a, e = self.env.forcing(0.0)
        rec = loaded.step_interevent(forcing.TimestepRecord(dt.datetime(2028, 1, 2, tzinfo=UTC),
                                                            dt.datetime(2028, 1, 3, tzinfo=UTC), 5.0, 5.0),
                                     added_water_m=a, potential_evaporation_m_s=e)
        self.assertTrue(rec['deferred_season_applied'])

    def test_no_overwrite_and_atomic_publication(self):
        s = make_session(self.env, self.catalog)
        path = s.save(self.dir / 'cp')
        with self.assertRaises(FileExistsError):
            s.save(self.dir / 'cp')
        self.assertTrue(os.path.isdir(path))
        real_rename = os.rename
        target = str(self.dir / 'cp2')

        def boom(src, dst):
            if dst == target:
                raise OSError('injected rename failure')
            return real_rename(src, dst)
        with mock.patch.object(os, 'rename', boom):
            with self.assertRaises(OSError):
                s.save(target)
        self.assertFalse(os.path.lexists(target))                                       # nothing half-written is visible
        self.assertEqual([n for n in os.listdir(self.dir) if n.startswith('.leaf-checkpoint')], [])
        s.save(self.dir / 'cp', overwrite=True)                                         # explicit overwrite works
        self.load(self.dir / 'cp')

    def _tamper(self, name, fn):
        s = make_session(self.env, self.catalog)
        run_days(s, self.env, 2)
        path = s.save(self.dir / f'cp-{name}')
        fn(pathlib.Path(path))
        return path

    def test_corruption_is_rejected_before_adoption(self):
        def flip(p):
            data = bytearray((p / 'restart_arrays.npz').read_bytes())
            data[len(data) // 2] ^= 0xFF
            (p / 'restart_arrays.npz').write_bytes(bytes(data))
        with self.assertRaises(checkpoint.CheckpointError):
            self.load(self._tamper('npz', flip))

        def trunc(p):
            (p / 'plants.csv').write_text((p / 'plants.csv').read_text()[:-40])
        with self.assertRaises(checkpoint.CheckpointError):
            self.load(self._tamper('csv', trunc))

        def edit_manifest(p):
            doc = json.loads((p / 'restart.json').read_text())
            doc['state']['season_due'] = True
            (p / 'restart.json').write_text(json.dumps(doc))
        with self.assertRaises(checkpoint.CheckpointError):
            self.load(self._tamper('manifest', edit_manifest))

        def drop(p):
            os.unlink(p / 'restart_arrays.npz')
        with self.assertRaises(checkpoint.CheckpointError):
            self.load(self._tamper('drop', drop))

    def test_consistent_but_wrong_content_is_rejected(self):
        '''A manifest + array hash that were recomputed over inconsistent pending arrays is still rejected.'''
        def rewrite(p):
            npz = dict(np.load(p / 'restart_arrays.npz', allow_pickle=False))
            npz['pending_delivered_m3'] = npz['pending_requested_m3'] * 10.0 + 1.0      # delivered > requested
            import io
            buf = io.BytesIO()
            np.savez(buf, **npz)
            (p / 'restart_arrays.npz').write_bytes(buf.getvalue())
            doc = json.loads((p / 'restart.json').read_text())
            doc['files']['restart_arrays.npz'] = checkpoint._sha(buf.getvalue())
            doc['array_hashes'] = {k: checkpoint._arr_hash(v) for k, v in npz.items()}
            body = {k: v for k, v in doc.items() if k != 'content_hash'}
            doc['content_hash'] = checkpoint._sha(checkpoint._canonical(body))
            (p / 'restart.json').write_text(json.dumps(doc))
        with self.assertRaises(checkpoint.CheckpointError):
            self.load(self._tamper('inconsistent', rewrite))

    def test_catalog_backend_native_and_source_mismatch_refused(self):
        s = make_session(self.env, self.catalog)
        run_days(s, self.env, 1)
        path = s.save(self.dir / 'cp')
        other = load_catalog(lambda d: d['species']['grass']['growth']['traits'].update(live_fraction=0.5))
        with self.assertRaises(checkpoint.CheckpointError):
            IntereventSession.load(path, other, operator=self.env.operator, backend=self.backend, max_edges=10 ** 7,
                                   max_bytes=BIG, verify_native_sources=VERIFY_NATIVE)
        with self.assertRaises(checkpoint.CheckpointError):
            self.load(path, backend='numba' if self.backend == 'numpy' else 'numpy')
        env2 = Env(self.backend, edges=(0.0, 0.2, 0.5))                               # different layers => other fingerprint
        with self.assertRaises(checkpoint.CheckpointError):
            self.load(path, operator=env2.operator)
        with mock.patch.object(checkpoint, 'module_hashes', return_value={'interevent': 'sha256:00'}):
            with self.assertRaises(checkpoint.CheckpointError):
                self.load(path)
            self.load(path, allow_source_change=True)
        with self.assertRaises(checkpoint.CheckpointError):
            self.load(path, root_builder='compiled')
        with self.assertRaises(IntereventAdmissionError):                              # restore budget is admitted
            self.load(path, max_bytes=1000)

    def test_restart_refuses_stored_roots_that_do_not_match_population(self):
        s = make_session(self.env, self.catalog)
        path = s.save(self.dir / 'cp')
        with mock.patch.object(roots, 'build_root_map', side_effect=roots.RootsError('cannot rebuild')):
            with self.assertRaises(roots.RootsError):
                self.load(path)                                                         # verification rebuild fails: no adoption
        self.load(path, verify_roots=False)

    def test_save_admission_counts_serialization_buffers(self):
        s = make_session(self.env, self.catalog)
        s.max_bytes = 5000
        with self.assertRaises(IntereventAdmissionError):
            s.save(self.dir / 'cp')
        self.assertFalse(os.path.lexists(self.dir / 'cp'))


@needs_native
class NoPlantsAndPopulations(unittest.TestCase):
    def test_no_live_plants_runs_native_only(self):
        env = Env('numpy')
        cat = load_catalog()
        plants = state.PlantState.from_records(cat, state.Domain(NX, NY, 1.0), [
            dict(plant_id=1, species='grass', x_m=1.0, y_m=1.0, establishment_ground_m=100.0, agb_dry_kg=0.3,
                 seasonal_agb_dry_kg=0.0, age_days=5.0, status='dead_standing')])
        s = make_session(env, cat, plants=plants)
        a, e = env.forcing(1e-8)
        rec = s.step_interevent(day(0), added_water_m=a, potential_evaporation_m_s=e)
        self.assertIsNone(rec['transpiration'])
        self.assertEqual(s.native_state.t_s, 86400.0)
        self.assertEqual(s.snapshot().age_days[0], 5.0)                                  # dead plants do not age
        s.advance_transport('wind', START + DAY, START + 2 * DAY)

    def test_snapshot_writer_and_canonical_csv(self):
        env = Env('numpy')
        s = make_session(env)
        run_days(s, env, 1)
        from leaf import recording
        with tempfile.TemporaryDirectory() as td:
            w = recording.SnapshotWriter(os.path.join(td, 'veg_{index:03d}.csv'))
            p = s.write_snapshot(w, 'initial')
            self.assertTrue(os.path.exists(p))
            self.assertEqual(s.cursor, START + DAY)
            snap = state.read_plant_csv(p, load_catalog())
            self.assertEqual(snap.n, 7)


# ------------------------------------------------------------------ other backends
@unittest.skipUnless(NATIVE and NUMBA, 'numba backend unavailable')
class SessionNumba(SessionCore):
    backend = 'numba'


@unittest.skipUnless(NATIVE and NUMBA, 'numba backend unavailable')
class RestartNumba(RestartTests):
    backend = 'numba'


@unittest.skipUnless(NATIVE and CUDA, 'CUDA backend unavailable')
class CudaParity(unittest.TestCase):
    def test_cpu_cuda_parity_and_device_residency(self):
        import cupy as cp
        cat = load_catalog()
        cpu_env, gpu_env = Env('numpy'), Env('cuda')
        s_cpu = make_session(cpu_env, cat)
        s_gpu = make_session(gpu_env, cat)
        dev = int(cp.cuda.Device().id)
        self.assertEqual(s_gpu._device, dev)
        self.assertEqual(int(s_gpu._b.B.device.id), dev)
        self.assertEqual(int(s_gpu.root_map.ids.device.id), dev)
        run_days(s_cpu, cpu_env, 8)
        run_days(s_gpu, gpu_env, 8)
        a, b = fp(s_cpu), fp(s_gpu)
        np.testing.assert_allclose(a['water'], b['water'], rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(a['B'], b['B'], rtol=1e-10)
        np.testing.assert_allclose(a['E'], b['E'], rtol=1e-10, atol=1e-14)
        for k in ('offsets', 'resources'):
            np.testing.assert_array_equal(a['roots'][k], b['roots'][k])
        self.assertEqual(a['cursor'], b['cursor'])
        with tempfile.TemporaryDirectory() as td:
            path = s_gpu.save(os.path.join(td, 'cp'))
            loaded = IntereventSession.load(path, cat, operator=gpu_env.operator, backend='cuda', max_edges=10 ** 7,
                                            max_bytes=BIG, verify_native_sources=VERIFY_NATIVE)
            self.assertEqual(int(loaded._b.B.device.id), dev)
            np.testing.assert_array_equal(fp(loaded)['B'], b['B'])
        with self.assertRaises(IntereventError):                                         # host operator, cuda session
            make_session(cpu_env, cat, backend='cuda')


if __name__ == '__main__':
    unittest.main()
