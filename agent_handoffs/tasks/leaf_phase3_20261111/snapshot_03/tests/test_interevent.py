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


# ------------------------------------------------------------------ revision 2: guards, closure, memory
@needs_native
class GuardRegressions(unittest.TestCase):
    backend = 'numpy'

    def setUp(self):
        self.env = Env(self.backend)
        self.catalog = load_catalog()

    # 1. scalar / global admission before any copy
    def test_zero_byte_constructor_refuses_before_population_copy(self):
        plants = make_plants(self.catalog)
        copies = []
        real_copy = state.PlantState.copy

        def observed(self_):
            copies.append(self_.n)
            return real_copy(self_)
        with mock.patch.object(state.PlantState, 'copy', new=observed), \
                mock.patch.object(state.PlantState, 'validate', side_effect=AssertionError('validated before admission')):
            for bad in (0, True, 1.5, -1, np.bool_(True)):
                with self.assertRaises(ValueError):
                    make_session(self.env, self.catalog, plants=plants, max_bytes=bad)
            with self.assertRaises(ValueError):                  # metadata lower bound: refused before validate / copy
                make_session(self.env, self.catalog, plants=plants, max_bytes=500)
            with self.assertRaises(ValueError):
                make_session(self.env, self.catalog, plants=plants, reserve_bytes=-5)
            with self.assertRaises(ValueError):
                make_session(self.env, self.catalog, plants=plants, reserve_bytes=True)
        self.assertEqual(copies, [])

    def test_load_validates_scalars_before_payload_work(self):
        missing = os.path.join(tempfile.gettempdir(), 'leaf-no-such-checkpoint')
        for kw in (dict(max_bytes=True), dict(max_bytes=1.5), dict(max_bytes=0), dict(max_edges=True),
                   dict(max_edges=-1), dict(reserve_bytes=-1), dict(reserve_bytes=2.0)):
            args = dict(operator=self.env.operator, backend=self.backend, max_edges=10 ** 6, max_bytes=BIG)
            args.update(kw)
            with self.assertRaises(IntereventError):         # IntereventError (scalar) not CheckpointError (path)
                IntereventSession.load(missing, self.catalog, **args)

    # 2. overrides are never coerced into other plants
    def test_override_coercion_is_refused_and_valid_overrides_survive(self):
        for ov in ({True: [.5, .5]}, {1.9: [.5, .5]}, {1: [.85, .15], 1.9: [.1, .9]}, {1: [True, False]},
                   {1: [0.5]}, {1: [0.7, 0.7]}, {1: [-0.1, 1.1]}, {1: [float('nan'), 1.0]}, {99: [0.5, 0.5]}, {}):
            with self.assertRaises(ValueError, msg=repr(ov)):
                make_session(self.env, self.catalog, overrides=ov)
        row = [0.1, 0.9]
        ov = {np.int64(2): row}
        s = make_session(self.env, self.catalog, overrides=ov)
        row[0], row[1] = 0.9, 0.1                              # no mutable alias of the caller's data
        ov[3] = [1.0, 0.0]
        self.assertEqual(s._overrides, {2: (0.1, 0.9)})
        run_days(s, self.env, 7)                               # rebuild keeps the override
        h = s.root_map.host_arrays()
        row_i = int(np.flatnonzero(h['ids'] == 2)[0])
        sl = slice(int(h['offsets'][row_i]), int(h['offsets'][row_i + 1]))
        lay = h['resources'][sl] // (NX * NY)
        np.testing.assert_allclose([h['weights'][sl][lay == k].sum() for k in range(L)], [0.1, 0.9], atol=1e-12)

    # 3. pending accumulation finite before publication
    def _huge_session(self):
        doc = self.catalog.document
        g = doc['species']['grass']
        g['growth']['traits'].update(monthly_activity=[1.0] * 12, temperature_min_c=0.0, temperature_opt_c=25.0,
                                     temperature_max_c=50.0, mature_biomass_kg=2.0, biomass_support_kg_year_per_m3=1e-305)
        g['roots']['layer_fractions'] = [.5, .5]
        g['water_use'] = {'law': 'biomass_specific_pet_v1',
                          'traits': {'demand_m3_per_kg_day': 1e308, 'reference_pet_mm_per_day': 1.0}}
        cat = traits.SpeciesCatalog(doc)
        plants = state.PlantState.from_records(cat, state.Domain(1, 1, 1.0), [
            dict(plant_id=1, species='grass', x_m=.5, y_m=.5, establishment_ground_m=0.0, agb_dry_kg=1.0,
                 seasonal_agb_dry_kg=0.0, age_days=100.0, status='live')])
        layers = ss.soil_layers([0.0, .25, 1.0], theta_residual=.05, theta_field_capacity=.3, theta_saturation=.5,
                                diffusivity_m2_s=0.0, theta_evaporation_stress=.1)
        column = ss.prepare_column(layers, np.ones(1, dtype=bool), 1.0, backend='numpy')
        op = st.prepare_transpiration(column)
        native = ss.column_state(column, np.array([[.05], [.15]]))
        s = IntereventSession(plants, cat, operator=op, native_state=native, start=START, backend='numpy',
                              growth_cadence=dt.timedelta(days=7), max_edges=100_000, max_bytes=BIG, root_builder='numpy',
                              verify_native_sources=VERIFY_NATIVE)
        return s

    def test_pending_demand_overflow_refused_boundary_preserved_and_valid_retry(self):
        s = self._huge_session()
        z, e = np.zeros((2, 1)), np.zeros(1)
        s.step_interevent(day(0, pet=1.0, temperature=25.0), added_water_m=z, potential_evaporation_m_s=e)
        self.assertTrue(np.all(np.isfinite(s._b.D)))
        before = fp(s)
        calls = {'step': 0, 'withdraw': 0}
        real_step, real_wd = ss.step, st.withdraw_transpiration

        def cstep(*a, **k):
            calls['step'] += 1
            return real_step(*a, **k)

        def cwd(*a, **k):
            calls['withdraw'] += 1
            return real_wd(*a, **k)
        with mock.patch.object(ss, 'step', cstep), mock.patch.object(st, 'withdraw_transpiration', cwd):
            with self.assertRaises(ValueError):
                s.step_interevent(day(1, pet=1.0, temperature=25.0), added_water_m=z, potential_evaporation_m_s=e)
            self.assertEqual(calls, {'step': 1, 'withdraw': 1})        # native staging was pure; nothing published
            assert_same(self, before, fp(s))
            rec = s.step_interevent(day(1, pet=0.0, temperature=25.0), added_water_m=z, potential_evaporation_m_s=e)
        self.assertEqual(s.sequence, 2)                                  # the valid retry is accounted ONCE
        self.assertEqual(s.native_state.t_s, 2 * 86400.0)
        self.assertTrue(np.all(np.isfinite(s._b.D)))
        self.assertTrue(rec['water_closure']['residual_within_tolerance'])

    def test_accumulate_check_rejects_nonfinite_and_inconsistent_values(self):
        s = make_session(self.env, self.catalog)
        w = s._b
        D = np.full(5, 1.0)
        ok = (np.zeros(2), np.zeros(2))
        rtol = s._overdelivery_rtol(w.root_map)
        with self.assertRaises(IntereventError):
            s._accumulate_checked(w, D, np.full(5, np.inf), *ok, Fraction1(), rtol)
        with self.assertRaises(IntereventError):                          # delivered > requested
            s._accumulate_checked(w, D, np.full(5, 2.0), *ok, Fraction1(), rtol)
        with self.assertRaises(IntereventError):                          # pending beyond the cadence
            s._accumulate_checked(w, D, D, *ok, Fraction1() * 8, rtol)
        s._accumulate_checked(w, D, D, *ok, Fraction1(), rtol)

    def test_same_overdelivery_tolerance_for_growth_and_pending(self):
        s = make_session(self.env, self.catalog)
        rtol = s._overdelivery_rtol(s.root_map)
        self.assertGreaterEqual(rtol, uptake.overdelivery_bound_rel(s.root_map))
        self.assertLessEqual(rtol, growth.MAX_OVERDELIVERY_RTOL)
        seen = []
        real = growth.grow

        def wrap(*a, **k):
            seen.append(k['overdelivery_rtol'])
            return real(*a, **k)
        with mock.patch.object(growth, 'grow', wrap):
            run_days(s, self.env, 7)
        self.assertEqual(seen, [rtol])

    # 4. end-to-end closure is ENFORCED
    def test_misreported_native_budget_is_refused_before_publication(self):
        s = make_session(self.env, self.catalog)
        a, e = self.env.forcing(0.0)
        before = fp(s)
        real_step = ss.step
        n = {'step': 0, 'withdraw': 0}

        def misreport(*args, **kwargs):
            n['step'] += 1
            r = real_step(*args, **kwargs)
            return dataclasses_replace(r, budget=dataclasses_replace(r.budget, evaporation_m3=r.budget.evaporation_m3 + 1e-3))

        real_wd = st.withdraw_transpiration

        def cwd(*args, **kwargs):
            n['withdraw'] += 1
            return real_wd(*args, **kwargs)
        with mock.patch.object(ss, 'step', misreport), mock.patch.object(st, 'withdraw_transpiration', cwd):
            with self.assertRaises(ValueError):
                s.step_interevent(day(0), added_water_m=a, potential_evaporation_m_s=e)
        self.assertEqual(n['step'], 1)
        assert_same(self, before, fp(s))
        s.step_interevent(day(0), added_water_m=a, potential_evaporation_m_s=e)       # a valid retry still works
        self.assertEqual(s.sequence, 1)

    def test_total_residual_perturbation_after_native_stage_is_refused(self):
        s = make_session(self.env, self.catalog)
        a, e = self.env.forcing(0.0)
        before = fp(s)
        real_wd = st.withdraw_transpiration

        def tamper(*args, **kwargs):
            r = real_wd(*args, **kwargs)
            try:
                return dataclasses_replace(r, budget=dataclasses_replace(r.budget, actual_m3=r.budget.actual_m3 + 1e-3))
            except TypeError:
                raise unittest.SkipTest('native TranspirationResult is not a dataclass in this version')
        with mock.patch.object(st, 'withdraw_transpiration', tamper):
            with self.assertRaises(ValueError):
                s.step_interevent(day(0), added_water_m=a, potential_evaporation_m_s=e)
        assert_same(self, before, fp(s))

    # 5. native clock tolerance is FP64 / ULP based
    def test_late_native_clock_half_second_error_refused_exact_handoff_accepted(self):
        env = self.env
        s = make_session(env, self.catalog, native_state=env.state(t_s=1e9))
        end = s.cursor + DAY
        w = s.native_state.water_m.copy()
        with self.assertRaises(IntereventError):
            s.advance_transport('rain', s.cursor, end, native_state=ss.SoilColumnState(w.copy(), 1e9 + 86400.5))
        with self.assertRaises(IntereventError):
            s.advance_transport('rain', s.cursor, end, native_state=ss.SoilColumnState(w.copy(), 1e9 + 86400.0 + 1e-3))
        res = ss.step(env.column, s.native_state, 86400.0, added_water_m=env.xp.zeros((L, env.N)),
                      potential_evaporation_m_s=env.xp.zeros(env.N))
        s.advance_transport('rain', s.cursor, end, native_state=res.state)           # real native endpoint arithmetic
        self.assertEqual(s.native_state.t_s, 1e9 + 86400.0)
        self.assertLess(interevent.TIME_TOL_ULPS * interevent.FP64_EPS * 1e9, 1e-4)  # never seconds

    # 6. snapshot ledger counts the whole materialisation
    def test_snapshot_ledger_counts_materialisation_scratch(self):
        s = make_session(self.env, self.catalog)
        seen = {}
        real = s._check_stage

        def obs(stage, total, mem):
            if stage == 'snapshot':
                seen['total'] = total
            return real(stage, total, mem)
        with mock.patch.object(s, '_check_stage', obs):
            s.snapshot()
        persistent = sum(s._persistent(s._b).values())
        self.assertGreaterEqual(seen['total'] - persistent, interevent.MATERIALIZE_BYTES_PER_PLANT * s._Pa)
        self.assertGreaterEqual(interevent.MATERIALIZE_BYTES_PER_PLANT, 2 * interevent.ROW_BYTES_PER_PLANT + 8)
        s.max_bytes = persistent + s._materialize_peak() - 1                         # one byte short: refused BEFORE work
        with mock.patch.object(state.PlantState, '_arrays', side_effect=AssertionError('materialised before admission')):
            with self.assertRaises(IntereventAdmissionError):
                s.snapshot()
        s.max_bytes = persistent + s._materialize_peak()
        s.snapshot()
        self.assertGreater(s._materialize_peak(), 2 * interevent.ROW_BYTES_PER_PLANT * s._Pa)

    # 7. accumulation stage uses the full bridge ownership
    def test_accumulate_stage_contains_full_bridge_ledger(self):
        s = make_session(self.env, self.catalog)
        fb = 8 * L * self.env.N + 8 * self.env.N
        st_ = s._stages(s._b, fire=False, carry=0, forcing_bytes=fb, native_extra=0)
        self.assertGreaterEqual(st_['accumulate']['total'], st_['bridge']['total'] + 16 * s._Pl)
        # every LN buffer the bridge still owns (availability, request depth, accepted volume) is inside the total
        self.assertGreaterEqual(st_['accumulate']['total'] - st_['bridge']['total'], 16 * s._Pl)


def make_many(catalog, n):
    '''n live grasses on a lattice inside the 4 x 4 m domain (enough roots / serialized state to be non-trivial).'''
    rec = []
    side = int(math.ceil(math.sqrt(n)))
    for i in range(n):
        rec.append(dict(plant_id=i + 1, species='grass' if i % 5 else 'shrub', x_m=0.1 + 3.8 * (i % side) / side,
                        y_m=0.1 + 3.8 * (i // side) / side, establishment_ground_m=100.0,
                        agb_dry_kg=0.2 if i % 5 else 1.0, seasonal_agb_dry_kg=0.0, age_days=100.0 + i, status='live'))
    return state.PlantState.from_records(catalog, state.Domain(NX, NY, 1.0), rec)


@needs_native
class CoexistenceRegressions(unittest.TestCase):
    '''Revision 3: the budget handed to the root builder / restore must count every coexisting candidate.'''
    backend = 'numpy'

    def setUp(self):
        self.env = Env(self.backend)
        self.catalog = load_catalog()

    def _week_with_spy(self, max_bytes=BIG, plants=None):
        s = make_session(self.env, self.catalog, plants=plants, max_bytes=max_bytes)
        run_days(s, self.env, 6)
        p0 = sum(s._persistent(s._b).values())
        seen = []
        real = roots.build_root_map

        def wrap(*a, **k):
            seen.append(k['max_bytes'])
            return real(*a, **k)
        a, e = self.env.forcing(0.0)
        with mock.patch.object(roots, 'build_root_map', wrap):
            rec = s.step_interevent(day(6), added_water_m=a, potential_evaporation_m_s=e)
        return s, p0, seen, rec

    def test_builder_budget_counts_fresh_pending_candidates_native_and_area(self):
        s, p0, seen, rec = self._week_with_spy(plants=make_many(self.catalog, 60))
        self.assertEqual(len(seen), 1)
        forcing_bytes = 8 * L * self.env.N + 8 * self.env.N
        # old persistent + forcing + new native state + B2/E2 + FRESH D/U + area + candidate PlantState (all distinct live)
        minimum = (p0 + forcing_bytes + 8 * L * self.env.N + 16 * s._Pl + 16 * s._Pl + 8 * s._Pl
                   + interevent.ROW_BYTES_PER_PLANT * s._Pa)
        self.assertGreaterEqual(BIG - seen[0], minimum)
        mem = rec['memory']
        self.assertGreaterEqual(mem['builder_external_bytes']['growth_roots_builder_external_bytes'], minimum - p0 - forcing_bytes)

    def test_too_small_global_cap_refuses_before_builder_and_keeps_boundary(self):
        plants = make_many(self.catalog, 60)
        _, _, seen, _ = self._week_with_spy(plants=plants)
        needed = BIG - seen[0]                                  # what the real transaction holds when the builder starts
        s = make_session(self.env, self.catalog, plants=plants)
        run_days(s, self.env, 6)
        before = fp(s)
        a, e = self.env.forcing(0.0)
        called = []
        s.max_bytes = needed - 1
        with mock.patch.object(roots, 'build_root_map', side_effect=lambda *a, **k: called.append(1)):
            with self.assertRaises(IntereventAdmissionError):
                s.step_interevent(day(6), added_water_m=a, potential_evaporation_m_s=e)
        self.assertEqual(called, [])                            # refused before ANY builder work
        s.max_bytes = BIG
        assert_same(self, before, fp(s))
        s.step_interevent(day(6), added_water_m=a, potential_evaporation_m_s=e)    # then a valid retry accounts once
        self.assertEqual(s.sequence, 7)

    def test_peak_estimate_is_the_largest_stage_total_not_a_capacity(self):
        s, _, seen, rec = self._week_with_spy()
        mem = rec['memory']
        self.assertFalse([k for k in mem['stage_totals'] if k.endswith('_builder_budget')])
        self.assertTrue(any(k.endswith('_builder_budget') for k in mem['builder_capacities']))
        self.assertEqual(mem['peak_estimate_bytes'], max(mem['stage_totals'].values()))
        self.assertIn('growth_roots_builder_admitted_peak', mem['stage_totals'])
        self.assertIn('growth_commit', mem['stage_totals'])
        self.assertLess(mem['peak_estimate_bytes'], BIG // 100)             # a cap-sized number would be ~BIG
        self.assertEqual(mem['builder_capacities']['growth_roots_builder_budget'], seen[0])
        self.assertLessEqual(mem['peak_estimate_bytes'], mem['max_bytes'])

    def test_restore_carries_decoded_arrays_through_roots_and_preserves_limits(self):
        s = make_session(self.env, self.catalog, plants=make_many(self.catalog, 60))
        run_days(s, self.env, 3)
        with tempfile.TemporaryDirectory() as td:
            path = s.save(os.path.join(td, 'cp'))
            with np.load(os.path.join(path, 'restart_arrays.npz'), allow_pickle=False) as z:
                decoded = sum(int(z[k].nbytes) for k in z.files)
            seen = []
            real = uptake.RootMap.from_arrays

            def wrap(*a, **k):
                seen.append(k['max_bytes'])
                return real(*a, **k)

            def load(max_bytes, reserve=0):
                return IntereventSession.load(path, self.catalog, operator=self.env.operator, backend=self.backend,
                                              max_edges=10_000_000, max_bytes=max_bytes, reserve_bytes=reserve,
                                              verify_native_sources=VERIFY_NATIVE)
            with mock.patch.object(uptake.RootMap, 'from_arrays', wrap):
                loaded = load(BIG, reserve=12345)
            self.assertEqual(len(seen), 1)
            self.assertGreaterEqual(BIG - seen[0], decoded)                  # decoded arrays are reserved during the upload
            self.assertEqual((loaded.max_bytes, loaded.reserve_bytes, loaded._transient_reserve), (BIG, 12345, 0))
            self.assertNotIn('restore', ''.join(k for k, v in loaded._persistent(loaded._b).items() if v))
            for k in ('water', 'B', 'E', 'D', 'U', 'A', 'T', 'pending'):
                a_, b_ = fp(s)[k], fp(loaded)[k]
                np.testing.assert_array_equal(a_, b_) if isinstance(a_, np.ndarray) else self.assertEqual(a_, b_)
            # a global cap just below what the real restore holds at the upload refuses safely (nothing adopted)
            held = BIG - seen[0]                                # bytes held (reserve included) when the upload starts
            seen.clear()
            with mock.patch.object(uptake.RootMap, 'from_arrays', wrap):
                with self.assertRaises(IntereventAdmissionError):
                    load(held - 1, reserve=12345)
            self.assertEqual(seen, [])
            self.assertEqual(s.sequence, 3)


def Fraction1():
    from fractions import Fraction
    return Fraction(1)


def dataclasses_replace(obj, **kw):
    import dataclasses
    return dataclasses.replace(obj, **kw)


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


# ================================================================== PHASE 2 (disturbance, biomass loss, mortality state)
# Written WITHOUT the ability to execute them (file tools only): NOT RUN by the author. All numbers are UNCALIBRATED
# fixtures (examples/phase2/species.json); the native water authority is the real SYRUP API exactly as above.
from leaf import demography as _demography      # noqa: E402
from leaf import disturbance as _disturbance    # noqa: E402
from leaf import events as _events              # noqa: E402

P2_PATH = ROOT / 'examples' / 'phase2' / 'species.json'
DEC29 = dt.datetime(2026, 12, 29, tzinfo=UTC)


def load_p2(modify=None):
    with open(P2_PATH, 'r', encoding='utf-8') as fh:
        doc = json.load(fh)
    if modify is not None:
        modify(doc)
    return traits.SpeciesCatalog(doc)


def p2_mortality(grass_bg=0.0, shrub_bg=0.0, weibull=False, stress=None):
    '''Catalog modifier: explicit background hazards, no Weibull unless asked; `stress` = {species: dict(updates)} or None.'''
    def mod(doc):
        for sid, rec in doc['species'].items():
            tr = rec['mortality']['traits']
            if not weibull:
                tr['age_hazard'] = None
            tr['background_hazard_per_year'] = grass_bg if sid == 'grass' else shrub_bg
            if stress is None or sid not in stress:
                tr['stress'] = None
            else:
                tr['stress'].update(stress[sid])
    return mod


def step_days(s, env, n, *, temperature=22.0, pet=5.0):
    a, e = env.forcing(0.0)
    out = []
    for _ in range(n):
        rec = forcing.TimestepRecord(s.cursor, s.cursor + DAY, temperature, pet)
        out.append(s.step_interevent(rec, added_water_m=a, potential_evaporation_m_s=e))
    return out


def wind(s, days=1, kind='wind', native_state=None):
    if s.has_pending_growth:
        s.flush_growth()
    start = s.cursor
    return s.advance_transport(kind, start, start + days * DAY, native_state=native_state)


def event_now(s, ids, source='maple_wind', nodose=False, **kw):
    '''A nominal wind event under an applicable abrasion law supplies an explicit ZERO dose in the declared unit when only
    burial / erosion is tested; `nodose=True` builds it without any dose (dedicated missing-dose / flux tests).'''
    tp = s.transport_pending
    if source == 'maple_wind' and not nodose and 'abrasion_dose' not in kw:
        kw['abrasion_dose'] = np.zeros(len(ids))
        kw['abrasion_unit'] = 'kg m-1'
    return _events.SurfaceEventDiagnostics(ids, tp['start_days'], tp['end_days'], source, **kw)


def surf_with(deltas):
    '''(NY, NX) surface at the establishment elevation 100 with {(row, col): delta_m} edits.'''
    s = np.full((NY, NX), 100.0)
    for (r, c), d in deltas.items():
        s[r, c] = 100.0 + d
    return s


# cells (row, col) of make_plants: 1 (0,0) grass, 2 (1,2) grass, 3 (3,3) grass, 4 (2,1) shrub, 5 (0,3) shrub,
# 6 (2,2) dead shrub, 7 (3,0) removed grass
CELL = {1: (0, 0), 2: (1, 2), 3: (3, 3), 4: (2, 1), 5: (0, 3), 6: (2, 2), 7: (3, 0)}


def fp2(s):
    '''fp() plus the Phase 2 accepted state.'''
    out = fp(s)
    b = s._b
    out.update({'p2_seed': s.rng_seed, 'p2_n_annual': s.n_annual_decisions, 'p2_n_dist': s.n_disturbance_events,
                'p2_due': s.annual_due_year, 'p2_tp': s.transport_pending, 'p2_last': s.last_decision,
                'p2_cursor0': s.annual_period_start, 'p2_stress': s.stress_counters(),
                'p2_live_ids': s.live_ids, 'p2_status': s.snapshot().status.copy()})
    acc = s.annual_accumulators()
    if acc is not None:
        for k, v in acc.items():
            out['p2_' + k] = v
    return out


def make_grasses(catalog, n, *, ages=400.0, reverse=False):
    '''n live grasses (B 0.3) on a lattice of the 4 x 4 m domain; ids 1..n; optionally in reversed row order.'''
    rec = []
    side = int(math.ceil(math.sqrt(n)))
    for i in range(n):
        rec.append(dict(plant_id=i + 1, species='grass', x_m=0.1 + 3.8 * (i % side) / side,
                        y_m=0.1 + 3.8 * (i // side) / side, establishment_ground_m=100.0, agb_dry_kg=0.3,
                        seasonal_agb_dry_kg=0.0, age_days=ages, status='live'))
    if reverse:
        rec = rec[::-1]
    return state.PlantState.from_records(catalog, state.Domain(NX, NY, 1.0), rec)


@needs_native
class Phase2Disturbance(unittest.TestCase):
    backend = 'numpy'

    def setUp(self):
        self.env = Env(self.backend)
        self.cat = load_p2(p2_mortality())          # active disturbance; mortality laws active but hazards are zero
        self.seed = 17

    def session(self, **kw):
        kw.setdefault('rng_seed', self.seed)
        return make_session(self.env, self.cat, **kw)

    def ready(self, **kw):
        '''A session right after a wind transport interval (1 day of interevent, flushed, then 1 day of wind).'''
        s = self.session(**kw)
        step_days(s, self.env, 1)
        wind(s)
        return s

    # ---- binding, replay, staleness
    def test_binding_replay_and_refusals_leave_the_boundary_untouched(self):
        s = self.session()
        early = _events.SurfaceEventDiagnostics([1], 0.0, 1.0, 'maple_wind', deposition_depth_m=[0.0625])
        with self.assertRaises(IntereventStateError):                 # no transport interval has been accepted
            s.apply_surface_event(early, surf_with({}))
        step_days(s, self.env, 1)
        wind(s)
        tp = s.transport_pending
        self.assertEqual((tp['kind'], tp['source']), ('wind', 'maple_wind'))
        before = fp2(s)
        ev = lambda **k: _events.SurfaceEventDiagnostics(**k)    # noqa: E731
        t0, t1 = tp['start_days'], tp['end_days']
        bad = [
            (IntereventError, lambda: s.apply_surface_event(event_now(s, [1], source='syrup_water'), surf_with({}))),
            (IntereventStateError, lambda: s.apply_surface_event(
                _events.SurfaceEventDiagnostics([1], t0 - 1, t1 - 1, 'maple_wind'), surf_with({}))),            # stale
            (IntereventStateError, lambda: s.apply_surface_event(
                _events.SurfaceEventDiagnostics([1], t0, t1 + 1, 'maple_wind'), surf_with({}))),                # future
            (IntereventStateError, lambda: s.apply_surface_event(
                _events.SurfaceEventDiagnostics([1], t0 + 0.25, t1, 'maple_wind'), surf_with({}))),           # wrong start
            (IntereventError, lambda: s.apply_surface_event(event_now(s, [99]), surf_with({}))),                # unknown ID
            (IntereventError, lambda: s.apply_surface_event(
                event_now(s, [1], abrasion_dose=[1.0], abrasion_unit='g m-1'), surf_with({}))),                 # wrong unit
            (IntereventError, lambda: s.apply_surface_event(
                event_now(s, [1], nodose=True, transport_flux=[1.0], flux_unit='kg m-1 s-1'), surf_with({}))),  # flux != dose
            (IntereventError, lambda: s.apply_surface_event(event_now(s, [1], nodose=True), surf_with({}))),   # no dose
            (IntereventError, lambda: s.apply_surface_event(event_now(s, [1]), np.zeros((NY + 1, NX)))),        # bad surface
            (IntereventError, lambda: s.apply_surface_event('not an event', surf_with({}))),
        ]
        for exc, call in bad:
            with self.assertRaises(exc):
                call()
        assert_same(self, before, fp2(s))
        rec = s.apply_surface_event(event_now(s, [1], deposition_depth_m=[0.0625]), surf_with({CELL[1]: 0.0625}))
        self.assertEqual(rec['kind'], 'disturbance')
        self.assertEqual(s.sequence, before['seq'] + 1)
        self.assertIsNone(s.transport_pending)
        self.assertEqual(s.n_disturbance_events, 1)
        after = fp2(s)
        with self.assertRaises(IntereventStateError):                  # replay of the same event
            s.apply_surface_event(_events.SurfaceEventDiagnostics([1], t0, t1, 'maple_wind', deposition_depth_m=[0.0625]),
                                  surf_with({CELL[1]: 0.0625}))
        with self.assertRaises(IntereventStateError):                  # a different event for the same interval
            s.apply_surface_event(_events.SurfaceEventDiagnostics([4], t0, t1, 'maple_wind', event_id='other'),
                                  surf_with({}))
        assert_same(self, after, fp2(s))
        step_days(s, self.env, 1)                                       # moving on invalidates any older binding
        wind(s)
        self.assertIsNotNone(s.transport_pending)

    # ---- species contrast, dead standing, water untouched, ages
    def test_species_contrast_dead_standing_water_and_ages(self):
        s = self.ready(plants=make_plants(self.cat, shrub_e=0.2))
        w0, t0 = host(s.native_state.water_m).copy(), s.native_state.t_s
        snap0 = s.snapshot()
        surf = surf_with({CELL[1]: 0.5, CELL[4]: 0.5})
        rec = s.apply_surface_event(event_now(s, [1, 4], deposition_depth_m=[0.5, 0.5]), surf)
        snap1 = s.snapshot()
        self.assertEqual(snap1.status_names()[0], 'dead_standing')            # grass: post-event burial / height >= 0.9
        self.assertEqual(float(snap1.agb_dry_kg[0]), float(snap0.agb_dry_kg[0]))
        self.assertEqual(snap1.status_names()[3], 'live')                     # shrub: tolerant, partial injury
        f = 0.3 * ((0.5 - 0.1) / (0.8 - 0.1))
        self.assertAlmostEqual(float(snap1.agb_dry_kg[3]), float(snap0.agb_dry_kg[3]) * (1.0 - f), places=12)
        self.assertAlmostEqual(float(snap1.seasonal_agb_dry_kg[3]), float(snap0.seasonal_agb_dry_kg[3]) * (1.0 - f),
                               places=12)                                           # E scales with the same retained fraction
        self.assertLessEqual(float(snap1.seasonal_agb_dry_kg[3]), float(snap1.agb_dry_kg[3]))
        np.testing.assert_array_equal(host(s.native_state.water_m), w0)       # native water is never touched
        self.assertEqual(s.native_state.t_s, t0)
        np.testing.assert_array_equal(snap1.x_m, snap0.x_m)
        np.testing.assert_array_equal(snap1.establishment_ground_m, snap0.establishment_ground_m)
        np.testing.assert_array_equal(snap1.ids, snap0.ids)
        self.assertEqual((rec['rebind']['n_live_before'], rec['rebind']['n_live_after']), (5, 4))
        self.assertEqual(s.n_live, 4)
        np.testing.assert_array_equal(s.live_ids, [2, 3, 4, 5])
        led = rec['disturbance']['ledger']
        self.assertLessEqual(abs(led['residual_kg']), led['bound_kg'])
        self.assertAlmostEqual(float(snap0.agb_dry_kg.sum() - snap1.agb_dry_kg.sum()), led['removed_dry_agb_kg'], places=12)
        self.assertTrue(rec['native_water_unchanged'])
        self.assertFalse(rec['rng_consumed'])
        # ages: the dead plant is frozen, live plants continue (no double aging, no reset)
        step_days(s, self.env, 1)
        snap2 = s.snapshot()
        live = snap2.status == state.LIVE
        np.testing.assert_array_equal(snap2.age_days[~live], snap1.age_days[~live])
        np.testing.assert_allclose(snap2.age_days[live], snap1.age_days[live] + 1.0, rtol=0, atol=1e-9)
        self.assertAlmostEqual(snap2.time_days, snap1.time_days + 1.0, places=9)

    def test_dead_plants_request_no_water_and_roots_follow_the_survivors(self):
        s = self.ready()
        s.apply_surface_event(event_now(s, [1], deposition_depth_m=[0.5]), surf_with({CELL[1]: 0.5}))
        store = []
        with mock.patch.object(growth, 'demand', spy(growth.demand, store)):
            step_days(s, self.env, 1)
        self.assertEqual(store[0]['kwargs']['B_kg'].shape, (4,))                 # only live plants request water
        self.assertEqual(store[0]['result']['requested_m3'].shape, (4,))
        np.testing.assert_array_equal(host(s.root_map.ids), [2, 3, 4, 5])
        self.assertEqual(s.root_map.n_plants, 4)

    def test_dislodgement_exports_biomass_and_the_shared_canopy_reflects_it(self):
        s = self.ready(plants=make_plants(self.cat, shrub_e=0.2))
        snap0 = s.snapshot()
        surf = surf_with({CELL[2]: -0.3125})
        rec = s.apply_surface_event(event_now(s, [2], erosion_depth_m=[0.3125]), surf)
        snap1 = s.snapshot()
        self.assertEqual(snap1.status_names()[1], 'removed')
        self.assertEqual((float(snap1.agb_dry_kg[1]), float(snap1.seasonal_agb_dry_kg[1])), (0.0, 0.0))
        self.assertEqual(rec['disturbance']['ledger']['removed_by_cause_kg']['dislodgement'], float(snap0.agb_dry_kg[1]))
        self.assertEqual(float(snap1.age_days[1]), float(snap0.age_days[1]))        # age frozen, record kept
        view = s.export_canopy_view(surf_with({}))
        ids = set(view['sidecar']['plant_id'].tolist())
        self.assertNotIn(2, ids)                                                      # removed: no canopy
        self.assertNotIn(7, ids)
        self.assertIn(6, ids)                                                         # dead standing: still a canopy
        self.assertIn(1, ids)
        self.assertEqual(snap1.n, snap0.n)                                            # identities never compacted

    def test_zero_injury_event_is_a_recorded_decision_without_rebind(self):
        s = self.ready()
        rec = s.apply_surface_event(event_now(s, [1, 4]), surf_with({}))
        self.assertFalse(rec['rebind']['performed'])
        self.assertEqual(s.n_live, 5)
        self.assertEqual(s.n_disturbance_events, 1)
        with self.assertRaises(IntereventStateError):
            s.apply_surface_event(event_now(s, [1, 4]) if s.transport_pending else
                                  _events.SurfaceEventDiagnostics([1, 4], 0.0, 0.0, 'maple_wind'), surf_with({}))

    def test_inactive_catalog_applies_no_response_but_binds_the_decision(self):
        cat1 = load_catalog()
        s = make_session(self.env, cat1)                             # Phase 1 catalog: no seed needed, laws inactive
        self.assertFalse(s.disturbance_active)
        self.assertFalse(s.demography_active)
        self.assertEqual(s._persistent(s._b)['phase2_tables'], 0)
        self.assertEqual(s._persistent(s._b)['phase2_state'], 0)
        step_days(s, self.env, 1)
        wind(s)
        rec = s.apply_surface_event(event_now(s, [1]), surf_with({CELL[1]: 0.9}))
        self.assertEqual(rec['disturbance']['n_injured'], 0)
        self.assertFalse(rec['rebind']['performed'])
        self.assertEqual(s.n_live, 5)
        with self.assertRaises(IntereventError):
            s.apply_annual_decision(2026)
        with self.assertRaises(IntereventError):                     # an ACTIVE mortality law needs an explicit seed
            make_session(self.env, self.cat, rng_seed=None)
        with self.assertRaises(IntereventError):
            make_session(self.env, self.cat, rng_seed=True)

    def test_id_lookup_is_admitted_first_and_never_caches_on_the_accepted_base(self):
        s = self.ready()
        self.assertIsNone(s._base._order)
        self.assertIsNone(s._base._sorted_ids)
        for ev_kw, surf in (({}, surf_with({})), ({'deposition_depth_m': [0.5]}, surf_with({CELL[1]: 0.5}))):
            with mock.patch.object(state.PlantState, 'indices_of', side_effect=AssertionError('cached lookup used')):
                s_ = self.ready()
                s_.apply_surface_event(event_now(s_, [1], **ev_kw), surf)
                self.assertIsNone(s_._base._order)                   # neither a zero-injury nor a rebinding event
                self.assertIsNone(s_._base._sorted_ids)
        # unknown / duplicate IDs still refuse; a one-byte-short cap refuses BEFORE surface validation / lookup scratch
        s = self.ready()
        before = fp2(s)
        with mock.patch.object(state.Domain, 'check_field', side_effect=AssertionError('validated before admission')):
            s.max_bytes = sum(s._persistent(s._b).values())
            with self.assertRaises(IntereventAdmissionError) as cm:
                s.apply_surface_event(event_now(s, [1]), surf_with({}))
            self.assertEqual(cm.exception.stage, 'event_lookup')
            s.max_bytes = BIG
        assert_same(self, before, fp2(s))
        self.assertIsNone(s._base._order)
        with self.assertRaises(IntereventError):
            s.apply_surface_event(event_now(s, [99]), surf_with({}))
        assert_same(self, before, fp2(s))

    # ---- transactional rebind
    def test_rebuild_failure_and_budget_refusal_keep_the_whole_state(self):
        s = self.ready()
        surf = surf_with({CELL[1]: 0.5})
        before = fp2(s)
        wrapper, box = fail_once(roots.build_root_map, roots.RootsError('injected'))
        with mock.patch.object(roots, 'build_root_map', wrapper):
            with self.assertRaises(roots.RootsError):
                s.apply_surface_event(event_now(s, [1], deposition_depth_m=[0.5]), surf)
            assert_same(self, before, fp2(s))
            self.assertEqual(s.n_live, 5)
            np.testing.assert_array_equal(s.live_ids, [1, 2, 3, 4, 5])
            rec = s.apply_surface_event(event_now(s, [1], deposition_depth_m=[0.5]), surf)      # the retry accounts ONCE
        self.assertEqual(box['n'], 1)
        self.assertEqual(s.n_disturbance_events, 1)
        self.assertEqual(s.sequence, before['seq'] + 1)
        self.assertEqual(s.n_live, 4)
        # a global cap one byte below the rebind preflight refuses before ANY geometry / builder work
        s2 = self.ready()
        limit = rec['memory']['stage_totals']['disturbance_rebind_preflight'] - 1
        before2 = fp2(s2)
        s2.max_bytes = limit
        with mock.patch.object(roots, 'build_root_map', side_effect=AssertionError('builder must not run')), \
                mock.patch.object(state.PlantState, 'update_plants', side_effect=AssertionError('no state edit')):
            with self.assertRaises(IntereventAdmissionError):
                s2.apply_surface_event(event_now(s2, [1], deposition_depth_m=[0.5]), surf)
        s2.max_bytes = BIG
        assert_same(self, before2, fp2(s2))
        s2.apply_surface_event(event_now(s2, [1], deposition_depth_m=[0.5]), surf)
        self.assertEqual(s2.n_live, 4)
        s3 = self.ready()
        s3.max_bytes = 100
        with self.assertRaises(IntereventAdmissionError):
            s3.apply_surface_event(event_now(s3, [1], deposition_depth_m=[0.5]), surf)

    def test_rebind_preserves_grid_deferred_season_and_pending_state(self):
        start = dt.datetime(2026, 12, 30, tzinfo=UTC)
        s = self.session(plants=make_plants(self.cat, shrub_e=0.2), start=start)
        step_days(s, self.env, 1)
        wind(s, days=4)                                               # 31 Dec .. 4 Jan: a year end INSIDE transport
        self.assertTrue(s.season_due)
        self.assertEqual(s.annual_due_year, 2026)
        grid, season, cursor, t_native = s.boundaries()['next_growth_boundary'], s.season_due, s.cursor, s.native_state.t_s
        w0 = host(s.native_state.water_m).copy()
        rec = s.apply_surface_event(event_now(s, [1], deposition_depth_m=[0.5]), surf_with({CELL[1]: 0.5}))
        self.assertEqual(rec['rebind']['n_live_after'], 4)
        self.assertEqual((s.boundaries()['next_growth_boundary'], s.season_due, s.cursor, s.native_state.t_s),
                         (grid, season, cursor, t_native))
        np.testing.assert_array_equal(host(s.native_state.water_m), w0)
        self.assertEqual(s.pending_days, 0)
        self.assertEqual(s.annual_due_year, 2026)                       # a disturbance never consumes the annual decision
        a, e = self.env.forcing(0.0)
        rec_open = forcing.TimestepRecord(s.cursor, s.cursor + DAY, 5.0, 5.0)
        with self.assertRaises(IntereventStateError):                  # the completed year's decision is due first
            s.step_interevent(rec_open, added_water_m=a, potential_evaporation_m_s=e)
        s.apply_annual_decision(2026)
        r = s.step_interevent(rec_open, added_water_m=a, potential_evaporation_m_s=e)
        self.assertTrue(r['deferred_season_applied'])                   # the turnover is applied at the next open interval
        self.assertEqual(s.pending_days, 1)                             # only the open day accrued (no catch-up)

    # ---- checkpoint of the Phase 2 accepted state
    def _load(self, path, catalog=None, **kw):
        args = dict(operator=self.env.operator, backend=self.backend, max_edges=10_000_000, max_bytes=BIG,
                    verify_native_sources=VERIFY_NATIVE)
        args.update(kw)
        return IntereventSession.load(path, catalog or self.cat, **args)

    def test_checkpoint_restores_decision_state_and_continues_identically(self):
        with tempfile.TemporaryDirectory() as td:
            s = self.ready(plants=make_plants(self.cat, shrub_e=0.2))
            pending_path = s.save(os.path.join(td, 'pending'))               # transport decision still outstanding
            surf = surf_with({CELL[1]: 0.5, CELL[4]: 0.25})
            ev = event_now(s, [1, 4], deposition_depth_m=[0.5, 0.25])
            loaded = self._load(pending_path)
            self.assertEqual(loaded.transport_pending, s.transport_pending)
            self.assertEqual(loaded.rng_seed, self.seed)
            s.apply_surface_event(ev, surf)
            loaded.apply_surface_event(ev, surf)
            a, b = fp2(s), fp2(loaded)
            for k in ('B', 'E', 'p2_status', 'p2_live_ids', 'p2_n_dist', 'p2_seed', 'water', 't'):
                np.testing.assert_array_equal(a[k], b[k], err_msg=k)
            np.testing.assert_allclose(a['age'], b['age'], rtol=1e-13)
            done_path = s.save(os.path.join(td, 'done'))
            again = self._load(done_path)
            self.assertEqual(again.n_disturbance_events, 1)
            self.assertIsNone(again.transport_pending)
            self.assertEqual(again.last_decision, s.last_decision)
            np.testing.assert_array_equal(again.live_ids, s.live_ids)
            with self.assertRaises(IntereventStateError):                      # the replay guard survived the restart
                again.apply_surface_event(ev, surf)
            with self.assertRaises(checkpoint.CheckpointError):                # the seed is accepted state
                self._load(done_path, rng_seed=self.seed + 1)

    def test_checkpoint_phase2_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            s = self.ready()
            path = pathlib.Path(s.save(os.path.join(td, 'cp')))

            def rehash(doc):
                body = {k: v for k, v in doc.items() if k != 'content_hash'}
                doc['content_hash'] = checkpoint._sha(checkpoint._canonical(body))
                (path / 'restart.json').write_text(json.dumps(doc))
            doc = json.loads((path / 'restart.json').read_text())
            doc['phase2']['transport_pending']['kind'] = 'rain'            # source / kind inconsistency
            rehash(doc)
            with self.assertRaises(checkpoint.CheckpointError):
                self._load(path)
            doc = json.loads((path / 'restart.json').read_text())
            doc['phase2']['transport_pending']['kind'] = 'wind'
            del doc['phase2']['rng_seed']
            rehash(doc)
            with self.assertRaises(checkpoint.CheckpointError):
                self._load(path)

    def test_checkpoint_format_version_three_is_explicit_and_strict(self):
        # Phase 3 bumps the complete-checkpoint layout to version 3 (adds the `phase3` reproduction block); versions 1 and
        # 2 are refused without migration (see Phase3Reproduction.test_old_checkpoint_versions_are_refused_not_migrated)
        self.assertEqual(checkpoint.SCHEMA_VERSION, 3)
        with tempfile.TemporaryDirectory() as td:
            s = self.ready()
            path = pathlib.Path(s.save(os.path.join(td, 'cp')))
            doc0 = json.loads((path / 'restart.json').read_text())
            self.assertIs(type(doc0['schema_version']), int)
            self.assertEqual(doc0['schema_version'], 3)
            self.assertEqual(self._load(path).sequence, s.sequence)                  # round trip
            with open(path / 'plants.csv', 'r', encoding='utf-8') as fh:
                self.assertEqual(json.loads(fh.readline()[2:])['schema_version'], 1)  # canonical CSV stays version 1
            # internally hashed manifests with another version are refused BEFORE arrays are decoded / state allocated
            for bad in (1, 2, 4, 3.0, True, '3', None):
                doc = json.loads((path / 'restart.json').read_text())
                doc['schema_version'] = bad
                body = {k: v for k, v in doc.items() if k != 'content_hash'}
                doc['content_hash'] = checkpoint._sha(checkpoint._canonical(body))
                (path / 'restart.json').write_text(json.dumps(doc))
                with mock.patch.object(np, 'load', side_effect=AssertionError('arrays decoded before version check')), \
                        mock.patch.object(state, 'parse_plant_csv', side_effect=AssertionError('population parsed')):
                    with self.assertRaises(checkpoint.CheckpointError) as cm:
                        self._load(path)
                self.assertIn('version', str(cm.exception))
                self.assertIn('3', str(cm.exception))
            (path / 'restart.json').write_text(json.dumps(doc0))
            self._load(path)                                                         # original restored: loads again

    def test_describe_and_memory_are_json_safe_and_linear(self):
        cat = load_p2(p2_mortality(stress={'grass': {}, 'shrub': {}}))     # stress laws ACTIVE: live-sized state exists
        s = make_session(self.env, cat, rng_seed=self.seed)
        step_days(s, self.env, 1)
        wind(s)
        json.dumps(s.describe(), allow_nan=False)
        p = s._persistent(s._b)
        self.assertEqual(p['phase2_state'], _demography.PERSISTENT_BYTES_PER_LIVE * s._Pl)     # stress laws active
        self.assertEqual(p['phase2_tables'], s._p2_table_bytes)
        for name in ('_history', 'history', 'event_ids', 'decisions'):
            self.assertFalse(hasattr(s, name))                           # no whole-history / event-ID list


@needs_native
class Phase2Annual(unittest.TestCase):
    backend = 'numpy'

    def setUp(self):
        self.env = Env(self.backend)

    def year_end_session(self, cat, plants=None, seed=5):
        s = make_session(self.env, cat, plants=plants, start=DEC29, rng_seed=seed)
        step_days(s, self.env, 3)                      # 29, 30, 31 December: the year-end growth update fires at 1 Jan
        return s

    def test_due_flag_ordering_replay_and_wrong_year_refusals(self):
        cat = load_p2(p2_mortality())
        s = make_session(self.env, cat, start=DEC29, rng_seed=5)
        self.assertIsNone(s.annual_due_year)
        step_days(s, self.env, 2)
        self.assertIsNone(s.annual_due_year)
        r = step_days(s, self.env, 1)[0]
        self.assertTrue(r['growth']['year_end'])
        self.assertEqual(s.annual_due_year, 2026)
        self.assertEqual(r['annual_decision_due_year'], 2026)
        before = fp2(s)
        a, e = self.env.forcing(0.0)
        with self.assertRaises(IntereventStateError):
            s.step_interevent(forcing.TimestepRecord(s.cursor, s.cursor + DAY, 5.0, 5.0), added_water_m=a,
                              potential_evaporation_m_s=e)
        for bad in (2025, 2027, True, 2026.0, '2026'):
            with self.assertRaises(IntereventError):
                s.apply_annual_decision(bad)
        assert_same(self, before, fp2(s))
        self.assertEqual(s.n_annual_decisions, 0)
        rec = s.apply_annual_decision(2026)
        self.assertEqual((rec['decision_counter'], rec['decision_counter_after']), (0, 1))
        self.assertEqual(s.n_annual_decisions, 1)
        self.assertIsNone(s.annual_due_year)
        self.assertEqual(s.annual_period_start, s.cursor)
        with self.assertRaises(IntereventStateError):                      # replay
            s.apply_annual_decision(2026)
        step_days(s, self.env, 1)                                           # open intervals resume
        w = fp2(s)
        self.assertIsNone(w['p2_due'])

    def test_outstanding_decision_is_never_overwritten_by_transport_and_multi_year_events_are_refused(self):
        cat = load_p2(p2_mortality())
        s = self.year_end_session(cat)                       # 1 Jan 2027, the decision for 2026 is outstanding
        self.assertEqual(s.annual_due_year, 2026)
        before = fp2(s)
        for kind, days in (('wind', 1), ('wind', 400)):
            with self.assertRaises(IntereventStateError):
                s.advance_transport(kind, s.cursor, s.cursor + days * DAY)
        with self.assertRaises(IntereventStateError):          # a closed-gate pause is a calendar advance as well
            s.step_interevent(forcing.TimestepRecord(s.cursor, s.cursor + DAY, 0.0, None, 'interevent', 'closed'))
        assert_same(self, before, fp2(s))                      # the original year and observation period are preserved
        self.assertEqual(s.annual_due_year, 2026)
        # exact restart keeps the outstanding decision and the same refusals
        with tempfile.TemporaryDirectory() as td:
            path = s.save(os.path.join(td, 'cp'))
            loaded = IntereventSession.load(path, cat, operator=self.env.operator, backend=self.backend,
                                            max_edges=10_000_000, max_bytes=BIG, verify_native_sources=VERIFY_NATIVE)
            self.assertEqual(loaded.annual_due_year, 2026)
            self.assertEqual(loaded.annual_period_start, s.annual_period_start)
            with self.assertRaises(IntereventStateError):
                loaded.advance_transport('wind', loaded.cursor, loaded.cursor + DAY)
        # valid order: decision first, then a transport crossing exactly ONE new year; two years are refused unchanged
        s.apply_annual_decision(2026)
        one_year = dt.datetime(2028, 1, 2, tzinfo=UTC) - s.cursor
        two_years = dt.datetime(2029, 1, 2, tzinfo=UTC) - s.cursor
        before = fp2(s)
        with self.assertRaises(IntereventStateError):
            s.advance_transport('wind', s.cursor, s.cursor + two_years)
        assert_same(self, before, fp2(s))
        s.advance_transport('wind', s.cursor, s.cursor + one_year)
        self.assertEqual(s.annual_due_year, 2027)            # exactly the one completed year, no relabelling
        self.assertEqual(s.annual_period_start, dt.datetime(2027, 1, 1, tzinfo=UTC))
        s.apply_annual_decision(2027)
        self.assertIsNone(s.annual_due_year)

    def test_pending_growth_must_be_flushed_before_the_decision(self):
        cat = load_p2(p2_mortality())
        s = make_session(self.env, cat, start=DEC29, rng_seed=5)
        step_days(s, self.env, 3)
        self.assertFalse(s.has_pending_growth)                  # the year-end growth already flushed the period
        b = s._b.clone()
        b.pending_days = Fraction1()                             # simulate pending growth: must be refused
        real = s._b
        s._b = b
        try:
            with self.assertRaises(IntereventStateError):
                s.apply_annual_decision(2026)
        finally:
            s._b = real
        s.apply_annual_decision(2026)

    def test_probability_one_kills_grass_keeps_B_E_and_shrubs_survive(self):
        cat = load_p2(p2_mortality(grass_bg=1e9))
        plants = make_plants(cat, shrub_e=0.2)
        s = self.year_end_session(cat, plants)
        snap0, w0, t0 = s.snapshot(), host(s.native_state.water_m).copy(), s.native_state.t_s
        rec = s.apply_annual_decision(2026)
        snap1 = s.snapshot()
        self.assertEqual(snap1.status_names().tolist(),
                         ['dead_standing'] * 3 + ['live', 'live', 'dead_standing', 'removed'])
        np.testing.assert_array_equal(snap1.agb_dry_kg, snap0.agb_dry_kg)       # dead plants keep their B / E
        np.testing.assert_array_equal(snap1.seasonal_agb_dry_kg, snap0.seasonal_agb_dry_kg)
        self.assertEqual(rec['n_died_age_background'], 3)
        np.testing.assert_array_equal(s.live_ids, [4, 5])
        np.testing.assert_array_equal(host(s.native_state.water_m), w0)
        self.assertEqual(s.native_state.t_s, t0)
        np.testing.assert_array_equal(snap1.age_days[:3], snap0.age_days[:3])        # ages freeze at death
        store = []
        with mock.patch.object(growth, 'demand', spy(growth.demand, store)):
            step_days(s, self.env, 1)
        self.assertEqual(store[0]['kwargs']['B_kg'].shape, (2,))                      # dead grasses request nothing
        self.assertEqual(s.root_map.n_plants, 2)
        view = s.export_canopy_view(surf_with({}))
        self.assertTrue({1, 2, 3}.issubset(set(view['sidecar']['plant_id'].tolist())))   # standing dead stay in the view

    def test_probability_zero_kills_nobody_and_mean_life_is_not_a_cutoff(self):
        cat = load_p2(p2_mortality(grass_bg=0.0, weibull=False))
        rec_plants = [dict(plant_id=1, species='grass', x_m=0.8, y_m=0.8, establishment_ground_m=100.0, agb_dry_kg=0.3,
                           seasonal_agb_dry_kg=0.0, age_days=40 * 365.0, status='live')]      # 5 x the mean life expectancy
        plants = state.PlantState.from_records(cat, state.Domain(NX, NY, 1.0), rec_plants)
        s = self.year_end_session(cat, plants)
        rec = s.apply_annual_decision(2026)
        self.assertEqual(s.n_live, 1)
        self.assertEqual(rec['n_died_age_background'] + rec['n_died_stress'], 0)
        self.assertFalse(rec['rebind']['performed'])

    def test_keyed_decisions_are_row_order_independent_and_split_run_equivalent(self):
        bg = math.log(2.0) * 365.0 / 3.0            # 3-day decision period: p = 1/2 for every grass
        cat = load_p2(p2_mortality(grass_bg=bg))
        n = 40
        a = self.year_end_session(cat, make_grasses(cat, n))
        b = self.year_end_session(cat, make_grasses(cat, n, reverse=True))
        a.apply_annual_decision(2026)
        b.apply_annual_decision(2026)
        sa, sb = a.snapshot(), b.snapshot()
        dead_a = set(sa.ids[sa.status == state.DEAD_STANDING].tolist())
        dead_b = set(sb.ids[sb.status == state.DEAD_STANDING].tolist())
        self.assertEqual(dead_a, dead_b)                    # same keyed draws whatever the row order
        self.assertTrue(0 < len(dead_a) < n)
        # split run: save with the decision outstanding, restore, decide there: identical deaths and counters
        with tempfile.TemporaryDirectory() as td:
            c = self.year_end_session(cat, make_grasses(cat, n))
            path = c.save(os.path.join(td, 'cp'))
            loaded = IntereventSession.load(path, cat, operator=self.env.operator, backend=self.backend,
                                            max_edges=10_000_000, max_bytes=BIG, verify_native_sources=VERIFY_NATIVE)
            self.assertEqual(loaded.annual_due_year, 2026)
            c.apply_annual_decision(2026)
            loaded.apply_annual_decision(2026)
            np.testing.assert_array_equal(c.snapshot().status, loaded.snapshot().status)
            self.assertEqual((c.n_annual_decisions, loaded.n_annual_decisions), (1, 1))
            np.testing.assert_array_equal(c.snapshot().status, sa.status)
        # a different seed is a different (but equally reproducible) draw
        d = self.year_end_session(cat, make_grasses(cat, n), seed=6)
        d.apply_annual_decision(2026)
        self.assertNotEqual(set(d.snapshot().ids[d.snapshot().status == state.DEAD_STANDING].tolist()), dead_a)

    def test_stress_accumulators_use_actual_growth_outputs_once_and_follow_the_documented_rule(self):
        stress = {'grass': dict(exempt_until_age_days=0.0, limit_years=1, counter_cap_years=3, mortality_probability=0.0,
                                biomass_loss_fraction=0.0, low_threshold=0.5, metric='support_ratio'),
                  'shrub': dict(exempt_until_age_days=0.0, limit_years=1, counter_cap_years=3, mortality_probability=0.0,
                                biomass_loss_fraction=0.0, low_threshold=0.4, metric='growth_ratio')}
        cat = load_p2(p2_mortality(stress=stress))
        env = Env(self.backend, theta=(0.0502, 0.0502))                        # dry soil: actual support << requested
        s = make_session(env, cat, start=dt.datetime(2026, 12, 18, tzinfo=UTC), rng_seed=3)
        store = []
        with mock.patch.object(growth, 'grow', spy(growth.grow, store)):
            step_days(s, env, 14)                                               # two weekly updates, the last at year end
        self.assertEqual(len(store), 2)
        acc = s.annual_accumulators()
        exp = {'required_support_kg_year': sum(r['result']['required_support_kg_year'] for r in store),
               'actual_support_kg_year': sum(r['result']['actual_support_kg_year'] for r in store),
               'planned_growth_kg': sum(r['result']['normal_plan_kg'] + r['result']['extra_plan_kg'] for r in store),
               'actual_growth_kg': sum(r['result']['normal_gain_kg'] + r['result']['extra_gain_kg'] for r in store)}
        for k, v in exp.items():
            np.testing.assert_allclose(acc[k], v, rtol=1e-12, atol=0.0, err_msg=k)         # accumulated exactly once
        self.assertEqual(s.annual_due_year, 2026)
        snap = s.snapshot()
        live = snap.status == state.LIVE
        age, sp = snap.age_days[live], snap.species_index[live]
        grass = sp == cat.index('grass')
        req, act, plan, gain = (acc['required_support_kg_year'], acc['actual_support_kg_year'],
                                acc['planned_growth_kg'], acc['actual_growth_kg'])
        n0 = np.zeros(s.n_live, dtype=np.int64)
        den = np.where(grass, req, plan)
        num = np.where(grass, act, gain)
        thr = np.where(grass, 0.5, 0.4)
        evaluated = (age > 0.0) & (den > 0.0)
        low = evaluated & (num / np.where(den > 0.0, den, 1.0) < thr)
        expected = np.where(low, np.minimum(n0 + 1, 3), np.where(evaluated, np.maximum(n0 - 1, 0), n0))
        rec = s.apply_annual_decision(2026)
        np.testing.assert_array_equal(s.stress_counters(), expected)
        for name in ('required_support_kg_year', 'actual_support_kg_year', 'planned_growth_kg', 'actual_growth_kg'):
            self.assertTrue(np.all(s.annual_accumulators()[name] == 0.0))               # reset after the decision
        # December is dormant for the grass (activity 0): required support is zero, so it is NOT counted as stress
        self.assertEqual(rec['per_species']['grass']['n_not_evaluated'], 3)
        self.assertTrue(np.all(s.stress_counters()[grass] == 0))


# ---- other backends for the Phase 2 session tests (skip when unavailable)
@unittest.skipUnless(NATIVE and NUMBA, 'numba backend unavailable')
class Phase2DisturbanceNumba(Phase2Disturbance):
    backend = 'numba'


@unittest.skipUnless(NATIVE and NUMBA, 'numba backend unavailable')
class Phase2AnnualNumba(Phase2Annual):
    backend = 'numba'


@unittest.skipUnless(NATIVE and CUDA, 'CUDA backend unavailable')
class Phase2CudaParity(unittest.TestCase):
    def test_cpu_cuda_phase2_parity_and_device_residency(self):
        import cupy as cp
        cat = load_p2(p2_mortality(grass_bg=math.log(2.0) * 365.0 / 3.0))
        cpu_env, gpu_env = Env('numpy'), Env('cuda')
        out = {}
        for name, env in (('cpu', cpu_env), ('gpu', gpu_env)):
            s = make_session(env, cat, plants=make_grasses(cat, 24), start=DEC29, rng_seed=9)
            step_days(s, env, 3)
            s.apply_annual_decision(2026)
            out[name] = s
        np.testing.assert_array_equal(out['cpu'].snapshot().status, out['gpu'].snapshot().status)
        gpu = out['gpu']
        self.assertIsNone(gpu._b.stress)                  # no stress law: the counter is correctly NOT allocated
        self.assertIsNone(gpu.stress_counters())
        self.assertEqual(int(gpu._b.B.device.id), int(cp.cuda.Device().id))

    def test_active_stress_law_is_device_resident_through_decision_and_restart(self):
        import cupy as cp
        dev = int(cp.cuda.Device().id)
        cat = load_p2(p2_mortality(grass_bg=math.log(2.0) * 365.0 / 3.0, stress={'grass': {}, 'shrub': {}}))
        env = Env('cuda')
        s = make_session(env, cat, plants=make_grasses(cat, 24), start=DEC29, rng_seed=9)
        step_days(s, env, 3)
        for arr in (s._b.stress, s._b.ann_req, s._b.ann_act, s._b.ann_plan, s._b.ann_gain):
            self.assertEqual(int(arr.device.id), dev)
        s.apply_annual_decision(2026)
        for arr in (s._b.stress, s._b.ann_req, s._b.ann_act, s._b.ann_plan, s._b.ann_gain):
            self.assertEqual(int(arr.device.id), dev)
            self.assertEqual(arr.shape, (s.n_live,))
        with tempfile.TemporaryDirectory() as td:
            path = s.save(os.path.join(td, 'cp'))
            loaded = IntereventSession.load(path, cat, operator=env.operator, backend='cuda', max_edges=10 ** 7,
                                            max_bytes=BIG, verify_native_sources=VERIFY_NATIVE)
            for arr in (loaded._b.stress, loaded._b.ann_req, loaded._b.ann_gain):
                self.assertEqual(int(arr.device.id), dev)
            np.testing.assert_array_equal(loaded.stress_counters(), s.stress_counters())
            np.testing.assert_array_equal(loaded.snapshot().status, s.snapshot().status)
            self.assertEqual(loaded.n_annual_decisions, 1)


# ------------------------------------------------------------------ Phase 3: annual reproduction (seeds + daughters)
P3_START = dt.datetime(2026, 12, 29, tzinfo=UTC)
P3_LAW = 'simple_seed_clonal_v1'
P3_LIMS = dict(max_new_plants=1000, max_total_records=100000, max_attempts=100000)
P3_SURFACE = 100.0 + 0.01 * np.arange(NX * NY, dtype=np.float64).reshape(NY, NX)       # SYNTHETIC non-constant ground


def p3_modify(*, seed_p=1.0, clonal_p=1.0, bg=None):
    '''UNCALIBRATED fixture reproduction numbers on the Phase 1 catalog: grass = clonal only, shrub = seed only.'''
    def mod(doc):
        g, sh = doc['species']['grass'], doc['species']['shrub']
        g['reproduction'] = {'law': P3_LAW, 'traits': {'seed': None, 'clonal': {
            'min_age_days': 250.0, 'min_agb_dry_kg': 0.2, 'probability': clonal_p, 'max_daughters_per_parent': 2,
            'initial_agb_dry_kg': 0.05, 'max_donation_fraction': 0.4, 'dispersal_radius_m': 0.5,
            'max_attempts_per_daughter': 8}}}
        sh['reproduction'] = {'law': P3_LAW, 'traits': {'seed': {
            'establishment_probability': seed_p, 'max_recruits_per_patch': 2, 'initial_agb_dry_kg': 0.03,
            'initial_seasonal_fraction': 0.5}, 'clonal': None}}
        if bg is not None:
            g['mortality'] = {'law': 'age_background_stress_v1', 'traits': {
                'age_hazard': None, 'background_hazard_per_year': bg, 'stress': None}}
    return mod


def p3_step(s, env, n):
    out = []
    for _ in range(n):
        a, e = env.forcing(0.0)
        t0 = s.cursor
        rec = forcing.TimestepRecord(t0, t0 + DAY, 22.0, 5.0)
        out.extend(s.step_interevent(p, added_water_m=a, potential_evaporation_m_s=e) for p in s.split_record(rec))
    return out


def p3_empty_plants(catalog):
    z = np.zeros(0)
    return state.PlantState(state.Domain(NX, NY, 1.0), catalog.species_ids, catalog.hash, ids=np.zeros(0, np.int64),
                            species_index=np.zeros(0, np.int32), x_m=z, y_m=z, establishment_ground_m=z,
                            agb_dry_kg=z, age_days=z, status=np.zeros(0, np.int8))


def p3_direct(env, cat, plants, **kw):
    return IntereventSession(plants, cat, operator=env.operator, native_state=env.state(), start=P3_START,
                             backend=env.backend, growth_cadence=dt.timedelta(days=7), max_edges=10_000_000,
                             max_bytes=BIG, root_builder='numpy', verify_native_sources=VERIFY_NATIVE, **kw)


@needs_native
class Phase3Reproduction(unittest.TestCase):
    backend = 'numpy'

    def setUp(self):
        self.env = Env(self.backend)
        self.cat = load_catalog(p3_modify())

    def year_end(self, cat=None, plants=None, seed=5):
        cat = cat or self.cat
        s = make_session(self.env, cat, plants=plants, start=P3_START, rng_seed=seed, reproduction_patch_size_m=2.0)
        p3_step(s, self.env, 3)                          # 29, 30, 31 December: the year-end growth fires at 1 January
        return s

    def reproduce(self, s, year=2026, **over):
        lim = dict(P3_LIMS)
        lim.update(over)
        return s.apply_reproduction(year, P3_SURFACE, **lim)

    def load(self, path, cat=None, **kw):
        return IntereventSession.load(path, cat or self.cat, operator=self.env.operator, backend=self.backend,
                                      max_edges=10_000_000, max_bytes=BIG, verify_native_sources=VERIFY_NATIVE, **kw)

    # ---- the transaction
    def test_seed_and_clonal_births_with_ledger_unchanged_water_and_next_step_uptake(self):
        s = self.year_end()
        s0, w0, t0, cur0 = s.snapshot(), host(s.native_state.water_m).copy(), s.native_state.t_s, s.cursor
        rep = self.reproduce(s)
        d = rep['reproduction']
        s1 = s.snapshot()
        n0, nb = s0.n, int(d['n_births'])
        self.assertEqual(s1.n, n0 + nb)
        self.assertGreaterEqual(d['n_seed_recruits'], 4)                 # 4 patches of 2 m, p = 1, 1..2 recruits each
        self.assertLessEqual(d['n_seed_recruits'], 8)
        self.assertTrue(2 <= d['n_daughters_placed'] <= 4)               # parents 1 and 3 are eligible (age, size)
        self.assertTrue(d['ledger']['closed'])
        # existing records: IDs, locations, ages, anchors, statuses persist; dead / removed retained deterministically
        for k in ('ids', 'species_index', 'x_m', 'y_m', 'establishment_ground_m', 'age_days', 'status'):
            np.testing.assert_array_equal(getattr(s1, k)[:n0], getattr(s0, k), err_msg=k)
        donors = np.flatnonzero(s1.agb_dry_kg[:n0] != s0.agb_dry_kg)
        self.assertTrue(set(s0.ids[donors].tolist()) <= {1, 3})
        self.assertTrue(np.all(s1.agb_dry_kg[donors] > 0.0))
        self.assertTrue(np.all(s1.seasonal_agb_dry_kg[:n0] <= s1.agb_dry_kg[:n0]))
        # newborns: own consecutive IDs above the maximum, age 0, live, ground sampled from the CURRENT surface
        np.testing.assert_array_equal(s1.ids[n0:], np.arange(int(s0.ids.max()) + 1, int(s0.ids.max()) + 1 + nb))
        self.assertTrue(np.all(s1.age_days[n0:] == 0.0))
        self.assertTrue(np.all(s1.status[n0:] == state.LIVE))
        np.testing.assert_array_equal(s1.establishment_ground_m[n0:],
                                      s1.domain.sample(P3_SURFACE, s1.x_m[n0:], s1.y_m[n0:]))
        daughters = s1.species_index[n0 + d['n_seed_recruits']:]
        self.assertTrue(np.all(daughters == self.cat.index('grass')))
        self.assertTrue(np.all(s1.species_index[n0:n0 + d['n_seed_recruits']] == self.cat.index('shrub')))
        # ledger: old total + external seed credit = new retained total (clonal transfer cancels)
        np.testing.assert_allclose(float(s1.agb_dry_kg.sum()), float(s0.agb_dry_kg.sum()) + d['ledger']['external_seed_agb_kg'],
                                   rtol=1e-12)
        # no phantom water, time, calendar or growth-grid change
        np.testing.assert_array_equal(host(s.native_state.water_m), w0)
        self.assertEqual((s.native_state.t_s, s.cursor), (t0, cur0))
        self.assertTrue(rep['rng_consumed'] and rep['native_water_unchanged'])
        self.assertEqual((s.n_reproduction_opportunities, s.reproduction_last_year), (1, 2026))
        self.assertEqual(s.n_live, 5 + nb)
        self.assertEqual(s.root_map.n_plants, s.n_live)
        # the NEXT open interval: new plants request water (D / U sized by the new live population)
        store = []
        with mock.patch.object(growth, 'demand', spy(growth.demand, store)):
            p3_step(s, self.env, 1)
        self.assertEqual(store[0]['kwargs']['B_kg'].shape, (s.n_live,))
        self.assertEqual(host(s._b.U).shape, (s.n_live,))

    def test_replay_stale_wrong_type_and_limit_refusals_leave_everything_unchanged(self):
        s = self.year_end()
        before = fp(s)
        for bad in (True, 2026.0, '2026', None):
            with self.assertRaises(IntereventError):
                s.apply_reproduction(bad, P3_SURFACE, **P3_LIMS)
        for bad in (True, 1.5, -1, None):
            with self.assertRaises(IntereventError):
                self.reproduce(s, max_new_plants=bad)
        for kw in (dict(max_new_plants=0), dict(max_total_records=7), dict(max_attempts=0)):
            with self.assertRaises(IntereventError):
                self.reproduce(s, **kw)
        with self.assertRaises(IntereventStateError):
            self.reproduce(s, year=2027)                       # not yet completed
        with self.assertRaises(IntereventStateError):
            self.reproduce(s, year=2025)                       # before the first opportunity (replay / stale)
        assert_same(self, before, fp(s))
        self.assertEqual(s.n_reproduction_opportunities, 0)    # rejected calls consumed nothing
        self.reproduce(s)
        with self.assertRaises(IntereventStateError):          # replay: an annual chance is never charged twice
            self.reproduce(s)
        self.assertEqual(s.n_reproduction_opportunities, 1)

    def test_ordering_pending_growth_pending_transport_and_unresolved_mortality(self):
        cat = load_catalog(p3_modify(bg=0.01))
        s = make_session(self.env, cat, start=P3_START, rng_seed=5, reproduction_patch_size_m=2.0)
        p3_step(s, self.env, 2)
        with self.assertRaises(IntereventStateError):
            self.reproduce(s)                                  # still December: nothing due
        p3_step(s, self.env, 1)
        self.assertEqual(s.annual_due_year, 2026)
        before = fp(s)
        with self.assertRaises(IntereventStateError):
            self.reproduce(s)                                  # annual mortality unresolved
        assert_same(self, before, fp(s))
        s.apply_annual_decision(2026)
        # a pending transport diagnostic blocks it; a later accepted interval clears it and the opportunity may then
        # be taken LATER than 1 January (newborns have age 0 at the cursor)
        s.advance_transport('wind', s.cursor, s.cursor + DAY)
        self.assertIsNotNone(s.transport_pending)
        with self.assertRaises(IntereventStateError):
            self.reproduce(s)
        p3_step(s, self.env, 1)
        self.assertTrue(s.has_pending_growth)
        with self.assertRaises(IntereventStateError):
            self.reproduce(s)                                  # pending growth must be completed first
        s.flush_growth()
        period0 = s.annual_period_start
        rep = self.reproduce(s)
        self.assertEqual(s.cursor, dt.datetime(2027, 1, 3, tzinfo=UTC))
        self.assertEqual(s.annual_period_start, period0)       # existing parents' real partial-year metrics preserved
        self.assertGreater(rep['reproduction']['n_births'], 0)

    def test_multi_year_gap_is_refused_and_must_be_declined_explicitly(self):
        s = self.year_end()
        s.advance_transport('wind', s.cursor, s.cursor + 400 * DAY)       # no mortality law: a multi-year event is allowed
        p3_step(s, self.env, 1)
        before = fp(s)
        with self.assertRaises(IntereventStateError) as cm:
            self.reproduce(s, year=2026)
        self.assertIn('decline', str(cm.exception))
        with self.assertRaises(IntereventStateError):
            self.reproduce(s, year=2027)                       # years are strictly sequential: 2026 first
        assert_same(self, before, fp(s))
        rec = s.decline_reproduction_opportunity(2026)
        self.assertFalse(rec['rng_consumed'])
        self.assertEqual((s.reproduction_last_year, s.n_reproduction_opportunities), (2026, 0))
        s.flush_growth()
        self.reproduce(s, year=2027)
        self.assertEqual(s.n_reproduction_opportunities, 1)

    def test_inactive_catalog_needs_no_seed_no_arrays_and_refuses_the_call(self):
        cat = load_catalog()
        s = make_session(self.env, cat)
        self.assertFalse(s.reproduction_active)
        self.assertEqual(s._repro.resident_bytes(), 0)
        self.assertIsNone(s._repro._f)
        self.assertEqual(s._persistent(s._b)['phase2_tables'], 0)
        with self.assertRaises(IntereventError):
            s.apply_reproduction(2026, P3_SURFACE, **P3_LIMS)
        with self.assertRaises(IntereventError):                 # an active law REQUIRES an explicit seed and a patch size
            make_session(self.env, self.cat, reproduction_patch_size_m=2.0)
        with self.assertRaises(IntereventError):
            make_session(self.env, self.cat, rng_seed=1)
        bad = load_catalog(lambda d: d['species']['grass'].update(reproduction={'law': 'mystery_v1'}))
        with self.assertRaises(IntereventError):
            make_session(self.env, bad, rng_seed=1)

    def test_zero_record_domain_gets_seed_recruits_and_requests_water_next_step(self):
        s = p3_direct(self.env, self.cat, p3_empty_plants(self.cat), rng_seed=3, reproduction_patch_size_m=2.0)
        self.assertEqual((s.n_live, s.n_total_records), (0, 0))
        p3_step(s, self.env, 3)
        rep = self.reproduce(s)
        self.assertEqual(rep['reproduction']['n_daughters_requested'], 0)
        self.assertGreaterEqual(s.n_live, 4)
        np.testing.assert_array_equal(s.snapshot().ids, np.arange(1, s.n_live + 1))
        self.assertEqual(s.root_map.n_plants, s.n_live)
        p3_step(s, self.env, 1)
        self.assertEqual(host(s._b.U).shape, (s.n_live,))

    def test_zero_live_with_dead_records_and_mortality_law_restarts_the_decision_period(self):
        cat = load_catalog(p3_modify(bg=0.01))
        rec = [dict(plant_id=9, species='grass', x_m=1.0, y_m=1.0, establishment_ground_m=100.0, agb_dry_kg=0.3,
                    seasonal_agb_dry_kg=0.0, age_days=500.0, status='dead_standing')]
        plants = state.PlantState.from_records(cat, state.Domain(NX, NY, 1.0), rec)
        s = p3_direct(self.env, cat, plants, rng_seed=3, reproduction_patch_size_m=2.0)
        p3_step(s, self.env, 3)
        self.assertIsNone(s.annual_due_year)                    # no live plant: no annual mortality decision exists
        self.reproduce(s)
        snap = s.snapshot()
        self.assertEqual(int(snap.ids[0]), 9)
        self.assertEqual(int(snap.status[0]), state.DEAD_STANDING)
        self.assertTrue(np.all(snap.ids[1:] >= 10))
        self.assertEqual(s.annual_period_start, s.cursor)       # the empty period is not inherited

    def test_signed_ids_and_int64_exhaustion_guard(self):
        def rec_(pid, sp, x, y, st_='live'):
            return dict(plant_id=pid, species=sp, x_m=x, y_m=y, establishment_ground_m=100.0, agb_dry_kg=0.3,
                        seasonal_agb_dry_kg=0.0, age_days=300.0, status=st_)
        ids = [-(2 ** 63), -4, 0, 2 ** 63 - 1]
        recs = [rec_(ids[0], 'grass', 0.8, 0.8), rec_(ids[1], 'grass', 3.0, 3.0), rec_(ids[2], 'grass', 2.0, 1.0),
                rec_(ids[3], 'shrub', 1.0, 3.0)]
        plants = state.PlantState.from_records(self.cat, state.Domain(NX, NY, 1.0), recs)
        s = self.year_end(plants=plants)
        before = fp(s)
        with self.assertRaises(IntereventError):                 # births need IDs above int64 max: refused, untouched
            self.reproduce(s)
        assert_same(self, before, fp(s))
        self.assertEqual(s.n_reproduction_opportunities, 0)
        # zero births: no failure merely because the population contains max-int64; the opportunity is consumed
        quiet = load_catalog(p3_modify(seed_p=0.0, clonal_p=0.0))
        s2 = self.year_end(cat=quiet, plants=state.PlantState.from_records(quiet, state.Domain(NX, NY, 1.0), recs))
        rep = self.reproduce(s2)
        self.assertEqual(rep['reproduction']['n_births'], 0)
        self.assertEqual(s2.n_reproduction_opportunities, 1)

    def test_surviving_root_overrides_preserved_and_daughters_use_species_roots(self):
        s = self.year_end()
        s2 = make_session(self.env, self.cat, start=P3_START, rng_seed=5, reproduction_patch_size_m=2.0,
                          overrides={1: [0.5, 0.5], 4: [0.1, 0.9]})
        p3_step(s2, self.env, 3)
        self.reproduce(s2)
        self.assertEqual(s2._overrides, {1: (0.5, 0.5), 4: (0.1, 0.9)})
        self.assertEqual(set(s2.live_ids.tolist()) & {1, 4}, {1, 4})
        self.assertGreater(s2.n_live, s.n_live - 1000)
        h = s2.root_map.host_arrays()
        np.testing.assert_array_equal(h['ids'], s2.live_ids)

    def test_row_permutation_gives_identical_by_id_births(self):
        recs = []
        for i, pid in enumerate((1, 2, 3, 4, 5)):
            recs.append(dict(plant_id=pid, species='grass' if pid <= 3 else 'shrub', x_m=0.8 + 0.5 * i, y_m=1.0 + 0.4 * i,
                             establishment_ground_m=100.0, agb_dry_kg=0.3 + 0.01 * i, seasonal_agb_dry_kg=0.0,
                             age_days=300.0 + i, status='live'))
        a = state.PlantState.from_records(self.cat, state.Domain(NX, NY, 1.0), recs)
        b = state.PlantState.from_records(self.cat, state.Domain(NX, NY, 1.0), recs[::-1])
        sa, sb = self.year_end(plants=a), self.year_end(plants=b)
        self.reproduce(sa)
        self.reproduce(sb)
        na, nb = sa.snapshot(), sb.snapshot()
        oa, ob = np.argsort(na.ids), np.argsort(nb.ids)
        for k in ('ids', 'species_index', 'x_m', 'y_m', 'establishment_ground_m', 'age_days', 'status'):
            np.testing.assert_array_equal(getattr(na, k)[oa], getattr(nb, k)[ob], err_msg=k)
        np.testing.assert_array_equal(na.agb_dry_kg[oa], nb.agb_dry_kg[ob])

    # ---- deferred prior-season finalisation before births (revision 2)
    def closed_year_end(self, plants, cat=None):
        '''Session that passes the year end inside a CLOSED-gate interval (season deferred, nothing else happens).'''
        cat = cat or self.cat
        s = p3_direct(self.env, cat, plants, rng_seed=3, reproduction_patch_size_m=2.0)
        end = dt.datetime(2027, 1, 1, tzinfo=UTC)
        s.step_interevent(forcing.TimestepRecord(s.cursor, end, 0.0, None, 'interevent', 'closed'))
        self.assertTrue(s.season_due)
        return s

    def test_reproduction_refuses_until_deferred_season_is_flushed_empty_population(self):
        s = self.closed_year_end(p3_empty_plants(self.cat))
        before = fp(s)
        with self.assertRaises(IntereventStateError) as cm:
            self.reproduce(s)
        self.assertIn('flush_deferred_season', str(cm.exception))
        assert_same(self, before, fp(s))
        rec = s.flush_deferred_season()
        self.assertTrue(rec['published'])
        self.assertFalse(s.season_due)
        self.assertEqual(s.flush_deferred_season()['published'], False)        # nothing deferred: no-op
        self.reproduce(s)
        born = s.snapshot()
        self.assertGreaterEqual(born.n, 4)
        p3_step(s, self.env, 1)
        after = s.snapshot()
        # newborn tissue never pays a prior-season loss
        np.testing.assert_array_equal(after.agb_dry_kg[:born.n] >= born.agb_dry_kg - 1e-15, True)
        self.assertNotIn('deferred_season_applied', s.last_receipt)
        self.assertEqual(s.last_receipt.get('deferred_season_applied', False), False)

    def test_flush_deferred_season_applies_old_season_to_live_plants_only_and_is_atomic(self):
        plants = make_plants(self.cat, shrub_e=0.5)               # shrub id 4 carries seasonal biomass
        s = self.closed_year_end(plants)
        s0, w0, t0, cur0, nb0 = s.snapshot(), host(s.native_state.water_m).copy(), s.native_state.t_s, s.cursor, s.boundaries()['next_growth_boundary']
        before = fp(s)
        real = roots.build_root_map
        wrapper, box = fail_once(real, RuntimeError('injected'))
        with mock.patch.object(roots, 'build_root_map', wrapper):
            with self.assertRaises(RuntimeError):
                s.flush_deferred_season()
        assert_same(self, before, fp(s))
        self.assertTrue(s.season_due)                              # failure: nothing changed, flag still set
        s.flush_deferred_season()
        s1 = s.snapshot()
        self.assertFalse(s.season_due)
        self.assertLess(float(s1.seasonal_agb_dry_kg[3]), float(s0.seasonal_agb_dry_kg[3]))     # true turnover of old tissue
        self.assertLessEqual(float(s1.agb_dry_kg[3]), float(s0.agb_dry_kg[3]))
        np.testing.assert_array_equal(host(s.native_state.water_m), w0)                         # no water / time / ET
        self.assertEqual((s.native_state.t_s, s.cursor, s.boundaries()['next_growth_boundary']), (t0, cur0, nb0))
        self.reproduce(s)
        s2 = s.snapshot()
        self.assertGreater(s2.n, s1.n)
        p3_step(s, self.env, 1)
        self.assertNotIn('deferred_season', s.last_receipt)           # nothing left to apply at the next open interval

    def test_flush_deferred_season_refuses_with_pending_growth(self):
        s = self.year_end()
        p3_step(s, self.env, 1)
        self.assertTrue(s.has_pending_growth)
        real = s._b
        s._b = real.clone()
        s._b.season_due = True                                    # simulate a deferred season beside pending growth
        try:
            with self.assertRaises(IntereventStateError):
                s.flush_deferred_season()
        finally:
            s._b = real

    # ---- strict restart metadata (revision 2)
    def test_inconsistent_but_hash_consistent_phase3_metadata_is_refused(self):
        def rehash(doc):
            body = {k: v for k, v in doc.items() if k != 'content_hash'}
            doc['content_hash'] = checkpoint._sha(checkpoint._canonical(body))
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(self.year_end().save(os.path.join(td, 'cp')))      # start 2026, nothing handled yet
            orig = json.loads((path / 'restart.json').read_text())
            rec_ok = {'year': 2026, 'counter': 0, 'interval_id': 'x', 'cursor': '2027-01-01T00:00:00+00:00',
                      'sequence': 1, 'n_seed_recruits': 1, 'n_daughters_placed': 0, 'n_daughters_rejected': 0, 'n_births': 1}
            muts = (lambda d: d['phase3'].update(n_reproduction=2 ** 64),
                    lambda d: d['phase3'].update(n_reproduction=1),
                    lambda d: d['phase3'].update(last_opportunity_year=2022),
                    lambda d: d['phase3'].update(start_year=2024),
                    lambda d: d['phase3'].update(last_reproduction={'year': 2023}),
                    lambda d: d['phase3'].update(last_opportunity_year=2026, last_reproduction=None),
                    lambda d: d['phase3'].update(last_opportunity_year=2026, n_reproduction=1,
                                                 last_reproduction=dict(rec_ok, counter=5)),
                    lambda d: d['phase3'].update(last_opportunity_year=2026, n_reproduction=1,
                                                 last_reproduction=dict(rec_ok, year=2025)),
                    lambda d: d['phase3'].update(last_opportunity_year=2026, n_reproduction=1,
                                                 last_reproduction=dict(rec_ok, n_births=3)),
                    lambda d: d['phase3'].update(last_opportunity_year=2026, n_reproduction=1,
                                                 last_reproduction=dict(rec_ok, cursor='2026-06-01T00:00:00+00:00')),
                    lambda d: d['phase3'].update(last_opportunity_year=2027))
            for m in muts:
                doc = json.loads(json.dumps(orig))
                m(doc)
                rehash(doc)
                (path / 'restart.json').write_text(json.dumps(doc))
                with self.assertRaises(checkpoint.CheckpointError):
                    self.load(path)
            # a coherent handled state loads
            doc = json.loads(json.dumps(orig))
            doc['phase3'].update(last_opportunity_year=2026, n_reproduction=1, last_reproduction=rec_ok)
            rehash(doc)
            (path / 'restart.json').write_text(json.dumps(doc))
            self.assertEqual(self.load(path).n_reproduction_opportunities, 1)

    def test_declined_year_roundtrips_with_fewer_accepted_than_handled(self):
        with tempfile.TemporaryDirectory() as td:
            s = self.year_end()
            s.decline_reproduction_opportunity(2026)
            loaded = self.load(s.save(os.path.join(td, 'cp')))
            self.assertEqual((loaded.reproduction_last_year, loaded.n_reproduction_opportunities), (2026, 0))
            self.assertTrue(loaded.phase3_state()['last_reproduction']['declined'])

    # ---- memory record of the full evaluate stage (revision 3)
    def test_all_rejected_requests_still_record_the_full_evaluate_stage_in_the_peak(self):
        def reject_all(doc):
            p3_modify(seed_p=0.0, clonal_p=1.0)(doc)
            c = doc['species']['grass']['reproduction']['traits']['clonal']
            c['dispersal_radius_m'] = 1.0e6              # a single attempt lands outside the 4 m domain (p ~ 1e-11)
            c['max_attempts_per_daughter'] = 1
        cat = load_catalog(reject_all)
        s = self.year_end(cat=cat)
        s0, w0 = s.snapshot(), host(s.native_state.water_m).copy()
        rep = self.reproduce(s)
        d = rep['reproduction']
        self.assertGreater(d['n_daughters_requested'], 0)
        self.assertEqual((d['n_daughters_placed'], d['n_births']), (0, 0))
        mem = rep['memory']
        self.assertIn('reproduction_preflight', mem['stage_totals'])
        self.assertIn('reproduction_evaluate', mem['stage_totals'])
        self.assertGreaterEqual(mem['stage_totals']['reproduction_evaluate'], d['memory_estimate']['total_bytes'])
        self.assertGreater(mem['stage_totals']['reproduction_evaluate'], mem['stage_totals']['reproduction_preflight'])
        self.assertEqual(mem['peak_estimate_bytes'], max(mem['stage_totals'].values()))
        self.assertGreaterEqual(mem['peak_estimate_bytes'], mem['stage_totals']['reproduction_evaluate'])
        s1 = s.snapshot()                                 # nothing donated, nothing born; only the opportunity is consumed
        for k in ('ids', 'x_m', 'y_m', 'agb_dry_kg', 'seasonal_agb_dry_kg', 'age_days', 'status'):
            np.testing.assert_array_equal(getattr(s1, k), getattr(s0, k), err_msg=k)
        np.testing.assert_array_equal(host(s.native_state.water_m), w0)
        self.assertEqual((s.n_reproduction_opportunities, s.reproduction_last_year), (1, 2026))

    # ---- failure atomicity / retry
    def test_fault_injection_leaves_state_unchanged_and_valid_retry_counts_once(self):
        s = self.year_end()
        before = fp(s)
        real = roots.build_root_map
        wrapper, box = fail_once(real, RuntimeError('injected root failure'))
        with mock.patch.object(roots, 'build_root_map', wrapper):
            with self.assertRaises(RuntimeError):
                self.reproduce(s)
        self.assertEqual(box['n'], 1)
        assert_same(self, before, fp(s))
        self.assertEqual((s.n_reproduction_opportunities, s.reproduction_last_year, s.n_total_records), (0, 2025, 7))
        s.max_bytes = 1000                                       # an admission refusal also leaves everything intact
        with self.assertRaises(IntereventAdmissionError):
            self.reproduce(s)
        s.max_bytes = BIG
        assert_same(self, before, fp(s))
        self.reproduce(s)                                        # the valid retry accounts exactly once
        self.assertEqual(s.n_reproduction_opportunities, 1)
        ref = self.year_end()
        self.reproduce(ref)
        np.testing.assert_array_equal(s.snapshot().x_m, ref.snapshot().x_m)     # same (seed, counter): same draws

    # ---- restart
    def test_checkpoint_roundtrip_state_and_identical_continuation(self):
        with tempfile.TemporaryDirectory() as td:
            c = self.year_end()
            path = c.save(os.path.join(td, 'cp'))               # opportunity outstanding at the save
            loaded = self.load(path)
            self.assertEqual((loaded.reproduction_last_year, loaded.n_reproduction_opportunities), (2025, 0))
            self.assertEqual(loaded.reproduction_patch_size_m, 2.0)
            self.reproduce(c)
            self.reproduce(loaded)
            a, b = c.snapshot(), loaded.snapshot()
            for k in ('ids', 'x_m', 'y_m', 'agb_dry_kg', 'seasonal_agb_dry_kg', 'age_days', 'status'):
                np.testing.assert_array_equal(getattr(a, k), getattr(b, k), err_msg=k)
            # a checkpoint AFTER the opportunity restores the counter, the guard and the patch; the replay is refused
            path2 = c.save(os.path.join(td, 'cp2'))
            loaded2 = self.load(path2)
            self.assertEqual((loaded2.reproduction_last_year, loaded2.n_reproduction_opportunities), (2026, 1))
            self.assertEqual(loaded2.last_receipt['kind'], 'reproduction')
            with self.assertRaises(IntereventStateError):
                self.reproduce(loaded2)
            np.testing.assert_array_equal(loaded2.live_ids, c.live_ids)
            doc = json.loads((pathlib.Path(path2) / 'restart.json').read_text())
            self.assertEqual(doc['phase3']['n_reproduction'], 1)
            self.assertEqual(doc['schema_version'], 3)

    def test_old_checkpoint_versions_are_refused_not_migrated(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(self.year_end().save(os.path.join(td, 'cp')))
            for old in (1, 2):
                doc = json.loads((path / 'restart.json').read_text())
                doc['schema_version'] = old
                body = {k: v for k, v in doc.items() if k != 'content_hash'}
                doc['content_hash'] = checkpoint._sha(checkpoint._canonical(body))
                (path / 'restart.json').write_text(json.dumps(doc))
                for allow in (False, True):                      # allow_source_change never authorises a migration
                    with self.assertRaises(checkpoint.CheckpointError):
                        self.load(path, allow_source_change=allow)

    def test_phase3_block_corruption_and_flag_mismatch_are_refused(self):
        def rehash(doc):
            body = {k: v for k, v in doc.items() if k != 'content_hash'}
            doc['content_hash'] = checkpoint._sha(checkpoint._canonical(body))
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(self.year_end().save(os.path.join(td, 'cp')))
            orig = json.loads((path / 'restart.json').read_text())
            muts = (lambda d: d['phase3'].update(n_reproduction='1'), lambda d: d['phase3'].update(n_reproduction=True),
                    lambda d: d['phase3'].update(patch_size_m=-1.0), lambda d: d['phase3'].update(patch_size_m=True),
                    lambda d: d['phase3'].update(last_opportunity_year=2.5),
                    lambda d: d['phase3'].update(reproduction_active=False), lambda d: d['phase3'].update(extra=1),
                    lambda d: d.pop('phase3'))
            for m in muts:
                doc = json.loads(json.dumps(orig))
                m(doc)
                rehash(doc)
                (path / 'restart.json').write_text(json.dumps(doc))
                with self.assertRaises(checkpoint.CheckpointError):
                    self.load(path)
            (path / 'restart.json').write_text(json.dumps(orig))
            self.load(path)
            # a different catalog reproduction law (hash differs) is refused: no silent catalog migration
            with self.assertRaises(checkpoint.CheckpointError):
                self.load(path, cat=load_catalog(p3_modify(seed_p=0.5)))


@unittest.skipUnless(NATIVE and CUDA, 'CUDA backend unavailable')
class Phase3ReproductionCuda(Phase3Reproduction):
    '''The same transactions on the actual CUDA backend (explicit host placement; B / E transfers are counted).'''
    backend = 'cuda'


if __name__ == '__main__':
    unittest.main()
