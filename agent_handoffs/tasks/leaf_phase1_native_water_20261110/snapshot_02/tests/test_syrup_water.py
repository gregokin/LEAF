'''LEAF Phase 1 native-water bridge tests (unittest) against the REAL maple_syrup native API.

Run from the candidate root with the pinned SYRUP package importable:  python -m unittest tests.test_syrup_water -v

* The native water authority is never faked: every case builds an actual `maple_syrup.subsurface` column, a
  prepared `prepare_transpiration` operator and a `SoilColumnState`, and the bridge calls the ACTUAL
  `withdraw_transpiration`. Spies only WRAP the real function (to count calls, record the request, or to hand
  the real operator a different real state / request / to tamper with the returned result for rejection tests).
* Native-dependent classes SKIP only when `leaf.syrup_water.native_status()` reports the package unavailable;
  numba / CUDA classes SKIP only when the backend reports unavailable. Everything else is a failure.
* CUDA tests never allocate on another device (no device-mismatch fabrication); the device-affinity guards are
  exercised only through host/device namespace mismatches that need no second device.
* Written without the ability to execute them (file tools only): Codex records the real runs.
'''
from __future__ import annotations

import contextlib
import dataclasses
import gc
import math
import os
import pathlib
import subprocess
import sys
import tracemalloc
import unittest
from unittest import mock

import numpy as np

from leaf import syrup_water as bridge
from leaf import uptake

ROOT = pathlib.Path(__file__).resolve().parents[1]
STATUS = bridge.native_status()
NATIVE = bool(STATUS['available'])
if NATIVE:
    from maple_syrup import subsurface as ss
    from maple_syrup import subsurface_transpiration as st

BIG = 1 << 28
THETA = dict(theta_residual=0.125, theta_field_capacity=0.5, theta_saturation=0.875)   # binary-exact depths
E_UPTAKE = uptake.UptakeError
needs_native = unittest.skipUnless(NATIVE, f'native SYRUP package unavailable: {STATUS["reason"]}')


# ------------------------------------------------------------------ helpers
def host(a):
    return np.asarray(a.get() if hasattr(a, 'get') else a)


def csr(rows, ids=None):
    '''rows: per plant a list of (resource, weight) -> (ids, offsets, resources, weights) typed arrays.'''
    ids = np.arange(1, len(rows) + 1, dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    res, w = [], []
    for i, row in enumerate(rows):
        offsets[i + 1] = offsets[i] + len(row)
        for r, ww in row:
            res.append(r)
            w.append(ww)
    return ids, offsets, np.array(res, dtype=np.int64), np.array(w, dtype=np.float64)


def make_map(rows, L, N, backend='numpy', ids=None, max_bytes=BIG, max_edges=10_000_000):
    i, o, r, w = csr(rows, ids)
    return uptake.RootMap.from_arrays(i, o, r, w, n_layers=L, n_cells=N, backend=backend, max_edges=max_edges,
                                      max_bytes=max_bytes)


def random_rows(rng, P, R, max_edges=5):
    rows = []
    for _ in range(P):
        k = int(rng.integers(1, max_edges + 1))
        res = rng.choice(R, size=k, replace=False)
        w = rng.uniform(0.05, 1.0, size=k)
        w /= w.sum()
        rows.append([(int(r), float(x)) for r, x in zip(res, w)])
    return rows


def oracle_proposal(rows, D, A):
    '''Scalar oracle of the user water rules (explicit loops; independent of the allocator).'''
    R = A.size
    Q = np.zeros(R)
    for p, row in enumerate(rows):
        for r, w in row:
            Q[r] += D[p] * w
    prop = np.array([min(A[r], Q[r]) if Q[r] > 0 else 0.0 for r in range(R)])
    return Q, prop


def oracle_delivered(rows, D, Q, accepted):
    out = np.zeros(len(rows))
    for p, row in enumerate(rows):
        for r, w in row:
            if Q[r] > 0:
                out[p] += D[p] * w * accepted[r] / Q[r]
    return out


def host_combos():
    '''(native backend, map backend) pairs on the CPU; numba pairs only when numba is genuinely available.'''
    combos = [('numpy', 'numpy')]
    if uptake.backend_status('numba')['available']:
        combos += [('numba', 'numba'), ('numba', 'numpy'), ('numpy', 'numba')]
    return combos


class Env:
    '''A REAL native column + prepared operator (+ state factory). depth edges in m, area in m2.'''

    def __init__(self, edges, area, active, backend='numpy', *, reserve_bytes=0, available_bytes_fn=None):
        self.layers = ss.soil_layers(edges, diffusivity_m2_s=0.0, theta_evaporation_stress=0.25, **THETA)
        self.L, self.N, self.area, self.backend = self.layers.n_layers, len(active), float(area), backend
        mask = np.asarray(active, dtype=bool)
        if backend == 'cuda':
            import cupy as cp
            self.xp, mask_xp = cp, cp.asarray(mask)
        else:
            self.xp, mask_xp = np, mask
        self.column = ss.prepare_column(self.layers, mask_xp, self.area, backend=backend, reserve_bytes=reserve_bytes,
                                        available_bytes_fn=available_bytes_fn)
        self.operator = st.prepare_transpiration(self.column)
        self.res = np.array(self.layers.residual_m)
        self.sat = np.array(self.layers.saturation_m)

    def water(self, avail_depth):
        return np.ascontiguousarray(self.res[:, None] + np.asarray(avail_depth, dtype=np.float64))

    def state(self, avail_depth, t_s=1234.5):
        return self.state_from_water(self.water(avail_depth), t_s)

    def state_from_water(self, water, t_s=1234.5):
        w = np.ascontiguousarray(water, dtype=np.float64)
        return ss.column_state(self.column, self.xp.asarray(w) if self.backend == 'cuda' else w, t_s)


def snap_inputs(state, m, D):
    return {'water': host(state.water_m).copy(), 't_s': state.t_s, 'map': m.host_arrays(), 'D': host(D).copy()}


def assert_unchanged(tc, snap, state, m, D):
    np.testing.assert_array_equal(host(state.water_m), snap['water'])
    tc.assertEqual(state.t_s, snap['t_s'])
    for k, v in m.host_arrays().items():
        np.testing.assert_array_equal(v, snap['map'][k])
    np.testing.assert_array_equal(host(D), snap['D'])


@contextlib.contextmanager
def native_spy(wrapper=None):
    '''Wrap the REAL withdraw_transpiration. `wrapper(real, operator, state, requested, **kw)` may call the real
    function with other (real) arguments or tamper with its result; the default just forwards.'''
    real = st.withdraw_transpiration
    calls = []

    def spy(operator, state, requested, **kw):
        calls.append({'requested': np.array(requested, copy=True) if isinstance(requested, np.ndarray) else None,
                      'state': state, 'kw': kw})
        if wrapper is None:
            return real(operator, state, requested, **kw)
        return wrapper(real, operator, state, requested, **kw)

    with mock.patch.object(st, 'withdraw_transpiration', spy):
        yield calls


def ledger_for(env, m, D, extra=0):
    return bridge.estimate_bridge_bytes(m.n_plants, m.n_edges, env.L, env.N, map_backend=m.backend,
                                        native_backend=env.backend, demand_bytes=int(D.nbytes),
                                        map_resident_bytes=m.resident_bytes,
                                        column_reserve_bytes=int(env.column.reserve_bytes), extra_reserve_bytes=extra)


# ------------------------------------------------------------------ pure / import tests (no native needed)
class TestImportsAndPureKernels(unittest.TestCase):
    def test_core_import_isolation(self):
        '''Importing leaf.uptake and leaf.syrup_water imports neither native SYRUP, numba nor cupy.'''
        code = ("import sys\n"
                "import leaf.uptake, leaf.syrup_water as b\n"
                "bad = [m for m in ('maple_syrup', 'numba', 'cupy') if m in sys.modules]\n"
                "assert not bad, bad\n"
                "assert callable(b.withdraw_plants) and callable(b.estimate_bridge_bytes)\n"
                "print('isolated')\n")
        proc = subprocess.run([sys.executable, '-c', code], cwd=str(ROOT), capture_output=True, text=True,
                              env=dict(os.environ))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn('isolated', proc.stdout)

    def test_plain_python_projection_kernel_matches_the_law(self):
        w = np.array([[37.5, 20.0, 900.0], [300.0, 5.0, 4.0]])
        res = np.array([12.5, 100.0])
        satb = np.array([1000.0, 1000.0])
        active = np.array([1, 0, 1], dtype=np.uint8)
        out = np.empty((2, 3))
        rows = np.empty((2, 3))
        bridge._k_project(w, res, satb, active, 2.0, out, rows)
        expected = 2.0 * np.maximum(w - res[:, None], 0.0) * active[None, :]
        np.testing.assert_array_equal(out, expected)
        np.testing.assert_array_equal(rows[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(rows[1], expected.sum(axis=0))
        w[1, 2] = np.nan
        bridge._k_project(w, res, satb, active, 2.0, out, rows)
        self.assertEqual(rows[0, 2], 1.0)
        w[1, 2] = 4.0
        w[0, 0] = 2000.0                                   # above the saturation bound
        bridge._k_project(w, res, satb, active, 2.0, out, rows)
        self.assertEqual(rows[0, 0], 1.0)

    def test_pin_and_documented_projection(self):
        self.assertEqual(bridge.NATIVE_PIN['repository_head'], 'c76c603dde4e8f79ec48b18d954a386580d8bc89')
        self.assertEqual(bridge.NATIVE_PIN['sha256']['subsurface.py'], '5e77f14df8af4e473e8a22278700b519ccb2888c722883a9f7da508abdf14ae9')
        self.assertEqual(bridge.NATIVE_PIN['sha256']['subsurface_transpiration.py'],
                         '3b9191872522755924271bbaba4ae2bac290e1cad4ecffebc2d2735c4a0a9fb9')
        self.assertEqual(bridge.SUPPORTED_NATIVE_VERSIONS, ('0.2.0',))
        self.assertIn('max(water_m[l, c] - residual_m[l], 0)', bridge.AVAILABILITY_PROJECTION)
        self.assertIn('NOT a separate water store', bridge.AVAILABILITY_PROJECTION)
        self.assertIn('exactly-once interval receipt', bridge.PUBLICATION_NOTE)

    def test_missing_native_is_a_clear_error_not_a_fallback(self):
        with mock.patch.dict(bridge._NATIVE, {}, clear=True):
            with mock.patch.dict(sys.modules, {'maple_syrup': None, 'maple_syrup.subsurface': None,
                                               'maple_syrup.subsurface_transpiration': None}):
                st_ = bridge.native_status()
                self.assertFalse(st_['available'])
                with self.assertRaises(bridge.NativeUnavailableError):
                    bridge.check_operator(object())


# ------------------------------------------------------------------ real native: water cases
@needs_native
class TestNativeContractAndCases(unittest.TestCase):
    def test_native_pin_and_provenance(self):
        self.assertTrue(STATUS['supported'], f'imported native VERSION {STATUS["version"]!r} is not the pinned one')
        prov = bridge.native_provenance()
        self.assertTrue(prov['version_supported'])
        self.assertFalse(prov['verified'])
        ver = bridge.native_provenance(verify=True)             # initialisation / run-evidence operation, not per call
        self.assertTrue(ver['verified'])
        for key in ('subsurface.py', 'subsurface_transpiration.py'):
            self.assertEqual(len(ver['sha256'][key]), 64)
            self.assertIn(key, ver['hash_matches'])

    def test_user_75_percent_and_deep_water_left(self):
        for nb, mb in host_combos():
            with self.subTest(native=nb, map=mb):
                env = Env([0.0, 100.0, 1000.0], 1.0, [True], nb)
                state = env.state([[25.0], [200.0]])                     # A = 25 / 200 m3 above residual
                m = make_map([[(0, 0.5), (1, 0.5)]], 2, 1, mb)
                D = np.array([100.0])
                snap = snap_inputs(state, m, D)
                with native_spy() as calls:
                    r = bridge.withdraw_plants(m, D, env.operator, state, max_bytes=BIG)
                self.assertEqual(len(calls), 1)                          # ONE native withdrawal
                np.testing.assert_array_equal(r.availability.volume_m3, [[25.0], [200.0]])
                np.testing.assert_array_equal(r.plan.request, [50.0, 50.0])
                np.testing.assert_array_equal(r.plan.proposed, [25.0, 50.0])
                np.testing.assert_array_equal(calls[0]['requested'], [[25.0], [50.0]])   # depth = volume / 1 m2
                np.testing.assert_array_equal(r.native.actual_m, [[25.0], [50.0]])
                np.testing.assert_array_equal(r.feedback.accepted, [25.0, 50.0])
                self.assertEqual(float(r.delivered_m3[0]), 75.0)
                self.assertEqual(float(r.unmet_m3[0]), 25.0)
                self.assertEqual(float(r.satisfaction[0]), 0.75)
                # leftover deep water stays with SYRUP (no compensation): shallow back to residual, deep 150 m above it
                np.testing.assert_array_equal(r.native_state.water_m, [[12.5], [262.5]])
                self.assertEqual(float(r.native.budget.actual_m3), 75.0)
                self.assertEqual(r.native_state.t_s, 1234.5)
                self.assertEqual(r.t_s, 1234.5)
                self.assertTrue(r.conservation['closed'])
                self.assertTrue(r.conservation['requested_matches_proposal'])
                self.assertTrue(all(r.conservation['checks'].values()))
                self.assertIsNot(r.native_state.water_m, state.water_m)
                assert_unchanged(self, snap, state, m, D)                # nothing published, inputs untouched

    def test_proportional_competition_unequal_roots(self):
        env = Env([0.0, 200.0], 1.0, [True, True], 'numpy')
        state = env.state([[62.5, 40.0]])
        rows = [[(0, 1.0)], [(0, 0.5), (1, 0.5)]]
        D = np.array([100.0, 50.0])
        m = make_map(rows, 1, 2)
        r = bridge.withdraw_plants(m, D, env.operator, state, max_bytes=BIG)
        np.testing.assert_array_equal(r.plan.request, [125.0, 25.0])
        np.testing.assert_array_equal(r.plan.proposed, [62.5, 25.0])
        np.testing.assert_array_equal(r.plan.fraction, [0.5, 1.0])
        np.testing.assert_array_equal(r.delivered_m3, [50.0, 37.5])      # same fraction 0.5 on the shared cell
        np.testing.assert_array_equal(r.unmet_m3, [50.0, 12.5])
        np.testing.assert_array_equal(r.satisfaction, [0.5, 0.75])
        np.testing.assert_array_equal(r.native_state.water_m, [[25.0, 40.0]])   # cell 1 keeps 15 m above residual
        # reordering individuals changes nothing per ID (no species / size / order priority)
        m2 = make_map(rows[::-1], 1, 2, ids=[2, 1])
        r2 = bridge.withdraw_plants(m2, D[::-1].copy(), env.operator, state, max_bytes=BIG)
        np.testing.assert_array_equal(r2.delivered_m3, [37.5, 50.0])
        np.testing.assert_array_equal(r2.native_state.water_m, r.native_state.water_m)

    def test_shallow_and_deep_profiles_and_no_compensation(self):
        env = Env([0.0, 100.0, 200.0], 1.0, [True], 'numpy')
        # shallow-rooted and deep-rooted plants with ample water in their own layers
        state = env.state([[30.0], [70.0]])
        m = make_map([[(0, 1.0)], [(1, 1.0)]], 2, 1)
        r = bridge.withdraw_plants(m, np.array([20.0, 60.0]), env.operator, state, max_bytes=BIG)
        np.testing.assert_array_equal(r.delivered_m3, [20.0, 60.0])
        np.testing.assert_array_equal(r.native_state.water_m, [[22.5], [22.5]])
        # dry shallow layer: the shallow half of the request stays unmet although the deep layer is wet
        dry = env.state([[0.0], [75.0]])
        m2 = make_map([[(0, 0.5), (1, 0.5)]], 2, 1)
        D = np.array([100.0])
        r2 = bridge.withdraw_plants(m2, D, env.operator, dry, max_bytes=BIG)
        np.testing.assert_array_equal(r2.plan.proposed, [0.0, 50.0])
        self.assertEqual(float(r2.delivered_m3[0]), 50.0)
        self.assertEqual(float(r2.unmet_m3[0]), 50.0)
        np.testing.assert_array_equal(r2.native_state.water_m, [[12.5], [37.5]])   # shallow stays at residual
        # fully dry profile: nothing is available, nothing is withdrawn, nothing negative appears
        none = env.state([[0.0], [0.0]])
        r3 = bridge.withdraw_plants(m2, D, env.operator, none, max_bytes=BIG)
        np.testing.assert_array_equal(r3.delivered_m3, [0.0])
        np.testing.assert_array_equal(r3.unmet_m3, [100.0])
        np.testing.assert_array_equal(r3.native_state.water_m, [[12.5], [12.5]])

    def test_inactive_cells_have_no_availability_and_no_renormalisation(self):
        for nb, mb in host_combos():
            with self.subTest(native=nb, map=mb):
                env = Env([0.0, 200.0], 1.0, [True, False, True], nb)
                state = env.state([[50.0, 80.0, 50.0]])                   # the inactive cell holds water above residual
                rows = [[(0, 0.5), (1, 0.5)], [(2, 0.5), (1, 0.5)]]
                D = np.array([100.0, 40.0])
                m = make_map(rows, 1, 3, mb)
                snap = snap_inputs(state, m, D)
                r = bridge.withdraw_plants(m, D, env.operator, state, max_bytes=BIG)
                np.testing.assert_array_equal(r.availability.volume_m3, [[50.0, 0.0, 50.0]])
                np.testing.assert_array_equal(r.plan.proposed, [50.0, 0.0, 20.0])
                np.testing.assert_array_equal(r.delivered_m3, [50.0, 20.0])  # inactive part NOT moved to active cells
                np.testing.assert_array_equal(r.unmet_m3, [50.0, 20.0])
                self.assertEqual(float(r.native_state.water_m[0, 1]), 105.0)  # inactive water untouched, bit exact
                self.assertEqual(r.availability.n_active, 2)
                assert_unchanged(self, snap, state, m, D)

    def test_nonunit_areas_preserve_water_units_and_exact_accepted_feedback(self):
        rows = [[(0, .4), (1, .4), (3, .2)], [(4, 1.0)], [(1, .5), (4, .5)], [(2, 1.0)]]
        depth = np.array([[30.0, 10.0, 0.0], [60.0, 5.0, 40.0]])
        for area in (0.0625, 1.0, 2.3, 400.0):
            for nb, mb in host_combos():
                with self.subTest(area=area, native=nb, map=mb):
                    env = Env([0.0, 100.0, 300.0], area, [True] * 3, nb)
                    state = env.state(depth)
                    D = np.array([10.0, 20.0, 40.0, 5.0]) * area
                    m = make_map(rows, 2, 3, mb)
                    with native_spy() as calls:
                        r = bridge.withdraw_plants(m, D, env.operator, state, max_bytes=BIG)
                    A = (area * np.maximum(host(state.water_m) - env.res[:, None], 0.0)).reshape(-1)
                    np.testing.assert_allclose(r.availability.volume_m3.reshape(-1), A, rtol=1e-15, atol=0)
                    Q, prop = oracle_proposal(rows, D, A)
                    np.testing.assert_allclose(r.plan.proposed, prop, rtol=1e-13, atol=0)
                    # division by the area exactly once, multiplication by the area exactly once
                    np.testing.assert_array_equal(calls[0]['requested'], r.plan.proposed_grid() / area)
                    np.testing.assert_array_equal(r.accepted_volume_m3, r.native.actual_m * area)
                    np.testing.assert_array_equal(r.feedback.accepted, (r.native.actual_m * area).reshape(-1))
                    np.testing.assert_allclose(r.delivered_m3, oracle_delivered(rows, D, Q, prop), rtol=1e-12, atol=0)
                    np.testing.assert_allclose(r.native_state.water_m.reshape(-1),
                                               host(state.water_m).reshape(-1) - prop / area, rtol=1e-12)
                    self.assertLessEqual(abs(float(np.sum(r.delivered_m3)) - r.native.budget.actual_m3),
                                         r.conservation['tolerance_m3'])
                    self.assertLessEqual(abs(r.conservation['area_conversion_roundoff_m3']), r.conservation['tolerance_m3'])
                    self.assertTrue(r.conservation['closed'])
                    self.assertEqual(float(r.delivered_m3[3]), 0.0)                      # dry resource: unmet stays unmet

    def test_random_overlaps_finite_nonnegative_unit_closure_and_purity(self):
        for area in (0.3, 2.3, 7.7):
            with self.subTest(area=area):
                rng = np.random.default_rng(20261111)
                L, N, P = 2, 9, 60
                rows = random_rows(rng, P, L * N)
                D = rng.uniform(0.0, 40.0, P) * area
                D[::7] = 0.0
                depth = rng.uniform(0.0, 60.0, (L, N))
                active = rng.random(N) > 0.25
                active[0] = True
                env = Env([0.0, 100.0, 300.0], area, active, 'numpy')
                state = env.state(depth)
                m = make_map(rows, L, N)
                snap = snap_inputs(state, m, D)
                r = bridge.withdraw_plants(m, D, env.operator, state, max_bytes=BIG)
                A = (area * np.maximum(host(state.water_m) - env.res[:, None], 0.0) * active[None, :]).reshape(-1)
                Q, prop = oracle_proposal(rows, D, A)
                np.testing.assert_allclose(r.delivered_m3, oracle_delivered(rows, D, Q, prop), rtol=1e-11, atol=0)
                for name in ('delivered', 'unmet', 'satisfaction', 'accepted'):
                    arr = getattr(r.feedback, name)
                    self.assertTrue(np.all(np.isfinite(arr)) and np.all(arr >= 0.0), name)
                self.assertTrue(np.all(r.satisfaction <= 1.0))
                np.testing.assert_array_equal(r.satisfaction[D == 0.0], 0.0)
                np.testing.assert_allclose(r.delivered_m3 + r.unmet_m3, D, rtol=1e-9, atol=1e-12)    # unit closure per plant
                self.assertTrue(np.all(np.isfinite(r.native_state.water_m)) and np.all(r.native_state.water_m >= 0.0))
                self.assertTrue(all(r.conservation['checks'].values()))
                np.testing.assert_array_equal(r.feedback.accepted, (r.native.actual_m * area).reshape(-1))  # no clipping
                # the pure function is repeatable: the same inputs give the same fresh state (nothing was published)
                r2 = bridge.withdraw_plants(m, D, env.operator, state, max_bytes=BIG)
                np.testing.assert_array_equal(r2.native_state.water_m, r.native_state.water_m)
                np.testing.assert_array_equal(r2.delivered_m3, r.delivered_m3)
                assert_unchanged(self, snap, state, m, D)

    def test_empty_population_is_a_zero_withdrawal(self):
        env = Env([0.0, 200.0], 1.0, [True, True], 'numpy')
        state = env.state([[10.0, 20.0]])
        m = make_map([], 1, 2)
        r = bridge.withdraw_plants(m, np.zeros(0), env.operator, state, max_bytes=BIG)
        np.testing.assert_array_equal(r.native_state.water_m, host(state.water_m))
        self.assertEqual(r.delivered_m3.shape, (0,))
        self.assertEqual(r.conservation['native_actual_m3'], 0.0)

    def test_accepted_shortfall_feedback_with_real_operator_and_drier_state(self):
        '''A narrow spy hands the REAL operator a drier REAL state (authority holds less than the snapshot said):
        native accepts only part of the proposal; plants gather from the ACTUAL accepted water.'''
        env = Env([0.0, 200.0], 1.0, [True], 'numpy')
        state = env.state([[100.0]])
        m = make_map([[(0, 1.0)]], 1, 1)
        D = np.array([80.0])

        def drier(real, operator, st_, requested, **kw):
            half = env.res[:, None] + 0.5 * (np.asarray(st_.water_m) - env.res[:, None])
            return real(operator, ss.SoilColumnState(np.ascontiguousarray(half), st_.t_s), requested, **kw)

        snap = snap_inputs(state, m, D)
        with native_spy(drier) as calls:
            r = bridge.withdraw_plants(m, D, env.operator, state, max_bytes=BIG)
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(r.plan.proposed, [80.0])
        np.testing.assert_array_equal(r.native.actual_m, [[50.0]])
        np.testing.assert_array_equal(r.feedback.accepted, [50.0])
        self.assertEqual(float(r.delivered_m3[0]), 50.0)
        self.assertEqual(float(r.unmet_m3[0]), 30.0)
        self.assertEqual(float(r.satisfaction[0]), 0.625)
        self.assertAlmostEqual(r.feedback.summary['shortfall_total_m3'], 30.0, places=12)
        np.testing.assert_array_equal(r.native_state.water_m, [[25.0]])
        self.assertTrue(r.conservation['closed'])
        self.assertEqual(r.native_state.t_s, state.t_s)
        assert_unchanged(self, snap, state, m, D)

    def test_native_requested_must_match_the_proposal(self):
        env = Env([0.0, 200.0], 1.0, [True], 'numpy')
        state = env.state([[100.0]])
        m = make_map([[(0, 1.0)]], 1, 1)
        D = np.array([80.0])
        snap = snap_inputs(state, m, D)

        def halved(real, operator, st_, requested, **kw):
            return real(operator, st_, requested * 0.5, **kw)

        with native_spy(halved) as calls:
            with self.assertRaises(bridge.SyrupWaterError) as cm:
                bridge.withdraw_plants(m, D, env.operator, state, max_bytes=BIG)
        self.assertEqual(len(calls), 1)
        self.assertIn('native_requested_matches_proposal', str(cm.exception))
        assert_unchanged(self, snap, state, m, D)

    def test_malformed_native_results_are_rejected(self):
        env = Env([0.0, 200.0], 1.0, [True, True], 'numpy')
        state = env.state([[100.0, 100.0]])
        m = make_map([[(0, 0.5), (1, 0.5)]], 1, 2)
        D = np.array([80.0])
        snap = snap_inputs(state, m, D)

        def tamper_budget(**fields):
            def wrapper(real, operator, st_, requested, **kw):
                res = real(operator, st_, requested, **kw)
                return dataclasses.replace(res, budget=dataclasses.replace(res.budget, **fields))
            return wrapper

        def new_time(real, operator, st_, requested, **kw):
            res = real(operator, st_, requested, **kw)
            return dataclasses.replace(res, state=ss.SoilColumnState(res.state.water_m, res.state.t_s + 1.0))

        def aliased(real, operator, st_, requested, **kw):
            return dataclasses.replace(real(operator, st_, requested, **kw), state=st_)

        def wrong_type(real, operator, st_, requested, **kw):
            return real(operator, st_, requested, **kw).budget

        def float32_actual(real, operator, st_, requested, **kw):
            res = real(operator, st_, requested, **kw)
            return dataclasses.replace(res, actual_m=res.actual_m.astype(np.float32))

        cases = {'nan actual': (tamper_budget(actual_m3=float('nan')), bridge.NativeContractError),
                 'inf unmet': (tamper_budget(unmet_m3=float('inf')), bridge.NativeContractError),
                 'negative actual': (tamper_budget(actual_m3=-1.0), bridge.NativeContractError),
                 'open budget': (tamper_budget(residual_m3=1.0), bridge.NativeContractError),
                 'actual above request': (tamper_budget(actual_m3=1000.0, residual_m3=0.0, request_residual_m3=0.0),
                                          bridge.SyrupWaterError),
                 'time changed': (new_time, bridge.NativeContractError),
                 'aliased input state': (aliased, bridge.NativeContractError),
                 'wrong result type': (wrong_type, bridge.NativeContractError),
                 'float32 actual': (float32_actual, bridge.NativeContractError)}
        for name, (wrapper, exc) in cases.items():
            with self.subTest(case=name):
                with native_spy(wrapper) as calls:
                    with self.assertRaises(exc):
                        bridge.withdraw_plants(m, D, env.operator, state, max_bytes=BIG)
                self.assertEqual(len(calls), 1)
                assert_unchanged(self, snap, state, m, D)


# ------------------------------------------------------------------ real native: guards, admission, atomicity
@needs_native
class TestGuardsAdmissionAndAtomicity(unittest.TestCase):
    def setUp(self):
        self.env = Env([0.0, 100.0, 300.0], 1.0, [True, True], 'numpy')
        self.state = self.env.state(np.full((2, 2), 10.0))
        self.m = make_map([[(0, 1.0)], [(3, 0.5), (1, 0.5)]], 2, 2)
        self.D = np.array([1.0, 2.0])
        self.snap = snap_inputs(self.state, self.m, self.D)

    def run_bridge(self, **kw):
        args = dict(root_map=self.m, demand_m3=self.D, operator=self.env.operator, soil_state=self.state)
        args.update(kw)
        return bridge.withdraw_plants(args['root_map'], args['demand_m3'], args['operator'], args['soil_state'],
                                      max_bytes=kw.get('max_bytes', BIG))

    def test_binding_refusals_before_any_allocation_or_native_call(self):
        class FakeDevice:
            __cuda_array_interface__ = {}
            shape, ndim, dtype, nbytes = (2,), 1, np.dtype(np.float64), 16

        env, L, N = self.env, 2, 2
        bad_state_water = {'wrong shape': np.zeros((2, 3)), 'float32': np.zeros((2, 2), dtype=np.float32),
                           'fortran order': np.asfortranarray(np.ones((2, 2)))}
        cases = []
        cases += [('map wrong layers', dict(root_map=make_map([[(0, 1.0)]], 1, 2)), bridge.BindingError),
                  ('map wrong cells', dict(root_map=make_map([[(0, 1.0)]], 2, 3)), bridge.BindingError),
                  ('not a map', dict(root_map='map'), bridge.BindingError)]
        cases += [('demand wrong length', dict(demand_m3=np.array([1.0, 2.0, 3.0])), bridge.BindingError),
                  ('demand list', dict(demand_m3=[1.0, 2.0]), bridge.BindingError),
                  ('demand bool', dict(demand_m3=np.array([True, False])), bridge.BindingError),
                  ('demand 2-D', dict(demand_m3=np.ones((2, 1))), bridge.BindingError),
                  ('demand object', dict(demand_m3=np.array(['a', 'b'])), bridge.BindingError),
                  ('demand device array on host column', dict(demand_m3=FakeDevice()), bridge.BindingError)]
        for name, w in bad_state_water.items():
            cases.append((f'state {name}', dict(soil_state=ss.SoilColumnState(w, 0.0)), bridge.BindingError))
        cases += [('state not a state', dict(soil_state=np.ones((2, 2))), bridge.BindingError),
                  ('state nan time', dict(soil_state=ss.SoilColumnState(np.ones((2, 2)), float('nan'))),
                   bridge.BindingError),
                  ('operator is the column', dict(operator=env.column), bridge.NativeContractError),
                  ('operator None', dict(operator=None), bridge.NativeContractError),
                  ('operator object', dict(operator=object()), bridge.NativeContractError)]
        with native_spy() as calls, mock.patch.object(uptake, 'propose', side_effect=AssertionError('allocator reached')), \
                mock.patch.object(bridge, '_snapshot', side_effect=AssertionError('snapshot allocated')):
            for name, kw, exc in cases:
                with self.subTest(case=name):
                    with self.assertRaises(exc):
                        self.run_bridge(**kw)
            for bad in (-1, 1.5, True, None, '10'):
                with self.subTest(max_bytes=bad):
                    with self.assertRaises(bridge.SyrupWaterError):
                        bridge.withdraw_plants(self.m, self.D, env.operator, self.state, max_bytes=bad)
            with self.assertRaises(bridge.SyrupWaterError):
                bridge.withdraw_plants(self.m, self.D, env.operator, self.state, max_bytes=BIG, extra_reserve_bytes=-5)
        with self.assertRaises(TypeError):                              # the budget is an explicit required keyword
            bridge.withdraw_plants(self.m, self.D, env.operator, self.state)
        self.assertEqual(calls, [])
        assert_unchanged(self, self.snap, self.state, self.m, self.D)

    def test_unsupported_native_version_is_refused(self):
        with native_spy() as calls, mock.patch.object(st, 'VERSION', '9.9.9'):
            with self.assertRaises(bridge.NativeContractError):
                self.run_bridge()
            with self.assertRaises(bridge.NativeContractError):
                bridge.availability_snapshot(self.env.operator, self.state, max_bytes=BIG)
        self.assertEqual(calls, [])

    def test_invalid_native_state_content_is_refused_before_proposal_and_native_call(self):
        env = self.env
        good = env.water(np.full((2, 2), 10.0))
        bad = {}
        for name, value, where in (('nan', np.nan, (0, 0)), ('inf', np.inf, (1, 1)), ('negative', -1.0, (0, 1)),
                                   ('above saturation', 1e6, (1, 0))):
            w = good.copy()
            w[where] = value
            bad[name] = w
        with native_spy() as calls, mock.patch.object(uptake, 'propose', side_effect=AssertionError('proposal reached')):
            for name, w in bad.items():
                with self.subTest(case=name):
                    state = env.state_from_water(w)
                    with self.assertRaises(bridge.SyrupWaterError):
                        bridge.withdraw_plants(self.m, self.D, env.operator, state, max_bytes=BIG)
                    with self.assertRaises(bridge.SyrupWaterError):
                        bridge.availability_snapshot(env.operator, state, max_bytes=BIG)
                    np.testing.assert_array_equal(host(state.water_m), w)
        self.assertEqual(calls, [])

    def test_invalid_demand_is_refused_before_native_call(self):
        with native_spy() as calls:
            for bad in (np.array([-1.0, 1.0]), np.array([np.nan, 1.0]), np.array([np.inf, 1.0])):
                with self.subTest(demand=bad):
                    with self.assertRaises(E_UPTAKE):
                        bridge.withdraw_plants(self.m, bad, self.env.operator, self.state, max_bytes=BIG)
        self.assertEqual(calls, [])
        assert_unchanged(self, self.snap, self.state, self.m, self.D)

    def test_byte_budget_refuses_before_allocation_and_native_execution(self):
        ledger = ledger_for(self.env, self.m, self.D)
        with native_spy() as calls, mock.patch.object(bridge, '_snapshot', side_effect=AssertionError('allocated')), \
                mock.patch.object(uptake, 'propose', side_effect=AssertionError('allocator reached')):
            for mb in (0, 1, ledger['total_bytes'] - 1):
                with self.subTest(max_bytes=mb):
                    with self.assertRaises(bridge.SyrupWaterAdmissionError) as cm:
                        self.run_bridge(max_bytes=mb)
                    self.assertEqual(cm.exception.stage, 'syrup_water.withdraw_plants')
                    self.assertEqual(cm.exception.estimate['total_bytes'], ledger['total_bytes'])
                    self.assertEqual(cm.exception.estimate['max_bytes'], mb)
        self.assertEqual(calls, [])
        assert_unchanged(self, self.snap, self.state, self.m, self.D)
        r = self.run_bridge(max_bytes=ledger['total_bytes'])                 # exactly the declared bound is admitted
        self.assertTrue(r.conservation['closed'])
        # caller reserve is part of the declared bound, never silently dropped
        ledger2 = ledger_for(self.env, self.m, self.D, extra=4096)
        self.assertEqual(ledger2['total_bytes'], ledger['total_bytes'] + 4096)
        with self.assertRaises(bridge.SyrupWaterAdmissionError):
            bridge.withdraw_plants(self.m, self.D, self.env.operator, self.state, max_bytes=ledger['total_bytes'],
                                   extra_reserve_bytes=4096)

    def test_native_available_bytes_is_checked_first_and_native_sees_the_same_requirement(self):
        box = [BIG]
        env = Env([0.0, 100.0, 300.0], 1.0, [True, True], 'numpy', reserve_bytes=1000,
                  available_bytes_fn=lambda: box[0])
        state = env.state(np.full((2, 2), 10.0))
        ledger = ledger_for(env, self.m, self.D, extra=500)
        self.assertEqual(ledger['reserve_bytes'], 1500)
        box[0] = ledger['native_required_bytes'] - 1
        with native_spy() as calls:
            with self.assertRaises(bridge.SyrupWaterAdmissionError):
                bridge.withdraw_plants(self.m, self.D, env.operator, state, max_bytes=BIG, extra_reserve_bytes=500)
        self.assertEqual(calls, [])
        box[0] = ledger['native_required_bytes']
        with native_spy() as calls:
            r = bridge.withdraw_plants(self.m, self.D, env.operator, state, max_bytes=BIG, extra_reserve_bytes=500)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['kw']['extra_reserve_bytes'], ledger['native_extra_reserve_bytes'])
        self.assertEqual(r.native.memory.required_bytes, ledger['native_required_bytes'])   # native admitted the same total
        self.assertEqual(ledger['native_own_bytes'] + ledger['native_extra_reserve_bytes'],
                         ledger['native_side_bytes'] + 500)

    def test_allocator_peak_is_prechecked_against_the_root_map_budget(self):
        m = make_map([[(0, 1.0)], [(3, 0.5), (1, 0.5)]], 2, 2)
        ledger = ledger_for(self.env, m, self.D)
        object.__setattr__(m, 'max_bytes', ledger['allocator_peak_bytes'] - 1)      # test-only: tighten the frozen cap
        with native_spy() as calls:
            with self.assertRaises(bridge.SyrupWaterAdmissionError):
                bridge.withdraw_plants(m, self.D, self.env.operator, self.state, max_bytes=BIG)
        self.assertEqual(calls, [])
        object.__setattr__(m, 'max_bytes', ledger['allocator_peak_bytes'])
        r = bridge.withdraw_plants(m, self.D, self.env.operator, self.state, max_bytes=BIG)   # allocator admits at its peak
        self.assertTrue(r.conservation['closed'])

    def test_failure_after_staging_preserves_inputs(self):
        # native failure
        def native_fail(real, operator, st_, requested, **kw):
            raise ss.SubsurfaceError('forced native failure')

        with native_spy(native_fail) as calls:
            with self.assertRaises(ss.SubsurfaceError):
                self.run_bridge()
        self.assertEqual(len(calls), 1)
        assert_unchanged(self, self.snap, self.state, self.m, self.D)
        # gather failure AFTER the native debit exists as a fresh (unpublished) state
        with native_spy() as calls, mock.patch.object(uptake, 'accept', side_effect=E_UPTAKE('forced gather failure')):
            with self.assertRaises(E_UPTAKE):
                self.run_bridge()
        self.assertEqual(len(calls), 1)
        assert_unchanged(self, self.snap, self.state, self.m, self.D)
        # native memory refusal on its own admission
        def native_oom(real, operator, st_, requested, **kw):
            raise ss.SubsurfaceMemoryError('forced native admission refusal')

        with native_spy(native_oom):
            with self.assertRaises(ss.SubsurfaceMemoryError):
                self.run_bridge()
        assert_unchanged(self, self.snap, self.state, self.m, self.D)
        # and a later healthy call still works on the very same inputs
        r = self.run_bridge()
        self.assertTrue(r.conservation['closed'])
        assert_unchanged(self, self.snap, self.state, self.m, self.D)


# ------------------------------------------------------------------ real native: availability snapshot
@needs_native
class TestAvailabilitySnapshot(unittest.TestCase):
    def test_projection_matches_the_native_law_on_every_cpu_backend(self):
        for nb in [c[0] for c in host_combos() if c[0] == c[1]]:
            with self.subTest(native=nb):
                area = 2.3
                env = Env([0.0, 100.0, 300.0], area, [True, False, True, True], nb)
                depth = np.array([[30.0, 10.0, 0.0, 5.0], [60.0, 5.0, 40.0, 0.0]])
                state = env.state(depth)
                snap = bridge.availability_snapshot(env.operator, state, max_bytes=BIG)
                expected = area * np.maximum(host(state.water_m) - env.res[:, None], 0.0)
                expected[:, 1] = 0.0                                          # inactive cell
                np.testing.assert_allclose(snap.volume_m3, expected, rtol=1e-15, atol=0)
                self.assertFalse(snap.volume_m3.flags.writeable)               # read-only projection
                self.assertAlmostEqual(snap.total_m3, float(expected.sum()), places=9)
                self.assertEqual((snap.n_layers, snap.n_cells, snap.n_active), (2, 4, 3))
                self.assertEqual(snap.t_s, state.t_s)
                self.assertEqual(snap.native_version, '0.2.0')
                self.assertIn('NOT a separate water store', snap.projection)

    def test_standalone_admission_counts_native_resident_copies(self):
        env = Env([0.0, 100.0, 300.0], 1.0, [True, True], 'numpy')
        state = env.state(np.full((2, 2), 10.0))
        ledger = bridge.estimate_availability_bytes(2, 2, native_backend='numpy', column_reserve_bytes=0)
        for name in ('native_state_input', 'native_operator_parameters', 'native_column_resident',
                     'native_layer_vectors_host', 'availability_buffer', 'availability_scratch'):
            self.assertIn(name, ledger['entries'])
        with mock.patch.object(bridge, '_project', side_effect=AssertionError('allocated')):
            with self.assertRaises(bridge.SyrupWaterAdmissionError):
                bridge.availability_snapshot(env.operator, state, max_bytes=ledger['total_bytes'] - 1)
        snap = bridge.availability_snapshot(env.operator, state, max_bytes=ledger['total_bytes'])
        np.testing.assert_array_equal(snap.volume_m3, np.full((2, 2), 10.0))


# ------------------------------------------------------------------ ledger scope (pure arithmetic against the native estimates)
@needs_native
class TestLedgerScope(unittest.TestCase):
    POP = {'root_map_resident', 'demand_input', 'allocator_plan', 'allocator_feedback'}

    def test_no_population_term_outside_map_demand_and_allocator(self):
        for nb, mb in (('numpy', 'numpy'), ('numba', 'numba'), ('cuda', 'cuda')):
            with self.subTest(native=nb):
                small = bridge.estimate_bridge_bytes(10, 30, 2, 5, map_backend=mb, native_backend=nb)
                large = bridge.estimate_bridge_bytes(10 ** 6, 3 * 10 ** 6, 2, 5, map_backend=mb, native_backend=nb)
                self.assertEqual(set(small['entries']), set(large['entries']))
                for key in small['entries']:
                    if key not in self.POP:
                        self.assertEqual(small['entries'][key]['bytes'], large['entries'][key]['bytes'], key)
                self.assertGreater(large['entries']['allocator_plan']['bytes'], small['entries']['allocator_plan']['bytes'])
                self.assertGreater(large['total_bytes'], small['total_bytes'])

    def test_native_kernel_parameter_layer_and_column_copies_are_counted(self):
        L, N = 3, 7
        par = 8 * ss.kernel_parameter_doubles(L)
        layer = 8 * ss.layer_vector_doubles(L)
        led = {nb: bridge.estimate_bridge_bytes(5, 10, L, N, map_backend=mb, native_backend=nb)
               for nb, mb in (('numpy', 'numpy'), ('numba', 'numba'), ('cuda', 'cuda'))}
        for nb, e in led.items():
            self.assertEqual(e['entries']['native_layer_vectors_host']['bytes'], layer)
            self.assertEqual(e['entries']['native_layer_vectors_host']['side'], 'host')
            est = ss.estimate_step_memory(L, N, backend=nb, reserve_bytes=0)
            self.assertEqual(e['entries']['native_column_resident']['bytes'], int(est.column_bytes))
            self.assertEqual(e['entries']['native_operator_parameters']['bytes'], 16 * L)
        self.assertNotIn('native_kernel_parameters_host', led['numpy']['entries'])
        self.assertEqual(led['numba']['entries']['native_kernel_parameters_host']['bytes'], par)
        self.assertNotIn('native_kernel_parameters_device', led['numba']['entries'])
        self.assertEqual(led['cuda']['entries']['native_kernel_parameters_host']['bytes'], par)
        self.assertEqual(led['cuda']['entries']['native_kernel_parameters_device']['bytes'], par)
        self.assertEqual(led['cuda']['entries']['native_kernel_parameters_device']['side'], 'device')
        # the cuda availability projection has NO scratch, only a 16-byte packet on each side
        self.assertNotIn('availability_scratch', led['cuda']['entries'])
        self.assertEqual(led['cuda']['entries']['availability_packet_device']['bytes'], 16)
        self.assertEqual(led['cuda']['entries']['availability_packet_host']['bytes'], 16)
        self.assertEqual(led['cuda']['entries']['allocator_host_packets']['bytes'],
                         8 * (uptake.NP_PACKET + uptake.NA_PACKET))
        # cuda: the device side carries the arrays, the host side only layers / parameter copies / packets
        self.assertGreater(led['cuda']['device_bytes'], led['cuda']['host_bytes'])
        self.assertEqual(led['cuda']['native_side_bytes'], led['cuda']['device_bytes'])
        self.assertEqual(led['numpy']['native_side_bytes'], led['numpy']['host_bytes'])
        self.assertEqual(led['numpy']['device_bytes'], 0)

    def test_ledger_covers_every_named_live_array_and_the_allocator_peak(self):
        L, N, P, E = 3, 11, 20, 50
        e = bridge.estimate_bridge_bytes(P, E, L, N, map_backend='numpy', native_backend='numpy', extra_reserve_bytes=123,
                                         column_reserve_bytes=77)
        LN = 8 * L * N
        lower_bound_fresh = 7 * LN                    # availability + request + 4 native outputs + accepted volume
        self.assertGreaterEqual(e['fresh_host_bytes'], lower_bound_fresh)
        self.assertEqual(e['reserve_bytes'], 200)
        self.assertEqual(e['total_bytes'], e['host_bytes'] + e['device_bytes'] + e['reserve_bytes'])
        plan, fb = e['plan_estimate'], e['feedback_estimate']
        map_res = e['entries']['root_map_resident']['bytes']
        self.assertEqual(e['allocator_peak_bytes'],
                         max(map_res + plan['total_bytes'], map_res + plan['owned_bytes'] + fb['total_bytes']))
        self.assertIn('NOT included', e['scope'])
        self.assertIn('Numba/NVRTC JIT', e['scope'])
        with self.assertRaises(bridge.BindingError):                                  # host map on a cuda column
            bridge.estimate_bridge_bytes(1, 1, 1, 1, map_backend='numpy', native_backend='cuda')
        with self.assertRaises(bridge.SyrupWaterError):
            bridge.estimate_bridge_bytes(1, 1, 1, 1, map_backend='numpy', native_backend='opencl')

    def test_cpu_fresh_allocations_stay_inside_the_declared_host_bound(self):
        '''tracemalloc (NumPy arrays included) over one call: peak fresh bytes <= the ledger's fresh host bytes plus a
        fixed Python-bookkeeping slack (excluded from the ledger by its declared scope). Population-heavy and
        grid-heavy shapes.'''
        slack = 128 * 1024
        for name, (L, N, P) in {'grid_heavy': (3, 20000, 6000), 'population_heavy': (2, 1, 200000)}.items():
            with self.subTest(shape=name):
                rng = np.random.default_rng(7)
                R = L * N
                edges = np.arange(L + 1, dtype=np.float64) * 100.0
                env = Env(edges, 1.0, np.ones(N, dtype=bool), 'numpy')
                state = env.state(rng.uniform(0.0, 70.0, (L, N)))
                r1 = rng.integers(0, R, P)
                if R > 1:
                    r2 = (r1 + 1 + rng.integers(0, R - 1, P)) % R
                    res = np.stack([r1, r2], axis=1).reshape(-1).astype(np.int64)
                    w = np.full(2 * P, 0.5)
                    off = 2 * np.arange(P + 1, dtype=np.int64)
                else:
                    res = np.zeros(P, dtype=np.int64)
                    w = np.ones(P)
                    off = np.arange(P + 1, dtype=np.int64)
                m = uptake.RootMap.from_arrays(np.arange(1, P + 1, dtype=np.int64), off, res, w, n_layers=L, n_cells=N,
                                               backend='numpy', max_edges=10_000_000, max_bytes=BIG)
                D = rng.uniform(0.0, 5.0, P)
                ledger = ledger_for(env, m, D)
                bridge.withdraw_plants(m, D, env.operator, state, max_bytes=BIG)      # warm-up (imports, first calls)
                gc.collect()
                tracemalloc.start()
                try:
                    base, _ = tracemalloc.get_traced_memory()
                    tracemalloc.reset_peak()
                    r = bridge.withdraw_plants(m, D, env.operator, state, max_bytes=ledger['total_bytes'])
                    _, peak = tracemalloc.get_traced_memory()
                finally:
                    tracemalloc.stop()
                fresh_peak = peak - base
                self.assertLessEqual(fresh_peak, ledger['fresh_host_bytes'] + slack,
                                     f'{name}: measured fresh peak {fresh_peak} exceeds ledger {ledger["fresh_host_bytes"]}')
                self.assertTrue(r.conservation['closed'])
                del r


# ------------------------------------------------------------------ CUDA (real device only; no foreign-device allocation)
def cuda_available():
    return bool(NATIVE and uptake.backend_status('cuda')['available'])


@unittest.skipUnless(NATIVE, 'native SYRUP package unavailable')
class TestCudaBridge(unittest.TestCase):
    def setUp(self):
        if not cuda_available():
            self.skipTest('cuda backend unavailable')
        import cupy
        self.cp = cupy

    def test_user_75_percent_on_the_device(self):
        cp = self.cp
        env = Env([0.0, 100.0, 1000.0], 1.0, [True], 'cuda')
        state = env.state([[25.0], [200.0]])
        m = make_map([[(0, 0.5), (1, 0.5)]], 2, 1, 'cuda')
        D = cp.asarray([100.0])
        snap = snap_inputs(state, m, D)
        with native_spy() as calls:
            r = bridge.withdraw_plants(m, D, env.operator, state, max_bytes=BIG)
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(r.native_state.water_m, cp.ndarray)
        self.assertIsInstance(r.feedback.delivered, cp.ndarray)
        self.assertEqual(float(r.delivered_m3[0]), 75.0)
        self.assertEqual(float(r.satisfaction[0]), 0.75)
        np.testing.assert_array_equal(host(r.native_state.water_m), [[12.5], [262.5]])
        self.assertEqual(r.native_state.t_s, 1234.5)
        self.assertTrue(r.conservation['closed'])
        self.assertEqual(r.binding['device_id'], int(env.column.device_id))
        assert_unchanged(self, snap, state, m, D)

    def test_cpu_gpu_parity_with_inactive_cells_and_nonunit_area(self):
        cp = self.cp
        rng = np.random.default_rng(11)
        L, N, P, area = 2, 13, 90, 2.3
        rows = random_rows(rng, P, L * N)
        D = rng.uniform(0.0, 40.0, P) * area
        depth = rng.uniform(0.0, 60.0, (L, N))
        active = rng.random(N) > 0.25
        active[0] = True
        edges = [0.0, 100.0, 300.0]
        cpu = Env(edges, area, active, 'numpy')
        gpu = Env(edges, area, active, 'cuda')
        rc = bridge.withdraw_plants(make_map(rows, L, N), D, cpu.operator, cpu.state(depth), max_bytes=BIG)
        gstate = gpu.state(depth)
        gm = make_map(rows, L, N, 'cuda')
        gD = cp.asarray(D)
        snap = snap_inputs(gstate, gm, gD)
        rg = bridge.withdraw_plants(gm, gD, gpu.operator, gstate, max_bytes=BIG)
        np.testing.assert_allclose(host(rg.availability.volume_m3), rc.availability.volume_m3, rtol=1e-14, atol=0)
        np.testing.assert_allclose(host(rg.delivered_m3), rc.delivered_m3, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(host(rg.native_state.water_m), rc.native_state.water_m, rtol=1e-12)
        np.testing.assert_array_equal(host(rg.feedback.accepted), host(rg.native.actual_m * area).reshape(-1))
        self.assertTrue(rg.conservation['closed'])
        assert_unchanged(self, snap, gstate, gm, gD)

    def test_namespace_guards_need_no_second_device(self):
        cp = self.cp
        env = Env([0.0, 200.0], 1.0, [True, True], 'cuda')
        state = env.state([[10.0, 10.0]])
        rows = [[(0, 0.5), (1, 0.5)]]
        gm = make_map(rows, 1, 2, 'cuda')
        hm = make_map(rows, 1, 2, 'numpy')
        with native_spy() as calls:
            with self.assertRaises(bridge.BindingError):                          # host map on a cuda column
                bridge.withdraw_plants(hm, cp.asarray([1.0]), env.operator, state, max_bytes=BIG)
            with self.assertRaises(bridge.BindingError):                          # host demand on a cuda column
                bridge.withdraw_plants(gm, np.array([1.0]), env.operator, state, max_bytes=BIG)
            host_env = Env([0.0, 200.0], 1.0, [True, True], 'numpy')
            with self.assertRaises(bridge.BindingError):                          # cuda map on a host column
                bridge.withdraw_plants(gm, np.array([1.0]), host_env.operator, host_env.state([[10.0, 10.0]]), max_bytes=BIG)
            with self.assertRaises(bridge.BindingError):                          # device demand on a host column
                bridge.withdraw_plants(hm, cp.asarray([1.0]), host_env.operator, host_env.state([[10.0, 10.0]]), max_bytes=BIG)
        self.assertEqual(calls, [])

    def test_gpu_availability_snapshot_and_invalid_state(self):
        env = Env([0.0, 100.0, 300.0], 2.3, [True, False, True], 'cuda')
        depth = np.array([[30.0, 10.0, 0.0], [60.0, 5.0, 40.0]])
        state = env.state(depth)
        snap = bridge.availability_snapshot(env.operator, state, max_bytes=BIG)
        expected = 2.3 * np.maximum(host(state.water_m) - env.res[:, None], 0.0)
        expected[:, 1] = 0.0
        np.testing.assert_allclose(host(snap.volume_m3), expected, rtol=1e-15, atol=0)
        self.assertAlmostEqual(snap.total_m3, float(expected.sum()), places=9)
        w = env.water(depth)
        w[1, 2] = np.nan
        with self.assertRaises(bridge.SyrupWaterError):
            bridge.availability_snapshot(env.operator, env.state_from_water(w), max_bytes=BIG)

    def test_failure_after_staging_preserves_device_inputs(self):
        cp = self.cp
        env = Env([0.0, 200.0], 1.0, [True, True], 'cuda')
        state = env.state([[10.0, 20.0]])
        m = make_map([[(0, 0.5), (1, 0.5)]], 1, 2, 'cuda')
        D = cp.asarray([5.0])
        snap = snap_inputs(state, m, D)

        def native_fail(real, operator, st_, requested, **kw):
            raise ss.SubsurfaceError('forced native failure')

        with native_spy(native_fail):
            with self.assertRaises(ss.SubsurfaceError):
                bridge.withdraw_plants(m, D, env.operator, state, max_bytes=BIG)
        with mock.patch.object(uptake, 'accept', side_effect=E_UPTAKE('forced gather failure')):
            with self.assertRaises(E_UPTAKE):
                bridge.withdraw_plants(m, D, env.operator, state, max_bytes=BIG)
        assert_unchanged(self, snap, state, m, D)
        with self.assertRaises(bridge.SyrupWaterAdmissionError):
            bridge.withdraw_plants(m, D, env.operator, state, max_bytes=1)
        assert_unchanged(self, snap, state, m, D)

    def test_declared_device_bound_covers_the_hooked_live_peak(self):
        '''Regression for the omitted device scope: hooked live pool bytes (rounded sizes) of one call must fit the
        ledger's fresh device bytes plus the pool's 512-byte rounding per allocation (rounding is NOT the defect).
        Shapes: population-heavy (P=200000, N=1, L=2) and grid-heavy (P=4096, N=65536, L=3).'''
        cp = self.cp
        try:
            from cupy.cuda import memory_hook
        except ImportError:
            self.skipTest('cupy memory hooks unavailable')

        class PeakHook(memory_hook.MemoryHook):
            name = 'leaf_syrup_water_peak'

            def __init__(self):
                self.live = 0
                self.peak = 0
                self.n_malloc = 0

            def malloc_postprocess(self, **kw):
                self.n_malloc += 1
                self.live += int(kw['mem_size'])
                self.peak = max(self.peak, self.live)

            def free_postprocess(self, **kw):
                self.live -= int(kw['mem_size'])

        for name, (L, N, P) in {'population_heavy': (2, 1, 200000), 'grid_heavy': (3, 65536, 4096)}.items():
            with self.subTest(shape=name):
                rng = np.random.default_rng(3)
                R = L * N
                edges = np.arange(L + 1, dtype=np.float64) * 100.0
                env = Env(edges, 1.0, np.ones(N, dtype=bool), 'cuda')
                state = env.state(rng.uniform(0.0, 70.0, (L, N)))
                if R > 1:
                    r1 = rng.integers(0, R, P)
                    r2 = (r1 + 1 + rng.integers(0, R - 1, P)) % R
                    res = np.stack([r1, r2], axis=1).reshape(-1).astype(np.int64)
                    w = np.full(2 * P, 0.5)
                    off = 2 * np.arange(P + 1, dtype=np.int64)
                else:
                    res, w, off = np.zeros(P, dtype=np.int64), np.ones(P), np.arange(P + 1, dtype=np.int64)
                m = uptake.RootMap.from_arrays(np.arange(1, P + 1, dtype=np.int64), off, res, w, n_layers=L, n_cells=N,
                                               backend='cuda', max_edges=10_000_000, max_bytes=BIG)
                D = cp.asarray(rng.uniform(0.0, 5.0, P))
                ledger = ledger_for(env, m, D)
                bridge.withdraw_plants(m, D, env.operator, state, max_bytes=BIG)      # warm-up: compile kernels
                cp.cuda.Device().synchronize()
                hook = PeakHook()
                with hook:
                    r = bridge.withdraw_plants(m, D, env.operator, state, max_bytes=ledger['total_bytes'])
                    cp.cuda.Device().synchronize()
                slack = 512 * hook.n_malloc
                self.assertLessEqual(hook.peak, ledger['fresh_device_bytes'] + slack,
                                     f'{name}: hooked live peak {hook.peak} B exceeds the declared fresh device bound '
                                     f'{ledger["fresh_device_bytes"]} B + rounding slack {slack} B')
                self.assertTrue(r.conservation['closed'])
                del r


if __name__ == '__main__':
    unittest.main()
