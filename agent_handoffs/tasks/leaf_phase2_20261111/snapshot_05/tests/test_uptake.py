'''LEAF Phase 1 part A: sparse uptake allocator tests (unittest, NumPy only unless a backend is requested).

Run from the candidate root:  python -m unittest tests.test_uptake -v
numba / cuda tests SKIP only when leaf.uptake.backend_status reports the backend unavailable; a real
implementation failure on an available backend fails the test. CUDA tests only touch the CURRENT device (they
never select or allocate on another device).
'''
from __future__ import annotations

import math
import tracemalloc
import types
import unittest

import numpy as np

from leaf import uptake

E = uptake.UptakeError
BIG = 1 << 26
INT64_MAX, INT64_MIN = 2 ** 63 - 1, -(2 ** 63)


def csr(rows, ids=None):
    '''rows: per plant, list of (resource, weight) -> typed (ids, offsets, resources, weights) arrays.'''
    ids = np.arange(1, len(rows) + 1, dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    res, w = [], []
    for i, row in enumerate(rows):
        offsets[i + 1] = offsets[i] + len(row)
        for r, ww in row:
            res.append(r)
            w.append(ww)
    return ids, offsets, np.array(res, dtype=np.int64), np.array(w, dtype=np.float64)


def make_map(rows, L, N, backend='numpy', ids=None, max_edges=100_000, max_bytes=BIG):
    ids, off, res, w = csr(rows, ids)
    return uptake.RootMap.from_arrays(ids, off, res, w, n_layers=L, n_cells=N, backend=backend,
                                      max_edges=max_edges, max_bytes=max_bytes)


def oracle_proposal(rows, D, A):
    '''Scalar oracle of the user water rules, written independently of the module with explicit loops.'''
    R = A.size
    Q = np.zeros(R)
    for p, row in enumerate(rows):
        for r, w in row:
            Q[r] += D[p] * w
    proposed = np.array([min(A[r], Q[r]) if Q[r] > 0 else 0.0 for r in range(R)])
    f = np.array([min(1.0, A[r] / Q[r]) if Q[r] > 0 else 0.0 for r in range(R)])
    return Q, proposed, f


def oracle_delivered(rows, D, Q, accepted):
    out = np.zeros(len(rows))
    for p, row in enumerate(rows):
        for r, w in row:
            if Q[r] > 0:
                out[p] += D[p] * w * accepted[r] / Q[r]
    return out


def random_rows(rng, P, R, max_edges=6):
    rows = []
    for _ in range(P):
        k = int(rng.integers(1, max_edges + 1))
        res = rng.choice(R, size=k, replace=False)
        w = rng.uniform(0.05, 1.0, size=k)
        w /= w.sum()
        rows.append([(int(r), float(x)) for r, x in zip(res, w)])
    return rows


class Trap(list):
    '''A sequence that fails the test if anything iterates, indexes or converts it (no materialisation).'''

    def __iter__(self):
        raise AssertionError('sequence was iterated')

    def __getitem__(self, i):
        raise AssertionError('sequence was indexed')

    def __array__(self, *a, **k):
        raise AssertionError('sequence was converted to an array')


class TestContractExamples(unittest.TestCase):
    def test_required_example_one_plant_two_layers_75_percent(self):
        # one plant, D = 100 m3, weights .5 / .5 in two layers of one cell, A = 25 / 200
        m = make_map([[(0, 0.5), (1, 0.5)]], L=2, N=1)
        A = np.array([[25.0], [200.0]])
        plan = uptake.propose(m, np.array([100.0]), A)
        np.testing.assert_array_equal(plan.request, [50.0, 50.0])
        np.testing.assert_array_equal(plan.proposed, [25.0, 50.0])
        np.testing.assert_array_equal(plan.fraction, [0.5, 1.0])
        fb = uptake.accept(plan, plan.proposed_grid())          # SYRUP accepts the proposal as is
        self.assertEqual(float(fb.delivered[0]), 75.0)
        self.assertEqual(float(fb.unmet[0]), 25.0)
        self.assertEqual(float(fb.satisfaction[0]), 0.75)
        np.testing.assert_array_equal(A.reshape(-1) - fb.accepted, [0.0, 150.0])    # deeper remaining water is NOT used
        self.assertTrue(fb.closure['closed'])
        self.assertEqual(fb.summary['n_roundoff_excess'], 0)
        self.assertEqual(plan.summary['n_resources_limited'], 1)

    def test_multi_plant_sharing_is_proportional_to_local_requests(self):
        # two plants on one resource: requests 100 and 300, supply 200 -> same fraction 0.5 for both
        m = make_map([[(0, 1.0)], [(0, 1.0)]], L=1, N=1)
        fb = uptake.propose(m, np.array([100.0, 300.0]), np.array([200.0])).accept_all()
        np.testing.assert_allclose(fb.delivered, [50.0, 150.0], rtol=1e-15)        # not equal litres (100/100)
        np.testing.assert_allclose(fb.satisfaction, [0.5, 0.5], rtol=1e-15)        # the SAME fraction
        m2 = make_map([[(0, 1.0)], [(0, 1.0)]], L=1, N=1, ids=[2, 1])               # order / ids do not matter
        fb2 = uptake.propose(m2, np.array([300.0, 100.0]), np.array([200.0])).accept_all()
        np.testing.assert_allclose(fb2.delivered, [150.0, 50.0], rtol=1e-15)
        self.assertTrue(fb.closure['closed'] and fb2.closure['closed'])

    def test_non_separable_weights_multiple_cells_and_layers(self):
        L, N = 2, 3
        rows = [[(0, 0.4), (1, 0.4), (N + 0, 0.2)],
                [(N + 1, 1.0)],
                [(1, 0.5), (N + 1, 0.5)]]
        D = np.array([10.0, 20.0, 40.0])
        A = np.array([[100.0, 10.0, 0.0], [1.0, 30.0, 5.0]])
        Q, proposed, f = oracle_proposal(rows, D, A.reshape(-1))
        m = make_map(rows, L, N)
        plan = uptake.propose(m, D, A)
        np.testing.assert_allclose(plan.request, Q, rtol=1e-15)
        np.testing.assert_allclose(plan.proposed, proposed, rtol=1e-15)
        np.testing.assert_allclose(plan.fraction, f, rtol=1e-15)
        fb = plan.accept_all()
        np.testing.assert_allclose(fb.delivered, oracle_delivered(rows, D, Q, proposed), rtol=1e-14)
        np.testing.assert_allclose(fb.unmet, D - fb.delivered, rtol=1e-14)
        self.assertTrue(np.all(fb.satisfaction <= 1.0) and np.all(fb.satisfaction >= 0.0))
        self.assertAlmostEqual(float(plan.fraction[1]), 10.0 / 24.0, places=15)    # cell 1 shallow: 4 + 20 requested vs 10
        self.assertTrue(fb.closure['closed'])

    def test_zero_demand_zero_supply_and_inactive_roots(self):
        m = make_map([[(0, 1.0)], [], [(0, 0.5), (1, 0.5)]], L=1, N=2)
        self.assertEqual(m.n_inactive_plants, 1)
        plan = uptake.propose(m, np.array([0.0, 0.0, 10.0]), np.array([0.0, 0.0]))   # dry everywhere
        np.testing.assert_array_equal(plan.proposed, [0.0, 0.0])
        fb = plan.accept_all()
        np.testing.assert_array_equal(fb.delivered, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(fb.satisfaction, [0.0, 0.0, 0.0])     # zero demand -> S = 0, not 1
        np.testing.assert_array_equal(fb.unmet, [0.0, 0.0, 10.0])
        self.assertEqual(fb.summary['n_plants_zero_demand'], 2)
        D = np.array([0.0, 1.0, 0.0])
        with self.assertRaises(E):                                           # positive demand on an empty root row
            uptake.propose(m, D, np.array([5.0, 5.0]))
        np.testing.assert_array_equal(D, [0.0, 1.0, 0.0])                    # inputs preserved
        empty = uptake.RootMap.from_arrays(np.zeros(0, np.int64), np.zeros(1, np.int64), np.zeros(0, np.int64),
                                           np.zeros(0), n_layers=2, n_cells=3, backend='numpy', max_edges=10, max_bytes=BIG)
        plan = uptake.propose(empty, np.zeros(0), np.ones((2, 3)))
        np.testing.assert_array_equal(plan.proposed, np.zeros(6))
        fb = plan.accept_all()
        self.assertEqual(fb.delivered.shape, (0,))
        self.assertTrue(fb.closure['closed'])

    def test_masked_cell_gets_nothing_and_weight_is_not_renormalised(self):
        m = make_map([[(0, 0.5), (1, 0.5)]], L=1, N=2)
        plan = uptake.propose(m, np.array([100.0]), np.array([0.0, 1000.0]))
        np.testing.assert_array_equal(plan.proposed, [0.0, 50.0])         # the dry cell's request is not moved
        fb = plan.accept_all()
        self.assertEqual(float(fb.delivered[0]), 50.0)
        self.assertEqual(float(fb.unmet[0]), 50.0)
        self.assertEqual(float(fb.satisfaction[0]), 0.5)
        self.assertEqual(float(1000.0 - fb.accepted[1]), 950.0)            # wet cell residual untouched
        np.testing.assert_array_equal(m.weights, [0.5, 0.5])

    def test_partial_acceptance_shortfall_and_honest_roundoff(self):
        rows = [[(0, 0.5), (1, 0.5)], [(1, 1.0)]]
        m = make_map(rows, L=1, N=2)
        D = np.array([100.0, 100.0])
        plan = uptake.propose(m, D, np.array([40.0, 90.0]))
        np.testing.assert_array_equal(plan.proposed, [40.0, 90.0])         # Q = [50, 150]
        half = plan.proposed * 0.5
        fb = uptake.accept(plan, half)                                        # SYRUP delivers half of the proposal
        np.testing.assert_allclose(fb.delivered, oracle_delivered(rows, D, plan.request, half), rtol=1e-14)
        np.testing.assert_array_equal(fb.accepted, half)                      # the native debit is kept unchanged
        self.assertAlmostEqual(fb.summary['shortfall_total_m3'], 65.0, places=12)
        self.assertTrue(fb.closure['closed'])
        before = plan.proposed.copy()
        for bad in (plan.proposed * 1.001, np.array([40.0, 90.0 + 1e-6]), np.array([-1.0, 0.0]),
                    np.array([np.nan, 0.0]), np.array([40.0, np.inf]), np.array([[40.0], [90.0]]), np.array([40.0]),
                    np.array([True, False]), np.array([40, 90]).astype(object), 'accepted', [40.0, 90.0]):
            with self.assertRaises(E, msg=repr(bad)):
                uptake.accept(plan, bad)
        np.testing.assert_array_equal(plan.proposed, before)
        # rounding-level excess (one ulp over the proposal) is KEPT as supplied and reported, never clipped
        tiny = plan.proposed * (1.0 + uptake.FP64_EPS)
        self.assertTrue(np.all(tiny > plan.proposed))
        fb2 = uptake.accept(plan, tiny)
        np.testing.assert_array_equal(fb2.accepted, tiny)                     # not modified
        self.assertEqual(fb2.summary['n_roundoff_excess'], 2)
        self.assertGreater(fb2.summary['max_roundoff_excess_m3'], 0.0)
        self.assertLessEqual(fb2.summary['max_roundoff_excess_m3'], 90.0 * 4 * uptake.FP64_EPS)
        np.testing.assert_allclose(fb2.delivered, oracle_delivered(rows, D, plan.request, tiny), rtol=1e-14)
        self.assertEqual(fb2.closure['accepted_total_m3'], float(tiny.sum()))  # closure is against the supplied water
        self.assertTrue(fb2.closure['closed'])
        # water on a resource nobody requested is refused (Q == 0 -> proposed 0)
        m0 = make_map([[(0, 1.0)]], L=1, N=2)
        plan0 = uptake.propose(m0, np.array([10.0]), np.array([5.0, 5.0]))
        with self.assertRaises(E):
            uptake.accept(plan0, np.array([5.0, 1.0]))
        # foreign objects and wrong shapes are refused
        with self.assertRaises(E):
            uptake.accept(plan.proposed, plan.proposed)
        with self.assertRaises(E):
            uptake.propose(m, D, np.array([1.0, 2.0, 3.0]))
        with self.assertRaises(E):
            uptake.propose('map', D, np.array([1.0, 2.0]))
        for bad_d in ([100.0, 100.0], np.array([-1.0, 1.0]), np.array([np.nan, 1.0]), np.array([True, False]), np.zeros(3)):
            with self.assertRaises(E, msg=repr(bad_d)):
                uptake.propose(m, bad_d, np.array([40.0, 90.0]))
        for bad_s in (np.array([-1.0, 1.0]), np.array([np.inf, 1.0]), np.array([np.nan, 1.0])):
            with self.assertRaises(E, msg=repr(bad_s)):
                uptake.propose(m, D, bad_s)


class TestFiniteResults(unittest.TestCase):
    def test_overflowing_totals_and_scatter_are_refused(self):
        # two plants on SEPARATE resources, finite D = A = 1e308 each: the aggregate demand / supply totals overflow
        m = make_map([[(0, 1.0)], [(1, 1.0)]], L=1, N=2)
        D, A = np.array([1e308, 1e308]), np.array([1e308, 1e308])
        with self.assertRaises(E):
            uptake.propose(m, D, A)
        np.testing.assert_array_equal(D, [1e308, 1e308])                     # inputs preserved
        # two plants on ONE resource: the scatter Q[r] itself overflows (both the request and the totals are non-finite)
        m1 = make_map([[(0, 1.0)], [(0, 1.0)]], L=1, N=1)
        with self.assertRaises(E):
            uptake.propose(m1, np.array([1e308, 1e308]), np.array([1e308]))
        # a single large but representable plan is fine and every reported number is finite
        ok = uptake.propose(m1, np.array([1e307, 1e307]), np.array([1e308])).accept_all()
        self.assertTrue(ok.closure['closed'])
        for k in ('accepted_total_m3', 'delivered_total_m3', 'unmet_total_m3'):
            self.assertTrue(math.isfinite(ok.summary[k]), k)
        self.assertTrue(np.all(np.isfinite(ok.delivered)) and np.all(np.isfinite(ok.satisfaction)))

    def test_nonfinite_accepted_and_closure_flag(self):
        m = make_map([[(0, 1.0)]], L=1, N=1)
        plan = uptake.propose(m, np.array([2.0]), np.array([1.0]))
        for bad in (np.array([np.inf]), np.array([np.nan]), np.array([-0.5])):
            with self.assertRaises(E):
                uptake.accept(plan, bad)
        fb = plan.accept_all()
        self.assertIs(fb.closure['closed'], True)
        self.assertTrue(math.isfinite(fb.closure['residual_m3']) and math.isfinite(fb.closure['tolerance_m3']))


class TestRootMapValidation(unittest.TestCase):
    def test_invalid_arrays_refused_and_inputs_preserved(self):
        ids, off, res, w = csr([[(0, 0.5), (1, 0.5)], [(1, 1.0)]])
        keep = [a.copy() for a in (ids, off, res, w)]
        ok = uptake.RootMap.from_arrays(ids, off, res, w, n_layers=1, n_cells=2, backend='numpy', max_edges=10, max_bytes=BIG)
        self.assertEqual((ok.n_plants, ok.n_edges, ok.n_resources), (2, 3, 2))
        bad_cases = [
            dict(ids=np.array([1, 1])),                                   # duplicate ids
            dict(ids=np.array([True, False])),                            # bool ids
            dict(ids=np.array([1.0, 2.0])),                               # float ids (lossy cast refused)
            dict(ids=np.array([2 ** 63, 1], dtype=np.uint64)),            # unsigned overflow
            dict(ids=np.array([1, 2, 3])),                                # shape mismatch with offsets
            dict(offsets=np.array([0, 2, 4])),                            # offsets end != E
            dict(offsets=np.array([1, 2, 3])),                            # offsets[0] != 0
            dict(offsets=np.array([0, 3, 3])),                            # valid shape, but row 0 then sums to 2
            dict(offsets=np.array([0, 2.0, 3.0])),                        # float offsets
            dict(offsets=np.array([0, 2, 3], dtype=np.uint64) + np.uint64(2 ** 63)),   # unsigned overflow offsets
            dict(resources=np.array([0, 1, 2])),                          # index out of range
            dict(resources=np.array([0, -1, 1])),                         # negative index
            dict(resources=np.array([0.0, 1.0, 1.0])),                    # float indices
            dict(weights=np.array([0.5, 0.5, 0.9])),                      # row sum != 1
            dict(weights=np.array([0.5, 0.6, 1.0])),                      # row sum != 1 (not renormalised)
            dict(weights=np.array([0.5, np.nan, 1.0])),                   # non-finite
            dict(weights=np.array([-0.5, 1.5, 1.0])),                     # negative
            dict(weights=np.array([True, False, True])),                  # bool weights
            dict(weights=np.array([0.5, 0.5])),                           # length mismatch
            dict(n_layers=0), dict(n_cells=1.5), dict(n_layers=True), dict(backend='opencl'),
        ]
        for case in bad_cases:
            kw = dict(ids=ids, offsets=off, resources=res, weights=w, n_layers=1, n_cells=2, backend='numpy',
                      max_edges=10, max_bytes=BIG)
            kw.update(case)
            with self.assertRaises(E, msg=repr(case)):
                uptake.RootMap.from_arrays(kw['ids'], kw['offsets'], kw['resources'], kw['weights'], n_layers=kw['n_layers'],
                                           n_cells=kw['n_cells'], backend=kw['backend'], max_edges=kw['max_edges'],
                                           max_bytes=kw['max_bytes'])
        for a, k in zip((ids, off, res, w), keep):
            np.testing.assert_array_equal(a, k)                           # failures never touch caller arrays
        # ownership: the map holds frozen copies; mutating the caller's arrays afterwards changes nothing
        w[0] = 0.9
        res[0] = 1
        off[1] = 3
        ids[0] = 99
        np.testing.assert_array_equal(ok.weights, [0.5, 0.5, 1.0])
        np.testing.assert_array_equal(ok.resources, [0, 1, 1])
        np.testing.assert_array_equal(ok.offsets, [0, 2, 3])
        np.testing.assert_array_equal(ok.ids, [1, 2])
        for arr in (ok.weights, ok.resources, ok.offsets, ok.ids, ok.counts):
            with self.assertRaises(ValueError):
                arr[0] = 1
        with self.assertRaises(AttributeError):
            ok.weights = np.zeros(3)
        self.assertEqual(ok.max_edges_per_resource, 2)
        self.assertEqual(ok.max_edges_per_plant, 2)
        self.assertIsNone(ok.device_id)
        self.assertIn('scope', ok.describe()['admission'])
        self.assertIn('exclusions', ok.describe()['admission'])
        make_map([[(0, 0.5), (1, 0.5 + 0.5e-9)]], L=1, N=2)               # within WEIGHT_SUM_TOL
        with self.assertRaises(E):
            make_map([[(0, 0.5), (1, 0.5 + 5e-9)]], L=1, N=2)            # beyond it: refused, never normalised

    def test_offsets_overflow_is_rejected_before_any_subtraction(self):
        # E = 0, P = 4: a wrapped np.diff of these offsets would give positive counts although nothing is in range
        bad_offsets = np.array([0, INT64_MAX, INT64_MIN + 1, -1, 0], dtype=np.int64)
        ids = np.arange(4, dtype=np.int64)
        keep = bad_offsets.copy()
        with self.assertRaises(E) as cm:
            uptake.RootMap.from_arrays(ids, bad_offsets, np.zeros(0, np.int64), np.zeros(0), n_layers=1, n_cells=4,
                                       backend='numpy', max_edges=10, max_bytes=BIG)
        self.assertIn('offset', str(cm.exception))
        np.testing.assert_array_equal(bad_offsets, keep)
        for off in ([0, 5, 2, 3], [0, 3, 2, 3], [0, 4, 3, 3], [0, 1, 0, 3]):     # out of range / decreasing, E = 3, P = 4
            o = np.array(off + [3], dtype=np.int64)
            with self.assertRaises(E, msg=repr(off)):
                uptake.RootMap.from_arrays(np.arange(len(o) - 1, dtype=np.int64), o, np.array([0, 1, 2], np.int64),
                                           np.full(3, 1.0), n_layers=1, n_cells=3, backend='numpy', max_edges=10,
                                           max_bytes=BIG)

    def test_typed_arrays_only_and_no_materialisation(self):
        ids, off, res, w = csr([[(0, 0.5), (1, 0.5)], [(1, 1.0)]])
        # a mixed bool list never becomes an integer ID vector; the sequence is not even iterated or converted
        for seq in ([True, 2], Trap([1, 2]), (1, 2), range(2)):
            with self.assertRaises(E, msg=repr(type(seq))):
                uptake.RootMap.from_arrays(seq, off, res, w, n_layers=1, n_cells=2, backend='numpy', max_edges=10, max_bytes=BIG)
            for pos, name in ((1, 'offsets'), (2, 'resources'), (3, 'weights')):
                args = [ids, off, res, w]
                args[pos] = seq
                with self.assertRaises(E, msg=name):
                    uptake.RootMap.from_arrays(*args, n_layers=1, n_cells=2, backend='numpy', max_edges=10, max_bytes=BIG)
        m = uptake.RootMap.from_arrays(ids, off, res, w, n_layers=1, n_cells=2, backend='numpy', max_edges=10, max_bytes=BIG)
        with self.assertRaises(E):
            uptake.propose(m, Trap([1.0, 1.0]), np.array([1.0, 1.0]))
        with self.assertRaises(E):
            uptake.propose(m, np.array([1.0, 1.0]), Trap([1.0, 1.0]))
        with self.assertRaises(E):
            uptake.propose(m, np.array([1.0, 1.0], dtype=object), np.array([1.0, 1.0]))
        with self.assertRaises(E):
            uptake.propose(m, np.array([1, 1], dtype=complex), np.array([1.0, 1.0]))

        class FakeDevice:
            __cuda_array_interface__ = {}
            shape, ndim, dtype = (2,), 1, np.dtype(np.float64)
        with self.assertRaises(E):                                        # host backends refuse device-like arrays
            uptake.propose(m, FakeDevice(), np.array([1.0, 1.0]))

    def test_device_identity_guards_without_any_device(self):
        fake = types.SimpleNamespace(device=types.SimpleNamespace(id=3))
        with self.assertRaises(E) as cm:
            uptake._check_device_array(fake, 'demand', 0)
        self.assertIn('device 3', str(cm.exception))
        uptake._check_device_array(fake, 'demand', 3)                      # same device: no error

    def test_memory_admission_refuses_before_allocation(self):
        ids, off, res, w = csr([[(0, 0.5), (1, 0.5)], [(1, 1.0)]])
        with self.assertRaises(uptake.AdmissionError) as cm:               # edge bound: 3 edges > max_edges 2
            uptake.RootMap.from_arrays(ids, off, res, w, n_layers=1, n_cells=2, backend='numpy', max_edges=2, max_bytes=BIG)
        self.assertEqual(cm.exception.estimate['n_edges'], 3)
        self.assertIn('scope', cm.exception.estimate)
        self.assertIn('exclusions', cm.exception.estimate)
        with self.assertRaises(uptake.AdmissionError) as cm:               # absurd resource count: scratch > cap, no MemoryError
            uptake.RootMap.from_arrays(ids, off, res, w, n_layers=1, n_cells=10 ** 12, backend='numpy', max_edges=10, max_bytes=BIG)
        self.assertGreater(cm.exception.estimate['total_bytes'], BIG)
        with self.assertRaises(uptake.AdmissionError):
            uptake.RootMap.from_arrays(ids, off, res, w, n_layers=1, n_cells=2, backend='numpy', max_edges=10, max_bytes=100)
        est = uptake.estimate_map_bytes(2, 3, 2, 'numpy')
        self.assertEqual(est['mirror_bytes'], 0)
        self.assertEqual(est['owned_bytes'], sum(est['owned'].values()))
        self.assertGreater(uptake.estimate_map_bytes(2, 3, 2, 'cuda', host_source=True)['staging_bytes'], 0)
        self.assertEqual(uptake.estimate_map_bytes(2, 3, 2, 'cuda', host_source=False)['staging_bytes'], 0)
        m = uptake.RootMap.from_arrays(ids, off, res, w, n_layers=1, n_cells=2, backend='numpy', max_edges=10, max_bytes=BIG)
        with self.assertRaises(uptake.AdmissionError) as cm:
            uptake.propose(m, np.array([1.0, 1.0]), np.array([1.0, 1.0]), max_bytes=m.resident_bytes + 8)
        self.assertEqual(cm.exception.stage, 'uptake.propose')
        plan = uptake.propose(m, np.array([1.0, 1.0]), np.array([1.0, 1.0]))
        with self.assertRaises(uptake.AdmissionError):
            uptake.accept(plan, plan.proposed, max_bytes=m.resident_bytes + plan.resident_bytes + 8)
        self.assertEqual(m.n_edges, 3)
        for bad in (-1, 1.5, True):
            with self.assertRaises(E):
                uptake.admit('x', est, max_edges=bad, max_bytes=BIG)
        # an operation budget may only TIGHTEN the map cap, never raise it
        with self.assertRaises(E) as cm:
            uptake.propose(m, np.array([1.0, 1.0]), np.array([1.0, 1.0]), max_bytes=m.max_bytes + 1)
        self.assertNotIsInstance(cm.exception, uptake.AdmissionError)
        with self.assertRaises(E):
            uptake.accept(plan, plan.proposed, max_bytes=m.max_bytes + 1)
        with self.assertRaises(E):
            m.with_backend('numpy', max_bytes=m.max_bytes + 1)

    def test_conversion_and_construction_preflight_before_any_copy(self):
        P = 20_000
        rows =[[(int(i % 50), 1.0)] for i in range(P)]
        ids, off, res, w = csr(rows)
        m = uptake.RootMap.from_arrays(ids, off, res, w, n_layers=1, n_cells=50, backend='numpy', max_edges=P, max_bytes=BIG)
        arrays_bytes = m.resident_bytes
        self.assertGreater(arrays_bytes, 400_000)

        def peak(fn):
            tracemalloc.start()
            tracemalloc.reset_peak()
            base = tracemalloc.get_traced_memory()[0]
            try:
                fn()
            except uptake.AdmissionError:
                pass
            else:
                self.fail('expected AdmissionError')
            finally:
                p = tracemalloc.get_traced_memory()[1] - base
                tracemalloc.stop()
            return p
        limit = 100_000     # far below the ~0.5 MB the arrays would need
        self.assertLess(peak(lambda: m.with_backend('numpy', max_bytes=0)), limit)
        self.assertLess(peak(lambda: m.with_backend('numpy', max_bytes=m.resident_bytes)), limit)   # source + candidate coexist
        self.assertLess(peak(lambda: uptake.RootMap.from_arrays(ids, off, res, w, n_layers=1, n_cells=50, backend='numpy',
                                                               max_edges=0, max_bytes=BIG)), limit)
        self.assertLess(peak(lambda: uptake.RootMap.from_arrays(ids, off, res, w, n_layers=1, n_cells=50, backend='numpy',
                                                               max_edges=P, max_bytes=0)), limit)
        # a successful conversion needs the candidate arrays on top of the resident source map
        c = m.with_backend('numpy', max_bytes=2 * m.resident_bytes)
        np.testing.assert_array_equal(c.weights, m.weights)
        self.assertFalse(np.shares_memory(c.weights, m.weights))
        self.assertEqual(c.provenance['converted_from'], 'numpy')
        self.assertEqual(c.admission['resident_bytes'], m.resident_bytes)


class TestClosureAndOrdering(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261110)
        self.L, self.N = 3, 7
        self.R = self.L * self.N
        self.P = 240                                     # many overlapping roots on 21 resources
        self.rows = random_rows(self.rng, self.P, self.R)
        self.D = self.rng.uniform(0.0, 5.0, self.P)
        self.D[::7] = 0.0
        self.A = self.rng.uniform(0.0, 20.0, self.R)
        self.A[[2, 9]] = 0.0

    def test_random_overlaps_closure_and_oracle(self):
        m = make_map(self.rows, self.L, self.N)
        plan = uptake.propose(m, self.D, self.A.reshape(self.L, self.N))
        Q, proposed, f = oracle_proposal(self.rows, self.D, self.A)
        np.testing.assert_allclose(plan.request, Q, rtol=1e-13, atol=0)
        np.testing.assert_allclose(plan.proposed, proposed, rtol=1e-13, atol=0)
        accepted = plan.proposed * self.rng.uniform(0.0, 1.0, self.R)
        fb = uptake.accept(plan, accepted)
        np.testing.assert_array_equal(fb.accepted, accepted)
        np.testing.assert_allclose(fb.delivered, oracle_delivered(self.rows, self.D, Q, accepted), rtol=1e-12, atol=0)
        tol = uptake.closure_tolerance_m3(math.fsum(accepted), self.P, m.n_edges, self.R)
        self.assertLessEqual(abs(math.fsum(fb.delivered) - math.fsum(accepted)), tol)
        self.assertLess(tol, 1e-10 * math.fsum(accepted))              # an FP64 tolerance, not an enlarged one
        self.assertTrue(fb.closure['closed'])
        self.assertTrue(np.all(fb.delivered <= self.D * (1 + 1e-9)))
        self.assertTrue(np.all(fb.unmet >= 0.0))
        zero = self.D == 0.0
        np.testing.assert_array_equal(fb.satisfaction[zero], 0.0)
        self.assertTrue(np.all(fb.satisfaction[~zero] <= 1.0))

    def test_reordering_individuals_does_not_change_priority(self):
        perm = self.rng.permutation(self.P)
        rows_p = [self.rows[i] for i in perm]
        ids = np.arange(1, self.P + 1, dtype=np.int64)
        m1 = make_map(self.rows, self.L, self.N, ids=ids)
        m2 = make_map(rows_p, self.L, self.N, ids=ids[perm])
        fb1 = uptake.propose(m1, self.D, self.A).accept_all()
        fb2 = uptake.propose(m2, self.D[perm], self.A).accept_all()
        np.testing.assert_allclose(fb2.delivered, fb1.delivered[perm], rtol=1e-12, atol=0)
        np.testing.assert_allclose(fb2.satisfaction, fb1.satisfaction[perm], rtol=1e-12, atol=0)
        np.testing.assert_allclose(fb2.plan.proposed, fb1.plan.proposed, rtol=1e-12, atol=0)

    def test_plain_python_kernels_are_the_same_algorithm(self):
        '''The authoritative per-edge definitions (executed here in pure Python on a tiny problem) agree with
        the vectorised NumPy reference, including the packet diagnostics.'''
        n = 12
        m = make_map(self.rows[:n], self.L, self.N)
        D = self.D[:n].copy()
        plan = uptake.propose(m, D, self.A)
        packet = np.zeros(len(uptake.PROPOSE_PACKET))
        uptake._k_check_demand(m.counts, D, packet)
        Q = np.zeros(self.R)
        uptake._k_scatter(m.offsets, m.resources, m.weights, D, Q)
        np.testing.assert_allclose(Q, plan.request, rtol=1e-14, atol=0)
        pr, fr = np.empty(self.R), np.empty(self.R)
        uptake._k_proposal(self.A, Q, pr, fr, packet)
        np.testing.assert_allclose(pr, plan.proposed, rtol=1e-14, atol=0)
        np.testing.assert_allclose(fr, plan.fraction, rtol=1e-14, atol=0)
        idx = {k: i for i, k in enumerate(uptake.PROPOSE_PACKET)}
        for k in ('demand_total_m3', 'supply_total_m3', 'request_total_m3', 'proposed_total_m3'):
            self.assertAlmostEqual(packet[idx[k]], plan.summary[k], delta=1e-12 * max(1.0, plan.summary[k]), msg=k)
        self.assertEqual(int(packet[idx['n_resources_limited']]), plan.summary['n_resources_limited'])
        acc = pr * 0.7
        fb = uptake.accept(plan, acc)
        ap = np.zeros(len(uptake.ACCEPT_PACKET))
        sc = np.empty(self.R)
        uptake._k_scale(acc, Q, pr, uptake.acceptance_tolerance_rel(m), sc, ap)
        dl, um, sa = np.empty(n), np.empty(n), np.empty(n)
        uptake._k_gather(m.offsets, m.resources, m.weights, D, sc, uptake.overdelivery_bound_rel(m), dl, um, sa, ap)
        np.testing.assert_allclose(dl, fb.delivered, rtol=1e-14, atol=0)
        np.testing.assert_allclose(um, fb.unmet, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(sa, fb.satisfaction, rtol=1e-14, atol=0)
        aidx = {k: i for i, k in enumerate(uptake.ACCEPT_PACKET)}
        self.assertAlmostEqual(ap[aidx['delivered_total_m3']], fb.summary['delivered_total_m3'], delta=1e-12)
        self.assertEqual(int(ap[aidx['n_plants_zero_demand']]), fb.summary['n_plants_zero_demand'])


class TestNumbaParity(unittest.TestCase):
    def setUp(self):
        st = uptake.backend_status('numba')
        if not st['available']:
            self.skipTest(f'numba backend unavailable: {st["reason"]}')
        self.rng = np.random.default_rng(3)

    def test_cpu_parity_with_reference(self):
        L, N, P = 2, 9, 150
        rows = random_rows(self.rng, P, L * N)
        D = self.rng.uniform(0.0, 3.0, P)
        A = self.rng.uniform(0.0, 4.0, (L, N))
        ref = make_map(rows, L, N, backend='numpy')
        nb = make_map(rows, L, N, backend='numba')
        self.assertEqual(nb.backend, 'numba')
        pr = uptake.propose(ref, D, A)
        pn = uptake.propose(nb, D, A)
        np.testing.assert_allclose(pn.request, pr.request, rtol=1e-13, atol=0)
        np.testing.assert_allclose(pn.proposed, pr.proposed, rtol=1e-13, atol=0)
        acc = pr.proposed * self.rng.uniform(0.0, 1.0, L * N)
        fr, fn = uptake.accept(pr, acc), uptake.accept(pn, acc)
        np.testing.assert_allclose(fn.delivered, fr.delivered, rtol=1e-12, atol=0)
        np.testing.assert_allclose(fn.satisfaction, fr.satisfaction, rtol=1e-12, atol=0)
        self.assertTrue(fn.closure['closed'])
        self.assertEqual(fn.summary['n_plants_unsatisfied'], fr.summary['n_plants_unsatisfied'])
        with self.assertRaises(E):
            uptake.propose(nb, np.zeros(P - 1), A)
        m = make_map([[(0, 0.5), (1, 0.5)]], L=2, N=1, backend='numba')           # the required example, compiled
        fb = uptake.propose(m, np.array([100.0]), np.array([[25.0], [200.0]])).accept_all()
        self.assertEqual(float(fb.delivered[0]), 75.0)
        self.assertEqual(float(fb.satisfaction[0]), 0.75)
        # non-finite totals are refused on the compiled path too
        m2 = make_map([[(0, 1.0)], [(0, 1.0)]], L=1, N=1, backend='numba')
        with self.assertRaises(E):
            uptake.propose(m2, np.array([1e308, 1e308]), np.array([1e308]))
        bad = make_map([[(0, 1.0)], [(1, 1.0)]], L=1, N=2, backend='numba')
        with self.assertRaises(E):
            uptake.propose(bad, np.array([1e308, 1e308]), np.array([1e308, 1e308]))

    def test_measured_feedback_scratch_stays_within_the_estimate(self):
        '''Regression for the under-reported scratch: P = E = 100000 on ONE resource. Measured allocation of the
        compiled accept() (pre-existing map, plan and caller arrays excluded) must fit the admitted estimate.'''
        P = 100_000
        ids = np.arange(P, dtype=np.int64)
        off = np.arange(P + 1, dtype=np.int64)
        m = uptake.RootMap.from_arrays(ids, off, np.zeros(P, np.int64), np.ones(P), n_layers=1, n_cells=1,
                                       backend='numba', max_edges=P, max_bytes=1 << 28)
        D = np.full(P, 1e-3)
        A = np.array([30.0])
        uptake.propose(make_map([[(0, 1.0)]], 1, 1, 'numba'), np.array([1.0]), np.array([1.0])).accept_all()   # JIT warm-up
        plan = uptake.propose(m, D, A)
        acc = plan.proposed.copy()
        est = uptake.estimate_feedback_bytes(P, P, 1, 'numba')
        tracemalloc.start()
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        fb = uptake.accept(plan, acc)
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        self.assertTrue(fb.closure['closed'])
        slack = 65_536      # interpreter-level dict / summary objects; the defect this guards was ~3 MB of array scratch
        self.assertLessEqual(peak, est['total_bytes'] + slack, msg=f'measured {peak} vs estimate {est["total_bytes"]}')
        self.assertGreaterEqual(est['total_bytes'], 3 * 8 * P)     # at least delivered + unmet + satisfaction are counted
        pest = uptake.estimate_plan_bytes(P, P, 1, 'numba')
        tracemalloc.start()
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        plan2 = uptake.propose(m, D, A)
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        self.assertLessEqual(peak, pest['total_bytes'] + slack, msg=f'measured {peak} vs estimate {pest["total_bytes"]}')
        self.assertEqual(float(plan2.request[0]), float(plan.request[0]))


class TestCudaParity(unittest.TestCase):
    def setUp(self):
        st = uptake.backend_status('cuda')
        if not st['available']:
            self.skipTest(f'cuda backend unavailable: {st["reason"]}')
        import cupy
        self.cp = cupy
        self.rng = np.random.default_rng(5)

    def test_gpu_parity_closure_and_no_implicit_transfers(self):
        cp = self.cp
        L, N, P = 3, 11, 400
        rows = random_rows(self.rng, P, L * N)
        D = self.rng.uniform(0.0, 3.0, P)
        A = self.rng.uniform(0.0, 4.0, (L, N))
        ref = make_map(rows, L, N, backend='numpy')
        gm = ref.with_backend('cuda')                                       # explicit host -> device load
        self.assertEqual(gm.backend, 'cuda')
        self.assertEqual(gm.device_id, int(cp.cuda.Device().id))            # bound to the CURRENT device, no switching
        self.assertTrue(isinstance(gm.weights, cp.ndarray))
        with self.assertRaises(E):                                           # host arrays are not transferred implicitly
            uptake.propose(gm, D, A)
        pr = uptake.propose(ref, D, A)
        pg = uptake.propose(gm, cp.asarray(D), cp.asarray(A))
        self.assertTrue(isinstance(pg.proposed, cp.ndarray))
        self.assertEqual(pg.summary['device_id'], gm.device_id)
        np.testing.assert_allclose(pg.request.get(), pr.request, rtol=1e-12, atol=0)
        np.testing.assert_allclose(pg.proposed.get(), pr.proposed, rtol=1e-12, atol=0)
        acc = pr.proposed * self.rng.uniform(0.0, 1.0, L * N)
        fr = uptake.accept(pr, acc)
        fg = uptake.accept(pg, cp.asarray(acc))
        np.testing.assert_allclose(fg.delivered.get(), fr.delivered, rtol=1e-11, atol=0)
        np.testing.assert_allclose(fg.satisfaction.get(), fr.satisfaction, rtol=1e-11, atol=0)
        np.testing.assert_array_equal(fg.accepted.get(), acc)                # the native debit is not modified on device either
        self.assertTrue(fg.closure['closed'])
        with self.assertRaises(E):
            uptake.accept(pg, acc)                                           # host accepted on a cuda plan
        excess = pg.proposed * 1.01                                          # genuinely above the PROPOSAL (1 %)
        self.assertGreater(float(excess.sum() - pg.proposed.sum()), 0.0)
        with self.assertRaises(E):
            uptake.accept(pg, excess)
        # device-resident map construction from device arrays (validation by scalar reductions)
        a = ref.host_arrays()
        gm2 = uptake.RootMap.from_arrays(cp.asarray(a['ids']), cp.asarray(a['offsets']), cp.asarray(a['resources']),
                                         cp.asarray(a['weights']), n_layers=L, n_cells=N, backend='cuda',
                                         max_edges=ref.max_edges, max_bytes=ref.max_bytes)
        self.assertEqual(gm2.n_edges, ref.n_edges)
        gm3 = gm2.with_backend('cuda')                                       # device -> device copy on the bound device
        self.assertEqual(gm3.device_id, gm2.device_id)
        self.assertFalse(bool(cp.shares_memory(gm3.weights, gm2.weights)))
        bad_w = cp.asarray(a['weights'])
        bad_w[0] = -1.0
        with self.assertRaises(E):
            uptake.RootMap.from_arrays(cp.asarray(a['ids']), cp.asarray(a['offsets']), cp.asarray(a['resources']), bad_w,
                                       n_layers=L, n_cells=N, backend='cuda', max_edges=ref.max_edges, max_bytes=ref.max_bytes)
        bad_off = cp.asarray(np.array([0, INT64_MAX, INT64_MIN + 1, -1, 0], dtype=np.int64))
        with self.assertRaises(E):                                           # overflowing offsets on device: refused too
            uptake.RootMap.from_arrays(cp.arange(4, dtype=cp.int64), bad_off, cp.zeros(0, cp.int64), cp.zeros(0),
                                       n_layers=1, n_cells=4, backend='cuda', max_edges=10, max_bytes=BIG)
        m = make_map([[(0, 0.5), (1, 0.5)]], L=2, N=1, backend='cuda')
        fb = uptake.propose(m, cp.asarray([100.0]), cp.asarray([[25.0], [200.0]])).accept_all()
        self.assertEqual(float(fb.delivered[0]), 75.0)
        mo = make_map([[(0, 1.0)], [(1, 1.0)]], L=1, N=2, backend='cuda')
        with self.assertRaises(E):                                           # non-finite totals refused on device
            uptake.propose(mo, cp.asarray([1e308, 1e308]), cp.asarray([1e308, 1e308]))

    def test_device_guards_do_not_touch_other_devices(self):
        cp = self.cp
        cur = int(cp.cuda.Device().id)
        m = make_map([[(0, 1.0)]], L=1, N=1, backend='cuda')
        with self.assertRaises(E):                                           # a map bound to another device id: refused, no switch
            uptake._require_current_device(cur + 1, 'test')
        self.assertEqual(int(cp.cuda.Device().id), cur)                      # the current device was not changed
        fake = types.SimpleNamespace(device=types.SimpleNamespace(id=cur + 1))
        with self.assertRaises(E):
            uptake._check_device_array(fake, 'demand', m.device_id)
        uptake._require_current_device(m.device_id, 'test')


if __name__ == '__main__':
    unittest.main()
