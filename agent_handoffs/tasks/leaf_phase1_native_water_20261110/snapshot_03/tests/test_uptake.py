'''LEAF Phase 1 part A: sparse uptake allocator tests (unittest, NumPy only).

Run from the candidate root:  python -m unittest tests.test_uptake -v
numba / cuda parity tests SKIP only when leaf.uptake.backend_status reports the backend unavailable; a
real implementation failure on an available backend fails the test.
'''
from __future__ import annotations

import math
import unittest

import numpy as np

from leaf import uptake

E = uptake.UptakeError
BIG = 1 << 26


def csr(rows, ids=None):
    '''rows: per plant, list of (resource, weight) -> (ids, offsets, resources, weights) arrays.'''
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
        residual = A.reshape(-1) - fb.accepted
        np.testing.assert_array_equal(residual, [0.0, 150.0])    # deeper remaining water is NOT used
        self.assertTrue(fb.closure['closed'])
        self.assertEqual(fb.summary['n_rounding_clipped'], 0)
        self.assertEqual(plan.summary['n_resources_limited'], 1)

    def test_multi_plant_sharing_is_proportional_to_local_requests(self):
        # two plants on one resource: requests 100 and 300, supply 200 -> same fraction 0.5 for both
        m = make_map([[(0, 1.0)], [(0, 1.0)]], L=1, N=1)
        plan = uptake.propose(m, np.array([100.0, 300.0]), np.array([200.0]))
        fb = plan.accept_all()
        np.testing.assert_allclose(fb.delivered, [50.0, 150.0], rtol=1e-15)        # not equal litres (100/100)
        np.testing.assert_allclose(fb.satisfaction, [0.5, 0.5], rtol=1e-15)        # the SAME fraction
        # size / order independent: swapping rows and ids gives the same per-id answer
        m2 = make_map([[(0, 1.0)], [(0, 1.0)]], L=1, N=1, ids=[2, 1])
        fb2 = uptake.propose(m2, np.array([300.0, 100.0]), np.array([200.0])).accept_all()
        np.testing.assert_allclose(fb2.delivered, [150.0, 50.0], rtol=1e-15)
        self.assertTrue(fb.closure['closed'] and fb2.closure['closed'])

    def test_non_separable_weights_multiple_cells_and_layers(self):
        # plant 1 roots mostly shallow in cells 0/1, plant 2 deep in cell 1 only; plant 3 shares both layers of cell 1
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
        # cell 1 shallow: requests 4 + 20 = 24 against A = 10 -> both plants get fraction 10/24 there
        self.assertAlmostEqual(float(plan.fraction[1]), 10.0 / 24.0, places=15)
        self.assertTrue(fb.closure['closed'])

    def test_zero_demand_zero_supply_and_inactive_roots(self):
        m = make_map([[(0, 1.0)], [], [(0, 0.5), (1, 0.5)]], L=1, N=2)
        self.assertEqual(m.n_inactive_plants, 1)
        plan = uptake.propose(m, np.array([0.0, 0.0, 10.0]), np.array([0.0, 0.0]))   # dry everywhere
        self.assertEqual(float(plan.request.sum()), 10.0)
        np.testing.assert_array_equal(plan.proposed, [0.0, 0.0])
        fb = plan.accept_all()
        np.testing.assert_array_equal(fb.delivered, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(fb.satisfaction, [0.0, 0.0, 0.0])     # zero demand -> S = 0, not 1
        np.testing.assert_array_equal(fb.unmet, [0.0, 0.0, 10.0])
        self.assertEqual(fb.summary['n_plants_zero_demand'], 2)
        with self.assertRaises(E):                                           # positive demand on an empty root row
            uptake.propose(m, np.array([0.0, 1.0, 0.0]), np.array([5.0, 5.0]))
        # an empty population is a valid (trivial) problem
        empty = uptake.RootMap.from_arrays(np.zeros(0, np.int64), np.zeros(1, np.int64), np.zeros(0, np.int64),
                                           np.zeros(0), n_layers=2, n_cells=3, backend='numpy', max_edges=10, max_bytes=BIG)
        plan = uptake.propose(empty, np.zeros(0), np.ones((2, 3)))
        np.testing.assert_array_equal(plan.proposed, np.zeros(6))
        fb = plan.accept_all()
        self.assertEqual(fb.delivered.shape, (0,))
        self.assertTrue(fb.closure['closed'])

    def test_masked_cell_gets_nothing_and_weight_is_not_renormalised(self):
        # plant with .5 in cell 0 (unavailable, A = 0) and .5 in cell 1 (plenty): only 50 % is delivered
        m = make_map([[(0, 0.5), (1, 0.5)]], L=1, N=2)
        plan = uptake.propose(m, np.array([100.0]), np.array([0.0, 1000.0]))
        np.testing.assert_array_equal(plan.proposed, [0.0, 50.0])         # the dry cell's request is not moved
        fb = plan.accept_all()
        self.assertEqual(float(fb.delivered[0]), 50.0)
        self.assertEqual(float(fb.unmet[0]), 50.0)
        self.assertEqual(float(fb.satisfaction[0]), 0.5)
        self.assertEqual(float(1000.0 - fb.accepted[1]), 950.0)            # wet cell residual untouched
        np.testing.assert_array_equal(m.weights, [0.5, 0.5])               # map weights unchanged

    def test_partial_acceptance_shortfall_and_tolerance(self):
        m = make_map([[(0, 0.5), (1, 0.5)], [(1, 1.0)]], L=1, N=2)
        D = np.array([100.0, 100.0])
        plan = uptake.propose(m, D, np.array([40.0, 90.0]))
        np.testing.assert_array_equal(plan.proposed, [40.0, 90.0])         # Q = [50, 150]
        half = plan.proposed * 0.5
        fb = uptake.accept(plan, half)                                        # SYRUP delivers half of the proposal
        np.testing.assert_allclose(fb.delivered, oracle_delivered([[(0, .5), (1, .5)], [(1, 1.)]], D, plan.request, half), rtol=1e-14)
        self.assertAlmostEqual(fb.summary['shortfall_total_m3'], 65.0, places=12)
        self.assertTrue(fb.closure['closed'])
        # material excess over the proposal is refused; nothing in the plan changes
        before = plan.proposed.copy()
        for bad in (plan.proposed * 1.001, np.array([40.0, 90.0 + 1e-6]), np.array([-1.0, 0.0]),
                    np.array([np.nan, 0.0]), np.array([40.0, np.inf]), np.array([[40.0], [90.0]]), np.array([40.0]),
                    np.array([True, False]), 'accepted'):
            with self.assertRaises(E, msg=repr(bad)):
                uptake.accept(plan, bad)
        np.testing.assert_array_equal(plan.proposed, before)
        # rounding-level excess (one ulp) is clipped and COUNTED, never silent
        tiny = plan.proposed * (1.0 + uptake.FP64_EPS)
        fb2 = uptake.accept(plan, tiny)
        self.assertEqual(fb2.summary['n_rounding_clipped'], 2)
        self.assertGreater(fb2.summary['max_rounding_excess_m3'], 0.0)
        np.testing.assert_array_equal(fb2.accepted, plan.proposed)
        # water on a resource nobody requested is refused (Q == 0 -> proposed 0)
        m0 = make_map([[(0, 1.0)]], L=1, N=2)
        plan0 = uptake.propose(m0, np.array([10.0]), np.array([5.0, 5.0]))
        with self.assertRaises(E):
            uptake.accept(plan0, np.array([5.0, 1.0]))
        # foreign objects are refused
        with self.assertRaises(E):
            uptake.accept(plan.proposed, plan.proposed)
        with self.assertRaises(E):
            uptake.propose(m, D, np.array([1.0, 2.0, 3.0]))
        with self.assertRaises(E):
            uptake.propose('map', D, np.array([1.0, 2.0]))


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
            dict(offsets=np.array([0, 3, 3])),                            # valid shape, but row 0 then sums to 2 -> refused
            dict(offsets=np.array([0, 2.0, 3.0])),                        # float offsets
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
        np.testing.assert_array_equal(ok.weights, [0.5, 0.5, 1.0])
        np.testing.assert_array_equal(ok.resources, [0, 1, 1])
        with self.assertRaises(ValueError):
            ok.weights[0] = 0.1
        with self.assertRaises(AttributeError):
            ok.weights = np.zeros(3)
        self.assertEqual(ok.max_edges_per_resource, 2)
        self.assertEqual(ok.max_edges_per_plant, 2)
        self.assertIn('scope', ok.describe()['admission'])
        # row sums within WEIGHT_SUM_TOL pass, beyond it fail (no silent normalisation either way)
        make_map([[(0, 0.5), (1, 0.5 + 0.5e-9)]], L=1, N=2)
        with self.assertRaises(E):
            make_map([[(0, 0.5), (1, 0.5 + 5e-9)]], L=1, N=2)
        # host backends refuse device-like arrays (no implicit transfer)
        class FakeDevice:
            __cuda_array_interface__ = {}
            shape, ndim, dtype = (2,), 1, np.dtype(np.float64)
        with self.assertRaises(E):
            uptake.propose(ok, FakeDevice(), np.array([1.0, 1.0]))

    def test_memory_admission_refuses_before_allocation(self):
        ids, off, res, w = csr([[(0, 0.5), (1, 0.5)], [(1, 1.0)]])
        # edge bound: 3 edges > max_edges 2
        with self.assertRaises(uptake.AdmissionError) as cm:
            uptake.RootMap.from_arrays(ids, off, res, w, n_layers=1, n_cells=2, backend='numpy', max_edges=2, max_bytes=BIG)
        self.assertEqual(cm.exception.estimate['n_edges'], 3)
        self.assertIn('scope', cm.exception.estimate)
        # byte bound: an absurd resource count would need ~ 8e12 bytes of validation scratch -> refused, no MemoryError
        with self.assertRaises(uptake.AdmissionError) as cm:
            uptake.RootMap.from_arrays(ids, off, res, w, n_layers=1, n_cells=10 ** 12, backend='numpy', max_edges=10, max_bytes=BIG)
        self.assertGreater(cm.exception.estimate['total_bytes'], BIG)
        with self.assertRaises(uptake.AdmissionError):
            uptake.RootMap.from_arrays(ids, off, res, w, n_layers=1, n_cells=2, backend='numpy', max_edges=10, max_bytes=100)
        est = uptake.estimate_map_bytes(2, 3, 2, 'numpy')
        self.assertEqual(est['owned_bytes'], 8 * 2 + 8 * 3 + 8 * 2 + 8 * 3 + 8 * 3)
        self.assertEqual(est['mirror_bytes'], 0)
        self.assertGreater(uptake.estimate_map_bytes(2, 3, 2, 'cuda', host_source=True)['staging_bytes'], 0)
        # operations: a tighter explicit budget refuses the plan / feedback before allocation; the map survives
        m = uptake.RootMap.from_arrays(ids, off, res, w, n_layers=1, n_cells=2, backend='numpy', max_edges=10, max_bytes=BIG)
        with self.assertRaises(uptake.AdmissionError) as cm:
            uptake.propose(m, np.array([1.0, 1.0]), np.array([1.0, 1.0]), max_bytes=m.resident_bytes + 8)
        self.assertEqual(cm.exception.stage, 'uptake.propose')
        plan = uptake.propose(m, np.array([1.0, 1.0]), np.array([1.0, 1.0]))
        with self.assertRaises(uptake.AdmissionError):
            uptake.accept(plan, plan.proposed, max_bytes=m.resident_bytes + plan.summary['admission']['owned_bytes'] + 8)
        self.assertEqual(m.n_edges, 3)
        for bad in (-1, 1.5, True):
            with self.assertRaises(E):
                uptake.admit('x', est, max_edges=bad, max_bytes=BIG)


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
        accepted = proposed * self.rng.uniform(0.0, 1.0, self.R)
        fb = uptake.accept(plan, accepted)
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
        the vectorised NumPy reference.'''
        rows = self.rows[:12]
        m = make_map(rows, self.L, self.N)
        D = self.D[:12]
        plan = uptake.propose(m, D, self.A)
        Q = np.empty(self.R)
        uptake._k_scatter(m.offsets, m.resources, m.weights, D, Q)
        np.testing.assert_allclose(Q, plan.request, rtol=1e-14, atol=0)
        pr, fr = np.empty(self.R), np.empty(self.R)
        uptake._k_proposal(self.A, Q, pr, fr)
        np.testing.assert_allclose(pr, plan.proposed, rtol=1e-14, atol=0)
        acc = pr * 0.7
        fb = uptake.accept(plan, acc)
        sc = np.empty(self.R)
        uptake._k_scale(acc, Q, sc)
        dl = np.empty(12)
        uptake._k_gather(m.offsets, m.resources, m.weights, D, sc, dl)
        np.testing.assert_allclose(dl, fb.delivered, rtol=1e-14, atol=0)


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
        with self.assertRaises(E):
            uptake.propose(nb, np.array([0.0] * (P - 1)), A)
        # the required example on the compiled path
        m = make_map([[(0, 0.5), (1, 0.5)]], L=2, N=1, backend='numba')
        fb = uptake.propose(m, np.array([100.0]), np.array([[25.0], [200.0]])).accept_all()
        self.assertEqual(float(fb.delivered[0]), 75.0)
        self.assertEqual(float(fb.satisfaction[0]), 0.75)


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
        self.assertTrue(isinstance(gm.weights, cp.ndarray))
        with self.assertRaises(E):                                           # host arrays are not transferred implicitly
            uptake.propose(gm, D, A)
        pr = uptake.propose(ref, D, A)
        pg = uptake.propose(gm, cp.asarray(D), cp.asarray(A))
        self.assertTrue(isinstance(pg.proposed, cp.ndarray))
        np.testing.assert_allclose(pg.request.get(), pr.request, rtol=1e-12, atol=0)
        np.testing.assert_allclose(pg.proposed.get(), pr.proposed, rtol=1e-12, atol=0)
        acc = pr.proposed * self.rng.uniform(0.0, 1.0, L * N)
        fr = uptake.accept(pr, acc)
        fg = uptake.accept(pg, cp.asarray(acc))
        np.testing.assert_allclose(fg.delivered.get(), fr.delivered, rtol=1e-11, atol=0)
        np.testing.assert_allclose(fg.satisfaction.get(), fr.satisfaction, rtol=1e-11, atol=0)
        self.assertTrue(fg.closure['closed'])
        with self.assertRaises(E):
            uptake.accept(pg, acc)                                           # host accepted on a cuda plan
        with self.assertRaises(E):
            uptake.accept(pg, cp.asarray(acc) * 1.01)
        # device-resident map construction from device arrays (validation by scalar reductions)
        a = ref.host_arrays()
        gm2 = uptake.RootMap.from_arrays(cp.asarray(a['ids']), cp.asarray(a['offsets']), cp.asarray(a['resources']),
                                         cp.asarray(a['weights']), n_layers=L, n_cells=N, backend='cuda',
                                         max_edges=ref.max_edges, max_bytes=ref.max_bytes)
        self.assertEqual(gm2.n_edges, ref.n_edges)
        bad_w = cp.asarray(a['weights'])
        bad_w[0] = -1.0
        with self.assertRaises(E):
            uptake.RootMap.from_arrays(cp.asarray(a['ids']), cp.asarray(a['offsets']), cp.asarray(a['resources']), bad_w,
                                       n_layers=L, n_cells=N, backend='cuda', max_edges=ref.max_edges, max_bytes=ref.max_bytes)
        m = make_map([[(0, 0.5), (1, 0.5)]], L=2, N=1, backend='cuda')
        fb = uptake.propose(m, cp.asarray([100.0]), cp.asarray([[25.0], [200.0]])).accept_all()
        self.assertEqual(float(fb.delivered[0]), 75.0)


if __name__ == '__main__':
    unittest.main()
