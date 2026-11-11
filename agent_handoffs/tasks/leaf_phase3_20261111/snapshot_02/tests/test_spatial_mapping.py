'''Small geometry-oracle unit tests for benchmarks/spatial_mapping.py (not performance tests).'''
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import spatial_mapping as sm  # noqa: E402

BIG = 10 ** 9


def inputs_from(points, canopy_r, root_r=None):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = pts.shape[0]
    rc = np.full(n, canopy_r, np.float64) if np.isscalar(canopy_r) else np.asarray(canopy_r, np.float64)
    if root_r is None:
        rr = rc.copy()
    else:
        rr = np.full(n, root_r, np.float64) if np.isscalar(root_r) else np.asarray(root_r, np.float64)
    return {'x': pts[:, 0].copy(), 'y': pts[:, 1].copy(), 'canopy_radius': rc,
            'root_radius': rr, 'ids': np.arange(n, dtype=np.int64)}


def oracle_cells(px, py, r, grid):
    '''Independent brute-force positive-overlap oracle over every cell (test-only).'''
    dx = grid.dx
    res = []
    for j in range(grid.ny):
        for i in range(grid.nx):
            cx = min(max(px, i * dx), (i + 1) * dx)
            cy = min(max(py, j * dx), (j + 1) * dx)
            if (px - cx) ** 2 + (py - cy) ** 2 < r * r:
                res.append(j * grid.nx + i)
    return res


def build(inputs, grid, layers=2, max_edges=10 ** 6, max_bytes=BIG):
    return sm.build_root_graph_cpu(inputs, grid, layers, max_edges, max_bytes)


def row(graph, p):
    sl = slice(int(graph.offsets[p]), int(graph.offsets[p + 1]))
    return graph.cells[sl], graph.weights[sl]


class RootPredicateTest(unittest.TestCase):
    def test_subcell_root_at_cell_corner_maps_four_cells(self):
        grid = sm.Grid(2, 2, 1.0)
        g = build(inputs_from([(1.0, 1.0)], 0.1), grid)
        cells, w = row(g, 0)
        self.assertEqual(sorted(cells.tolist()), [0, 1, 2, 3])
        np.testing.assert_allclose(w, 0.25)

    def test_subcell_root_at_cell_centre_maps_one_cell(self):
        grid = sm.Grid(3, 3, 1.0)
        g = build(inputs_from([(1.5, 1.5)], 0.05), grid)
        cells, w = row(g, 0)
        self.assertEqual(cells.tolist(), [4])
        self.assertEqual(w.tolist(), [1.0])

    def test_tangent_contact_is_not_an_edge(self):
        grid = sm.Grid(3, 3, 1.0)
        g = build(inputs_from([(1.5, 1.5)], 0.5), grid)
        cells, _ = row(g, 0)
        self.assertEqual(cells.tolist(), [4])
        g2 = build(inputs_from([(1.5, 1.5)], 0.5000001), grid)
        self.assertEqual(sorted(row(g2, 0)[0].tolist()), [1, 3, 4, 5, 7])

    def test_boundary_clipping_renormalizes(self):
        grid = sm.Grid(2, 2, 1.0)
        g = build(inputs_from([(0.0, 0.5), (2.0, 2.0), (0.5, 1.0)], 0.3), grid)
        c0, w0 = row(g, 0)
        self.assertEqual(c0.tolist(), [0])
        self.assertEqual(w0.tolist(), [1.0])
        c1, w1 = row(g, 1)
        self.assertEqual(c1.tolist(), [3])
        c2, w2 = row(g, 2)
        self.assertEqual(sorted(c2.tolist()), [0, 2])
        np.testing.assert_allclose(w2, 0.5)

    def test_csr_matches_oracle_and_is_normalized(self):
        rng = np.random.default_rng(7)
        grid = sm.Grid(5, 4, 0.5)
        n = 60
        pts = np.column_stack([rng.uniform(0, grid.width, n), rng.uniform(0, grid.height, n)])
        rr = rng.uniform(0.02, 1.2, n)
        inp = inputs_from(pts, rr)
        g = build(inp, grid, layers=3)
        self.assertEqual(g.offsets.dtype, np.int64)
        self.assertEqual(g.cells.dtype, np.int32)
        self.assertEqual(g.weights.dtype, np.float64)
        self.assertEqual(int(g.offsets[-1]), g.cells.shape[0])
        self.assertTrue(np.all(np.diff(g.offsets) >= 1))
        for p in range(n):
            cells, w = row(g, p)
            self.assertEqual(cells.tolist(), oracle_cells(pts[p, 0], pts[p, 1], rr[p], grid))
            self.assertAlmostEqual(float(w.sum()), 1.0, places=12)
            self.assertEqual(len(set(cells.tolist())), len(cells))

    def test_root_predicate_independent_of_canopy_radius(self):
        grid = sm.Grid(2, 2, 1.0)
        a = build(inputs_from([(1.0, 1.0)], 0.01, 0.2), grid)
        b = build(inputs_from([(1.0, 1.0)], 5.0, 0.2), grid)
        self.assertEqual(row(a, 0)[0].tolist(), row(b, 0)[0].tolist())


class ReductionTest(unittest.TestCase):
    def test_plants_sharing_one_cell(self):
        grid = sm.Grid(2, 2, 1.0)
        g = build(inputs_from([(0.3, 0.3), (0.5, 0.5), (0.7, 0.6)], 0.1), grid, layers=3)
        out = sm.layered_reduction_cpu(g)
        self.assertEqual(out.shape, (3, 4))
        np.testing.assert_allclose(out[:, 0], 1.0)
        self.assertAlmostEqual(float(out.sum()), 3.0, places=12)

    def test_layered_sum_equals_plant_count(self):
        rng = np.random.default_rng(11)
        grid = sm.Grid(6, 6, 1.0)
        for layers in (1, 2, 3, 5):
            n = 40
            pts = np.column_stack([rng.uniform(0, 6, n), rng.uniform(0, 6, n)])
            g = build(inputs_from(pts, rng.uniform(0.05, 1.5, n)), grid, layers=layers)
            out = sm.layered_reduction_cpu(g)
            audit = sm.audit_reduction(out, g.counts, layers)
            self.assertTrue(audit['passed'], audit)
            self.assertAlmostEqual(float(out.sum()), float(n), places=10)
            np.testing.assert_allclose(out.sum(axis=1), n / layers, rtol=1e-10)


class AuditAndStatusRegressionTest(unittest.TestCase):
    def _case(self, **over):
        return dict(GenerationAndAdmissionTest.CASE, **over)

    def test_audit_fails_on_zero_edge_plant(self):
        counts = np.array([1, 0], dtype=np.int64)
        out = np.zeros((2, 4))
        out[:, 0] = 0.5
        audit = sm.audit_reduction(out, counts, 2)
        self.assertFalse(audit['passed'])
        self.assertEqual(audit['zero_edge_plants'], 1)
        self.assertEqual(audit['expected_demand'], 2)
        good = sm.audit_reduction(np.array([[0.5, 0.0], [0.5, 0.0]]), np.array([1], dtype=np.int64), 2)
        self.assertTrue(good['passed'], good)
        bad = sm.audit_reduction(np.array([[np.nan, 0.0], [0.5, 0.0]]), np.array([1], dtype=np.int64), 2)
        self.assertFalse(bad['passed'])
        wrong_shape = sm.audit_reduction(np.array([[1.0, 0.0]]), np.array([1], dtype=np.int64), 2)
        self.assertFalse(wrong_shape['passed'])

    def test_rejection_happens_before_host_generation(self):
        with mock.patch.object(sm, 'generate_inputs') as gen:
            rec = sm.run_case(self._case(max_bytes=16), 'cpu', 1)
        gen.assert_not_called()
        self.assertEqual(rec['status'], 'rejected')
        self.assertEqual(rec['n_plants'], 40)
        self.assertFalse(rec['admission']['accepted'])
        self.assertIn('case_wall_seconds', rec)

    def test_gpu_rejection_before_transfer(self):
        grid = sm.Grid(2, 2, 1.0)
        inp = inputs_from([(1.0, 1.0)], 0.1)
        with mock.patch.object(sm, 'gpu_status', return_value={'available': True}), \
                mock.patch.object(sm, 'to_device') as h2d, \
                mock.patch.object(sm, '_gpu_module') as mod:
            rec = sm.run_gpu(inp, grid, 2, 10 ** 6, 16, 1)
        h2d.assert_not_called()
        mod.assert_not_called()
        self.assertEqual(rec['status'], 'rejected')

    def test_partial_status_when_gpu_unavailable(self):
        fake = {'backend': 'gpu', 'status': 'unavailable', 'reason': 'mocked'}
        with mock.patch.object(sm, 'run_gpu', side_effect=lambda *a, **k: dict(fake)):
            both = sm.run_case(self._case(), 'both', 1)
            only = sm.run_case(self._case(), 'gpu', 1)
        self.assertEqual(both['status'], 'partial')
        self.assertEqual(both['cpu']['status'], 'ok')
        self.assertEqual(only['status'], 'unavailable')
        self.assertNotIn('cpu', only)
        with self.assertRaises(ValueError):
            sm.run_case(self._case(), 'tpu', 1)
        with self.assertRaises(ValueError):
            sm.run_case(self._case(), 'cpu', 0)

    def test_clump_parameters_bounded(self):
        k, sigma = sm.clump_parameters(100000, sm.Grid(10, 20, 1.0))
        self.assertEqual(k, 8)
        self.assertAlmostEqual(sigma, 0.2)
        self.assertEqual(sm.clump_parameters(40, sm.Grid(8, 6, 1.0))[0], 1)
        rec = sm.run_case(self._case(), 'cpu', 1)
        self.assertEqual(rec['generation']['cluster_count'], 1)


class CanopyTest(unittest.TestCase):
    def test_small_plant_in_coarse_cell_is_reported_missed(self):
        grid = sm.Grid(1, 1, 10.0)
        inp = inputs_from([(3.3, 3.3)], 0.1)
        res = sm.canopy_cover_cpu(inp, grid)
        self.assertEqual(res.missed_plants, 1)
        self.assertEqual(res.cover.tolist(), [0.0])
        g = build(inp, grid)
        self.assertEqual(row(g, 0)[0].tolist(), [0])

    def test_sampled_cover_is_approximate_and_capped(self):
        grid = sm.Grid(1, 1, 10.0)
        res = sm.canopy_cover_cpu(inputs_from([(5.0, 5.0)], 5.0), grid)
        self.assertEqual(res.missed_plants, 0)
        self.assertGreater(res.cover[0], 0.7)
        self.assertLess(res.cover[0], 0.87)
        res2 = sm.canopy_cover_cpu(inputs_from([(5.0, 5.0), (5.0, 5.0)], 50.0), grid)
        self.assertEqual(res2.cover.tolist(), [1.0])
        self.assertEqual(res2.capped_cells, 1)
        self.assertAlmostEqual(res2.uncapped_sum, 2.0)


class GenerationAndAdmissionTest(unittest.TestCase):
    CASE = {'name': 't', 'nx': 8, 'ny': 6, 'dx': 1.0, 'grass_count': 30, 'shrub_count': 10,
            'grass_diameter': 0.3, 'shrub_diameter': 1.0, 'root_multiplier': 1.5,
            'pattern': 'clumped', 'seed': 5, 'layers': 2, 'max_edges': 100000, 'max_bytes': BIG}

    def test_deterministic_generation(self):
        a = sm.generate_inputs(self.CASE)
        b = sm.generate_inputs(self.CASE)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
        self.assertEqual(sm.validate_inputs(a, sm.Grid(8, 6, 1.0)), 40)
        np.testing.assert_allclose(a['root_radius'], a['canopy_radius'] * 1.5)

    def test_invalid_inputs_rejected(self):
        grid = sm.Grid(2, 2, 1.0)
        with self.assertRaises(ValueError):
            sm.validate_inputs(inputs_from([(0.5, 0.5)], -0.1), grid)
        with self.assertRaises(ValueError):
            sm.validate_inputs(inputs_from([(2.5, 0.5)], 0.1), grid)
        with self.assertRaises(ValueError):
            sm.validate_inputs(inputs_from([(np.nan, 0.5)], 0.1), grid)
        with self.assertRaises(ValueError):
            sm.Grid(2, 2, 0.0)
        with self.assertRaises(ValueError):
            sm.validate_case(dict(self.CASE, pattern='random'))

    def test_capacity_refusal_before_allocation(self):
        grid = sm.Grid(2, 2, 1.0)
        inp = inputs_from([(1.0, 1.0)], 0.1)
        with self.assertRaises(sm.AdmissionRejected) as cm:
            build(inp, grid, max_edges=3)
        self.assertEqual(cm.exception.stage, 'post-count')
        self.assertEqual(cm.exception.estimate['n_edges'], 4)
        with self.assertRaises(sm.AdmissionRejected) as cm2:
            build(inp, grid, max_bytes=16)
        self.assertEqual(cm2.exception.stage, 'pre-count')
        rec = sm.run_case(dict(self.CASE, max_edges=1), 'cpu', 1)
        self.assertEqual(rec['status'], 'rejected')
        self.assertFalse(rec['cpu']['admission']['accepted'])

    def test_run_case_cpu_small(self):
        rec = sm.run_case(self.CASE, 'cpu', 2)
        self.assertEqual(rec['status'], 'ok', rec.get('traceback'))
        self.assertTrue(rec['cpu']['audit']['passed'])
        self.assertEqual(len(rec['cpu']['reduction']['repeats_seconds']), 2)
        self.assertNotIn('arrays', rec['cpu'])


@unittest.skipUnless(sm.gpu_status()['available'], 'GPU unavailable: %s' % sm.gpu_status().get('reason'))
class GpuParityTest(unittest.TestCase):
    def test_cpu_gpu_parity_within_tolerance(self):
        rng = np.random.default_rng(3)
        grid = sm.Grid(6, 5, 1.0)
        n = 200
        pts = np.column_stack([rng.uniform(0, 6, n), rng.uniform(0, 5, n)])
        inp = inputs_from(pts, rng.uniform(0.05, 1.2, n))
        cpu = sm.run_cpu(inp, grid, 3, 10 ** 6, BIG, 1)
        gpu = sm.run_gpu(inp, grid, 3, 10 ** 6, BIG, 1)
        self.assertEqual(cpu['status'], 'ok')
        self.assertEqual(gpu['status'], 'ok', gpu.get('traceback'))
        parity = sm.compare_backends(cpu['arrays'], gpu['arrays'])
        self.assertTrue(parity['passed'], parity)


if __name__ == '__main__':
    unittest.main()
