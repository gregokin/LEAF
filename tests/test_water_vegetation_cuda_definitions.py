'''leaf.water_vegetation CUDA test DEFINITIONS (AUTHORED, NOT RUN by the author; Root executes on hardware).

Skipped unless LEAF_RELEASE_GPU=1. Run (Root, GPU0 after the free-device check, existing MAPLE .venv-cupy):
    LEAF_RELEASE_GPU=1 LEAF_GPU_DEVICE=0 python -B -m unittest tests.test_water_vegetation_cuda_definitions -v
Definitions: (1) the CUDA reduction is BIT-IDENTICAL to the NumPy reference for a randomised multi-species / overlapping / dead+live / masked (active-first) case at 1 and 3 layers;
(2) an explicit device and an explicit separate DEVICE pool (`max_device_bytes`, `device_reserve_bytes`, `available_device_bytes_fn`) are mandatory, there is no fallback; (3) the CURRENT
device must equal the requested one (a mismatch refuses before any device allocation; no implicit switching; needs >= 2 devices, else skipped); (4) a device bound below
`estimate_device_bytes(...)['device_bytes']` refuses BEFORE any upload, and the host pool is gated first; (5) the canonical outputs are HOST arrays returned by ONE contiguous (3, N) D2H call of
exactly `d2h_bytes`, sealed read-only on immutable bytes; they are NOT resident GPU maps; (6) the identity records backend=cuda, the device id and the declared transfers, and the content
hash equals the NumPy projection's content hash; (7) the host peak of the CUDA path (one shared (3, N) host backing) is within the CUDA model; (8) no per-plant / per-step transfer: the H2D is the entry arrays once (declared `h2d_bytes`). Root measures transfers and device peak.'''
from __future__ import annotations

import os
import unittest

import numpy as np

from leaf import water_vegetation as wv
from tests.test_water_vegetation import BIG, EDGES, catalog, plants_state

GPU = os.environ.get('LEAF_RELEASE_GPU') == '1'
DEVICE = int(os.environ.get('LEAF_GPU_DEVICE', '0'))


def case(layers):
    cat = catalog(layers=layers, grass=[0] + [1] * (layers - 1), shrub=[2] * layers)
    rng = np.random.default_rng(11)
    pl = [(i, 'grass' if i % 2 else 'shrub', float(rng.uniform(1, 19)), float(rng.uniform(1, 19)), float(rng.uniform(0.2, 1.1)),
           'live' if i % 4 else 'dead_standing') for i in range(1, 400)]
    st = plants_state(cat, pl, nx=20, ny=20)
    mask = np.ones((20, 20), dtype=bool)
    mask[3:6, 3:9] = False
    return cat, st, mask


@unittest.skipUnless(GPU, 'CUDA definitions: set LEAF_RELEASE_GPU=1 on the Root hardware lane')
class CudaDefinitions(unittest.TestCase):
    def test_cuda_equals_numpy_bit_for_bit(self):
        for L in (1, 3):
            cat, st, mask = case(L)
            a = wv.project_from_initial_state(st, cat, np.zeros(st.domain.shape), active_mask=mask, max_edges=EDGES, max_bytes=BIG, backend='numpy')
            b = wv.project_from_initial_state(st, cat, np.zeros(st.domain.shape), active_mask=mask, max_edges=EDGES, max_bytes=BIG, backend='cuda', device=DEVICE, max_device_bytes=1 << 30)
            for name in ('splash_cover', 'shading_cover', 'matrix_root_index'):
                np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
                self.assertFalse(getattr(b, name).flags.writeable)
            self.assertEqual(a.identity['content_sha256'], b.identity['content_sha256'])
            self.assertEqual((b.identity['backend'], b.identity['device']), ('cuda', DEVICE))
            self.assertEqual(b.identity['transfers']['d2h_bytes'], 3 * 8 * 400)
            self.assertEqual((b.identity['transfers']['h2d_calls'], b.identity['transfers']['d2h_calls']), (5, 1))        # ONE device-to-host call, five cold H2D entry arrays

    def test_explicit_device_and_no_fallback(self):
        cat, st, _ = case(1)
        with self.assertRaises(wv.VegetationBackendError):
            wv.project_from_initial_state(st, cat, np.zeros(st.domain.shape), max_edges=EDGES, max_bytes=BIG, backend='cuda', max_device_bytes=1 << 30)

    def test_device_byte_bound_refuses_before_upload_and_the_host_pool_is_gated_first(self):
        cat, st, mask = case(1)
        ref = wv.project_from_initial_state(st, cat, np.zeros(st.domain.shape), active_mask=mask, max_edges=EDGES, max_bytes=BIG)
        need = wv.estimate_device_bytes(400, ref.n_entries, 2)['device_bytes']
        with self.assertRaises(wv.VegetationBudgetError):
            wv.project_from_initial_state(st, cat, np.zeros(st.domain.shape), active_mask=mask, max_edges=EDGES, max_bytes=BIG, backend='cuda', device=DEVICE, max_device_bytes=need - 1)
        with self.assertRaises(wv.VegetationBudgetError):
            wv.project_from_initial_state(st, cat, np.zeros(st.domain.shape), active_mask=mask, max_edges=EDGES, max_bytes=1, backend='cuda', device=DEVICE, max_device_bytes=BIG)
        wv.project_from_initial_state(st, cat, np.zeros(st.domain.shape), active_mask=mask, max_edges=EDGES, max_bytes=BIG, backend='cuda', device=DEVICE, max_device_bytes=need)

    def test_host_peak_of_the_cuda_path_is_within_the_cuda_model_at_large_n(self):
        # correction04 (Root's finding): the host side of a CUDA run holds ONE shared (3, N) backing; its traced host peak must be within `estimate_projection_bytes(..., backend='cuda')`
        # (and above the NumPy model), warm context / source / kernel, originals built before the tracer. UNRUN here; Root measures it on the guarded GPU.
        import gc
        import tracemalloc
        cat = catalog(layers=2, grass=[0, 1], shrub=[0, 1])
        st = plants_state(cat, [], nx=500, ny=400)
        surf = np.zeros(st.domain.shape)
        kw = dict(max_edges=0, backend='cuda', device=DEVICE, max_device_bytes=1 << 30)
        wv.project_from_initial_state(st, cat, surf, max_bytes=1 << 40, **kw)                       # warm the context / kernel / source digests
        model = wv.estimate_projection_bytes(0, 0, 0, 200_000, 2, 2, backend='cuda')
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        wv.project_from_initial_state(st, cat, surf, max_bytes=model['peak_bytes'], **kw)
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        self.assertLessEqual(peak, model['peak_bytes'])
        self.assertGreater(peak, wv.estimate_projection_bytes(0, 0, 0, 200_000, 2, 2)['peak_bytes'])

    def test_current_device_must_equal_the_requested_device(self):
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 2:
            self.skipTest('needs two devices')
        other = 1 if DEVICE == 0 else 0
        cat, st, _ = case(1)
        with cp.cuda.Device(DEVICE):
            with self.assertRaises(wv.VegetationBackendError):
                wv.project_from_initial_state(st, cat, np.zeros(st.domain.shape), max_edges=EDGES, max_bytes=BIG, backend='cuda', device=other, max_device_bytes=1 << 30)


if __name__ == '__main__':
    unittest.main()
