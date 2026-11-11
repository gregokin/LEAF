"""Independent operation allocation measurements; not an ecological simulation.

CPU tracemalloc includes Python bookkeeping as well as NumPy allocations. CUDA
hooks separately measure requested and pool-rounded live bytes, excluding CUDA
context, driver allocations and other allocators. Inputs predate each measured
operation and are subtracted; no plant/resource trajectories are retained.
"""
import gc
import json
import sys
import tracemalloc

import numpy as np

from leaf import uptake


def make_hook(cp):
    class Hook(cp.cuda.MemoryHook):
        name = 'leaf_phase1_independent_live_allocation'

        def __init__(self):
            self.active = {}
            self.raw = self.rounded = self.raw_peak = self.rounded_peak = 0
            self.unknown_frees = 0

        def malloc_postprocess(self, **kw):
            if not kw['mem_ptr']:
                return
            key = (kw['device_id'], kw['pmem_id'])
            if key in self.active:
                raise RuntimeError('Duplicate live allocation in profiler')
            raw, rounded = int(kw['size']), int(kw['mem_size'])
            self.active[key] = (raw, rounded)
            self.raw += raw
            self.rounded += rounded
            self.raw_peak = max(self.raw_peak, self.raw)
            self.rounded_peak = max(self.rounded_peak, self.rounded)

        def free_postprocess(self, **kw):
            key = (kw['device_id'], kw['pmem_id'])
            if key not in self.active:
                self.unknown_frees += 1
                return
            raw, rounded = self.active.pop(key)
            self.raw -= raw
            self.rounded -= rounded

    return Hook()


def arrays(p, edges_per_plant, resources):
    ids = np.arange(p, dtype=np.int64)
    offsets = np.arange(p + 1, dtype=np.int64) * edges_per_plant
    edges = p * edges_per_plant
    res = np.arange(edges, dtype=np.int64) % resources
    weights = np.full(edges, 1.0 / edges_per_plant)
    return ids, offsets, res, weights


def measured(label, fn, estimate, hook, cp):
    gc.collect()
    if hook is None:
        tracemalloc.start()
        result = fn()
        retained, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        report = {'stage': label, 'estimated_additional_bytes': estimate,
                  'traced_peak_additional_bytes': peak,
                  'traced_retained_additional_bytes': retained,
                  'array_scale_underestimate': peak > estimate + 16_384,
                  'measurement_note': 'Includes Python bookkeeping; 16 KiB distinguishes bookkeeping '
                                      'from array-scale error, not an increased admission cap'}
    else:
        cp.cuda.Device().synchronize()
        raw_base, rounded_base = hook.raw, hook.rounded
        hook.raw_peak, hook.rounded_peak = raw_base, rounded_base
        result = fn()
        cp.cuda.Device().synchronize()
        raw_peak = hook.raw_peak - raw_base
        report = {'stage': label, 'estimated_additional_bytes': estimate,
                  'requested_peak_additional_bytes': raw_peak,
                  'pool_rounded_peak_additional_bytes': hook.rounded_peak - rounded_base,
                  'requested_retained_additional_bytes': hook.raw - raw_base,
                  'array_scale_underestimate': raw_peak > estimate,
                  'pool_used_bytes_after': cp.get_default_memory_pool().used_bytes(),
                  'pool_reserved_bytes_after': cp.get_default_memory_pool().total_bytes(),
                  'measurement_note': 'CUDA pool hook, raw requested algorithm allocations compared '
                                      'to estimate; rounded pool/context are separate'}
    return result, report


def run_case(backend, p, k, r, *, device_source=False):
    cp = None
    if backend == 'cuda':
        import cupy as cp
    hook = make_hook(cp) if cp else None
    if hook:
        hook.__enter__()
    try:
        source = arrays(p, k, r)
        if device_source:
            source = tuple(cp.asarray(a) for a in source)
        xp = cp if cp else np
        demand = xp.full(p, 0.0001)
        supply = xp.full(r, 0.00005)
        map_estimate = uptake.estimate_map_bytes(p, p*k, r, backend, host_source=not device_source)['total_bytes']
        m, map_report = measured('map', lambda: uptake.RootMap.from_arrays(
            *source, n_layers=1, n_cells=r, backend=backend,
            max_edges=2_000_000, max_bytes=1_500_000_000), map_estimate, hook, cp)
        plan_estimate = uptake.estimate_plan_bytes(p, p*k, r, backend)['total_bytes']
        plan, plan_report = measured('plan', lambda: uptake.propose(m, demand, supply), plan_estimate, hook, cp)
        feedback_estimate = uptake.estimate_feedback_bytes(p, p*k, r, backend)['total_bytes']
        feedback, feedback_report = measured('feedback', plan.accept_all, feedback_estimate, hook, cp)
        assert feedback.closure['closed']
        reports = [map_report, plan_report, feedback_report]
        del feedback, plan, m, demand, supply, source
        gc.collect()
        if cp:
            cp.cuda.Device().synchronize()
        return {'backend': backend, 'n_plants': p, 'edges_per_plant': k, 'n_resources': r,
                'device_source': device_source, 'operations': reports,
                'unknown_free_events': hook.unknown_frees if hook else None,
                'requested_live_at_end_bytes': hook.raw if hook else None}
    finally:
        if hook:
            hook.__exit__(None, None, None)


def main():
    backend = sys.argv[1]
    if backend not in ('numpy', 'numba', 'cuda'):
        raise ValueError('Expected numpy, numba or cuda')
    # Warm compiled kernels outside measurements. CUDA inventory/selection is
    # enforced by the separate controller before this process starts.
    xp = np
    if backend == 'cuda':
        import cupy as xp
    tiny = uptake.RootMap.from_arrays(*arrays(1, 1, 1), n_layers=1, n_cells=1,
                                      backend=backend, max_edges=10, max_bytes=10**6)
    uptake.propose(tiny, xp.ones(1), xp.ones(1)).accept_all()
    del tiny
    cases = [(200_000, 1, 1), (1, 100_000, 100_000), (4096, 4, 49_152), (0, 1, 49_152)]
    results = [run_case(backend, *case) for case in cases]
    if backend == 'cuda':
        results.append(run_case(backend, 200_000, 1, 1, device_source=True))
    bad = [r for r in results if any(op['array_scale_underestimate'] for op in r['operations'])]
    print(json.dumps({'scope': 'Independent allocation profiling, not ecological time integration',
                      'backend': backend, 'cases': results,
                      'array_scale_underestimate_cases': len(bad)}, indent=2))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
