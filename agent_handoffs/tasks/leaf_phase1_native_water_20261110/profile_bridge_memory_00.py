"""Independent allocation probes for the saved bridge composition, not a model run."""
from pathlib import Path
import gc
import json
import sys
import tracemalloc

import numpy as np

from leaf import syrup_water as bridge
from verify_syrup_bridge import fixture

sys.path.append(str(Path(__file__).resolve().parent.parent / 'leaf_phase1_20261110'))
from profile_uptake_memory import make_hook


def case(backend, plants, cells, layers):
    cp = None
    if backend == 'cuda':
        import cupy as cp
    hook = make_hook(cp) if cp else None
    if hook:
        hook.__enter__()
    try:
        p = np.arange(plants, dtype=np.int64)
        args = dict(demand=np.full(plants, 1e-4), offsets=np.arange(plants+1)*2,
                    resources=np.column_stack((p % cells, (layers-1)*cells + (p*17) % cells)).ravel(),
                    weights=np.full(2*plants, 0.5), available=np.full((layers, cells), .001), area=1.0)
        m, d, op, state = fixture(backend, backend, **args)
        column = op.column
        gc.collect()
        if hook:
            cp.cuda.Device().synchronize()
            baseline = hook.raw
            hook.raw_peak, hook.rounded_peak = hook.raw, hook.rounded
        else:
            borrowed = (m.resident_bytes + d.nbytes + state.water_m.nbytes + op.residual_xp.nbytes
                        + op.saturation_xp.nbytes + column.memory.column_bytes
                        + column.memory.layer_bytes + column.memory.parameter_bytes)
            tracemalloc.start()
        result = bridge.withdraw_plants(m, d, op, state, max_bytes=100_000_000)
        assert result.conservation['closed']
        if hook:
            cp.cuda.Device().synchronize()
            report = {'observed_live_requested_peak_bytes': hook.raw_peak,
                      'observed_live_rounded_peak_bytes': hook.rounded_peak,
                      'live_requested_at_call_start_bytes': baseline,
                      'declared_native_required_bytes': result.memory['native_required_bytes'],
                      'array_scale_underestimate': hook.raw_peak > result.memory['native_required_bytes'],
                      'pool_reserved_bytes': cp.get_default_memory_pool().total_bytes()}
        else:
            retained, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            report = {'borrowed_resident_bytes_from_native_component_estimates': borrowed,
                      'traced_additional_peak_bytes': peak, 'traced_retained_additional_bytes': retained,
                      'declared_total_bytes': result.memory['total_bytes'],
                      'array_scale_underestimate': borrowed + peak > result.memory['total_bytes'] + 16_384,
                      'note': 'CPU trace includes Python bookkeeping; 16 KiB distinguishes that from '
                              'array-scale error, not an increase in admission budget'}
        report.update(backend=backend, plants=plants, cells=cells, layers=layers,
                      declared_entries=result.memory['entries'])
        del result, m, d, op, state, column, args
        gc.collect()
        if hook:
            cp.cuda.Device().synchronize()
            report.update(unknown_free_events=hook.unknown_frees, requested_live_at_end_bytes=hook.raw)
        return report
    finally:
        if hook:
            hook.__exit__(None, None, None)


def main():
    backend = sys.argv[1]
    # Warm native and allocator compiled paths before measurement.
    m, d, op, state = fixture(backend, backend, demand=[1e-4], offsets=[0, 2], resources=[0, 1],
                              weights=[.5, .5], available=[[.001], [.001]], area=1.0)
    result = bridge.withdraw_plants(m, d, op, state, max_bytes=100_000_000)
    del m, d, op, state, result
    gc.collect()
    results = [case(backend, 200_000, 1, 2), case(backend, 4096, 65_536, 3)]
    failures = sum(r['array_scale_underestimate'] for r in results)
    print(json.dumps({'scope': 'live allocation probes of review composition; no phase acceptance',
                      'backend': backend, 'results': results, 'underestimate_cases': failures}, indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    main()
