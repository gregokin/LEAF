"""Independent live-array profiles of pure kernels; no ecological model run."""
import gc
import json
from pathlib import Path
import sys
import tracemalloc

import numpy as np

from leaf import growth, traits

sys.path.append(str(Path(__file__).resolve().parent.parent / 'leaf_phase1_20261110'))
from profile_uptake_memory import make_hook


def inputs(xp, operation, count):
    args = {'B_kg': xp.full(count, 0.2, dtype=xp.float64),
            'species_index': xp.zeros(count, dtype=xp.int32),
            'live': xp.ones(count, dtype=xp.bool_)}
    if operation == 'demand':
        args.update(canopy_area_m2=xp.full(count, .1, dtype=xp.float64),
                    month=6, temperature_c=25., pet_mm_per_day=3., dt_days=1.)
    else:
        args.update(E_kg=xp.zeros(count, dtype=xp.float64),
                    requested_m3=xp.full(count, .001, dtype=xp.float64),
                    delivered_m3=xp.full(count, .00075, dtype=xp.float64),
                    activity_integral=np.full(2, 7/365, dtype=np.float64),
                    temperature_integral=np.full(2, .8*7/365, dtype=np.float64),
                    year_end=False)
    return args


def case(backend, operation, count, catalog):
    cp = None
    if backend == 'cuda':
        import cupy as cp
    xp = cp if cp else np
    hook = make_hook(cp) if cp else None
    if hook:
        hook.__enter__()
    try:
        table = growth.compile_species(catalog, backend, 1 << 29,
                                   device=int(xp.cuda.runtime.getDevice()) if backend == 'cuda' else None)
        args = inputs(xp, operation, count)
        gc.collect()
        if hook:
            cp.cuda.Device().synchronize()
            hook.raw_peak, hook.rounded_peak = hook.raw, hook.rounded
            baseline = hook.raw
        else:
            assert not tracemalloc.is_tracing()
            tracemalloc.start()
        result = getattr(growth, operation)(table, **args, max_bytes=1 << 29)
        memory = result['memory']
        report = {'backend': backend, 'operation': operation, 'plants': count,
                  'declared_total_bytes': memory['total_bytes'],
                  'declared_allocating_bytes': memory['allocating_bytes']}
        if hook:
            cp.cuda.Device().synchronize()
            report.update(live_requested_at_start_bytes=baseline,
                          live_requested_peak_bytes=hook.raw_peak,
                          live_rounded_peak_bytes=hook.rounded_peak,
                          declared_device_bytes=memory['device_bytes'],
                          array_underestimate=hook.raw_peak > memory['device_bytes'])
        else:
            current, peak = tracemalloc.get_traced_memory()
            snapshot = tracemalloc.take_snapshot()
            domain = np.lib.tracemalloc_domain
            retained_data = sum(t.size for t in snapshot.traces if t.domain == domain)
            tracemalloc.stop()
            report.update(traced_allocation_peak_bytes=peak,
                          traced_retained_bytes=current,
                          numpy_data_retained_bytes=retained_data,
                          array_underestimate=retained_data > memory['allocating_bytes'],
                          all_trace_exceeds_bound=peak > memory['allocating_bytes'],
                          note='All-trace peak includes excluded Python metadata; retained NumPy data is '
                               'separate. A retained-data excess is definite; total-trace excess alone is not.')
        del result, args, table
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
    catalog = traits.SpeciesCatalog.from_file(Path.cwd() / 'examples/phase1/species.json')
    xp = np
    if backend == 'cuda':
        import cupy as xp
    table = growth.compile_species(catalog, backend, 1 << 29,
                                   device=int(xp.cuda.runtime.getDevice()) if backend == 'cuda' else None)
    for operation in ('demand', 'grow'):
        answer = getattr(growth, operation)(table, **inputs(xp, operation, 1), max_bytes=1 << 29)
        del answer
    del table
    gc.collect()
    rows = [case(backend, operation, count, catalog)
            for count in (0, 4096, 200_000) for operation in ('demand', 'grow')]
    failures = sum(row['array_underestimate'] or row.get('unknown_free_events', 0)
                   or row.get('requested_live_at_end_bytes', 0) for row in rows)
    print(json.dumps({'scope': 'pure current kernel array profiles, not session/phase qualification',
                      'results': rows, 'failure_count': int(failures)}, indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
