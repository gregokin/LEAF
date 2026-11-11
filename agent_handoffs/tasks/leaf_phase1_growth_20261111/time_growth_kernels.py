"""Kernel microbenchmark, not a water/growth trajectory or coupled model run."""
from datetime import date
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from leaf import growth, traits
from profile_growth_03 import inputs


def one_size(backend, count, repeats=200):
    started = time.perf_counter()
    xp = np
    if backend == 'cuda':
        import cupy as xp
    catalog = traits.SpeciesCatalog.from_file(Path.cwd() / 'examples/phase1/species.json')
    table = growth.compile_species(catalog, backend, 1 << 29,
                                   device=int(xp.cuda.runtime.getDevice()) if backend == 'cuda' else None)
    dargs = inputs(xp, 'demand', count)
    gargs = inputs(xp, 'grow', count)
    reserves = {'demand': sum(v.nbytes for v in gargs.values() if hasattr(v, 'nbytes')),
                'grow': sum(v.nbytes for v in dargs.values() if hasattr(v, 'nbytes'))}
    for operation, args in (('demand', dargs), ('grow', gargs)):
        answer = getattr(growth, operation)(table, **args, max_bytes=1 << 29,
                                           extra_reserve_bytes=reserves[operation])
        del answer
    if backend == 'cuda':
        xp.cuda.Device().synchronize()
    startup = time.perf_counter() - started
    timings = {}
    for operation, args in (('demand', dargs), ('grow', gargs)):
        for _ in range(5):
            answer = getattr(growth, operation)(table, **args, max_bytes=1 << 29,
                                               extra_reserve_bytes=reserves[operation])
            del answer
        if backend == 'cuda':
            xp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            answer = getattr(growth, operation)(table, **args, max_bytes=1 << 29,
                                               extra_reserve_bytes=reserves[operation])
            del answer
        if backend == 'cuda':
            xp.cuda.Device().synchronize()
        elapsed = time.perf_counter() - t0
        timings[operation] = {'calls': repeats, 'wall_seconds': elapsed,
                              'wall_seconds_per_call': elapsed / repeats}
    start = date(2026, 1, 1)
    days = (date(2056, 1, 1) - start).days
    weekly = set(range(7, days + 1, 7))
    year_ends = {(date(year, 1, 1) - start).days for year in range(2027, 2057)}
    calls_grow = len(weekly | year_ends)
    compute_only = (days * timings['demand']['wall_seconds_per_call']
                    + calls_grow * timings['grow']['wall_seconds_per_call'])
    return {'backend': backend, 'plants': count, 'observed_setup_and_first_calls_seconds': startup,
            'setup_note': 'Observed in this process with existing cache directories; no claim of empty caches.',
            'steady_state': timings,
            'isolated_30year_compute_extrapolation': {
                'start': start.isoformat(), 'end': '2056-01-01', 'daily_demand_calls': days,
                'weekly_plus_year_end_growth_calls': calls_grow, 'estimated_wall_seconds': compute_only,
                'excludes': 'root scatter/gather and native hydrology, root/canopy rebuilds and transfers, '
                            'event orchestration, snapshots/restart I/O and changing population/traits'},
            'fixture_note': 'Repeated identical valid one-step inputs; biomass and water do not evolve. '
                            'This is NOT a 30-year simulation or an end-to-end runtime estimate.'}


def main():
    backend = sys.argv[1]
    source = Path(growth.__file__)
    print(json.dumps({'scope': 'pure demand/growth microbenchmark only', 'source': str(source),
                      'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                      'python': sys.version, 'numpy': np.__version__,
                      'results': [one_size(backend, count) for count in (35_000, 200_000)]}, indent=2))


if __name__ == '__main__':
    main()
