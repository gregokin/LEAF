"""Independent allocation-before-admission probe of exported root helpers."""
import json
from unittest import mock

import numpy as np

from benchmarks import spatial_mapping  # warm/import before intercepting NumPy
from leaf import roots, uptake


def main():
    count = 100_000
    x = np.full(count, 0.5, dtype=np.float64)
    y = x.copy()
    radius = np.full(count, 0.01, dtype=np.float64)
    events = []
    real_arange = np.arange

    def observed_arange(*args, **kwargs):
        arr = real_arange(*args, **kwargs)
        events.append({'shape': list(arr.shape), 'bytes': arr.nbytes})
        return arr

    with mock.patch.object(roots.np, 'arange', side_effect=observed_arange):
        try:
            roots.horizontal_edges_compiled(
                x, y, radius, 1, 1, 1.0, max_edges=0, max_bytes=0)
        except uptake.AdmissionError as exc:
            refused = {'type': type(exc).__name__, 'stage': exc.stage}
        else:
            raise AssertionError('Zero-byte/edge helper call unexpectedly accepted')
    bad = any(event['bytes'] >= 8 * count for event in events)
    print(json.dumps({'case': 'exported compiled helper, zero edge/byte cap',
                      'refused': refused, 'allocations_before_refusal': events,
                      'population_array_before_admission': bad}, indent=2))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
