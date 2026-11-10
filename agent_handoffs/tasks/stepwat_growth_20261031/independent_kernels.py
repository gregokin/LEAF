"""Independent numerical oracles for the STEP-style growth kernels."""
from pathlib import Path
import argparse
import json
import math
import numpy as np

from leaf import stepwat as growth
from benchmarks import spatial_mapping as sm

TASK = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    rows = json.loads((TASK / 'analytical_expectations.json').read_text())['cases']
    xp = sm.cp if args.gpu else np
    assert xp is not None
    tested = []
    for year_end in (False, True):
        selected = [r for r in rows if r['input']['year_end'] == year_end]
        data = [r['input'] for r in selected]
        def vector(key):
            return xp.asarray([r[key] for r in data], dtype=xp.float64)
        p = len(data)
        B, E, uptake = vector('B_kg'), vector('seasonal_kg'), vector('uptake_m3')
        ids = xp.arange(p, dtype=xp.int32)
        mature, rate = vector('mature_kg'), vector('rate_per_year')
        support, live = vector('support_kg_per_m3'), vector('live_fraction')
        extra = vector('extra_fraction')
        density = xp.full(p, 2.5, dtype=xp.float64)
        ptr = xp.full(p, .003, dtype=xp.float64)
        activity = xp.asarray([[r['activity']] * 12 for r in data], dtype=xp.float64)
        tint = xp.asarray([r['temperature_factor'] * r['activity'] * r['period_days'] / 365
                           for r in data], dtype=xp.float64)
        outs = tuple(xp.zeros(p, dtype=xp.float64) for _ in range(7))
        inputs = (B, E, uptake, ids, mature, rate, support, live, density, ptr, extra, activity, tint)
        if args.gpu:
            growth._gpu_kernel()(((p + 255) // 256,), (256,), inputs + (
                np.int32(0), np.int32(1), np.float64(3.), np.float64(7.), np.int32(year_end),
                *outs, np.int64(p)))
            xp.cuda.Device().synchronize()
        else:
            growth._grow_cpu(*inputs, 0, 1, 3., 7., int(year_end), *outs)
        Bc, Ec, gain, loss, canopy, root, demand = [a.get() if args.gpu else a for a in outs]
        for i, case in enumerate(selected):
            expected = case['expected']
            mass = expected['total_aboveground_kg']
            radius = math.sqrt(mass / (math.pi * 2.5))
            wants = [mass, expected['seasonal_kg'],
                     expected['normal_gain_kg'] + expected['extra_gain_kg'],
                     expected['loss_kg'], radius, 3 * radius,
                     math.pi * radius ** 2 * .003 * data[i]['live_fraction'] * data[i]['activity']]
            actual = [Bc[i], Ec[i], gain[i], loss[i], canopy[i], root[i], demand[i]]
            np.testing.assert_allclose(actual, wants, rtol=3e-13, atol=1e-14,
                                       err_msg=case['name'])
            assert np.isfinite(actual).all()
            tested.append(case['name'])
    print(json.dumps({'ok': True, 'backend': 'cuda' if args.gpu else 'numba',
                      'source': str(Path(growth.__file__).resolve()),
                      'cases': tested, 'count': len(tested)}))


if __name__ == '__main__':
    main()
