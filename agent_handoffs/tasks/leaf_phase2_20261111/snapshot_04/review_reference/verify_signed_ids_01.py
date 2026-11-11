"""Regression for the existing signed-int64 plant-ID contract (including zero)."""
import json
import sys

import numpy as np
from leaf import demography, state
from probe_entrypoints_02 import catalog


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else 'numba'
    xp = np
    if backend == 'cuda':
        import cupy as xp
    cat = catalog()
    ids = np.array([np.iinfo(np.int64).min, -1, 0, np.iinfo(np.int64).max], dtype=np.int64)
    records = [dict(plant_id=int(key), species='grass', x_m=.5, y_m=.5,
                    establishment_ground_m=0., agb_dry_kg=.1, seasonal_agb_dry_kg=0.,
                    age_days=1000., status='live') for key in ids]
    plants = state.PlantState.from_records(cat, state.Domain(1, 1, 1.), records)
    np.testing.assert_array_equal(plants.ids, ids)
    table = demography.compile_species(cat, backend, device=0 if backend == 'cuda' else None,
                                        max_bytes=10_000_000)
    kwargs = dict(species_index=xp.zeros(4, dtype=np.int32), plant_ids=ids,
                  age_end_days=np.full(4, 1000.), stress=xp.zeros(4, dtype=np.int32),
                  required_support=xp.ones(4), actual_support=xp.ones(4),
                  planned_growth=xp.ones(4), actual_growth=xp.ones(4),
                  period_days=365., seed=123, counter=0, max_bytes=10_000_000)
    result = demography.decide(table, **kwargs)
    permutation = np.array([3, 1, 0, 2])
    shuffled = {k: v[permutation] if hasattr(v, 'shape') and v.shape == (4,) else v
                for k, v in kwargs.items()}
    permuted = demography.decide(table, **shuffled)
    for field in ['cause', 'loss_fraction', 'stress_new', 'flag', 'p_age']:
        a, b = result[field], permuted[field]
        if hasattr(a, 'get'):
            a, b = a.get(), b.get()
        np.testing.assert_array_equal(a[permutation], b)
    u = demography.keyed_uniform(123, 0, demography.STREAM_AGE, ids)
    expected = np.array([demography.keyed_uniform_py(123, 0, demography.STREAM_AGE, int(k)) for k in ids])
    np.testing.assert_array_equal(u, expected)
    assert np.all((u >= 0.) & (u < 1.))
    print(json.dumps({'backend': backend, 'signed_int64_extremes_and_zero': 'pass',
                      'canonical_state_compatibility': 'pass', 'keyed_rng_and_row_permutation': 'pass'}))


if __name__ == '__main__':
    main()
