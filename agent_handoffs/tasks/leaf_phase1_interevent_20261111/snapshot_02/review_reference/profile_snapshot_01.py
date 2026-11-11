"""Count a lower bound of simultaneously live snapshot arrays, excluding metadata.

The real NumPy uniqueness validator is wrapped only to observe its returned array
while the real constructor/materializer arrays are still alive. No RSS, pool
rounding or Python object overhead is confused with array-data bytes.
"""
import datetime as dt
import gc
import json
from pathlib import Path
import sys
from unittest import mock

import numpy as np

from leaf import state, traits
from leaf.interevent import IntereventSession
from maple_syrup import subsurface as ss
from maple_syrup import subsurface_transpiration as st


def fixture(backend, count):
    xp = np
    if backend == 'cuda':
        import cupy as xp
    catalog = traits.SpeciesCatalog.from_file(Path.cwd() / 'examples/phase1/species.json')
    domain = state.Domain(1, 1, 1.)
    plants = state.PlantState(domain, catalog.species_ids, catalog.hash,
                              ids=np.arange(1, count+1, dtype=np.int64),
                              species_index=np.zeros(count, dtype=np.int32),
                              x_m=np.full(count, .5), y_m=np.full(count, .5),
                              establishment_ground_m=np.zeros(count),
                              establishment_unresolved=np.zeros(count, dtype=bool),
                              agb_dry_kg=np.full(count, .05), seasonal_agb_dry_kg=np.zeros(count),
                              age_days=np.full(count, 100.), status=np.ones(count, dtype=np.uint8)*state.LIVE)
    layers = ss.soil_layers([0., .25, 1.], theta_residual=.05, theta_field_capacity=.3,
                            theta_saturation=.5, diffusivity_m2_s=0., theta_evaporation_stress=.1)
    column = ss.prepare_column(layers, xp.ones(1, dtype=bool), 1., backend=backend)
    operator = st.prepare_transpiration(column)
    water = ss.column_state(column, xp.asarray(np.array([[.05], [.15]])))
    return IntereventSession(plants, catalog, operator=operator, native_state=water,
                              start=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc), backend=backend,
                              growth_cadence=dt.timedelta(days=7), max_edges=1_000_000, max_bytes=1 << 29)


def pointer(array):
    return int(array.__array_interface__['data'][0])


def main():
    backend = sys.argv[1]
    count = 100_000
    session = fixture(backend, count)
    borrowed = list(session._base._arrays().values())
    borrowed += [session._live_rows, session._live_ids, session._sidx, session._live]
    borrowed += [getattr(session._b, key) for key in ('B', 'E', 'D', 'U', 'A', 'T', 'area')]
    borrowed += [getattr(session.root_map, key) for key in ('ids', 'counts', 'offsets', 'resources', 'weights')]
    borrowed_ptrs = {pointer(array) for array in borrowed if isinstance(array, np.ndarray) and array.size}
    original_check = session._check_stage
    original_unique = np.unique
    declarations, observations = [], []

    def observed_check(stage, total, mem):
        if stage == 'snapshot':
            resident = sum(session._persistent(session._b).values())
            declarations.append({'declared_total_bytes': total, 'resident_bytes': resident,
                                 'declared_additional_bytes': total-resident})
        return original_check(stage, total, mem)

    def observed_unique(*args, **kwargs):
        result = original_unique(*args, **kwargs)
        arrays = [('unique_output', result)]
        frame = sys._getframe().f_back
        names = set()
        while frame is not None:
            name = frame.f_code.co_name
            if name in ('_validate_arrays', '_materialize'):
                names.add(name)
                values = frame.f_locals.get('a', {})
                arrays.extend((f'{name}.{key}', value) for key, value in values.items())
                if name == '_materialize':
                    arrays.extend((f'{name}.{key}', frame.f_locals.get(key)) for key in ('B_live', 'E_live'))
            frame = frame.f_back
        if names == {'_validate_arrays', '_materialize'}:
            data = {}
            for label, array in arrays:
                if isinstance(array, np.ndarray) and array.size and pointer(array) not in borrowed_ptrs:
                    key = pointer(array)
                    if key not in data or array.nbytes > data[key]['bytes']:
                        data[key] = {'label': label, 'bytes': array.nbytes}
            observations.append({'simultaneous_additional_array_bytes_lower_bound': sum(v['bytes'] for v in data.values()),
                                 'arrays': list(data.values())})
        return result

    with mock.patch.object(session, '_check_stage', new=observed_check), \
            mock.patch.object(state.np, 'unique', new=observed_unique):
        snapshot = session.snapshot()
    assert snapshot.n == count
    measured = max(row['simultaneous_additional_array_bytes_lower_bound'] for row in observations)
    declared = declarations[0]['declared_additional_bytes']
    print(json.dumps({'scope': 'actual simultaneous array-data lower bound during snapshot validation',
                      'backend': backend, 'plants': count, 'declaration': declarations[0],
                      'observed_additional_lower_bound_bytes': measured,
                      'underestimate_confirmed': measured > declared,
                      'observations': observations}, indent=2))
    del snapshot, session
    gc.collect()
    return 1 if measured > declared else 0


if __name__ == '__main__':
    raise SystemExit(main())
