"""Independent lower bound on the actual CUDA daily-step array peak."""
import datetime as dt
import gc
import json
from pathlib import Path
import sys

import numpy as np

from leaf import forcing, state, traits
from leaf.interevent import IntereventSession
from maple_syrup import subsurface as ss
from maple_syrup import subsurface_transpiration as st

sys.path.append(str(Path(__file__).resolve().parent.parent / 'leaf_phase1_20261110'))
from profile_uptake_memory import make_hook


def main():
    import cupy as cp
    count, nx = 100_000, 64
    cells = nx * nx
    catalog = traits.SpeciesCatalog.from_file(Path.cwd() / 'examples/phase1/species.json')
    domain = state.Domain(nx, nx, 1.)
    plants = state.PlantState(domain, catalog.species_ids, catalog.hash,
                              ids=np.arange(1, count+1, dtype=np.int64),
                              species_index=np.zeros(count, dtype=np.int32),
                              x_m=np.full(count, .5), y_m=np.full(count, .5),
                              establishment_ground_m=np.zeros(count),
                              establishment_unresolved=np.zeros(count, dtype=bool),
                              agb_dry_kg=np.full(count, .05), seasonal_agb_dry_kg=np.zeros(count),
                              age_days=np.full(count, 100.), status=np.full(count, state.LIVE, dtype=np.uint8))
    with make_hook(cp) as hook:
        layers = ss.soil_layers([0., .25, 1.], theta_residual=.05, theta_field_capacity=.3,
                                theta_saturation=.5, diffusivity_m2_s=0., theta_evaporation_stress=.1)
        column = ss.prepare_column(layers, cp.ones(cells, dtype=bool), 1., backend='cuda')
        operator = st.prepare_transpiration(column)
        water = ss.column_state(column, cp.asarray(np.broadcast_to([[.05], [.15]], (2, cells)).copy()))
        session = IntereventSession(plants, catalog, operator=operator, native_state=water,
                                    start=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc), backend='cuda',
                                    growth_cadence=dt.timedelta(days=7), max_edges=1_000_000, max_bytes=1 << 29)
        del water
        added, evaporation = cp.zeros((2, cells)), cp.zeros(cells)
        host_arrays = list(session._base._arrays().values())
        host_arrays += [session._live_rows, session._live_ids, session._b.A, session._b.T]
        host_arrays += list(session._table.host.values())
        by_pointer = {}
        for array in host_arrays:
            if isinstance(array, np.ndarray) and array.size:
                pointer = int(array.__array_interface__['data'][0])
                by_pointer[pointer] = max(by_pointer.get(pointer, 0), array.nbytes)
        host_lower_bound = sum(by_pointer.values())
        del host_arrays
        gc.collect()
        cp.cuda.Device().synchronize()
        baseline = hook.raw
        hook.raw_peak, hook.rounded_peak = hook.raw, hook.rounded
        record = forcing.TimestepRecord(session.cursor, session.cursor + dt.timedelta(days=1),
                                        25., pet_mm_per_day=3.)
        receipt = session.step_interevent(record, added_water_m=added,
                                          potential_evaporation_m_s=evaporation)
        cp.cuda.Device().synchronize()
        declared = receipt['memory']['peak_estimate_bytes']
        observed = hook.raw_peak + host_lower_bound
        report = {'scope': 'daily actual native step, synthetic hot-resource population, no growth boundary',
                  'plants': count, 'cells': cells, 'layers': 2,
                  'host_resident_array_lower_bound_bytes': host_lower_bound,
                  'device_live_at_start_bytes': baseline, 'device_live_requested_peak_bytes': hook.raw_peak,
                  'global_array_peak_lower_bound_bytes': observed, 'declared_global_peak_bytes': declared,
                  'underestimate_confirmed': observed > declared,
                  'stage_estimates': receipt['memory']['stage_totals'],
                  'note': 'Host lower bound omits native host parameters and temporary host packets; '
                          'raw device bytes include every hooked array and exclude pool rounding/context.'}
        del session, column, operator, added, evaporation
        gc.collect()
        cp.cuda.Device().synchronize()
        report.update(unknown_free_events=hook.unknown_frees, requested_live_at_end_bytes=hook.raw)
    print(json.dumps(report, indent=2))
    return 1 if report['underestimate_confirmed'] or report['unknown_free_events'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
