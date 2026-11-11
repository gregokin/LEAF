'''UNCALIBRATED example: a dated LEAF interevent session on a synthetic soil column (no field data, no storm / wind).

Uses the complete examples/phase1/species.json fixture catalog, LEAF's own population generator, an actual
maple_syrup layered soil column / prepared transpiration operator and SYNTHETIC explicit weather (a sinusoidal
temperature, constant PET, a constant potential soil-evaporation rate and a constant soil intake). Every one of
these numbers is a fixture for demonstration; none is a calibration or a SYRUP / MASY weather product. The backend
and the global memory budget are REQUIRED arguments (there is no default and no fallback). No package or
environment setup is performed: run it with an interpreter that already imports numpy and maple_syrup (numba /
cupy only for those backends), for example from the candidate root:

    PYTHONPATH="$PWD:/home/okin/SYRUP/src" python examples/phase1/run_interevent.py \
        --backend numpy --max-bytes 500000000 --days 21 --out-dir /tmp/leaf-interevent-example

It runs whole-day records, flushes the partial growth period at the end, saves a restart checkpoint, restores it,
and prints compact JSON receipts. It shows executable use only; it is NOT a qualification and NOT a full
MASY / MAPLE storm or wind scenario.
'''
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from leaf import forcing, interevent, recording, state, traits  # noqa: E402
from leaf.initialization import generate_population  # noqa: E402

UNCALIBRATED = ('UNCALIBRATED synthetic example: fixture species traits, synthetic weather and soil; not a calibration, '
                'not a field simulation, no storm / wind transport')


def generation_config(nx, ny, dx, seed):
    return {'schema': 'leaf.generation_config', 'schema_version': 1, 'domain': {'nx': nx, 'ny': ny, 'dx_m': dx},
            'seed': seed, 'overlap': 'allowed', 'cover_definition': 'per_species_union', 'cover_tolerance': 0.02,
            'cover_subsamples': 4, 'max_iterations': 12, 'time_days': 0.0,
            'limits': {'max_plants': 200000, 'max_attempts': 50000, 'max_cover_cells': 4000000,
                       'max_cover_bytes': 8000000, 'max_working_bytes': 1048576},
            'species': {'grass': {'target_cover': 0.05, 'diameter_mean_m': 0.3, 'diameter_std_m': 0.05,
                                  'initial_age_days': 365},
                        'shrub': {'target_cover': 0.05, 'diameter_mean_m': 2.0, 'diameter_std_m': 0.3,
                                  'initial_age_days': 3650}}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=UNCALIBRATED)
    ap.add_argument('--backend', required=True, choices=interevent.BACKENDS)
    ap.add_argument('--max-bytes', required=True, type=int, help='global byte budget of the session (host + device)')
    ap.add_argument('--max-edges', type=int, default=20_000_000)
    ap.add_argument('--reserve-bytes', type=int, default=0, help='bytes the caller holds elsewhere')
    ap.add_argument('--builder', default='numpy', choices=('numpy', 'compiled'), help="root builder ('compiled' needs numba)")
    ap.add_argument('--days', type=int, default=21)
    ap.add_argument('--nx', type=int, default=32)
    ap.add_argument('--dx', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=20261111)
    ap.add_argument('--start', default='2026-05-01', help='YYYY-MM-DD (local midnight, UTC offset 0)')
    ap.add_argument('--pet-mm-day', type=float, default=5.0, help='SYNTHETIC shared PET')
    ap.add_argument('--e0-mm-day', type=float, default=1.0, help='SYNTHETIC explicit potential soil evaporation')
    ap.add_argument('--intake-mm-day', type=float, default=0.5, help='SYNTHETIC water directed to the soil (layer 0)')
    ap.add_argument('--out-dir', default=None, help='write the initial / final CSV and the checkpoint here')
    args = ap.parse_args(argv)

    from maple_syrup import subsurface as ss
    from maple_syrup import subsurface_transpiration as st

    catalog = traits.load_catalog(ROOT / 'examples' / 'phase1' / 'species.json')
    ny = args.nx
    surface = np.full((ny, args.nx), 100.0)
    plants, report = generate_population(catalog, generation_config(args.nx, ny, args.dx, args.seed), surface=surface)
    print(json.dumps({'note': UNCALIBRATED, 'n_plants': plants.n, 'counts': plants.counts()}), flush=True)

    layers = ss.soil_layers([0.0, 0.1, 0.5], diffusivity_m2_s=2e-7, theta_evaporation_stress=0.2, theta_residual=0.05,
                            theta_field_capacity=0.2, theta_saturation=0.45)
    n_cells = args.nx * ny
    mask = np.ones(n_cells, dtype=bool)
    if args.backend == 'cuda':
        import cupy as xp
        mask = xp.asarray(mask)
    else:
        xp = np
    column = ss.prepare_column(layers, mask, args.dx * args.dx, backend=args.backend, reserve_bytes=0)
    operator = st.prepare_transpiration(column)
    native = ss.uniform_state(column, [0.12, 0.15], 0.0)

    start = dt.datetime.fromisoformat(args.start).replace(tzinfo=dt.timezone.utc)
    session = interevent.IntereventSession(
        plants, catalog, operator=operator, native_state=native, start=start, backend=args.backend,
        growth_cadence=dt.timedelta(days=7), max_edges=args.max_edges, max_bytes=args.max_bytes,
        reserve_bytes=args.reserve_bytes, root_builder=args.builder)
    out = pathlib.Path(args.out_dir) if args.out_dir else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        state.write_plant_csv(session.snapshot(), out / 'initial.csv', initial=False)

    added = xp.zeros((2, n_cells), dtype=np.float64)
    added[0] += args.intake_mm_day * 1e-3
    evap = xp.full((n_cells,), args.e0_mm_day * 1e-3 / 86400.0, dtype=np.float64)
    last = None
    for i in range(args.days):
        t0 = start + dt.timedelta(days=i)
        temp = 18.0 + 8.0 * math.sin(2.0 * math.pi * (t0.timetuple().tm_yday - 100) / 365.0)   # SYNTHETIC
        rec = forcing.TimestepRecord(t0, t0 + dt.timedelta(days=1), temp, args.pet_mm_day)
        for piece in session.split_record(rec):            # no-op for whole days; shows the boundary helper
            last = session.step_interevent(piece, added_water_m=added, potential_evaporation_m_s=evap)
        if last['growth'].get('performed'):
            print(json.dumps({'day': i + 1, 'growth': {k: last['growth'][k] for k in ('period_days', 'year_end')},
                              'water_closure': last['water_closure'], 'rebuild': last['growth'].get('rebuild')}),
                  flush=True)
    flushed = session.flush_growth()              # finish the partial period so a consumer can export the canopy
    print(json.dumps({'flush': flushed['growth'].get('period_days'), 'describe': session.describe()}), flush=True)
    if out is not None:
        state.write_plant_csv(session.snapshot(), out / 'final.csv')
        path = session.save(out / 'checkpoint', overwrite=False)
        restored = interevent.IntereventSession.load(
            path, catalog, operator=operator, backend=args.backend, max_edges=args.max_edges, max_bytes=args.max_bytes,
            reserve_bytes=args.reserve_bytes, root_builder=args.builder)
        print(json.dumps({'checkpoint': str(path), 'restored_cursor': restored.cursor.isoformat(),
                          'restored_matches': restored.cursor == session.cursor}), flush=True)
        if os.environ.get('LEAF_EXAMPLE_SNAPSHOT_WRITER'):
            writer = recording.SnapshotWriter(str(out / 'veg_{index:03d}.csv'))
            session.write_snapshot(writer, 'initial')
    print(json.dumps({'last_receipt_memory': last['memory'] if last else None, 'note': UNCALIBRATED}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
