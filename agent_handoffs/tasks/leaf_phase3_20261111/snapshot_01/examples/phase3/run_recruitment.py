'''UNCALIBRATED example: one annual reproduction opportunity (seed establishment + clonal daughters) on a synthetic column.

Uses the complete examples/phase3/species.json FIXTURE catalog (grass = clonal only, shrub = seed only, forb = both,
moss_inactive = inactive; EVERY reproduction number is an uncalibrated fixture, not a field default), a small hand-made
population, an actual maple_syrup layered soil column / prepared transpiration operator and SYNTHETIC explicit weather and
surface. The backend and the global memory budget are REQUIRED arguments (no default, no fallback). No package or
environment setup is performed; run with an interpreter that already imports numpy and maple_syrup (numba only for
`--builder compiled`, cupy only for `--backend cuda`), for example from the candidate root:

    PYTHONPATH="$PWD:/home/okin/SYRUP/src" python examples/phase3/run_recruitment.py \
        --backend numpy --builder compiled --max-bytes 500000000 --out-dir /tmp/leaf-phase3-example

Timeline (UTC, whole-day records): open interevent days until the new year, the explicit annual mortality decision (the
grass fixture has a background hazard), then `apply_reproduction(year, surface, ...)` with explicit limits, one more open
day (newborn plants now request water), a version-3 checkpoint and its restore. The ecological patch size is an explicit
physical length (`--patch-m`), independent of `--dx`. The population placement is a HOST step (on cuda B / E are
transferred and counted); the phase is not GPU resident. This is executable usage only: NOT a qualification and NOT a MAPLE
/ SYRUP / MASY scenario. Limitations: external-propagule seeds, canopy overlap allowed, no surface remapping, no seed bank.
'''
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from leaf import forcing, interevent, state, traits  # noqa: E402

UNCALIBRATED = ('UNCALIBRATED synthetic example: fixture reproduction numbers, synthetic weather / soil / surface; not a '
                'calibration, not a field simulation')


def make_population(catalog, domain):
    rec = []

    def add(pid, sp, x, y, b, e, age):
        rec.append(dict(plant_id=pid, species=sp, x_m=x, y_m=y, establishment_ground_m=100.0, agb_dry_kg=b,
                        seasonal_agb_dry_kg=e, age_days=age, status='live'))

    add(10, 'grass', 2.0, 2.0, 0.5, 0.1, 800.0)
    add(11, 'grass', 5.0, 5.0, 0.4, 0.0, 900.0)
    add(12, 'forb', 3.5, 6.0, 0.2, 0.05, 400.0)
    add(13, 'shrub', 6.0, 2.0, 5.0, 0.0, 2000.0)
    add(14, 'moss_inactive', 1.0, 6.5, 0.04, 0.0, 300.0)
    return state.PlantState.from_records(catalog, domain, rec, time_days=0.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=UNCALIBRATED)
    ap.add_argument('--backend', required=True, choices=interevent.BACKENDS)
    ap.add_argument('--max-bytes', required=True, type=int, help='global byte budget of the session (host + device)')
    ap.add_argument('--max-edges', type=int, default=20_000_000)
    ap.add_argument('--reserve-bytes', type=int, default=0)
    ap.add_argument('--builder', default='numpy', choices=('numpy', 'compiled'), help="root builder ('compiled' needs numba)")
    ap.add_argument('--nx', type=int, default=8)
    ap.add_argument('--dx', type=float, default=1.0, help='hydraulic cell size (m); the patch size is independent')
    ap.add_argument('--patch-m', type=float, default=4.0, help='ecological patch edge (m), explicit and fixed')
    ap.add_argument('--rng-seed', type=int, default=424242)
    ap.add_argument('--start', default='2026-12-29', help='YYYY-MM-DD (local midnight, UTC offset 0)')
    ap.add_argument('--max-new-plants', type=int, default=1000)
    ap.add_argument('--max-total-records', type=int, default=100000)
    ap.add_argument('--max-attempts', type=int, default=100000)
    ap.add_argument('--pet-mm-day', type=float, default=5.0, help='SYNTHETIC shared PET')
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args(argv)

    from maple_syrup import subsurface as ss
    from maple_syrup import subsurface_transpiration as st

    catalog = traits.load_catalog(ROOT / 'examples' / 'phase3' / 'species.json')
    ny = args.nx
    domain = state.Domain(args.nx, ny, args.dx)
    plants = make_population(catalog, domain)
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
        reserve_bytes=args.reserve_bytes, root_builder=args.builder, rng_seed=args.rng_seed,
        reproduction_patch_size_m=args.patch_m)
    out = pathlib.Path(args.out_dir) if args.out_dir else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        state.write_plant_csv(session.snapshot(), out / 'initial.csv')
    added = xp.zeros((2, n_cells), dtype=np.float64)
    added[0] += 0.5e-3
    evap = xp.full((n_cells,), 1.0e-3 / 86400.0, dtype=np.float64)

    def open_day():
        t0 = session.cursor
        rec = forcing.TimestepRecord(t0, t0 + dt.timedelta(days=1), 3.0, args.pet_mm_day)   # SYNTHETIC cold-season day
        return [session.step_interevent(p, added_water_m=added, potential_evaporation_m_s=evap)
                for p in session.split_record(rec)][-1]

    while session.cursor.year == start.year:
        open_day()
    year = session.annual_due_year
    if year is not None:
        dec = session.apply_annual_decision(year)
        print(json.dumps({'annual_decision': {k: dec[k] for k in ('year', 'n_died_age_background', 'n_live_before',
                                                                  'n_live_after')}}), flush=True)
    year = start.year
    surface = np.full(domain.shape, 100.0)
    rep = session.apply_reproduction(year, surface, max_new_plants=args.max_new_plants,
                                     max_total_records=args.max_total_records, max_attempts=args.max_attempts)
    d = rep['reproduction']
    print(json.dumps({'reproduction': {k: d[k] for k in ('n_seed_recruits', 'n_daughters_requested', 'n_daughters_placed',
                                                         'n_daughters_rejected', 'n_births', 'id_range', 'ledger')},
                      'rebind': rep['rebind'], 'water_unchanged': rep['native_water_unchanged'],
                      'note': UNCALIBRATED}), flush=True)
    open_day()                      # newborn plants request water only from here on
    if out is not None:
        state.write_plant_csv(session.snapshot(), out / 'final.csv')
        session.flush_growth()
        path = session.save(out / 'checkpoint', overwrite=False)
        restored = interevent.IntereventSession.load(
            path, catalog, operator=operator, backend=args.backend, max_edges=args.max_edges,
            max_bytes=args.max_bytes, reserve_bytes=args.reserve_bytes, root_builder=args.builder)
        print(json.dumps({'checkpoint': str(path), 'restored_opportunities': restored.n_reproduction_opportunities,
                          'restored_last_year': restored.reproduction_last_year}), flush=True)
    print(json.dumps({'phase3': session.describe()['phase3'], 'note': UNCALIBRATED}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
