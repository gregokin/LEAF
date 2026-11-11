'''UNCALIBRATED example: a post-wind surface-event response and the annual demography decision on a synthetic column.

Uses the complete examples/phase2/species.json FIXTURE catalog (species `grass`, `shrub`; every coefficient is an
uncalibrated fixture), LEAF's own population generator, an actual maple_syrup layered soil column / prepared
transpiration operator and SYNTHETIC explicit weather and a SYNTHETIC surface event (a hand-made deposition patch, an
erosion patch and a wind abrasion dose). None of these numbers is a calibration, a MAPLE wind result or a SYRUP water
result. The backend and the global memory budget are REQUIRED arguments (no default, no fallback). No package or
environment setup is performed: run it with an interpreter that already imports numpy and maple_syrup (numba / cupy
only for those backends), for example from the candidate root:

    PYTHONPATH="$PWD:/home/okin/SYRUP/src" python examples/phase2/run_disturbance.py \
        --backend numpy --max-bytes 500000000 --out-dir /tmp/leaf-phase2-example

Timeline (all UTC, whole-day records): `--pre-days` open interevent days, a flush, ONE day of wind transport
(`advance_transport('wind', ...)`: native water unchanged), `apply_surface_event` with the synthetic event
(units: depths in m; abrasion DOSE in `kg m-1`, the exact unit declared in the catalog; NO flux is used), the remaining
open days up to the new year, then `apply_annual_decision(year)` (keyed age / background hazard and the persistent
stress counter; the seed is explicit). Compact JSON receipts are printed. It shows executable use only; it is NOT a
qualification and NOT a MAPLE / SYRUP / MASY scenario. Limitations: the event is synthetic; the surface is not
remapped by LEAF; aboveground dry biomass only; the example runs one transport event and one annual decision.
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

from leaf import events, forcing, interevent, state, traits  # noqa: E402
from leaf.initialization import generate_population  # noqa: E402

UNCALIBRATED = ('UNCALIBRATED synthetic example: fixture species traits, synthetic weather / soil / surface event; not a '
                'calibration, not a field simulation, no MAPLE wind or SYRUP water result')


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


def synthetic_event(session, surface_after, t0_days, t1_days):
    '''SYNTHETIC SurfaceEventDiagnostics for the live plants: depths from the establishment-vs-current surface, a
    constant wind abrasion dose in the catalog's declared unit on the western half of the domain. No flux is given.'''
    snap = session.snapshot()
    live = snap.status == state.LIVE
    ids = snap.ids[live]
    delta = snap.domain.sample(surface_after, snap.x_m[live], snap.y_m[live]) - snap.establishment_ground_m[live]
    west = snap.x_m[live] < snap.domain.origin_x_m + 0.5 * snap.domain.width_m
    return events.SurfaceEventDiagnostics(ids, t0_days, t1_days, 'maple_wind', erosion_depth_m=np.maximum(-delta, 0.0),
                                          deposition_depth_m=np.maximum(delta, 0.0),
                                          abrasion_dose=np.where(west, 2.0, 0.0), abrasion_unit='kg m-1',
                                          event_id='synthetic-wind-event-1')


def main(argv=None):
    ap = argparse.ArgumentParser(description=UNCALIBRATED)
    ap.add_argument('--backend', required=True, choices=interevent.BACKENDS)
    ap.add_argument('--max-bytes', required=True, type=int, help='global byte budget of the session (host + device)')
    ap.add_argument('--max-edges', type=int, default=20_000_000)
    ap.add_argument('--reserve-bytes', type=int, default=0, help='bytes the caller holds elsewhere')
    ap.add_argument('--builder', default='numpy', choices=('numpy', 'compiled'), help="root builder ('compiled' needs numba)")
    ap.add_argument('--pre-days', type=int, default=3, help='open interevent days before the wind transport')
    ap.add_argument('--nx', type=int, default=32)
    ap.add_argument('--dx', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=20261111, help='population generation seed')
    ap.add_argument('--rng-seed', type=int, default=424242, help='EXPLICIT seed of the keyed demography randomness')
    ap.add_argument('--start', default='2026-12-26', help='YYYY-MM-DD (local midnight, UTC offset 0)')
    ap.add_argument('--pet-mm-day', type=float, default=5.0, help='SYNTHETIC shared PET')
    ap.add_argument('--out-dir', default=None, help='write the initial / final CSV and the checkpoint here')
    args = ap.parse_args(argv)

    from maple_syrup import subsurface as ss
    from maple_syrup import subsurface_transpiration as st

    catalog = traits.load_catalog(ROOT / 'examples' / 'phase2' / 'species.json')
    ny = args.nx
    surface0 = np.full((ny, args.nx), 100.0)
    plants, _ = generate_population(catalog, generation_config(args.nx, ny, args.dx, args.seed), surface=surface0)
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
        reserve_bytes=args.reserve_bytes, root_builder=args.builder, rng_seed=args.rng_seed)
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

    for _ in range(args.pre_days):
        open_day()
    session.flush_growth()
    wind = session.advance_transport('wind', session.cursor, session.cursor + dt.timedelta(days=1))
    print(json.dumps({'transport': {k: wind[k] for k in ('kind', 'start', 'end', 'native_time_s')}}), flush=True)

    # SYNTHETIC surface after the event: a 0.2 m deposition patch in the south-west quarter, a 0.3 m erosion patch NE
    after = surface0.copy()
    after[: ny // 2, : args.nx // 2] += 0.2
    after[ny // 2:, args.nx // 2:] -= 0.3
    tp = session.transport_pending
    event = synthetic_event(session, after, tp['start_days'], tp['end_days'])
    rec = session.apply_surface_event(event, after)
    print(json.dumps({'disturbance': {k: rec['disturbance'][k] for k in ('n_injured', 'n_dead_standing', 'n_removed')},
                      'ledger': rec['disturbance']['ledger'], 'rebind': rec['rebind'],
                      'water_unchanged': rec['native_water_unchanged']}), flush=True)

    while session.annual_due_year is None:
        open_day()
        if session.cursor.year > start.year + 1:
            raise SystemExit('no year boundary reached: increase --pre-days / choose a December start')
    year = session.annual_due_year
    dec = session.apply_annual_decision(year)
    print(json.dumps({'annual_decision': {k: dec[k] for k in ('year', 'decision_counter', 'n_died_age_background',
                                                              'n_died_stress', 'n_dieback', 'n_live_before',
                                                              'n_live_after')}, 'ledger': dec['ledger']}), flush=True)
    if out is not None:
        state.write_plant_csv(session.snapshot(), out / 'final.csv')
        path = session.save(out / 'checkpoint', overwrite=False)
        restored = interevent.IntereventSession.load(
            path, catalog, operator=operator, backend=args.backend, max_edges=args.max_edges,
            max_bytes=args.max_bytes, reserve_bytes=args.reserve_bytes, root_builder=args.builder)
        print(json.dumps({'checkpoint': str(path), 'restored_cursor': restored.cursor.isoformat(),
                          'restored_decisions': restored.n_annual_decisions, 'rng_seed': restored.rng_seed}), flush=True)
    print(json.dumps({'phase2': session.describe()['phase2'], 'note': UNCALIBRATED}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
