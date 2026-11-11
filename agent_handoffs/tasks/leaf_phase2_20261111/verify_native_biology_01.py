"""Independent actual-native mortality, root, age, replay and restart checks."""
import datetime as dt
import json
from pathlib import Path
import tempfile

import numpy as np
from leaf import events, forcing, state, traits
from leaf.interevent import IntereventSession
from maple_syrup import subsurface as ss
from maple_syrup import subsurface_transpiration as st


def make(backend='numba'):
    raw = json.loads(Path('examples/phase1/species.json').read_text())
    for sid, item in raw['species'].items():
        item['disturbance'] = {'law': 'mechanical_response_v1', 'traits': {
            'burial': {'increment_basis': 'net', 'scale': 'm', 'injury_threshold': .01,
                       'injury_saturation': .5, 'max_injury_fraction': .2, 'curve': 'linear',
                       'lethal_depth_m': .2 if sid == 'grass' else .5, 'lethal_height_fraction': None},
            'exposure': {'increment_basis': 'net', 'injury_threshold_m': .01, 'injury_saturation_m': .2,
                         'max_injury_fraction': .2, 'curve': 'linear', 'lethal_depth_m': None,
                         'dislodgement_depth_m': .3}, 'abrasion': None}}
        item['mortality'] = {'law': 'age_background_stress_v1', 'traits': {
            'age_hazard': None, 'background_hazard_per_year': 0., 'stress': {
                'metric': 'support_ratio', 'low_threshold': .5, 'exempt_until_age_days': 365.,
                'limit_years': 2, 'counter_cap_years': 10, 'mortality_probability': 0.,
                'biomass_loss_fraction': .1}}}
    cat = traits.SpeciesCatalog(raw)
    records = [dict(plant_id=i + 1, species=sid, x_m=.5 + i, y_m=.5, establishment_ground_m=0.,
                    agb_dry_kg=b, seasonal_agb_dry_kg=e, age_days=1000., status='live')
               for i, (sid, b, e) in enumerate([('grass', .1, .02), ('shrub', 2., .4)])]
    plants = state.PlantState.from_records(cat, state.Domain(2, 1, 1.), records)
    xp = np
    if backend == 'cuda':
        import cupy as xp
    layers = ss.soil_layers([0., .25, 1.], theta_residual=.05, theta_field_capacity=.3,
                            theta_saturation=.5, diffusivity_m2_s=0., theta_evaporation_stress=.1)
    col = ss.prepare_column(layers, xp.ones(2, dtype=bool), 1., backend=backend)
    op = st.prepare_transpiration(col)
    water = ss.column_state(col, xp.asarray([[.05, .05], [.15, .15]], dtype=np.float64))
    start = dt.datetime(2024, 12, 30, tzinfo=dt.timezone.utc)
    session = IntereventSession(plants, cat, operator=op, native_state=water, start=start, backend=backend,
                                growth_cadence=dt.timedelta(days=7), max_edges=1000,
                                max_bytes=100_000_000, root_builder='compiled', rng_seed=123)
    return session, cat, op, xp


def host(a):
    return a.get() if hasattr(a, 'get') else np.asarray(a)


def main():
    import sys
    backend = sys.argv[1] if len(sys.argv) > 1 else 'numba'
    reports = []
    for remove_shrub in [False, True]:
        session, cat, op, xp = make(backend)
        start = session.cursor
        session.advance_transport('wind', start, start + dt.timedelta(days=1), interval_id='wind')
        stock = host(session.native_state.water_m).copy()
        event = events.SurfaceEventDiagnostics(np.array([1, 2], np.int64), 0., 1., 'maple_wind',
                                               deposition_depth_m=[.3, 0.],
                                               erosion_depth_m=[0., .4 if remove_shrub else 0.], event_id='damage')
        surface = np.array([[.3, -.4 if remove_shrub else 0.]])
        receipt = session.apply_surface_event(event, surface, interval_id='response')
        snap = session.snapshot()
        np.testing.assert_array_equal(host(session.native_state.water_m), stock)
        np.testing.assert_array_equal(snap.status, [state.DEAD_STANDING, state.REMOVED if remove_shrub else state.LIVE])
        np.testing.assert_array_equal(snap.establishment_ground_m, [0., 0.])
        np.testing.assert_array_equal(snap.age_days, [1001., 1001.])
        np.testing.assert_array_equal(snap.agb_dry_kg, [.1, 0. if remove_shrub else 2.])
        if remove_shrub:
            assert session.root_map is None
        else:
            np.testing.assert_array_equal(host(session.root_map.ids), [2])
        with tempfile.TemporaryDirectory(prefix='leaf-phase2-independent-') as temp:
            saved = session.save(Path(temp) / 'restart')
            restored = IntereventSession.load(saved, cat, operator=op, backend=backend,
                                               max_edges=1000, max_bytes=100_000_000, root_builder='compiled')
            for run in [session, restored]:
                try:
                    run.apply_surface_event(event, surface)
                except ValueError:
                    pass
                else:
                    raise AssertionError('duplicate injury accepted')
                rec = forcing.TimestepRecord(run.cursor, run.cursor + dt.timedelta(days=1), 25., pet_mm_per_day=5.)
                daily = run.step_interevent(rec, added_water_m=xp.zeros((2, 2)), potential_evaporation_m_s=xp.zeros(2))
                if remove_shrub:
                    assert daily['transpiration'] is None
                else:
                    assert daily['transpiration']['native_budget']['actual_m3'] > 0.
                before_annual = host(run.native_state.water_m).copy()
                run.apply_annual_decision(2024)
                np.testing.assert_array_equal(host(run.native_state.water_m), before_annual)
                np.testing.assert_array_equal(run.snapshot().age_days, [1001., 1001. if remove_shrub else 1002.])
            for field in snap._arrays():
                np.testing.assert_array_equal(session.snapshot()._arrays()[field], restored.snapshot()._arrays()[field])
            np.testing.assert_array_equal(host(session.native_state.water_m), host(restored.native_state.water_m))
        reports.append({'all_live_plants_removed': remove_shrub, 'restart_exact': True,
                        'dead_roots_no_uptake': True, 'dead_ages_frozen': True,
                        'biological_operations_preserve_native_water': True})
    print(json.dumps({'backend': backend, 'cases': reports}, indent=2))


if __name__ == '__main__':
    main()
