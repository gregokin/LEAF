"""Independent early dated native-column fixture; no full transport/model run."""
import datetime as dt
import json
from pathlib import Path
import tempfile

import numpy as np

from leaf import forcing, state, traits
from leaf.interevent import IntereventSession
from maple_syrup import subsurface as ss
from maple_syrup import subsurface_transpiration as st


def fixture():
    catalog = traits.SpeciesCatalog.from_file(Path.cwd() / 'examples/phase1/species.json')
    domain = state.Domain(2, 2, 1.)
    records = [dict(plant_id=1, species='grass', x_m=.5, y_m=.5,
                    establishment_ground_m=0., agb_dry_kg=.05, seasonal_agb_dry_kg=.005,
                    age_days=100., status='live'),
               dict(plant_id=2, species='shrub', x_m=1.5, y_m=1.5,
                    establishment_ground_m=0., agb_dry_kg=2., seasonal_agb_dry_kg=.1,
                    age_days=1000., status='live')]
    plants = state.PlantState.from_records(catalog, domain, records)
    layers = ss.soil_layers([0., .25, 1.], theta_residual=.05,
                            theta_field_capacity=.3, theta_saturation=.5,
                            diffusivity_m2_s=0., theta_evaporation_stress=.1)
    column = ss.prepare_column(layers, np.ones(4, dtype=bool), 1., backend='numba')
    operator = st.prepare_transpiration(column)
    water = np.broadcast_to(np.array([[.05], [.15]]), (2, 4)).copy()
    native = ss.column_state(column, water, t_s=10.25)
    start = dt.datetime(2024, 2, 27, tzinfo=dt.timezone.utc)
    session = IntereventSession(plants, catalog, operator=operator, native_state=native,
                                start=start, backend='numba', growth_cadence=dt.timedelta(days=7),
                                max_edges=100_000, max_bytes=1 << 28)
    return session, catalog, operator


def advance_day(session):
    rec = forcing.TimestepRecord(session.cursor, session.cursor + dt.timedelta(days=1),
                                 25., pet_mm_per_day=3.)
    receipt = session.step_interevent(rec, added_water_m=np.zeros((2, 4)),
                                      potential_evaporation_m_s=np.zeros(4))
    if receipt.get('water_closure'):
        assert receipt['water_closure']['residual_within_tolerance']
    return receipt


def main():
    session, catalog, operator = fixture()
    first = []
    for _ in range(3):
        first.append(advance_day(session))
    assert session.cursor.date() == dt.date(2024, 3, 1)
    with tempfile.TemporaryDirectory(prefix='leaf_dated_smoke_') as td:
        destination = Path(td) / 'restart'
        session.save(destination)
        restored = IntereventSession.load(destination, catalog, operator=operator, backend='numba',
                                          max_edges=100_000, max_bytes=1 << 28)
        for _ in range(7):
            advance_day(session)
            advance_day(restored)
        a, b = session.snapshot(), restored.snapshot()
        for name in state.PlantState.ARRAY_NAMES:
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        np.testing.assert_array_equal(session.native_state.water_m, restored.native_state.water_m)
        assert session.cursor == restored.cursor
        assert a.time_days == b.time_days
    print(json.dumps({'scope': 'early independent actual-native interevent fixture',
                      'days': 10, 'crossed_leap_day': True, 'restart_at_day': 3,
                      'restart_equivalent': True, 'final_cursor': session.cursor.isoformat(),
                      'native_time_s': session.native_state.t_s,
                      'agb_dry_kg': a.agb_dry_kg.tolist(), 'age_days': a.age_days.tolist(),
                      'whole_phase_accepted': False}, indent=2))


if __name__ == '__main__':
    main()
