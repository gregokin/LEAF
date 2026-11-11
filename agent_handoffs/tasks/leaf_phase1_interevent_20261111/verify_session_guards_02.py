"""Independent adversarial session guards; actual native water, tiny fixtures."""
import datetime as dt
from dataclasses import replace
import json
import math
from pathlib import Path
from unittest import mock

import numpy as np

from leaf import forcing, state, traits
from leaf.interevent import IntereventSession, IntereventError
from maple_syrup import subsurface as ss
from maple_syrup import subsurface_transpiration as st
from early_session_smoke import fixture


def create(plants, catalog, operator, native, start, **kwargs):
    options = dict(operator=operator, native_state=native, start=start, backend='numba',
                   growth_cadence=dt.timedelta(days=7), max_edges=100_000, max_bytes=1 << 28)
    options.update(kwargs)
    return IntereventSession(plants, catalog, **options)


def main():
    base, catalog, operator = fixture()
    plants = base.snapshot()
    reports = []
    copies = []
    real_copy = state.PlantState.copy

    def observed_copy(self):
        copies.append(self.n)
        return real_copy(self)

    with mock.patch.object(state.PlantState, 'copy', new=observed_copy):
        try:
            create(plants, catalog, operator, base.native_state, base.cursor, max_bytes=0)
        except ValueError as exc:
            reason = str(exc)
        else:
            raise AssertionError('Zero-byte session accepted')
    reports.append({'case': 'zero-byte constructor before copy', 'passed': not copies,
                    'population_copies_before_refusal': copies, 'reason': reason})

    for label, overrides in [('bool_id', {True: [.5, .5]}),
                             ('fractional_id', {1.9: [.5, .5]}),
                             ('id_alias_collision', {1: [.85, .15], 1.9: [.1, .9]}),
                             ('bool_fractions', {1: [True, False]})]:
        try:
            answer = create(plants, catalog, operator, base.native_state, base.cursor,
                            layer_fraction_overrides=overrides)
        except ValueError as exc:
            reports.append({'case': label, 'passed': True, 'rejected': str(exc)})
        else:
            reports.append({'case': label, 'passed': False, 'coerced_to': answer._overrides})

    late = create(plants, catalog, operator,
                  ss.SoilColumnState(base.native_state.water_m.copy(), 1e9), base.cursor)
    end = late.cursor + dt.timedelta(days=1)
    candidate = ss.SoilColumnState(late.native_state.water_m.copy(), 1e9 + 86400.5)
    try:
        late.advance_transport('rain', late.cursor, end, native_state=candidate)
    except IntereventError as exc:
        reports.append({'case': 'late native clock off by half second', 'passed': True, 'rejected': str(exc)})
    else:
        reports.append({'case': 'late native clock off by half second', 'passed': False,
                        'accepted_time_error_s': late.native_state.t_s - (1e9 + 86400)})

    document = catalog.document
    grass = document['species']['grass']
    grass['growth']['traits'].update(monthly_activity=[1.] * 12, temperature_min_c=0.,
                                     temperature_opt_c=25., temperature_max_c=50., mature_biomass_kg=2.,
                                     biomass_support_kg_year_per_m3=1e-305)
    grass['roots']['layer_fractions'] = [.5, .5]
    grass['water_use'] = {'law': 'biomass_specific_pet_v1',
                         'traits': {'demand_m3_per_kg_day': 1e308, 'reference_pet_mm_per_day': 1.}}
    huge_catalog = traits.SpeciesCatalog(document)
    domain = state.Domain(1, 1, 1.)
    huge_plants = state.PlantState.from_records(huge_catalog, domain, [
        dict(plant_id=1, species='grass', x_m=.5, y_m=.5, establishment_ground_m=0.,
             agb_dry_kg=1., seasonal_agb_dry_kg=0., age_days=100., status='live')])
    layers = ss.soil_layers([0., .25, 1.], theta_residual=.05, theta_field_capacity=.3,
                            theta_saturation=.5, diffusivity_m2_s=0., theta_evaporation_stress=.1)
    column = ss.prepare_column(layers, np.ones(1, dtype=bool), 1., backend='numba')
    op = st.prepare_transpiration(column)
    native = ss.column_state(column, np.array([[.05], [.15]]))
    huge = create(huge_plants, huge_catalog, op, native, base.cursor)
    prior = None
    for index in range(2):
        rec = forcing.TimestepRecord(huge.cursor, huge.cursor + dt.timedelta(days=1), 25., pet_mm_per_day=1.)
        old_cursor = huge.cursor
        old_D = huge._b.D.copy()
        try:
            huge.step_interevent(rec, added_water_m=np.zeros((2, 1)),
                                 potential_evaporation_m_s=np.zeros(1))
        except (ValueError, ArithmeticError) as exc:
            if index == 0:
                reports.append({'case': 'pending demand overflow', 'passed': False,
                                'inconclusive_first_step_refused': str(exc)})
            else:
                intact = huge.cursor == old_cursor and np.array_equal(huge._b.D, old_D)
                reports.append({'case': 'pending demand overflow', 'passed': bool(intact),
                                'rejected': str(exc), 'boundary_preserved': bool(intact)})
            break
        prior = index
    else:
        finite = bool(np.all(np.isfinite(huge._b.D)))
        reports.append({'case': 'pending demand overflow', 'passed': finite,
                        'finite_pending_demand': finite,
                        'published_pending_demand': huge._b.D.tolist(),
                        'published_cursor': huge.cursor.isoformat()})
    clean, _, _ = fixture()
    old_boundary = clean._b
    real_step = ss.step
    calls = []

    def misreported_evaporation(*args, **kwargs):
        calls.append(1)
        result = real_step(*args, **kwargs)
        return replace(result, budget=replace(result.budget,
                                              evaporation_m3=result.budget.evaporation_m3 + .001))

    record = forcing.TimestepRecord(clean.cursor, clean.cursor + dt.timedelta(days=1),
                                    25., pet_mm_per_day=3.)
    with mock.patch.object(ss, 'step', new=misreported_evaporation):
        try:
            receipt = clean.step_interevent(record, added_water_m=np.zeros((2, 4)),
                                             potential_evaporation_m_s=np.zeros(4))
        except ValueError as exc:
            reports.append({'case': 'enforce end-to-end water closure',
                            'passed': clean._b is old_boundary, 'rejected': str(exc),
                            'boundary_preserved': clean._b is old_boundary, 'real_native_calls': len(calls)})
        else:
            reports.append({'case': 'enforce end-to-end water closure', 'passed': False,
                            'reported_closed': receipt['water_closure']['residual_within_tolerance'],
                            'reported_residual_m3': receipt['water_closure']['residual_m3'],
                            'published_failed_boundary': clean._b is not old_boundary,
                            'real_native_calls': len(calls)})

    failures = sum(not row['passed'] for row in reports)
    print(json.dumps({'scope': 'independent tiny native session guard fixtures',
                      'results': reports, 'failures': failures}, indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
