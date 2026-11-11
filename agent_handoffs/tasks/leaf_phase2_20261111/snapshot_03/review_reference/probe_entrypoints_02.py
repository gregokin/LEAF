"""Independent scientific input/admission probes; synthetic records only."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from leaf import demography, disturbance, events, state, traits


def catalog():
    raw = json.loads(Path('examples/phase1/species.json').read_text())
    for item in raw['species'].values():
        item['disturbance'] = {'law': 'mechanical_response_v1', 'traits': {
            'burial': None, 'exposure': None, 'abrasion': None}}
        item['mortality'] = {'law': 'age_background_stress_v1', 'traits': {
            'age_hazard': None, 'background_hazard_per_year': 0., 'stress': {
                'metric': 'support_ratio', 'low_threshold': .5, 'exempt_until_age_days': 0.,
                'limit_years': 2, 'counter_cap_years': 10, 'mortality_probability': 0.,
                'biomass_loss_fraction': .1}}}
    return traits.SpeciesCatalog(raw)


def main():
    cat = catalog()
    results = []
    for backend in ['numpy', 'numba']:
        table = demography.compile_species(cat, backend)
        nominal = dict(species_index=np.array([0, 1], np.int32), plant_ids=np.array([1, 2], np.int64),
                       age_end_days=np.full(2, 1000.), stress=np.zeros(2, np.int32),
                       required_support=np.ones(2), actual_support=np.ones(2),
                       planned_growth=np.ones(2), actual_growth=np.ones(2),
                       period_days=365., seed=123, counter=0, max_bytes=10_000_000)
        demography.decide(table, **nominal)
        variants = [('negative_species_index', 'species_index', np.array([-1, 0], np.int32)),
                    ('nan_required_support', 'required_support', np.array([np.nan, 1.])),
                    ('negative_actual_support', 'actual_support', np.array([-1., 1.])),
                    ('negative_stress_counter', 'stress', np.array([-2, 0], np.int32)),
                    ('actual_growth_exceeds_planned', 'actual_growth', np.array([2., 1.]))]
        for name, key, value in variants:
            kw = copy.deepcopy(nominal)
            kw[key] = value
            try:
                demography.decide(table, **kw)
                refused, error = False, None
            except demography.DemographyError as exc:
                refused, error = True, str(exc)
            except Exception as exc:
                refused, error = False, f'{type(exc).__name__}: {exc} (secondary failure, not pre-kernel validation)'
            results.append(dict(name=name, backend=backend, refused=refused, error=error))
    records = [dict(plant_id=i + 1, species=sid, x_m=.5, y_m=.5, establishment_ground_m=0.,
                    agb_dry_kg=.1, seasonal_agb_dry_kg=0., age_days=1000., status='live')
               for i, sid in enumerate(cat.species_ids)]
    plants = state.PlantState.from_records(cat, state.Domain(1, 1, 1.), records)
    event = events.SurfaceEventDiagnostics(np.zeros(0, np.int64), 0., 0., 'other', event_id='empty')
    copies = []
    original = state.PlantState.copy
    def counted_copy(obj):
        copies.append(obj.n)
        return original(obj)
    with patch.object(state.PlantState, 'copy', counted_copy):
        try:
            disturbance.respond_state(plants, cat, event, 0., max_bytes=1)
            refused, error = False, None
        except disturbance.DisturbanceBudgetError as exc:
            refused, error = True, str(exc)
    results.append(dict(name='state_response_global_admission', backend='numpy', refused=refused,
                        population_copies=copies, error=error))
    print(json.dumps({'probes': results, 'all_refused': all(x['refused'] for x in results)}, indent=2))
    return 0 if all(x['refused'] for x in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
