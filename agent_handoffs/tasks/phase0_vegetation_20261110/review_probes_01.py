"""Independent adversarial state/catalog probes; not candidate-authored tests."""
import json
import math
import numpy as np
from leaf.traits import SpeciesCatalog
from leaf.state import Domain, PlantState, plant_csv_text, parse_plant_csv, constant_surface


def catalog():
    return SpeciesCatalog({
        'schema': 'leaf.species_catalog', 'schema_version': 1,
        'units': {'biomass': 'kg', 'length': 'm', 'time': 'days'}, 'root_layers': 2,
        'species': {'BOER': {
            'functional_type': 'grass',
            'allometry': {'model': 'power_law_v1', 'diameter': {'a': 1.5, 'b': 0.5},
                          'height': {'a': 2.0, 'b': 0.5}},
            'roots': {'lateral_radius_multiplier': 4.0, 'layer_fractions': [0.8, 0.2]},
            'porosity': {'value': 0.2, 'status': 'constant fixture'},
            'growth': None, 'water_use': None, 'reproduction': None,
            'life_expectancy': {'mean': 10, 'unit': 'years'},
            'mortality': None, 'disturbance': None,
        }}})


def state(c, *, anchor=3.0, age=2.0):
    return PlantState.from_records(c, Domain(2, 2, 1.0), [{
        'plant_id': 17, 'species': 'BOER', 'x_m': 0.5, 'y_m': 0.5,
        'establishment_ground_m': anchor, 'agb_dry_kg': 0.01,
        'seasonal_agb_dry_kg': 0.002, 'age_days': age, 'status': 'live'}])


def main():
    out = {}
    c = catalog()
    s = state(c)
    restored = parse_plant_csv(plant_csv_text(s), c)
    out['fp64_roundtrip'] = {'pass': all(np.array_equal(getattr(s, k), getattr(restored, k))
                                               for k in s.ARRAY_NAMES)}

    initial = plant_csv_text(state(c, anchor='SURFACE'), initial=True)
    line, body = initial.split('\n', 1)
    meta = json.loads(line[1:]); meta.update(initial=False, time_days=7.0)
    try:
        bad = parse_plant_csv('# '+json.dumps(meta)+'\n'+body, c,
                              surface=constant_surface(s.domain, 55.0))
        out['reject_noninitial_surface'] = {'pass': False,
            'observed': f'accepted and resolved to {bad.establishment_ground_m.tolist()} at t={bad.time_days}'}
    except ValueError as exc:
        out['reject_noninitial_surface'] = {'pass': True, 'error': str(exc)}

    try:
        bad = PlantState(s.domain, c.species_ids, c.hash, ids=[1], species_index=[0],
                         x_m=[0.5], y_m=[0.5], establishment_ground_m=[float('nan')],
                         agb_dry_kg=[0.01], age_days=[2.0], status=['live'])
        out['reject_implicit_nan_surface'] = {'pass': False,
            'observed': f'accepted NaN as {bad.establishment_unresolved.tolist()}'}
    except ValueError as exc:
        out['reject_implicit_nan_surface'] = {'pass': True, 'error': str(exc)}

    s = state(c, age=1e308)
    old_time, old_age = s.time_days, s.age_days.copy()
    try:
        with np.errstate(over='ignore'):
            s.advance_time(1e308)
        out['age_overflow_transaction'] = {'pass': False,
            'observed': {'time': s.time_days, 'age_finite': bool(np.isfinite(s.age_days).all())}}
    except ValueError as exc:
        out['age_overflow_transaction'] = {'pass': s.time_days == old_time and np.array_equal(s.age_days, old_age),
                                           'error': str(exc)}

    s = state(c)
    text = plant_csv_text(s)
    lines = text.splitlines()
    lines[-1] = lines[-1].rsplit(',', 1)[0]+',"live'
    try:
        parse_plant_csv('\n'.join(lines)+'\n', c)
        out['reject_unterminated_quote'] = {'pass': False, 'observed': 'accepted truncated closing quote'}
    except ValueError as exc:
        out['reject_unterminated_quote'] = {'pass': True, 'error': str(exc)}

    big = catalog().document
    big['species']['BOER']['allometry']['diameter']['b'] = 2.0
    bc = SpeciesCatalog(big)
    try:
        with np.errstate(over='ignore'):
            val = bc.diameter_from_biomass('BOER', 1e308)
        out['reject_allometry_overflow'] = {'pass': False, 'observed_finite': math.isfinite(val)}
    except ValueError as exc:
        out['reject_allometry_overflow'] = {'pass': True, 'error': str(exc)}

    print(json.dumps(out, indent=2, allow_nan=False))
    return 0 if all(v['pass'] for v in out.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
