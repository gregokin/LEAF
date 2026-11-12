"""Read-only schema, independent source-arithmetic and preservation checks; no model steps."""
from pathlib import Path
import hashlib
import json
import math
import sys
import tomllib

import numpy as np
from leaf import traits, growth, demography, disturbance, recruitment, initialization, state

ROOT = Path('/home/okin/leaf')
TASK = Path(__file__).resolve().parent
CASE_REL = Path('cases/jornada_grass05_shrub10_256')
base = Path(sys.argv[1]).resolve()
case = base / CASE_REL
baseline = json.loads((TASK / 'baseline.json').read_text())['source_manifest']
checks = []


def check(name, ok, evidence):
    if not ok:
        raise AssertionError(f'{name}: {evidence}')
    checks.append({'name': name, 'pass': True, 'evidence': evidence})


catalog = traits.load_catalog(case / 'species.json')
check('active native catalog', catalog.species_ids == ('BOER', 'PROGLA'),
      {'id': catalog.document['catalog_id'], 'hash': catalog.hash})
config = json.loads((case / 'generation.json').read_text())
gen = initialization.validate_generation_config(config, catalog)
check('generation schema and user age rule',
      [v['age_sampling'] for v in gen['species']] == [
          {'distribution': 'uniform', 'min_days': 0.0, 'max_days': 5475.0},
          {'distribution': 'uniform', 'min_days': 0.0, 'max_days': 29200.0}],
      {v['id']: v['age_sampling'] for v in gen['species']})
# This only validates two small parameter tables; no populations, water or growth are advanced.
growth.compile_species(catalog, 'numpy', 1024 * 1024)
demography.compile_species(catalog, 'numpy', max_bytes=1024 * 1024)
disturbance.compile_species(catalog, 'numpy', max_bytes=1024 * 1024)
recruitment.compile_species(catalog, max_bytes=1024 * 1024)
check('all native trait consumers', True, 'growth/demand, demography, mechanical response and recruitment tables parse on NumPy')
sp = catalog.document['species']
source_C = math.exp(6.960) / 1000.0
geometry = []
for diameter_m in (0.5, 1.0, 2.0, 4.0):
    area = math.pi * diameter_m ** 2 / 4.0
    source_B = source_C * area ** 1.492
    source_H = math.exp(0.11) * area ** 0.38
    actual_D = float(catalog.diameter_from_biomass('PROGLA', source_B))
    actual_H = float(catalog.height_from_biomass('PROGLA', source_B))
    inverse_B = float(catalog.biomass_from_diameter('PROGLA', diameter_m))
    check(f'source-equation geometry D={diameter_m}m',
          math.isclose(actual_D, diameter_m, rel_tol=2e-13) and
          math.isclose(actual_H, source_H, rel_tol=2e-13) and
          math.isclose(inverse_B, source_B, rel_tol=2e-13),
          {'source_B_kg_dry_AGB': source_B, 'source_H_m': source_H,
           'native_D_m': actual_D, 'native_H_m': actual_H})
    geometry.append({'diameter_m': diameter_m, 'agb_dry_kg': source_B, 'height_m': source_H})
check('target-species live fraction', sp['PROGLA']['growth']['traits']['live_fraction'] == 1 - 0.094,
      'Northup 2005 Table 1: complement of mean standing-dead dry mass fraction 9.4%; includes live wood')
for sid, rec in sp.items():
    gr = rec['growth']['traits']; q = rec['water_use']['traits']['demand_m3_per_kg_day']
    check(f'{sid} support coefficient',
          math.isclose(gr['biomass_support_kg_year_per_m3'], gr['live_fraction'] / (365 * q), rel_tol=1e-14),
          {'live_fraction': gr['live_fraction'], 'q_m3_per_kg_day': q,
           'k_kg_year_per_m3': gr['biomass_support_kg_year_per_m3']})
    check(f'{sid} life expectancy/Weibull consistency',
          math.isclose(rec['mortality']['traits']['age_hazard']['weibull_scale_years'] * math.gamma(1.5),
                       rec['life_expectancy']['mean'], rel_tol=1e-13),
          'Unchanged assumed shape 2 and mean; generation still Uniform(0, catalog life expectancy)')
check('STEP blue-grama growth prior', sp['BOER']['growth']['traits']['intrinsic_rate_per_year'] == .474,
      'Pinned STEPWAT2 species.in bogr irate .474; model-derived related-species prior')
cl = sp['BOER']['reproduction']['traits']['clonal']
check('STEP unit-size proxy and donation affordability',
      cl['initial_agb_dry_kg'] == 1.5096 / 1000 and
      math.isclose(cl['min_agb_dry_kg'] * cl['max_donation_fraction'], cl['initial_agb_dry_kg'], rel_tol=1e-15),
      {'daughter_kg_dry_AGB': cl['initial_agb_dry_kg'], 'min_parent_kg_dry_AGB': cl['min_agb_dry_kg'],
       'max_donation_fraction': cl['max_donation_fraction']})
soil = tomllib.loads((case / 'soil.toml').read_text())
s = soil['soil_layers']
check('FAO soil prior and storage ordering',
      s['theta_field_capacity'] == .23 and
      s['theta_residual'] < s['theta_evaporation_stress'] < s['theta_field_capacity'] < s['theta_saturation'] and
      all(s['theta_residual'] <= theta <= s['theta_field_capacity'] for theta in soil['initial_state']['theta_initial']),
      'FAO-56 Table 19 sandy-loam FC range .18-.28; Example 35 .23. Other soil coefficients remain independent priors.')
for p in case.rglob('*'):
    if p.suffix == '.json':
        traits.parse_strict_json(p.read_text(), what=str(p.relative_to(case)))
    if p.suffix == '.toml':
        tomllib.loads(p.read_text())
check('all scenario JSON/TOML parse', True, 'Strict JSON refuses duplicate keys/NaN; stdlib TOML syntax checked')
frozen_file = case / 'preview/species_at_initialization.json'
frozen = traits.load_catalog(frozen_file)
check('frozen exact baseline catalog',
      hashlib.sha256(frozen_file.read_bytes()).hexdigest() == baseline[str(CASE_REL / 'species.json')]['sha256'],
      {'frozen_hash': frozen.hash, 'active_hash': catalog.hash})
plants = state.read_plant_csv(case / 'preview/initial_plants.csv', frozen)
check('historical plain CSV companion readability', len(plants.ids) == 25589,
      {'plant_records': len(plants.ids), 'catalog_hash': plants.catalog_hash})
try:
    state.read_plant_csv(case / 'preview/initial_plants.csv', catalog)
except state.StateError as exc:
    check('historical data reject active-catalog reinterpretation', 'catalog' in str(exc).lower(), str(exc))
else:
    raise AssertionError('New catalog silently rebound historical CSV')
immutable = []
for rel, item in baseline.items():
    path = Path(rel)
    if (str(path).startswith(str(CASE_REL / 'preview') + '/') and path.name not in {'README.md'}) or \
            str(path).startswith('leaf/') or rel in {'AGENTS.md', 'claude.md', 'agent_handoffs/AGENTS.md'} or \
            (str(path).startswith(str(CASE_REL) + '/') and path.name not in {'species.json','soil.toml','README.md','PARAMETERS.md','SOURCES.md'}):
        actual = base / rel
        if not actual.exists():
            actual = ROOT / rel
        check('preserve ' + rel, hashlib.sha256(actual.read_bytes()).hexdigest() == item['sha256'], item['sha256'])
        immutable.append(rel)
# Require prose source columns in every quantitative science table and retain explicit unknowns.
parameters = (case / 'PARAMETERS.md').read_text()
for url in ['10.1016/j.jaridenv.2004.09.019', '10.1016/j.jaridenv.2012.08.015', '10.1002/ecy.70384',
            '10.1111/nph.18909', '10.1007/s00442-004-1688-y', '10.1029/WR024i005p00755',
            'x0490e0c.htm', '003cb9b4adbf13d64a39d6d62fd95280adac27fc']:
    check('parameter table contains source ' + url, url in parameters, url)
check('no production plants/surface or model output created',
      not (case / 'inputs/plants.csv').exists() and not (case / 'inputs/surface.npy').exists() and
      sorted(p.name for p in (case / 'outputs').iterdir()) == ['README.md'],
      'Historical preview only; production initialization, forward run and rendering not performed')
result = {'base': str(base), 'catalog_hash': catalog.hash, 'checks': checks, 'check_count': len(checks),
          'all_passed': True, 'geometry_at_reference_diameters': geometry,
          'no_model_execution': True, 'no_population_generation': True, 'no_rendering': True,
          'preserved_files': immutable}
print(json.dumps(result, indent=2))
