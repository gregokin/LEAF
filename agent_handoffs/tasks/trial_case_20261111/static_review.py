"""Static input review only. Never imports or calls any scientific model module."""
from __future__ import annotations

import ast
import calendar
import csv
import datetime as dt
from decimal import Decimal
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import tomllib

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[2]
CANDIDATE = TASK / 'candidate'
CASE_REL = Path('cases/jornada_grass05_shrub10_256')
CASE = CANDIDATE / CASE_REL
ERRORS: list[str] = []
CHECKS: list[str] = []


def check(condition, message):
    (CHECKS if condition else ERRORS).append(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key {key}')
        result[key] = value
    return result


def read_json(path):
    def invalid(value):
        raise ValueError(f'nonfinite JSON value {value}')
    return json.loads(path.read_text(), object_pairs_hook=no_duplicates, parse_constant=invalid)


def keyset(value, expected, label):
    check(set(value) == set(expected), label + ': exact keys')


def constants(path):
    found = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                found[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return found


baseline = read_json(TASK / 'baseline.json')
allowed = set(read_json(TASK / 'launch_02.json')['write_files'])
context = read_json(TASK / 'source-manifest.json')
inventory = {}
for path in CANDIDATE.rglob('*'):
    if not path.is_file():
        continue
    rel = str(path.relative_to(CANDIDATE))
    inventory[rel] = {'sha256': digest(path), 'size_bytes': path.stat().st_size}
    check(rel in allowed or rel in context or rel == 'SOURCE_INDEX.txt', 'candidate scope: ' + rel)
for rel, identity in context.items():
    check(inventory.get(rel, {}).get('sha256') == identity['sha256'], 'immutable context: ' + rel)
for rel in allowed:
    check((CANDIDATE / rel).is_file(), 'required output: ' + rel)
check(len(list(CASE.rglob('*'))) > 0, 'case exists')
check(not (CASE / 'inputs/surface.npy').exists(), 'terrain not generated')
check(not (CASE / 'inputs/plants.csv').exists(), 'population not generated')
check(all(p.name == 'README.md' for p in (CASE / 'outputs').rglob('*') if p.is_file()), 'no simulation outputs')

documents = {}
for path in CASE.rglob('*'):
    if not path.is_file():
        continue
    content = path.read_text()
    check(content.endswith('\n') and '\r' not in content, 'LF and final newline: ' + path.name)
    check(all(line == line.rstrip() for line in content.splitlines()), 'no trailing whitespace: ' + path.name)
    if path.suffix == '.json':
        documents[str(path.relative_to(CASE))] = read_json(path)
    elif path.suffix == '.toml':
        documents[str(path.relative_to(CASE))] = tomllib.loads(content)
    elif path.suffix == '.csv':
        rows = list(csv.reader(io.StringIO(content), strict=True))
        check(all(len(row) == len(rows[0]) for row in rows), 'CSV widths: ' + path.name)

s = documents['species.json']
g = documents['generation.json']
case = documents['case.toml']
soil = documents['soil.toml']
land = documents['landscape.toml']
schedule = documents['scheduler/settings.toml']
trait_keys = constants(ROOT / 'leaf/traits.py')
generation_keys = constants(ROOT / 'leaf/initialization.py')
growth_keys = constants(ROOT / 'leaf/growth.py')
keyset(s, trait_keys['TOP_LEVEL_KEYS'], 'catalog')
check(s['units'] == trait_keys['REQUIRED_UNITS'], 'native catalog units')
check(s['schema'] == 'leaf.species_catalog' and s['schema_version'] == 1, 'catalog schema')
check(s['root_layers'] == 3 and set(s['species']) == {'BOER', 'PROGLA'}, 'two species / three layers')
check(set(g) <= set(generation_keys['TOP_KEYS']), 'generation top-level keys')
check(g['schema'] == 'leaf.generation_config' and g['schema_version'] == 1, 'generation schema')
check(g['domain'] == {'nx': 256, 'ny': 256, 'dx_m': .25, 'dy_m': .25,
                      'origin_x_m': 0., 'origin_y_m': 0.}, 'reference grid')
check(g['overlap'] == 'disallowed' and g['cover_definition'] == 'per_species_union', 'initial placement choices')
check(g['limits']['max_cover_cells'] == (256 * 8) ** 2, 'cover raster cell cap')
check(g['limits']['max_cover_bytes'] == 2 * (256 * 8) ** 2, 'cover raster byte cap')

parameter_arithmetic = {}
for sid, beta, mean, sd, cover in [('BOER', .977, .1, .02, .05), ('PROGLA', .983, 1., .1, .1)]:
    rec = s['species'][sid]
    check(set(trait_keys['REQUIRED_SECTIONS']) <= set(rec), sid + ' required trait sections')
    check(set(rec) <= set(trait_keys['REQUIRED_SECTIONS'] + trait_keys['OPTIONAL_SPECIES_KEYS']), sid + ' no unknown sections')
    for section in ['growth', 'water_use', 'mortality', 'disturbance', 'reproduction']:
        check(set(rec[section]) <= set(growth_keys['_SECTION_KEYS']), sid + ' ' + section + ' section keys')
    gr = rec['growth']['traits']
    keyset(gr, growth_keys['GROWTH_TRAITS'], sid + ' growth traits')
    check(rec['growth']['law'] == 'stepwat_requested_support_v1', sid + ' growth law')
    check(rec['water_use']['law'] == 'biomass_specific_pet_v1', sid + ' demand law')
    keyset(rec['water_use']['traits'], ['demand_m3_per_kg_day', 'reference_pet_mm_per_day'], sid + ' water traits')
    q = rec['water_use']['traits']['demand_m3_per_kg_day']
    check(math.isclose(gr['biomass_support_kg_year_per_m3'], gr['live_fraction'] / (365 * q), rel_tol=1e-14), sid + ' support prior arithmetic')
    check(0 <= gr['intrinsic_rate_per_year'] <= 1 and 0 <= gr['extra_growth_rate_per_year'] <= 1, sid + ' admitted rates')
    check(len(gr['monthly_activity']) == 12 and all(0 <= a <= 1 for a in gr['monthly_activity']), sid + ' activity fractions')
    check(gr['temperature_min_c'] < gr['temperature_opt_c'] < gr['temperature_max_c'], sid + ' cardinal temperatures')
    depths = [0, 20, 50, 120]
    expected = [(beta ** depths[i] - beta ** depths[i + 1]) / (1 - beta ** 120) for i in range(3)]
    actual = rec['roots']['layer_fractions']
    check(len(actual) == 3 and math.isclose(sum(actual), 1, abs_tol=1e-14), sid + ' root fraction sum')
    check(all(math.isclose(a, b, rel_tol=1e-14) for a, b in zip(actual, expected)), sid + ' root profile integration')
    check(rec['roots']['lateral_radius_multiplier'] == (3. if sid == 'BOER' else 5.), sid + ' root radius range')
    dm = rec['mortality']['traits']
    dm_keys = constants(ROOT / 'leaf/demography.py')
    keyset(dm, dm_keys['_TRAIT_KEYS'], sid + ' mortality traits')
    keyset(dm['age_hazard'], dm_keys['_AGE_KEYS'], sid + ' age traits')
    keyset(dm['stress'], dm_keys['_STRESS_KEYS'], sid + ' stress traits')
    age = dm['age_hazard']
    check(math.isclose(age['weibull_scale_years'] * math.gamma(1 + 1 / age['weibull_shape']), rec['life_expectancy']['mean'], rel_tol=1e-14), sid + ' Weibull scale derivation')
    dk = constants(ROOT / 'leaf/disturbance.py')
    di = rec['disturbance']['traits']
    keyset(di, dk['_TRAIT_KEYS'], sid + ' disturbance traits')
    keyset(di['burial'], dk['_BURIAL_KEYS'], sid + ' burial traits')
    keyset(di['exposure'], dk['_EXPOSURE_KEYS'], sid + ' exposure traits')
    check(di['abrasion'] is None, sid + ' abrasion inactive')
    rk = constants(ROOT / 'leaf/recruitment.py')
    reproduction = rec['reproduction']['traits']
    keyset(reproduction, ['seed', 'clonal'], sid + ' reproduction pathways')
    for pathway in ['seed', 'clonal']:
        if reproduction[pathway] is not None:
            keyset(reproduction[pathway], rk['_' + pathway.upper() + '_KEYS'], sid + ' ' + pathway + ' traits')
    gen = g['species'][sid]
    check((gen['diameter_mean_m'], gen['diameter_std_m'], gen['target_cover']) == (mean, sd, cover), sid + ' user targets')
    check(set(gen) <= set(generation_keys['SPECIES_KEYS']), sid + ' generation species keys')
    parameter_arithmetic[sid] = {'root_sum': sum(actual), 'support_kg_year_per_m3': gr['biomass_support_kg_year_per_m3'],
                               'expected_initial_count_estimate': 4096 * cover / (math.pi / 4 * (mean * mean + sd * sd))}

ss_ast = ast.parse((ROOT.parent / 'SYRUP/src/maple_syrup/subsurface.py').read_text())
ss_fn = next(n for n in ss_ast.body if isinstance(n, ast.FunctionDef) and n.name == 'soil_layers')
ss_args = [n.arg for n in ss_fn.args.args + ss_fn.args.kwonlyargs]
keyset(soil['soil_layers'], ss_args, 'soil-layer table public arguments')
depth = soil['soil_layers']['depth_edges_m']
check(depth == [0., .2, .5, 1.2], 'soil / root layer boundaries')
th = soil['soil_layers']
check(th['theta_residual'] < th['theta_field_capacity'] < th['theta_saturation'], 'soil retention ordering')
check(len(soil['initial_state']['theta_initial']) == 3 and all(th['theta_residual'] <= v <= th['theta_field_capacity'] for v in soil['initial_state']['theta_initial']), 'antecedent moisture bounds')
check(soil['column']['cell_area_m2'] == .25 * .25 and soil['column']['n_cells'] == 256 * 256, 'column grid geometry')
check(math.isclose(land['surface']['reference_level_m'], .5 * land['surface']['sediment_voxels']['count'] * land['surface']['sediment_voxels']['dz_m']), 'terrain reference datum')
check(land['surface']['call_arguments'] == {'ny': 256, 'nx': 256, 'dx_m': .25, 'dy_m': .25,
                                         'correlation_length_x_m': 1., 'correlation_length_y_m': 1., 'relief_m': .05, 'seed': 1173}, 'MAPLE random-field recipe arguments')
for field in ['generation_config', 'species_catalog', 'weather', 'monthly_normals', 'landscape', 'soil', 'scheduler_settings', 'scheduler_plan']:
    check((CASE / case['paths'][field]).is_file(), 'case-root input path ' + field)
check(case['runtime']['growth_cadence_days'] == 7 and case['snapshots']['cadence_days'] == 30., 'growth / reporting cadence')
check(case['checkpoints']['overwrite'] is False and case['checkpoints']['policy'] == 'after_each_accepted_transport_and_final', 'explicit checkpoint example policy')
check(case['snapshots']['event_snapshot_path_template'] != case['snapshots']['path_template'], 'separate event snapshot paths')

w = documents['climate/weather_2027.json']
keyset(w, ['schema', 'weather_id', 'utc_offset', 'sampling', 'units', 'temperature', 'humidity', 'radiation', 'pet', 'provenance', 'rows'], 'weather')
check(w['schema'] == 'masy.weather.v1' and w['utc_offset'] == '-07:00' and w['sampling'] == 'daily', 'weather declarations')
check(w['temperature'] == {'tmean': 'arithmetic_mean_of_tmin_tmax', 'extrema': True}, 'weather extrema / mean declarations')
check(w['humidity'] == {'supplied': 'none'} and w['radiation'] == {'supplied': False}, 'no invented humidity or radiation')
check(w['pet'] == {'method': 'ecohyd_hargreaves'}, 'PET delegated, never evaluated')
check(w['units'] == constants(ROOT.parent / 'MASY/src/masy/weather.py')['CANONICAL_UNITS'], 'native weather units')
mins = [34.5, 38.9, 45.5, 53.3, 62.1, 70.6, 73., 71.8, 65.4, 54., 42., 34.4]
maxs = [58.6, 64.1, 71.9, 80., 88.7, 97.1, 95.8, 94., 88.3, 79.4, 67., 57.8]
means = [46.5, 51.5, 58.7, 66.6, 75.4, 83.9, 84.4, 82.9, 76.9, 66.7, 54.5, 46.1]
inches = [Decimal(x) for x in ['.39', '.40', '.24', '.17', '.43', '.73', '1.58', '1.67', '1.52', '.59', '.43', '.63']]
expected_precip = [x * Decimal('25.4') for x in inches]
check(len(w['rows']) == 365, '365 daily weather rows')
for i, row in enumerate(w['rows']):
    date = dt.date(2027, 1, 1) + dt.timedelta(days=i)
    keyset(row, ['date', 'tmin_c', 'tmax_c'], 'weather row ' + str(i))
    check(row['date'] == date.isoformat(), 'weather date ' + str(i))
    check(math.isclose(row['tmin_c'], (mins[date.month - 1] - 32) * 5 / 9, abs_tol=1e-8), 'NOAA Tmin conversion ' + str(i))
    check(math.isclose(row['tmax_c'], (maxs[date.month - 1] - 32) * 5 / 9, abs_tol=1e-8), 'NOAA Tmax conversion ' + str(i))
monthly = list(csv.DictReader((CASE / 'climate/monthly_normals.csv').open()))
check(len(monthly) == 12, '12 monthly normal records')
for i, row in enumerate(monthly):
    check(int(row['month']) == i + 1 and int(row['days_2027']) == calendar.monthrange(2027, i + 1)[1], 'monthly calendar ' + str(i + 1))
    check((float(row['tmin_f']), float(row['tmean_f_published']), float(row['tmax_f'])) == (mins[i], means[i], maxs[i]), 'published NOAA temperatures ' + str(i + 1))
    check(Decimal(row['precip_in']) == inches[i] and Decimal(row['precip_mm']) == expected_precip[i], 'published NOAA precipitation ' + str(i + 1))

start = dt.datetime.fromisoformat(case['calendar']['start'])
end = dt.datetime.fromisoformat(case['calendar']['end'])
check((end - start).total_seconds() == 365 * 86400, '365-day fixed-offset calendar')
rows = list(csv.DictReader((CASE / 'scheduler/annual_plan.csv').open()))
check(len(rows) == 72, '72 transport rows')
check(len({row['event_id'] for row in rows}) == 72, 'unique occurrence IDs')
rain_sums = [Decimal(0) for _ in range(12)]
rain_counts = [0] * 12
previous_envelope = 0
max_display_error_mm = 0.
for i, row in enumerate(rows):
    kind = 'wind' if i % 2 == 0 else 'rain'
    check(row['kind'] == kind and int(row['seq']) == i + 1 and int(row['pair']) == i // 2 + 1, 'alternating event ' + str(i + 1))
    when = dt.datetime.fromisoformat(row['start_local'])
    nominal_end = dt.datetime.fromisoformat(row['nominal_end_local'])
    duration = int(row['duration_s'])
    offset = int(row['start_offset_s'])
    guard = int(row['recession_guard_s'] or 0)
    check(start <= when < end and when.utcoffset() == start.utcoffset(), 'event date range ' + str(i + 1))
    check(int((when - start).total_seconds()) == offset, 'event start offset ' + str(i + 1))
    check(nominal_end == when + dt.timedelta(seconds=duration), 'nominal forcing end ' + str(i + 1))
    check(int((nominal_end - start).total_seconds()) == int(row['nominal_end_offset_s']), 'nominal end offset ' + str(i + 1))
    envelope = int(row['forcing_plus_rain_guard_end_offset_s'])
    check(envelope == int(row['nominal_end_offset_s']) + guard, 'forcing-plus-rain-guard arithmetic ' + str(i + 1))
    check(offset >= previous_envelope, 'static forcing/guard separation ' + str(i + 1))
    previous_envelope = envelope
    check((CASE / row['forcing_ref']).is_file(), 'case-root event pattern path ' + str(i + 1))
    if kind == 'rain':
        month = when.month - 1
        rain_sums[month] += Decimal(row['rain_depth_mm'])
        rain_counts[month] += 1
        max_display_error_mm = max(max_display_error_mm, abs(float(row['rain_depth_mm']) - float(row['rain_intensity_mm_h']) * duration / 3600))
        check(guard == 21600 and Decimal(row['month_total_mm_normals']) == expected_precip[month], 'rain guard/normal ' + str(i + 1))
    else:
        check(duration == 43200 and guard == 0, 'wind forcing duration ' + str(i + 1))
check(rain_sums == expected_precip, 'exact 12 monthly rainfall budgets')
check(rain_counts == [2, 2, 3, 3, 3, 3, 4, 4, 4, 3, 3, 2], 'monthly rain counts')
check(sum(rain_sums) == Decimal('223.012'), 'annual rainfall budget')
check(max_display_error_mm <= 3e-9, 'rounded intensity display error bounded')
wind = documents['scheduler/examples/wind.json']
rain = documents['scheduler/examples/rain.json']
inter = documents['scheduler/examples/interevent.json']
keyset(wind, ['schema', 'event_id', 'wind_forcing'], 'native wind example')
keyset(rain, ['schema', 'event_id', 'water_profile_id', 'intervals', 'recession_guard_s'], 'native rain example')
keyset(inter, ['schema', 'event_id', 'duration_s', 'processes'], 'native interevent example')
for event in [wind, rain, inter]:
    check(re.fullmatch(r'[a-z0-9][a-z0-9._-]{0,63}', event['event_id']) is not None, 'native event ID ' + event['event_id'])
check(sum(b['duration_s'] for b in wind['wind_forcing']['blocks']) == 43200, 'wind example duration')
check(math.isclose(sum(b['duration_s'] * b['intensity_mm_h'] / 3600 for b in rain['intervals']), 10.033, rel_tol=1e-14), 'July rain example depth')
check(rain['water_profile_id'] == soil['events']['rain_water_profile_id'], 'water profile ID consistency')
check(inter['duration_s'] == 86400 and all(inter['processes'].values()), 'daily interevent template')
check(constants(ROOT.parent / 'MASY/src/masy/scenario.py')['GAP_POLICIES'] == ('refuse', 'idle'), 'actual native MASY gap choices')
check(schedule['status']['qualified_end_to_end'] is False, 'no false coupled runtime qualification')

for path in CASE.rglob('*.md'):
    for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
        if '://' not in target and not target.startswith('#'):
            check((path.parent / target.split('#')[0]).exists(), 'local Markdown link ' + target)
check('../../report.md' not in (CASE / 'README.md').read_text(), 'no missing candidate report link')
check('../' not in json.dumps(s), 'species notes use case-root references')

allowed_controls = {'agent_handoffs/current_prompt.md', 'agent_handoffs/orchestration_state.md',
                    'agent_handoffs/codex_review.md', 'agent_handoffs/verification.md', 'agent_handoffs/claude_report.md'}
for rel, original in baseline['tracked_files'].items():
    if rel not in allowed_controls:
        check((ROOT / rel).is_file() and digest(ROOT / rel) == original['sha256'], 'protected tracked file ' + rel)
check(digest(ROOT / 'AGENTS.md') == '90a8b6217553480283efd0dcba514306c5472809b35b1d5a0e80e3009e48ef31', 'original AGENTS preserved')
check(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == baseline['head'], 'no commits')

sessions = []
for revision in [1, 2, 3, 4]:
    launch = read_json(TASK / f'launch_{revision:02d}.json')
    exit_record = read_json(TASK / f'exit_{revision:02d}.json')
    events = [json.loads(line) for line in (TASK / f'claude_{revision:02d}.stdout.jsonl').read_text().splitlines() if line.strip()]
    terminal = [event for event in events if event.get('type') == 'result'][-1]
    check(exit_record['child_reaped'], 'Claude writer reaped ' + str(revision))
    check((exit_record['returncode'] == 1 and terminal['is_error']) if revision == 1 else (exit_record['returncode'] == 0 and not terminal['is_error']), 'Claude structured exit disposition ' + str(revision))
    check(not terminal.get('permission_denials'), 'no permission denial ' + str(revision))
    tool_counts = {}
    for event in events:
        blocks = event.get('message', {}).get('content', [])
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if block.get('type') != 'tool_use':
                continue
            name = block['name']
            tool_counts[name] = tool_counts.get(name, 0) + 1
            check(name in {'Read', 'Edit', 'Write'}, 'author tool allowed ' + name)
            raw = block.get('input', {}).get('file_path')
            if raw:
                path = Path(raw).resolve()
                if name == 'Read':
                    check(path.is_relative_to(CANDIDATE) or path == (TASK / f'prompt_{revision:02d}.md'), 'author read scope ' + str(path))
                else:
                    check(str(path.relative_to(CANDIDATE)) in allowed, 'author write scope ' + str(path))
    sessions.append({'revision': revision, 'returncode': exit_record['returncode'], 'is_error': terminal['is_error'],
                     'child_reaped': exit_record['child_reaped'], 'tools': tool_counts,
                     'failure_class': 'rate_limit; reset unknown; resolved by Sonnet continuation' if revision == 1 else None})

# The last correction was explicitly restricted to one prose table plus the report.
last_baseline = read_json(TASK / 'baseline_revision_04.json')['candidate_files']
last_changes = [rel for rel, value in inventory.items() if last_baseline.get(rel, {}).get('sha256') != value['sha256']]
check(set(last_changes) == {str(CASE_REL / 'PARAMETERS.md'), 'report.md'}, 'last revision only permitted wording/report changes')
original_weather = read_json(TASK / 'revision_02' / CASE_REL / 'climate/weather_2027.json')
original_weather['provenance']['description'] = original_weather['provenance']['description'].replace(
    'see ../landscape.toml)', 'see landscape.toml in the case root)')
check(original_weather == w, 'weather unchanged apart from approved path-reference correction')
for rel in ['climate/monthly_normals.csv', 'generation.json',
            'scheduler/examples/wind.json', 'scheduler/examples/rain.json', 'scheduler/examples/interevent.json']:
    check(digest(CASE / rel) == digest(TASK / 'revision_02' / CASE_REL / rel), 'stable unmodified input ' + rel)


def numerical_fields(value, prefix=''):
    result = {}
    if type(value) in (int, float, bool):
        result[prefix] = value
    elif isinstance(value, dict):
        for key, child in value.items():
            result.update(numerical_fields(child, prefix + '/' + key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.update(numerical_fields(child, prefix + '/' + str(index)))
    return result


for rel in ['species.json', 'generation.json', 'climate/weather_2027.json']:
    original = read_json(TASK / 'revision_02' / CASE_REL / rel)
    check(numerical_fields(original) == numerical_fields(documents[rel]), 'all scientific numeric fields unchanged ' + rel)

record = {'kind': 'STATIC INPUT REVIEW ONLY', 'scientific_modules_imported': False,
          'native_validators_tests_generators_models_run': False, 'passed_checks': len(CHECKS),
          'errors': ERRORS, 'case_files': {str(p.relative_to(CASE)): {'sha256': digest(p), 'size_bytes': p.stat().st_size}
                                        for p in CASE.rglob('*') if p.is_file()},
          'parameter_arithmetic': parameter_arithmetic,
          'weather_rows': len(w['rows']), 'transport_rows': len(rows), 'rain_counts': rain_counts,
          'monthly_rain_mm': [str(x) for x in rain_sums], 'annual_rain_mm': str(sum(rain_sums)),
          'max_rounded_intensity_depth_error_mm': max_display_error_mm, 'sessions': sessions,
          'limits': ['No native loading or execution; formats compared statically to current source.',
                     'Event completion unmeasured; calendar is a design example, not an executable coupled driver.',
                     'Scientific trial coefficients remain uncalibrated; source proxies are explicitly labelled.']}
(TASK / 'static_review.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({k: v for k, v in record.items() if k not in {'case_files', 'sessions'}}, indent=2))
raise SystemExit(1 if ERRORS else 0)
