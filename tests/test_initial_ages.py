'''Species-linked initial-age rule of the population generator (unittest, NumPy only).

Run from the candidate root:  python -m unittest tests.test_initial_ages -v

User steering under test: "The plant generator should generate plants with a range of ages, uniformly distributed from 0 to the
life expectancy". The DEFAULT (initial_age_days omitted) and the explicit selector {"distribution":
"uniform_to_life_expectancy"} draw each individual's age independently and uniformly on [0, L) days, L = that species'
catalog life expectancy (years = 365 days). Explicit constant / numeric / bounded-uniform ages remain. Small synthetic
generations only (no field case, no model run). Written WITHOUT the ability to execute them: Codex records the real runs.
'''
from __future__ import annotations

import copy
import io
import json
import os
import pathlib
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

import numpy as np

from leaf import initialization, state, traits
from leaf.__main__ import main as cli_main

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPECIES_JSON = ROOT / 'examples' / 'phase0' / 'species.json'
GENERATION_JSON = ROOT / 'examples' / 'phase0' / 'generation.json'
CASE = ROOT / 'cases' / 'jornada_grass05_shrub10_256'
SELECTOR = {'distribution': 'uniform_to_life_expectancy'}
GRASS_DAYS = 3000.0              # life expectancy of the test catalog: grass 3000 days ...
SHRUB_YEARS = 12                 # ... shrub 12 years (= 4380 days): two species, two units, two bounds


def catalog(grass=None, shrub=None, null_shrub=False):
    doc = traits.SpeciesCatalog.from_file(SPECIES_JSON).document
    doc['species']['grass']['life_expectancy'] = grass or {'mean': GRASS_DAYS, 'unit': 'days'}
    doc['species']['shrub']['life_expectancy'] = None if null_shrub else (shrub or {'mean': SHRUB_YEARS, 'unit': 'years'})
    return traits.SpeciesCatalog(doc)


def config(grass_age='omit', shrub_age='omit', **over):
    '''Small seeded synthetic config. grass_age / shrub_age: 'omit' (no initial_age_days key = the default) or a spec.'''
    sp = {'grass': {'target_cover': 0.15, 'diameter_mean_m': 0.4, 'diameter_std_m': 0.1},
          'shrub': {'target_cover': 0.05, 'diameter_mean_m': 1.0, 'diameter_std_m': 0.2}}
    for sid, age in (('grass', grass_age), ('shrub', shrub_age)):
        if not (isinstance(age, str) and age == 'omit'):
            sp[sid]['initial_age_days'] = age
    cfg = {'schema': 'leaf.generation_config', 'schema_version': 1, 'domain': {'nx': 32, 'ny': 32, 'dx_m': 1.0}, 'seed': 11,
           'overlap': 'allowed', 'cover_tolerance': 0.01, 'cover_subsamples': 8, 'max_iterations': 12,
           'limits': {'max_plants': 50000, 'max_attempts': 2000000}, 'species': sp}
    cfg.update(over)
    return cfg


def ks_uniform(u):
    '''Kolmogorov-Smirnov distance of a sample on [0, 1] from the continuous uniform law.'''
    u = np.sort(np.asarray(u, dtype=float))
    n = u.size
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - u), np.max(u - (i - 1) / n)))


def ages_of(s, cat, sid):
    return s.age_days[s.species_index == cat.index(sid)]


def same_state(tc, a, b):
    for k in state.PlantState.ARRAY_NAMES:
        np.testing.assert_array_equal(getattr(a, k), getattr(b, k), err_msg=k)


def run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli_main(argv)
    return code, out.getvalue(), err.getvalue()


class TestSpeciesLinkedAges(unittest.TestCase):
    def setUp(self):
        self.cat = catalog()
        self.bounds = {'grass': GRASS_DAYS, 'shrub': SHRUB_YEARS * 365.0}

    def test_default_is_uniform_from_zero_to_each_species_life_expectancy(self):
        s, r = initialization.generate_population(self.cat, config())          # initial_age_days omitted
        self.assertEqual(self.cat.life_expectancy_days('grass'), GRASS_DAYS)   # the bound comes from the catalog ...
        self.assertEqual(self.cat.life_expectancy_days('shrub'), 4380.0)       # ... in days, years = 365
        for sid, bound in self.bounds.items():
            a = ages_of(s, self.cat, sid)
            self.assertGreater(a.size, 40, sid)
            self.assertTrue(np.all(a >= 0.0) and np.all(a < bound), sid)       # continuous uniform on [0, L)
            self.assertGreater(float(a.std()), 0.0)                            # not constant
            self.assertLess(ks_uniform(a / bound), 2.5 / np.sqrt(a.size), sid)  # compatible with the uniform law
        grass = ages_of(s, self.cat, 'grass')
        self.assertLess(float(grass.min()), 0.05 * GRASS_DAYS)                 # the whole range is used, not a corner
        self.assertGreater(float(grass.max()), 0.95 * GRASS_DAYS)
        self.assertAlmostEqual(float(grass.mean()), GRASS_DAYS / 2, delta=0.06 * GRASS_DAYS)
        self.assertGreater(float(ages_of(s, self.cat, 'shrub').max()), GRASS_DAYS)   # shrubs reach beyond the grass bound
        self.assertTrue(np.all(s.status == state.LIVE))
        # ages are independent of size: no age-biomass relation was invented (weak correlation on a large sample)
        m = s.species_index == self.cat.index('grass')
        self.assertLess(abs(float(np.corrcoef(s.age_days[m], s.agb_dry_kg[m])[0, 1])), 0.15)

    def test_selector_is_identical_to_the_default_and_to_explicit_uniform_bounds(self):
        d, _ = initialization.generate_population(self.cat, config())
        sel, _ = initialization.generate_population(self.cat, config(SELECTOR, SELECTOR))
        same_state(self, d, sel)
        # the same single vectorised rng.uniform call: identical to an explicit uniform whose bounds equal 0 .. L
        explicit, _ = initialization.generate_population(self.cat, config(
            {'distribution': 'uniform', 'min_days': 0, 'max_days': GRASS_DAYS},
            {'distribution': 'uniform', 'min_days': 0, 'max_days': 4380}))
        same_state(self, d, explicit)
        # fixed seed reproducibility, and another seed changes the ages
        again, _ = initialization.generate_population(self.cat, config())
        same_state(self, d, again)
        other, _ = initialization.generate_population(self.cat, config(seed=12))
        self.assertFalse(np.array_equal(d.age_days[:20], other.age_days[:20]))

    def test_two_species_two_units_two_bounds_follow_the_catalog_not_the_config(self):
        alt = catalog(grass={'mean': 2, 'unit': 'years'}, shrub={'mean': 9000, 'unit': 'days'})
        s, r = initialization.generate_population(alt, config())
        self.assertLess(float(ages_of(s, alt, 'grass').max()), 730.0)          # 2 years = 730 d
        self.assertGreater(float(ages_of(s, alt, 'grass').max()), 0.9 * 730.0)
        self.assertLess(float(ages_of(s, alt, 'shrub').max()), 9000.0)
        self.assertGreater(float(ages_of(s, alt, 'shrub').max()), 0.7 * 9000.0)
        self.assertNotEqual(alt.hash, self.cat.hash)                           # another catalog is another identity
        # per-species report: resolved bounds + catalog origin, outside any CSV
        grass_rec = next(p for p in r['species'] if p['species'] == 'grass')['age_requested']
        self.assertEqual((grass_rec['min_days'], grass_rec['max_days']), (0.0, 730.0))
        self.assertEqual(grass_rec['life_expectancy'], {'mean': 2, 'unit': 'years'})
        self.assertEqual(grass_rec['days_per_unit'], 365.0)

    def test_resolved_bounds_origin_and_source_rule_are_reported_and_replay_is_exact(self):
        s, r = initialization.generate_population(self.cat, config())
        # the SOURCE rule is kept in the replayable config (written explicitly even though the original omitted it) ...
        for sid in ('grass', 'shrub'):
            self.assertEqual(r['generation_config']['species'][sid]['initial_age_days'], SELECTOR)
        # ... and the resolved bounds / origin / default-vs-explicit provenance live in derived / the report / the provenance
        g = r['derived']['species']['grass']['initial_age']
        self.assertEqual(g['rule'], 'uniform_to_life_expectancy')
        self.assertIn('DEFAULT', g['rule_source'])
        self.assertEqual((g['distribution'], g['min_days'], g['max_days'], g['life_expectancy_days']), ('uniform', 0.0, 3000.0, 3000.0))
        self.assertEqual(g['catalog_hash'], self.cat.hash)
        self.assertIn("species['grass'].life_expectancy", g['catalog_origin'])
        self.assertIn('not a hard death age', g['note'])
        sh = r['derived']['species']['shrub']['initial_age']
        self.assertEqual((sh['max_days'], sh['life_expectancy'], sh['days_per_unit']), (4380.0, {'mean': 12, 'unit': 'years'}, 365.0))
        self.assertIn('initial_age_note', r)
        self.assertEqual(s.provenance['derived']['species']['grass']['initial_age']['max_days'], 3000.0)
        # an explicit selector is labelled as such
        _, r2 = initialization.generate_population(self.cat, config(SELECTOR, SELECTOR))
        self.assertIn('explicit selector', r2['derived']['species']['grass']['initial_age']['rule_source'])
        # JSON round trip of the saved config replays array-for-array and reproduces the record
        saved = json.loads(json.dumps(r['generation_config']))
        replay, r3 = initialization.generate_population(self.cat, saved)
        same_state(self, s, replay)
        self.assertEqual(r3['generation_config'], r['generation_config'])
        self.assertEqual(r3['derived']['species']['grass']['initial_age']['max_days'], 3000.0)
        # the rule is re-resolved from the catalog: the SAME config with another catalog gives another range (hash differs)
        alt = catalog(grass={'mean': 6000, 'unit': 'days'})
        s_alt, r_alt = initialization.generate_population(alt, saved)
        self.assertEqual(r_alt['derived']['species']['grass']['initial_age']['max_days'], 6000.0)
        self.assertGreater(float(ages_of(s_alt, alt, 'grass').max()), 3000.0)
        # validate / config_record round trip
        cfg = initialization.validate_generation_config(saved, self.cat)
        self.assertEqual(initialization.config_record(cfg), r['generation_config'])
        self.assertEqual(cfg['species'][0]['initial_age_days'], SELECTOR)
        self.assertEqual(cfg['species'][0]['age_sampling'], {'distribution': 'uniform', 'min_days': 0.0, 'max_days': 3000.0})

    def test_missing_or_null_life_expectancy_fails_for_default_and_selector_only(self):
        nul = catalog(null_shrub=True)
        self.assertIsNone(nul.life_expectancy_days('shrub'))
        E = traits.CatalogError
        for name, cfg in (('default', config()), ('selector', config(SELECTOR, SELECTOR)),
                          ('selector_shrub_only', config(grass_age=5.0, shrub_age=SELECTOR))):
            with self.subTest(name):
                with self.assertRaises(E) as cm:
                    initialization.validate_generation_config(cfg, nul)
                self.assertIn('life_expectancy', str(cm.exception))
                self.assertIn("'shrub'", str(cm.exception))                   # names the species
                with self.assertRaises(E):
                    initialization.generate_population(nul, cfg)              # nothing is generated or guessed
        with self.assertRaises(E) as cm:
            initialization.validate_generation_config(config(), nul)
        self.assertIn('DEFAULT', str(cm.exception))
        with self.assertRaises(E) as cm:
            initialization.validate_generation_config(config(grass_age=5.0, shrub_age=SELECTOR), nul)
        self.assertNotIn('DEFAULT', str(cm.exception))
        # explicit legacy ages still work with life_expectancy null
        s, r = initialization.generate_population(nul, config(
            {'distribution': 'constant', 'value_days': 100}, {'distribution': 'uniform', 'min_days': 365, 'max_days': 3650}))
        np.testing.assert_array_equal(ages_of(s, nul, 'grass'), 100.0)
        sh = ages_of(s, nul, 'shrub')
        self.assertTrue(np.all((sh >= 365.0) & (sh <= 3650.0)) and float(sh.std()) > 0.0)
        n, _ = initialization.generate_population(nul, config(730, 1825))      # plain numbers (the old fixtures)
        np.testing.assert_array_equal(ages_of(n, nul, 'grass'), 730.0)
        np.testing.assert_array_equal(ages_of(n, nul, 'shrub'), 1825.0)
        self.assertEqual(r['derived']['species']['shrub']['initial_age']['rule'], 'explicit')
        # a species that is not configured never needs a life expectancy
        only_grass = config()
        del only_grass['species']['shrub']
        s2, _ = initialization.generate_population(nul, only_grass)
        self.assertTrue(np.all(s2.species_index == nul.index('grass')))

    def test_selector_takes_no_parameters_and_unknown_distributions_are_refused(self):
        E = traits.CatalogError
        for bad in (dict(SELECTOR, max_days=5), dict(SELECTOR, life_expectancy_days=5), {'distribution': 'uniform_to_lifespan'},
                    {'distribution': 'uniform'}, 'uniform_to_life_expectancy', None, True, -1):
            with self.subTest(repr(bad)):
                with self.assertRaises(E):
                    initialization.validate_generation_config(config(bad, SELECTOR), self.cat)
        # a missing initial_age_days is the ONLY way to omit it; the key may not be null
        cfg = config()
        cfg['species']['grass']['initial_age_days'] = None
        with self.assertRaises(E):
            initialization.validate_generation_config(cfg, self.cat)

    def test_explicit_rules_are_unchanged(self):
        s, r = initialization.generate_population(self.cat, config(730, {'distribution': 'uniform', 'min_days': 365, 'max_days': 3650}))
        np.testing.assert_array_equal(ages_of(s, self.cat, 'grass'), 730.0)
        sh = ages_of(s, self.cat, 'shrub')
        self.assertTrue(np.all((sh >= 365.0) & (sh <= 3650.0)))
        self.assertEqual(r['generation_config']['species']['shrub']['initial_age_days']['distribution'], 'uniform')
        self.assertEqual(r['generation_config']['species']['grass']['initial_age_days'], {'distribution': 'constant', 'value_days': 730.0})

    def test_exclusion_mode_uses_the_same_age_rule(self):
        cfg = config(domain={'nx': 16, 'ny': 16, 'dx_m': 1.0}, overlap='disallowed')
        cfg['species'] = {'grass': {'target_cover': 0.10, 'diameter_mean_m': 0.3, 'diameter_std_m': 0.05},
                          'shrub': {'target_cover': 0.08, 'diameter_mean_m': 1.0, 'diameter_std_m': 0.0}}
        s, r = initialization.generate_population(self.cat, cfg, surface=state.constant_surface(state.Domain(16, 16, 1.0), 5.0))
        g = ages_of(s, self.cat, 'grass')
        self.assertTrue(np.all((g >= 0.0) & (g < GRASS_DAYS)) and float(g.std()) > 0.0)
        self.assertTrue(s.establishment_resolved)
        self.assertEqual(r['derived']['species']['grass']['initial_age']['max_days'], GRASS_DAYS)


class TestFilesAndShippedConfigs(unittest.TestCase):
    def test_csv_round_trip_keeps_generated_ages_and_handwritten_ages_are_untouched(self):
        cat = catalog()
        s, r = initialization.generate_population(cat, config(), surface=state.constant_surface(state.Domain(32, 32, 1.0), 7.0))
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'pop.csv')
            state.write_plant_csv(s, path)
            with open(path, 'r', encoding='utf-8') as fh:
                text = fh.read()
            self.assertNotIn('life_expectancy', text)                         # the age rule is metadata, never in the plain CSV
            self.assertNotIn('#', text)
            back = state.read_plant_csv(path, cat)
            np.testing.assert_array_equal(back.age_days, s.age_days)          # FP64-exact age records
            with open(state.metadata_path(path), 'r', encoding='utf-8') as fh:
                meta = json.load(fh)
            self.assertEqual(meta['provenance']['derived']['species']['grass']['initial_age']['max_days'], GRASS_DAYS)
            self.assertEqual(meta['provenance']['derived']['species']['shrub']['initial_age']['max_days'], 4380.0)
        # a handwritten plain list keeps whatever ages it states (no generator involvement), even above the life expectancy
        plain = ('plant_id,species,x_m,y_m,establishment_ground_m,agb_dry_kg,seasonal_agb_dry_kg,age_days,status\n'
                 '1,grass,2.5,3.5,5.0,0.12,0.0,12.5,live\n'
                 '2,shrub,10.0,10.0,5.0,2.5,0.0,99999.0,live\n')
        h = state.parse_plant_csv(plain, cat, domain=state.Domain(20, 20, 1.0))
        self.assertEqual(h.age_days[h.indices_of([1])[0]], 12.5)
        self.assertEqual(h.age_days[h.indices_of([2])[0]], 99999.0)           # older than the shrub's 4380 d: just a record

    def test_shipped_example_config_uses_the_species_linked_rule(self):
        cat = traits.SpeciesCatalog.from_file(SPECIES_JSON)
        with open(GENERATION_JSON, 'r', encoding='utf-8') as fh:
            raw = json.load(fh)
        for sid in ('grass', 'shrub'):
            self.assertEqual(raw['species'][sid]['initial_age_days'], SELECTOR)
        cfg = initialization.validate_generation_config(raw, cat)
        got = {sp['id']: sp['age_sampling'] for sp in cfg['species']}
        self.assertEqual(got['grass'], {'distribution': 'uniform', 'min_days': 0.0, 'max_days': 8 * 365.0})     # fixture 8 years
        self.assertEqual(got['shrub'], {'distribution': 'uniform', 'min_days': 0.0, 'max_days': 40 * 365.0})    # fixture 40 years

    def test_jornada_trial_config_validates_to_0_15_and_0_80_year_ranges_without_generating(self):
        spc, gen = CASE / 'species.json', CASE / 'generation.json'
        if not (spc.is_file() and gen.is_file()):
            self.skipTest('Jornada trial case files are not present in this tree')
        cat = traits.SpeciesCatalog.from_file(spc)
        with open(gen, 'r', encoding='utf-8') as fh:
            raw = json.load(fh)
        for sid in ('BOER', 'PROGLA'):
            self.assertEqual(raw['species'][sid]['initial_age_days'], SELECTOR)
        cfg = initialization.validate_generation_config(raw, cat)               # PURE validation: no population is generated
        got = {sp['id']: sp['age_resolved'] for sp in cfg['species']}
        self.assertEqual((got['BOER']['min_days'], got['BOER']['max_days']), (0.0, 15 * 365.0))     # 0 .. 15 years
        self.assertEqual((got['PROGLA']['min_days'], got['PROGLA']['max_days']), (0.0, 80 * 365.0))  # 0 .. 80 years
        self.assertEqual(got['BOER']['life_expectancy'], {'mean': 15, 'unit': 'years'})
        self.assertEqual(got['PROGLA']['life_expectancy'], {'mean': 80, 'unit': 'years'})
        self.assertEqual(got['BOER']['catalog_hash'], cat.hash)

    def test_cli_generate_uses_the_default_rule_and_reports_the_resolved_bounds(self):
        cat = catalog()
        with tempfile.TemporaryDirectory() as td:
            doc_path = os.path.join(td, 'species.json')
            with open(doc_path, 'w', encoding='utf-8') as fh:
                json.dump(cat.document, fh)
            cfg_path = os.path.join(td, 'gen.json')
            with open(cfg_path, 'w', encoding='utf-8') as fh:
                json.dump(config(), fh)
            out, rep = os.path.join(td, 'g.csv'), os.path.join(td, 'g_report.json')
            code, _, err = run_cli(['generate', '--species', doc_path, '--config', cfg_path, '--out', out, '--report', rep,
                                    '--surface-constant', '3.0'])
            self.assertEqual(code, 0, err)
            s = state.read_plant_csv(out, traits.load_catalog(doc_path))
            for sid, bound in (('grass', GRASS_DAYS), ('shrub', 4380.0)):
                a = s.age_days[s.species_index == cat.index(sid)]
                self.assertTrue(np.all((a >= 0.0) & (a < bound)) and float(a.std()) > 0.0, sid)
            with open(rep, 'r', encoding='utf-8') as fh:
                report = json.load(fh)
            self.assertEqual(report['derived']['species']['shrub']['initial_age']['max_days'], 4380.0)
            self.assertEqual(report['generation_config']['species']['grass']['initial_age_days'], SELECTOR)
            # a null life expectancy fails the default through the CLI with a clear message and writes nothing
            nul_doc = catalog(null_shrub=True).document
            nul_path = os.path.join(td, 'species_null.json')
            with open(nul_path, 'w', encoding='utf-8') as fh:
                json.dump(nul_doc, fh)
            code, _, err = run_cli(['generate', '--species', nul_path, '--config', cfg_path, '--out', os.path.join(td, 'n.csv')])
            self.assertEqual(code, 1)
            self.assertIn('life_expectancy', err)
            self.assertFalse(any(n.startswith('n.') for n in os.listdir(td)))


if __name__ == '__main__':
    unittest.main()
