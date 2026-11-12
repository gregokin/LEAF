'''Plain plant CSV + companion metadata regression tests (unittest, NumPy only; no native SYRUP / numba needed).

Run from the candidate root:  python -m unittest tests.test_plant_csv -v

Project rule under test: a LEAF plant file is a PLAIN nine-column CSV (no embedded metadata / comment / JSON); the saved-file
metadata lives in the separate companion "<stem>.metadata.json", bound to the CSV by SHA-256. The complete-restart
(checkpoint) cases that need the native water authority live in tests/test_interevent.py (RestartTests).
Written WITHOUT the ability to execute them (file tools only): Codex records the real runs.
'''
from __future__ import annotations

import copy
import csv
import importlib.util
import inspect
import io
import json
import os
import pathlib
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import numpy as np

from leaf import recording, state, traits
from leaf.__main__ import main as cli_main

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / 'examples' / 'phase0'
SPECIES_JSON = EXAMPLES / 'species.json'
PLANTS_CSV = EXAMPLES / 'plants.csv'
PREVIEW_SCRIPT = ROOT / 'cases' / 'jornada_grass05_shrub10_256' / 'preview' / 'render_preview.py'
DOM = state.Domain(20, 20, 1.0)


def rec(pid, sp, x, y, est, B, E=0.0, age=0.0, status='live'):
    return {'plant_id': pid, 'species': sp, 'x_m': x, 'y_m': y, 'establishment_ground_m': est, 'agb_dry_kg': B,
            'seasonal_agb_dry_kg': E, 'age_days': age, 'status': status}


def catalog():
    return traits.SpeciesCatalog.from_file(SPECIES_JSON)


def other_catalog():
    doc = catalog().document
    doc['species']['grass']['allometry']['diameter']['a'] = 0.91     # different content -> different hash
    return traits.SpeciesCatalog(doc)


def six(cat, resolved=True, time_days=0.0, provenance=None):
    '''Six plants (4 live, 1 dead_standing, 1 removed); ids 1, 2, 4 use SURFACE until resolved.'''
    recs = [rec(1, 'grass', 2.5, 3.5, 'SURFACE', 0.12, 0.0, 730.0),
            rec(2, 'grass', 7.25, 12.75, 'SURFACE', 0.08, 0.01, 365.0),
            rec(3, 'shrub', 10.0, 10.0, 99.9, 2.5, 0.0, 3650.0),
            rec(4, 'shrub', 15.5, 4.5, 'SURFACE', 1.8, 0.2, 1825.0),
            rec(5, 'grass', 18.0, 18.0, 100.0, 0.05, 0.0, 1095.0, 'dead_standing'),
            rec(6, 'shrub', 3.0, 16.0, 99.8, 0.0, 0.0, 5475.0, 'removed')]
    surface = state.constant_surface(DOM, 100.0) if resolved else None
    return state.PlantState.from_records(cat, DOM, recs, time_days=time_days, surface=surface, provenance=provenance)


def changed(cat, time_days):
    '''six() with a different biomass of plant 3: a DIFFERENT CSV (the time lives only in the companion).'''
    s = six(cat, time_days=time_days)
    s.update_plants([3], agb_dry_kg=[2.6])
    return s


def read_text(path):
    with open(path, 'r', encoding='utf-8', newline='') as fh:
        return fh.read()


def read_bytes(path):
    with open(path, 'rb') as fh:
        return fh.read()


def write_text(path, text):
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        fh.write(text)


def edit_companion(path, fn):
    '''Edit the on-disk companion JSON of a plant CSV path in place (fn mutates the parsed dict).'''
    mp = state.metadata_path(path)
    doc = json.loads(read_text(mp))
    fn(doc)
    write_text(mp, json.dumps(doc))


def legacy_text(s, *, csv_initial=False, **meta_over):
    '''The PRE-companion embedded form "# {json}\\n<csv>" (read compatibility only: LEAF never writes it).'''
    meta = {'schema': state.CSV_SCHEMA, 'schema_version': state.CSV_SCHEMA_VERSION, 'model': 'LEAF', 'time_days': s.time_days,
            'n_plants': s.n, 'catalog_hash': s.catalog_hash, 'species_names': list(s.species_names),
            'domain': s.domain.to_dict(), 'units': dict(state.UNITS), 'coordinate_frame': state.COORDINATE_FRAME,
            'establishment_resolved': s.establishment_resolved, 'initial': bool(csv_initial),
            'provenance': {'legacy': True}}
    meta.update(meta_over)
    return '# ' + json.dumps(meta, sort_keys=True) + '\n' + state.plant_csv_text(s, initial=csv_initial)


def assert_plain_csv(tc, text, n):
    '''Strict CSV: first row the nine-column header, every data row nine fields, no metadata / comment / JSON.'''
    rows = list(csv.reader(io.StringIO(text, newline=''), strict=True))
    tc.assertEqual(rows[0], list(state.CSV_COLUMNS))
    tc.assertEqual(len(rows), n + 1)
    tc.assertTrue(all(len(r) == 9 for r in rows))
    tc.assertFalse(any(r[0].startswith('#') for r in rows))
    tc.assertNotIn('{', text)
    tc.assertNotIn('"initial"', text)


def run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli_main(argv)
    return code, out.getvalue(), err.getvalue()


def generation_config():
    return {'schema': 'leaf.generation_config', 'schema_version': 1, 'domain': {'nx': 32, 'ny': 32, 'dx_m': 1.0}, 'seed': 11,
            'overlap': 'allowed', 'cover_tolerance': 0.01, 'cover_subsamples': 8, 'max_iterations': 12,
            'limits': {'max_plants': 50000, 'max_attempts': 2000000},
            'species': {'grass': {'target_cover': 0.15, 'diameter_mean_m': 0.4, 'diameter_std_m': 0.1, 'initial_age_days': 730},
                        'shrub': {'target_cover': 0.05, 'diameter_mean_m': 1.0, 'diameter_std_m': 0.2,
                                  'initial_age_days': {'distribution': 'uniform', 'min_days': 365, 'max_days': 3650}}}}


def fail_nth_atomic_write(n, exc=None):
    '''Wrap state.atomic_write_bytes so that its n-th call raises (1-based); every other call is real.'''
    real = state.atomic_write_bytes
    calls = []

    def wrapper(path, data, **kw):
        calls.append(os.fspath(path))
        if len(calls) == n:
            raise exc or OSError(5, 'injected write failure')
        return real(path, data, **kw)
    return wrapper, calls


# =============================================================== plain CSV + companion round trip
class TestPlainSerialization(unittest.TestCase):
    def setUp(self):
        self.cat = catalog()

    def test_serializer_returns_plain_csv_and_has_no_embedded_metadata_option(self):
        for resolved in (True, False):
            s = six(self.cat, resolved=resolved)
            text = state.plant_csv_text(s, initial=not resolved)
            assert_plain_csv(self, text, 6)
        self.assertIn('SURFACE', state.plant_csv_text(six(self.cat, resolved=False), initial=True))
        # no code path can WRITE embedded metadata, not even opt-in: the keyword does not exist anywhere
        for fn in (state.plant_csv_text, state.write_plant_csv, state.write_plant_files, state.plant_files_text):
            self.assertNotIn('metadata', inspect.signature(fn).parameters, fn.__name__)
        with self.assertRaises(TypeError):
            state.plant_csv_text(six(self.cat), metadata=True)
        with self.assertRaises(TypeError):
            state.plant_csv_text(six(self.cat), provenance={'a': 1})
        self.assertNotIn('metadata', inspect.signature(recording.SnapshotWriter.__init__).parameters)
        self.assertFalse(hasattr(state, 'csv_metadata'))

    def test_metadata_path_convention(self):
        j = os.path.join
        self.assertEqual(state.metadata_path(j('d', 'initial_plants.csv')), j('d', 'initial_plants.metadata.json'))
        self.assertEqual(state.metadata_path(j('d', 'plants.csv')), j('d', 'plants.metadata.json'))
        self.assertEqual(state.metadata_path(j('d', 'a.b.csv')), j('d', 'a.b.metadata.json'))
        self.assertEqual(state.metadata_path(j('d', 'fixed')), j('d', 'fixed.metadata.json'))
        self.assertEqual(state.metadata_path(pathlib.Path('x') / 'veg_001.csv'), os.path.join('x', 'veg_001.metadata.json'))
        with self.assertRaises(state.StateError):                    # the companion is not itself a plant CSV path
            state.metadata_path(j('d', 'plants.metadata.json'))

    def test_real_files_round_trip_with_companion_hash_and_human_readable_json(self):
        rng = np.random.default_rng(5)
        n = 150
        dom = state.Domain(37, 23, 0.7, 1.3, -12.345678901234567, 3.0000000001)
        x, y = rng.uniform(dom.x_min_m, dom.x_max_m, n), rng.uniform(dom.y_min_m, dom.y_max_m, n)
        B = np.exp(rng.uniform(-9, 3, n))
        s = state.PlantState(dom, self.cat.species_ids, self.cat.hash, ids=rng.permutation(n).astype(np.int64) * 7919 - 100,
                             species_index=rng.integers(0, 2, n).astype(np.int32), x_m=x, y_m=y,
                             establishment_ground_m=rng.normal(1000.0, 10.0, n), agb_dry_kg=B,
                             seasonal_agb_dry_kg=B * rng.uniform(0, 1, n), age_days=rng.uniform(0, 2e4, n),
                             status=np.zeros(n, np.int8), time_days=0.1 + 0.2, provenance={'case': 'round-trip'})
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'veg.csv')
            csv_path, meta_path = state.write_plant_files(s, path, provenance={'case': 'round-trip', 'k': [1, 2]})
            self.assertEqual((csv_path, meta_path), (path, os.path.join(td, 'veg.metadata.json')))
            self.assertEqual(sorted(os.listdir(td)), ['veg.csv', 'veg.metadata.json'])    # nothing else, no temp files
            text = read_text(path)
            assert_plain_csv(self, text, n)
            meta_text = read_text(meta_path)
            self.assertTrue(meta_text.endswith('\n'))
            self.assertIn('\n  "', meta_text)                                              # indented, human readable
            meta = json.loads(meta_text)
            self.assertEqual((meta['schema'], meta['schema_version']), (state.METADATA_SCHEMA, state.METADATA_SCHEMA_VERSION))
            self.assertEqual((meta['n_plants'], meta['time_days'], meta['initial']), (n, 0.1 + 0.2, False))
            self.assertEqual(meta['catalog_hash'], self.cat.hash)
            self.assertEqual(meta['species_names'], list(self.cat.species_ids))
            self.assertEqual(meta['domain']['nx'], 37)
            self.assertEqual(meta['units'], state.UNITS)
            self.assertEqual(meta['provenance'], {'case': 'round-trip', 'k': [1, 2]})
            self.assertEqual(meta['csv']['sha256'], state._sha256(read_bytes(path)))
            self.assertEqual((meta['csv']['n_bytes'], meta['csv']['file']), (len(read_bytes(path)), 'veg.csv'))
            self.assertEqual(meta['csv']['columns'], list(state.CSV_COLUMNS))
            back = state.read_plant_csv(path, self.cat)                  # companion loaded and verified automatically
            for k in state.PlantState.ARRAY_NAMES:
                np.testing.assert_array_equal(getattr(back, k), getattr(s, k), err_msg=k)
            self.assertEqual((back.time_days, back.domain), (0.1 + 0.2, dom))
            self.assertEqual(back.provenance, {'case': 'round-trip', 'k': [1, 2]})
            # caller context that agrees with the companion is accepted
            self.assertEqual(len(state.read_plant_csv(path, self.cat, domain=dom, time_days=0.1 + 0.2)), n)
            # in memory: the serializer's text + plant_metadata dict verify each other without any file
            mem = state.parse_plant_csv(text, self.cat, metadata=meta)
            np.testing.assert_array_equal(mem.x_m, s.x_m)
            self.assertEqual(state.plant_metadata(s, text)['csv']['sha256'], meta['csv']['sha256'])
            csv_text, meta_text2 = state.plant_files_text(s)
            self.assertEqual(csv_text, text)
            self.assertEqual(json.loads(meta_text2)['csv']['sha256'], meta['csv']['sha256'])
            # a plain-CSV reader that ignores the companion needs the domain from the caller (explicit opt-out)
            with self.assertRaises(state.StateError):
                state.read_plant_csv(path, self.cat, metadata=False)
            self.assertEqual(len(state.read_plant_csv(path, self.cat, domain=dom, metadata=False)), n)

    def test_metadata_size_counts_the_actual_companion_json_without_building_it(self):
        s = six(self.cat, time_days=123456.789)
        for prov in ({}, {'k': 'v' * 100000, 'nested': {'a': [1.5, 2, 'café']}}):
            _, meta_text = state.plant_files_text(s, provenance=prov)
            actual = len(meta_text.encode('utf-8'))
            n = state.plant_metadata_size(s, provenance=prov)
            self.assertGreaterEqual(n, actual)                                     # conservative ...
            self.assertLessEqual(n - actual, state.METADATA_SIZE_SLACK_BYTES + 128)  # ... and tight (scalar placeholders only)
        big = state.plant_metadata_size(s, provenance={'k': 'v' * 1_000_000})
        small = state.plant_metadata_size(s, provenance={'k': 'v' * 10})
        self.assertGreaterEqual(big - small, 1_000_000 - 10)                       # provenance is counted whole

    def test_row_bytes_are_the_same_as_the_legacy_body_so_conversion_is_a_pure_split(self):
        s = six(self.cat)
        legacy = legacy_text(s)
        first, body = legacy.split('\n', 1)
        self.assertTrue(first.startswith('# {'))
        self.assertEqual(body, state.plant_csv_text(s))          # identical header + rows: only the metadata line moved
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'p.csv')
            state.write_plant_csv(s, path)
            self.assertEqual(read_text(path), body)
            self.assertFalse(read_bytes(path).startswith(b'#'))


# =============================================================== edited / missing / mismatched metadata
class TestCompanionVerification(unittest.TestCase):
    def setUp(self):
        self.cat = catalog()
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.path = os.path.join(self.td.name, 'snap.csv')
        self.s = six(self.cat, time_days=12.5, provenance={'who': 'test'})
        state.write_plant_csv(self.s, self.path)

    def test_edited_or_reformatted_csv_is_refused_by_the_hash_binding(self):
        original = read_text(self.path)
        edits = {'value': original.replace('0.12', '0.13', 1), 'extra_newline': original + '\n',
                 'crlf': original.replace('\n', '\r\n'), 'dropped_row': ''.join(original.splitlines(True)[:-1]),
                 'whitespace': original.replace(',live', ', live', 1)}
        for name, text in edits.items():
            self.assertNotEqual(text, original, name)
            write_text(self.path, text)
            with self.assertRaises(state.StateError, msg=name) as cm:
                state.read_plant_csv(self.path, self.cat)
            self.assertIn('match', str(cm.exception), name)
        write_text(self.path, original)
        self.assertEqual(len(state.read_plant_csv(self.path, self.cat)), 6)         # restored: accepted again

    def test_stale_pair_from_another_write_is_refused(self):
        other = os.path.join(self.td.name, 'other.csv')
        state.write_plant_csv(changed(self.cat, 99.0), other)
        self.assertNotEqual(read_text(other), read_text(self.path))
        write_text(self.path, read_text(other))                  # companion of snapshot A beside the CSV of snapshot B
        with self.assertRaises(state.StateError):
            state.read_plant_csv(self.path, self.cat)
        # the opposite mix (companion of B beside the CSV of A) is refused as well
        state.write_plant_csv(self.s, self.path, overwrite=True)
        write_text(state.metadata_path(self.path), read_text(state.metadata_path(other)))
        with self.assertRaises(state.StateError):
            state.read_plant_csv(self.path, self.cat)

    def test_edited_companion_fields_are_refused(self):
        edits = {
            'catalog_hash': lambda d: d.update(catalog_hash='sha256:' + '1' * 64),
            'species_order': lambda d: d.update(species_names=['shrub', 'grass']),
            'domain_convention': lambda d: d['domain'].update(edges='periodic'),
            'domain_unknown_key': lambda d: d['domain'].update(bogus=1),
            'n_plants': lambda d: d.update(n_plants=d['n_plants'] + 1),
            'schema': lambda d: d.update(schema='leaf.other'),
            'schema_version': lambda d: d.update(schema_version=2),
            'schema_version_float': lambda d: d.update(schema_version=1.0),
            'units': lambda d: d['units'].update(agb_dry_kg='g'),
            'frame': lambda d: d.update(coordinate_frame='origin-relative'),
            'csv_sha256': lambda d: d['csv'].update(sha256='sha256:' + '0' * 64),
            'csv_sha256_form': lambda d: d['csv'].update(sha256='not-a-hash'),
            'csv_columns': lambda d: d['csv'].update(columns=['plant_id']),
            'csv_n_bytes_wrong': lambda d: d['csv'].update(n_bytes=d['csv']['n_bytes'] + 1),
            'csv_n_bytes_zero': lambda d: d['csv'].update(n_bytes=0),
            'csv_n_bytes_negative': lambda d: d['csv'].update(n_bytes=-1),
            'csv_n_bytes_float': lambda d: d['csv'].update(n_bytes=float(d['csv']['n_bytes'])),
            'csv_n_bytes_bool': lambda d: d['csv'].update(n_bytes=True),
            'csv_n_bytes_string': lambda d: d['csv'].update(n_bytes=str(d['csv']['n_bytes'])),
            'csv_n_bytes_missing': lambda d: d['csv'].pop('n_bytes'),
            'drop_csv_block': lambda d: d.pop('csv'),
            'drop_domain': lambda d: d.pop('domain'),
            'drop_time': lambda d: d.pop('time_days'),
            'drop_initial': lambda d: d.pop('initial'),
            'initial_not_bool': lambda d: d.update(initial='false'),
            'resolved_contradicts_csv': lambda d: d.update(establishment_resolved=False),
            'time_not_finite_type': lambda d: d.update(time_days='soon'),
        }
        mp = state.metadata_path(self.path)
        good = read_text(mp)                                     # the pristine fixture companion written by setUp
        for name, fn in edits.items():
            with self.subTest(name):
                # restore the pristine companion DIRECTLY in the temp test directory (never through write_plant_csv:
                # the production overwrite guard correctly refuses to replace a companion whose schema was corrupted)
                write_text(mp, good)
                edit_companion(self.path, fn)
                with self.assertRaises(state.StateError):
                    state.read_plant_csv(self.path, self.cat)
        write_text(mp, good)
        self.assertEqual(len(state.read_plant_csv(self.path, self.cat)), 6)       # the pristine pair is accepted again
        for name, text in (('not_json', '{this is not json'), ('duplicate_key', good.replace('{\n', '{\n  "schema": "x",\n', 1)),
                           ('nan', good.replace('"time_days": 12.5', '"time_days": NaN')), ('list', '[1, 2]\n'), ('empty', '')):
            with self.subTest(name):
                write_text(mp, text)
                with self.assertRaises(state.StateError):
                    state.read_plant_csv(self.path, self.cat)
        write_text(mp, good)
        self.assertEqual(len(state.read_plant_csv(self.path, self.cat)), 6)

    def test_catalog_time_and_grid_guards(self):
        with self.assertRaises(state.StateError):                                       # another catalog
            state.read_plant_csv(self.path, other_catalog())
        with self.assertRaises(state.StateError):                                       # another grid supplied by the caller
            state.read_plant_csv(self.path, self.cat, domain=state.Domain(20, 21, 1.0))
        with self.assertRaises(state.StateError):
            state.read_plant_csv(self.path, self.cat, domain=state.Domain(20, 20, 1.0, origin_x_m=1.0))
        with self.assertRaises(state.StateError):                                       # another clock supplied by the caller
            state.read_plant_csv(self.path, self.cat, time_days=0.0)
        ok = state.read_plant_csv(self.path, self.cat, domain=DOM, time_days=12.5)
        self.assertEqual((ok.time_days, ok.domain), (12.5, DOM))
        # a deliberate hand edit of the companion time is the declared context; a caller clock that disagrees is refused
        edit_companion(self.path, lambda d: d.update(time_days=40.0))
        self.assertEqual(state.read_plant_csv(self.path, self.cat).time_days, 40.0)
        with self.assertRaises(state.StateError):
            state.read_plant_csv(self.path, self.cat, time_days=12.5)

    def test_missing_companion_reads_as_a_plain_list_that_needs_caller_context(self):
        os.unlink(state.metadata_path(self.path))
        with self.assertRaises(state.StateError) as cm:
            state.read_plant_csv(self.path, self.cat)
        self.assertIn('Domain', str(cm.exception))
        s = state.read_plant_csv(self.path, self.cat, domain=DOM)
        self.assertEqual((s.time_days, s.n, s.provenance), (0.0, 6, {}))                 # no companion: time defaults to 0
        self.assertEqual(state.read_plant_csv(self.path, self.cat, domain=DOM, time_days=3.0).time_days, 3.0)
        with self.assertRaises(FileNotFoundError):
            state.read_plant_csv(os.path.join(self.td.name, 'no_such.csv'), self.cat, domain=DOM)

    def test_orphan_companion_without_csv_fails_loudly(self):
        os.unlink(self.path)
        self.assertTrue(os.path.isfile(state.metadata_path(self.path)))
        with self.assertRaises(FileNotFoundError):
            state.read_plant_csv(self.path, self.cat)
        with self.assertRaises(state.StateError):
            state.read_plant_csv(state.metadata_path(self.path), self.cat, domain=DOM)  # a companion is not a plant CSV


# =============================================================== plain initial input with caller context + SURFACE guards
class TestInitialInputAndSurface(unittest.TestCase):
    def setUp(self):
        self.cat = catalog()
        self.later = state.constant_surface(DOM, 55.0)

    def test_plain_initial_list_with_caller_context(self):
        plain = ('plant_id,species,x_m,y_m,establishment_ground_m,agb_dry_kg,seasonal_agb_dry_kg,age_days,status\n'
                 '1,grass,2.5,3.5,SURFACE,0.12,0.0,730.0,live\n'
                 '2,shrub,10.0,10.0,99.9,2.5,0.0,3650.0,live\n')
        with self.assertRaises(state.StateError):
            state.parse_plant_csv(plain, self.cat)                                         # no companion, no domain
        s = state.parse_plant_csv(plain, self.cat, domain=DOM, time_days=4.0, surface=self.later)
        self.assertEqual((s.time_days, s.domain, s.establishment_resolved), (4.0, DOM, True))
        self.assertEqual(s.establishment_ground_m[s.indices_of([1])[0]], 55.0)             # resolved ONCE from the caller surface
        self.assertEqual(s.establishment_ground_m[s.indices_of([2])[0]], 99.9)
        unresolved = state.parse_plant_csv(plain, self.cat, domain=DOM)
        self.assertFalse(unresolved.establishment_resolved)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'hand.csv')
            write_text(path, plain)
            self.assertFalse(state.read_plant_csv(path, self.cat, domain=DOM).establishment_resolved)
            self.assertEqual(os.listdir(td), ['hand.csv'])                                   # reading creates no companion

    def test_initial_companion_allows_surface_and_resolution_happens_once(self):
        s = six(self.cat, resolved=False, provenance={'stage': 'initial'})
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'initial_plants.csv')
            state.write_plant_csv(s, path, initial=True)
            self.assertIn('SURFACE', read_text(path))
            self.assertTrue(os.path.isfile(os.path.join(td, 'initial_plants.metadata.json')))
            meta = json.loads(read_text(os.path.join(td, 'initial_plants.metadata.json')))
            self.assertEqual((meta['initial'], meta['establishment_resolved']), (True, False))
            raw = state.read_plant_csv(path, self.cat)
            self.assertFalse(raw.establishment_resolved)
            res = state.read_plant_csv(path, self.cat, surface=self.later)
            self.assertTrue(res.establishment_resolved)
            self.assertEqual(res.establishment_ground_m[res.indices_of([1])[0]], 55.0)
            self.assertEqual(res.establishment_ground_m[res.indices_of([3])[0]], 99.9)
            self.assertNotIn('SURFACE', state.plant_csv_text(res))                          # a saved snapshot carries concrete anchors

    def test_surface_in_a_file_not_declared_initial_is_refused_before_any_surface_is_sampled(self):
        s = six(self.cat, resolved=False)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'x.csv')
            state.write_plant_csv(s, path, initial=True)
            for name, fn in (('initial_false', lambda d: d.update(initial=False)),
                             ('resolved_true', lambda d: d.update(establishment_resolved=True))):
                with self.subTest(name):
                    state.write_plant_csv(s, path, initial=True, overwrite=True)
                    edit_companion(path, fn)
                    with mock.patch.object(state.Domain, 'sample', side_effect=AssertionError('surface sampled before refusal')):
                        with self.assertRaises(state.StateError):
                            state.read_plant_csv(path, self.cat, surface=self.later)
                        with self.assertRaises(state.StateError):
                            state.read_plant_csv(path, self.cat)

    def test_unresolved_state_cannot_be_written_or_described_as_a_snapshot(self):
        s = six(self.cat, resolved=False)
        with self.assertRaises(state.StateError):
            state.plant_csv_text(s)
        with self.assertRaises(state.StateError):
            state.plant_metadata(s, 'x')
        with self.assertRaises(state.StateError):
            state.plant_files_text(s)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(state.StateError):
                state.write_plant_csv(s, os.path.join(td, 'snap.csv'))
            self.assertEqual(os.listdir(td), [])                                             # neither file, no temp file
            state.write_plant_csv(s, os.path.join(td, 'initial.csv'), initial=True)          # initial files are allowed
            self.assertEqual(sorted(os.listdir(td)), ['initial.csv', 'initial.metadata.json'])


# =============================================================== legacy embedded-metadata READ compatibility
class TestLegacyReads(unittest.TestCase):
    def setUp(self):
        self.cat = catalog()
        self.later = state.constant_surface(DOM, 55.0)

    def test_legacy_embedded_snapshot_is_still_read_with_its_context(self):
        s = six(self.cat, time_days=33.0)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'old.csv')
            write_text(path, legacy_text(s))
            back = state.read_plant_csv(path, self.cat)
            for k in state.PlantState.ARRAY_NAMES:
                np.testing.assert_array_equal(getattr(back, k), getattr(s, k), err_msg=k)
            self.assertEqual((back.time_days, back.domain, back.provenance), (33.0, DOM, {'legacy': True}))
            self.assertEqual(os.listdir(td), ['old.csv'])                                    # reading never writes anything
        self.assertEqual(len(state.parse_plant_csv(legacy_text(s).encode('utf-8'), self.cat)), 6)

    def test_legacy_checks_are_kept(self):
        s = six(self.cat)
        good = legacy_text(s)
        for name, text, kw in (
                ('catalog', good, {}),
                ('time', good, {'time_days': 1.0}),
                ('domain', good, {'domain': state.Domain(20, 21, 1.0)}),
                ('truncated', ''.join(good.splitlines(True)[:-1]), {}),
                ('bad_version', legacy_text(s, schema_version=7), {}),
                ('bad_schema', legacy_text(s, schema='leaf.other'), {}),
                ('bad_units', legacy_text(s, units={'x_m': 'ft'}), {}),
                ('not_json_comment', '# this is just a comment\n' + state.plant_csv_text(s), {}),
                ('non_object', '# [1, 2]\n' + state.plant_csv_text(s), {})):
            with self.subTest(name):
                cat = other_catalog() if name == 'catalog' else self.cat
                with self.assertRaises(state.StateError):
                    state.parse_plant_csv(text, cat, **kw)

    def test_legacy_surface_rules(self):
        unres = six(self.cat, resolved=False)
        ok = state.parse_plant_csv(legacy_text(unres, csv_initial=True), self.cat, surface=self.later)
        self.assertEqual(ok.establishment_ground_m[ok.indices_of([1])[0]], 55.0)
        for name, text in (('not_initial', legacy_text(unres, csv_initial=True, initial=False)),
                           ('undeclared', legacy_text(unres, csv_initial=True, initial=None)),
                           ('resolved_flag', legacy_text(unres, csv_initial=True, establishment_resolved=True))):
            with self.subTest(name):
                with mock.patch.object(state.Domain, 'sample', side_effect=AssertionError('sampled before refusal')):
                    with self.assertRaises(state.StateError):
                        state.parse_plant_csv(text, self.cat, surface=self.later)

    def test_legacy_line_with_companion_is_ambiguous_and_refused(self):
        s = six(self.cat)
        text, meta = state.plant_csv_text(s), state.plant_metadata(s, state.plant_csv_text(s))
        with self.assertRaises(state.StateError):
            state.parse_plant_csv(legacy_text(s), self.cat, metadata=meta)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'old.csv')
            write_text(path, legacy_text(s))
            write_text(state.metadata_path(path), state.plant_metadata_text(state.plant_metadata(s, text)))
            with self.assertRaises(state.StateError):
                state.read_plant_csv(path, self.cat)
            # an explicit opt-out ignores the stray companion and reads the legacy embedded form on its own
            self.assertEqual(len(state.read_plant_csv(path, self.cat, metadata=False)), 6)


# =============================================================== two-file write safety
class TestTwoFileWrites(unittest.TestCase):
    def setUp(self):
        self.cat = catalog()
        self.s = six(self.cat, time_days=1.5)
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.dir = self.td.name

    def listing(self):
        return sorted(os.listdir(self.dir))

    def test_no_clobber_of_csv_or_companion(self):
        path = os.path.join(self.dir, 'a.csv')
        mp = state.metadata_path(path)
        state.write_plant_csv(self.s, path)
        before = (read_bytes(path), read_bytes(mp))
        with self.assertRaises(FileExistsError):
            state.write_plant_csv(self.s, path)
        self.assertEqual((read_bytes(path), read_bytes(mp)), before)                     # neither file touched
        # an existing companion alone (CSV absent) also blocks the write ...
        os.unlink(path)
        with self.assertRaises(FileExistsError):
            state.write_plant_csv(self.s, path)
        self.assertFalse(os.path.exists(path))
        self.assertEqual(read_bytes(mp), before[1])
        # ... while overwrite=True may replace a LEAF companion (an orphan of an interrupted write)
        state.write_plant_csv(self.s, path, overwrite=True)
        self.assertEqual(read_bytes(path), before[0])
        self.assertEqual(len(state.read_plant_csv(path, self.cat)), 6)

    def test_unrelated_metadata_is_never_overwritten(self):
        path = os.path.join(self.dir, 'b.csv')
        mp = state.metadata_path(path)
        for content in ('{"unrelated": true}\n', '{"schema": "something.else"}\n', 'not json at all', '[]'):
            write_text(mp, content)
            for ow in (False, True):
                with self.assertRaises(FileExistsError, msg=(content, ow)):
                    state.write_plant_csv(self.s, path, overwrite=ow)
                self.assertFalse(os.path.exists(path))
                self.assertEqual(read_text(mp), content)
            os.unlink(mp)
        os.mkdir(mp)                                                                       # a directory in the way
        with self.assertRaises(OSError):
            state.write_plant_csv(self.s, path)
        self.assertFalse(os.path.exists(path))

    def test_preflight_creates_nothing_and_checks_both_destinations(self):
        path = os.path.join(self.dir, 'c.csv')
        self.assertEqual(state.preflight_plant_files(path), (path, os.path.join(self.dir, 'c.metadata.json')))
        self.assertEqual(self.listing(), [])
        with self.assertRaises(FileNotFoundError):
            state.preflight_plant_files(os.path.join(self.dir, 'missing', 'c.csv'))
        with self.assertRaises(FileNotFoundError):
            state.write_plant_csv(self.s, os.path.join(self.dir, 'missing', 'c.csv'))
        self.assertEqual(self.listing(), [])

    def test_injected_failure_publishing_the_csv_leaves_no_new_output(self):
        path = os.path.join(self.dir, 'd.csv')
        wrapper, calls = fail_nth_atomic_write(2)                  # 1 = companion, 2 = CSV
        with mock.patch.object(state, 'atomic_write_bytes', wrapper):
            with self.assertRaises(OSError):
                state.write_plant_csv(self.s, path)
        self.assertEqual([os.path.basename(c) for c in calls], ['d.metadata.json', 'd.csv'])
        self.assertEqual(self.listing(), [])                       # the newly created companion was removed, no temp files
        state.write_plant_csv(self.s, path)                        # a clean retry works
        self.assertEqual(self.listing(), ['d.csv', 'd.metadata.json'])

    def test_injected_failure_publishing_the_companion_writes_nothing(self):
        path = os.path.join(self.dir, 'e.csv')
        wrapper, calls = fail_nth_atomic_write(1)
        with mock.patch.object(state, 'atomic_write_bytes', wrapper):
            with self.assertRaises(OSError):
                state.write_plant_csv(self.s, path)
        self.assertEqual(len(calls), 1)                            # the CSV publication was never attempted
        self.assertEqual(self.listing(), [])

    def test_injected_failure_on_overwrite_restores_the_previous_pair(self):
        path = os.path.join(self.dir, 'f.csv')
        mp = state.metadata_path(path)
        state.write_plant_csv(self.s, path)
        before = (read_bytes(path), read_bytes(mp))
        newer = six(self.cat, time_days=77.0)
        wrapper, calls = fail_nth_atomic_write(2)
        with mock.patch.object(state, 'atomic_write_bytes', wrapper):
            with self.assertRaises(OSError):
                state.write_plant_csv(newer, path, overwrite=True)
        self.assertEqual(len(calls), 3)                            # companion, failed CSV, restore of the old companion
        self.assertEqual((read_bytes(path), read_bytes(mp)), before)
        self.assertEqual(state.read_plant_csv(path, self.cat).time_days, 1.5)             # still the consistent OLD pair
        self.assertEqual(self.listing(), ['f.csv', 'f.metadata.json'])

    def test_a_pair_interrupted_between_the_two_publications_is_detected_on_read(self):
        '''Two renames are not one atomic step: simulate the process dying after the new companion was published but
        before the new CSV replaced the old one. The reader refuses the pair instead of using it.'''
        path = os.path.join(self.dir, 'g.csv')
        state.write_plant_csv(self.s, path)
        newer = changed(self.cat, 88.0)
        csv_text, meta_text = state.plant_files_text(newer, csv_file='g.csv')
        self.assertNotEqual(csv_text, read_text(path))
        write_text(state.metadata_path(path), meta_text)           # step 1 happened, step 2 (the CSV) did not
        with self.assertRaises(state.StateError):
            state.read_plant_csv(path, self.cat)
        write_text(path, csv_text)                                 # step 2 completed: the new pair is consistent
        self.assertEqual(state.read_plant_csv(path, self.cat).time_days, 88.0)

    def test_hard_link_failure_refuses_without_leaving_files(self):
        path = os.path.join(self.dir, 'h.csv')
        real_link = os.link

        def no_link(src, dst, *a, **k):
            raise OSError(95, 'Operation not supported')
        os.link = no_link
        try:
            with self.assertRaises(OSError):
                state.write_plant_csv(self.s, path)
        finally:
            os.link = real_link
        self.assertEqual(self.listing(), [])


# =============================================================== snapshot writer, CLI, preview helper
class TestSnapshotsAndCLI(unittest.TestCase):
    def setUp(self):
        self.cat = catalog()
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.dir = self.td.name

    def test_snapshot_writer_writes_plain_csv_plus_companion_and_rejects_surface(self):
        s = six(self.cat)
        w = recording.SnapshotWriter(os.path.join(self.dir, 'veg_{index:03d}_{kind}.csv'), cadence_days=10.0,
                                     provenance={'run': 'unit'})
        p0 = w.write_initial(s)
        s.advance_time(10.0)
        p1 = w.maybe_write(s)
        s.advance_time(3.0)
        p2 = w.write_final(s)
        for p, kind, t in ((p0, 'initial', 0.0), (p1, 'cadence', 10.0), (p2, 'final', 13.0)):
            assert_plain_csv(self, read_text(p), 6)
            meta = json.loads(read_text(state.metadata_path(p)))
            self.assertEqual((meta['time_days'], meta['n_plants'], meta['initial']), (t, 6, False))
            self.assertEqual(meta['provenance']['snapshot_kind'], kind)
            self.assertEqual(meta['provenance']['run'], 'unit')
            self.assertEqual(state.read_plant_csv(p, self.cat).time_days, t)
        self.assertEqual([r['metadata_path'] for r in w.records], [state.metadata_path(p) for p in (p0, p1, p2)])
        self.assertEqual(w.manifest()['records'][0]['metadata_path'], state.metadata_path(p0))
        unres = six(self.cat, resolved=False)
        unres.time_days = 99.0                                     # later than the last snapshot: only SURFACE can refuse
        before = os.listdir(self.dir)
        with self.assertRaises(state.StateError):
            w.write_final(unres)                                   # snapshots never carry SURFACE
        self.assertEqual(os.listdir(self.dir), before)

    def test_snapshot_writer_failed_pair_write_changes_no_bookkeeping(self):
        s = six(self.cat)
        w = recording.SnapshotWriter(os.path.join(self.dir, 'v_{index}.csv'), cadence_days=7.0)
        manifest = w.manifest()
        wrapper, _ = fail_nth_atomic_write(2)
        with mock.patch.object(state, 'atomic_write_bytes', wrapper):
            with self.assertRaises(OSError):
                w.write_initial(s)
        self.assertEqual(w.manifest(), manifest)
        self.assertIsNone(w.start_time_days)
        self.assertEqual(os.listdir(self.dir), [])
        p = w.write_initial(s)                                     # the retry succeeds and then advances the writer
        self.assertEqual(w.start_time_days, 0.0)
        self.assertEqual(os.path.basename(p), 'v_0.csv')

    def test_cli_generate_and_inspect_use_the_companion(self):
        cfg = os.path.join(self.dir, 'gen.json')
        write_text(cfg, json.dumps(generation_config()))
        out, rep = os.path.join(self.dir, 'pop.csv'), os.path.join(self.dir, 'pop_report.json')
        code, text, err = run_cli(['generate', '--species', str(SPECIES_JSON), '--config', cfg, '--out', out, '--report', rep,
                                   '--surface-constant', '5.0'])
        self.assertEqual(code, 0, err)
        brief = json.loads(text)
        self.assertEqual(brief['metadata_written'], os.path.join(self.dir, 'pop.metadata.json'))
        assert_plain_csv(self, read_text(out), brief['n_plants'])
        meta = json.loads(read_text(brief['metadata_written']))
        self.assertEqual((meta['n_plants'], meta['initial'], meta['establishment_resolved']), (brief['n_plants'], False, True))
        self.assertEqual(meta['provenance']['command'], 'leaf generate')
        self.assertEqual(meta['csv']['sha256'], state._sha256(read_bytes(out)))
        self.assertEqual(json.loads(read_text(rep))['metadata_written'], brief['metadata_written'])
        # inspect needs no domain: the companion supplies it; a caller domain that disagrees is refused
        code, text, err = run_cli(['inspect', '--species', str(SPECIES_JSON), '--plants', out])
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(text)['summary']['n_plants'], brief['n_plants'])
        self.assertEqual(json.loads(text)['summary']['domain']['nx'], 32)
        code, _, err = run_cli(['inspect', '--species', str(SPECIES_JSON), '--plants', out, '--nx', '33', '--ny', '32', '--dx-m', '1.0'])
        self.assertEqual(code, 1)
        self.assertIn('differs', err)
        # an edited CSV is refused by every reader
        original = read_text(out)
        write_text(out, original + '\n')
        code, _, err = run_cli(['inspect', '--species', str(SPECIES_JSON), '--plants', out])
        self.assertEqual(code, 1)
        self.assertIn('match', err)
        write_text(out, original)
        # without a surface the generated file is an INITIAL input (SURFACE kept, companion initial = true)
        raw = os.path.join(self.dir, 'raw.csv')
        code, _, err = run_cli(['generate', '--species', str(SPECIES_JSON), '--config', cfg, '--out', raw])
        self.assertEqual(code, 0, err)
        raw_meta = json.loads(read_text(os.path.join(self.dir, 'raw.metadata.json')))
        self.assertEqual((raw_meta['initial'], raw_meta['establishment_resolved']), (True, False))
        self.assertIn('SURFACE', read_text(raw))
        assert_plain_csv(self, read_text(raw), raw_meta['n_plants'])

    def test_cli_generate_collisions_and_injected_failure_roll_back_everything(self):
        cfg = os.path.join(self.dir, 'gen.json')
        write_text(cfg, json.dumps(generation_config()))
        base = ['generate', '--species', str(SPECIES_JSON), '--config', cfg]
        out = os.path.join(self.dir, 'x.csv')
        for name, report in (('out', out), ('companion', os.path.join(self.dir, 'x.metadata.json'))):
            code, _, err = run_cli(base + ['--out', out, '--report', report, '--overwrite'])
            self.assertEqual(code, 1, name)
            self.assertEqual(sorted(os.listdir(self.dir)), ['gen.json'], name)             # refused before any output
        # a failure while publishing the CSV removes the report AND the companion that were created
        wrapper, calls = fail_nth_atomic_write(3)                  # 1 = report, 2 = companion, 3 = CSV
        with mock.patch.object(state, 'atomic_write_bytes', wrapper):
            code, _, err = run_cli(base + ['--out', out, '--report', os.path.join(self.dir, 'x_report.json')])
        self.assertEqual(code, 1)
        self.assertIn('injected', err)
        self.assertEqual([os.path.basename(c) for c in calls], ['x_report.json', 'x.metadata.json', 'x.csv'])
        self.assertEqual(sorted(os.listdir(self.dir)), ['gen.json'])
        # an existing companion (unrelated or not) blocks the run before anything is generated or written
        write_text(os.path.join(self.dir, 'x.metadata.json'), '{"unrelated": true}\n')
        code, _, err = run_cli(base + ['--out', out, '--report', os.path.join(self.dir, 'x_report.json')])
        self.assertEqual(code, 1)
        self.assertIn('exists', err)
        self.assertEqual(sorted(os.listdir(self.dir)), ['gen.json', 'x.metadata.json'])
        self.assertEqual(read_text(os.path.join(self.dir, 'x.metadata.json')), '{"unrelated": true}\n')

    def test_cli_init_chain_from_a_plain_handwritten_list(self):
        hand = os.path.join(self.dir, 'hand.csv')
        write_text(hand, read_text(PLANTS_CSV))
        raw, resolved = os.path.join(self.dir, 'raw.csv'), os.path.join(self.dir, 'resolved.csv')
        code, text, err = run_cli(['init', '--species', str(SPECIES_JSON), '--plants', hand, '--nx', '20', '--ny', '20',
                                   '--dx-m', '1.0', '--time-days', '9.0', '--out', raw])
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(text)['metadata_written'], os.path.join(self.dir, 'raw.metadata.json'))
        self.assertEqual(sorted(os.listdir(self.dir)), ['hand.csv', 'raw.csv', 'raw.metadata.json'])
        assert_plain_csv(self, read_text(raw), json.loads(read_text(os.path.join(self.dir, 'raw.metadata.json')))['n_plants'])
        # the second step needs no domain / time: the companion of the plain INITIAL file carries them
        code, _, err = run_cli(['init', '--species', str(SPECIES_JSON), '--plants', raw, '--surface-constant', '100.0',
                                '--out', resolved])
        self.assertEqual(code, 0, err)
        s = state.read_plant_csv(resolved, self.cat)
        self.assertTrue(s.establishment_resolved)
        self.assertEqual((s.time_days, s.domain), (9.0, DOM))
        self.assertEqual(s.establishment_ground_m[s.indices_of([1])[0]], 100.0)
        self.assertEqual(s.establishment_ground_m[s.indices_of([3])[0]], 99.9)
        meta = json.loads(read_text(os.path.join(self.dir, 'resolved.metadata.json')))
        self.assertEqual((meta['initial'], meta['provenance']['command']), (False, 'leaf init'))
        # collisions: either destination blocks the command before anything is written
        before = sorted(os.listdir(self.dir))
        code, _, err = run_cli(['init', '--species', str(SPECIES_JSON), '--plants', raw, '--surface-constant', '100.0',
                                '--out', resolved])
        self.assertEqual(code, 1)
        self.assertIn('exists', err)
        os.unlink(resolved)
        code, _, err = run_cli(['init', '--species', str(SPECIES_JSON), '--plants', raw, '--surface-constant', '100.0',
                                '--out', resolved])
        self.assertEqual(code, 1)                                  # the orphan companion of the removed CSV still blocks
        self.assertEqual(sorted(os.listdir(self.dir)), [n for n in before if n != 'resolved.csv'])
        # a conflicting caller domain against the companion is refused and writes nothing
        code, _, err = run_cli(['init', '--species', str(SPECIES_JSON), '--plants', raw, '--nx', '21', '--ny', '20',
                                '--dx-m', '1.0', '--out', os.path.join(self.dir, 'conflict.csv')])
        self.assertEqual(code, 1)
        self.assertFalse(any(n.startswith('conflict') for n in os.listdir(self.dir)))

    def test_preview_helper_declares_the_companion_as_an_output(self):
        if not PREVIEW_SCRIPT.is_file():
            self.skipTest('preview helper not present in this tree')
        spec = importlib.util.spec_from_file_location('leaf_case_render_preview', str(PREVIEW_SCRIPT))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertIn('initial_plants.csv', mod.OUTPUTS)
        self.assertIn(os.path.basename(state.metadata_path('initial_plants.csv')), mod.OUTPUTS)
        self.assertEqual(len(mod.OUTPUTS), 6)
        self.assertFalse(hasattr(mod, 'rerender'))            # serialization-only task: no extra rendering mode


if __name__ == '__main__':
    unittest.main()
