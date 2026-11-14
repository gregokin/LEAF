'''Species catalog v2: optional per-species preferential-infiltration receiving weights (unittest, NumPy only).

Run from the candidate root:  python -m unittest tests.test_species_hydrology_traits -v
Scope: the TRAIT ONLY (raw specified weights, validation, versioning, merge, state compatibility). No cover, g, infiltration or
physics mapping exists or is tested here; SYRUP normalises positive vectors later.
'''
from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import tempfile
import unittest

import numpy as np

from leaf import state, traits

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPECIES_JSON = ROOT / 'examples' / 'phase0' / 'species.json'
KEY = traits.RECEIVER_WEIGHTS_KEY


def base_doc(layers=2):
    doc = json.loads(SPECIES_JSON.read_text())
    if layers != doc['root_layers']:
        doc['root_layers'] = layers
        for rec in doc['species'].values():
            rec['roots']['layer_fractions'] = [1.0 / layers] * layers
    return doc


def v2_doc(layers=2, weights=None):
    doc = base_doc(layers)
    doc['schema_version'] = 2
    for sid, w in (weights or {}).items():
        doc['species'][sid]['hydrology'] = {KEY: list(w)}
    return doc


def write(path, doc):
    path.write_text(json.dumps(doc))
    return str(path)


class V1Unchanged(unittest.TestCase):
    def test_v1_example_is_bit_exact_and_has_no_trait(self):
        text = SPECIES_JSON.read_text()
        cat = traits.SpeciesCatalog.from_text(text)
        doc = json.loads(text)
        expected = 'sha256:' + hashlib.sha256(traits.canonical_json(doc).encode('ascii')).hexdigest()
        self.assertEqual(cat.hash, expected)
        self.assertEqual(cat.schema_version, 1)
        self.assertEqual(cat.metadata()['schema_version'], 1)
        self.assertEqual(traits.SCHEMA_VERSION, 1)
        self.assertEqual(cat.species_ids, ('grass', 'shrub'))
        self.assertFalse(cat.has_preferential_receiver_weights('grass'))
        with self.assertRaisesRegex(traits.CatalogError, 'no .*trait'):
            cat.preferential_receiver_weights('grass')
        with self.assertRaisesRegex(traits.CatalogError, 'no .*trait'):
            cat.preferential_receiver_weight_table()

    def test_v1_refuses_the_hydrology_block_and_float_one_is_still_accepted(self):
        doc = base_doc()
        doc['species']['grass']['hydrology'] = {KEY: [0, 1]}
        with self.assertRaisesRegex(traits.CatalogError, 'unknown keys'):
            traits.SpeciesCatalog(doc)
        doc = base_doc()
        doc['schema_version'] = 1.0                           # accepted exactly as before this change
        self.assertEqual(traits.SpeciesCatalog(doc).schema_version, 1)

    def test_v1_core_arrays_unchanged_by_the_v2_extension(self):
        a = traits.SpeciesCatalog(base_doc())
        b = traits.SpeciesCatalog(v2_doc(weights={'grass': [0, 1], 'shrub': [1, 1]}))
        for name in ('allometry_arrays', 'root_layer_table', 'porosity_array', 'root_multiplier_array'):
            x, y = getattr(a, name)(), getattr(b, name)()
            if isinstance(x, dict):
                self.assertEqual(sorted(x), sorted(y))
                for k in x:
                    np.testing.assert_array_equal(x[k], y[k]) if isinstance(x[k], np.ndarray) else self.assertEqual(x[k], y[k])
            else:
                np.testing.assert_array_equal(x, y)
        self.assertNotEqual(a.hash, b.hash)                   # a v2 document is a different document


class V2Trait(unittest.TestCase):
    def check(self, layers, grass, shrub):
        cat = traits.SpeciesCatalog(v2_doc(layers, {'grass': grass, 'shrub': shrub}))
        self.assertEqual(cat.schema_version, 2)
        self.assertEqual(cat.metadata()['schema_version'], 2)
        np.testing.assert_array_equal(cat.preferential_receiver_weights('grass'), np.array(grass, float))
        tab = cat.preferential_receiver_weight_table()
        self.assertEqual(tab.shape, (2, layers))
        np.testing.assert_array_equal(tab, np.array([grass, shrub], float))     # sorted table order: grass, shrub
        np.testing.assert_array_equal(cat.preferential_receiver_weight_table(['shrub']), np.array([shrub], float))
        return cat

    def test_one_two_three_and_four_layers(self):
        self.check(1, [0.0], [3.5])
        self.check(2, [0, 1], [0.25, 0.75])
        self.check(3, [0, 1, 1], [2.0, 0.0, 6.0])
        self.check(4, [1, 2, 3, 4], [0, 0, 0, 0])

    def test_raw_values_are_returned_not_normalised_and_all_zero_means_disabled_data(self):
        cat = self.check(3, [0, 1, 1], [0, 0, 0])
        self.assertEqual(float(cat.preferential_receiver_weights('grass').sum()), 2.0)      # not scaled to 1
        self.assertEqual(float(cat.preferential_receiver_weights('shrub').sum()), 0.0)      # explicit all-zero stays all-zero
        self.assertTrue(cat.has_preferential_receiver_weights('shrub'))

    def test_huge_finite_values_are_returned_raw_without_a_sum(self):
        big = 1.7e308
        cat = traits.SpeciesCatalog(v2_doc(2, {'grass': [big, big], 'shrub': [0, 1]}))
        np.testing.assert_array_equal(cat.preferential_receiver_weights('grass'), np.array([big, big]))

    def test_trait_is_per_species_optional_and_not_inferred(self):
        cat = traits.SpeciesCatalog(v2_doc(2, {'grass': [0, 1]}))                          # shrub has no block
        self.assertTrue(cat.has_preferential_receiver_weights('grass'))
        self.assertFalse(cat.has_preferential_receiver_weights('shrub'))
        np.testing.assert_array_equal(cat.preferential_receiver_weights('grass'), [0.0, 1.0])
        with self.assertRaisesRegex(traits.CatalogError, "species 'shrub' has no"):
            cat.preferential_receiver_weights('shrub')
        with self.assertRaisesRegex(traits.CatalogError, "species 'shrub' has no"):
            cat.preferential_receiver_weight_table()
        with self.assertRaises(traits.CatalogError):
            cat.preferential_receiver_weights('unknown')
        self.assertEqual(cat.preferential_receiver_weight_table(['grass']).tolist(), [[0.0, 1.0]])
        # root fractions are a separate trait and never become weights
        self.assertNotEqual(cat.root_layer_fractions('shrub'), None)

    def test_invalid_declarations_are_refused(self):
        L = 3
        cases = {
            'wrong length': [0, 1],
            'too long': [0, 1, 1, 1],
            'negative': [0, -1e-300, 1],
            'nan': [float('nan'), 1, 1],
            'bool': [True, 0, 1],
            'string': ['1', 0, 1],
            'null entry': [None, 0, 1],
            'nested': [[0], 1, 1],
        }
        for label, w in cases.items():
            doc = v2_doc(L)
            doc['species']['grass']['hydrology'] = {KEY: w}
            with self.subTest(label), self.assertRaises(traits.CatalogError):
                traits.SpeciesCatalog(doc) if label != 'nan' else traits.SpeciesCatalog.from_text(json.dumps(doc).replace('NaN', 'NaN'))
        for label, hy in {'null block': None, 'empty block': {}, 'unknown key': {KEY: [0, 1, 1], 'weights': [1, 1, 1]}, 'null weights': {KEY: None},
                          'scalar weights': {KEY: 1.0}, 'list block': [0, 1, 1]}.items():
            doc = v2_doc(L)
            doc['species']['grass']['hydrology'] = hy
            with self.subTest(label), self.assertRaises(traits.CatalogError):
                traits.SpeciesCatalog(doc)
        doc = v2_doc(L, {'grass': [0, 1, 1]})
        doc['species']['grass']['hydrology']['notes'] = 'documented'      # the only other documented key
        traits.SpeciesCatalog(doc)
        doc['species']['grass']['hydrology']['notes'] = {'x': float('inf')}
        with self.assertRaises(traits.CatalogError):
            traits.SpeciesCatalog(doc)

    def test_version_declarations(self):
        for ver in (True, 2.0, '2', None, 3, 0, [2], 1.5):
            doc = v2_doc(2, {'grass': [0, 1]})
            doc['schema_version'] = ver
            with self.subTest(ver), self.assertRaisesRegex(traits.CatalogError, 'unsupported species catalog schema_version'):
                traits.SpeciesCatalog(doc)
        doc = v2_doc(2, {'grass': [0, 1]})
        del doc['schema_version']
        with self.assertRaises(traits.CatalogError):
            traits.SpeciesCatalog(doc)
        self.assertEqual(traits.document_version(v2_doc()), 2)

    def test_species_functional_type_does_not_enable_or_disable(self):
        doc = v2_doc(2, {'grass': [0, 0], 'shrub': [0, 1]})
        doc['species']['grass']['functional_type'] = 'shrub'
        doc['species']['shrub']['functional_type'] = 'perennial_bunchgrass'
        cat = traits.SpeciesCatalog(doc)
        self.assertEqual(cat.preferential_receiver_weights('grass').tolist(), [0.0, 0.0])
        self.assertEqual(cat.preferential_receiver_weights('shrub').tolist(), [0.0, 1.0])

    def test_returned_arrays_and_document_are_copies(self):
        cat = traits.SpeciesCatalog(v2_doc(2, {'grass': [0, 1], 'shrub': [1, 1]}))
        h0 = cat.hash
        w = cat.preferential_receiver_weights('grass')
        w[:] = 99.0
        t = cat.preferential_receiver_weight_table()
        t[:] = 77.0
        d = cat.document
        d['species']['grass']['hydrology'][KEY][0] = 5
        rec = cat.species_record('grass')
        rec['hydrology'][KEY][1] = 8
        self.assertEqual(cat.preferential_receiver_weights('grass').tolist(), [0.0, 1.0])
        self.assertEqual(cat.preferential_receiver_weight_table().tolist(), [[0.0, 1.0], [1.0, 1.0]])
        self.assertEqual(traits.catalog_hash(cat.document), h0)
        self.assertEqual(cat.hash, h0)
        for arr in (cat._rw_table, cat._rw_has):
            self.assertFalse(arr.flags.writeable)

    def test_weights_change_the_hash_and_json_key_order_does_not(self):
        a = traits.SpeciesCatalog(v2_doc(2, {'grass': [0, 1], 'shrub': [1, 1]}))
        b = traits.SpeciesCatalog(v2_doc(2, {'grass': [0, 1], 'shrub': [1, 2]}))
        self.assertNotEqual(a.hash, b.hash)
        d = v2_doc(2, {'grass': [0, 1], 'shrub': [1, 1]})
        reordered = {k: d[k] for k in reversed(list(d))}
        reordered['species'] = {k: reordered['species'][k] for k in reversed(list(reordered['species']))}
        self.assertEqual(traits.SpeciesCatalog(reordered).hash, a.hash)


class Correction01(unittest.TestCase):
    def num_cat(self):
        doc = v2_doc(2, {'grass': [0, 1], 'shrub': [1, 1]})
        doc['species']['1'] = copy.deepcopy(doc['species']['grass'])
        doc['species']['1']['hydrology'] = {KEY: [3, 4]}
        return traits.SpeciesCatalog(doc)

    def test_generic_section_serves_hydrology_for_v2_only(self):
        cat = traits.SpeciesCatalog(v2_doc(2, {'grass': [0, 1]}))
        self.assertEqual(cat.section('grass', 'hydrology'), {KEY: [0, 1]})
        self.assertIsNone(cat.section('shrub', 'hydrology'))             # optional and absent
        sec = cat.section('grass', 'hydrology')
        sec[KEY][0] = 9
        self.assertEqual(cat.section('grass', 'hydrology'), {KEY: [0, 1]})     # defensive deep copy
        self.assertEqual(cat.preferential_receiver_weights('grass').tolist(), [0.0, 1.0])
        v1 = traits.SpeciesCatalog(base_doc())
        with self.assertRaisesRegex(traits.CatalogError, 'unknown species section'):
            v1.section('grass', 'hydrology')
        for bad in ('nothing', 'Hydrology', 'preferential_infiltration_receiver_weights'):
            with self.assertRaisesRegex(traits.CatalogError, 'unknown species section'):
                cat.section('grass', bad)
        self.assertEqual(v1.section('grass', 'porosity')['value'], 0.2)        # existing sections unchanged

    def test_table_ids_are_strict_strings_without_coercion(self):
        cat = self.num_cat()
        self.assertIn('1', cat.species_ids)
        with self.assertRaises(traits.CatalogError):
            cat.preferential_receiver_weights(1)                     # individual getter refuses an int id
        self.assertEqual(cat.preferential_receiver_weight_table(['1']).tolist(), [[3.0, 4.0]])
        for label, arg in {'int id': [1], 'int tuple': (1,), 'True': [True], 'None element': [None], 'float': [1.0], 'numpy int': [np.int64(1)],
                           'bare string': 'grass', 'bare numeric string': '1', 'bare int': 1, 'bool': True, 'dict': {'grass': 1}, 'set': {'grass'},
                           'generator': (x for x in ['grass']), 'bytes': [b'grass'], 'nested': [['grass']], 'unknown': ['grass', 'nope'],
                           '2d array': np.array([['grass']])}.items():
            with self.subTest(label), self.assertRaises(traits.CatalogError):
                cat.preferential_receiver_weight_table(arg)

    def test_table_valid_vectors_empty_duplicates_and_order(self):
        cat = traits.SpeciesCatalog(v2_doc(3, {'grass': [0, 1, 1], 'shrub': [2, 0, 5]}))
        self.assertEqual(cat.preferential_receiver_weight_table().tolist(), [[0.0, 1.0, 1.0], [2.0, 0.0, 5.0]])
        self.assertEqual(cat.preferential_receiver_weight_table(['shrub', 'grass', 'shrub']).tolist(),
                         [[2.0, 0.0, 5.0], [0.0, 1.0, 1.0], [2.0, 0.0, 5.0]])            # order and repeats preserved
        self.assertEqual(cat.preferential_receiver_weight_table(('grass',)).tolist(), [[0.0, 1.0, 1.0]])
        self.assertEqual(cat.preferential_receiver_weight_table(np.array(['shrub'])).tolist(), [[2.0, 0.0, 5.0]])
        for empty in ([], ()):
            t = cat.preferential_receiver_weight_table(empty)
            self.assertEqual(t.shape, (0, 3))
            self.assertEqual(t.dtype, np.float64)
        t = cat.preferential_receiver_weight_table(['grass'])
        self.assertTrue(t.flags.writeable)
        self.assertFalse(np.shares_memory(t, cat._rw_table))                      # one fresh array, never a view of the stored table
        full = cat.preferential_receiver_weight_table()
        self.assertFalse(np.shares_memory(full, cat._rw_table))
        self.assertEqual((t.dtype, full.dtype), (np.float64, np.float64))
        t[:] = 5
        self.assertEqual(cat.preferential_receiver_weights('grass').tolist(), [0.0, 1.0, 1.0])

    def test_table_missing_trait_is_refused_before_allocation_and_v1_refuses(self):
        cat = traits.SpeciesCatalog(v2_doc(2, {'grass': [0, 1]}))
        with self.assertRaisesRegex(traits.CatalogError, "species 'shrub' has no"):
            cat.preferential_receiver_weight_table(['grass', 'shrub'])
        with self.assertRaisesRegex(traits.CatalogError, "species 'shrub' has no"):
            cat.preferential_receiver_weight_table()
        with self.assertRaisesRegex(traits.CatalogError, 'has no'):
            traits.SpeciesCatalog(base_doc()).preferential_receiver_weight_table(['grass'])
        self.assertEqual(cat.preferential_receiver_weight_table([]).shape, (0, 2))


class MergeFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = pathlib.Path(self.tmp.name)

    def split(self, version, weights):
        doc = v2_doc(2, weights) if version == 2 else base_doc(2)
        a, b = copy.deepcopy(doc), copy.deepcopy(doc)
        a['species'] = {'grass': doc['species']['grass']}
        b['species'] = {'shrub': doc['species']['shrub']}
        return a, b, doc

    def test_same_version_merges_deterministically_and_reports_the_actual_version(self):
        a, b, whole = self.split(2, {'grass': [0, 1], 'shrub': [1, 1]})
        pa, pb = write(self.dir / 'a.json', a), write(self.dir / 'b.json', b)
        c1, c2 = traits.SpeciesCatalog.from_files([pa, pb]), traits.SpeciesCatalog.from_files([pb, pa])
        self.assertEqual(c1.hash, c2.hash)
        self.assertEqual(c1.schema_version, 2)
        self.assertEqual(c1.metadata()['schema_version'], 2)
        self.assertEqual(c1.hash, traits.SpeciesCatalog(whole).hash)
        self.assertEqual(c1.preferential_receiver_weight_table().tolist(), [[0.0, 1.0], [1.0, 1.0]])
        self.assertEqual(c1.species_sources, {'grass': pa, 'shrub': pb})

    def test_v1_merge_is_unchanged(self):
        a, b, whole = self.split(1, {})
        c = traits.SpeciesCatalog.from_files([write(self.dir / 'a.json', a), write(self.dir / 'b.json', b)])
        self.assertEqual(c.schema_version, 1)
        self.assertEqual(c.hash, traits.SpeciesCatalog(whole).hash)
        self.assertEqual(json.loads(c.canonical_text())['schema_version'], 1)

    def test_mixed_versions_are_refused_without_migration(self):
        a, _, _ = self.split(1, {})
        _, b, _ = self.split(2, {'grass': [0, 1], 'shrub': [1, 1]})
        with self.assertRaisesRegex(traits.CatalogError, 'different schema versions'):
            traits.SpeciesCatalog.from_files([write(self.dir / 'a.json', a), write(self.dir / 'b.json', b)])

    def test_duplicate_species_and_layer_mismatch_still_refused(self):
        a, b, _ = self.split(2, {'grass': [0, 1], 'shrub': [1, 1]})
        b['species']['grass'] = a['species']['grass']
        with self.assertRaisesRegex(traits.CatalogError, 'defined in both'):
            traits.SpeciesCatalog.from_files([write(self.dir / 'a.json', a), write(self.dir / 'b.json', b)])
        a, b, _ = self.split(2, {'grass': [0, 1], 'shrub': [1, 1]})
        c3 = v2_doc(3, {'shrub': [0, 1, 1]})
        c3['species'] = {'shrub': c3['species']['shrub']}
        with self.assertRaisesRegex(traits.CatalogError, 'root_layers differ'):
            traits.SpeciesCatalog.from_files([write(self.dir / 'a.json', a), write(self.dir / 'c.json', c3)])

    def test_identical_v2_notes_metadata_is_retained_and_differing_is_omitted(self):
        a, b, _ = self.split(2, {'grass': [0, 1], 'shrub': [1, 1]})
        a['notes'] = b['notes'] = 'same'
        c = traits.SpeciesCatalog.from_files([write(self.dir / 'a.json', a), write(self.dir / 'b.json', b)])
        self.assertEqual(json.loads(c.canonical_text())['notes'], 'same')
        b['notes'] = 'other'
        c = traits.SpeciesCatalog.from_files([write(self.dir / 'a2.json', a), write(self.dir / 'b2.json', b)])
        self.assertNotIn('notes', json.loads(c.canonical_text()))


class StateCompatibility(unittest.TestCase):
    DOM = state.Domain(20, 20, 1.0)

    def records(self):
        return [{'plant_id': 1, 'species': 'grass', 'x_m': 3.0, 'y_m': 4.0, 'establishment_ground_m': 0.0, 'agb_dry_kg': 0.2,
                 'seasonal_agb_dry_kg': 0.0, 'age_days': 0.0, 'status': 'live'},
                {'plant_id': 2, 'species': 'shrub', 'x_m': 9.0, 'y_m': 10.0, 'establishment_ground_m': 0.0, 'agb_dry_kg': 2.0,
                 'seasonal_agb_dry_kg': 0.0, 'age_days': 5.0, 'status': 'live'}]

    def test_plantstate_csv_roundtrip_with_a_v2_catalog_and_hash_guard(self):
        cat = traits.SpeciesCatalog(v2_doc(2, {'grass': [0, 1], 'shrub': [0, 0]}))
        st = state.PlantState.from_records(cat, self.DOM, self.records(), time_days=3.0)
        before = (cat.hash, cat.preferential_receiver_weight_table().tobytes())
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'plants.csv')
            state.write_plant_files(st, path)
            back = state.read_plant_csv(path, cat)
            self.assertEqual(len(back), 2)
            self.assertEqual(back.catalog_hash, cat.hash)
            np.testing.assert_array_equal(back.agb_dry_kg, st.agb_dry_kg)
            # a catalog with different weights is a different hash: the state refuses it (the guard is not weakened)
            other = traits.SpeciesCatalog(v2_doc(2, {'grass': [1, 1], 'shrub': [0, 0]}))
            self.assertNotEqual(other.hash, cat.hash)
            with self.assertRaisesRegex((state.StateError, traits.CatalogError, ValueError), 'hash|catalog'):
                state.read_plant_csv(path, other)
            # the original v1 catalog also differs by hash
            with self.assertRaisesRegex((state.StateError, traits.CatalogError, ValueError), 'hash|catalog'):
                state.read_plant_csv(path, traits.SpeciesCatalog(base_doc()))
        self.assertEqual(before, (cat.hash, cat.preferential_receiver_weight_table().tobytes()))
        self.assertEqual(st.catalog_hash, cat.hash)


if __name__ == '__main__':
    unittest.main()
