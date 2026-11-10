'''LEAF Phase 0 acceptance + revision-2 regression tests (unittest, NumPy only; legacy-driver tests skip without numba).

Run from the candidate root:  python -m unittest tests.test_phase0 -v
'''
from __future__ import annotations

import copy
import io
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import tracemalloc
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout

import numpy as np

from leaf import canopy, events, initialization, recording, runtime, state, traits
from leaf.__main__ import main as cli_main

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / 'examples' / 'phase0'
SPECIES_JSON = EXAMPLES / 'species.json'
PLANTS_CSV = EXAMPLES / 'plants.csv'
GENERATION_JSON = EXAMPLES / 'generation.json'


def rec(pid, sp, x, y, est, B, E=0.0, age=0.0, status='live'):
    return {'plant_id': pid, 'species': sp, 'x_m': x, 'y_m': y, 'establishment_ground_m': est, 'agb_dry_kg': B,
            'seasonal_agb_dry_kg': E, 'age_days': age, 'status': status}


def example_catalog():
    return traits.SpeciesCatalog.from_file(SPECIES_JSON)


def consistent_catalog(mult=2.0):
    '''Catalog whose diameter LAW is the legacy area law B = pi r^2 * 1 kg m^-2 (b_d = 0.5, a_d = 2/sqrt(pi)).'''
    doc = example_catalog().document
    for sid in doc['species']:
        doc['species'][sid]['allometry']['diameter'] = {'a': 2.0 / math.sqrt(math.pi), 'b': 0.5}
        doc['species'][sid]['roots']['lateral_radius_multiplier'] = mult
    doc['catalog_id'] = 'consistent-with-legacy-area-law'
    return traits.SpeciesCatalog(doc)


def small_state(cat, resolved=True, time_days=0.0):
    '''Six plants: four live, one dead_standing, one removed; ids 1, 2, 4 use SURFACE, 3, 5, 6 explicit anchors.'''
    dom = state.Domain(20, 20, 1.0)
    recs = [rec(1, 'grass', 2.5, 3.5, 'SURFACE', 0.12, 0.0, 730.0),
            rec(2, 'grass', 7.25, 12.75, 'SURFACE', 0.08, 0.01, 365.0),
            rec(3, 'shrub', 10.0, 10.0, 99.9, 2.5, 0.0, 3650.0),
            rec(4, 'shrub', 15.5, 4.5, 'SURFACE', 1.8, 0.2, 1825.0),
            rec(5, 'grass', 18.0, 18.0, 100.0, 0.05, 0.0, 1095.0, 'dead_standing'),
            rec(6, 'shrub', 3.0, 16.0, 99.8, 0.0, 0.0, 5475.0, 'removed')]
    surface = state.constant_surface(dom, 100.0) if resolved else None
    return state.PlantState.from_records(cat, dom, recs, time_days=time_days, surface=surface)


def snapshot(s):
    return {k: getattr(s, k).copy() for k in state.PlantState.ARRAY_NAMES}


def assert_unchanged(tc, s, snap):
    for k in state.PlantState.ARRAY_NAMES:
        np.testing.assert_array_equal(getattr(s, k), snap[k], err_msg=k)


def assert_nonoverlap(tc, x, y, r):
    '''Row-wise exact check (test-only; production never does all pairs).'''
    for i in range(len(x) - 1):
        d = np.hypot(x[i + 1:] - x[i], y[i + 1:] - y[i])
        tc.assertTrue(bool(np.all(d >= r[i + 1:] + r[i] - 1e-12)), f'canopy overlap at row {i}')


def run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli_main(argv)
    return code, out.getvalue(), err.getvalue()


# =============================================================== species catalog
class TestCatalog(unittest.TestCase):
    def setUp(self):
        self.cat = example_catalog()

    def test_load_hash_stability_and_sorted_table(self):
        self.assertEqual(self.cat.species_ids, ('grass', 'shrub'))
        self.assertTrue(self.cat.hash.startswith('sha256:'))
        pretty = json.dumps(self.cat.document, indent=4, sort_keys=True)
        self.assertEqual(traits.SpeciesCatalog.from_text(pretty).hash, self.cat.hash)
        doc = self.cat.document
        reordered = dict(doc)
        reordered['species'] = {'shrub': doc['species']['shrub'], 'grass': doc['species']['grass']}  # JSON key order swapped
        cat2 = traits.SpeciesCatalog(reordered)
        self.assertEqual(cat2.hash, self.cat.hash)
        self.assertEqual(cat2.species_ids, self.cat.species_ids)   # deterministic name/index table
        self.assertEqual(cat2.index('shrub'), 1)
        doc['species']['grass']['allometry']['diameter']['a'] = 0.91
        self.assertNotEqual(traits.SpeciesCatalog(doc).hash, self.cat.hash)
        with self.assertRaises(traits.CatalogError):
            self.cat.check_species_table(('shrub', 'grass'))

    def test_document_is_a_defensive_copy(self):
        digest, d0 = self.cat.hash, self.cat.diameter_from_biomass('grass', 1.0)
        public = self.cat.document
        public['species']['grass']['allometry']['diameter']['a'] = 100.0
        self.assertEqual(self.cat.hash, digest)
        self.assertEqual(self.cat.diameter_from_biomass('grass', 1.0), d0)
        self.assertNotIn('100.0', self.cat.canonical_text())
        self.assertEqual(self.cat.document['species']['grass']['allometry']['diameter']['a'], 0.9)
        arrs = self.cat.allometry_arrays()
        arrs['a_d'][0] = 5.0                                   # returned arrays are copies ...
        self.assertEqual(self.cat.diameter_from_biomass('grass', 1.0), d0)
        self.assertFalse(self.cat._a_d.flags.writeable)        # ... and the private tables are frozen

    def test_allometry_positive_monotone_inverse_and_output_validation(self):
        B = np.array([0.01, 0.1, 1.0, 10.0])
        d = self.cat.diameter_from_biomass('grass', B)
        h = self.cat.height_from_biomass('grass', B)
        self.assertTrue(np.all(d > 0) and np.all(np.diff(d) > 0) and np.all(h > 0) and np.all(np.diff(h) > 0))
        self.assertAlmostEqual(self.cat.diameter_from_biomass('grass', 1.0), 0.9)
        np.testing.assert_allclose(self.cat.biomass_from_diameter('grass', d), B, rtol=1e-12)
        idx = np.array([0, 1, 1])
        np.testing.assert_allclose(self.cat.biomass_from_diameter(idx, self.cat.diameter_from_biomass(idx, np.array([0.2, 2.0, 3.0]))),
                                   [0.2, 2.0, 3.0], rtol=1e-12)
        self.assertEqual(self.cat.diameter_from_biomass('shrub', 0.0), 0.0)
        E = traits.CatalogError
        for bad in (-1.0, True, 1 + 2j, '1.0', np.array([True]), np.array(['1']), np.array([1.0, np.nan]), None):
            with self.assertRaises(E, msg=repr(bad)):
                self.cat.diameter_from_biomass('grass', bad)
        with self.assertRaises(E):
            self.cat.index('BOER')
        big = self.cat.document
        big['species']['grass']['allometry']['diameter']['b'] = 2.0
        bc = traits.SpeciesCatalog(big)
        with self.assertRaises(E):   # finite input, overflowing forward output (B^2)
            bc.diameter_from_biomass('grass', 1e308)
        self.assertTrue(math.isfinite(bc.biomass_from_diameter('grass', 1e308)))   # inverse exponent 1/2: finite, valid
        half = self.cat.document
        half['species']['grass']['allometry']['diameter']['b'] = 0.5
        hc = traits.SpeciesCatalog(half)
        with self.assertRaises(E):   # inverse exponent 2: a huge finite diameter overflows -> refused
            hc.biomass_from_diameter('grass', 1e308)
        self.assertTrue(math.isfinite(hc.diameter_from_biomass('grass', 1e308)))

    def test_traits_sections_and_lifespan_metadata(self):
        self.assertEqual(self.cat.life_expectancy_days('grass'), 8 * 365.0)
        self.assertEqual(self.cat.root_layers, 2)
        self.assertEqual(self.cat.root_layer_fractions('grass'), [0.8, 0.2])
        self.assertEqual(self.cat.porosity('shrub'), 0.2)
        self.assertEqual(self.cat.law('grass', 'mortality'), 'unimplemented')
        sec = self.cat.section('grass', 'disturbance')
        self.assertIn('fixture_sensitivities', sec)
        sec['law'] = 'mutated'
        self.assertEqual(self.cat.law('grass', 'disturbance'), 'unimplemented')
        with self.assertRaises(traits.CatalogError):
            self.cat.stepwat_traits()
        doc = self.cat.document
        row = {'mature_biomass_kg': 1.0, 'intrinsic_rate_per_year': 0.5, 'biomass_support_kg_per_m3': 10.0, 'live_fraction': 0.5,
               'canopy_biomass_density_kg_m2': 1.0, 'potential_transpiration_m_per_day': 0.003, 'extra_growth_fraction': 0.5,
               'monthly_activity': [1.0] * 12}
        for sid in doc['species']:
            doc['species'][sid]['growth'] = {'law': traits.STEPWAT_GROWTH_LAW, 'traits': row}
        self.assertEqual(traits.SpeciesCatalog(doc).stepwat_traits()['shrub']['mature_biomass_kg'], 1.0)

    def test_schema_errors(self):
        E = traits.CatalogError
        with self.assertRaises(E):
            traits.parse_strict_json('{"a": 1, "a": 2}')
        with self.assertRaises(E):
            traits.parse_strict_json('{"a": NaN}')
        with self.assertRaises(E):
            traits.parse_strict_json('{"a": -Infinity}')
        base = self.cat.document

        def bad(mutate):
            doc = copy.deepcopy(base)
            mutate(doc)
            with self.assertRaises(E):
                traits.SpeciesCatalog(doc)

        bad(lambda d: d.__setitem__('schema_version', 2))
        bad(lambda d: d.__setitem__('schema', 'other'))
        bad(lambda d: d.__setitem__('units', {'biomass': 'g', 'length': 'm', 'time': 'days'}))
        bad(lambda d: d['species']['grass']['allometry']['diameter'].__setitem__('a', True))
        bad(lambda d: d['species']['grass']['allometry']['diameter'].__setitem__('b', 0.0))
        bad(lambda d: d['species']['grass']['allometry'].__setitem__('model', 'linear'))
        bad(lambda d: d['species']['grass']['roots'].__setitem__('layer_fractions', [0.5, 0.3, 0.2]))
        bad(lambda d: d['species']['grass']['roots'].__setitem__('layer_fractions', [0.8, 0.1]))
        bad(lambda d: d['species']['grass']['roots'].__setitem__('layer_fractions', [1.2, -0.2]))
        bad(lambda d: d['species']['grass']['porosity'].__setitem__('value', 1.5))
        bad(lambda d: d['species']['grass'].pop('mortality'))
        bad(lambda d: d['species']['grass'].__setitem__('growth', {'rate': 1.0}))
        bad(lambda d: d['species']['grass']['life_expectancy'].__setitem__('unit', 'months'))
        bad(lambda d: d['species']['grass'].__setitem__('unknown_key', 1))
        bad(lambda d: d['species'].__setitem__('bad id', d['species']['grass']))
        bad(lambda d: d.__setitem__('root_layers', 2.0))


# =============================================================== domain / surface
class TestDomain(unittest.TestCase):
    def test_validation_immutability_and_metadata_conflicts(self):
        with self.assertRaises(state.StateError):
            state.Domain(0, 4, 1.0)
        with self.assertRaises(state.StateError):
            state.Domain(4, 4, -1.0)
        with self.assertRaises(state.StateError):
            state.Domain(True, 4, 1.0)
        with self.assertRaises(state.StateError):
            state.Domain(4, 4, float('nan'))
        with self.assertRaises(state.StateError):
            state.Domain(4, 4, 1e308, 1e308)   # non-finite area
        d = state.Domain(4, 3, 0.5, 2.0, 10.0, -5.0)
        self.assertEqual(d.shape, (3, 4))
        self.assertEqual(state.Domain.from_dict(d.to_dict()), d)
        self.assertNotEqual(d, state.Domain(4, 3, 0.5))
        with self.assertRaises(AttributeError):
            d.nx = 5
        with self.assertRaises(AttributeError):
            d.dx_m = 1.0
        md = d.to_dict()
        md['edges'] = 'periodic'
        with self.assertRaises(state.StateError):
            state.Domain.from_dict(md)
        md = d.to_dict()
        md['cell_sampling'] = 'bilinear'
        with self.assertRaises(state.StateError):
            state.Domain.from_dict(md)
        md = d.to_dict()
        md['bogus'] = 1
        with self.assertRaises(state.StateError):
            state.Domain.from_dict(md)

    def test_containing_cell_subcell_edges_offset_nonsquare(self):
        d = state.Domain(4, 3, 0.5, 2.0, 10.0, -5.0)
        field = np.arange(12, dtype=float).reshape(3, 4) * 1.0  # value = row*4 + col
        x = np.array([10.0, 10.74, 12.0, 11.3, 10.26])
        y = np.array([-5.0, -5.0, 1.0, -3.9, -1.0])
        row, col = d.cell_indices(x, y)
        np.testing.assert_array_equal(col, [0, 1, 3, 2, 0])
        np.testing.assert_array_equal(row, [0, 0, 2, 0, 2])
        np.testing.assert_array_equal(d.sample(field, x, y), [0.0, 1.0, 11.0, 2.0, 8.0])
        for bx, by in ((12.0001, 0.0), (9.999, 0.0), (11.0, 1.0001), (11.0, -5.001)):  # no wrapping, closed edges
            with self.assertRaises(state.StateError):
                d.cell_indices(np.array([bx]), np.array([by]))
        with self.assertRaises(state.StateError):
            d.cell_indices(np.array([np.nan]), np.array([0.0]))
        with self.assertRaises(state.StateError):
            d.check_field(field.T)  # transposed shape refused
        bad = field.copy()
        bad[1, 1] = np.nan
        with self.assertRaises(state.StateError):
            d.check_field(bad)
        with self.assertRaises(state.StateError):
            d.check_field(np.ones((3, 4), dtype=bool))


# =============================================================== plant state and CSV
class TestPlantState(unittest.TestCase):
    def setUp(self):
        self.cat = example_catalog()
        self.dom = state.Domain(20, 20, 1.0)

    def test_surface_token_resolved_once_and_anchor_preserved(self):
        s = small_state(self.cat, resolved=False)
        self.assertFalse(s.establishment_resolved)
        self.assertEqual(int(s.establishment_unresolved.sum()), 3)
        with self.assertRaises(state.StateError):
            state.plant_csv_text(s)  # non-initial snapshot with SURFACE refused
        with self.assertRaises(state.StateError):
            s.advance_time(1.0)      # unresolved initial placements cannot move forward in time
        text = state.plant_csv_text(s, initial=True)
        self.assertIn('SURFACE', text)
        self.assertIn('"initial": true', text)
        surf = np.arange(400, dtype=float).reshape(20, 20) + 100.0  # value = 100 + row*20 + col
        self.assertEqual(s.resolve_establishment(surf), 3)
        self.assertTrue(s.establishment_resolved)
        tok = s.indices_of([1, 2, 4])          # rows resolved from `surf`
        explicit = s.indices_of([3, 5, 6])     # rows with explicit anchors
        i1 = tok[0]
        self.assertEqual(s.establishment_ground_m[i1], 100.0 + 3 * 20 + 2)  # (2.5, 3.5) -> row 3, col 2
        np.testing.assert_array_equal(s.establishment_ground_m[explicit], [99.9, 100.0, 99.8])  # untouched
        self.assertEqual(s.resolve_establishment(surf + 50.0), 0)  # resolved ONCE
        self.assertEqual(s.establishment_ground_m[i1], 162.0)
        anchors = s.establishment_ground_m.copy()
        for shift in (0.3, -0.2):
            cur = surf + shift
            geo = canopy.derive_geometry(s, self.cat, current_surface=cur)
            expected = self.dom.sample(cur, s.x_m, s.y_m) - anchors   # delta for ALL rows = sampled current - stored anchor
            np.testing.assert_allclose(geo['surface_delta_m'], expected, atol=1e-12)
            np.testing.assert_allclose(geo['burial_depth_m'], np.maximum(expected, 0.0), atol=1e-12)
            np.testing.assert_allclose(geo['exposure_depth_m'], np.maximum(-expected, 0.0), atol=1e-12)
            np.testing.assert_allclose(geo['surface_delta_m'][tok], shift, atol=1e-12)   # exact on the SURFACE subset only
            np.testing.assert_array_equal(s.establishment_ground_m, anchors)            # anchors never rewritten
        np.testing.assert_allclose(geo['burial_depth_m'][tok], 0.0)
        np.testing.assert_allclose(geo['exposure_depth_m'][tok], 0.2)

    def test_constructor_requires_explicit_unresolved_mask_and_owns_arrays(self):
        E = state.StateError
        ids = np.array([1, 2], dtype=np.int64)
        kw = dict(species_index=[0, 1], x_m=[1.0, 2.0], y_m=[1.0, 2.0], agb_dry_kg=[0.1, 0.2], age_days=[1.0, 2.0], status=['live', 'live'])
        with self.assertRaises(E):   # bare NaN is NOT a surface request
            state.PlantState(self.dom, self.cat.species_ids, self.cat.hash, ids=ids, establishment_ground_m=[np.nan, 5.0], **kw)
        with self.assertRaises(E):
            state.PlantState(self.dom, self.cat.species_ids, self.cat.hash, ids=ids, establishment_ground_m=[np.inf, 5.0], **kw)
        with self.assertRaises(E):   # mask says resolved but value is not finite
            state.PlantState(self.dom, self.cat.species_ids, self.cat.hash, ids=ids, establishment_ground_m=[np.nan, 5.0],
                             establishment_unresolved=[False, False], **kw)
        s = state.PlantState(self.dom, self.cat.species_ids, self.cat.hash, ids=ids, establishment_ground_m=[np.nan, 5.0],
                             establishment_unresolved=[True, False], **kw)
        self.assertEqual(int(s.establishment_unresolved.sum()), 1)
        B = np.array([0.1, 0.2])
        s2 = state.PlantState(self.dom, self.cat.species_ids, self.cat.hash, ids=ids, establishment_ground_m=[5.0, 5.0],
                              species_index=np.array([0, 1], np.int32), x_m=[1.0, 2.0], y_m=[1.0, 2.0], agb_dry_kg=B,
                              age_days=[1.0, 2.0], status=['live', 'live'])
        B[0] = 99.0
        ids[0] = 77
        self.assertEqual(s2.agb_dry_kg[0], 0.1)            # no aliasing of caller arrays
        self.assertEqual(s2.ids[0], 1)
        with self.assertRaises(ValueError):
            s2.ids[0] = 5                                   # identity fields are read-only
        with self.assertRaises(ValueError):
            s2.agb_dry_kg[0] = 5.0
        with self.assertRaises(AttributeError):
            s2.ids = np.array([9, 8])
        self.assertEqual(s2.indices_of([2])[0], 1)
        s2.update_plants([2], agb_dry_kg=[0.5])
        self.assertEqual(s2.agb_dry_kg[s2.indices_of([2])[0]], 0.5)
        self.assertFalse(s2.agb_dry_kg.flags.writeable)
        with self.assertRaises(E):
            s2.time_days = float('nan')

    def test_invalid_rows_refused_without_partial_state(self):
        base = [rec(1, 'grass', 1.0, 1.0, 100.0, 0.1)]
        bad_rows = [
            [rec(1, 'grass', 1.0, 1.0, 100.0, 0.1), rec(1, 'shrub', 2.0, 2.0, 100.0, 0.1)],  # duplicate id
            [rec(1, 'BOER', 1.0, 1.0, 100.0, 0.1)],                                            # unknown species
            [rec(1, 'grass', 1.0, 1.0, 100.0, 0.1, E=0.2)],                                    # E > B
            [rec(1, 'grass', 1.0, 1.0, 100.0, 0.1, age=-1.0)],                                 # negative age
            [rec(1, 'grass', 1.0, 1.0, 100.0, -0.1)],                                          # negative mass
            [rec(1, 'grass', 1.0, 1.0, 100.0, 0.0)],                                           # zero mass live
            [rec(1, 'grass', 1.0, 1.0, 100.0, 0.0, status='dead_standing')],                   # zero mass dead
            [rec(1, 'grass', 20.5, 1.0, 100.0, 0.1)],                                          # outside domain
            [rec(1, 'grass', float('nan'), 1.0, 100.0, 0.1)],                                  # nonfinite position
            [rec(1, 'grass', 1.0, 1.0, 100.0, 0.1, status='zombie')],                          # unsupported status
            [rec(1, 'grass', 1.0, 1.0, 'nan', 0.1)],                                           # nan is not SURFACE
            [rec(1, 'grass', 1.0, 1.0, '', 0.1)],                                              # empty is not SURFACE
            [rec(1, 'grass', 1.0, 1.0, 'NA', 0.1)],
            [rec(1, 'grass', 1.0, 1.0, 'surface', 0.1)],                                       # token is case-sensitive
            [rec(1, 'grass', 1.0, 1.0, 'inf', 0.1)],
            [rec(1, 'grass', 1.0, 1.0, 100.0, 'inf')],
            [rec(1.5, 'grass', 1.0, 1.0, 100.0, 0.1)],
        ]
        state.PlantState.from_records(self.cat, self.dom, base)  # sanity: the base row is valid
        for rows in bad_rows:
            with self.assertRaises(state.StateError, msg=str(rows)):
                state.PlantState.from_records(self.cat, self.dom, rows)
        ok = state.PlantState.from_records(self.cat, self.dom, [rec(1, 'grass', 1.0, 1.0, 100.0, 0.0, status='removed')])
        self.assertEqual(ok.counts()['removed'], 1)

    def test_empty_population_round_trip(self):
        s = state.PlantState.empty(self.cat, self.dom, time_days=12.5)
        self.assertEqual(len(s), 0)
        text = state.plant_csv_text(s)
        back = state.parse_plant_csv(text, self.cat)
        self.assertEqual(len(back), 0)
        self.assertEqual(back.time_days, 12.5)
        self.assertEqual(back.domain, self.dom)
        self.assertTrue(back.establishment_resolved)
        geo = canopy.derive_geometry(back, self.cat)
        self.assertEqual(geo['n'], 0)
        view = canopy.maple_vegetation_view(back, self.cat, state.constant_surface(self.dom, 0.0))
        self.assertEqual(view['meta']['n_objects'], 0)
        self.assertTrue(view['meta']['native_maple_valid'])
        with self.assertRaises(state.StateError):
            back.indices_of([1])

    def test_exact_fp64_round_trip_and_metadata_checks(self):
        rng = np.random.default_rng(7)
        n = 200
        dom = state.Domain(37, 23, 0.7, 1.3, -12.345678901234567, 3.0000000001)
        ids = rng.permutation(np.arange(1_000_000, 1_000_000 + n, dtype=np.int64)) * 7919
        x = rng.uniform(dom.x_min_m, dom.x_max_m, n)
        y = rng.uniform(dom.y_min_m, dom.y_max_m, n)
        x[0], y[0] = dom.x_max_m, dom.y_min_m  # closed edges included
        B = np.exp(rng.uniform(-9, 3, n))
        B[5] = 1e-300
        E = B * rng.uniform(0, 1, n)
        E[7] = B[7]
        age = rng.uniform(0, 20000, n)
        est = rng.normal(1000.0, 10.0, n)
        st = np.array([state.LIVE] * n, dtype=np.int8)
        st[3], st[4] = state.DEAD_STANDING, state.REMOVED
        B[4], E[4] = 0.0, 0.0
        sp = rng.integers(0, 2, n).astype(np.int32)
        s = state.PlantState(dom, self.cat.species_ids, self.cat.hash, ids=ids, species_index=sp, x_m=x, y_m=y,
                             establishment_ground_m=est, agb_dry_kg=B, seasonal_agb_dry_kg=E, age_days=age, status=st,
                             time_days=0.1 + 0.2, provenance={'test': 'round-trip'})
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'veg.csv')
            state.write_plant_csv(s, path)
            with self.assertRaises(FileExistsError):
                state.write_plant_csv(s, path)  # no overwrite by default
            state.write_plant_csv(s, path, overwrite=True)
            back = state.read_plant_csv(path, self.cat)
            for k in state.PlantState.ARRAY_NAMES:
                np.testing.assert_array_equal(getattr(back, k), getattr(s, k), err_msg=k)
            self.assertEqual(back.time_days, 0.1 + 0.2)
            self.assertEqual(back.domain, dom)
            self.assertEqual(back.provenance['test'], 'round-trip')
            self.assertEqual(back.catalog_hash, self.cat.hash)
            with open(path, 'r', encoding='utf-8') as fh:
                text = fh.read()
            with self.assertRaises(state.StateError):
                state.parse_plant_csv(text, consistent_catalog())        # catalog mismatch
            with self.assertRaises(state.StateError):
                state.parse_plant_csv(text, self.cat, domain=state.Domain(37, 23, 0.7))   # domain disagreement
            self.assertEqual(len(state.parse_plant_csv(text, self.cat, domain=dom)), n)
            lines = text.splitlines(True)
            with self.assertRaises(state.StateError):
                state.parse_plant_csv(''.join(lines[:-1]), self.cat)     # truncated (n_plants mismatch)
            with self.assertRaises(state.StateError):
                state.parse_plant_csv(''.join(lines[:-1]) + lines[-1].rsplit(',', 2)[0] + '\n', self.cat)  # malformed row
            with self.assertRaises(state.StateError):
                state.parse_plant_csv(lines[0] + lines[1].replace('plant_id', 'id') + ''.join(lines[2:]), self.cat)
            meta = json.loads(lines[0][1:])
            meta['schema_version'] = 99
            with self.assertRaises(state.StateError):
                state.parse_plant_csv('# ' + json.dumps(meta) + '\n' + ''.join(lines[1:]), self.cat)
            meta = json.loads(lines[0][1:])
            meta['domain']['edges'] = 'periodic'
            with self.assertRaises(state.StateError):
                state.parse_plant_csv('# ' + json.dumps(meta) + '\n' + ''.join(lines[1:]), self.cat)
            with self.assertRaises(state.StateError):
                state.parse_plant_csv(''.join(lines[1:]), self.cat)      # handwritten CSV needs a domain
            self.assertEqual(len(state.parse_plant_csv(''.join(lines[1:]), self.cat, domain=dom)), n)
            # standard strict CSV: an unterminated quoted last field is refused ...
            last_status = state.STATUS_NAMES[int(s.status[-1])]
            quoted = ''.join(lines[:-1]) + lines[-1].rsplit(',', 1)[0] + ',"' + last_status + '\n'
            with self.assertRaises(state.StateError):
                state.parse_plant_csv(quoted, self.cat)
            # ... while ordinary quoted status / numeric fields (spreadsheet style) round-trip exactly
            quoted_ok = ''.join(lines[:-1]) + lines[-1].rsplit(',', 1)[0] + ',"' + last_status + '"\n'
            qb = state.parse_plant_csv(quoted_ok, self.cat)
            for k in state.PlantState.ARRAY_NAMES:
                np.testing.assert_array_equal(getattr(qb, k), getattr(s, k), err_msg=k)
            f = lines[2].rstrip('\n').split(',')
            row_q = ','.join([f[0], '"' + f[1] + '"'] + ['"' + v + '"' for v in f[2:8]] + [f[8]]) + '\n'
            qn = state.parse_plant_csv(lines[0] + lines[1] + row_q + ''.join(lines[3:]), self.cat)
            for k in state.PlantState.ARRAY_NAMES:
                np.testing.assert_array_equal(getattr(qn, k), getattr(s, k), err_msg=k)
            with self.assertRaises(state.StateError):   # stray quote inside a field is malformed CSV
                state.parse_plant_csv(lines[0] + lines[1] + lines[2].replace(',live', ',li"ve') + ''.join(lines[3:]), self.cat)

    def test_metadata_noninitial_surface_token_refused_before_sampling(self):
        s = small_state(self.cat, resolved=False)
        text = state.plant_csv_text(s, initial=True)
        line, body = text.split('\n', 1)
        meta = json.loads(line[1:])
        later = state.constant_surface(self.dom, 55.0)
        for mutate in (lambda m: m.update(initial=False, time_days=7.0),
                       lambda m: m.pop('initial'),
                       lambda m: m.update(establishment_resolved=True)):
            m = dict(meta)
            mutate(m)
            with self.assertRaises(state.StateError):
                state.parse_plant_csv('# ' + json.dumps(m) + '\n' + body, self.cat, surface=later)
        ok = state.parse_plant_csv(text, self.cat, surface=later)   # declared initial: resolved from the INITIAL surface once
        self.assertTrue(ok.establishment_resolved)
        self.assertEqual(ok.establishment_ground_m[ok.indices_of([1])[0]], 55.0)
        self.assertEqual(ok.establishment_ground_m[ok.indices_of([3])[0]], 99.9)
        # a saved (non-initial) snapshot of the resolved state carries concrete anchors only
        self.assertNotIn('SURFACE', state.plant_csv_text(ok))

    def test_updates_are_transactional_and_by_id(self):
        s = small_state(self.cat)
        before = snapshot(s)
        with self.assertRaises(state.StateError):
            s.update_plants([1, 2], agb_dry_kg=[0.5, 0.005])  # id 2 has E=0.01 > new B
        with self.assertRaises(state.StateError):
            s.update_plants([1, 99], agb_dry_kg=[0.5, 0.5])    # unknown id
        with self.assertRaises(state.StateError):
            s.update_plants([1, 1], agb_dry_kg=[0.5, 0.5])     # duplicate ids in request
        with self.assertRaises(state.StateError):
            s.update_plants([6], status='live')                # removed -> live refused
        with self.assertRaises(state.StateError):
            s.update_plants([5], status='live')                # dead_standing -> live refused
        with self.assertRaises(state.StateError):
            s.update_plants([1], agb_dry_kg=[-1.0])
        assert_unchanged(self, s, before)
        idx = s.update_plants([4, 1], agb_dry_kg=[2.0, 0.3], seasonal_agb_dry_kg=[0.5, 0.0])
        np.testing.assert_array_equal(s.ids[idx], [4, 1])
        self.assertEqual(s.agb_dry_kg[s.indices_of([1])[0]], 0.3)
        self.assertEqual(s.seasonal_agb_dry_kg[s.indices_of([4])[0]], 0.5)
        self.assertEqual(s.counts(), {'live': 4, 'dead_standing': 1, 'removed': 1})
        s.update_plants([1], status='dead_standing')
        s.update_plants([1], status='removed', agb_dry_kg=[0.0], seasonal_agb_dry_kg=[0.0])
        self.assertEqual(s.counts(), {'live': 3, 'dead_standing': 1, 'removed': 2})  # ids 2, 3, 4 live; 5 dead; 1, 6 removed

    def test_age_advances_with_calendar_only_for_live_and_no_automatic_death(self):
        s = small_state(self.cat)
        age0 = s.age_days.copy()
        s.advance_time(365.25)                   # e.g. a wind-only / transport interval: no ecological step needed
        live = s.live_mask()
        np.testing.assert_allclose(s.age_days[live], age0[live] + 365.25)
        np.testing.assert_array_equal(s.age_days[~live], age0[~live])  # dead/removed ages frozen
        self.assertEqual(s.time_days, 365.25)
        text1 = state.plant_csv_text(s)
        text2 = state.plant_csv_text(s)          # repeated export never ages a plant
        self.assertEqual(text1, text2)
        with self.assertRaises(state.StateError):
            s.advance_time(-1.0)
        life = self.cat.life_expectancy_days('grass')
        s.advance_time(life * 3)
        self.assertTrue(np.all(s.age_days[live] > life))
        self.assertEqual(int(s.live_mask().sum()), int(live.sum()))  # mean life expectancy != hard maximum

    def test_advance_time_preflights_overflow_and_commits_atomically(self):
        s = state.PlantState.from_records(self.cat, self.dom, [rec(17, 'grass', 0.5, 0.5, 3.0, 0.01, 0.002, 1e308)])
        before, t0 = snapshot(s), s.time_days
        with np.errstate(over='ignore'):
            with self.assertRaises(state.StateError):
                s.advance_time(1e308)
        assert_unchanged(self, s, before)
        self.assertEqual(s.time_days, t0)
        s2 = state.PlantState.from_records(self.cat, self.dom, [rec(17, 'grass', 0.5, 0.5, 3.0, 0.01, 0.002, 1.0)], time_days=1e308)
        with self.assertRaises(state.StateError):
            s2.advance_time(1e308)                # clock overflow
        self.assertEqual(s2.time_days, 1e308)
        self.assertEqual(s2.age_days[0], 1.0)
        s2.advance_time(0.0)
        self.assertEqual(s2.age_days[0], 1.0)

    def test_ids_resolve_regardless_of_row_order(self):
        s = small_state(self.cat)
        perm = np.array([5, 2, 0, 4, 1, 3])
        s2 = state.PlantState(s.domain, s.species_names, s.catalog_hash, ids=s.ids[perm], species_index=s.species_index[perm],
                              x_m=s.x_m[perm], y_m=s.y_m[perm], establishment_ground_m=s.establishment_ground_m[perm],
                              agb_dry_kg=s.agb_dry_kg[perm], seasonal_agb_dry_kg=s.seasonal_agb_dry_kg[perm],
                              age_days=s.age_days[perm], status=s.status[perm])
        for pid in (1, 3, 6):
            self.assertEqual(s2.x_m[s2.indices_of([pid])[0]], s.x_m[s.indices_of([pid])[0]])

    def test_atomic_no_overwrite_never_falls_back_to_replace(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, 'a.txt')
            state.atomic_write_text(p, 'one')
            with self.assertRaises(FileExistsError):
                state.atomic_write_text(p, 'two')
            with open(p) as fh:
                self.assertEqual(fh.read(), 'one')
            original_link = os.link

            def no_link(src, dst, *a, **k):
                raise OSError(95, 'Operation not supported')

            os.link = no_link
            try:
                with self.assertRaises(OSError):
                    state.atomic_write_text(os.path.join(td, 'b.txt'), 'x')   # refused, not replaced
                self.assertFalse(os.path.exists(os.path.join(td, 'b.txt')))
                self.assertEqual([f for f in os.listdir(td) if f.startswith('.leaf-')], [])   # temp removed
                state.atomic_write_text(os.path.join(td, 'b.txt'), 'x', overwrite=True)      # explicit overwrite path
            finally:
                os.link = original_link
            with self.assertRaises(FileNotFoundError):
                state.atomic_write_text(os.path.join(td, 'missing', 'c.txt'), 'x')


# =============================================================== canopy geometry / MAPLE view / cover
class TestCanopy(unittest.TestCase):
    def setUp(self):
        self.cat = example_catalog()

    def test_geometry_derived_pure_and_constant_porosity(self):
        s = small_state(self.cat)
        snap = snapshot(s)
        geo = canopy.derive_geometry(s, self.cat)
        i3 = s.indices_of([3])[0]
        self.assertAlmostEqual(geo['diameter_m'][i3], 1.6 * 2.5 ** 0.4)
        self.assertAlmostEqual(geo['height_m'][i3], 1.0 * 2.5 ** 0.35)
        self.assertAlmostEqual(geo['root_radius_m'][i3], 3.0 * 0.5 * 1.6 * 2.5 ** 0.4)
        np.testing.assert_array_equal(geo['porosity'], 0.2)
        self.assertEqual(geo['diameter_m'][s.indices_of([6])[0]], 0.0)  # removed, zero mass
        geo['x_m'][:] = -1.0
        geo['porosity'][:] = 0.9
        assert_unchanged(self, s, snap)
        with self.assertRaises(state.StateError):
            canopy.derive_geometry(s, consistent_catalog())  # catalog hash mismatch

    def test_maple_view_safe_default_uses_current_surface_and_anchor(self):
        dom = state.Domain(10, 10, 1.0, 1.0, 50.0, -20.0)
        recs = [rec(11, 'grass', 52.0, -18.0, 7.0, 0.1), rec(12, 'shrub', 55.5, -15.5, 7.0, 2.0),
                rec(13, 'grass', 58.0, -12.0, 7.0, 0.05, status='dead_standing'),
                rec(14, 'shrub', 51.0, -11.0, 7.0, 0.0, status='removed')]
        s = state.PlantState.from_records(self.cat, dom, recs)
        flat = state.constant_surface(dom, 7.0)
        with self.assertRaises(ValueError):
            canopy.maple_vegetation_view(s, self.cat, None)       # the current surface is required
        v = canopy.maple_vegetation_view(s, self.cat, flat)
        self.assertEqual(list(v['fields']), list(canopy.MAPLE_FIELDS))
        self.assertEqual(v['meta']['n_objects'], 3)  # removed excluded, dead_standing included
        self.assertTrue(v['meta']['native_maple_valid'])
        self.assertEqual(v['meta']['height_mode'], 'exposed')
        np.testing.assert_array_equal(v['sidecar']['plant_id'], [11, 12, 13])
        np.testing.assert_allclose(v['fields']['center_x_m'], [2.0, 5.5, 8.0])      # domain-local (origin subtracted)
        np.testing.assert_allclose(v['fields']['center_y_m'], [2.0, 4.5, 8.0])
        np.testing.assert_array_equal(v['fields']['semi_major_m'], v['fields']['semi_minor_m'])
        np.testing.assert_array_equal(v['fields']['orientation_deg'], 0.0)
        np.testing.assert_array_equal(v['fields']['porosity'], 0.2)
        np.testing.assert_array_equal(v['fields']['drag_coefficient'], 1.0)
        np.testing.assert_allclose(v['fields']['height_m'], v['sidecar']['intact_height_m'])  # no burial on a flat surface
        self.assertEqual(v['meta']['catalog_hash'], self.cat.hash)
        self.assertEqual(canopy.maple_vegetation_view(s, self.cat, flat, statuses=('live',))['meta']['n_objects'], 2)
        h_intact = v['sidecar']['intact_height_m'].copy()
        # partial burial reduces the published height; erosion (elevated base) is unsupported
        cur = flat.copy()
        cur[2, 2] = 7.05            # grass 11 at (52,-18) -> row 2, col 2: partially buried by 0.05
        cur[4, 5] = 6.5             # shrub 12 at (55.5,-15.5) -> row 4, col 5: eroded (elevated base)
        with self.assertRaises(state.StateError):
            canopy.maple_vegetation_view(s, self.cat, cur)         # default policy: error
        ex = canopy.maple_vegetation_view(s, self.cat, cur, unsupported_policy='exclude')
        self.assertEqual(ex['meta']['n_objects'], 2)
        self.assertNotIn(12, ex['sidecar']['plant_id'].tolist())
        self.assertAlmostEqual(ex['fields']['height_m'][0], h_intact[0] - 0.05)
        self.assertTrue(ex['meta']['native_maple_valid'])
        fl = canopy.maple_vegetation_view(s, self.cat, cur, unsupported_policy='flag')
        self.assertEqual(fl['meta']['n_objects'], 3)
        self.assertFalse(fl['meta']['native_maple_valid'])
        self.assertTrue(bool(fl['sidecar']['elevated_base'][1]))
        self.assertFalse(bool(fl['sidecar']['native_maple_valid'][1]))
        self.assertEqual(fl['meta']['n_unsupported'], 1)
        # a fully buried plant must never reappear at full height on top of deposited soil
        buried = flat.copy()
        buried[2, 2] = 10.0
        with self.assertRaises(state.StateError):
            canopy.maple_vegetation_view(s, self.cat, buried)
        exb = canopy.maple_vegetation_view(s, self.cat, buried, unsupported_policy='exclude')
        self.assertNotIn(11, exb['sidecar']['plant_id'].tolist())
        self.assertTrue(exb['meta']['native_maple_valid'])
        flb = canopy.maple_vegetation_view(s, self.cat, buried, unsupported_policy='flag')
        self.assertEqual(flb['fields']['height_m'][0], 0.0)
        self.assertFalse(flb['meta']['native_maple_valid'])
        diag = canopy.maple_vegetation_view(s, self.cat, buried, height_mode='intact_diagnostic', unsupported_policy='flag')
        self.assertAlmostEqual(diag['fields']['height_m'][0], h_intact[0])
        self.assertFalse(diag['meta']['native_maple_valid'])
        self.assertIn('DIAGNOSTIC', diag['meta']['height_definition'])
        with self.assertRaises(state.StateError):
            canopy.maple_vegetation_view(s, self.cat, buried, height_mode='intact_diagnostic')   # still refused by default
        unres = small_state(self.cat, resolved=False)
        with self.assertRaises(state.StateError):
            canopy.maple_vegetation_view(unres, self.cat, state.constant_surface(unres.domain, 1.0))

    def test_cover_analytic_circles_edges_groups_and_bounds(self):
        dom = state.Domain(10, 10, 1.0)
        m = canopy.measure_cover(dom, [5.0], [5.0], [2.0], subsamples=32)
        self.assertAlmostEqual(m['groups'][0]['union_fraction'], 4 * math.pi / 100, delta=0.002)
        self.assertAlmostEqual(m['groups'][0]['nominal_fraction'], 4 * math.pi / 100, places=12)
        q = canopy.measure_cover(dom, [0.0], [0.0], [2.0], subsamples=32)  # clipped quarter circle
        self.assertAlmostEqual(q['groups'][0]['union_fraction'], math.pi / 100, delta=0.001)
        self.assertAlmostEqual(q['groups'][0]['nominal_fraction'], 4 * math.pi / 100, places=12)
        two = canopy.measure_cover(dom, [5.0, 5.0], [5.0, 5.0], [2.0, 2.0], subsamples=32)  # same group counts once
        self.assertAlmostEqual(two['groups'][0]['union_fraction'], m['groups'][0]['union_fraction'], places=12)
        self.assertAlmostEqual(two['groups'][0]['nominal_fraction'], 8 * math.pi / 100, places=12)
        g = canopy.measure_cover(dom, [5.0, 5.0], [5.0, 5.0], [2.0, 2.0], [0, 1], 2, subsamples=32)  # cross-group
        self.assertAlmostEqual(g['groups'][1]['union_fraction'], m['groups'][0]['union_fraction'], places=12)
        self.assertAlmostEqual(g['total_union_fraction'], m['groups'][0]['union_fraction'], places=12)
        self.assertAlmostEqual(g['sum_of_group_union_fractions'], 2 * m['groups'][0]['union_fraction'], places=12)
        off = state.Domain(8, 4, 0.5, 2.0, 100.0, 200.0)  # non-square cells, offset origin
        o = canopy.measure_cover(off, [102.0], [204.0], [1.0], subsamples=32)
        self.assertAlmostEqual(o['groups'][0]['union_fraction'], math.pi / 32.0, delta=0.003)
        with self.assertRaises(canopy.ResourceBoundError):
            canopy.measure_cover(dom, [5.0], [5.0], [2.0], subsamples=32, max_cover_cells=1000)
        with self.assertRaises(canopy.ResourceBoundError):
            canopy.measure_cover(dom, [5.0], [5.0], [2.0], subsamples=32, max_cover_bytes=1000)
        with self.assertRaises(canopy.ResourceBoundError):
            canopy.measure_cover(dom, [5.0], [5.0], [2.0], subsamples=32, max_working_bytes=100)
        with self.assertRaises(state.StateError):
            canopy.measure_cover(dom, [11.0], [5.0], [2.0])  # centre outside the domain
        z = canopy.measure_cover(dom, [], [], [], subsamples=2)
        self.assertEqual(z['groups'][0]['union_fraction'], 0.0)
        self.assertEqual(z['total_union_fraction'], 0.0)
        # row tiling gives identical results at any chunk size
        a = canopy.measure_cover(dom, [5.0, 1.0], [5.0, 9.0], [2.0, 3.0], subsamples=16, max_working_bytes=16 * 320 + 9 * 160 + 1)
        b = canopy.measure_cover(dom, [5.0, 1.0], [5.0, 9.0], [2.0, 3.0], subsamples=16)
        self.assertEqual(a['rows_per_chunk'], 1)
        self.assertEqual(a['groups'][0]['covered_cells'], b['groups'][0]['covered_cells'])

    def test_cover_working_memory_is_admitted_and_scoped(self):
        dom = state.Domain(512, 512, 1.0)
        adm = canopy.cover_admission(dom, 1, total=True, max_cover_bytes=2 * 512 * 512)
        self.assertEqual(adm['raster_bytes'], 2 * 512 * 512)
        self.assertLessEqual(adm['working_bytes'], canopy.DEFAULT_MAX_WORKING_BYTES)
        tracemalloc.start()
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        result = canopy.measure_cover(dom, [256.0], [256.0], [400.0], subsamples=1, max_cover_cells=512 * 512,
                                      max_cover_bytes=2 * 512 * 512)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        self.assertEqual(result['raster_bytes'], 2 * 512 * 512)
        self.assertEqual(result['working_bytes'], adm['working_bytes'])
        self.assertLessEqual(peak - base, result['raster_bytes'] + result['working_bytes'] + 256 * 1024,
                             f"traced peak {peak - base} exceeds declared raster+working {result['raster_bytes'] + result['working_bytes']}")
        self.assertAlmostEqual(result['groups'][0]['union_fraction'], 1.0, places=12)   # r=400 covers the 512x512 domain

    def test_cover_independent_finer_resolution_agrees(self):
        rng = np.random.default_rng(3)
        dom = state.Domain(16, 16, 1.0)
        n = 300
        x, y = rng.uniform(0, 16, n), rng.uniform(0, 16, n)
        r = rng.lognormal(math.log(0.3), 0.3, n)
        coarse = canopy.measure_cover(dom, x, y, r, subsamples=8)['groups'][0]['union_fraction']
        fine = canopy.measure_cover(dom, x, y, r, subsamples=64)['groups'][0]['union_fraction']
        self.assertGreater(fine, 0.1)
        self.assertAlmostEqual(coarse, fine, delta=0.01)
        self.assertLessEqual(fine, math.pi * float((r * r).sum()) / 256.0 + 1e-9)  # union <= nominal


# =============================================================== generator
class TestGeneration(unittest.TestCase):
    def setUp(self):
        self.cat = example_catalog()

    def config(self, **over):
        cfg = {'schema': 'leaf.generation_config', 'schema_version': 1, 'domain': {'nx': 32, 'ny': 32, 'dx_m': 1.0},
               'seed': 11, 'overlap': 'allowed', 'cover_tolerance': 0.01, 'cover_subsamples': 8, 'max_iterations': 12,
               'limits': {'max_plants': 50000, 'max_attempts': 2000000},
               'species': {'grass': {'target_cover': 0.15, 'diameter_mean_m': 0.4, 'diameter_std_m': 0.1, 'initial_age_days': 730},
                           'shrub': {'target_cover': 0.05, 'diameter_mean_m': 1.0, 'diameter_std_m': 0.2,
                                     'initial_age_days': {'distribution': 'uniform', 'min_days': 365, 'max_days': 3650}}}}
        cfg.update(over)
        return cfg

    def test_count_formulas_use_size_variance(self):
        A = 1000.0
        ea0 = initialization.expected_crown_area_m2(0.5, 0.0)
        ea1 = initialization.expected_crown_area_m2(0.5, 0.3)
        self.assertAlmostEqual(ea0, math.pi / 4 * 0.25)
        self.assertAlmostEqual(ea1, math.pi / 4 * (0.25 + 0.09))
        self.assertLess(initialization.initial_count(A, 0.3, ea1, True), initialization.initial_count(A, 0.3, ea0, True))
        self.assertEqual(initialization.initial_count(A, 0.3, ea0, True), round(-A * math.log(0.7) / ea0))
        self.assertEqual(initialization.initial_count(A, 0.3, ea0, False), round(A * 0.3 / ea0))
        self.assertEqual(initialization.initial_count(A, 0.0, ea0, True), 0)
        mu, sigma = initialization.lognormal_parameters(0.5, 0.3)
        d = np.random.default_rng(0).lognormal(mu, sigma, 400000)
        self.assertAlmostEqual(float(d.mean()), 0.5, delta=0.01)
        self.assertAlmostEqual(float(d.std()), 0.3, delta=0.02)
        self.assertIsNone(initialization.lognormal_parameters(0.5, 0.0))
        with self.assertRaises(traits.CatalogError):   # Python float overflow converted to the declared error
            initialization.expected_crown_area_m2(1e200, 1e200)
        with self.assertRaises(traits.CatalogError):
            initialization.lognormal_parameters(1e-200, 1e200)
        self.assertAlmostEqual(initialization.expected_crown_area_m2(1e100, 0.0), 0.25 * math.pi * 1e200)   # finite stays valid

    def test_reproducible_overlap_allowed_species_cover(self):
        s1, r1 = initialization.generate_population(self.cat, self.config())
        s2, r2 = initialization.generate_population(self.cat, self.config())
        for k in state.PlantState.ARRAY_NAMES:
            np.testing.assert_array_equal(getattr(s1, k), getattr(s2, k), err_msg=k)
        s3, _ = initialization.generate_population(self.cat, self.config(seed=12))
        self.assertFalse(np.array_equal(s1.x_m[:10], s3.x_m[:10]))
        self.assertGreater(len(s1), 500)
        self.assertFalse(s1.establishment_resolved)
        self.assertTrue(np.all(s1.status == state.LIVE))
        self.assertEqual(s1.ids.tolist(), list(range(1, len(s1) + 1)))
        for sp in r1['species']:
            self.assertTrue(sp['accepted'], sp)
            self.assertLessEqual(abs(sp['measured_union_cover'] - sp['target_cover']), 0.01)
            self.assertGreater(sp['count'], 0)
            self.assertAlmostEqual(sp['diameter_sample_m']['mean'], sp['diameter_requested']['mean_m'], delta=0.1)
            self.assertGreater(sp['diameter_sample_m']['std'], 0.0)
        grass = r1['species'][0]
        self.assertEqual(grass['species'], 'grass')
        self.assertGreater(grass['nominal_crown_area_fraction'], grass['measured_union_cover'])  # overlaps counted once
        self.assertEqual(r1['seed'], 11)
        self.assertEqual(r1['catalog_hash'], self.cat.hash)
        self.assertIn('midpoint', r1['cover_measurement']['method'])
        # complete reproducibility record: SOURCE config (directly replayable) + derived provenance
        gc = r1['generation_config']
        self.assertEqual(gc['species']['shrub']['initial_age_days']['distribution'], 'uniform')
        self.assertEqual(gc['limits']['max_plants'], 50000)
        self.assertNotIn('ordering_convention', gc)
        self.assertEqual(r1['derived']['species']['grass']['diameter_distribution']['type'], 'lognormal')
        self.assertEqual(r1['derived']['ordering_convention'], r1['ordering_convention'])
        self.assertEqual(r1['derived']['species']['grass']['initial_count_estimate'], r1['species'][0]['initial_count_estimate'])
        self.assertEqual(r1['rng']['bit_generator'], 'PCG64')
        self.assertEqual(r1['rng']['numpy_version'], np.__version__)
        self.assertIn('generation_config', s1.provenance)
        self.assertIn('working_bytes', r1['cover_measurement'])
        saved = json.loads(json.dumps(gc))                      # JSON round trip of the saved config
        s4, r4 = initialization.generate_population(self.cat, saved)
        for k in state.PlantState.ARRAY_NAMES:
            np.testing.assert_array_equal(getattr(s4, k), getattr(s1, k), err_msg=k)
        self.assertEqual(r4['generation_config'], gc)
        self.assertEqual(s4.time_days, s1.time_days)
        geo = canopy.derive_geometry(s1, self.cat)
        gi = s1.species_index == 0
        self.assertAlmostEqual(float(geo['diameter_m'][gi].mean()), grass['diameter_sample_m']['mean'], places=9)
        self.assertTrue(np.all(s1.age_days[gi] == 730.0))
        sh = s1.age_days[~gi]
        self.assertTrue(np.all((sh >= 365.0) & (sh <= 3650.0)) and sh.std() > 0)

    def test_global_nonoverlap_large_first_and_surface_resolution(self):
        cfg = self.config(overlap='disallowed', domain={'nx': 16, 'ny': 16, 'dx_m': 1.0},
                          species={'grass': {'target_cover': 0.10, 'diameter_mean_m': 0.3, 'diameter_std_m': 0.05, 'initial_age_days': 10},
                                   'shrub': {'target_cover': 0.08, 'diameter_mean_m': 1.0, 'diameter_std_m': 0.0, 'initial_age_days': 100}})
        dom = state.Domain(16, 16, 1.0)
        surf = state.constant_surface(dom, 5.0)
        s, r = initialization.generate_population(self.cat, cfg, surface=surf)
        self.assertTrue(s.establishment_resolved)
        np.testing.assert_array_equal(s.establishment_ground_m, 5.0)
        geo = canopy.derive_geometry(s, self.cat)
        assert_nonoverlap(self, geo['x_m'], geo['y_m'], geo['radius_m'])
        self.assertTrue(np.all(np.diff(geo['diameter_m']) <= 1e-12))   # ids follow descending-size placement order
        for sp in r['species']:
            self.assertTrue(sp['accepted'], sp)
            self.assertLessEqual(abs(sp['measured_union_cover'] - sp['target_cover']), 0.01)
        self.assertIn('large_canopy_first', r['ordering_convention'])
        self.assertGreaterEqual(r['attempts'], len(s))
        self.assertGreaterEqual(r['placement_passes'], 1)
        s2, _ = initialization.generate_population(self.cat, cfg, surface=surf)
        np.testing.assert_array_equal(s.x_m, s2.x_m)
        saved = json.loads(json.dumps(r['generation_config']))   # saved exclusion config replays with the same surface
        s3, _ = initialization.generate_population(self.cat, saved, surface=surf)
        for k in state.PlantState.ARRAY_NAMES:
            np.testing.assert_array_equal(getattr(s3, k), getattr(s, k), err_msg=k)

    def test_mixed_size_exclusion_regression_128_quarter_metre(self):
        '''Grass 6 % (0.15 m +- 0.05 m) with shrubs 5 % (2 m +- 0.3 m): random interleaving needed millions of
        rejections; large-canopy-first must construct global nonoverlap within max_attempts=50000.'''
        cfg = {'domain': {'nx': 128, 'ny': 128, 'dx_m': 0.25}, 'seed': 7751, 'overlap': 'disallowed', 'cover_tolerance': 0.005,
               'cover_subsamples': 4, 'max_iterations': 12, 'limits': {'max_attempts': 50000, 'max_plants': 100000},
               'species': {'grass': {'target_cover': 0.06, 'diameter_mean_m': 0.15, 'diameter_std_m': 0.05, 'initial_age_days': 100.0},
                           'shrub': {'target_cover': 0.05, 'diameter_mean_m': 2.0, 'diameter_std_m': 0.3,
                                     'initial_age_days': {'distribution': 'uniform', 'min_days': 300.0, 'max_days': 3000.0}}}}
        dom = state.Domain(128, 128, 0.25)
        s, r = initialization.generate_population(self.cat, cfg, surface=state.constant_surface(dom, 4.25))
        self.assertLessEqual(r['attempts'], 50000)
        self.assertGreater(len(s), 2000)
        geo = canopy.derive_geometry(s, self.cat)
        assert_nonoverlap(self, geo['x_m'], geo['y_m'], geo['radius_m'])
        for sp in r['species']:
            self.assertTrue(sp['accepted'], sp)
            self.assertLessEqual(abs(sp['measured_union_cover'] - sp['target_cover']), 0.005)
        shrub = [sp for sp in r['species'] if sp['species'] == 'shrub'][0]
        self.assertGreaterEqual(shrub['count'], 10)
        self.assertAlmostEqual(shrub['diameter_sample_m']['mean'], 2.0, delta=0.4)
        np.testing.assert_allclose(self.cat.biomass_from_diameter(s.species_index, geo['diameter_m']), s.agb_dry_kg, rtol=2e-15)

    def test_bounded_impossible_and_degenerate_cases(self):
        E = initialization.GenerationError
        with self.assertRaises(E):  # exclusion request far above what sequential placement reaches -> attempts exhausted
            initialization.generate_population(self.cat, self.config(
                overlap='disallowed', domain={'nx': 10, 'ny': 10, 'dx_m': 1.0}, limits={'max_attempts': 3000, 'max_plants': 100000},
                species={'shrub': {'target_cover': 0.85, 'diameter_mean_m': 1.0, 'diameter_std_m': 0.0, 'initial_age_days': 1}}))
        with self.assertRaises(E):  # max_plants checked before sampling
            initialization.generate_population(self.cat, self.config(limits={'max_plants': 10}))
        with self.assertRaises(E):  # raster bound checked before allocation
            initialization.generate_population(self.cat, self.config(limits={'max_cover_cells': 100}))
        with self.assertRaises(E):  # working-memory bound checked before allocation
            initialization.generate_population(self.cat, self.config(limits={'max_working_bytes': 10}))
        with self.assertRaises(E):  # tolerance unattainable in zero iterations
            initialization.generate_population(self.cat, self.config(max_iterations=0, cover_tolerance=1e-6))
        C = traits.CatalogError
        with self.assertRaises(C):  # acceptance cannot be disabled
            initialization.generate_population(self.cat, self.config(cover_tolerance=None))
        with self.assertRaises(C):
            initialization.generate_population(self.cat, self.config(cover_tolerance=0.0))
        with self.assertRaises(C):
            initialization.generate_population(self.cat, self.config(species={'BOER': {'target_cover': 0.1, 'diameter_mean_m': 1.0,
                                                                                        'diameter_std_m': 0.0, 'initial_age_days': 1}}))
        with self.assertRaises(C):
            initialization.generate_population(self.cat, self.config(bogus=1))
        with self.assertRaises(C):
            initialization.generate_population(self.cat, self.config(species={'grass': {'target_cover': 1.0, 'diameter_mean_m': 1.0,
                                                                                         'diameter_std_m': 0.0, 'initial_age_days': 1}}))
        z, rz = initialization.generate_population(self.cat, self.config(
            species={'grass': {'target_cover': 0.0, 'diameter_mean_m': 0.4, 'diameter_std_m': 0.1, 'initial_age_days': 1},
                     'shrub': {'target_cover': 0.05, 'diameter_mean_m': 1.0, 'diameter_std_m': 0.0, 'initial_age_days': 1}}))
        self.assertEqual(rz['species'][0]['count'], 0)
        self.assertTrue(np.all(z.species_of() == 'shrub'))
        e, re_ = initialization.generate_population(self.cat, self.config(species={}, time_days=3.5))
        self.assertEqual(len(e), 0)
        self.assertEqual(e.time_days, 3.5)
        self.assertEqual(re_['n_plants'], 0)
        e2, _ = initialization.generate_population(self.cat, self.config(species={}, overlap='disallowed'))
        self.assertEqual(len(e2), 0)


# =============================================================== recording cadence
class TestRecording(unittest.TestCase):
    def setUp(self):
        self.cat = example_catalog()

    def test_cadence_initial_final_skip_and_errors(self):
        s = small_state(self.cat)
        with tempfile.TemporaryDirectory() as td:
            tpl = os.path.join(td, 'veg_{index:03d}_{kind}_{time_days:.1f}.csv')
            with self.assertRaises(recording.RecordingError):
                recording.SnapshotWriter(os.path.join(td, 'same.csv'), cadence_days=10.0)  # would overwrite itself
            with self.assertRaises(recording.RecordingError):
                recording.SnapshotWriter(tpl, cadence_days=0.0)
            w = recording.SnapshotWriter(tpl, cadence_days=10.0)
            with self.assertRaises(recording.RecordingError):
                w.maybe_write(s)  # no origin yet
            p0 = w.write_initial(s)
            self.assertTrue(os.path.exists(p0))
            s.advance_time(5.0)
            self.assertIsNone(w.maybe_write(s))
            s.advance_time(5.0)
            p1 = w.maybe_write(s)
            self.assertIsNotNone(p1)
            self.assertIsNone(w.maybe_write(s))  # not due twice at the same time
            s.advance_time(25.0)                 # t = 35: requested times 20 AND 30 were missed
            p2 = w.maybe_write(s)
            self.assertEqual(w.records[-1]['skipped_due_times_days'], [20.0, 30.0])
            self.assertEqual(w.records[-1]['time_days'], 35.0)   # the ACTUAL state time, nothing fabricated
            self.assertEqual(w.next_due_days, 40.0)
            self.assertEqual(w.skipped_total, 2)
            s.advance_time(2.0)
            p3 = w.write_final(s)
            self.assertEqual([r['kind'] for r in w.records], ['initial', 'cadence', 'cadence', 'final'])
            self.assertEqual([r['time_days'] for r in w.records], [0.0, 10.0, 35.0, 37.0])
            for p in (p0, p1, p2, p3):
                back = state.read_plant_csv(p, self.cat)
                self.assertEqual(len(back), 6)
            self.assertEqual(back.time_days, 37.0)
            self.assertEqual(back.provenance['snapshot_kind'], 'final')
            man = w.manifest()
            self.assertEqual(man['n_snapshots'], 4)
            self.assertTrue(os.path.exists(w.write_manifest(os.path.join(td, 'manifest.json'))))
            # after a snapshot at 10, a snapshot exactly at t=30 misses only 20
            s30 = small_state(self.cat)
            w30 = recording.SnapshotWriter(os.path.join(td, 'b_{index}_{time_days:.1f}.csv'), cadence_days=10.0, start_time_days=0.0)
            s30.advance_time(10.0)
            w30.maybe_write(s30)
            self.assertEqual(w30.records[-1]['skipped_due_times_days'], [])
            s30.advance_time(20.0)
            w30.maybe_write(s30)
            self.assertEqual(w30.records[-1]['skipped_due_times_days'], [20.0])
            self.assertEqual(w30.records[-1]['time_days'], 30.0)
            self.assertEqual(w30.next_due_days, 40.0)
            # time must not go backwards; unresolved state refused without side effects
            s.time_days = 1.0
            with self.assertRaises(recording.RecordingError):
                w.write_final(s)
            unres = small_state(self.cat, resolved=False)
            unres.time_days = 99.0
            n_before = len(os.listdir(td))
            with self.assertRaises(state.StateError):
                w.write_final(unres)
            self.assertEqual(len(os.listdir(td)), n_before)
            self.assertEqual(len(w.records), 4)
            w3 = recording.SnapshotWriter(os.path.join(td, 'fixed.csv'), overwrite=True)
            w3.write_initial(s)
            s.time_days = 2.0
            w3.write_final(s)  # explicit overwrite allowed
            self.assertIsNone(w3.maybe_write(s))  # no cadence -> only explicit snapshots

    def test_failed_writes_leave_scheduling_untouched(self):
        s = small_state(self.cat)
        with tempfile.TemporaryDirectory() as td:
            # failed initial write (missing directory) changes nothing, including the cadence origin
            w = recording.SnapshotWriter(os.path.join(td, 'missing', 'plants_{index}.csv'), cadence_days=7.0)
            before = w.manifest()
            with self.assertRaises(OSError):
                w.write_initial(s)
            self.assertEqual(w.manifest(), before)
            self.assertIsNone(w.next_due_days)
            self.assertIsNone(w.start_time_days)
            # failed initial validation (unresolved) changes nothing either
            unres = small_state(self.cat, resolved=False)
            w = recording.SnapshotWriter(os.path.join(td, 'u_{index}.csv'), cadence_days=7.0)
            with self.assertRaises(state.StateError):
                w.write_initial(unres)
            self.assertEqual(w.records, [])
            self.assertIsNone(w.start_time_days)
            # ACTUAL destination collisions (same kind/index/time) are refused and do not advance either writer
            tpl = os.path.join(td, 'veg_{kind}_{time_days:.1f}.csv')
            wa = recording.SnapshotWriter(tpl, cadence_days=10.0)
            wb = recording.SnapshotWriter(tpl, cadence_days=10.0, start_time_days=0.0)
            pa = wa.write_initial(s)
            self.assertEqual(pa, os.path.join(td, 'veg_initial_0.0.csv'))
            with self.assertRaises(FileExistsError):
                wb.write_initial(s)              # same kind, same time -> same path
            self.assertEqual(wb.records, [])
            self.assertEqual(wb.next_due_days, 10.0)
            s.advance_time(10.0)
            pc = wa.maybe_write(s)
            self.assertEqual(pc, os.path.join(td, 'veg_cadence_10.0.csv'))
            self.assertTrue(wb.due(10.0))
            with self.assertRaises(FileExistsError):
                wb.maybe_write(s)                # same cadence destination exists
            self.assertEqual(wb.next_due_days, 10.0)   # bookkeeping untouched after the failed write
            self.assertEqual(wb.records, [])
            self.assertEqual(wb.skipped_total, 0)
            with open(pc, 'r', encoding='utf-8') as fh:
                self.assertIn('"time_days": 10.0', fh.readline())


# =============================================================== events
class TestEvents(unittest.TestCase):
    def setUp(self):
        self.cat = example_catalog()

    def test_construction_validation_immutability_and_dict_round_trip(self):
        E = events.EventError
        ero = np.array([0.01, 0.0])
        ev = events.SurfaceEventDiagnostics([1, 3], 0.0, 2.0, 'maple_wind', erosion_depth_m=ero,
                                            deposition_depth_m=[0.0, 0.3], abrasion_dose=[5.0, 0.0], abrasion_unit='kg m^-1')
        ero[0] = -5.0                                             # caller array is not aliased
        self.assertEqual(ev.erosion_depth_m[0], 0.01)
        np.testing.assert_allclose(ev.net_surface_delta_m, [-0.01, 0.3])
        with self.assertRaises(ValueError):
            ev.erosion_depth_m[0] = -1.0                          # read-only arrays: negative depths impossible
        with self.assertRaises(ValueError):
            ev.plant_id[0] = 999
        with self.assertRaises(AttributeError):
            ev.plant_id = np.array([999])                          # attributes cannot be rebound
        with self.assertRaises(AttributeError):
            ev.source = 'other'
        back = events.SurfaceEventDiagnostics.from_dict(json.loads(json.dumps(ev.to_dict())))
        np.testing.assert_array_equal(back.plant_id, ev.plant_id)
        np.testing.assert_array_equal(back.abrasion_dose, ev.abrasion_dose)
        self.assertEqual(back.abrasion_unit, 'kg m^-1')
        with self.assertRaises(E):
            events.SurfaceEventDiagnostics([1, 1], 0.0, 1.0, 'maple_wind')               # duplicate ids
        with self.assertRaises(E):
            events.SurfaceEventDiagnostics([1], 0.0, 1.0, 'maple_wind', erosion_depth_m=[-0.1])
        with self.assertRaises(E):
            events.SurfaceEventDiagnostics([1], 2.0, 1.0, 'maple_wind')                  # end < start
        with self.assertRaises(E):
            events.SurfaceEventDiagnostics([1], 0.0, 1.0, 'wind')                        # unknown source
        with self.assertRaises(E):
            events.SurfaceEventDiagnostics([1], 0.0, 1.0, 'maple_wind', abrasion_dose=[1.0])  # unit missing
        with self.assertRaises(E):
            events.SurfaceEventDiagnostics([1], 0.0, float('nan'), 'maple_wind')
        with self.assertRaises(E):
            events.SurfaceEventDiagnostics([1.5], 0.0, 1.0, 'maple_wind')
        for mutate in (lambda d: d.__setitem__('schema_version', 7), lambda d: d.__setitem__('depth_unit', 'cm'),
                       lambda d: d.__setitem__('extra', 1), lambda d: d.__setitem__('erosion_depth_m', [-1.0, 0.0])):
            d = ev.to_dict()
            mutate(d)
            with self.assertRaises(E):
                events.SurfaceEventDiagnostics.from_dict(d)

    def test_apply_defaults_to_diagnostics_and_rolls_back_failed_handlers(self):
        s = small_state(self.cat)
        snap = snapshot(s)
        ev = events.SurfaceEventDiagnostics([1, 4], 0.0, 2.0, 'syrup_water', deposition_depth_m=[0.5, 0.1])
        with self.assertRaises(events.EventError):
            events.apply_event_diagnostics(s, ev)  # population clock (0) has not reached the event end (2)
        s.advance_time(2.0)
        snap = snapshot(s)
        rep = events.apply_event_diagnostics(s, ev)
        self.assertEqual(rep['n_plants'], 2)
        self.assertTrue(rep['response'].startswith('none'))
        np.testing.assert_array_equal(s.ids[rep['indices']], [1, 4])
        assert_unchanged(self, s, snap)
        bad_ids = events.SurfaceEventDiagnostics([1, 999], 0.0, 1.0, 'maple_wind')
        with self.assertRaises(events.EventError):
            events.apply_event_diagnostics(s, bad_ids)
        assert_unchanged(self, s, snap)
        dom_before = s.domain
        ids_before = ev.plant_id.copy()

        def evil(view, event, idx):
            attempts = 0
            for action in (lambda: event.plant_id.__setitem__(0, 999),
                           lambda: event.erosion_depth_m.__setitem__(slice(None), -1.0),
                           lambda: setattr(event, 'plant_id', np.array([5, 6])),
                           lambda: setattr(view.domain, 'nx', 1),
                           lambda: view.agb_dry_kg.__setitem__(slice(None), 0.0),
                           lambda: setattr(view, 'status', None),
                           lambda: idx.__setitem__(0, 5)):
                try:
                    action()
                except (ValueError, AttributeError):
                    attempts += 1
            raise RuntimeError(f'species law exploded after {attempts} blocked mutations')

        with self.assertRaises(events.EventError):
            events.apply_event_diagnostics(s, ev, handler=evil)
        assert_unchanged(self, s, snap)
        self.assertIs(s.domain, dom_before)
        self.assertEqual(s.domain.nx, 20)
        np.testing.assert_array_equal(ev.plant_id, ids_before)
        np.testing.assert_array_equal(ev.erosion_depth_m, 0.0)

        def redirect(view, event, idx):
            object.__setattr__(event, 'plant_id', np.array([2, 3], dtype=np.int64))   # bypasses the guard on purpose
            return {'agb_dry_kg': np.array([0.01, 0.01])}

        ev2 = events.SurfaceEventDiagnostics([1, 4], 0.0, 2.0, 'syrup_water', deposition_depth_m=[0.5, 0.1])
        with self.assertRaises(events.EventError):
            events.apply_event_diagnostics(s, ev2, handler=redirect)                   # IDs changed -> refused
        assert_unchanged(self, s, snap)

        def invalid(view, event, idx):
            return {'agb_dry_kg': np.zeros(len(idx))}  # live plants with zero mass -> rejected

        with self.assertRaises(events.EventError):
            events.apply_event_diagnostics(s, ev, handler=invalid)
        assert_unchanged(self, s, snap)

        def law(view, event, idx):
            B = view.agb_dry_kg[idx] * 0.5
            status = ['dead_standing' if d > 0.4 else 'live' for d in event.deposition_depth_m]
            return {'agb_dry_kg': B, 'seasonal_agb_dry_kg': np.zeros(len(idx)), 'status': status}

        rep = events.apply_event_diagnostics(s, ev, handler=law)
        self.assertGreater(rep['changed'], 0)
        self.assertEqual(s.status[s.indices_of([1])[0]], state.DEAD_STANDING)
        self.assertEqual(s.status[s.indices_of([4])[0]], state.LIVE)
        self.assertAlmostEqual(s.agb_dry_kg[s.indices_of([4])[0]], 0.9)
        self.assertAlmostEqual(s.agb_dry_kg[s.indices_of([1])[0]], 0.06)


# =============================================================== runtime bridge
class FakeSim:
    def __init__(self, ids, B, pid, names, E=None, day=3, ids_dtype=np.int64, pid_dtype=np.int32):
        self.host_inputs = {'ids': np.asarray(ids, dtype=ids_dtype)}
        self.B = np.asarray(B, dtype=np.float64)
        self.host_profile_ids = np.asarray(pid, dtype=pid_dtype)
        self.profile_names = list(names)
        self.stepwat = None if E is None else types.SimpleNamespace(E=np.asarray(E, dtype=np.float64))
        self.day = day


class TestRuntimeBridge(unittest.TestCase):
    def setUp(self):
        self.cat = example_catalog()
        self.dom = state.Domain(8, 8, 1.0)
        self.pop = state.PlantState.from_records(self.cat, self.dom, [
            rec(10, 'grass', 1.0, 1.0, 0.0, 0.1, age=100.0), rec(20, 'grass', 2.0, 2.0, 0.0, 0.2, age=200.0),
            rec(30, 'shrub', 4.0, 4.0, 0.0, 2.0, age=300.0), rec(40, 'shrub', 6.0, 6.0, 0.0, 0.0, status='removed')])
        self.names = ['grass', 'shrub']

    def test_capture_by_id_with_reordering_profiles_ages_and_type_checks(self):
        names = self.names
        sim = FakeSim([30, 10, 20], [3.0, 0.5, 0.7], [1, 0, 0], names, E=[0.25, 0.0, 0.1])
        out = runtime.capture_snapshot(self.pop, types.SimpleNamespace(vegetation=sim))  # CoupledSimulation-like
        self.assertIs(out, self.pop)
        self.assertEqual(self.pop.agb_dry_kg[self.pop.indices_of([30])[0]], 3.0)
        self.assertEqual(self.pop.agb_dry_kg[self.pop.indices_of([10])[0]], 0.5)
        self.assertEqual(self.pop.seasonal_agb_dry_kg[self.pop.indices_of([20])[0]], 0.1)
        self.assertEqual(self.pop.age_days[self.pop.indices_of([10])[0]], 100.0)  # ages/clock untouched by capture
        self.assertEqual(self.pop.time_days, 0.0)
        self.assertEqual(self.pop.agb_dry_kg[self.pop.indices_of([40])[0]], 0.0)   # removed identity retained
        self.assertEqual(self.pop.provenance['bridge']['captured_from'], 'CoupledSimulation')
        snap = snapshot(self.pop)
        B = runtime.BridgeError
        cases = [FakeSim([30, 10], [3.0, 0.5], [1, 0], names, E=[0.0, 0.0]),                      # live plant 20 missing
                 FakeSim([30, 10, 20, 99], [3.0, 0.5, 0.7, 1.0], [1, 0, 0, 0], names, E=[0.0] * 4),  # unknown id
                 FakeSim([30, 10, 20, 40], [3.0, 0.5, 0.7, 1.0], [1, 0, 0, 1], names, E=[0.0] * 4),  # removed plant in driver
                 FakeSim([30, 10, 20], [3.0, 0.5, 0.7], [0, 0, 0], names, E=[0.0] * 3),             # profile mismatch for id 30
                 FakeSim([30, 10, 20], [3.0, 0.5, 0.7], [1, 0, 0], names, E=[4.0, 0.0, 0.0]),       # E > B
                 FakeSim([30, 10, 20], [3.0, -0.5, 0.7], [1, 0, 0], names, E=[0.0] * 3),
                 FakeSim([30, 10, 10], [3.0, 0.5, 0.7], [1, 0, 0], names, E=[0.0] * 3),             # duplicate driver ids
                 FakeSim([30.0, 10.0, 20.0], [3.0, 0.5, 0.7], [1, 0, 0], names, E=[0.0] * 3, ids_dtype=np.float64),
                 FakeSim([30, 10, 20], [3.0, 0.5, 0.7], [1.0, 0.0, 0.0], names, E=[0.0] * 3, pid_dtype=np.float64),
                 FakeSim([30, 10, 20], [3.0, 0.5, 0.7], [1, 0, 5], names, E=[0.0] * 3),             # profile id out of range
                 FakeSim([2 ** 63 + 5, 10, 20], [3.0, 0.5, 0.7], [1, 0, 0], names, E=[0.0] * 3, ids_dtype=np.uint64),
                 FakeSim([30, 10, 20], [3.0, 0.5, 0.7], [1, 0, 0], names, E=[0.0] * 3, pid_dtype=bool)]
        for sim in cases:
            with self.assertRaises(B, msg=repr(sim.host_inputs['ids'])):
                runtime.capture_snapshot(self.pop, sim)
            assert_unchanged(self, self.pop, snap)
        copy_out = runtime.capture_snapshot(self.pop, FakeSim([30, 10, 20], [1.0, 1.0, 1.0], [1, 0, 0], names, E=[0.0] * 3), in_place=False)
        self.assertEqual(self.pop.agb_dry_kg[self.pop.indices_of([30])[0]], 3.0)
        self.assertEqual(copy_out.agb_dry_kg[copy_out.indices_of([30])[0]], 1.0)

    def test_capture_from_non_step_driver_never_erases_seasonal_mass_silently(self):
        sim = FakeSim([30, 10, 20], [3.0, 0.5, 0.7], [1, 0, 0], self.names)   # legacy driver: no E
        runtime.capture_snapshot(self.pop, sim)                                # population E was 0: fine
        self.pop.update_plants([20], seasonal_agb_dry_kg=[0.3])
        snap = snapshot(self.pop)
        with self.assertRaises(runtime.BridgeError):
            runtime.capture_snapshot(self.pop, sim)
        assert_unchanged(self, self.pop, snap)
        kept = runtime.capture_snapshot(self.pop, sim, in_place=False, seasonal_policy='keep')
        self.assertEqual(kept.seasonal_agb_dry_kg[kept.indices_of([20])[0]], 0.3)
        zeroed = runtime.capture_snapshot(self.pop, sim, in_place=False, seasonal_policy='zero')
        self.assertEqual(zeroed.seasonal_agb_dry_kg[zeroed.indices_of([20])[0]], 0.0)
        with self.assertRaises(runtime.BridgeError):   # keep would violate E <= new B
            runtime.capture_snapshot(self.pop, FakeSim([30, 10, 20], [3.0, 0.5, 0.2], [1, 0, 0], self.names), seasonal_policy='keep')
        assert_unchanged(self, self.pop, snap)

    def test_legacy_inputs_check_the_species_law_not_one_biomass(self):
        B = runtime.BridgeError
        with self.assertRaises(B):  # two live species with different lateral multipliers (legacy has one)
            runtime.legacy_inputs_from_state(self.pop, self.cat)
        self.pop.update_plants([30], status='removed', agb_dry_kg=[0.0])
        with self.assertRaises(B):  # power allometry (b=0.45) != legacy square-root area law -> refused by default
            runtime.legacy_inputs_from_state(self.pop, self.cat)
        res = runtime.legacy_inputs_from_state(self.pop, self.cat, biomass_policy='accept_driver_recompute')
        self.assertFalse(res['biomass_check']['law_consistent'])
        self.assertEqual(sorted(res['excluded_ids']), [30, 40])
        np.testing.assert_array_equal(res['inputs']['ids'], [10, 20])
        np.testing.assert_array_equal(res['inputs']['root_profile_id'], [0, 0])
        # a law that matches B at ONE size only: a_d chosen so B=0.1 agrees exactly, exponent 0.45 -> still refused
        doc = self.cat.document
        a_match = 2.0 / math.sqrt(math.pi) * 0.1 ** (0.5 - 0.45)
        doc['species']['grass']['allometry']['diameter'] = {'a': a_match, 'b': 0.45}
        one_size = traits.SpeciesCatalog(doc)
        pop1 = state.PlantState.from_records(one_size, self.dom, [rec(1, 'grass', 1.0, 1.0, 0.0, 0.1)])
        with self.assertRaises(B):
            runtime.legacy_inputs_from_state(pop1, one_size)
        acc = runtime.legacy_inputs_from_state(pop1, one_size, biomass_policy='accept_driver_recompute')
        self.assertLess(acc['biomass_check']['initial_biomass_max_relative_deviation'], 1e-9)
        self.assertFalse(acc['biomass_check']['law_consistent'])
        cat2 = consistent_catalog()
        pop2 = state.PlantState.from_records(cat2, self.dom, [rec(1, 'grass', 1.0, 1.0, 0.0, 0.3), rec(2, 'shrub', 5.0, 5.0, 0.0, 2.0)])
        res2 = runtime.legacy_inputs_from_state(pop2, cat2)
        self.assertTrue(res2['biomass_check']['law_consistent'])
        np.testing.assert_allclose(math.pi * res2['inputs']['canopy_radius'] ** 2, [0.3, 2.0], rtol=1e-12)
        np.testing.assert_allclose(res2['inputs']['root_radius'], 2.0 * res2['inputs']['canopy_radius'])
        self.assertEqual(res2['root_layer_fractions'], {'grass': [0.8, 0.2], 'shrub': [0.25, 0.75]})
        # STEP density: the law coefficient must match 2/sqrt(pi*rho); NaN density cannot bypass refusal
        traits_row = {'canopy_biomass_density_kg_m2': 4.0}
        with self.assertRaises(B):
            runtime.legacy_inputs_from_state(pop2, cat2, stepwat_traits={'grass': traits_row, 'shrub': traits_row})
        with self.assertRaises(B):
            runtime.legacy_inputs_from_state(pop2, cat2, stepwat_traits={'grass': {'canopy_biomass_density_kg_m2': float('nan')},
                                                                         'shrub': {'canopy_biomass_density_kg_m2': float('nan')}})
        for bad_rtol in (float('nan'), -1e-9, True):
            with self.assertRaises(B):
                runtime.legacy_inputs_from_state(pop2, cat2, rtol=bad_rtol)
        with self.assertRaises(B):
            runtime.legacy_inputs_from_state(pop2, cat2, profile_names=['grass', 'BOER'])    # unknown profile name
        with self.assertRaises(B):
            runtime.legacy_inputs_from_state(pop2, cat2, profile_names=['grass'])            # live shrub not covered
        off = state.Domain(8, 8, 1.0, 1.0, 5.0, 0.0)
        pop3 = state.PlantState.from_records(cat2, off, [rec(1, 'grass', 6.0, 1.0, 0.0, 0.3)])
        with self.assertRaises(B):
            runtime.legacy_inputs_from_state(pop3, cat2)
        # species table mismatch (same hash cannot happen; a different catalog is refused by hash)
        with self.assertRaises(B):
            runtime.legacy_inputs_from_state(pop2, self.cat)

    def test_legacy_inputs_seasonal_mass_policies_and_restore(self):
        cat2 = consistent_catalog()
        pop = state.PlantState.from_records(cat2, self.dom, [rec(1, 'grass', 1.0, 1.0, 0.0, 0.3, E=0.1), rec(2, 'shrub', 5.0, 5.0, 0.0, 2.0)])
        snap = snapshot(pop)
        with self.assertRaises(runtime.BridgeError):   # fresh driver starts E = 0
            runtime.legacy_inputs_from_state(pop, cat2)
        assert_unchanged(self, pop, snap)
        ack = runtime.legacy_inputs_from_state(pop, cat2, seasonal_policy='acknowledge_reset')
        self.assertEqual(ack['seasonal_agb_dry_kg']['n_nonzero'], 1)
        assert_unchanged(self, pop, snap)                       # the population keeps its E
        res = runtime.legacy_inputs_from_state(pop, cat2, seasonal_policy='restore')
        np.testing.assert_array_equal(res['seasonal_agb_dry_kg']['values'], [0.1, 0.0])
        B_drv = math.pi * res['inputs']['canopy_radius'] ** 2

        def fresh(**kw):
            return FakeSim(res['inputs']['ids'], B_drv, res['inputs']['root_profile_id'], res['profile_names'], E=[0.0, 0.0], day=0, **kw)

        # fresh-driver contract: a mid-run driver (day != 0) or coupled driver with accepted days is refused untouched
        mid = FakeSim(res['inputs']['ids'], B_drv, res['inputs']['root_profile_id'], res['profile_names'], E=[0.0, 0.0], day=5)
        with self.assertRaises(runtime.BridgeError):
            runtime.restore_seasonal_mass(pop, mid)
        np.testing.assert_array_equal(mid.stepwat.E, [0.0, 0.0])
        coupled_mid = types.SimpleNamespace(vegetation=fresh(), days=2)
        with self.assertRaises(runtime.BridgeError):
            runtime.restore_seasonal_mass(pop, coupled_mid)
        np.testing.assert_array_equal(coupled_mid.vegetation.stepwat.E, [0.0, 0.0])
        # same association checks as capture, before any mutation
        partial = FakeSim(res['inputs']['ids'][:1], B_drv[:1], res['inputs']['root_profile_id'][:1], res['profile_names'], E=[0.0], day=0)
        with self.assertRaises(runtime.BridgeError):
            runtime.restore_seasonal_mass(pop, partial)
        np.testing.assert_array_equal(partial.stepwat.E, [0.0])
        too_small = FakeSim(res['inputs']['ids'], [0.05, 2.0], res['inputs']['root_profile_id'], res['profile_names'], E=[0.0, 0.0], day=0)
        with self.assertRaises(runtime.BridgeError):   # E (0.1) > actual driver B (0.05)
            runtime.restore_seasonal_mass(pop, too_small)
        np.testing.assert_array_equal(too_small.stepwat.E, [0.0, 0.0])
        sim = fresh()
        self.assertEqual(runtime.restore_seasonal_mass(pop, sim), 2)
        np.testing.assert_allclose(sim.stepwat.E, [0.1, 0.0])
        coupled_fresh = types.SimpleNamespace(vegetation=fresh(), days=0)
        self.assertEqual(runtime.restore_seasonal_mass(pop, coupled_fresh), 2)
        with self.assertRaises(runtime.BridgeError):   # non-STEP driver cannot restore
            runtime.restore_seasonal_mass(pop, FakeSim(res['inputs']['ids'], [0.3, 2.0], [0, 1], res['profile_names'], day=0))
        back = runtime.capture_snapshot(pop, sim, in_place=False)
        self.assertEqual(back.seasonal_agb_dry_kg[back.indices_of([1])[0]], 0.1)

    def test_device_vector_metadata_is_checked_before_transfer(self):
        class DeviceVector:
            def __init__(self, shape, value=0.01, dtype='float64', received=None):
                self.shape, self.dtype, self.calls, self.value, self.received = shape, np.dtype(dtype), 0, value, received

            def get(self):
                self.calls += 1
                if self.received is not None:
                    return self.received
                if self.shape != (1,):
                    raise AssertionError('unexpected large device-to-host transfer')
                return np.array([self.value])

        def driver(B, E=None):
            return types.SimpleNamespace(host_inputs={'ids': np.array([17], np.int64)}, B=B, host_profile_ids=np.array([0], np.int32),
                                         profile_names=['grass'], stepwat=None if E is None else types.SimpleNamespace(E=E), day=0)

        bad_B = DeviceVector((3, 2_000_000))
        with self.assertRaises(runtime.BridgeError):
            runtime.read_driver_arrays(driver(bad_B))
        self.assertEqual(bad_B.calls, 0)
        bad_E = DeviceVector((3, 2_000_000))
        with self.assertRaises(runtime.BridgeError):
            runtime.read_driver_arrays(driver(np.array([0.02]), bad_E))
        self.assertEqual(bad_E.calls, 0)
        bad_dtype = DeviceVector((1,), dtype='bool')
        with self.assertRaises(runtime.BridgeError):
            runtime.read_driver_arrays(driver(bad_dtype))
        self.assertEqual(bad_dtype.calls, 0)
        no_meta = types.SimpleNamespace(get=lambda: np.array([0.02]))      # device-like object without shape/dtype
        with self.assertRaises(runtime.BridgeError):
            runtime.read_driver_arrays(driver(no_meta))
        lying = DeviceVector((1,), received=np.zeros((3,)))               # metadata fine, received vector wrong
        with self.assertRaises(runtime.BridgeError):
            runtime.read_driver_arrays(driver(lying))
        self.assertEqual(lying.calls, 1)
        B, E = DeviceVector((1,), value=0.02), DeviceVector((1,), value=0.001)
        got = runtime.read_driver_arrays(driver(B, E))
        self.assertEqual((B.calls, E.calls), (1, 1))                       # exactly one host read per valid vector
        self.assertEqual((got['B'][0], got['E'][0]), (0.02, 0.001))
        self.assertTrue(got['stepwat'])

    def test_capture_from_real_legacy_driver(self):
        try:
            from benchmarks import daily_water_growth as dwg
        except Exception as exc:  # numba / benchmarks unavailable in this interpreter
            self.skipTest(f'legacy driver unavailable: {exc!r}')
        cat2 = consistent_catalog()
        pop = state.PlantState.from_records(cat2, self.dom, [rec(7, 'grass', 1.5, 1.5, 0.0, 0.3), rec(3, 'shrub', 5.5, 5.5, 0.0, 2.0),
                                                            rec(5, 'grass', 2.5, 6.5, 0.0, 0.1)])
        res = runtime.legacy_inputs_from_state(pop, cat2)
        self.assertTrue(res['biomass_check']['law_consistent'])
        case = {'name': 'bridge', 'nx': 8, 'ny': 8, 'dx': 1.0, 'grass_count': 3, 'shrub_count': 0, 'grass_diameter': 0.5,
                'shrub_diameter': 1.0, 'root_multiplier': res['case_fragment']['root_multiplier'], 'pattern': 'uniform', 'seed': 1,
                'layers': 2, 'max_edges': 100000, 'max_bytes': 50_000_000, 'root_layer_fractions': res['root_layer_fractions']}
        sim = dwg.Simulation(case, 'cpu', inputs=res['inputs'], vegetation_only=True)
        pop.update_plants([7, 3, 5], agb_dry_kg=[1.0, 1.0, 1.0], seasonal_agb_dry_kg=[0.0, 0.0, 0.0])
        runtime.capture_snapshot(pop, sim)
        np.testing.assert_allclose(pop.agb_dry_kg[pop.indices_of([7, 3, 5])], [0.3, 2.0, 0.1], rtol=1e-9)


# =============================================================== CLI, examples, import isolation, legacy exports
class TestCLIAndExamples(unittest.TestCase):
    def test_help_and_usage(self):
        with self.assertRaises(SystemExit) as cm:
            run_cli(['--help'])
        self.assertEqual(cm.exception.code, 0)
        for sub in ('init', 'generate', 'inspect', 'export-canopy', 'validate-species'):
            with self.assertRaises(SystemExit) as cm:
                run_cli([sub, '--help'])
            self.assertEqual(cm.exception.code, 0)
        code, _, _ = run_cli([])
        self.assertEqual(code, 2)

    def test_validate_species_example(self):
        code, out, _ = run_cli(['validate-species', '--species', str(SPECIES_JSON)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)['species_ids'], ['grass', 'shrub'])

    def test_init_inspect_export_round_trip(self):
        cat = example_catalog()
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, 'init.csv')
            code, text, err = run_cli(['init', '--species', str(SPECIES_JSON), '--plants', str(PLANTS_CSV), '--nx', '20', '--ny', '20',
                                       '--dx-m', '1.0', '--surface-constant', '100.0', '--out', out])
            self.assertEqual(code, 0, err)
            s = state.read_plant_csv(out, cat)
            self.assertTrue(s.establishment_resolved)
            self.assertEqual(s.establishment_ground_m[s.indices_of([1])[0]], 100.0)
            self.assertEqual(s.establishment_ground_m[s.indices_of([3])[0]], 99.9)   # explicit anchor preserved
            self.assertEqual(s.counts(), {'live': 4, 'dead_standing': 1, 'removed': 1})
            code, _, err = run_cli(['init', '--species', str(SPECIES_JSON), '--plants', str(PLANTS_CSV), '--nx', '20', '--ny', '20',
                                    '--dx-m', '1.0', '--surface-constant', '100.0', '--out', out])
            self.assertEqual(code, 1)  # exists, no --overwrite
            self.assertIn('exists', err)
            # without a surface the SURFACE token is preserved in an INITIAL file; it cannot be published
            raw = os.path.join(td, 'raw.csv')
            code, _, err = run_cli(['init', '--species', str(SPECIES_JSON), '--plants', str(PLANTS_CSV), '--nx', '20', '--ny', '20',
                                    '--dx-m', '1.0', '--out', raw])
            self.assertEqual(code, 0, err)
            with open(raw, 'r', encoding='utf-8') as fh:
                head = fh.read()
            self.assertIn('SURFACE', head)
            self.assertIn('"initial": true', head)
            code, _, err = run_cli(['export-canopy', '--species', str(SPECIES_JSON), '--plants', raw, '--surface-constant', '100.0',
                                    '--out', os.path.join(td, 'v.json')])
            self.assertEqual(code, 1)
            # export requires the CURRENT surface
            code, _, err = run_cli(['export-canopy', '--species', str(SPECIES_JSON), '--plants', out, '--out', os.path.join(td, 'v.json')])
            self.assertEqual(code, 1)
            self.assertFalse(os.path.exists(os.path.join(td, 'v.json')))
            # inspect with a current surface and a cover measurement
            surf = os.path.join(td, 'surface.npy')
            np.save(surf, np.full((20, 20), 100.5))
            code, text, err = run_cli(['inspect', '--species', str(SPECIES_JSON), '--plants', out, '--surface', surf, '--cover-subsamples', '8'])
            self.assertEqual(code, 0, err)
            info = json.loads(text)
            self.assertAlmostEqual(info['burial_exposure']['burial_depth_m']['max'], 0.6)
            self.assertIn('grass', info['cover']['per_species'])
            self.assertIn('working_bytes', info['cover']['admission'])
            np.save(surf, np.full((20, 21), 100.5))
            code, _, err = run_cli(['inspect', '--species', str(SPECIES_JSON), '--plants', out, '--surface', surf])
            self.assertEqual(code, 1)   # transposed surface refused
            # canopy export JSON + NPZ against the current (= initial) surface
            vj = os.path.join(td, 'view.json')
            code, text, err = run_cli(['export-canopy', '--species', str(SPECIES_JSON), '--plants', out, '--surface-constant', '100.0', '--out', vj])
            self.assertEqual(code, 0, err)
            self.assertTrue(json.loads(text)['native_maple_valid'])
            with open(vj, 'r', encoding='utf-8') as fh:
                view = json.load(fh)
            self.assertEqual(view['meta']['n_objects'], 5)
            self.assertEqual(view['sidecar']['plant_id'], [1, 2, 3, 4, 5])
            self.assertTrue(view['meta']['native_maple_valid'])
            i3 = view['sidecar']['plant_id'].index(3)
            self.assertAlmostEqual(view['fields']['height_m'][i3], view['sidecar']['intact_height_m'][i3] - 0.1)  # buried 0.1 m
            vz = os.path.join(td, 'view.npz')
            code, _, err = run_cli(['export-canopy', '--species', str(SPECIES_JSON), '--plants', out, '--surface-constant', '100.0',
                                    '--out', vz, '--statuses', 'live'])
            self.assertEqual(code, 0, err)
            with np.load(vz, allow_pickle=False) as z:
                self.assertEqual(z['center_x_m'].shape, (4,))
                self.assertEqual(json.loads(str(z['meta']))['n_objects'], 4)
                self.assertEqual(z['sidecar_plant_id'].tolist(), [1, 2, 3, 4])
            code, _, _ = run_cli(['export-canopy', '--species', str(SPECIES_JSON), '--plants', out, '--surface-constant', '100.0', '--out', vz])
            self.assertEqual(code, 1)  # no overwrite
            # deposition burying the small grasses: refused by default, explicit exclusion works and is labelled
            code, _, err = run_cli(['export-canopy', '--species', str(SPECIES_JSON), '--plants', out, '--surface-constant', '101.0',
                                    '--out', os.path.join(td, 'buried.json')])
            self.assertEqual(code, 1)
            self.assertFalse(os.path.exists(os.path.join(td, 'buried.json')))
            code, text, err = run_cli(['export-canopy', '--species', str(SPECIES_JSON), '--plants', out, '--surface-constant', '101.0',
                                       '--out', os.path.join(td, 'buried.json'), '--unsupported', 'exclude'])
            self.assertEqual(code, 0, err)
            self.assertGreater(json.loads(text)['n_unsupported_excluded'], 0)
            code, _, _ = run_cli(['inspect', '--species', str(SPECIES_JSON), '--plants', os.path.join(td, 'missing.csv')])
            self.assertEqual(code, 1)

    def test_generate_example_config_and_report_collision(self):
        cat = example_catalog()
        with tempfile.TemporaryDirectory() as td:
            out, rep = os.path.join(td, 'gen.csv'), os.path.join(td, 'gen_report.json')
            code, _, err = run_cli(['generate', '--species', str(SPECIES_JSON), '--config', str(GENERATION_JSON), '--out', out,
                                    '--report', out, '--overwrite'])
            self.assertEqual(code, 1)              # --report must not alias --out
            self.assertFalse(os.path.exists(out))  # refused before any output
            code, text, err = run_cli(['generate', '--species', str(SPECIES_JSON), '--config', str(GENERATION_JSON), '--out', out,
                                       '--report', rep, '--surface-constant', '12.0'])
            self.assertEqual(code, 0, err)
            brief = json.loads(text)
            self.assertTrue(all(sp['accepted'] for sp in brief['species']))
            s = state.read_plant_csv(out, cat)
            self.assertEqual(len(s), brief['n_plants'])
            self.assertTrue(s.establishment_resolved)
            np.testing.assert_array_equal(s.establishment_ground_m, 12.0)
            with open(rep, 'r', encoding='utf-8') as fh:
                report = json.load(fh)
            self.assertEqual(report['seed'], 20261110)
            self.assertEqual(report['domain']['nx'], 128)
            self.assertEqual(report['generation_config']['species']['grass']['diameter_mean_m'], 0.15)
            for sp in report['species']:
                self.assertLessEqual(abs(sp['measured_union_cover'] - sp['target_cover']), 0.005)
            view = canopy.maple_vegetation_view(s, cat, state.constant_surface(s.domain, 12.0))
            self.assertEqual(view['meta']['n_objects'], len(s))
            self.assertTrue(view['meta']['native_maple_valid'])
            # existing report without --overwrite: refused before generation, CSV untouched
            mtime = os.path.getmtime(out)
            code, _, err = run_cli(['generate', '--species', str(SPECIES_JSON), '--config', str(GENERATION_JSON), '--out',
                                    os.path.join(td, 'gen2.csv'), '--report', rep])
            self.assertEqual(code, 1)
            self.assertFalse(os.path.exists(os.path.join(td, 'gen2.csv')))
            self.assertEqual(os.path.getmtime(out), mtime)

    def test_import_isolation_subprocess(self):
        code = ('import sys\n'
                'import leaf, leaf.state, leaf.traits, leaf.canopy, leaf.initialization, leaf.recording, leaf.events, '
                'leaf.runtime, leaf.__main__\n'
                "bad = [m for m in ('numba', 'cupy', 'maple_syrup', 'maple', 'benchmarks', 'benchmarks.daily_water_growth', "
                "'benchmarks.spatial_mapping', 'leaf.coupling', 'leaf.stepwat') if m in sys.modules]\n"
                'print(bad)\n')
        env = dict(os.environ)
        env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
        res = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env, timeout=300)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(res.stdout.strip(), '[]')
        res = subprocess.run([sys.executable, '-m', 'leaf', '--help'], capture_output=True, text=True, env=env, timeout=300)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn('export-canopy', res.stdout)

    def test_legacy_lazy_exports_and_all_consistency(self):
        import leaf
        self.assertEqual(leaf.__version__, '0.1.0')
        for name in leaf.__all__:
            if name in ('CoupledSimulation', 'CoupledPhaseError', 'stepwat', 'coupling'):
                self.assertIn(name, leaf._LAZY_ATTRS)
            elif name != '__version__':
                self.assertIn(name, leaf._LAZY_ATTRS)
                self.assertTrue(hasattr(leaf, name))
        with self.assertRaises(AttributeError):
            getattr(leaf, 'definitely_missing_attribute')
        try:
            from leaf import CoupledSimulation, CoupledPhaseError, stepwat
        except ImportError as exc:
            self.skipTest(f'legacy coupled growth extras unavailable: {exc!r}')
        self.assertTrue(issubclass(CoupledPhaseError, ValueError))
        self.assertTrue(callable(CoupledSimulation))
        self.assertEqual(stepwat.MODEL_ID, 'stepwat_weekly_aboveground_v0.1')
        self.assertIs(leaf.coupling.CoupledSimulation, CoupledSimulation)


if __name__ == '__main__':
    unittest.main()
