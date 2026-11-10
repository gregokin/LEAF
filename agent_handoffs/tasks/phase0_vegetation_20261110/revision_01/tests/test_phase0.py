'''LEAF Phase 0 acceptance tests (unittest, NumPy only; the legacy-driver tests skip without numba).

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
    '''Catalog whose diameter law B = pi r^2 * 1 kg m^-2 matches the legacy fixture (b_d = 0.5, a_d = 2/sqrt(pi)).'''
    doc = copy.deepcopy(example_catalog().document)
    for sid in doc['species']:
        doc['species'][sid]['allometry']['diameter'] = {'a': 2.0 / math.sqrt(math.pi), 'b': 0.5}
        doc['species'][sid]['roots']['lateral_radius_multiplier'] = mult
    doc['catalog_id'] = 'consistent-with-legacy-area-law'
    return traits.SpeciesCatalog(doc)


def small_state(cat, resolved=True, time_days=0.0):
    dom = state.Domain(20, 20, 1.0)
    recs = [rec(1, 'grass', 2.5, 3.5, 'SURFACE', 0.12, 0.0, 730.0),
            rec(2, 'grass', 7.25, 12.75, 'SURFACE', 0.08, 0.01, 365.0),
            rec(3, 'shrub', 10.0, 10.0, 100.25, 2.5, 0.0, 3650.0),
            rec(4, 'shrub', 15.5, 4.5, 'SURFACE', 1.8, 0.2, 1825.0),
            rec(5, 'grass', 18.0, 18.0, 100.0, 0.05, 0.0, 1095.0, 'dead_standing'),
            rec(6, 'shrub', 3.0, 16.0, 100.1, 0.0, 0.0, 5475.0, 'removed')]
    surface = state.constant_surface(dom, 100.0) if resolved else None
    return state.PlantState.from_records(cat, dom, recs, time_days=time_days, surface=surface)


def run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli_main(argv)
    return code, out.getvalue(), err.getvalue()


# =============================================================== species catalog
class TestCatalog(unittest.TestCase):
    def setUp(self):
        self.cat = example_catalog()

    def test_load_and_hash_stability(self):
        self.assertEqual(self.cat.species_ids, ('grass', 'shrub'))
        self.assertTrue(self.cat.hash.startswith('sha256:'))
        pretty = json.dumps(self.cat.document, indent=4, sort_keys=True)  # whitespace + key order differ
        self.assertEqual(traits.SpeciesCatalog.from_text(pretty).hash, self.cat.hash)
        doc = copy.deepcopy(self.cat.document)
        doc['species']['grass']['allometry']['diameter']['a'] = 0.91
        self.assertNotEqual(traits.SpeciesCatalog(doc).hash, self.cat.hash)

    def test_allometry_positive_monotone_and_exact_inverse(self):
        B = np.array([0.01, 0.1, 1.0, 10.0])
        d = self.cat.diameter_from_biomass('grass', B)
        h = self.cat.height_from_biomass('grass', B)
        self.assertTrue(np.all(d > 0) and np.all(np.diff(d) > 0) and np.all(h > 0) and np.all(np.diff(h) > 0))
        self.assertAlmostEqual(self.cat.diameter_from_biomass('grass', 1.0), 0.9)
        back = self.cat.biomass_from_diameter('grass', d)
        np.testing.assert_allclose(back, B, rtol=1e-12)
        idx = np.array([0, 1, 1])
        np.testing.assert_allclose(self.cat.biomass_from_diameter(idx, self.cat.diameter_from_biomass(idx, np.array([0.2, 2.0, 3.0]))),
                                   [0.2, 2.0, 3.0], rtol=1e-12)
        self.assertEqual(self.cat.diameter_from_biomass('shrub', 0.0), 0.0)
        with self.assertRaises(traits.CatalogError):
            self.cat.diameter_from_biomass('grass', -1.0)
        with self.assertRaises(traits.CatalogError):
            self.cat.index('BOER')

    def test_traits_sections_and_lifespan_metadata(self):
        self.assertEqual(self.cat.life_expectancy_days('grass'), 8 * 365.0)
        self.assertEqual(self.cat.root_layers, 2)
        self.assertEqual(self.cat.root_layer_fractions('grass'), [0.8, 0.2])
        self.assertEqual(self.cat.porosity('shrub'), 0.2)
        self.assertEqual(self.cat.law('grass', 'mortality'), 'unimplemented')
        sec = self.cat.section('grass', 'disturbance')
        self.assertIn('fixture_sensitivities', sec)
        sec['law'] = 'mutated'
        self.assertEqual(self.cat.law('grass', 'disturbance'), 'unimplemented')  # deep copy returned
        with self.assertRaises(traits.CatalogError):
            self.cat.stepwat_traits()
        doc = copy.deepcopy(self.cat.document)
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
    def test_validation(self):
        with self.assertRaises(state.StateError):
            state.Domain(0, 4, 1.0)
        with self.assertRaises(state.StateError):
            state.Domain(4, 4, -1.0)
        with self.assertRaises(state.StateError):
            state.Domain(True, 4, 1.0)
        with self.assertRaises(state.StateError):
            state.Domain(4, 4, float('nan'))
        d = state.Domain(4, 3, 0.5, 2.0, 10.0, -5.0)
        self.assertEqual(d.shape, (3, 4))
        self.assertEqual(state.Domain.from_dict(d.to_dict()), d)
        self.assertNotEqual(d, state.Domain(4, 3, 0.5))

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
        text = state.plant_csv_text(s, initial=True)
        self.assertIn('SURFACE', text)
        self.assertTrue(text.startswith('# {'))
        surf = np.arange(400, dtype=float).reshape(20, 20) + 100.0  # value = 100 + row*20 + col
        self.assertEqual(s.resolve_establishment(surf), 3)
        self.assertTrue(s.establishment_resolved)
        i1 = s.indices_of([1])[0]
        self.assertEqual(s.establishment_ground_m[i1], 100.0 + 3 * 20 + 2)  # (2.5, 3.5) -> row 3, col 2
        self.assertEqual(s.establishment_ground_m[s.indices_of([3])[0]], 100.25)  # explicit anchor untouched
        self.assertEqual(s.resolve_establishment(surf + 50.0), 0)  # resolved ONCE
        self.assertEqual(s.establishment_ground_m[i1], 162.0)
        anchored = s.indices_of([1, 2, 4])  # the plants whose anchors came from `surf`
        geo = canopy.derive_geometry(s, self.cat, current_surface=surf + 0.3)  # deposition after establishment
        np.testing.assert_allclose(geo['burial_depth_m'][anchored], 0.3, atol=1e-12)
        np.testing.assert_allclose(geo['exposure_depth_m'][anchored], 0.0)
        np.testing.assert_allclose(geo['surface_delta_m'][anchored], 0.3, atol=1e-12)
        self.assertEqual(s.establishment_ground_m[i1], 162.0)  # anchor preserved after deposition
        geo = canopy.derive_geometry(s, self.cat, current_surface=surf - 0.2)
        np.testing.assert_allclose(geo['exposure_depth_m'][anchored], 0.2, atol=1e-12)
        np.testing.assert_allclose(geo['burial_depth_m'][anchored], 0.0)
        self.assertEqual(s.establishment_ground_m[i1], 162.0)
        self.assertEqual(s.establishment_ground_m[s.indices_of([3])[0]], 100.25)

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
        view = canopy.maple_vegetation_view(back, self.cat)
        self.assertEqual(view['meta']['n_objects'], 0)
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
            # catalog mismatch
            other = consistent_catalog()
            with self.assertRaises(state.StateError):
                state.parse_plant_csv(text, other)
            # supplied domain must agree with the metadata domain
            with self.assertRaises(state.StateError):
                state.parse_plant_csv(text, self.cat, domain=state.Domain(37, 23, 0.7))
            self.assertEqual(len(state.parse_plant_csv(text, self.cat, domain=dom)), n)
            # truncated file (row count differs from metadata n_plants)
            lines = text.splitlines(True)
            with self.assertRaises(state.StateError):
                state.parse_plant_csv(''.join(lines[:-1]), self.cat)
            # malformed row (missing fields) and bad header
            with self.assertRaises(state.StateError):
                state.parse_plant_csv(''.join(lines[:-1]) + lines[-1].rsplit(',', 2)[0] + '\n', self.cat)
            with self.assertRaises(state.StateError):
                state.parse_plant_csv(lines[0] + lines[1].replace('plant_id', 'id') + ''.join(lines[2:]), self.cat)
            # unsupported metadata schema version
            meta = json.loads(lines[0][1:])
            meta['schema_version'] = 99
            with self.assertRaises(state.StateError):
                state.parse_plant_csv('# ' + json.dumps(meta) + '\n' + ''.join(lines[1:]), self.cat)
            # handwritten CSV without metadata needs a domain
            with self.assertRaises(state.StateError):
                state.parse_plant_csv(''.join(lines[1:]), self.cat)
            self.assertEqual(len(state.parse_plant_csv(''.join(lines[1:]), self.cat, domain=dom)), n)

    def test_updates_are_transactional_and_by_id(self):
        s = small_state(self.cat)
        before = {k: getattr(s, k).copy() for k in state.PlantState.ARRAY_NAMES}
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
        for k in state.PlantState.ARRAY_NAMES:
            np.testing.assert_array_equal(getattr(s, k), before[k], err_msg=k)
        idx = s.update_plants([4, 1], agb_dry_kg=[2.0, 0.3], seasonal_agb_dry_kg=[0.5, 0.0])
        np.testing.assert_array_equal(s.ids[idx], [4, 1])
        self.assertEqual(s.agb_dry_kg[s.indices_of([1])[0]], 0.3)
        self.assertEqual(s.seasonal_agb_dry_kg[s.indices_of([4])[0]], 0.5)
        s.update_plants([1], status='dead_standing')
        s.update_plants([1], status='removed', agb_dry_kg=[0.0], seasonal_agb_dry_kg=[0.0])
        self.assertEqual(s.counts(), {'live': 2, 'dead_standing': 1, 'removed': 2})

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

    def test_ids_resolve_regardless_of_row_order(self):
        s = small_state(self.cat)
        perm = np.array([5, 2, 0, 4, 1, 3])
        s2 = state.PlantState(s.domain, s.species_names, s.catalog_hash, ids=s.ids[perm], species_index=s.species_index[perm],
                              x_m=s.x_m[perm], y_m=s.y_m[perm], establishment_ground_m=s.establishment_ground_m[perm],
                              agb_dry_kg=s.agb_dry_kg[perm], seasonal_agb_dry_kg=s.seasonal_agb_dry_kg[perm],
                              age_days=s.age_days[perm], status=s.status[perm])
        for pid in (1, 3, 6):
            self.assertEqual(s2.x_m[s2.indices_of([pid])[0]], s.x_m[s.indices_of([pid])[0]])


# =============================================================== canopy geometry / MAPLE view / cover
class TestCanopy(unittest.TestCase):
    def setUp(self):
        self.cat = example_catalog()

    def test_geometry_derived_pure_and_constant_porosity(self):
        s = small_state(self.cat)
        snap = {k: getattr(s, k).copy() for k in state.PlantState.ARRAY_NAMES}
        geo = canopy.derive_geometry(s, self.cat)
        i3 = s.indices_of([3])[0]
        self.assertAlmostEqual(geo['diameter_m'][i3], 1.6 * 2.5 ** 0.4)
        self.assertAlmostEqual(geo['height_m'][i3], 1.0 * 2.5 ** 0.35)
        self.assertAlmostEqual(geo['root_radius_m'][i3], 3.0 * 0.5 * 1.6 * 2.5 ** 0.4)
        np.testing.assert_array_equal(geo['porosity'], 0.2)
        self.assertEqual(geo['diameter_m'][s.indices_of([6])[0]], 0.0)  # removed, zero mass
        geo['x_m'][:] = -1.0
        geo['porosity'][:] = 0.9
        for k in state.PlantState.ARRAY_NAMES:
            np.testing.assert_array_equal(getattr(s, k), snap[k], err_msg=k)
        with self.assertRaises(state.StateError):
            canopy.derive_geometry(s, consistent_catalog())  # catalog hash mismatch

    def test_maple_view_fields_ids_coordinates_and_unsupported_cases(self):
        dom = state.Domain(10, 10, 1.0, 1.0, 50.0, -20.0)
        recs = [rec(11, 'grass', 52.0, -18.0, 7.0, 0.1), rec(12, 'shrub', 55.5, -15.5, 7.0, 2.0),
                rec(13, 'grass', 58.0, -12.0, 7.0, 0.05, status='dead_standing'),
                rec(14, 'shrub', 51.0, -11.0, 7.0, 0.0, status='removed')]
        s = state.PlantState.from_records(self.cat, dom, recs)
        v = canopy.maple_vegetation_view(s, self.cat)
        self.assertEqual(list(v['fields']), list(canopy.MAPLE_FIELDS))
        self.assertEqual(v['meta']['n_objects'], 3)  # removed excluded, dead_standing included
        np.testing.assert_array_equal(v['sidecar']['plant_id'], [11, 12, 13])
        np.testing.assert_allclose(v['fields']['center_x_m'], [2.0, 5.5, 8.0])      # domain-local
        np.testing.assert_allclose(v['fields']['center_y_m'], [2.0, 4.5, 8.0])
        np.testing.assert_array_equal(v['fields']['semi_major_m'], v['fields']['semi_minor_m'])
        np.testing.assert_array_equal(v['fields']['orientation_deg'], 0.0)
        np.testing.assert_array_equal(v['fields']['porosity'], 0.2)
        np.testing.assert_array_equal(v['fields']['drag_coefficient'], 1.0)
        self.assertEqual(v['meta']['catalog_hash'], self.cat.hash)
        live_only = canopy.maple_vegetation_view(s, self.cat, statuses=('live',))
        self.assertEqual(live_only['meta']['n_objects'], 2)
        # burial: exposed mode reduces height; erosion: elevated base unsupported
        h_intact = v['fields']['height_m'].copy()
        buried = state.constant_surface(dom, 7.0)
        buried[:, :] = 7.0
        buried[2, 2] = 7.05            # grass 11 at (52,-18) -> row 2, col 2 partially buried
        buried[4, 5] = 6.5             # shrub 12 at (55.5,-15.5) -> row 4, col 5 eroded (elevated base)
        with self.assertRaises(state.StateError):
            canopy.maple_vegetation_view(s, self.cat, current_surface=buried)  # default policy: error
        ex = canopy.maple_vegetation_view(s, self.cat, current_surface=buried, unsupported_policy='exclude')
        self.assertEqual(ex['meta']['n_objects'], 2)
        self.assertNotIn(12, ex['sidecar']['plant_id'].tolist())
        fl = canopy.maple_vegetation_view(s, self.cat, current_surface=buried, unsupported_policy='flag', height_mode='exposed')
        self.assertEqual(fl['meta']['n_objects'], 3)
        self.assertAlmostEqual(fl['fields']['height_m'][0], h_intact[0] - 0.05)
        self.assertTrue(bool(fl['sidecar']['elevated_base'][1]))
        self.assertEqual(fl['meta']['n_unsupported'], 1)
        with self.assertRaises(ValueError):
            canopy.maple_vegetation_view(s, self.cat, height_mode='exposed')  # needs a surface
        fully = state.constant_surface(dom, 7.0)
        fully[2, 2] = 50.0
        with self.assertRaises(state.StateError):
            canopy.maple_vegetation_view(s, self.cat, current_surface=fully, height_mode='exposed')
        unres = small_state(self.cat, resolved=False)
        with self.assertRaises(state.StateError):
            canopy.maple_vegetation_view(unres, self.cat)  # unresolved anchors cannot be published

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
        with self.assertRaises(state.StateError):
            canopy.measure_cover(dom, [11.0], [5.0], [2.0])  # centre outside the domain
        z = canopy.measure_cover(dom, [], [], [], subsamples=2)
        self.assertEqual(z['groups'][0]['union_fraction'], 0.0)
        self.assertEqual(z['total_union_fraction'], 0.0)

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
            self.assertIn('history', sp)
            self.assertAlmostEqual(sp['diameter_sample_m']['mean'], sp['diameter_requested']['mean_m'], delta=0.1)
            self.assertGreater(sp['diameter_sample_m']['std'], 0.0)
        grass = r1['species'][0]
        self.assertGreater(grass['nominal_crown_area_fraction'], grass['measured_union_cover'])  # overlaps counted once
        self.assertEqual(r1['seed'], 11)
        self.assertEqual(r1['catalog_hash'], self.cat.hash)
        self.assertIn('midpoint', r1['cover_measurement']['method'])
        # biomass follows the species inverse allometry of the sampled diameters
        geo = canopy.derive_geometry(s1, self.cat)
        gi = s1.species_index == 0
        self.assertAlmostEqual(float(geo['diameter_m'][gi].mean()), grass['diameter_sample_m']['mean'], places=9)
        # ages: grass constant, shrub uniform
        self.assertTrue(np.all(s1.age_days[gi] == 730.0))
        sh = s1.age_days[~gi]
        self.assertTrue(np.all((sh >= 365.0) & (sh <= 3650.0)) and sh.std() > 0)

    def test_global_nonoverlap_and_surface_resolution(self):
        cfg = self.config(overlap='disallowed', domain={'nx': 16, 'ny': 16, 'dx_m': 1.0},
                          species={'grass': {'target_cover': 0.10, 'diameter_mean_m': 0.3, 'diameter_std_m': 0.05, 'initial_age_days': 10},
                                   'shrub': {'target_cover': 0.08, 'diameter_mean_m': 1.0, 'diameter_std_m': 0.0, 'initial_age_days': 100}})
        dom = state.Domain(16, 16, 1.0)
        surf = state.constant_surface(dom, 5.0)
        s, r = initialization.generate_population(self.cat, cfg, surface=surf)
        self.assertTrue(s.establishment_resolved)
        np.testing.assert_array_equal(s.establishment_ground_m, 5.0)
        geo = canopy.derive_geometry(s, self.cat)
        x, y, rad = geo['x_m'], geo['y_m'], geo['radius_m']
        d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])  # all pairs: tiny population, test only
        rs = rad[:, None] + rad[None, :]
        np.fill_diagonal(d, np.inf)
        self.assertTrue(np.all(d >= rs), 'canopies overlap across species in exclusion mode')
        for sp in r['species']:
            self.assertTrue(sp['accepted'], sp)
            self.assertLessEqual(abs(sp['measured_union_cover'] - sp['target_cover']), 0.01)
        self.assertEqual(r['ordering_convention'], 'random_interleave')
        self.assertGreaterEqual(r['attempts'], len(s))
        self.assertEqual(s.species_of()[0] in ('grass', 'shrub'), True)
        s2, _ = initialization.generate_population(self.cat, cfg, surface=surf)
        np.testing.assert_array_equal(s.x_m, s2.x_m)

    def test_bounded_impossible_and_degenerate_cases(self):
        E = initialization.GenerationError
        with self.assertRaises(E):  # exclusion request far above packing limits -> attempts exhausted, no output
            initialization.generate_population(self.cat, self.config(
                overlap='disallowed', domain={'nx': 10, 'ny': 10, 'dx_m': 1.0}, limits={'max_attempts': 3000, 'max_plants': 100000},
                species={'shrub': {'target_cover': 0.85, 'diameter_mean_m': 1.0, 'diameter_std_m': 0.0, 'initial_age_days': 1}}))
        with self.assertRaises(E):  # max_plants checked before sampling
            initialization.generate_population(self.cat, self.config(limits={'max_plants': 10}))
        with self.assertRaises(E):  # raster bound checked before allocation
            initialization.generate_population(self.cat, self.config(limits={'max_cover_cells': 100}))
        with self.assertRaises(E):  # tolerance unattainable in zero iterations
            initialization.generate_population(self.cat, self.config(max_iterations=0, cover_tolerance=1e-6))
        with self.assertRaises(traits.CatalogError):
            initialization.generate_population(self.cat, self.config(species={'BOER': {'target_cover': 0.1, 'diameter_mean_m': 1.0,
                                                                                        'diameter_std_m': 0.0, 'initial_age_days': 1}}))
        with self.assertRaises(traits.CatalogError):
            initialization.generate_population(self.cat, self.config(bogus=1))
        with self.assertRaises(traits.CatalogError):
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


# =============================================================== recording cadence
class TestRecording(unittest.TestCase):
    def setUp(self):
        self.cat = example_catalog()

    def test_cadence_initial_final_skip_overwrite_and_errors(self):
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
            s.advance_time(25.0)                 # skips the due times 20 and 30
            p2 = w.maybe_write(s)
            self.assertEqual(w.records[-1]['skipped_due_times_days'], [20.0, 30.0])
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
            mp = w.write_manifest(os.path.join(td, 'manifest.json'))
            self.assertTrue(os.path.exists(mp))
            # overwrite refusal: a second writer on the same template must not clobber or advance
            w2 = recording.SnapshotWriter(tpl, cadence_days=10.0, start_time_days=0.0)
            with self.assertRaises(FileExistsError):
                w2.write_initial(s)
            self.assertEqual(w2.records, [])
            s.time_days = 10.0
            self.assertTrue(w2.due(10.0))
            with self.assertRaises(FileExistsError):
                w2.maybe_write(s)
            self.assertEqual(w2.next_due_days, 10.0)  # bookkeeping untouched after the failed write
            with open(p1, 'r', encoding='utf-8') as fh:
                self.assertIn('"time_days": 10.0', fh.readline())
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


# =============================================================== events
class TestEvents(unittest.TestCase):
    def setUp(self):
        self.cat = example_catalog()

    def test_construction_validation_and_dict_round_trip(self):
        E = events.EventError
        ev = events.SurfaceEventDiagnostics([1, 3], 0.0, 2.0, 'maple_wind', erosion_depth_m=[0.01, 0.0],
                                            deposition_depth_m=[0.0, 0.3], abrasion_dose=[5.0, 0.0], abrasion_unit='kg m^-1')
        np.testing.assert_allclose(ev.net_surface_delta_m, [-0.01, 0.3])
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
        d = ev.to_dict()
        d['schema_version'] = 7
        with self.assertRaises(E):
            events.SurfaceEventDiagnostics.from_dict(d)
        d = ev.to_dict()
        d['depth_unit'] = 'cm'
        with self.assertRaises(E):
            events.SurfaceEventDiagnostics.from_dict(d)
        d = ev.to_dict()
        d['extra'] = 1
        with self.assertRaises(E):
            events.SurfaceEventDiagnostics.from_dict(d)

    def test_apply_defaults_to_diagnostics_and_rolls_back_failed_handlers(self):
        s = small_state(self.cat)
        snap = {k: getattr(s, k).copy() for k in state.PlantState.ARRAY_NAMES}

        def unchanged():
            for k in state.PlantState.ARRAY_NAMES:
                np.testing.assert_array_equal(getattr(s, k), snap[k], err_msg=k)

        ev = events.SurfaceEventDiagnostics([1, 4], 0.0, 2.0, 'syrup_water', deposition_depth_m=[0.5, 0.1])
        with self.assertRaises(events.EventError):
            events.apply_event_diagnostics(s, ev)  # population clock (0) has not reached the event end (2)
        s.advance_time(2.0)
        rep = events.apply_event_diagnostics(s, ev)
        self.assertEqual(rep['n_plants'], 2)
        self.assertTrue(rep['response'].startswith('none'))
        np.testing.assert_array_equal(s.ids[rep['indices']], [1, 4])
        snap['age_days'] = s.age_days.copy()
        unchanged()
        bad_ids = events.SurfaceEventDiagnostics([1, 999], 0.0, 1.0, 'maple_wind')
        with self.assertRaises(events.EventError):
            events.apply_event_diagnostics(s, bad_ids)
        unchanged()

        def boom(view, event, idx):
            raise RuntimeError('species law exploded')

        with self.assertRaises(events.EventError):
            events.apply_event_diagnostics(s, ev, handler=boom)
        unchanged()

        def invalid(view, event, idx):
            return {'agb_dry_kg': np.zeros(len(idx))}  # live plants with zero mass -> rejected

        with self.assertRaises(events.EventError):
            events.apply_event_diagnostics(s, ev, handler=invalid)
        unchanged()

        def mutate_view(view, event, idx):
            view.agb_dry_kg[:] = 0.0  # a handler cannot reach the real arrays
            return None

        rep = events.apply_event_diagnostics(s, ev, handler=mutate_view)
        unchanged()

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
    def __init__(self, ids, B, pid, names, E=None, day=3):
        self.host_inputs = {'ids': np.asarray(ids, dtype=np.int64)}
        self.B = np.asarray(B, dtype=np.float64)
        self.host_profile_ids = np.asarray(pid, dtype=np.int32)
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

    def test_capture_by_id_with_reordering_profiles_and_ages(self):
        names = ['grass', 'shrub']
        sim = FakeSim([30, 10, 20], [3.0, 0.5, 0.7], [1, 0, 0], names, E=[0.25, 0.0, 0.1])
        out = runtime.capture_snapshot(self.pop, types.SimpleNamespace(vegetation=sim))  # CoupledSimulation-like
        self.assertIs(out, self.pop)
        self.assertEqual(self.pop.agb_dry_kg[self.pop.indices_of([30])[0]], 3.0)
        self.assertEqual(self.pop.agb_dry_kg[self.pop.indices_of([10])[0]], 0.5)
        self.assertEqual(self.pop.seasonal_agb_dry_kg[self.pop.indices_of([20])[0]], 0.1)
        self.assertEqual(self.pop.age_days[self.pop.indices_of([10])[0]], 100.0)  # ages untouched by capture
        self.assertEqual(self.pop.agb_dry_kg[self.pop.indices_of([40])[0]], 0.0)   # removed identity retained
        self.assertEqual(self.pop.provenance['bridge']['captured_from'], 'CoupledSimulation')
        snap = {k: getattr(self.pop, k).copy() for k in state.PlantState.ARRAY_NAMES}
        B = runtime.BridgeError
        cases = [FakeSim([30, 10], [3.0, 0.5], [1, 0], names),                        # live plant 20 missing
                 FakeSim([30, 10, 20, 99], [3.0, 0.5, 0.7, 1.0], [1, 0, 0, 0], names),  # unknown id
                 FakeSim([30, 10, 20, 40], [3.0, 0.5, 0.7, 1.0], [1, 0, 0, 1], names),  # removed plant in driver
                 FakeSim([30, 10, 20], [3.0, 0.5, 0.7], [0, 0, 0], names),             # profile mismatch for id 30
                 FakeSim([30, 10, 20], [3.0, 0.5, 0.7], [1, 0, 0], names, E=[4.0, 0.0, 0.0]),  # E > B
                 FakeSim([30, 10, 20], [3.0, -0.5, 0.7], [1, 0, 0], names),
                 FakeSim([30, 10, 10], [3.0, 0.5, 0.7], [1, 0, 0], names)]              # duplicate driver ids
        for sim in cases:
            with self.assertRaises(B):
                runtime.capture_snapshot(self.pop, sim)
            for k in state.PlantState.ARRAY_NAMES:
                np.testing.assert_array_equal(getattr(self.pop, k), snap[k], err_msg=k)
        copy_out = runtime.capture_snapshot(self.pop, FakeSim([30, 10, 20], [1.0, 1.0, 1.0], [1, 0, 0], names), in_place=False)
        self.assertEqual(self.pop.agb_dry_kg[self.pop.indices_of([30])[0]], 3.0)
        self.assertEqual(copy_out.agb_dry_kg[copy_out.indices_of([30])[0]], 1.0)

    def test_legacy_inputs_refuse_silent_allometry_mismatch(self):
        B = runtime.BridgeError
        with self.assertRaises(B):  # two live species with different lateral multipliers (legacy has one)
            runtime.legacy_inputs_from_state(self.pop, self.cat)
        self.pop.update_plants([30], status='removed', agb_dry_kg=[0.0])
        with self.assertRaises(B):  # power allometry != pi r^2 * 1 kg/m^2 -> refused by default
            runtime.legacy_inputs_from_state(self.pop, self.cat)
        res = runtime.legacy_inputs_from_state(self.pop, self.cat, biomass_policy='accept_driver_recompute')
        self.assertFalse(res['biomass_check']['consistent'])
        self.assertEqual(sorted(res['excluded_ids']), [30, 40])
        np.testing.assert_array_equal(res['inputs']['ids'], [10, 20])
        np.testing.assert_array_equal(res['inputs']['root_profile_id'], [0, 0])
        cat2 = consistent_catalog()
        pop2 = state.PlantState.from_records(cat2, self.dom, [rec(1, 'grass', 1.0, 1.0, 0.0, 0.3), rec(2, 'shrub', 5.0, 5.0, 0.0, 2.0)])
        res2 = runtime.legacy_inputs_from_state(pop2, cat2)
        self.assertTrue(res2['biomass_check']['consistent'])
        np.testing.assert_allclose(math.pi * res2['inputs']['canopy_radius'] ** 2, [0.3, 2.0], rtol=1e-12)
        np.testing.assert_allclose(res2['inputs']['root_radius'], 2.0 * res2['inputs']['canopy_radius'])
        self.assertEqual(res2['root_layer_fractions'], {'grass': [0.8, 0.2], 'shrub': [0.25, 0.75]})
        off = state.Domain(8, 8, 1.0, 1.0, 5.0, 0.0)
        pop3 = state.PlantState.from_records(cat2, off, [rec(1, 'grass', 6.0, 1.0, 0.0, 0.3)])
        with self.assertRaises(B):
            runtime.legacy_inputs_from_state(pop3, cat2)

    def test_capture_from_real_legacy_driver(self):
        try:
            from benchmarks import daily_water_growth as dwg
        except Exception as exc:  # numba / benchmarks unavailable in this interpreter
            self.skipTest(f'legacy driver unavailable: {exc!r}')
        cat2 = consistent_catalog()
        pop = state.PlantState.from_records(cat2, self.dom, [rec(7, 'grass', 1.5, 1.5, 0.0, 0.3), rec(3, 'shrub', 5.5, 5.5, 0.0, 2.0),
                                                            rec(5, 'grass', 2.5, 6.5, 0.0, 0.1)])
        res = runtime.legacy_inputs_from_state(pop, cat2)
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
            self.assertEqual(s.establishment_ground_m[s.indices_of([3])[0]], 100.25)
            self.assertEqual(s.counts(), {'live': 4, 'dead_standing': 1, 'removed': 1})
            code, _, err = run_cli(['init', '--species', str(SPECIES_JSON), '--plants', str(PLANTS_CSV), '--nx', '20', '--ny', '20',
                                    '--dx-m', '1.0', '--surface-constant', '100.0', '--out', out])
            self.assertEqual(code, 1)  # exists, no --overwrite
            self.assertIn('exists', err)
            # without a surface the SURFACE token is preserved in an INITIAL file
            raw = os.path.join(td, 'raw.csv')
            code, _, err = run_cli(['init', '--species', str(SPECIES_JSON), '--plants', str(PLANTS_CSV), '--nx', '20', '--ny', '20',
                                    '--dx-m', '1.0', '--out', raw])
            self.assertEqual(code, 0, err)
            with open(raw, 'r', encoding='utf-8') as fh:
                self.assertIn('SURFACE', fh.read())
            code, _, err = run_cli(['export-canopy', '--species', str(SPECIES_JSON), '--plants', raw, '--out', os.path.join(td, 'v.json')])
            self.assertEqual(code, 1)  # unresolved anchors cannot be published
            # inspect with a surface and a cover measurement
            surf = os.path.join(td, 'surface.npy')
            np.save(surf, np.full((20, 20), 100.5))
            code, text, err = run_cli(['inspect', '--species', str(SPECIES_JSON), '--plants', out, '--surface', surf, '--cover-subsamples', '8'])
            self.assertEqual(code, 0, err)
            info = json.loads(text)
            self.assertAlmostEqual(info['burial_exposure']['burial_depth_m']['max'], 0.5)
            self.assertIn('grass', info['cover']['per_species'])
            # transposed surface refused
            np.save(surf, np.full((20, 21), 100.5))
            code, _, err = run_cli(['inspect', '--species', str(SPECIES_JSON), '--plants', out, '--surface', surf])
            self.assertEqual(code, 1)
            # canopy export JSON + NPZ
            vj = os.path.join(td, 'view.json')
            code, _, err = run_cli(['export-canopy', '--species', str(SPECIES_JSON), '--plants', out, '--out', vj])
            self.assertEqual(code, 0, err)
            with open(vj, 'r', encoding='utf-8') as fh:
                view = json.load(fh)
            self.assertEqual(view['meta']['n_objects'], 5)
            self.assertEqual(view['sidecar']['plant_id'], [1, 2, 3, 4, 5])
            self.assertEqual(len(view['fields']['height_m']), 5)
            vz = os.path.join(td, 'view.npz')
            code, _, err = run_cli(['export-canopy', '--species', str(SPECIES_JSON), '--plants', out, '--out', vz, '--statuses', 'live'])
            self.assertEqual(code, 0, err)
            with np.load(vz, allow_pickle=False) as z:
                self.assertEqual(z['center_x_m'].shape, (4,))
                self.assertEqual(json.loads(str(z['meta']))['n_objects'], 4)
                self.assertEqual(z['sidecar_plant_id'].tolist(), [1, 2, 3, 4])
            code, _, _ = run_cli(['export-canopy', '--species', str(SPECIES_JSON), '--plants', out, '--out', vz])
            self.assertEqual(code, 1)  # no overwrite
            code, _, _ = run_cli(['inspect', '--species', str(SPECIES_JSON), '--plants', os.path.join(td, 'missing.csv')])
            self.assertEqual(code, 1)

    def test_generate_example_config(self):
        cat = example_catalog()
        with tempfile.TemporaryDirectory() as td:
            out, rep = os.path.join(td, 'gen.csv'), os.path.join(td, 'gen_report.json')
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
            self.assertEqual(report['domain']['nx'], 64)
            for sp in report['species']:
                self.assertLessEqual(abs(sp['measured_union_cover'] - sp['target_cover']), 0.01)
            view = canopy.maple_vegetation_view(s, cat)
            self.assertEqual(view['meta']['n_objects'], len(s))

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

    def test_legacy_lazy_exports(self):
        import leaf
        self.assertEqual(leaf.__version__, '0.1.0')
        self.assertIn('CoupledSimulation', dir(leaf))
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
