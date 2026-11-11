'''LEAF Phase 1 part A: root-map builder tests + co-located multi-file species loading (unittest, NumPy only).

Run from the candidate root:  python -m unittest tests.test_roots -v
The compiled-builder parity test SKIPS only when numba is unavailable. The multi-file catalog tests live
here because this part's allowed test files are test_uptake / test_roots / test_forcing.
'''
from __future__ import annotations

import json
import math
import os
import pathlib
import tempfile
import unittest

import numpy as np

from leaf import canopy, roots, state, traits, uptake

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPECIES_JSON = ROOT / 'examples' / 'phase0' / 'species.json'
BIG = 1 << 26


def rec(pid, sp, x, y, B, status='live'):
    return {'plant_id': pid, 'species': sp, 'x_m': x, 'y_m': y, 'establishment_ground_m': 0.0, 'agb_dry_kg': B,
            'seasonal_agb_dry_kg': 0.0, 'age_days': 100.0, 'status': status}


def example_catalog():
    return traits.SpeciesCatalog.from_file(SPECIES_JSON)


def small_state(cat, dom=None):
    dom = state.Domain(12, 10, 1.0, origin_x_m=100.0, origin_y_m=-50.0) if dom is None else dom
    recs = [rec(11, 'grass', 102.5, -46.5, 0.12),
            rec(12, 'shrub', 108.0, -44.0, 2.5),
            rec(13, 'grass', 111.9, -40.1, 0.4),                       # near the north-east edge: clipped
            rec(14, 'grass', 104.0, -42.0, 0.05, 'dead_standing'),
            rec(15, 'shrub', 101.0, -41.0, 0.0, 'removed')]
    return state.PlantState.from_records(cat, dom, recs)


def brute_force_cells(dom, px, py, r):
    '''All cells whose closed square positively overlaps the circle (closest point strictly inside), row-major.'''
    out = []
    for j in range(dom.ny):
        for i in range(dom.nx):
            cx = min(max(px, i * dom.dx_m), (i + 1) * dom.dx_m)
            cy = min(max(py, j * dom.dx_m), (j + 1) * dom.dx_m)
            if (px - cx) ** 2 + (py - cy) ** 2 < r * r:
                out.append(j * dom.nx + i)
    return out


def expected_rows(st, cat, geo, statuses=('live',), overrides=None):
    dom = st.domain
    codes = [state.STATUS_CODES[s] for s in statuses]
    rows, ids = [], []
    table = cat.root_layer_table()
    for k in range(st.n):
        if int(st.status[k]) not in codes:
            continue
        pid = int(st.ids[k])
        frac = table[int(st.species_index[k])] if not overrides or pid not in overrides else np.asarray(overrides[pid])
        r = float(geo['root_radius_m'][k])
        cells = brute_force_cells(dom, float(st.x_m[k]) - dom.origin_x_m, float(st.y_m[k]) - dom.origin_y_m, r) if r > 0 else []
        row = []
        for l in range(cat.root_layers):
            if frac[l] > 0.0:
                for c in cells:
                    row.append((l * dom.ncells + c, frac[l] * (1.0 / len(cells))))
        rows.append(row)
        ids.append(pid)
    return ids, rows


def assert_map_equals(tc, m, ids, rows):
    h = m.host_arrays()
    np.testing.assert_array_equal(h['ids'], ids)
    tc.assertEqual(m.n_plants, len(rows))
    for p, row in enumerate(rows):
        o0, o1 = int(h['offsets'][p]), int(h['offsets'][p + 1])
        np.testing.assert_array_equal(h['resources'][o0:o1], [r for r, _ in row], err_msg=f'plant row {p}')
        np.testing.assert_allclose(h['weights'][o0:o1], [w for _, w in row], rtol=1e-15, atol=0, err_msg=f'plant row {p}')


class TestRootBuilder(unittest.TestCase):
    def setUp(self):
        self.cat = example_catalog()
        self.st = small_state(self.cat)
        self.geo = canopy.derive_geometry(self.st, self.cat)

    def test_numpy_builder_matches_brute_force_geometry_and_labels_the_law(self):
        m = roots.build_root_map(self.st, self.cat, max_edges=100_000, max_bytes=BIG, builder='numpy')
        ids, rows = expected_rows(self.st, self.cat, self.geo)
        assert_map_equals(self, m, ids, rows)
        self.assertEqual(m.n_plants, 3)                                   # live only by default
        self.assertEqual(m.provenance['horizontal_law'], roots.HORIZONTAL_LAW)
        self.assertIn('NOT measured root density', m.provenance['horizontal_law_label'])
        self.assertEqual(m.provenance['statuses'], ['live'])
        self.assertEqual(m.provenance['catalog_hash'], self.cat.hash)
        self.assertEqual(m.provenance['admission']['horizontal_builder']['builder'], 'numpy')
        # the edge plant is clipped to the finite domain and still sums to one (normalised at build time)
        h = m.host_arrays()
        for p in range(m.n_plants):
            self.assertAlmostEqual(float(h['weights'][h['offsets'][p]:h['offsets'][p + 1]].sum()), 1.0, places=12)
        k13 = ids.index(13)
        self.assertLess(len(rows[k13]), 2 * len(brute_force_cells(state.Domain(40, 40, 1.0), 20.0, 20.0,
                                                                     float(self.geo['root_radius_m'][self.st.indices_of([13])[0]]))))
        # the rows follow the state row order and the map is consumable by the uptake allocator
        D = np.array([1.0, 2.0, 3.0])
        plan = uptake.propose(m, D, np.full((2, self.st.domain.ncells), 10.0))
        fb = plan.accept_all()
        np.testing.assert_allclose(fb.satisfaction, 1.0, rtol=1e-12)
        self.assertTrue(fb.closure['closed'])

    def test_species_fractions_and_individual_overrides(self):
        ov = {11: [1.0, 0.0], 12: [0.1, 0.9]}
        m = roots.build_root_map(self.st, self.cat, max_edges=100_000, max_bytes=BIG, builder='numpy',
                                 layer_fraction_overrides=ov)
        ids, rows = expected_rows(self.st, self.cat, self.geo, overrides=ov)
        assert_map_equals(self, m, ids, rows)
        self.assertEqual(m.provenance['n_overrides'], 2)
        h = m.host_arrays()
        p11 = ids.index(11)
        self.assertTrue(np.all(h['resources'][h['offsets'][p11]:h['offsets'][p11 + 1]] < self.st.domain.ncells))  # layer 0 only
        R = roots.RootsError
        for bad in ({11: [0.5, 0.4]}, {11: [1.0]}, {11: [1.0, 0.0, 0.0]}, {11: [1.2, -0.2]}, {11: [np.nan, 1.0]},
                    {11: 'ab'}, {True: [1.0, 0.0]}, {11.0: [1.0, 0.0]}, {99: [1.0, 0.0]}, {14: [1.0, 0.0]}, {}, [1.0, 0.0]):
            with self.assertRaises(R, msg=repr(bad)):
                roots.build_root_map(self.st, self.cat, max_edges=100_000, max_bytes=BIG, builder='numpy',
                                     layer_fraction_overrides=bad)

    def test_statuses_inactive_rows_and_empty_selection(self):
        m = roots.build_root_map(self.st, self.cat, max_edges=100_000, max_bytes=BIG, builder='numpy',
                                 statuses=('live', 'dead_standing', 'removed'))
        ids, rows = expected_rows(self.st, self.cat, self.geo, statuses=('live', 'dead_standing', 'removed'))
        assert_map_equals(self, m, ids, rows)
        self.assertEqual(m.n_plants, 5)
        self.assertEqual(m.n_inactive_plants, 1)                            # removed plant with B = 0: no roots
        self.assertEqual(m.provenance['n_with_roots'], 4)
        D = np.zeros(5)
        D[list(ids).index(15)] = 1.0
        with self.assertRaises(uptake.UptakeError):                         # positive demand on the inactive row
            uptake.propose(m, D, np.ones((2, self.st.domain.ncells)))
        D[list(ids).index(15)] = 0.0
        uptake.propose(m, D, np.ones((2, self.st.domain.ncells)))
        empty = state.PlantState.empty(self.cat, self.st.domain)
        me = roots.build_root_map(empty, self.cat, max_edges=10, max_bytes=BIG, builder='numpy')
        self.assertEqual((me.n_plants, me.n_edges), (0, 0))
        with self.assertRaises(state.StateError):
            roots.build_root_map(self.st, self.cat, max_edges=10, max_bytes=BIG, builder='numpy', statuses=('zombie',))

    def test_nonsquare_domain_refused_before_allocation(self):
        dom = state.Domain(12, 10, 1.0, 2.0)
        st = state.PlantState.from_records(self.cat, dom, [rec(1, 'grass', 2.5, 3.5, 0.12)])
        with self.assertRaises(roots.RootsError) as cm:
            roots.build_root_map(st, self.cat, max_edges=100, max_bytes=BIG, builder='numpy')
        self.assertIn('non-square', str(cm.exception))
        with self.assertRaises(roots.RootsError):
            roots.build_root_map(self.st, self.cat, max_edges=100, max_bytes=BIG, builder='gpu')
        with self.assertRaises(roots.RootsError):
            roots.build_root_map(self.st, self.cat, max_edges=100, max_bytes=BIG, builder='numpy', backend='metal')

    def test_expanded_layer_edges_admitted_before_allocation(self):
        full = roots.build_root_map(self.st, self.cat, max_edges=100_000, max_bytes=BIG, builder='numpy')
        E_h, E = full.provenance['n_horizontal_edges'], full.n_edges
        self.assertGreater(E, E_h)                                           # two positive layers per species
        with self.assertRaises(uptake.AdmissionError) as cm:
            roots.build_root_map(self.st, self.cat, max_edges=E - 1, max_bytes=BIG, builder='numpy')
        self.assertEqual(cm.exception.stage, 'roots.expand_layers')
        self.assertEqual(cm.exception.estimate['n_edges'], E)
        with self.assertRaises(uptake.AdmissionError) as cm:
            roots.build_root_map(self.st, self.cat, max_edges=E_h - 1, max_bytes=BIG, builder='numpy')
        self.assertEqual(cm.exception.stage, 'roots.horizontal_numpy')
        with self.assertRaises(uptake.AdmissionError) as cm:
            roots.build_root_map(self.st, self.cat, max_edges=100_000, max_bytes=16, builder='numpy')
        self.assertEqual(cm.exception.stage, 'roots.preflight')
        # the population is never truncated to pass: the full map keeps every selected plant
        self.assertEqual(full.n_plants, 3)

    def test_masked_cell_in_uptake_keeps_weights_fixed(self):
        m = roots.build_root_map(self.st, self.cat, max_edges=100_000, max_bytes=BIG, builder='numpy')
        h = m.host_arrays()
        ids = list(h['ids'])
        p = ids.index(11)
        row_res = h['resources'][h['offsets'][p]:h['offsets'][p + 1]]
        row_w = h['weights'][h['offsets'][p]:h['offsets'][p + 1]]
        A = np.full(m.n_resources, 1e6)
        masked = int(row_res[0])                                              # make one of plant 11's resources unavailable
        A[masked] = 0.0
        D = np.zeros(3)
        D[p] = 10.0
        fb = uptake.propose(m, D, A).accept_all()
        self.assertAlmostEqual(float(fb.satisfaction[p]), 1.0 - float(row_w[0]), places=12)
        self.assertAlmostEqual(float(fb.unmet[p]), 10.0 * float(row_w[0]), places=12)
        np.testing.assert_array_equal(m.host_arrays()['weights'], h['weights'])  # nothing renormalised

    def test_custom_general_map_can_replace_the_builder(self):
        N = self.st.domain.ncells
        rows = [[(0, 0.3), (5, 0.3), (N + 7, 0.4)], [(N + 7, 1.0)], [(3, 0.5), (N + 3, 0.25), (N + 4, 0.25)]]
        ids = [11, 12, 13]
        off = np.cumsum([0] + [len(r) for r in rows])
        m = uptake.RootMap.from_arrays(np.array(ids), off, np.array([r for row in rows for r, _ in row]),
                                       np.array([w for row in rows for _, w in row]), n_layers=2, n_cells=N,
                                       backend='numpy', max_edges=100, max_bytes=BIG)
        fb = uptake.propose(m, np.array([1.0, 1.0, 1.0]), np.full((2, N), 0.25)).accept_all()
        self.assertTrue(fb.closure['closed'])
        self.assertLess(float(fb.satisfaction[1]), 1.0)

    def test_compiled_builder_parity(self):
        st = uptake.backend_status('numba')
        if not st['available']:
            self.skipTest(f'compiled builder unavailable: {st["reason"]}')
        a = roots.build_root_map(self.st, self.cat, max_edges=100_000, max_bytes=BIG, builder='numpy',
                                 statuses=('live', 'dead_standing'))
        b = roots.build_root_map(self.st, self.cat, max_edges=100_000, max_bytes=BIG, builder='compiled',
                                 statuses=('live', 'dead_standing'))
        ha, hb = a.host_arrays(), b.host_arrays()
        for k in ('ids', 'offsets', 'resources'):
            np.testing.assert_array_equal(ha[k], hb[k], err_msg=k)
        np.testing.assert_allclose(ha['weights'], hb['weights'], rtol=1e-15, atol=0)
        self.assertEqual(b.provenance['admission']['horizontal_builder']['builder'], 'compiled')
        with self.assertRaises(uptake.AdmissionError):
            roots.build_root_map(self.st, self.cat, max_edges=1, max_bytes=BIG, builder='compiled')


class TestMultiFileCatalog(unittest.TestCase):
    def setUp(self):
        self.cat = example_catalog()
        self.doc = self.cat.document
        self.td = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def write(self, name, doc):
        p = self.dir / name
        p.write_text(json.dumps(doc, indent=1), encoding='utf-8')
        return str(p)

    def split(self, mutate=None):
        g, s = dict(self.doc), dict(self.doc)
        g['species'] = {'grass': self.doc['species']['grass']}
        s['species'] = {'shrub': self.doc['species']['shrub']}
        if mutate:
            mutate(g, s)
        return self.write('grass.json', g), self.write('shrub.json', s)

    def test_deterministic_merge_equals_single_file_identity(self):
        g, s = self.split()
        a = traits.SpeciesCatalog.from_files([g, s])
        b = traits.SpeciesCatalog.from_files([s, g])
        self.assertEqual(a.hash, b.hash)
        self.assertEqual(a.hash, self.cat.hash)                              # same complete content -> same identity
        self.assertEqual(a.species_ids, ('grass', 'shrub'))
        self.assertEqual(b.species_ids, ('grass', 'shrub'))
        self.assertEqual(a.canonical_text(), self.cat.canonical_text())
        self.assertEqual(a.sources, (g, s))
        self.assertEqual(b.sources, (s, g))                                  # provenance keeps the given order ...
        self.assertEqual(a.species_sources, {'grass': g, 'shrub': s})
        self.assertIsNone(a.source)
        self.assertEqual(a.metadata()['sources'], [g, s])
        self.assertEqual(a.root_layer_fractions('shrub'), [0.25, 0.75])
        self.assertEqual(a.law('grass', 'growth'), 'unimplemented')          # per-species content preserved
        # one file through from_files is the same catalog as from_file
        one = self.write('one.json', self.doc)
        c1, c2 = traits.SpeciesCatalog.from_files([one]), traits.SpeciesCatalog.from_file(one)
        self.assertEqual(c1.hash, c2.hash)
        self.assertEqual(c1.source, c2.source)
        self.assertEqual(c1.sources, c2.sources)
        self.assertEqual(c1.species_sources, c2.species_sources)
        self.assertEqual(traits.load_catalog([g, s]).hash, a.hash)
        self.assertEqual(traits.load_catalog(pathlib.Path(one)).hash, c2.hash)
        # a state built against the merged catalog is compatible with the original
        st = small_state(a)
        self.assertEqual(st.catalog_hash, self.cat.hash)
        canopy.derive_geometry(st, self.cat)
        # document ownership survives merging
        pub = a.document
        pub['species']['grass']['roots']['layer_fractions'] = [0.1, 0.9]
        self.assertEqual(a.root_layer_fractions('grass'), [0.8, 0.2])

    def test_duplicates_incomplete_records_and_mismatches_refused(self):
        E = traits.CatalogError
        g, s = self.split()
        g2 = self.write('grass_copy.json', traits.parse_strict_json(pathlib.Path(g).read_text(encoding='utf-8')))
        with self.assertRaises(E):
            traits.SpeciesCatalog.from_files([g, s, g2])                    # identical duplicate species refused
        with self.assertRaises(E):
            traits.SpeciesCatalog.from_files([g, g])                        # same file twice
        with self.assertRaises(E):
            traits.SpeciesCatalog.from_files(g)                             # a single string is not a sequence
        with self.assertRaises(E):
            traits.SpeciesCatalog.from_files([])
        with self.assertRaises(E):
            traits.SpeciesCatalog.from_files([g, 7])

        def more_layers(gd, sd):
            sd['root_layers'] = 3
            sd['species']['shrub']['roots']['layer_fractions'] = [0.25, 0.5, 0.25]
        g3, s3 = self.split(more_layers)
        with self.assertRaises(E):                                           # root_layers must match across files
            traits.SpeciesCatalog.from_files([g3, s3])

        def incomplete(gd, sd):
            del sd['species']['shrub']['mortality']
        g4, s4 = self.split(incomplete)
        with self.assertRaises(E):                                           # every record must be complete in its file
            traits.SpeciesCatalog.from_files([g4, s4])

        def other_units(gd, sd):
            sd['units'] = {'biomass': 'g', 'length': 'm', 'time': 'days'}
        g5, s5 = self.split(other_units)
        with self.assertRaises(E):
            traits.SpeciesCatalog.from_files([g5, s5])

    def test_catalog_id_and_notes_kept_only_when_identical(self):
        def other_id(gd, sd):
            sd['catalog_id'] = 'shrub-file'
        g, s = self.split(other_id)
        a = traits.SpeciesCatalog.from_files([g, s])
        b = traits.SpeciesCatalog.from_files([s, g])
        self.assertEqual(a.hash, b.hash)                                     # still order independent ...
        self.assertNotEqual(a.hash, self.cat.hash)                           # ... but no longer the fixture identity
        self.assertNotIn('catalog_id', a.document)
        self.assertEqual(a.document['notes'], self.doc['notes'])
        self.assertEqual(a.species_record('shrub'), self.cat.species_record('shrub'))


if __name__ == '__main__':
    unittest.main()
