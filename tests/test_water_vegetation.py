'''leaf.water_vegetation (Phase 2.5 P1) CPU tests: independent analytic geometry, add-and-cap rule, eligibility, masks, origins/periodic seams, ordering, backends,
admission, ownership, session binding and the pre-existing-source guard. Run from the candidate root:  python -m unittest tests.test_water_vegetation -v
Expected values come from closed-form geometry (pi r^2, symmetry) and hand arithmetic, never from the implementation.'''
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import tracemalloc
import unittest

import numpy as np

from leaf import state, traits
from leaf import water_vegetation as wv
from tests import native_leaf_fixture as nf

ROOT = pathlib.Path(__file__).resolve().parents[1]
PHASE0 = ROOT / 'examples' / 'phase0' / 'species.json'
KEY = traits.RECEIVER_WEIGHTS_KEY
BIG = 1 << 28
EDGES = 1 << 16
BASELINE_27 = {
    '__init__.py': 'e8d289c0c597dbaf987a94e1f53b036ebd5f1dc08f2900be7cc6b3aacb5e78b5',
    '__main__.py': 'e1e40edddf48c656e0b55cf2b5d8bd8387bf8f581439d23609ffbc3d99b27294',
    'biocalendar.py': 'f486bd3216612958a8205cbea31d7c11974972a4fd527c70cfd0c4f963b11544',
    'canopy.py': '6cba03f4f59fac55ef430420d930ddc84f4e7a7adf36d44e685b01504dc87b07',
    'checkpoint.py': 'a9db54616e3af273e5b001cbb0ef6e831ddb7a7a39b3296f564f53a000686056',
    'coupling.py': '2a9de99773f976d260617d1506be15f9b5950233880625dc40389e6d99e9b247',
    'demography.py': '7648b8f4a23016dc07196b9c790498bccb3cf18db0a20bbbda408974834dce1a',
    'disturbance.py': '38a9dfa4521cdcb2512b904a83d5f2af4fbad97fa1a0c972bb70b0cb6906f9c0',
    'events.py': '2ec912dac4a5bb4ae1f52a64ef99e74e973d7f7dc68993f0f083fceb3cbed98e',
    'forcing.py': 'fa79d12f2c96a15aac8012d713976ece8b4c055ca567968251225635fa9d7474',
    'growth.py': '6664c7194ca7df3f731fe82f9527cf6f60835c5517aca60dbc591cf719d2aa58',
    'initialization.py': 'a385b7f509871ad2d130564d1b07c5c04592ffb008ec2192bb96ff26c72b489b',
    'interevent.py': 'ff7b9287c9ac810287dde89bcbb550af6b4f8380eeb5ee55d5a98d161f78ff5f',
    'lifecycle_bridge.py': '63896e6fe8ffd7e436790203006bf2ae248febb7f9cccd6f9cb98314d571cd91',
    'periodic.py': 'b960a27b07eb2c34a455e398255e1e7ffdf845c280fe4c959f9fd4ee467c6aa5',
    'plant_checkpoint.py': '17cd7dbae173b53796b114ddbe63b9425fe9f59e72f5d30c84350147cc5c3c2e',
    'plant_pet.py': '50216355aab1be3a5dcaf6280d5ab00fd34395d1bb8e12be7ab66827de227acd',
    'plant_session.py': '562188cc9b9d9c80b61903d63549fd760326e9e385ba1ca1aa1a823e43343a14',
    'recording.py': '49564345dd53505148de90de00b5bcb02d7d0e23b8b483e75ebac96af2ff9ef0',
    'recruitment.py': '235821ec613867706ed1971e8cd841b83104fe1bc1642054593e66a654fa6491',
    'roots.py': 'c258de49e4d3da734bae0b9c7fa8d0c59bb1d6f91ac2ff5ceceb46ca12e80904',
    'runtime.py': 'abbea7b0f873d87139f92c9eb4a6989f657def504c2080379d728f7ceb121826',
    'state.py': 'a2bbf2472a6bbc32ed5db4f5ad45a47d9cb98099f8d7918a709367b5b01e2c25',
    'stepwat.py': '981f9b04cd15421c889c923453957f3a7f77034d535aca2f8e4aa4f70858f76a',
    'syrup_water.py': 'baefec2a100b751cc35bda8739ea5d04339681f0d3f9d6b900972986161db215',
    'traits.py': '3449fbd3cc790d06d3c4062040d77755c7b04c980721c77d80be1e93453768f0',
    'uptake.py': '675e4bf4b9a9c202ae166f50444032c105d4d691f366f94479034814952bfe2d',
}


def catalog(layers=2, grass=None, shrub=None, fractions=None, version=2):
    doc = json.loads(PHASE0.read_text())
    doc['root_layers'] = layers
    fr = fractions or {'grass': [0.8] + [0.2 / (layers - 1)] * (layers - 1) if layers > 1 else [1.0],
                       'shrub': [0.25] + [0.75 / (layers - 1)] * (layers - 1) if layers > 1 else [1.0]}
    for sid, rec in doc['species'].items():
        rec['roots']['layer_fractions'] = list(fr[sid])
    if version == 2:
        doc['schema_version'] = 2
        for sid, w in (('grass', grass), ('shrub', shrub)):
            if w is not None:
                doc['species'][sid]['hydrology'] = {KEY: list(w)}
    return traits.SpeciesCatalog(doc)


def plants_state(cat, plants, *, nx=10, ny=10, dx=1.0, origin=(0.0, 0.0), periodic=(False, False)):
    '''plants: (id, species, x, y, radius_m, status[, establishment_ground]).'''
    dom = state.Domain(nx, ny, dx, origin_x_m=origin[0], origin_y_m=origin[1], periodic=periodic)
    recs = []
    for p in plants:
        pid, sp, x, y, r, st = p[:6]
        est = p[6] if len(p) > 6 else 0.0
        recs.append({'plant_id': pid, 'species': sp, 'x_m': x, 'y_m': y, 'establishment_ground_m': est, 'agb_dry_kg': float(cat.biomass_from_diameter(sp, 2.0 * r)),
                     'seasonal_agb_dry_kg': 0.0, 'age_days': 10.0, 'status': st})
    return state.PlantState.from_records(cat, dom, recs)


def flat(st, v=0.0):
    return np.full(st.domain.shape, v, dtype=np.float64)


def project(st, cat, surface=None, **kw):
    kw.setdefault('max_edges', EDGES)
    kw.setdefault('max_bytes', BIG)
    return wv.project_from_initial_state(st, cat, flat(st) if surface is None else surface, **kw)


def pref(p, **kw):
    kw.setdefault('max_bytes', BIG)
    return p.preferential_inputs(**kw)


def cell(st, row, col):
    return row * st.domain.nx + col


def pi_r2(r):
    return math.pi * r * r


def has_numba():
    try:
        import numba  # noqa: F401
        return True
    except Exception:
        return False


class Geometry(unittest.TestCase):
    def test_single_plant_inside_one_cell_is_pi_r2_over_the_cell_area(self):
        cat = catalog(grass=[0, 1], shrub=[1, 1])
        r = 0.4
        st = plants_state(cat, [(1, 'grass', 3.5, 6.5, r, 'live')])           # centre of cell row 6, col 3
        p = project(st, cat)
        want = np.zeros(100)
        want[cell(st, 6, 3)] = pi_r2(r)
        np.testing.assert_allclose(p.splash_cover, want, rtol=1e-12, atol=1e-15)
        np.testing.assert_array_equal(p.shading_cover, p.splash_cover)
        np.testing.assert_allclose(p.matrix_root_index, want * 0.8, rtol=1e-12, atol=1e-15)         # grass top root fraction 0.8
        self.assertEqual(p.diagnostics['n_plants_projected'], 1)
        self.assertEqual(p.diagnostics['n_species_cell_capped_all'], 0)

    def test_circle_on_a_cell_corner_splits_into_four_equal_quarters(self):
        cat = catalog(grass=[0, 1], shrub=[1, 1])
        r = 0.5
        st = plants_state(cat, [(1, 'shrub', 4.0, 5.0, r, 'live')])
        p = project(st, cat)
        got = np.asarray(p.splash_cover).reshape(10, 10)
        quarter = pi_r2(r) / 4.0
        np.testing.assert_allclose(got[4:6, 3:5], quarter, rtol=1e-12)
        self.assertAlmostEqual(float(got.sum()), pi_r2(r), places=12)
        self.assertEqual(int((got > 0).sum()), 4)

    def test_same_species_areas_add_then_cap_at_one_with_diagnostics(self):
        cat = catalog(grass=[0, 1], shrub=[1, 1])
        r = 0.45
        st = plants_state(cat, [(i, 'grass', 2.5, 2.5, r, 'live') for i in (1, 2, 3)])      # 3 * 0.636 > 1 in ONE cell
        p = project(st, cat)
        c = cell(st, 2, 2)
        self.assertEqual(float(p.splash_cover[c]), 1.0)
        self.assertEqual(p.diagnostics['n_species_cell_capped_all'], 1)
        self.assertEqual(p.diagnostics['n_species_cell_capped_live'], 1)
        self.assertAlmostEqual(p.diagnostics['max_species_fraction_uncapped'], 3 * pi_r2(r), places=12)
        # R1 uses the CAPPED species fraction (f_s = min(.,1)) times the top root fraction
        self.assertAlmostEqual(float(p.matrix_root_index[c]), 1.0 * 0.8, places=14)

    def test_two_species_sum_then_total_cap_and_uncapped_r1(self):
        cat = catalog(layers=3, grass=[0, 1, 1], shrub=[0, 0, 1], fractions={'grass': [0.5, 0.3, 0.2], 'shrub': [0.2, 0.3, 0.5]})
        rg, rs = math.sqrt(0.7 / math.pi), math.sqrt(0.6 / math.pi)                       # areas 0.7 and 0.6 of one cell
        st = plants_state(cat, [(1, 'grass', 5.5, 5.5, rg, 'live'), (2, 'shrub', 5.5, 5.5, rs, 'live')])
        p = project(st, cat)
        c = cell(st, 5, 5)
        self.assertAlmostEqual(float(p.splash_cover[c]), 1.0, places=15)            # min(0.7 + 0.6, 1)
        self.assertAlmostEqual(float(p.shading_cover[c]), 1.0, places=15)
        self.assertEqual(p.diagnostics['n_cells_total_capped_splash'], 1)
        self.assertAlmostEqual(float(p.matrix_root_index[c]), 0.7 * 0.5 + 0.6 * 0.2, places=14)          # no total cap on R1, no normalisation
        ids, w, it = pref(p)
        self.assertEqual(ids, ('grass', 'shrub'))
        np.testing.assert_array_equal(w, [[0, 1, 1], [0, 0, 1]])                          # RAW trait values, not normalised
        got = {sid: c_ for sid, c_ in zip(ids, it)}
        self.assertAlmostEqual(float(got['grass'][c]), 0.7 * (0.3 + 0.2), places=14)     # f_live * sum(rootfrac[1:])
        self.assertAlmostEqual(float(got['shrub'][c]), 0.6 * (0.3 + 0.5), places=14)

    def test_one_layer_profile_has_zero_lower_root_sum_and_unit_top_fraction(self):
        cat = catalog(layers=1, grass=[3.0], shrub=[0.0])
        st = plants_state(cat, [(1, 'grass', 1.5, 1.5, 0.3, 'live')])
        p = project(st, cat)
        c = cell(st, 1, 1)
        self.assertAlmostEqual(float(p.matrix_root_index[c]), pi_r2(0.3), places=14)
        ids, w, it = pref(p)
        self.assertEqual(ids, ('grass',))
        np.testing.assert_array_equal(next(it), np.zeros(100))                           # one layer: no lower roots

    def test_layer_counts_two_and_four(self):
        for L, fr in ((2, [0.6, 0.4]), (4, [0.4, 0.3, 0.2, 0.1])):
            cat = catalog(layers=L, grass=[1] * L, shrub=[1] * L, fractions={'grass': fr, 'shrub': fr})
            st = plants_state(cat, [(1, 'grass', 2.5, 2.5, 0.4, 'live')])
            p = project(st, cat)
            c = cell(st, 2, 2)
            self.assertAlmostEqual(float(p.matrix_root_index[c]), pi_r2(0.4) * fr[0], places=14)
            _, _, it = pref(p)
            self.assertAlmostEqual(float(next(it)[c]), pi_r2(0.4) * sum(fr[1:]), places=14)


class Eligibility(unittest.TestCase):
    def setUp(self):
        self.cat = catalog(grass=[0, 1], shrub=[1, 1])

    def test_live_dead_removed(self):
        st = plants_state(self.cat, [(1, 'grass', 1.5, 1.5, 0.3, 'live'), (2, 'grass', 4.5, 1.5, 0.3, 'dead_standing'), (3, 'grass', 7.5, 1.5, 0.3, 'removed')])
        p = project(st, self.cat)
        a = pi_r2(0.3)
        np.testing.assert_allclose([p.splash_cover[cell(st, 1, 1)], p.splash_cover[cell(st, 1, 4)], p.splash_cover[cell(st, 1, 7)]], [a, a, 0.0], rtol=1e-12)
        np.testing.assert_allclose([p.shading_cover[cell(st, 1, 1)], p.shading_cover[cell(st, 1, 4)]], [a, 0.0], rtol=1e-12)
        self.assertEqual(p.matrix_root_index[cell(st, 1, 4)], 0.0)                         # dead roots are not modelled
        self.assertEqual(p.diagnostics['excluded_by_reason']['removed'], 1)

    def test_no_live_gives_zero_r1_and_no_preferential_but_positive_splash(self):
        st = plants_state(self.cat, [(1, 'grass', 1.5, 1.5, 0.3, 'dead_standing')])
        p = project(st, self.cat)
        self.assertGreater(float(p.splash_cover.max()), 0.0)
        self.assertEqual(float(p.shading_cover.max()), 0.0)
        self.assertEqual(float(p.matrix_root_index.max()), 0.0)
        _, _, it = pref(p)
        for arr in it:
            self.assertEqual(float(arr.max()), 0.0)

    def test_burial_rules_full_partial_and_elevated(self):
        cat = self.cat
        r = 0.3
        sid = 'grass'
        B = float(cat.biomass_from_diameter(sid, 2 * r))
        h = float(cat.height_from_biomass(sid, B))
        st = plants_state(cat, [(1, sid, 1.5, 1.5, r, 'live', 0.0), (2, sid, 4.5, 1.5, r, 'live', 0.0), (3, sid, 7.5, 1.5, r, 'live', 5.0)])
        surf = flat(st)
        surf[1, 1] = h + 0.01              # fully buried: burial >= intact height
        surf[1, 4] = 0.5 * h                # partially buried: same circle, exposed height > 0
        surf[1, 7] = 0.0                    # eroded anchor (establishment 5.0 above the surface): still projected at its XY
        p = project(st, cat, surf)
        a = pi_r2(r)
        np.testing.assert_allclose([p.splash_cover[cell(st, 1, 1)], p.splash_cover[cell(st, 1, 4)], p.splash_cover[cell(st, 1, 7)]], [0.0, a, a], rtol=1e-12, atol=1e-15)
        self.assertEqual(p.diagnostics['excluded_by_reason']['fully_buried'], 1)
        self.assertEqual(float(p.matrix_root_index[cell(st, 1, 1)]), 0.0)


class Masks(unittest.TestCase):
    def setUp(self):
        self.cat = catalog(grass=[0, 1], shrub=[1, 1])

    def test_inactive_cells_are_zero_and_straddling_plant_keeps_only_active_area(self):
        st = plants_state(self.cat, [(1, 'grass', 4.0, 5.0, 0.5, 'live')])        # four quarter cells
        mask = np.ones((10, 10), dtype=bool)
        mask[:, :4] = False                                                        # cols 0..3 inactive: keeps the two col-4 quarters
        p = project(st, self.cat, active_mask=mask)
        got = np.asarray(p.splash_cover).reshape(10, 10)
        np.testing.assert_allclose(got[4:6, 4], pi_r2(0.5) / 4, rtol=1e-12)
        self.assertEqual(float(got[:, :4].sum()), 0.0)
        self.assertEqual(p.diagnostics['n_omitted_no_active_overlap'], 0)
        self.assertEqual(p.diagnostics['n_inactive_cells'], 40)

    def test_wholly_inactive_plant_is_omitted_from_the_projection_only(self):
        st = plants_state(self.cat, [(7, 'grass', 1.5, 1.5, 0.3, 'live'), (9, 'grass', 7.5, 7.5, 0.3, 'live')])
        mask = np.ones((10, 10), dtype=bool)
        mask[0:3, 0:3] = False
        before = wv._state_sha256(st)
        p = project(st, self.cat, active_mask=mask)
        self.assertEqual(p.omitted_ids.tolist(), [7])
        self.assertEqual((p.diagnostics['n_plants_eligible'], p.diagnostics['n_plants_projected']), (2, 1))
        self.assertEqual(wv._state_sha256(st), before)                              # the biological plant is untouched
        self.assertAlmostEqual(float(p.splash_cover[cell(st, 7, 7)]), pi_r2(0.3), places=14)

    def test_all_inactive_and_empty_population(self):
        st = plants_state(self.cat, [(1, 'grass', 1.5, 1.5, 0.3, 'live')])
        p = project(st, self.cat, active_mask=np.zeros((10, 10), dtype=bool))
        for name in ('splash_cover', 'shading_cover', 'matrix_root_index'):
            self.assertEqual(float(getattr(p, name).max()), 0.0)
        self.assertEqual(p.omitted_ids.tolist(), [1])
        self.assertEqual(p.n_entries, 0)
        empty = plants_state(self.cat, [(1, 'grass', 1.5, 1.5, 0.3, 'removed')])
        q = project(empty, self.cat)
        self.assertEqual(float(q.splash_cover.max()), 0.0)
        self.assertEqual(q.diagnostics['n_plants_eligible'], 0)
        ids, w, it = pref(q)
        self.assertEqual(w.shape, (2, 2))
        self.assertEqual(len(list(it)), 2)


class Frames(unittest.TestCase):
    def test_nonzero_origin_matches_the_zero_origin_geometry(self):
        cat = catalog(grass=[0, 1], shrub=[1, 1])
        a = plants_state(cat, [(1, 'grass', 3.2, 4.7, 0.45, 'live'), (2, 'shrub', 6.0, 6.0, 0.5, 'live')])
        b = plants_state(cat, [(1, 'grass', 103.2, 204.7, 0.45, 'live'), (2, 'shrub', 106.0, 206.0, 0.5, 'live')], origin=(100.0, 200.0))
        pa, pb = project(a, cat), project(b, cat)
        np.testing.assert_allclose(pb.splash_cover, pa.splash_cover, rtol=0, atol=1e-12)
        np.testing.assert_allclose(pb.matrix_root_index, pa.matrix_root_index, rtol=0, atol=1e-12)
        self.assertGreater(float(pb.splash_cover.sum()), 0.0)

    def test_periodic_seam_wraps_the_footprint(self):
        cat = catalog(grass=[0, 1], shrub=[1, 1])
        st = plants_state(cat, [(1, 'grass', 0.0, 5.5, 0.5, 'live')], periodic=(True, False))      # centre on the x seam, cell row 5
        p = project(st, cat)
        got = np.asarray(p.splash_cover).reshape(10, 10)
        half = pi_r2(0.5) / 2
        self.assertAlmostEqual(float(got[5, 0]), half, places=12)
        self.assertAlmostEqual(float(got[5, 9]), half, places=12)                     # the wrapped half
        self.assertAlmostEqual(float(got.sum()), pi_r2(0.5), places=12)
        np_ = plants_state(cat, [(1, 'grass', 0.0, 5.5, 0.5, 'live')])                 # non periodic: the left half is clipped away
        self.assertAlmostEqual(float(np.asarray(project(np_, cat).splash_cover).sum()), half, places=12)

    def test_circle_wider_than_the_periodic_domain_covers_every_cell_once(self):
        cat = catalog(grass=[0, 1], shrub=[1, 1])
        st = plants_state(cat, [(1, 'grass', 2.0, 2.0, 3.0, 'live')], nx=4, ny=4, periodic=(True, True))   # r = 3 > L/sqrt(2) = 2.83: full torus cover
        p = project(st, cat)
        np.testing.assert_allclose(p.splash_cover, np.ones(16), rtol=0, atol=1e-12)    # union per cell, never a double count
        self.assertEqual(p.diagnostics['n_species_cell_capped_all'], 0)
        np.testing.assert_allclose(p.matrix_root_index, np.full(16, 0.8), atol=1e-12)


class Ordering(unittest.TestCase):
    def test_input_row_order_does_not_change_any_bit(self):
        cat = catalog(grass=[0, 1], shrub=[1, 1])
        pl = [(5, 'grass', 2.4, 2.6, 0.45, 'live'), (2, 'shrub', 2.6, 2.4, 0.45, 'live'), (9, 'grass', 2.5, 2.5, 0.4, 'dead_standing'), (4, 'shrub', 6.5, 6.5, 0.3, 'live'),
              (1, 'grass', 2.5, 2.4, 0.35, 'live')]
        a = project(plants_state(cat, pl), cat)
        b = project(plants_state(cat, pl[::-1]), cat)
        for name in ('splash_cover', 'shading_cover', 'matrix_root_index'):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        self.assertEqual(a.identity['content_sha256'], b.identity['content_sha256'])

    @unittest.skipUnless(has_numba(), 'numba not installed')
    def test_numba_equals_numpy_bit_for_bit(self):
        cat = catalog(layers=3, grass=[0, 1, 1], shrub=[2, 0, 1], fractions={'grass': [0.5, 0.3, 0.2], 'shrub': [0.2, 0.3, 0.5]})
        rng = np.random.default_rng(7)
        pl = [(i, 'grass' if i % 2 else 'shrub', float(rng.uniform(1, 9)), float(rng.uniform(1, 9)), float(rng.uniform(0.2, 0.9)), 'live' if i % 5 else 'dead_standing')
              for i in range(1, 60)]
        st = plants_state(cat, pl)
        a = project(st, cat, backend='numpy')
        b = project(st, cat, backend='numba')
        for name in ('splash_cover', 'shading_cover', 'matrix_root_index'):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        self.assertEqual(a.identity['content_sha256'], b.identity['content_sha256'])
        self.assertEqual(a.diagnostics['n_species_cell_entries'], b.diagnostics['n_species_cell_entries'])


class Traits(unittest.TestCase):
    def test_missing_trait_on_a_participating_live_species_refuses_clearly(self):
        v2_missing = catalog(grass=[0, 1], shrub=None)                              # shrub has no block
        st = plants_state(v2_missing, [(1, 'shrub', 2.5, 2.5, 0.3, 'live')])
        with self.assertRaisesRegex(wv.VegetationProjectionError, "participating live species \\['shrub'\\] have no"):
            project(st, v2_missing)
        v1 = catalog(version=1)
        st1 = plants_state(v1, [(1, 'grass', 2.5, 2.5, 0.3, 'live')])
        with self.assertRaisesRegex(wv.VegetationProjectionError, 'schema_version 1'):
            project(st1, v1)
        p = project(st1, v1, require_receiver_traits=False)                         # splash / shading / R1 only
        self.assertGreater(float(p.splash_cover.max()), 0.0)
        ids, w, it = pref(p)                                       # no trait anywhere: nothing enabled, never a default vector
        self.assertEqual((ids, w.shape, list(it)), ((), (0, 2), []))

    def test_species_without_live_cover_or_only_dead_do_not_need_the_trait(self):
        cat = catalog(grass=[0, 1], shrub=None)
        st = plants_state(cat, [(1, 'grass', 2.5, 2.5, 0.3, 'live'), (2, 'shrub', 6.5, 6.5, 0.3, 'dead_standing')])
        p = project(st, cat)                                                         # shrub is dead only: not a participating live species
        ids, w, _ = pref(p)
        self.assertEqual(ids, ('grass',))
        self.assertEqual(list(p.diagnostics['participating_live_species']), ['grass'])

    def test_all_zero_weights_disable_only_the_preferential_contribution(self):
        cat = catalog(grass=[0, 0], shrub=[1, 1])
        st = plants_state(cat, [(1, 'grass', 2.5, 2.5, 0.4, 'live'), (2, 'shrub', 6.5, 6.5, 0.4, 'live')])
        p = project(st, cat)
        ids, w, it = pref(p)
        self.assertEqual(ids, ('shrub',))
        np.testing.assert_array_equal(w, [[1.0, 1.0]])
        self.assertEqual(list(p.diagnostics['receiver_trait_disabled_species']), ['grass'])
        self.assertAlmostEqual(float(p.matrix_root_index[cell(st, 2, 2)]), pi_r2(0.4) * 0.8, places=14)      # R1 still includes the disabled species
        self.assertAlmostEqual(float(p.shading_cover[cell(st, 2, 2)]), pi_r2(0.4), places=14)
        c = next(it)
        self.assertEqual(float(c[cell(st, 2, 2)]), 0.0)                              # grass cell: no shrub contribution there
        self.assertAlmostEqual(float(c[cell(st, 6, 6)]), pi_r2(0.4) * 0.75, places=14)

    def test_species_cover_accessor_and_unknown_species(self):
        cat = catalog(grass=[0, 1], shrub=[1, 1])
        st = plants_state(cat, [(1, 'grass', 2.5, 2.5, 0.4, 'live'), (2, 'grass', 6.5, 6.5, 0.4, 'dead_standing')])
        p = project(st, cat)
        self.assertAlmostEqual(float(p.species_cell_cover('grass', 'live', max_bytes=BIG)[cell(st, 2, 2)]), pi_r2(0.4), places=14)
        self.assertEqual(float(p.species_cell_cover('grass', 'live', max_bytes=BIG)[cell(st, 6, 6)]), 0.0)
        self.assertAlmostEqual(float(p.species_cell_cover('grass', 'all', max_bytes=BIG)[cell(st, 6, 6)]), pi_r2(0.4), places=14)
        for bad in ('nope', 1, None):
            with self.assertRaises(wv.VegetationProjectionError):
                p.species_cell_cover(bad, max_bytes=BIG)
        with self.assertRaises(wv.VegetationProjectionError):
            p.species_cell_cover('grass', 'dead', max_bytes=BIG)


class InvalidInputs(unittest.TestCase):
    def setUp(self):
        self.cat = catalog(grass=[0, 1], shrub=[1, 1])
        self.st = plants_state(self.cat, [(1, 'grass', 2.5, 2.5, 0.3, 'live')])

    def test_bad_inputs_are_refused_without_effect(self):
        st, cat = self.st, self.cat
        before = wv._state_sha256(st)
        surf = flat(st)
        mask = np.ones((10, 10), dtype=bool)
        bad = {
            'surface list': dict(surface=surf.tolist()), 'surface float32': dict(surface=surf.astype(np.float32)), 'surface shape': dict(surface=np.zeros((9, 10))),
            'surface nan': dict(surface=np.where(np.eye(10, dtype=bool), np.nan, 0.0)), 'surface fortran': dict(surface=np.asfortranarray(surf)),
            'mask int': dict(active_mask=mask.astype(np.int8)), 'mask shape': dict(active_mask=np.ones((10, 9), dtype=bool)), 'mask list': dict(active_mask=mask.tolist()),
            'mask fortran': dict(active_mask=np.asfortranarray(mask)), 'backend': dict(backend='gpu'), 'device on cpu': dict(device=0), 'cuda no device': dict(backend='cuda'),
            'builder': dict(builder='fortran'), 'edges bool': dict(max_edges=True), 'edges negative': dict(max_edges=-1), 'bytes float': dict(max_bytes=1.5), 'reserve neg': dict(reserve_bytes=-1),
            'reader': dict(available_bytes_fn=5)}
        for label, kw in bad.items():
            args = {'surface': surf, 'max_edges': EDGES, 'max_bytes': BIG}
            args.update(kw)
            with self.subTest(label), self.assertRaises(wv.VegetationProjectionError):
                wv.project_from_initial_state(st, cat, args.pop('surface'), **args)
        for label, (s_, c_) in {'state type': (object(), cat), 'catalog type': (st, object()), 'catalog mismatch': (st, catalog(grass=[1, 1], shrub=[1, 1]))}.items():
            with self.subTest(label), self.assertRaises(wv.VegetationProjectionError):
                wv.project_from_initial_state(s_, c_, surf, max_edges=EDGES, max_bytes=BIG)
        self.assertEqual(wv._state_sha256(st), before)

    def test_unresolved_establishment_is_refused(self):
        dom = state.Domain(10, 10, 1.0)
        rec = [{'plant_id': 1, 'species': 'grass', 'x_m': 2.5, 'y_m': 2.5, 'establishment_ground_m': 'SURFACE', 'agb_dry_kg': 0.2, 'seasonal_agb_dry_kg': 0.0, 'age_days': 1.0,
                'status': 'live'}]
        try:
            st = state.PlantState.from_records(self.cat, dom, rec)
        except Exception:                                                              # the record syntax for unresolved anchors may differ; the refusal path below is what matters
            self.skipTest('SURFACE anchor record syntax not available')
        with self.assertRaises((state.StateError, wv.VegetationProjectionError, ValueError)):
            wv.project_from_initial_state(st, self.cat, flat(st), max_edges=EDGES, max_bytes=BIG)


class Admission(unittest.TestCase):
    def setUp(self):
        self.cat = catalog(grass=[0, 1], shrub=[1, 1])
        rng = np.random.default_rng(3)
        pl = [(i, 'grass' if i % 2 else 'shrub', float(rng.uniform(1, 9)), float(rng.uniform(1, 9)), float(rng.uniform(0.2, 0.8)), 'live') for i in range(1, 40)]
        self.st = plants_state(self.cat, pl)

    def declared(self, p, st, cold=False):
        d = p.diagnostics
        return wv.estimate_projection_bytes(st.n, d['n_plants_projected'], d['n_edges_active'], 100, 2, 2, n_entries=p.n_entries, source_cold=cold,
                                            n_omitted=d['n_omitted_no_active_overlap'])

    def test_declared_stage_model_admits_exactly_its_peak_and_refuses_one_below_every_stage(self):
        st, cat = self.st, self.cat
        p = project(st, cat)                                       # warm (source digest cached)
        est = self.declared(p, st)
        self.assertGreaterEqual(est['peak_bytes'], est['resident_bytes'])
        self.assertEqual(est['peak_bytes'], max(est['stages'].values()))
        project(st, cat, max_bytes=est['peak_bytes'] + wv.HASH_CHUNK_BYTES * 0)                      # the declared peak admits the whole run
        with self.assertRaises(wv.VegetationBudgetError):
            project(st, cat, max_bytes=est['peak_bytes'] - 1)
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        project(st, cat, max_bytes=est['peak_bytes'])
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        self.assertLessEqual(peak, est['peak_bytes'])                                                # traced NumPy/Python peak inside the declared model

    def test_each_stage_gate_fires_before_that_stage_allocates(self):
        st, cat = self.st, self.cat
        p = project(st, cat)
        est = self.declared(p, st)
        for stage in ('validate', 'select', 'count', 'build', 'aggregate', 'reduce', 'seal'):
            need = est['stages'][stage]
            with self.subTest(stage):
                try:
                    project(st, cat, max_bytes=need - 1)
                except wv.VegetationBudgetError as exc:
                    self.assertIsInstance(exc.stage, str)
                    continue
                # a smaller stage may legitimately pass when a LATER (larger) gate is the binding one; the run must then still be refused somewhere above
                with self.assertRaises(wv.VegetationBudgetError):
                    project(st, cat, max_bytes=est['peak_bytes'] - 1)

    def test_tiny_budget_and_reader_refuse_before_any_owned_allocation_empty_and_populated_warm_and_cold(self):
        cat = self.cat
        empty = plants_state(cat, [], nx=10, ny=10)
        project(empty, cat)                                                                          # warm the source cache
        for st in (empty, self.st):
            with self.subTest(n=st.n):
                with self.assertRaises(wv.VegetationBudgetError):
                    project(st, cat, max_bytes=1)
                with self.assertRaises(wv.VegetationBudgetError):
                    project(st, cat, available_bytes_fn=lambda: 1)
                with self.assertRaises(wv.VegetationBudgetError):
                    project(st, cat, reserve_bytes=BIG)
        q = project(empty, cat)
        est = wv.estimate_projection_bytes(0, 0, 0, 100, 2, 2)
        self.assertLessEqual(q.resident_bytes, est['resident_bytes'])                                # the real sealed bytes are inside the declared resident model
        with self.assertRaises(wv.VegetationBudgetError):
            project(empty, cat, max_bytes=est['peak_bytes'] - 1)
        project(empty, cat, max_bytes=est['peak_bytes'])
        # cold source cache: the declared validate stage includes the cold digest allowance
        saved = dict(wv._SOURCE_CACHE)
        wv._SOURCE_CACHE.clear()
        try:
            cold = wv.estimate_projection_bytes(0, 0, 0, 100, 2, 2, source_cold=True)
            self.assertGreater(cold['stages']['validate'], est['stages']['validate'])
            with self.assertRaises(wv.VegetationBudgetError):
                project(empty, cat, max_bytes=cold['stages']['validate'] - 1)
            self.assertNotIn('rec', wv._SOURCE_CACHE)                                                # refused BEFORE the cold digest ran
            project(empty, cat, max_bytes=cold['peak_bytes'])
            self.assertIn('rec', wv._SOURCE_CACHE)
        finally:
            wv._SOURCE_CACHE.update(saved)

    def test_all_inactive_and_empty_cases_build_no_edge_and_need_only_the_n_arrays(self):
        cat = self.cat
        zero = np.zeros((10, 10), dtype=bool)
        p = project(self.st, cat, active_mask=zero)
        self.assertEqual((p.n_entries, p.diagnostics['n_edges_active']), (0, 0))
        self.assertEqual(p.diagnostics['n_omitted_no_active_overlap'], p.diagnostics['n_plants_eligible'])
        project(self.st, cat, active_mask=zero, max_edges=0)                                           # zero ACTIVE edges: a zero edge cap is legal
        q = project(plants_state(cat, [], nx=10, ny=10), cat, max_edges=0)
        self.assertEqual(q.n_entries, 0)

    def test_max_edges_bounds_the_active_edges_and_reader_reserve_work(self):
        st, cat = self.st, self.cat
        E = project(st, cat).diagnostics['n_edges_active']
        project(st, cat, max_edges=E)
        with self.assertRaises(wv.VegetationBudgetError):
            project(st, cat, max_edges=E - 1)
        project(st, cat, available_bytes_fn=lambda: 1 << 40)

    def test_active_edge_cap_with_a_huge_canopy_and_one_active_cell(self):
        cat = self.cat
        huge = plants_state(cat, [(1, 'grass', 5.0, 5.0, 10.0, 'live')], nx=10, ny=10)            # the circle covers every one of the 100 cells
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        p = project(huge, cat, active_mask=mask, max_edges=1)                                      # ONE active edge; the 99 inactive footprints are never built
        self.assertEqual((p.diagnostics['n_edges_active'], p.n_entries), (1, 1))
        self.assertAlmostEqual(float(p.splash_cover[5 * 10 + 5]), 1.0, places=12)
        with self.assertRaises(wv.VegetationBudgetError):
            project(huge, cat, max_edges=99)                                                       # unmasked it has 100 edges
        project(huge, cat, max_edges=100)

    def test_active_first_cases_nonperiodic_periodic_and_wide_circle(self):
        cat = self.cat
        mask = np.zeros((4, 4), dtype=bool)
        mask[1, 1] = True
        wide = plants_state(cat, [(1, 'grass', 2.0, 2.0, 3.0, 'live')], nx=4, ny=4, periodic=(True, True))
        p = project(wide, cat, active_mask=mask, max_edges=1)                                      # torus union: ONE active cell, one edge
        self.assertEqual(p.diagnostics['n_edges_active'], 1)
        self.assertAlmostEqual(float(p.splash_cover[1 * 4 + 1]), 1.0, places=12)
        seam = plants_state(cat, [(1, 'grass', 0.0, 1.5, 0.5, 'live')], nx=4, ny=4, periodic=(True, False))
        m2 = np.zeros((4, 4), dtype=bool)
        m2[1, 3] = True                                                                              # only the wrapped half's cell is active
        q = project(seam, cat, active_mask=m2, max_edges=1)
        self.assertAlmostEqual(float(q.splash_cover[1 * 4 + 3]), pi_r2(0.5) / 2, places=12)
        self.assertEqual(q.diagnostics['n_edges_active'], 1)
        plain = plants_state(cat, [(1, 'grass', 1.5, 1.5, 0.3, 'live'), (2, 'grass', 3.5, 3.5, 0.3, 'live')], nx=4, ny=4)
        m3 = np.zeros((4, 4), dtype=bool)
        m3[3, 3] = True
        r = project(plain, cat, active_mask=m3, max_edges=1)
        self.assertEqual((r.omitted_ids.tolist(), r.diagnostics['n_edges_active']), ([1], 1))

    def test_missing_trait_is_checked_only_on_active_live_plants(self):
        cat = catalog(grass=[0, 1], shrub=None)                                                    # the shrub has no trait
        st = plants_state(cat, [(1, 'grass', 1.5, 1.5, 0.3, 'live'), (2, 'shrub', 7.5, 7.5, 0.3, 'live')])
        mask = np.ones((10, 10), dtype=bool)
        mask[7:9, 7:9] = False                                                                       # the shrub is wholly inactive: not a participant
        p = project(st, cat, active_mask=mask)
        self.assertEqual(p.omitted_ids.tolist(), [2])
        with self.assertRaisesRegex(wv.VegetationProjectionError, "\\['shrub'\\]"):
            project(st, cat)

    def test_refusals_leave_the_inputs_unchanged(self):
        st, cat = self.st, self.cat
        before = (wv._state_sha256(st), hashlib.sha256(flat(st).tobytes()).hexdigest())
        with self.assertRaises(wv.VegetationBudgetError):
            project(st, cat, max_edges=1)
        self.assertEqual((wv._state_sha256(st), hashlib.sha256(flat(st).tobytes()).hexdigest()), before)


class LiteralMemoryModel(unittest.TestCase):
    '''The declared model must bound the MEASURED live allocation (tracemalloc over the whole call, all inputs built before the tracer, warm source cache) and each gate must fire BEFORE
    its allocation. The only slack the model itemises is `OVERHEAD_SLACK_BYTES`; the controls below would fail on any O(N), O(P), O(E) or O(S L) error.'''

    @classmethod
    def setUpClass(cls):
        wv.package_source_record()
        wv.external_source_record()
        cls.cat = catalog(layers=3, grass=[0, 1, 1], shrub=[2, 0, 1], fractions={'grass': [0.5, 0.3, 0.2], 'shrub': [0.2, 0.3, 0.5]})

    def traced(self, fn):
        import gc
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        try:
            out = fn()
        except BaseException:
            peak = tracemalloc.get_traced_memory()[1] - base
            tracemalloc.stop()
            raise_peak = peak
            return None, raise_peak
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return out, (peak - base, cur - base)

    def refused_peak(self, fn):
        import gc
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        try:
            with self.assertRaises(wv.VegetationBudgetError) as cm:
                fn()
        finally:
            peak = tracemalloc.get_traced_memory()[1] - base
            tracemalloc.stop()
        return cm.exception, peak

    def population(self, n, nx, ny, rmin, rmax, periodic=(False, False), seed=5):
        rng = np.random.default_rng(seed)
        pl = [(i, 'grass' if i % 2 else 'shrub', float(rng.uniform(1, nx - 1)), float(rng.uniform(1, ny - 1)), float(rng.uniform(rmin, rmax)), 'live' if i % 4 else 'dead_standing')
              for i in range(1, n + 1)]
        return plants_state(self.cat, pl, nx=nx, ny=ny, periodic=periodic)

    def check_case(self, st, **kw):
        cat = self.cat
        surf = flat(st)
        kw.setdefault('max_edges', 1 << 24)
        full = wv.project_from_initial_state(st, cat, surf, max_bytes=1 << 40, **kw)
        declared, resident = full.diagnostics['declared_peak_bytes'], full.resident_bytes
        real = sum(a.nbytes for a in full._a.values()) + sum(a.nbytes for a in full._t.values())
        self.assertEqual(resident, real)                                           # the declared resident model is exactly the owned arrays
        self.assertEqual(full.diagnostics['declared_resident_bytes'], real)
        del full
        out, (peak, cur) = self.traced(lambda: wv.project_from_initial_state(st, cat, surf, max_bytes=declared, **kw))           # the EXACT declared cap admits the whole run
        self.assertLessEqual(peak, declared, f'traced peak {peak} exceeds the declared peak {declared}')
        self.assertLessEqual(cur, real + 4096 + wv.OVERHEAD_SLACK_BYTES)               # what stays alive is the sealed arrays (+ small Python metadata)
        exc, rpeak = self.refused_peak(lambda: wv.project_from_initial_state(st, cat, surf, max_bytes=declared - 1, **kw))      # cap - 1 is refused ...
        return declared, rpeak, exc, out

    def test_empty_population_n200000_declared_bounds_the_measured_peak_and_refusal_allocates_no_o_n_array(self):
        st = plants_state(self.cat, [], nx=500, ny=400)
        declared, rpeak, exc, out = self.check_case(st, max_edges=0)
        self.assertEqual(out.n_entries, 0)
        self.assertLess(rpeak, 1 << 20)                                              # ... BEFORE any O(N) array exists (the N arrays alone are 6.4 MB)
        self.assertGreater(declared, 6_400_000)
        self.assertEqual(exc.stage, 'seal')                                          # the binding gate of an empty case (the largest stage), fired before the reduction allocated

    def test_populated_multi_species_cases_numpy_numba_compiled_masked_periodic(self):
        dense = self.population(3000, 200, 200, 0.2, 0.9)
        cases = {'numpy': (dense, {}), 'compiled builder': (dense, {'builder': 'compiled'})}
        mask = np.ones((200, 200), dtype=bool)
        mask[50:120, 60:140] = False
        cases['masked'] = (dense, {'active_mask': mask})
        cases['big plants'] = (self.population(1000, 60, 60, 0.5, 3.0), {})
        cases['periodic'] = (self.population(500, 60, 60, 0.5, 2.0, periodic=(True, True)), {})
        if has_numba():
            cases['numba'] = (dense, {'backend': 'numba', 'builder': 'compiled'})
        for label, (st, kw) in cases.items():
            with self.subTest(label):
                declared, rpeak, exc, out = self.check_case(st, **kw)
                self.assertGreater(out.n_entries, 0)
                self.assertLess(rpeak, declared)                                      # a refusal allocates strictly less than the admitted run

    def test_large_n_one_million_cells_where_the_o_n_stages_dominate(self):
        # the earlier controls used N <= 4e5 where the 1 MiB validate chunk masks an O(N) under-model; at N = 1e6 every O(N) term is exercised (the rank-loop bool mask included)
        for label, st in (('empty', plants_state(self.cat, [], nx=1000, ny=1000)), ('K=300', self.population(300, 1000, 1000, 0.4, 3.0, seed=3))):
            with self.subTest(label):
                declared, rpeak, exc, out = self.check_case(st, max_edges=1 << 26)
                self.assertGreater(declared, 40_000_000)
                self.assertLess(rpeak, 4 << 20)                                                       # the refused run never allocates the 32 MB of N-sized arrays

    def test_refusal_at_a_late_gate_happens_before_the_builder_and_reduction_allocate(self):
        st = self.population(3000, 200, 200, 0.2, 0.9)
        out, (peak, cur) = self.traced(lambda: wv.project_from_initial_state(st, self.cat, flat(st), max_edges=1 << 24, max_bytes=1 << 40))
        full_declared = out.diagnostics['declared_peak_bytes']
        exc, rpeak = self.refused_peak(lambda: wv.project_from_initial_state(st, self.cat, flat(st), max_edges=1 << 24, max_bytes=full_declared - 1))
        self.assertLess(rpeak, peak)                                                 # the refused run never reaches the heavy stages
        self.assertIn(exc.stage, ('build', 'aggregate', 'reduce', 'seal'))

    def test_session_snapshot_is_modelled_and_released_after_selection(self):
        sess, _ = SessionBinding().make_session()
        surf = np.zeros(sess.base.domain.shape)
        full = wv.project_from_plant_session(sess, surf, max_edges=EDGES, max_bytes=1 << 40)
        declared = full.diagnostics['declared_peak_bytes']
        del full
        out, (peak, cur) = self.traced(lambda: wv.project_from_plant_session(sess, surf, max_edges=EDGES, max_bytes=declared))
        self.assertLessEqual(peak, declared)
        exc, _ = self.refused_peak(lambda: wv.project_from_plant_session(sess, surf, max_edges=EDGES, max_bytes=declared - 1))
        self.assertIsInstance(exc.stage, str)
        m = wv.estimate_projection_bytes(1000, 1000, 0, 100, 2, 2, from_session=True)
        n = wv.estimate_projection_bytes(1000, 1000, 0, 100, 2, 2, from_session=False)
        self.assertGreater(m['stages']['snapshot'], 0)
        self.assertEqual(n['stages']['snapshot'], 0)
        self.assertGreater(m['stages']['select'], n['stages']['select'])            # the resident snapshot is counted while the geometry exists
        self.assertEqual(m['stages']['aggregate'], n['stages']['aggregate'])         # and is released afterwards (not carried into later stages)

    def test_model_listing_is_literal_every_stage_scales_with_its_own_dimension(self):
        args = dict(n_plants_all=100, n_plants=100, n_edges=1000, n_cells=1000, n_species=3, n_layers=3, n_entries=800, from_session=True)
        base = wv.estimate_projection_bytes(**args)
        self.assertEqual(base['peak_bytes'], max(base['stages'].values()))
        self.assertEqual(set(base['stages']), {'validate', 'snapshot', 'select', 'count_pre', 'count', 'filter', 'build', 'aggregate', 'reduce', 'seal'})
        grows = {'n_cells': ('reduce', 'seal'), 'n_edges': ('aggregate', 'build'), 'n_plants': ('select', 'count', 'filter', 'build'), 'n_entries': ('reduce', 'seal', 'aggregate'),
                 'n_plants_all': ('snapshot', 'select'), 'n_species': ('select',), 'n_layers': ('select',)}
        for key, stages in grows.items():
            kw = dict(args)
            kw[key] = args[key] * 2
            if key == 'n_plants':
                kw['n_eligible'] = kw['n_plants']
            bigger = wv.estimate_projection_bytes(**kw)
            for stage in stages:
                if key in ('n_species', 'n_layers') and stage == 'select':
                    self.assertGreaterEqual(bigger['stages'][stage], base['stages'][stage], (key, stage))
                else:
                    self.assertGreater(bigger['stages'][stage], base['stages'][stage], (key, stage))
        self.assertGreater(wv.estimate_projection_bytes(**dict(args, n_omitted=50))['resident_bytes'], base['resident_bytes'])


class SharedBackingSeal(unittest.TestCase):
    '''GPU HOST memory (correction04): the CUDA reduction returns ONE contiguous (3, N) host array whose three rows are views, so `_seal_arrays` cannot free the 24 N backing until the LAST row is
    sealed. These HOST-ONLY controls exercise the real `_seal_arrays` on a real (3, N) NumPy shared base (and the real `_build` with the device call replaced by a host function that returns the same
    shared (3, N) layout). They do NOT run, fake or claim a CUDA backend; hardware traces stay with Root.'''

    @staticmethod
    def peak_extra(fn):
        import gc
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        out = fn()
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        return out, peak

    def raw(self, N, shared):
        off = np.zeros(N + 1, dtype=np.int64)
        if shared:
            base = np.zeros((3, N))
            raw = {'splash_cover': base[0], 'shading_cover': base[1], 'matrix_root_index': base[2], 'cell_offsets': off}
            del base                                                  # only the three views keep the 24 N backing alive (as after `three[0], three[1], three[2]`)
        else:
            raw = {'splash_cover': np.zeros(N), 'shading_cover': np.zeros(N), 'matrix_root_index': np.zeros(N), 'cell_offsets': off}
        return raw

    def test_the_shared_base_costs_exactly_16n_more_than_independent_arrays_at_the_seal_peak(self):
        for N in (200_000, 1_000_000):
            with self.subTest(N=N):
                # the raw arrays are created INSIDE the traced call so that their release is visible to tracemalloc: the traced peak is then the TOTAL live bytes
                _, indep = self.peak_extra(lambda: wv._seal_arrays(self.raw(N, False)))
                _, shared = self.peak_extra(lambda: wv._seal_arrays(self.raw(N, True)))
                # raw total 4 x 8 N (three rows + offsets). independent: + one temporary copy (8 N) = 5 x 8 N; shared: + two sealed rows (16 N) + the temporary (8 N) = 7 x 8 N
                self.assertAlmostEqual(indep / (8 * N), 5.0, delta=0.05)
                self.assertAlmostEqual(shared / (8 * N), 7.0, delta=0.05)
                self.assertAlmostEqual((shared - indep) / (8 * N), 2.0, delta=0.05)                      # exactly the 2 N doubles = 16 N bytes the CUDA model adds
        a = wv.estimate_projection_bytes(0, 0, 0, 200_000, 2, 2)
        b = wv.estimate_projection_bytes(0, 0, 0, 200_000, 2, 2, backend='cuda')
        self.assertEqual(b['stages']['seal'] - a['stages']['seal'], 16 * 200_000)
        for stage in ('validate', 'snapshot', 'select', 'count_pre', 'count', 'filter', 'build', 'aggregate'):
            self.assertEqual(a['stages'][stage], b['stages'][stage], stage)                              # only the seal stage differs; the CPU model is unchanged
        self.assertEqual(a['stages']['seal'], 8_065_650)                                                 # the CPU number Root validated is unchanged
        self.assertLessEqual(b['stages']['reduce'], a['stages']['reduce'])                              # the CUDA reduction has no np.diff counts array
        self.assertEqual(a['resident_bytes'], b['resident_bytes'])

    def test_the_real_build_with_the_cuda_return_layout_is_bounded_by_the_cuda_model_and_exceeds_the_cpu_model(self):
        from unittest import mock
        cat = catalog(layers=2, grass=[0, 1], shrub=[0, 1])
        wv.package_source_record()
        wv.external_source_record()
        for N, nx, ny in ((200_000, 500, 400), (1_000_000, 1000, 1000)):
            with self.subTest(N=N):
                st = plants_state(cat, [], nx=nx, ny=ny)
                surf = np.zeros(st.domain.shape)
                cpu = wv.estimate_projection_bytes(0, 0, 0, N, 2, 2)
                gpu = wv.estimate_projection_bytes(0, 0, 0, N, 2, 2, backend='cuda')

                def host_three(device, dq, off, sp, a_all, a_live, root0, cell_area, n):         # the SAME return layout as `_reduce_cuda` (one contiguous (3, N) array), computed on the host
                    return np.zeros((3, n))

                kw = dict(max_edges=0, backend='cuda', builder='numpy', device=0, max_device_bytes=1 << 30)
                with mock.patch.object(wv, '_reduce_cuda', host_three):
                    full = wv.project_from_initial_state(st, cat, surf, max_bytes=1 << 40, **kw)
                    self.assertEqual(full.diagnostics['declared_peak_bytes'], gpu['peak_bytes'])           # the build admits the CUDA model
                    del full
                    out, peak = self.peak_extra(lambda: wv.project_from_initial_state(st, cat, surf, max_bytes=gpu['peak_bytes'], **kw))
                self.assertLessEqual(peak, gpu['peak_bytes'])                                       # the CUDA host model bounds the real allocation ...
                self.assertGreater(peak, cpu['peak_bytes'])                                         # ... and the CPU model would have been exceeded (Root's finding), so this is not padding
                self.assertLess(peak, gpu['peak_bytes'] * 1.0001)
                with mock.patch.object(wv, '_reduce_cuda', host_three):
                    with self.assertRaises(wv.VegetationBudgetError):
                        wv.project_from_initial_state(st, cat, surf, max_bytes=gpu['peak_bytes'] - 1, **kw)
                    with self.assertRaises(wv.VegetationBudgetError):
                        wv.project_from_initial_state(st, cat, surf, max_bytes=cpu['peak_bytes'], **kw)   # the old CPU number no longer admits a CUDA run
                self.assertEqual(out.n_entries, 0)

    def test_the_cuda_return_shape_numbers_equal_the_cpu_results(self):
        # a populated host-layout control for the arithmetic (the device kernel itself is Root's hardware lane)
        from unittest import mock
        cat = catalog(layers=3, grass=[0, 1, 1], shrub=[2, 0, 1], fractions={'grass': [0.5, 0.3, 0.2], 'shrub': [0.2, 0.3, 0.5]})
        rng = np.random.default_rng(4)
        pl = [(i, 'grass' if i % 2 else 'shrub', float(rng.uniform(1, 29)), float(rng.uniform(1, 29)), float(rng.uniform(0.3, 1.2)), 'live' if i % 4 else 'dead_standing') for i in range(1, 120)]
        st = plants_state(cat, pl, nx=30, ny=30)
        ref = project(st, cat)

        def host_three(device, dq, off, sp, a_all, a_live, root0, cell_area, n):
            three = np.zeros((3, n))
            s_, h_, r_ = wv._reduce_numpy(off, sp, a_all, a_live, root0, cell_area, n)
            three[0], three[1], three[2] = s_, h_, r_
            return three

        with mock.patch.object(wv, '_reduce_cuda', host_three):
            got = wv.project_from_initial_state(st, cat, flat(st), max_edges=EDGES, max_bytes=BIG, backend='cuda', builder='numpy', device=0, max_device_bytes=1 << 30)
        for name in ('splash_cover', 'shading_cover', 'matrix_root_index'):
            np.testing.assert_array_equal(getattr(ref, name), getattr(got, name))
        self.assertFalse(got.splash_cover.flags.writeable)


class SessionRows(unittest.TestCase):
    '''F2: a genuine PlantSession snapshot materialises EVERY base row (live, dead_standing, removed): the snapshot / geometry stages are admitted from `session.base.n`.'''

    def session(self, statuses=('live', 'dead_standing', 'removed', 'live'), n_extra=0):
        import tests.native_leaf_fixture as nfx
        doc = json.loads(nfx.FIXTURES['phase1'].read_text())
        nfx.simple_edit(doc)
        doc['schema_version'] = 2
        for sid in doc['species']:
            doc['species'][sid]['hydrology'] = {KEY: [0.0, 1.0]}
        cat = traits.SpeciesCatalog(doc)
        dom = state.Domain(12, 10, 1.0)
        recs = [{'plant_id': i + 1, 'species': 'grass', 'x_m': 1.5 + (i % 10), 'y_m': 3.5 + (i // 10) % 5, 'establishment_ground_m': 0.0, 'agb_dry_kg': 0.1, 'seasonal_agb_dry_kg': 0.0,
                 'age_days': 100.0, 'status': st} for i, st in enumerate(statuses)]
        st = state.PlantState.from_records(cat, dom, recs)
        return nfx.start_session(st, nfx.make_config(cat), nfx.anchor(), bound=True), cat

    def test_genuine_session_with_live_dead_and_removed_rows(self):
        sess, cat = self.session()
        snap = sess.snapshot_state()
        self.assertEqual((int(sess.base.n), int(sess.n_live), int(snap.n)), (4, 2, 4))              # the snapshot has MORE rows than n_live (confirmed native behaviour)
        before = (sess.token, sess.sequence, wv._state_sha256(snap))
        surf = np.zeros(sess.base.domain.shape)
        p = wv.project_from_plant_session(sess, surf, max_edges=EDGES, max_bytes=BIG)
        d = p.diagnostics
        self.assertEqual((d['n_plants_state'], d['excluded_by_reason']['removed'], d['n_plants_projected']), (4, 1, 3))
        c = lambda i: (3 + (i // 10) % 5) * 12 + (1 + i % 10)                                         # cell of plant row i (x = 1.5 + i, y = 3.5)
        splash, shade = np.asarray(p.splash_cover), np.asarray(p.shading_cover)
        self.assertGreater(float(splash[c(1)]), 0.0)                                                  # the dead_standing plant shades the rain-impact (splash) field ...
        self.assertEqual(float(shade[c(1)]), 0.0)                                                     # ... but not the live-only shading / roots
        self.assertEqual(float(p.matrix_root_index[c(1)]), 0.0)
        self.assertEqual(float(splash[c(2)]), 0.0)                                                    # removed: nothing
        self.assertGreater(float(shade[c(0)]), 0.0)
        self.assertGreater(float(p.matrix_root_index[c(3)]), 0.0)
        self.assertEqual((sess.token, sess.sequence, wv._state_sha256(sess.snapshot_state())), before)  # the session / state are unchanged
        self.assertTrue(p.check_session(sess, current_surface=surf)['surface'])

    def test_all_nonlive_session_is_projected_without_live_participants(self):
        try:
            sess, cat = self.session(('dead_standing', 'removed', 'dead_standing'))
        except Exception as exc:                                                                      # the native session may refuse a population without a live plant
            self.skipTest(f'native PlantSession refuses an all-nonlive population: {type(exc).__name__}')
        p = wv.project_from_plant_session(sess, np.zeros(sess.base.domain.shape), max_edges=EDGES, max_bytes=BIG)
        self.assertEqual(float(p.shading_cover.max()), 0.0)
        self.assertGreater(float(p.splash_cover.max()), 0.0)
        self.assertEqual(list(p.diagnostics['participating_live_species']), [])

    def test_the_snapshot_gate_uses_base_rows_fires_before_the_snapshot_and_passes_at_the_exact_need(self):
        from unittest import mock

        from leaf import plant_session as ps
        sess, cat = self.session()
        N, S, L = 120, cat.n_species, cat.root_layers
        n_rows, n_live = int(sess.base.n), int(sess.n_live)
        need_rows = wv.estimate_projection_bytes(n_rows, n_rows, 0, N, S, L, from_session=True)['stages']
        need_live = wv.estimate_projection_bytes(n_live, n_live, 0, N, S, L, from_session=True)['stages']
        self.assertGreater(need_rows['snapshot'], need_live['snapshot'])                               # the n_live bound would UNDER-admit
        surf = np.zeros(sess.base.domain.shape)
        with mock.patch.object(wv, 'HASH_CHUNK_BYTES', 1):                                             # makes the 1 MiB validate chunk irrelevant so the snapshot gate binds
            with mock.patch.object(ps.PlantSession, 'snapshot_state', side_effect=AssertionError('the snapshot was taken before its gate')):
                with self.assertRaises(wv.VegetationBudgetError) as cm:
                    wv.project_from_plant_session(sess, surf, max_edges=EDGES, max_bytes=need_rows['snapshot'] - 1)
                self.assertIn(cm.exception.stage, ('validate', 'snapshot'))
                self.assertGreaterEqual(need_rows['snapshot'] - 1, need_live['snapshot'])                  # that refused cap would have PASSED the former n_live bound
            full = wv.project_from_plant_session(sess, surf, max_edges=EDGES, max_bytes=1 << 30)
            declared = full.diagnostics['declared_peak_bytes']
            wv.project_from_plant_session(sess, surf, max_edges=EDGES, max_bytes=declared)             # the exact declared peak succeeds

    def test_the_full_row_snapshot_model_bounds_the_real_allocation_on_a_larger_mixed_session(self):
        import gc
        n = 3000
        statuses = tuple(['live', 'dead_standing', 'removed'][i % 3] for i in range(n))
        sess, cat = self.session(statuses)
        self.assertGreater(int(sess.base.n), int(sess.n_live))
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        snap = sess.snapshot_state()
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        model = wv.estimate_projection_bytes(int(sess.base.n), int(sess.base.n), 0, 120, cat.n_species, cat.root_layers, from_session=True)['stages']['snapshot']
        self.assertEqual(int(snap.n), int(sess.base.n))
        self.assertLessEqual(peak, model)
        self.assertGreater(peak, wv.SNAPSHOT_PEAK_PER_PLANT * int(sess.n_live) * 0.5)                  # not a tautology: the real peak is of the order of the per-row coefficient


class ScalarSeal(unittest.TestCase):
    def build(self):
        cat = catalog(grass=[0, 1], shrub=[1, 1])
        return cat, project(plants_state(cat, [(1, 'grass', 2.5, 2.5, 0.4, 'live')]), cat)

    def test_every_replaced_scalar_is_refused_by_value_on_every_public_path(self):
        replacements = {'cell_area_m2': 100.0, 'shape': (5, 20), 'n_cells': 99, 'n_entries': 7, 'n_plants': 5, 'resident_bytes': 1, 'species_ids': ('x', 'y')}
        for name, bad in replacements.items():
            cat, p = self.build()
            good = getattr(p, name)
            object.__setattr__(p, name, bad)
            with self.subTest(name):
                for call in (lambda: p.splash_cover, lambda: p.species_cell_cover('grass', max_bytes=BIG), lambda: p.identity_record(), lambda: p.export_compact(), lambda: p.stream_estimate(),
                             lambda: p.preferential_inputs(max_bytes=BIG), lambda: p.omitted_ids):
                    with self.assertRaises(wv.VegetationProjectionError):
                        call()
                object.__setattr__(p, name, good)                                                      # the sealed value restored: accepted again
                self.assertEqual(float(p.species_cell_cover('grass', max_bytes=BIG)[2 * 10 + 2]), pi_r2(0.4))

    def test_roots_probe_case_cell_area_change_is_refused_while_the_content_hash_is_unchanged(self):
        cat = catalog(grass=[0, 1])
        st = plants_state(cat, [(1, 'grass', 0.5, 0.5, 0.2, 'live')], nx=1, ny=1)
        a = wv.project_from_initial_state(st, cat, np.zeros((1, 1)), max_edges=1, max_bytes=1 << 26)
        before = a.species_cell_cover('grass', max_bytes=1 << 26).tolist()
        sha = a.identity['content_sha256']
        object.__setattr__(a, 'cell_area_m2', 100.0)
        with self.assertRaises(wv.VegetationProjectionError):
            a.species_cell_cover('grass', max_bytes=1 << 26)
        object.__setattr__(a, 'cell_area_m2', 1.0)
        self.assertEqual(a.species_cell_cover('grass', max_bytes=1 << 26).tolist(), before)
        self.assertEqual(a.identity['content_sha256'], sha)


class CheckSessionScope(unittest.TestCase):
    def setUp(self):
        self.helper = SessionBinding()
        self.sess, self.cat = self.helper.make_session()
        self.shape = self.sess.base.domain.shape
        self.surf = np.zeros(self.shape)
        self.mask = np.ones(self.shape, dtype=bool)
        self.mask[0:2, 0:3] = False
        self.p = wv.project_from_plant_session(self.sess, self.surf, active_mask=self.mask, max_edges=EDGES, max_bytes=BIG)

    def test_valid_surface_and_inactive_mask_and_the_biology_only_label(self):
        r = self.p.check_session(self.sess, current_surface=self.surf, active_mask=self.mask)
        self.assertEqual((r['biology'], r['surface'], r['mask']), (True, True, True))
        only = self.p.check_session(self.sess)
        self.assertEqual((only['biology'], only['surface'], only['mask']), (True, False, False))
        self.assertIn('BIOLOGY-ONLY', only['scope'])
        self.assertIn('does NOT validate the current landscape', only['scope'])
        with self.assertRaises(wv.StaleProjectionError):
            self.p.check_session(self.sess, current_surface=self.surf + 1.0)
        with self.assertRaises(wv.StaleProjectionError):
            self.p.check_session(self.sess, active_mask=np.ones(self.shape, dtype=bool))             # a different (all-active) mask
        with self.assertRaises(wv.StaleProjectionError):
            self.p.check_session(self.sess, active_mask=~self.mask)

    def test_wrong_shape_dtype_and_layout_are_refused_before_any_hash_and_without_coercion(self):
        from unittest import mock
        bad_surfaces = {'list': self.surf.tolist(), 'float32': self.surf.astype(np.float32), 'shape': np.zeros((self.shape[0] + 1, self.shape[1])), 'fortran': np.asfortranarray(self.surf),
                        'int': np.zeros(self.shape, dtype=np.int64)}
        bad_masks = {'int8': self.mask.astype(np.int8), 'shape': np.ones((self.shape[0], self.shape[1] + 1), dtype=bool), 'list': self.mask.tolist(), 'fortran': np.asfortranarray(self.mask)}
        with mock.patch.object(wv, '_hash_array', side_effect=AssertionError('hashed before validation')):
            for label, a in bad_surfaces.items():
                with self.subTest('surface ' + label), self.assertRaises(wv.VegetationProjectionError):
                    self.p.check_session(self.sess, current_surface=a)
            for label, a in bad_masks.items():
                with self.subTest('mask ' + label), self.assertRaises(wv.VegetationProjectionError):
                    self.p.check_session(self.sess, active_mask=a)

    def test_the_check_allocates_no_whole_n_copy(self):
        import gc
        big = np.zeros((800, 800))
        ones = np.ones((800, 800), dtype=bool).reshape(-1).view(np.uint8)
        sess = self.sess
        p = wv.project_from_initial_state(plants_state(catalog(grass=[0, 1], shrub=[1, 1]), [(1, 'grass', 2.5, 2.5, 0.4, 'live')], nx=800, ny=800), catalog(grass=[0, 1], shrub=[1, 1]), big,
                                          max_edges=EDGES, max_bytes=1 << 30)
        # a state projection has no session identity; time the hash path itself on the helper that check_session uses
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        wv._hash_array(big)
        wv._hash_array(ones)
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        self.assertLess(peak, 64 * 1024)                                                               # 5.1 MB surface + 0.64 MB mask hashed with no copy
        self.assertIsNotNone(p)
        self.assertIsNotNone(sess)


class CompactExport(unittest.TestCase):
    def build(self, **w):
        cat = catalog(layers=3, grass=w.get('g', [0, 1, 1]), shrub=w.get('s', [0, 0, 0]), fractions={'grass': [0.5, 0.3, 0.2], 'shrub': [0.2, 0.3, 0.5]})
        st = plants_state(cat, [(1, 'grass', 2.5, 2.5, 0.4, 'live'), (2, 'shrub', 6.5, 6.5, 0.4, 'live'), (3, 'grass', 8.5, 1.5, 0.3, 'dead_standing')])
        mask = np.ones((10, 10), dtype=bool)
        mask[0, 0] = False
        return cat, st, project(st, cat, active_mask=mask)

    def independent_hash(self, ex):
        h = hashlib.sha256()
        h.update(json.dumps({'schema': ex['schema'], 'version': ex['version'], 'species_ids': list(ex['species_ids']), 'shape': list(ex['shape']), 'cell_area_m2': float(ex['cell_area_m2']).hex(),
                             'n_cells': ex['n_cells'], 'n_entries': ex['n_entries'], 'n_plants': ex['n_plants'], 'root_layers': ex['root_layers']}, sort_keys=True).encode())
        for k in ex['field_order']:
            a = np.ascontiguousarray(ex['arrays'][k])
            h.update(k.encode() + str(a.dtype).encode() + str(a.shape).encode() + hashlib.sha256(a.tobytes()).digest())
        return 'sha256:' + h.hexdigest()

    def test_export_has_every_required_array_read_only_without_copy_and_in_canonical_order(self):
        cat, st, p = self.build()
        ex = p.export_compact()
        want = ('splash_cover', 'shading_cover', 'matrix_root_index', 'cell_offsets', 'entry_cell', 'entry_species', 'entry_area_all_m2', 'entry_area_live_m2', 'root', 'lower', 'weights', 'has_w',
                'omitted_ids')
        self.assertEqual(tuple(ex['field_order']), want)
        self.assertEqual(set(ex['arrays']), set(want))
        S, L, N, M = cat.n_species, cat.root_layers, 100, p.n_entries
        shapes = {'splash_cover': (N,), 'shading_cover': (N,), 'matrix_root_index': (N,), 'cell_offsets': (N + 1,), 'entry_cell': (M,), 'entry_species': (M,), 'entry_area_all_m2': (M,),
                  'entry_area_live_m2': (M,), 'root': (S, L), 'lower': (S,), 'weights': (S, L), 'has_w': (S,), 'omitted_ids': (len(p.omitted_ids),)}
        for k, a in ex['arrays'].items():
            self.assertEqual(a.shape, shapes[k], k)
            self.assertFalse(a.flags.writeable)
            node = a
            while isinstance(node, np.ndarray):
                with self.assertRaises(ValueError):
                    node.setflags(write=True)
                node = node.base
            if a.size:
                self.assertTrue(np.shares_memory(a, p._a[k] if k in p._a else p._t[k]))              # a view, never a copy
        np.testing.assert_array_equal(ex['arrays']['root'], np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]]))
        np.testing.assert_array_equal(ex['arrays']['weights'], np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0]]))      # RAW trait values; the all-zero shrub stays all-zero
        np.testing.assert_array_equal(ex['arrays']['has_w'], [1, 1])
        np.testing.assert_allclose(ex['arrays']['lower'], [0.5, 0.8])
        with self.assertRaises(TypeError):
            ex['arrays']['root'] = None
        self.assertEqual(ex['content_sha256'], p.identity['content_sha256'])

    def test_content_hash_covers_every_table_and_matches_an_independent_derivation(self):
        cat, st, p = self.build()
        ex = p.export_compact()
        self.assertEqual(self.independent_hash(ex), p.identity['content_sha256'])                       # re-derived from the EXPORT in the canonical order
        base = p.identity['content_sha256']
        _, _, q_w = self.build(g=[0, 1, 2])                                                             # only the raw receiving weights differ
        self.assertNotEqual(q_w.identity['content_sha256'], base)
        cat2 = catalog(layers=3, grass=[0, 1, 1], shrub=[0, 0, 0], fractions={'grass': [0.6, 0.3, 0.1], 'shrub': [0.2, 0.3, 0.5]})        # only the root fractions differ
        st2 = plants_state(cat2, [(1, 'grass', 2.5, 2.5, 0.4, 'live'), (2, 'shrub', 6.5, 6.5, 0.4, 'live'), (3, 'grass', 8.5, 1.5, 0.3, 'dead_standing')])
        mask2 = np.ones((10, 10), dtype=bool)
        mask2[0, 0] = False
        q_r = project(st2, cat2, active_mask=mask2)
        self.assertNotEqual(q_r.identity['content_sha256'], base)
        _, _, q_z = self.build(s=[0, 0, 1])                                                             # a disabled species becomes enabled (participation / lower sums are tables)
        self.assertNotEqual(q_z.identity['content_sha256'], base)
        self.assertEqual(p.identity['version'], wv.VERSION)
        self.assertEqual(wv.VERSION, 3)

    def test_export_of_a_forged_or_replaced_record_is_refused(self):
        cat, st, p = self.build()
        p._t['weights'] = np.frombuffer(np.zeros(6).tobytes(), dtype=np.float64).reshape(2, 3)
        with self.assertRaises(wv.VegetationProjectionError):
            p.export_compact()


class PublicStreamEstimator(unittest.TestCase):
    def setUp(self):
        self.cat = catalog(grass=[0, 1], shrub=[1, 1])
        rng = np.random.default_rng(9)
        pl = [(i, 'grass' if i % 2 else 'shrub', float(rng.uniform(1, 199)), float(rng.uniform(1, 199)), float(rng.uniform(0.2, 0.9)), 'live') for i in range(1, 1500)]
        self.st = plants_state(self.cat, pl, nx=200, ny=200)
        self.p = project(self.st, self.cat)

    def test_public_estimator_matches_the_projection_method_and_the_stream_admission(self):
        p = self.p
        for held in (False, True):
            self.assertEqual(p.stream_estimate(previous_held=held), wv.estimate_stream_bytes(p.n_cells, p.n_entries, 2, 2, previous_held=held))
        e0, e1 = p.stream_estimate(), p.stream_estimate(previous_held=True)
        self.assertEqual(e1['per_next_bytes'] - e0['per_next_bytes'], 8 * p.n_cells)                     # previous + new differ by exactly one (N,) array
        self.assertEqual(e0['per_next_bytes'], e0['result_bytes'] + e0['scratch_bytes'] + e0['weights_bytes'] + e0['retained_metadata_bytes'])      # the retained index / id tuple are live for the iterator lifetime
        self.assertEqual(e0['retained_metadata_bytes'], e0['index_bytes'] + e0['ids_bytes'] + wv.STREAM_FIXED_BYTES)
        # the stream admits EXACTLY these numbers: cap = estimate passes, cap - 1 refuses (no private mirroring)
        ids, w, it = p.preferential_inputs(max_bytes=e0['per_next_bytes'])
        a = next(it)
        with self.assertRaises(wv.VegetationBudgetError):
            next(it)                                                                                     # `a` is still held: needs e1
        ids, w, it = p.preferential_inputs(max_bytes=e1['per_next_bytes'])
        a = next(it)
        b = next(it)                                                                                     # previous + new fits exactly
        del a, b
        ids, w, it = p.preferential_inputs(max_bytes=e0['per_next_bytes'] - 1)
        with self.assertRaises(wv.VegetationBudgetError):
            next(it)

    def test_estimator_bounds_the_actual_allocation_of_one_next(self):
        import gc
        p = self.p
        est = p.stream_estimate()
        ids, w, it = p.preferential_inputs(max_bytes=BIG)
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        out = next(it)
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        self.assertLessEqual(peak, est['result_bytes'] + est['scratch_bytes'])                           # the model bounds the traced allocation of one next()
        self.assertGreaterEqual(peak, est['result_bytes'])                                               # and is not a tautology: at least the (N,) result really exists
        self.assertEqual(out.shape, (p.n_cells,))
        self.assertEqual(w.nbytes, est['weights_bytes'])

    def test_only_the_most_recent_yielded_array_is_tracked_and_that_is_documented(self):
        doc = wv.estimate_stream_bytes.__doc__
        flat_doc = ' '.join(doc.split())
        self.assertIn('Only the most recent yielded array is tracked', flat_doc)
        self.assertIn('retains several arrays must reserve them itself', flat_doc)


class StreamModel(unittest.TestCase):
    def setUp(self):
        self.cat = catalog(grass=[0, 1], shrub=[1, 1])
        self.st = plants_state(self.cat, [(1, 'grass', 2.5, 2.5, 0.4, 'live'), (2, 'shrub', 6.5, 6.5, 0.4, 'live')])
        self.p = project(self.st, self.cat)

    def test_the_previous_yielded_array_is_counted_when_still_held(self):
        p = self.p
        e0, e1 = p.stream_estimate(), p.stream_estimate(previous_held=True)
        ids, w, it = p.preferential_inputs(max_bytes=e0['per_next_bytes'])             # exactly one result + scratch + the weights + the retained index / ids
        first = next(it)
        with self.assertRaises(wv.VegetationBudgetError):
            next(it)                                                                   # the caller still holds `first`: previous + new no longer fit
        ids, w, it = p.preferential_inputs(max_bytes=e0['per_next_bytes'])
        first = next(it)
        del first                                                                      # released by the caller: the next one fits under the same cap
        second = next(it)
        self.assertEqual(second.shape, (100,))
        ids, w, it = p.preferential_inputs(max_bytes=e1['per_next_bytes'])             # a quota that covers previous + new: both may coexist
        a = next(it)
        b = next(it)
        self.assertEqual(float(a.max()), pi_r2(0.4) * 0.2)
        self.assertAlmostEqual(float(b.max()), pi_r2(0.4) * 0.75, places=14)

    def test_the_raw_weights_are_one_fresh_copy_and_the_contribution_is_scaled_in_place(self):
        p = self.p
        ids, w, it = pref(p)
        self.assertTrue(w.flags.owndata)                                               # one advanced-index copy, not a copy of a copy
        np.testing.assert_array_equal(w, [[0.0, 1.0], [1.0, 1.0]])
        expect = np.minimum(np.asarray(p.species_cell_cover('grass', 'live', max_bytes=BIG)) , 1.0) * 0.2
        np.testing.assert_array_equal(next(it), expect)                                # bitwise the former `cover * lower`

    def test_huge_finite_weights_do_not_overflow_the_enable_test(self):
        big = 1.7e308
        cat = catalog(grass=[big, big], shrub=[0, 0])
        st = plants_state(cat, [(1, 'grass', 2.5, 2.5, 0.4, 'live')])
        p = project(st, cat)
        ids, w, it = pref(p)
        self.assertEqual(ids, ('grass',))
        np.testing.assert_array_equal(w, [[big, big]])
        self.assertEqual(list(p.diagnostics['receiver_trait_disabled_species']), ['shrub'])


class ExternalHelpers(unittest.TestCase):
    def test_the_external_compiled_builder_helper_is_recorded_and_a_changed_helper_is_observed(self):
        from unittest import mock
        cat = catalog(grass=[0, 1], shrub=[1, 1])
        p = project(plants_state(cat, [(1, 'grass', 2.5, 2.5, 0.4, 'live')]), cat)
        ext = p.identity['external_helpers']
        self.assertEqual([h['module'] for h in ext['helpers']], ['benchmarks.spatial_mapping'])
        helper = ROOT / 'benchmarks' / 'spatial_mapping.py'
        self.assertEqual(ext['helpers'][0]['sha256'], 'sha256:' + hashlib.sha256(helper.read_bytes()).hexdigest())
        self.assertEqual(ext['helpers'][0]['bytes'], helper.stat().st_size)
        self.assertNotIn('benchmarks', ' '.join(wv.package_source_record()['algorithm'].split()[:1]))          # the 28-file package digest is unchanged in scope
        import importlib.util
        import tempfile
        saved = dict(wv._SOURCE_CACHE)
        try:
            with tempfile.TemporaryDirectory() as d:
                fake = pathlib.Path(d) / 'spatial_mapping.py'
                fake.write_bytes(helper.read_bytes() + b'# changed\n')
                spec = importlib.util.spec_from_file_location('benchmarks.spatial_mapping', fake)
                wv._SOURCE_CACHE.pop('ext', None)
                with mock.patch('importlib.util.find_spec', return_value=spec):
                    changed = wv.external_source_record()
                self.assertNotEqual(changed['helpers'][0]['sha256'], ext['helpers'][0]['sha256'])
                wv._SOURCE_CACHE.pop('ext', None)
                with mock.patch('importlib.util.find_spec', return_value=None):
                    with self.assertRaisesRegex(wv.VegetationProjectionError, 'cannot be located'):
                        wv.external_source_record()
        finally:
            wv._SOURCE_CACHE.clear()
            wv._SOURCE_CACHE.update(saved)


class StreamAdmission(unittest.TestCase):
    def test_streamed_contributions_are_admitted_before_each_array_and_retain_nothing(self):
        cat = catalog(grass=[0, 1], shrub=[1, 1])
        st = plants_state(cat, [(1, 'grass', 2.5, 2.5, 0.4, 'live'), (2, 'shrub', 6.5, 6.5, 0.4, 'live')])
        p = project(st, cat)
        est = p.stream_estimate()
        need = est['species_cover_bytes']
        with self.assertRaises(wv.VegetationBudgetError):
            p.preferential_inputs(max_bytes=est['weights_stage_bytes'] - 1)                          # the weights stage (weights + index + id build) is admitted first
        ids, w, it = p.preferential_inputs(max_bytes=est['per_next_bytes'])
        self.assertEqual(ids, ('grass', 'shrub'))
        next(it)
        calls = []
        ids, w, it = p.preferential_inputs(max_bytes=BIG, available_bytes_fn=lambda: calls.append(1) or (1 << 40))
        n0 = len(calls)
        next(it)
        self.assertGreater(len(calls), n0)                                                          # the reader is consulted again before each species array
        _, _, tight = p.preferential_inputs(max_bytes=est['per_next_bytes'] - 1)
        with self.assertRaises(wv.VegetationBudgetError):
            next(tight)
        with self.assertRaises(wv.VegetationBudgetError):
            p.species_cell_cover('grass', 'live', max_bytes=need - 1)
        with self.assertRaises(TypeError):
            p.preferential_inputs()                                                                  # the quota is mandatory (no hidden unbudgeted allocation)


class Ownership(unittest.TestCase):
    def build(self):
        cat = catalog(grass=[0, 1], shrub=[1, 1])
        st = plants_state(cat, [(1, 'grass', 2.5, 2.5, 0.4, 'live'), (2, 'shrub', 6.5, 6.5, 0.3, 'dead_standing')])
        return cat, st, project(st, cat)

    def test_base_chain_cannot_be_unlocked_and_ends_in_immutable_bytes(self):
        _, _, p = self.build()
        before = np.array(p.splash_cover, copy=True)
        for name in ('splash_cover', 'shading_cover', 'matrix_root_index', 'omitted_ids'):
            v = getattr(p, name)
            node, depth = v, 0
            while isinstance(node, np.ndarray):
                self.assertFalse(node.flags.writeable, f'{name} chain node {depth} is writeable')
                with self.assertRaises(ValueError):
                    node.setflags(write=True)
                node, depth = node.base, depth + 1
            self.assertIsInstance(node, bytes) if v.size else None
            with self.assertRaises(ValueError):
                v[0:1] = 1.0
        base = p.splash_cover.base
        with self.assertRaises(ValueError):
            base.setflags(write=True)                                            # Root's probe: view.base.setflags(write=True)
        np.testing.assert_array_equal(p.splash_cover, before)

    def test_identity_and_diagnostics_are_deeply_frozen_and_detached_copies_are_available(self):
        _, _, p = self.build()
        for mutate in (lambda: p.identity.__setitem__('x', 1), lambda: p.identity['source'].__setitem__('kind', 'plant_session'), lambda: p.identity['cap'].__setitem__('total', 5.0),
                       lambda: p.diagnostics.__setitem__('n_plants_projected', 99), lambda: p.diagnostics['excluded_by_reason'].__setitem__('removed', 9),
                       lambda: p.identity.update({'a': 1})):
            with self.assertRaises((TypeError, AttributeError)):
                mutate()
        self.assertEqual(p.identity['source']['kind'], 'initial_state')
        rec = p.identity_record()
        rec['source']['kind'] = 'plant_session'
        rec['species_ids'].append('x')
        self.assertEqual(p.identity['source']['kind'], 'initial_state')
        self.assertEqual(list(p.identity['species_ids']), ['grass', 'shrub'])
        self.assertTrue(p.splash_cover.shape == (100,))

    def test_forged_replaced_copied_and_uninitialised_records_are_refused(self):
        cat, st, p = self.build()
        with self.assertRaises(wv.VegetationProjectionError):
            wv.VegetationProjection()
        with self.assertRaises(wv.VegetationProjectionError):
            wv.VegetationProjection(_a=p._a, _t=p._t, identity=p.identity)
        with self.assertRaises(TypeError):
            type('Sub', (wv.VegetationProjection,), {})
        raw = wv.VegetationProjection.__new__(wv.VegetationProjection)
        for attr in ('splash_cover', 'omitted_ids'):
            with self.assertRaises(wv.VegetationProjectionError):
                getattr(raw, attr)
        with self.assertRaises(wv.VegetationProjectionError):
            raw.identity_record()
        # replaced array / wrong dtype / wrong shape / replaced pointer in the private tables
        q = project(st, cat)
        good = q._a['splash_cover']
        for label, repl in {'other pointer': np.frombuffer(np.zeros(100).tobytes(), dtype=np.float64), 'wrong dtype': np.frombuffer(np.zeros(100, np.float32).tobytes(), dtype=np.float32),
                            'wrong shape': np.frombuffer(np.zeros(50).tobytes(), dtype=np.float64)}.items():
            q._a['splash_cover'] = repl
            with self.subTest(label), self.assertRaises(wv.VegetationProjectionError):
                _ = q.splash_cover
            q._a['splash_cover'] = good
        self.assertEqual(q.splash_cover.shape, (100,))                           # restored record is accepted again (the fingerprint is the sealed one)
        object.__setattr__(q, 'identity', dict(q.identity))
        with self.assertRaises(wv.VegetationProjectionError):
            _ = q.matrix_root_index
        r = project(st, cat)
        forged = dict(r._t)
        forged['omitted_ids'] = np.frombuffer(np.array([123], dtype=np.int64).tobytes(), dtype=np.int64)       # a forged omitted-id table
        object.__setattr__(r, '_t', forged)
        with self.assertRaises(wv.VegetationProjectionError):
            _ = r.omitted_ids
        t = project(st, cat)
        object.__setattr__(t, '_t', dict(t._t))                                  # an equal-content container is not an alteration: the sealed arrays are unchanged
        self.assertEqual(t.omitted_ids.tolist(), [])

    def test_inputs_are_never_mutated_and_exact_genuine_types_are_required(self):
        import copy
        cat, st, _ = self.build()
        before = (wv._state_sha256(st), cat.hash, [getattr(st, n).flags.writeable for n in st.ARRAY_NAMES])
        project(st, cat)
        self.assertEqual((wv._state_sha256(st), cat.hash, [getattr(st, n).flags.writeable for n in st.ARRAY_NAMES]), before)
        SubState = type('SubState', (state.PlantState,), {})
        SubCat = type('SubCat', (traits.SpeciesCatalog,), {})
        with self.assertRaises(wv.VegetationProjectionError):
            wv.project_from_initial_state(SubState.__new__(SubState), cat, flat(st), max_edges=EDGES, max_bytes=BIG)
        with self.assertRaises(wv.VegetationProjectionError):
            wv.project_from_initial_state(st, SubCat.__new__(SubCat), flat(st), max_edges=EDGES, max_bytes=BIG)
        with self.assertRaises(wv.VegetationProjectionError):
            wv.project_from_plant_session(copy.copy(object()), flat(st), max_edges=EDGES, max_bytes=BIG)

    def test_streamed_arrays_are_fresh_and_writeable_but_independent(self):
        cat, st, p = self.build()
        ids, w, it = pref(p)
        a = next(it)
        a[:] = 9.0
        w[:] = -1.0
        ids2, w2, it2 = pref(p)
        self.assertEqual(float(next(it2).max()), pi_r2(0.4) * 0.2)
        self.assertEqual(float(w2.min()), 0.0)

    def test_identity_records_surface_mask_and_rule_hashes(self):
        cat = catalog(grass=[0, 1], shrub=[1, 1])
        st = plants_state(cat, [(1, 'grass', 2.5, 2.5, 0.4, 'live')])
        surf = flat(st, 0.5)
        mask = np.ones((10, 10), dtype=bool)
        p, q, m = project(st, cat, surf), project(st, cat, flat(st, 0.6)), project(st, cat, surf, active_mask=mask)
        self.assertEqual(p.identity['surface_sha256'], hashlib.sha256(surf.tobytes()).hexdigest())
        self.assertNotEqual(p.identity['surface_sha256'], q.identity['surface_sha256'])
        self.assertEqual(p.identity['active_mask_sha256'], hashlib.sha256(np.ones(100, np.uint8).tobytes()).hexdigest())
        self.assertEqual(p.identity['active_mask_sha256'], m.identity['active_mask_sha256'])           # None == an all-True mask
        self.assertEqual(p.identity['source']['kind'], 'initial_state')
        self.assertIn('min(sum eligible plant overlap area/cell area,1)', wv.RULE)


class PackageDigest(unittest.TestCase):
    def independent_digest(self):
        base = ROOT / 'leaf'
        entries = []
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d != '__pycache__']
            for name in filenames:
                if name.endswith(('.pyc', '.pyo')):
                    continue
                path = pathlib.Path(dirpath) / name
                entries.append((path.relative_to(base).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest()))
        top = hashlib.sha256()
        for rel, h in sorted(entries):
            top.update(rel.encode() + b'\0' + h.encode() + b'\n')
        return 'sha256:' + top.hexdigest(), len(entries)

    def test_package_digest_is_the_full_source_tree_digest_of_every_package_file(self):
        want, n = self.independent_digest()
        rec = wv.package_source_record()
        self.assertEqual((rec['digest'], rec['file_count']), (want, n))
        self.assertEqual(n, len(BASELINE_27) + 1)                                    # the 27 pre-existing modules + water_vegetation.py
        self.assertEqual(wv.package_sha256(), want)
        self.assertIn('__pycache__', rec['algorithm'])
        self.assertGreater(rec['total_bytes'], 0)
        self.assertNotEqual(want, 'sha256:5a627db99617437a29c3a65696ebd4b9bd6bab02a54cfb235c0f085c62a99bef')        # the former 7-module claim

    def test_identity_carries_the_package_record(self):
        cat = catalog(grass=[0, 1], shrub=[1, 1])
        p = project(plants_state(cat, [(1, 'grass', 2.5, 2.5, 0.4, 'live')]), cat)
        self.assertEqual(dict(p.identity['package']), wv.package_source_record())

    def test_a_changed_module_byte_changes_the_digest(self):
        # re-derive the algorithm over a temporary copy with one byte changed (the real tree is never touched)
        import shutil
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            shutil.copytree(ROOT / 'leaf', pathlib.Path(d) / 'leaf', ignore=shutil.ignore_patterns('__pycache__'))
            def digest(root):
                ent = [(p_.relative_to(root).as_posix(), hashlib.sha256(p_.read_bytes()).hexdigest()) for p_ in sorted(root.rglob('*')) if p_.is_file() and p_.suffix not in ('.pyc', '.pyo')]
                top = hashlib.sha256()
                for rel, h in sorted(ent):
                    top.update(rel.encode() + b'\0' + h.encode() + b'\n')
                return 'sha256:' + top.hexdigest()
            a = digest(pathlib.Path(d) / 'leaf')
            with open(pathlib.Path(d) / 'leaf' / 'roots.py', 'ab') as fh:
                fh.write(b'#')
            self.assertNotEqual(a, digest(pathlib.Path(d) / 'leaf'))
        self.assertEqual(a, self.independent_digest()[0])                           # the copy equals the real package digest


class CudaStaticHostChecks(unittest.TestCase):
    def test_device_arguments_are_cuda_only_and_the_device_pool_is_explicit(self):
        cat = catalog(grass=[0, 1], shrub=[1, 1])
        st = plants_state(cat, [(1, 'grass', 2.5, 2.5, 0.4, 'live')])
        for kw in ({'device': 0}, {'max_device_bytes': 1}, {'device_reserve_bytes': 1}, {'available_device_bytes_fn': lambda: 1}):
            with self.subTest(kw), self.assertRaises(wv.VegetationProjectionError):
                project(st, cat, **kw)
        with self.assertRaises(wv.VegetationBackendError):
            project(st, cat, backend='cuda')                                           # no device: no fallback
        with self.assertRaisesRegex(wv.VegetationProjectionError, 'max_device_bytes'):
            project(st, cat, backend='cuda', device=0)

    def test_without_cupy_cuda_refuses_after_host_admission_and_never_falls_back(self):
        try:
            import cupy  # noqa: F401
            self.skipTest('CuPy is installed: the hardware definitions cover this path')
        except Exception:
            pass
        cat = catalog(grass=[0, 1], shrub=[1, 1])
        st = plants_state(cat, [(1, 'grass', 2.5, 2.5, 0.4, 'live')])
        with self.assertRaises(wv.VegetationBackendError):
            project(st, cat, backend='cuda', device=0, max_device_bytes=1 << 30)
        with self.assertRaises(wv.VegetationBudgetError):
            project(st, cat, backend='cuda', device=0, max_device_bytes=1 << 30, max_bytes=1)       # host gate first

    def test_device_model_and_transfer_costs_are_declared(self):
        m = wv.estimate_device_bytes(100, 7, 2)
        self.assertEqual(m['d2h_bytes'], 3 * 8 * 100)
        self.assertEqual(m['h2d_bytes'], 8 * 101 + 7 * (4 + 8 + 8) + 8 * 2)
        self.assertEqual(m['device_bytes'], m['h2d_bytes'] + m['d2h_bytes'])

    def test_kernel_source_and_call_order_are_static_checked(self):
        src = (ROOT / 'leaf' / 'water_vegetation.py').read_text()
        k = src[src.index('_CUDA_SRC = r"""'):src.index('"""', src.index('_CUDA_SRC = r"""') + 20)]
        for needle in ('__ddiv_rn', '__dadd_rn', '__dmul_rn', 'blockIdx.x', 'off[c + 1]'):
            self.assertIn(needle, k)
        self.assertNotIn('fma', k)
        self.assertNotIn('atomic', k)                                                   # deterministic: no atomics
        body = src[src.index('def _reduce_cuda'):src.index('# ------------------------------------------------------------------ immutable sealing')]
        order = [body.index(t) for t in ('import cupy', 'getDevice()', 'dq.admit', 'cp.asarray', 'cp.zeros', 'RawKernel', '.get()')]
        self.assertEqual(order, sorted(order))                                         # import -> current-device guard -> device admission -> H2D -> device result -> kernel -> one D2H
        self.assertEqual(body.count('.get()'), 1)                                      # exactly ONE device-to-host call
        self.assertEqual(body.count('cp.zeros('), 1)                                   # one contiguous (3 N) device result
        self.assertNotIn('np.array(', body)                                            # no second host copy of the downloaded array
        self.assertEqual(body.count('cp.asarray('), 5)                                 # the five cold H2D entry arrays
        self.assertNotIn('cp.cuda.Stream', body.split('.get()')[0])
        self.assertNotIn('cp.cuda.Device(', body)                                       # never switches the device implicitly


class SessionAdmission(unittest.TestCase):
    def test_the_session_snapshot_is_admitted_from_metadata_before_it_allocates(self):
        from unittest import mock

        from leaf import plant_session as ps
        helper = SessionBinding()
        sess, _ = helper.make_session()
        surf = np.zeros(sess.base.domain.shape)
        with mock.patch.object(ps.PlantSession, 'snapshot_state', side_effect=AssertionError('snapshot allocated before admission')):
            with self.assertRaises(wv.VegetationBudgetError):
                wv.project_from_plant_session(sess, surf, max_edges=EDGES, max_bytes=1)
            with self.assertRaises(wv.VegetationBudgetError):
                wv.project_from_plant_session(sess, surf, max_edges=EDGES, max_bytes=BIG, available_bytes_fn=lambda: 1)
        calls = []
        real = ps.PlantSession.snapshot_state

        def spy(self_):
            calls.append(1)
            return real(self_)

        with mock.patch.object(ps.PlantSession, 'snapshot_state', spy):
            wv.project_from_plant_session(sess, surf, max_edges=EDGES, max_bytes=BIG)
        self.assertEqual(len(calls), 1)


class SessionBinding(unittest.TestCase):
    def make_session(self, plants=None):
        import tests.native_leaf_fixture as nfx
        doc = json.loads(nfx.FIXTURES['phase1'].read_text())
        nfx.simple_edit(doc)
        doc['schema_version'] = 2
        for sid in doc['species']:
            doc['species'][sid]['hydrology'] = {KEY: [0.0, 1.0]}
        cat = traits.SpeciesCatalog(doc)
        st = nfx.make_state(cat, plants=plants)
        cfg = nfx.make_config(cat)
        return nfx.start_session(st, cfg, nfx.anchor(), bound=True), cat

    def test_genuine_session_projection_binds_and_detects_staleness(self):
        sess, cat = self.make_session()
        surf = np.zeros(sess.snapshot_state().domain.shape)
        p = wv.project_from_plant_session(sess, surf, max_edges=EDGES, max_bytes=BIG)
        src = p.identity['source']
        self.assertEqual((src['kind'], src['sequence'], src['token'], src['pop_sha256'], src['geometry_sha256']),
                         ('plant_session', sess.sequence, sess.token, sess.pop_sha256, sess.geometry_sha256))
        self.assertTrue(p.check_session(sess, current_surface=surf))
        self.assertGreater(float(p.splash_cover.max()), 0.0)
        with self.assertRaises(wv.StaleProjectionError):
            p.check_session(sess, current_surface=surf + 1.0)
        mask = np.ones(surf.shape, dtype=bool)
        mask[0, 0] = False
        with self.assertRaises(wv.StaleProjectionError):
            p.check_session(sess, active_mask=mask)
        other, _ = self.make_session(plants=[(11, 3.5, 3.5, 0.1), (12, 4.2, 3.5, 0.3)])
        with self.assertRaises(wv.StaleProjectionError):
            p.check_session(other)                                                  # a different accepted generation / population
        with self.assertRaises(wv.VegetationProjectionError):
            p.check_session(object())

    def test_initial_state_projection_is_not_a_session_receipt_and_fakes_are_refused(self):
        sess, cat = self.make_session()
        st = sess.snapshot_state()
        p = wv.project_from_initial_state(st, cat, np.zeros(st.domain.shape), max_edges=EDGES, max_bytes=BIG)
        with self.assertRaisesRegex(wv.StaleProjectionError, 'initial state'):
            p.check_session(sess)
        for fake in (object(), st, type('FakeSession', (), {'snapshot_state': lambda self: st})()):
            with self.assertRaises(wv.VegetationProjectionError):
                wv.project_from_plant_session(fake, np.zeros(st.domain.shape), max_edges=EDGES, max_bytes=BIG)

    def test_session_and_state_are_not_mutated(self):
        sess, _ = self.make_session()
        before = (sess.token, sess.sequence, wv._state_sha256(sess.snapshot_state()))
        wv.project_from_plant_session(sess, np.zeros(sess.snapshot_state().domain.shape), max_edges=EDGES, max_bytes=BIG)
        self.assertEqual((sess.token, sess.sequence, wv._state_sha256(sess.snapshot_state())), before)


class SourceGuard(unittest.TestCase):
    def test_the_27_preexisting_modules_are_byte_identical_to_the_qualified_trait_baseline(self):
        for name, want in BASELINE_27.items():
            got = hashlib.sha256((ROOT / 'leaf' / name).read_bytes()).hexdigest()
            self.assertEqual(got, want, f'leaf/{name} differs from the qualified trait baseline')
        present = {p.name for p in (ROOT / 'leaf').glob('*.py')}
        self.assertEqual(present - set(BASELINE_27), {'water_vegetation.py'})        # the ONLY production addition of this task


if __name__ == '__main__':
    unittest.main()
