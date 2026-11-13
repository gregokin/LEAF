'''Tests for the per-axis PERIODIC boundary support (leaf.state.Domain.periodic, leaf.periodic, canopy cover, exclusion placement,
canopy-area-weighted PET, root footprints, session / checkpoint propagation, recruitment daughters). NOT RUN by the author (file tools
only): Codex records the real runs. Expectations are brute-force / independent oracles (tiled image rasters, minimum-image distances,
the existing exact circle-rectangle integrator on explicitly shifted rectangles, a fine midpoint grid for self-overlapping canopies),
not re-statements of the implementation.

Run from the candidate root:  python -m unittest tests.test_periodic_boundaries -v
The compiled-kernel / numba-backend parity tests SKIP only when numba is unavailable, the CUDA tests only when no CuPy device exists, and
the real native N15 test only when LEAF_NATIVE_BUNDLE / LEAF_NATIVE_CASE_DIR are not configured (a configured but unusable native source
is a failure, never a skip).
'''
from __future__ import annotations

import copy
import dataclasses
import hashlib
import itertools
import json
import math
import os
import pathlib
import shutil
import sys
import tempfile
import tracemalloc
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from leaf import biocalendar, canopy, initialization as init, periodic, plant_checkpoint as pc, plant_pet as pp  # noqa: E402
from leaf import plant_session as ps, recruitment as rec, roots, state, traits, uptake  # noqa: E402
import native_leaf_fixture as nf  # noqa: E402

try:
    import numba  # noqa: F401
    HAVE_NUMBA = True
except Exception:   # pragma: no cover
    HAVE_NUMBA = False


def _have_cuda():
    '''Runtime probe (CuPy is imported ONLY inside the CUDA test, never at module import: the genuine CPU native source guard refuses a
    process that imported CuPy, and the native test class runs before the CUDA test in alphabetical class order).'''
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


SPECIES0 = ROOT / 'examples' / 'phase0' / 'species.json'
PHASE3 = ROOT / 'examples' / 'phase3' / 'species.json'
BIG = 1 << 26
TT, TF, FT, FF = (True, True), (True, False), (False, True), (False, False)


def cat0():
    return traits.SpeciesCatalog.from_file(SPECIES0)


def rec_row(pid, sp, x, y, B, status='live'):
    return {'plant_id': pid, 'species': sp, 'x_m': x, 'y_m': y, 'establishment_ground_m': 0.0, 'agb_dry_kg': B,
            'seasonal_agb_dry_kg': 0.0, 'age_days': 100.0, 'status': status}


# ------------------------------------------------------------------ independent oracles
def reached_cells(dom, px, py, r):
    '''Brute force: cells whose closed square is positively overlapped (closest point STRICTLY inside) by the circle OR ANY of its
    periodic images (explicit shifts), clipped only on nonperiodic axes. px, py are domain-local.'''
    nx, ny, dx = dom.nx, dom.ny, dom.dx_m
    lx, ly = nx * dx, ny * dx
    kx_n = int(r // lx) + 2 if dom.periodic[0] else 0
    ky_n = int(r // ly) + 2 if dom.periodic[1] else 0
    out = []
    for j in range(ny):
        for i in range(nx):
            best = math.inf
            for kx in range(-kx_n, kx_n + 1):
                for ky in range(-ky_n, ky_n + 1):
                    qx, qy = px + kx * lx, py + ky * ly
                    cx = min(max(qx, i * dx), (i + 1) * dx)
                    cy = min(max(qy, j * dx), (j + 1) * dx)
                    best = min(best, (qx - cx) ** 2 + (qy - cy) ** 2)
            if best < r * r:
                out.append(j * nx + i)
    return out


def image_fragment_areas(dom, px, py, r):
    '''{cell: area} as the SUM over explicit image shifts of the existing exact circle-rectangle integrator (valid when the circle does
    not overlap its own images, i.e. 2 r <= every periodic span). Nonperiodic axes use the unshifted circle only (domain clipping).'''
    nx, ny, dx = dom.nx, dom.ny, dom.dx_m
    lx, ly = nx * dx, ny * dx
    kxs = (-1, 0, 1) if dom.periodic[0] else (0,)
    kys = (-1, 0, 1) if dom.periodic[1] else (0,)
    out = {}
    for j in range(ny):
        for i in range(nx):
            tot = 0.0
            for kx in kxs:
                for ky in kys:
                    qx, qy = px + kx * lx, py + ky * ly
                    tot += pp.circle_rectangle_area(r, i * dx - qx, (i + 1) * dx - qx, j * dx - qy, (j + 1) * dx - qy)
            if tot > 1e-14:
                out[j * nx + i] = tot
    return out


def grid_union_areas(dom, px, py, r, m=120):
    '''{cell: area} of the TORUS UNION of the circle with all its images by a fine midpoint grid per cell (minimum-image distance);
    independent of the kernels, used for self-overlapping canopies (error ~ boundary length / m).'''
    nx, ny, dx = dom.nx, dom.ny, dom.dx_m
    lx, ly = nx * dx, ny * dx
    sub = (np.arange(m) + 0.5) / m * dx
    out = {}
    for j in range(ny):
        dyv = j * dx + sub - py
        if dom.periodic[1]:
            dyv = dyv - ly * np.rint(dyv / ly)
        for i in range(nx):
            dxv = i * dx + sub - px
            if dom.periodic[0]:
                dxv = dxv - lx * np.rint(dxv / lx)
            cov = (dxv[None, :] ** 2 + dyv[:, None] ** 2) < r * r
            a = float(cov.mean()) * dx * dx
            if a > 0.0:
                out[j * nx + i] = a
    return out


def tiled_cover(dom, x, y, r, s, group=None, kmax=2):
    '''Independent midpoint-raster union cover by explicit image TILES (periodic axes) of every circle; returns the covered fraction.'''
    NX, NY = dom.nx * s, dom.ny * s
    hx, hy = dom.dx_m / s, dom.dy_m / s
    mx = dom.origin_x_m + (np.arange(NX) + 0.5) * hx
    my = dom.origin_y_m + (np.arange(NY) + 0.5) * hy
    lx, ly = dom.width_m, dom.height_m
    kxs = range(-kmax, kmax + 1) if dom.periodic[0] else (0,)
    kys = range(-kmax, kmax + 1) if dom.periodic[1] else (0,)
    cov = np.zeros((NY, NX), dtype=bool)
    for p in range(len(x)):
        if group is not None and not group[p]:
            continue
        for kx in kxs:
            for ky in kys:
                cx, cy = x[p] + kx * lx, y[p] + ky * ly
                cov |= ((mx[None, :] - cx) ** 2 + (my[:, None] - cy) ** 2) < r[p] ** 2
    return float(cov.mean())


def expected_root_rows(st, cat, overrides=None):
    '''{plant_id: {resource: weight}} by brute force: layer fractions x uniform weight over the UNIQUE reached cells.'''
    dom = st.domain
    geo = canopy.derive_geometry(st, cat)
    table = cat.root_layer_table()
    out = {}
    for k in range(st.n):
        if int(st.status[k]) != state.LIVE:
            continue
        pid = int(st.ids[k])
        frac = np.asarray(overrides[pid]) if overrides and pid in overrides else table[int(st.species_index[k])]
        r = float(geo['root_radius_m'][k])
        cells = reached_cells(dom, float(st.x_m[k]) - dom.origin_x_m, float(st.y_m[k]) - dom.origin_y_m, r)
        row = {}
        for layer in range(cat.root_layers):
            if frac[layer] > 0.0:
                for c in cells:
                    row[layer * dom.ncells + c] = float(frac[layer]) / len(cells)
        out[pid] = row
    return out


def map_rows(m):
    '''{plant_id: {resource: weight}} of a RootMap; asserts every (plant, resource) pair occurs exactly once.'''
    h = m.host_arrays()
    out = {}
    for p in range(m.n_plants):
        res = h['resources'][h['offsets'][p]:h['offsets'][p + 1]].tolist()
        w = h['weights'][h['offsets'][p]:h['offsets'][p + 1]].tolist()
        assert len(set(res)) == len(res), f'duplicate plant-resource pair in row {p}'
        out[int(h['ids'][p])] = dict(zip(res, w))
    return out


class Case(unittest.TestCase):
    def assertRowsEqual(self, got, want, rtol=1e-12):
        self.assertEqual(sorted(got), sorted(want))
        for pid in want:
            self.assertEqual(sorted(got[pid]), sorted(want[pid]), msg=f'plant {pid}: resource sets differ')
            for res, w in want[pid].items():
                self.assertAlmostEqual(got[pid][res], w, delta=rtol * max(w, 1e-300), msg=f'plant {pid} resource {res}')


# ------------------------------------------------------------------ 1. Domain declaration, serialisation, sampling
class DomainTests(Case):
    LEGACY = {'nx': 4, 'ny': 3, 'dx_m': 0.5, 'dy_m': 0.5, 'origin_x_m': 10.0, 'origin_y_m': -2.0,
              'axes': '(ny, nx): row = y north, col = x east', 'cell_sampling': 'containing_cell_closed_upper',
              'edges': 'finite closed domain, non-periodic',
              'coordinate_frame': 'landscape frame (same frame as origin_x_m/origin_y_m; NOT origin-relative)'}

    def test_default_domain_is_nonperiodic_and_serialises_exactly_as_before(self):
        d = state.Domain(4, 3, 0.5, origin_x_m=10.0, origin_y_m=-2.0)
        self.assertEqual(d.periodic, (False, False))
        self.assertFalse(d.any_periodic)
        self.assertEqual(d.to_dict(), self.LEGACY)                     # no new key, legacy texts
        self.assertNotIn('periodic', d.to_dict())
        self.assertNotIn('periodic', repr(d))
        self.assertEqual(state.Domain.from_dict(self.LEGACY), d)
        self.assertEqual(state.Domain.from_dict({'nx': 4, 'ny': 3, 'dx_m': 0.5, 'origin_x_m': 10.0, 'origin_y_m': -2.0}), d)
        # an explicit [false, false] is the same domain and serialises without the key
        e = state.Domain.from_dict(dict(self.LEGACY, periodic=[False, False]))
        self.assertEqual(e, d)
        self.assertEqual(e.to_dict(), self.LEGACY)

    def test_periodic_dict_roundtrip_equality_hash_and_repr(self):
        d0 = state.Domain(4, 3, 0.5)
        px, py, pxy = state.Domain(4, 3, 0.5, periodic=TF), state.Domain(4, 3, 0.5, periodic=FT), state.Domain(4, 3, 0.5, periodic=TT)
        for d in (px, py, pxy):
            dd = d.to_dict()
            self.assertEqual(dd['periodic'], [d.periodic[0], d.periodic[1]])
            self.assertNotEqual(dd['edges'], 'finite closed domain, non-periodic')
            self.assertNotEqual(dd['cell_sampling'], 'containing_cell_closed_upper')
            self.assertEqual(state.Domain.from_dict(json.loads(json.dumps(dd))), d)      # JSON round trip
            self.assertIn('periodic', repr(d))
        self.assertEqual(px.to_dict()['edges'], 'periodic (torus) in x; finite closed non-periodic in y')
        self.assertEqual(pxy.to_dict()['edges'], 'periodic (torus) in x+y')
        self.assertEqual(len({d0, px, py, pxy}), 4)
        self.assertNotEqual(d0, pxy)
        self.assertNotEqual(px, py)
        self.assertEqual(pxy, state.Domain(4, 3, 0.5, periodic=(np.True_, np.True_)))
        self.assertEqual(hash(pxy), hash(state.Domain(4, 3, 0.5, periodic=[True, True])))
        self.assertNotEqual(hash(d0), hash(pxy))
        self.assertEqual({pxy: 1}[state.Domain(4, 3, 0.5, periodic=TT)], 1)

    def test_strict_boolean_declaration(self):
        for bad in ((1, 0), [1, 0], ('x', 'y'), None, True, [True], [True, False, True], 'tt', (None, False), (0.0, 1.0),
                    {'x': True, 'y': False}, np.array([True, False])):
            with self.subTest(bad=bad), self.assertRaises(state.StateError):
                state.Domain(4, 3, 0.5, periodic=bad)
        for bad in ([1, 0], 'yes', [True], [True, 'no'], None, 1, {'x': True}):
            with self.subTest(dict_bad=bad), self.assertRaises(state.StateError):
                state.Domain.from_dict({'nx': 4, 'ny': 3, 'dx_m': 0.5, 'periodic': bad})

    def test_conflicting_legacy_edge_metadata_is_refused(self):
        base = {'nx': 4, 'ny': 3, 'dx_m': 0.5}
        with self.assertRaises(state.StateError):                      # an old "periodic" edge text WITHOUT a declaration
            state.Domain.from_dict(dict(base, edges='periodic'))
        with self.assertRaises(state.StateError):
            state.Domain.from_dict(dict(base, edges='periodic (torus) in x+y'))
        with self.assertRaises(state.StateError):                      # declaration contradicting the finite text
            state.Domain.from_dict(dict(base, periodic=[True, False], edges='finite closed domain, non-periodic'))
        with self.assertRaises(state.StateError):
            state.Domain.from_dict(dict(base, periodic=[True, True], cell_sampling='containing_cell_closed_upper'))
        with self.assertRaises(state.StateError):                      # periodic text of the OTHER axis
            state.Domain.from_dict(dict(base, periodic=[True, False], edges='periodic (torus) in y; finite closed non-periodic in x'))
        ok = state.Domain.from_dict(dict(base, periodic=[True, False], edges='periodic (torus) in x; finite closed non-periodic in y',
                                         cell_sampling=state.CELL_SAMPLING_PERIODIC))
        self.assertEqual(ok.periodic, (True, False))

    def test_cell_sampling_origin_offset_mixed_axes_and_upper_edge(self):
        d = state.Domain(4, 3, 0.5, origin_x_m=10.0, origin_y_m=-2.0, periodic=TF)     # x in [10, 12], y in [-2, -0.5]
        row, col = d.cell_indices(np.array([10.0, 10.25, 11.99, 12.0]), np.array([-2.0, -1.5, -0.75, -0.5]))
        self.assertEqual(col.tolist(), [0, 0, 3, 0])                    # periodic x: the upper edge IS the lower edge
        self.assertEqual(row.tolist(), [0, 1, 2, 2])                    # nonperiodic y: closed upper -> last row
        n = state.Domain(4, 3, 0.5, origin_x_m=10.0, origin_y_m=-2.0)
        self.assertEqual(n.cell_indices(np.array([12.0]), np.array([-0.5]))[1].tolist(), [3])       # legacy closed upper, unchanged
        b = state.Domain(4, 3, 0.5, origin_x_m=10.0, origin_y_m=-2.0, periodic=FT)
        row, col = b.cell_indices(np.array([12.0]), np.array([-0.5]))
        self.assertEqual((col.tolist(), row.tolist()), ([3], [0]))
        field = np.arange(12.0).reshape(3, 4)
        self.assertEqual(float(d.sample(field, [12.0], [-0.5])[0]), field[2, 0])

    def test_wrap_positions_is_explicit_and_ordinary_input_is_never_normalised(self):
        d = state.Domain(4, 3, 0.5, origin_x_m=10.0, origin_y_m=-2.0, periodic=TF)
        x, y = d.wrap_positions(np.array([12.25, 9.75, 12.0, 10.0, 21.5]), np.array([-1.0, -1.0, -1.0, -1.0, -3.0]))
        np.testing.assert_array_equal(x, [10.25, 11.75, 10.0, 10.0, 11.5])
        np.testing.assert_array_equal(y, [-1.0, -1.0, -1.0, -1.0, -3.0])      # the nonperiodic axis is returned unchanged
        self.assertFalse(d.contains(x[-1], y[-1]))                             # ... so the caller still rejects it
        nx_, ny_ = state.Domain(4, 3, 0.5).wrap_positions(np.array([99.0]), np.array([-99.0]))
        self.assertEqual((float(nx_[0]), float(ny_[0])), (99.0, -99.0))        # nonperiodic domain: identity
        with self.assertRaises(state.StateError):
            d.wrap_positions(np.array([np.nan]), np.array([0.0]))
        cat = cat0()
        with self.assertRaises(state.StateError):                              # ordinary inputs are validated, never silently wrapped
            state.PlantState.from_records(cat, d, [rec_row(1, 'grass', 12.25, -1.0, 0.1)])
        st = state.PlantState.from_records(cat, d, [rec_row(1, 'grass', 12.0, -1.0, 0.1)])   # the upper edge itself is allowed
        self.assertEqual(float(st.x_m[0]), 12.0)

    def test_plain_csv_and_companion_metadata_roundtrip_and_mismatch(self):
        cat = cat0()
        for dom in (state.Domain(6, 5, 1.0, origin_x_m=100.0, origin_y_m=-50.0),
                    state.Domain(6, 5, 1.0, origin_x_m=100.0, origin_y_m=-50.0, periodic=TT)):
            st = state.PlantState.from_records(cat, dom, [rec_row(7, 'shrub', 106.0, -45.0, 1.0), rec_row(8, 'grass', 100.5, -49.5, 0.1)],
                                               surface=np.zeros(dom.shape))
            text, meta_text = state.plant_files_text(st)
            self.assertTrue(text.startswith('plant_id,species,x_m,y_m'))
            self.assertNotIn('#', text)                                            # plain CSV: no embedded metadata
            meta = json.loads(meta_text)
            self.assertEqual(meta['domain'], dom.to_dict())
            self.assertEqual(('periodic' in meta['domain']), dom.any_periodic)
            self.assertEqual(meta['cell_sampling'], dom.cell_sampling_convention)
            back = state.parse_plant_csv(text, cat, metadata=meta_text)
            self.assertEqual(back.domain, dom)
            np.testing.assert_array_equal(back.x_m, st.x_m)
            other = state.Domain(6, 5, 1.0, origin_x_m=100.0, origin_y_m=-50.0, periodic=FF if dom.any_periodic else TT)
            with self.assertRaises(state.StateError):                              # the boundary choice is part of the domain identity
                state.parse_plant_csv(text, cat, domain=other, metadata=meta_text)
        # a handwritten plain list needs the (periodic) domain from the caller
        dom = state.Domain(6, 5, 1.0, origin_x_m=100.0, origin_y_m=-50.0, periodic=TF)
        st = state.parse_plant_csv(text, cat, domain=dom, metadata=False)
        self.assertEqual(st.domain.periodic, (True, False))


# ------------------------------------------------------------------ 2. generation: exclusion, bins, cover, no inset
class Q:
    '''Scripted uniform() source: returns the queued values in order (x then y per proposal), ignoring the bounds.'''

    def __init__(self, vals):
        self.vals = list(vals)

    def uniform(self, lo, hi):
        return self.vals.pop(0)


def gen_config(periodic=None, overlap='disallowed', seed=11, n=40, sh_cover=0.15, gr_cover=0.04, sh_mean=0.8):
    dom = {'nx': n, 'ny': n, 'dx_m': 0.25}
    if periodic is not None:
        dom['periodic'] = list(periodic)
    return {'schema': 'leaf.generation_config', 'schema_version': 1, 'domain': dom, 'seed': seed, 'overlap': overlap,
            'cover_tolerance': 0.01, 'cover_subsamples': 4, 'max_iterations': 20,
            'limits': {'max_plants': 5000, 'max_attempts': 3_000_000, 'max_cover_cells': 4_000_000, 'max_cover_bytes': 8_000_000,
                       'max_working_bytes': 1 << 20},
            'species': {'grass': {'target_cover': gr_cover, 'diameter_mean_m': 0.2, 'diameter_std_m': 0.03, 'initial_age_days': 100},
                        'shrub': {'target_cover': sh_cover, 'diameter_mean_m': sh_mean, 'diameter_std_m': 0.1, 'initial_age_days': 100}}}


class ExclusionPlacementTests(Case):
    def place(self, dom, d, vals):
        rng = Q(vals)
        X, Y, order, used = init._place_large_first(rng, dom, np.asarray(d, dtype=np.float64), 1000, 0)
        self.assertEqual(rng.vals, [], 'the scripted proposals must be consumed exactly')
        return X, Y, used

    def test_seam_overlap_is_rejected_and_exact_tangency_is_allowed(self):
        dom = state.Domain(10, 10, 1.0, periodic=TT)
        X, Y, used = self.place(dom, [1.0, 1.0], [0.25, 5.0, 9.75, 5.0, 9.25, 5.0])   # wrapped distance 0.5 -> reject; 1.0 = tangent -> ok
        self.assertEqual((X.tolist(), Y.tolist(), used), ([0.25, 9.25], [5.0, 5.0], 3))
        nonper = state.Domain(10, 10, 1.0)                                              # the finite domain has no seam: accepted at once
        X, Y, used = self.place(nonper, [1.0, 1.0], [0.25, 5.0, 9.75, 5.0])
        self.assertEqual((X.tolist(), used), ([0.25, 9.75], 2))

    def test_corner_overlap_uses_both_wrapped_axes(self):
        dom = state.Domain(10, 10, 1.0, periodic=TT)
        X, Y, used = self.place(dom, [1.0, 1.0], [0.25, 0.25, 9.75, 9.75, 9.25, 9.25])  # diagonal wrapped distance 0.707 < 1 -> reject
        self.assertEqual((X.tolist(), Y.tolist(), used), ([0.25, 9.25], [0.25, 9.25], 3))

    def test_mixed_axes_wrap_only_the_periodic_axis(self):
        px = state.Domain(10, 10, 1.0, periodic=TF)
        X, Y, used = self.place(px, [1.0, 1.0], [0.25, 5.0, 9.75, 5.0, 9.25, 5.0])
        self.assertEqual((X.tolist(), used), ([0.25, 9.25], 3))                         # x wraps -> reject the first copy
        py = state.Domain(10, 10, 1.0, periodic=FT)
        X, Y, used = self.place(py, [1.0, 1.0], [0.25, 5.0, 9.75, 5.0])                 # y is periodic only: no x seam
        self.assertEqual((X.tolist(), used), ([0.25, 9.75], 2))
        X, Y, used = self.place(py, [1.0, 1.0], [5.0, 0.25, 5.0, 9.75, 5.0, 9.25])      # the y seam rejects
        self.assertEqual((Y.tolist(), used), ([0.25, 9.25], 3))

    def test_awkward_bin_to_domain_ratio_never_misses_a_seam_neighbour(self):
        dom = state.Domain(10, 10, 1.0, periodic=TF)                                    # L = 10, bin edge 3: nb = 3 exact bins of 10/3
        X, Y, used = self.place(dom, [3.0, 3.0], [0.5, 5.0, 9.9, 5.5, 9.0, 5.0, 7.5, 5.0])
        # 9.9: wrapped dx 0.6; 9.0: wrapped dx 1.5 (both < 3 -> reject); 7.5: wrapped dx 3.0 = tangent -> accepted
        self.assertEqual((X.tolist(), used), ([0.5, 7.5], 4))

    def test_bin_layout_is_correct_for_any_ratio(self):
        rng = np.random.default_rng(3)
        for L, size in ((10.0, 3.0), (10.0, 4.5), (10.0, 7.0), (10.0, 10.0), (10.0, 25.0), (7.3, 1.1), (1.0, 2.5)):
            dom = state.Domain(10, 10, L / 10.0, periodic=TT)
            lay = periodic.bin_layout(dom, size)
            nb, width = lay[0]
            self.assertEqual(nb, max(1, int(L // size)))
            self.assertAlmostEqual(nb * width, L, places=12)                            # exact tiling, no short last bin
            p = rng.uniform(0.0, L, 2000)
            q = rng.uniform(0.0, L, 2000)
            near = np.abs(periodic.min_image(p - q, L)) < size
            for a, b in zip(p[near], q[near]):
                self.assertIn(periodic.bin_key(b, 0.0, lay[0]), periodic.bin_neighbours(periodic.bin_key(a, 0.0, lay[0]), lay[0]))
        nonper = periodic.bin_layout(state.Domain(10, 10, 1.0), 3.0)
        self.assertEqual(nonper[0], (None, 3.0))                                        # legacy bins on a nonperiodic axis

    def test_self_overlapping_canopy_is_refused_without_silent_changes(self):
        dom = state.Domain(10, 10, 1.0, periodic=TT)
        with self.assertRaises(init.GenerationError) as cm:
            self.place(dom, [11.0], [])
        self.assertIn('periodic span', str(cm.exception))
        X, Y, used = self.place(dom, [10.0], [5.0, 5.0])                                # diameter == span: tangent to its own image, allowed
        self.assertEqual(used, 1)
        self.place(state.Domain(10, 10, 1.0), [11.0], [5.0, 5.0])                       # the finite domain has no self image
        for per in (TF, FT):
            with self.assertRaises(init.GenerationError):                               # either periodic span (10 m) is exceeded
                self.place(state.Domain(10, 10, 1.0, periodic=per), [11.0], [])
        self.place(state.Domain(2, 10, 1.0, periodic=FT), [3.0], [1.0, 5.0])           # x (2 m) is NOT periodic: only the 10 m y span counts
        self.assertTrue(periodic.self_overlapping(dom, np.array([9.9, 10.0, 10.1])).tolist() == [False, False, True])
        periodic.check_no_self_overlap(state.Domain(10, 10, 1.0), np.array([99.0]))     # nonperiodic: never refused


class GenerationTests(Case):
    @classmethod
    def setUpClass(cls):
        cls.cat = cat0()
        cls.per, cls.rep = init.generate_population(cls.cat, gen_config(TT), surface=np.zeros((40, 40)))

    def test_nonperiodic_generation_is_unchanged_by_the_new_declaration(self):
        a, ra = init.generate_population(self.cat, gen_config(None, seed=5), surface=np.zeros((40, 40)))
        b, rb = init.generate_population(self.cat, gen_config(FF, seed=5), surface=np.zeros((40, 40)))
        c, rc = init.generate_population(self.cat, gen_config(None, seed=5), surface=np.zeros((40, 40)))
        for k in state.PlantState.ARRAY_NAMES:
            np.testing.assert_array_equal(getattr(a, k), getattr(b, k))
            np.testing.assert_array_equal(getattr(a, k), getattr(c, k))                # reproducible
        self.assertEqual(a.domain.to_dict(), b.domain.to_dict())
        self.assertNotIn('periodic', a.domain.to_dict())
        self.assertNotIn('boundary', ra)                                               # the nonperiodic report keeps its key set
        self.assertEqual(ra['ordering_convention'], init.ORDERING['disallowed'])
        self.assertIn('finite non-periodic', ra['cover_measurement']['definition'])
        self.assertEqual(set(ra), set(rb))
        # overlap-allowed mode too
        d, _ = init.generate_population(self.cat, gen_config(None, overlap='allowed', seed=5, sh_cover=0.1), surface=np.zeros((40, 40)))
        e, _ = init.generate_population(self.cat, gen_config(FF, overlap='allowed', seed=5, sh_cover=0.1), surface=np.zeros((40, 40)))
        np.testing.assert_array_equal(d.x_m, e.x_m)

    def test_periodic_exclusion_by_brute_force_wrapped_distances_one_record_per_plant(self):
        st = self.per
        L = st.domain.width_m
        geo = canopy.derive_geometry(st, self.cat)
        r = geo['radius_m']
        n = st.n
        self.assertEqual(n, self.rep['n_plants'])
        self.assertEqual(len(set(st.ids.tolist())), n)                                  # no duplicate / image records
        dx = periodic.min_image(st.x_m[:, None] - st.x_m[None, :], L)
        dy = periodic.min_image(st.y_m[:, None] - st.y_m[None, :], L)
        d2 = dx * dx + dy * dy
        need = (r[:, None] + r[None, :]) ** 2
        np.fill_diagonal(d2, np.inf)
        self.assertTrue(bool(np.all(d2 >= need * (1.0 - 1e-12))), 'two canopies overlap by wrapped distance')
        self.assertEqual(self.rep['boundary']['periodic'], [True, True])
        self.assertEqual(st.domain.periodic, (True, True))

    def test_full_domain_sampling_no_inset_and_wrapping_canopies_exist(self):
        st = self.per
        L = st.domain.width_m
        geo = canopy.derive_geometry(st, self.cat)
        r = geo['radius_m']
        edge_gap = np.minimum.reduce([st.x_m, L - st.x_m, st.y_m, L - st.y_m])
        self.assertLess(float(edge_gap.min()), 0.3)                                     # centres reach the border: no margin is applied
        self.assertLess(float((edge_gap - r).min()), 0.0)                               # canopies that cross a seam exist and stay ONE plant
        self.assertTrue(np.all((st.x_m >= 0.0) & (st.x_m <= L) & (st.y_m >= 0.0) & (st.y_m <= L)))
        sh = geo['species'] == 'shrub'
        self.assertTrue(bool(sh.any()))

    def test_reported_cover_matches_an_independent_tiled_image_raster(self):
        st = self.per
        geo = canopy.derive_geometry(st, self.cat)
        for rec_sp in self.rep['species']:
            m = geo['species'] == rec_sp['species']
            cov = tiled_cover(st.domain, st.x_m, st.y_m, geo['radius_m'], 4, group=m)
            self.assertAlmostEqual(rec_sp['measured_union_cover'], cov, places=12)
            self.assertLessEqual(abs(rec_sp['measured_union_cover'] - rec_sp['target_cover']), 0.01 + 1e-12)
        self.assertIn('periodic axes wrapped', self.rep['cover_measurement']['definition'])
        self.assertIn('periodic', self.rep['cover_measurement']['boundary_note'])
        self.assertIn('periodic axes', self.rep['ordering_convention'])
        self.assertIn('wrapped', self.rep['placement'])

    def test_overlap_allowed_keeps_independent_uniform_centres_and_union_cover(self):
        st, rep = init.generate_population(self.cat, gen_config(TT, overlap='allowed', seed=3, sh_cover=0.3), surface=np.zeros((40, 40)))
        geo = canopy.derive_geometry(st, self.cat)
        for rec_sp in rep['species']:
            m = geo['species'] == rec_sp['species']
            self.assertAlmostEqual(rec_sp['measured_union_cover'], tiled_cover(st.domain, st.x_m, st.y_m, geo['radius_m'], 4, group=m),
                                   places=12)
        self.assertNotIn('refused', rep['boundary']['overlap_test'])
        self.assertEqual(rep['boundary']['centres'], 'uniform over the FULL domain (no margin / inset)')

    def test_oversized_canopy_with_exclusion_refuses_the_generation(self):
        cfg = gen_config(TT, n=16, sh_cover=0.0)                                         # 4 m torus
        cfg['species'] = {'shrub': {'target_cover': 0.2, 'diameter_mean_m': 6.0, 'diameter_std_m': 0.0, 'initial_age_days': 100}}
        with self.assertRaises(init.GenerationError) as cm:
            init.generate_population(self.cat, cfg)
        self.assertIn('periodic span', str(cm.exception))


# ------------------------------------------------------------------ 3. measured cover
class CoverTests(Case):
    def cover(self, dom, x, y, r, s=4, groups=None, n_groups=1):
        return canopy.measure_cover(dom, x, y, r, groups, n_groups, subsamples=s)

    def test_wrapped_fragments_corners_and_union_match_independent_tiles(self):
        r = np.array([1.0, 0.8, 0.6, 1.2])
        x = np.array([0.25, 7.9, 4.0, 4.0])
        y = np.array([0.25, 4.0, 7.9, 4.0])
        for per in (TT, TF, FT, FF):
            dom = state.Domain(8, 8, 1.0, periodic=per)
            got = self.cover(dom, x, y, r)['groups'][0]
            want = tiled_cover(dom, x, y, r, 4)
            self.assertEqual(got['union_fraction'], want, msg=f'periodic={per}')
            self.assertEqual(got['covered_cells'], round(want * 32 * 32))
        c_tt = self.cover(state.Domain(8, 8, 1.0, periodic=TT), x, y, r)['groups'][0]['union_fraction']
        c_ff = self.cover(state.Domain(8, 8, 1.0), x, y, r)['groups'][0]['union_fraction']
        self.assertGreater(c_tt, c_ff)                                                   # fragments that the finite domain clips are kept

    def test_seam_shrub_of_the_maple_case_is_one_plant_with_wrapped_cover(self):
        dom = state.Domain(64, 64, 1.0, periodic=TT)
        got = self.cover(dom, np.array([63.8]), np.array([32.0]), np.array([0.5]), s=8)
        want = tiled_cover(dom, np.array([63.8]), np.array([32.0]), np.array([0.5]), 8, kmax=1)
        self.assertEqual(got['groups'][0]['union_fraction'], want)
        self.assertEqual(got['groups'][0]['count'], 1)
        self.assertAlmostEqual(got['groups'][0]['union_area_m2'], math.pi * 0.25, delta=0.05)    # full circle area, not a clipped half

    def test_self_overlapping_circle_is_counted_once_and_nominal_may_exceed_the_union(self):
        dom = state.Domain(8, 8, 1.0, periodic=TT)
        x, y, r = np.array([3.0]), np.array([5.0]), np.array([5.0])                      # diameter 10 > the 8 m span
        g = self.cover(dom, x, y, r)['groups'][0]
        self.assertEqual(g['union_fraction'], tiled_cover(dom, x, y, r, 4, kmax=3))
        self.assertLessEqual(g['union_fraction'], 1.0)
        self.assertGreater(g['nominal_fraction'], 1.0)                                    # pi r^2 / A = 1.227: not a physical cover
        two = self.cover(dom, np.array([3.0, 6.0]), np.array([5.0, 1.0]), np.array([5.0, 0.5]), n_groups=2,
                         groups=np.array([0, 1]))['groups']
        self.assertEqual(two[1]['union_fraction'], tiled_cover(dom, np.array([6.0]), np.array([1.0]), np.array([0.5]), 4))

    def test_nonperiodic_path_matches_a_clipped_oracle_and_default(self):
        dom = state.Domain(8, 8, 1.0, origin_x_m=3.0, origin_y_m=-4.0)
        x, y, r = np.array([3.25, 10.9, 7.0]), np.array([-3.5, 3.9, 0.0]), np.array([1.0, 0.8, 1.5])
        self.assertEqual(self.cover(dom, x, y, r)['groups'][0]['union_fraction'], tiled_cover(dom, x, y, r, 4))
        self.assertEqual(self.cover(state.Domain(8, 8, 1.0, origin_x_m=3.0, origin_y_m=-4.0, periodic=FF), x, y, r)['groups'][0]['covered_cells'],
                         self.cover(dom, x, y, r)['groups'][0]['covered_cells'])
        self.assertIn('finite non-periodic', self.cover(dom, x, y, r)['definition'])
        self.assertEqual(self.cover(dom, x, y, r)['boundary_note'], canopy.BOUNDARY_NOTE)

    def test_translation_equivariance_with_origin_offset(self):
        r = np.array([1.0, 0.8])
        a = state.Domain(8, 8, 1.0, periodic=TT)
        b = state.Domain(8, 8, 1.0, origin_x_m=100.0, origin_y_m=-50.0, periodic=TT)
        x, y = np.array([0.25, 7.5]), np.array([7.75, 3.0])
        self.assertEqual(self.cover(a, x, y, r)['groups'][0]['covered_cells'],
                         self.cover(b, x + 100.0, y - 50.0, r)['groups'][0]['covered_cells'])

    def test_admission_counts_periodic_scratch_and_stays_sparse(self):
        a = canopy.cover_admission(state.Domain(64, 64, 1.0), 4)
        b = canopy.cover_admission(state.Domain(64, 64, 1.0, periodic=TF), 4)
        self.assertGreater(b['working_bytes'], a['working_bytes'] - 1)
        self.assertEqual(b['raster_bytes'], a['raster_bytes'])                           # still ONE (ny*s, nx*s) boolean raster per group
        with self.assertRaises(canopy.ResourceBoundError):
            canopy.cover_admission(state.Domain(64, 64, 1.0, periodic=TT), 4, max_working_bytes=32 * 512 + 8)
        with self.assertRaises(canopy.ResourceBoundError):
            canopy.measure_cover(state.Domain(64, 64, 1.0, periodic=TT), [1.0], [1.0], [0.5], subsamples=4, max_cover_cells=100)

    def test_shared_canopy_view_is_one_object_per_plant_with_boundary_metadata(self):
        cat = cat0()
        dom = state.Domain(64, 64, 1.0, origin_x_m=10.0, origin_y_m=20.0, periodic=TT)
        st = state.PlantState.from_records(cat, dom, [rec_row(5, 'shrub', 73.8, 52.0, 0.3088)])
        for fn in (canopy.canopy_view, canopy.maple_vegetation_view):
            v = fn(st, cat, np.zeros(dom.shape))
            self.assertEqual(v['meta']['n_objects'], 1)
            self.assertEqual(v['fields']['center_x_m'].tolist(), [73.8 - 10.0])                 # the stored centre, no wrapped copy / shift
            self.assertEqual(v['sidecar']['plant_id'].tolist(), [5])
            self.assertEqual(v['meta']['boundary']['periodic'], [True, True])
            self.assertTrue(v['meta']['boundary']['one_record_per_plant'])
            self.assertEqual(v['meta']['domain']['periodic'], [True, True])
            self.assertEqual(v['meta']['boundary_note'], periodic.BOUNDARY_NOTE_PERIODIC)
        nonper = state.PlantState.from_records(cat, state.Domain(64, 64, 1.0), [rec_row(5, 'shrub', 63.8, 32.0, 0.3088)])
        meta = canopy.canopy_view(nonper, cat, np.zeros((64, 64)))['meta']
        self.assertNotIn('boundary', meta)
        self.assertEqual(meta['boundary_note'], canopy.BOUNDARY_NOTE)

    def test_wrapped_row_tiles_do_not_retain_a_mask_beyond_the_declared_memory(self):
        # one synthetic oversized circle (no plant population): several row tiles, a footprint spanning the whole 1024 x 1024 raster
        dom = state.Domain(256, 256, 0.25, periodic=TT)
        working = 4 << 20
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            rep = canopy.measure_cover(dom, np.array([0.0]), np.array([0.0]), np.array([80.0]), subsamples=4, max_cover_cells=2 << 20,
                                       max_cover_bytes=4 << 20, max_working_bytes=working)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertGreater(1024 // rep['rows_per_chunk'], 1)                                  # really several tiles
        self.assertEqual(rep['total_union_fraction'], 1.0)
        declared = rep['raster_bytes'] + rep['working_bytes'] + rep['index_bytes']
        self.assertLessEqual(peak, declared + (32 << 10),
                             msg=f'traced peak {peak} exceeds the declared raster + working + index bytes {declared} (+32 KiB metadata)')

    def test_default_cover_report_keeps_the_legacy_key_set_and_only_periodic_adds_fields(self):
        legacy = {'method', 'subsamples', 'raster_shape', 'resolution_m', 'raster_cells', 'raster_bytes', 'working_bytes',
                  'rows_per_chunk', 'index_bytes', 'admission', 'domain_area_m2', 'n_circles', 'groups', 'total_union_fraction',
                  'sum_of_group_union_fractions', 'definition', 'boundary_note', 'error_basis'}
        for dom in (state.Domain(8, 8, 1.0), state.Domain(8, 8, 1.0, periodic=FF)):
            out = self.cover(dom, np.array([1.0]), np.array([1.0]), np.array([0.5]))
            self.assertEqual(set(out), legacy)
            self.assertNotIn('periodic', out)
            self.assertEqual(out['boundary_note'], canopy.BOUNDARY_NOTE)
        per = self.cover(state.Domain(8, 8, 1.0, periodic=TF), np.array([1.0]), np.array([1.0]), np.array([0.5]))
        self.assertEqual(set(per) - legacy, {'periodic'})
        self.assertEqual(per['periodic'], [True, False])


# ------------------------------------------------------------------ 3b. exact upper endpoint == lower endpoint (decimal grids / origins)
class UpperAliasTests(Case):
    CASES = ((3, 0.2, 0.4), (7, 0.1, 0.3), (3, 0.7, 0.6), (7, 0.3, 1.1), (3, 0.1, 100000.1), (5, 0.5, 0.0),
             # exact tuples whose wrap_positions once returned x_min + 1 ulp (input subtraction rounded ABOVE the span)
             (3, 0.1, 0.3), (3, 0.2, 0.6), (3, 0.3, 0.6), (7, 0.2, 0.4), (7, 0.7, 1.1))

    def test_wrap_positions_broad_lower_interior_upper_and_outside(self):
        for nx, dx, ox in itertools.product((3, 7), (0.1, 0.2, 0.3, 0.7), (0.1, 0.3, 0.4, 0.6, 1.1, 100000.1)):
            d = state.Domain(nx, 2, dx, origin_x_m=ox, origin_y_m=-ox, periodic=TT)
            lo, hi = (d.x_min_m, d.y_min_m), (d.x_max_m, d.y_max_m)
            for k, (a, b) in enumerate(((lo, hi),)):
                wx, wy = d.wrap_positions(np.array([lo[0], hi[0], np.nextafter(hi[0], 0.0)]), np.array([lo[1], hi[1], np.nextafter(hi[1], -np.inf)]))
                self.assertEqual((float(wx[0]), float(wy[0])), lo, msg=f'lower {nx} {dx} {ox}')
                self.assertEqual((float(wx[1]), float(wy[1])), lo, msg=f'EXACT upper == exact lower {nx} {dx} {ox}')
                self.assertEqual(float(wx[2]), float(np.nextafter(hi[0], 0.0)), msg=f'just inside unchanged {nx} {dx} {ox}')
            mid = np.array([lo[0] + 0.5 * d.width_m, lo[0] + 0.1 * d.width_m])
            midy = np.array([lo[1] + 0.5 * d.height_m, lo[1] + 0.9 * d.height_m])
            wx, wy = d.wrap_positions(mid, midy)
            np.testing.assert_array_equal(wx, mid)                                           # interior values are bit-for-bit unchanged
            np.testing.assert_array_equal(wy, midy)
            # outside the closed domain: wrapped by whole spans into [min, max), never left outside
            wx, wy = d.wrap_positions(np.array([lo[0] - 0.25 * d.width_m, hi[0] + 0.25 * d.width_m]),
                                      np.array([hi[1] + 0.25 * d.height_m, lo[1] - 0.25 * d.height_m]))
            self.assertTrue(bool(np.all(d.contains(wx, wy))) and bool(np.all((wx >= lo[0]) & (wx < hi[0]) & (wy >= lo[1]) & (wy < hi[1]))))
            np.testing.assert_allclose(wx, [lo[0] + 0.75 * d.width_m, lo[0] + 0.25 * d.width_m], rtol=0, atol=1e-9 * max(1.0, abs(ox)))

    def test_wrap_positions_refuses_non_finite_for_every_domain_and_keeps_valid_draws(self):
        for d in (state.Domain(4, 3, 0.5), state.Domain(4, 3, 0.5, periodic=TF), state.Domain(4, 3, 0.5, periodic=TT)):
            for bad in (np.nan, np.inf, -np.inf):
                with self.subTest(domain=d, bad=bad), self.assertRaises(state.StateError):
                    d.wrap_positions(np.array([bad]), np.array([0.25]))
                with self.assertRaises(state.StateError):
                    d.wrap_positions(np.array([0.25]), np.array([bad]))
        x, y = np.array([1.25, 9.0]), np.array([-3.0, 0.5])
        wx, wy = state.Domain(4, 3, 0.5).wrap_positions(x, y)                                 # nonperiodic: values returned unchanged
        np.testing.assert_array_equal(wx, x)
        np.testing.assert_array_equal(wy, y)

    def test_exact_upper_endpoint_samples_wraps_exports_and_maps_as_the_lower_endpoint(self):
        cat = cat0()
        builders = ('numpy',) + (('compiled',) if HAVE_NUMBA else ())
        d0 = state.Domain(3, 2, 0.2, origin_x_m=0.4, periodic=TF)                            # the review example: local 0.6 < span 0.6000000000000001
        self.assertEqual(int(d0.cell_indices(np.array([d0.x_max_m]), np.array([d0.y_min_m]))[1][0]), 0)
        self.assertEqual(float(d0.wrap_positions(np.array([d0.x_max_m]), np.array([0.1]))[0][0]), d0.x_min_m)
        for nx, dx, ox in self.CASES:
            d = state.Domain(nx, 2, dx, origin_x_m=ox, origin_y_m=-ox, periodic=TT)
            xu, yu = np.array([d.x_max_m]), np.array([d.y_max_m])
            row, col = d.cell_indices(xu, yu)
            self.assertEqual((int(row[0]), int(col[0])), (0, 0), msg=f'{nx} {dx} {ox}')
            wx, wy = d.wrap_positions(xu, yu)
            self.assertEqual((float(wx[0]), float(wy[0])), (d.x_min_m, d.y_min_m), msg=f'{nx} {dx} {ox}')

            def mk(x, y, dom=d):
                return state.PlantState.from_records(cat, dom, [rec_row(9, 'grass', float(x), float(y), 0.1)])
            up, low = mk(xu[0], yu[0]), mk(d.x_min_m, d.y_min_m)           # the legal upper point is accepted as ordinary input
            self.assertEqual((float(up.x_m[0]), float(up.y_m[0])), (float(xu[0]), float(yu[0])))      # the stored record is NOT changed
            for fn in (canopy.canopy_view, canopy.maple_vegetation_view):
                v = fn(up, cat, np.zeros(d.shape))
                self.assertEqual((v['fields']['center_x_m'].tolist(), v['fields']['center_y_m'].tolist()), ([0.0], [0.0]))
                self.assertEqual(v['meta']['n_objects'], 1)
                self.assertEqual(v['sidecar']['plant_id'].tolist(), [9])
            for builder in builders:
                a, b = (roots.build_root_map(p, cat, max_edges=10_000, max_bytes=BIG, builder=builder).host_arrays() for p in (up, low))
                for k in ('ids', 'offsets', 'resources', 'weights'):
                    np.testing.assert_array_equal(a[k], b[k], err_msg=f'root {k} {builder} {nx} {dx} {ox}')
                a, b = (pp.canopy_map_from_state(p, cat, max_edges=10_000, max_bytes=BIG, builder=builder).host_arrays() for p in (up, low))
                for k in ('offsets', 'cells', 'areas_m2', 'area_total_m2'):
                    np.testing.assert_array_equal(a[k], b[k], err_msg=f'canopy {k} {builder} {nx} {dx} {ox}')
            # a REAL just-inside position is not moved onto the seam: it keeps the last cell and a public centre inside [0, L)
            xi, yi = np.nextafter(xu, -np.inf), np.nextafter(yu, -np.inf)
            row, col = d.cell_indices(xi, yi)
            self.assertEqual((int(row[0]), int(col[0])), (d.ny - 1, d.nx - 1), msg=f'{nx} {dx} {ox}')
            wi = d.wrap_positions(xi, yi)
            self.assertEqual((float(wi[0][0]), float(wi[1][0])), (float(xi[0]), float(yi[0])))     # inside [min, max): unchanged bit-for-bit
            v = canopy.canopy_view(mk(xi[0], yi[0]), cat, np.zeros(d.shape))
            self.assertTrue(0.0 < v['fields']['center_x_m'][0] < d.width_m and 0.0 < v['fields']['center_y_m'][0] < d.height_m)

    def test_mixed_axis_mappers_accept_both_upper_edges_and_apply_only_the_periodic_alias(self):
        cat = cat0()
        builders = ('numpy',) + (('compiled',) if HAVE_NUMBA else ())
        for flags, dx, ox in itertools.product((TF, FT), (0.1, 0.2, 0.3, 0.7), (0.1, 0.3, 0.4, 100000.1)):
            d = state.Domain(3, 7, dx, origin_x_m=ox, origin_y_m=-ox, periodic=flags)
            upper = (d.x_max_m, d.y_max_m)
            equiv = (d.x_min_m if flags[0] else d.x_max_m, d.y_min_m if flags[1] else d.y_max_m)

            def pop(x, y, dom=d):
                return state.PlantState.from_records(cat, dom, [rec_row(3, 'grass', float(x), float(y), 0.1)])
            up, eq = pop(*upper), pop(*equiv)
            for builder in builders:
                a, b = (roots.build_root_map(p, cat, max_edges=10_000, max_bytes=BIG, builder=builder).host_arrays() for p in (up, eq))
                self.assertEqual(dict(zip(a['resources'].tolist(), a['weights'].tolist())), dict(zip(b['resources'].tolist(), b['weights'].tolist())),
                                 msg=f'roots {flags} {dx} {ox} {builder}')
                a, b = (pp.canopy_map_from_state(p, cat, max_edges=10_000, max_bytes=BIG, builder=builder).host_arrays() for p in (up, eq))
                self.assertEqual(dict(zip(a['cells'].tolist(), a['areas_m2'].tolist())), dict(zip(b['cells'].tolist(), b['areas_m2'].tolist())),
                                 msg=f'canopy {flags} {dx} {ox} {builder}')
            # just OUTSIDE the closed domain is refused before any clipping (also by the PET builder), on either axis
            ids = np.array([3], dtype=np.int64)
            r = np.array([0.1])
            for xo, yo in ((np.nextafter(d.x_max_m, np.inf), d.y_max_m), (d.x_max_m, np.nextafter(d.y_max_m, np.inf)),
                           (np.nextafter(d.x_min_m, -np.inf), d.y_min_m), (d.x_min_m, np.nextafter(d.y_min_m, -np.inf))):
                with self.assertRaises(pp.PlantPetError, msg=f'outside {flags} {dx} {ox}'):
                    pp.build_canopy_map(d, ids, np.array([float(xo)]), np.array([float(yo)]), r, max_edges=1000, max_bytes=BIG)
        d = state.Domain(3, 7, 0.1, origin_x_m=0.3, origin_y_m=-0.3, periodic=FT)               # the review example, both exact upper edges
        pm = build_map(d, [d.x_max_m], [d.y_max_m], [0.05])
        self.assertTrue(len(row_of(pm)[0]) >= 1)

    def test_mixed_axes_and_nonperiodic_edge_keep_their_rules(self):
        d = state.Domain(3, 2, 0.2, origin_x_m=0.4, origin_y_m=0.1, periodic=TF)
        row, col = d.cell_indices(np.array([d.x_max_m]), np.array([d.y_max_m]))
        self.assertEqual((int(row[0]), int(col[0])), (1, 0))                                  # y nonperiodic: closed upper -> last row
        n = state.Domain(3, 2, 0.2, origin_x_m=0.4)
        self.assertEqual(int(n.cell_indices(np.array([n.x_max_m]), np.array([0.1]))[1][0]), 2)   # legacy closed upper -> last column
        cat = cat0()
        st = state.PlantState.from_records(cat, d, [rec_row(9, 'grass', d.x_max_m, d.y_max_m, 0.1)])
        v = canopy.canopy_view(st, cat, np.zeros(d.shape))
        self.assertEqual(v['fields']['center_x_m'].tolist(), [0.0])
        self.assertEqual(v['fields']['center_y_m'].tolist(), [d.y_max_m - d.origin_y_m])      # the nonperiodic centre is the plain subtraction


# ------------------------------------------------------------------ 4. canopy-area-weighted PET
def build_map(dom, x, y, r, builder='numpy', mask=None, ids=None, **kw):
    ids = np.arange(1, len(x) + 1, dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
    return pp.build_canopy_map(dom, ids, np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(r, dtype=np.float64),
                               max_edges=kw.pop('max_edges', 100_000), max_bytes=kw.pop('max_bytes', BIG), builder=builder,
                               valid_cell_mask=mask, **kw)


def row_of(cm, p=0):
    h = cm.host_arrays()
    sl = slice(int(h['offsets'][p]), int(h['offsets'][p + 1]))
    cells = h['cells'][sl].tolist()
    assert len(set(cells)) == len(cells), 'one sparse entry per plant-cell pair'
    return dict(zip(cells, h['areas_m2'][sl].tolist())), float(h['area_total_m2'][p])


class CanopyPetTests(Case):
    def assertAreas(self, got, want, rtol=1e-11, atol=1e-13):
        self.assertEqual(sorted(got), sorted(want))
        for c in want:
            self.assertAlmostEqual(got[c], want[c], delta=rtol * want[c] + atol, msg=f'cell {c}')

    def test_seam_plant_sums_wrapped_fragments_exactly(self):
        dom = state.Domain(4, 4, 1.0, periodic=TT)
        for builder in ('numpy',) + (('compiled',) if HAVE_NUMBA else ()):
            for px, py, r in ((0.25, 2.0, 0.5), (3.75, 0.5, 0.75), (0.5, 3.5, 0.9), (2.0, 2.0, 1.0)):
                got, total = row_of(build_map(dom, [px], [py], [r], builder))
                self.assertAreas(got, image_fragment_areas(dom, px, py, r))
                self.assertAlmostEqual(total, math.pi * r * r, delta=1e-12 * math.pi * r * r)      # the full circle, wrapped
                self.assertAlmostEqual(sum(got.values()), total, delta=1e-12)

    def test_corner_and_upper_edge_are_the_same_four_cells(self):
        dom = state.Domain(4, 4, 1.0, periodic=TT)
        for px, py in ((0.0, 0.0), (4.0, 4.0), (0.0, 4.0), (4.0, 0.0)):
            got, total = row_of(build_map(dom, [px], [py], [0.5]))
            self.assertEqual(sorted(got), [0, 3, 12, 15])
            for a in got.values():
                self.assertAlmostEqual(a, math.pi * 0.25 / 4.0, delta=1e-13)
            self.assertAlmostEqual(total, math.pi * 0.25, delta=1e-13)

    def test_tangent_contact_is_excluded_and_one_cell_remains(self):
        dom = state.Domain(8, 8, 1.0, periodic=TT)
        got, total = row_of(build_map(dom, [0.5], [0.5], [0.5]))                         # wrapped neighbours touch at a single point
        self.assertEqual(list(got), [0])
        self.assertAlmostEqual(total, math.pi * 0.25, delta=1e-13)

    def test_spatially_varying_pet_reads_across_the_seam(self):
        dom = state.Domain(4, 4, 1.0, periodic=TT)
        field = np.arange(16, dtype=np.float64) + 1.0
        cm = build_map(dom, [0.25], [2.0], [0.5])
        out = pp.canopy_pet(cm, field, max_bytes=BIG)['pet_mm_per_day']
        want = image_fragment_areas(dom, 0.25, 2.0, 0.5)
        expect = sum(a * field[c] for c, a in want.items()) / sum(want.values())
        self.assertAlmostEqual(float(out[0]), expect, delta=1e-12 * expect)
        cells_left_only = [c for c in want if c % 4 == 0]
        self.assertTrue(cells_left_only and any(c % 4 == 3 for c in want))                # both sides of the seam contribute
        nonper = build_map(state.Domain(4, 4, 1.0), [0.25], [2.0], [0.5])
        self.assertNotAlmostEqual(float(pp.canopy_pet(nonper, field, max_bytes=BIG)['pet_mm_per_day'][0]), expect, places=6)
        const = pp.canopy_pet(cm, np.full(16, 4.321), max_bytes=BIG)['pet_mm_per_day']
        self.assertEqual(float(const[0]), 4.321)                                            # a uniform field is reproduced exactly

    def test_mask_excludes_cells_and_zero_support_refuses(self):
        dom = state.Domain(4, 4, 1.0, periodic=TT)
        full = image_fragment_areas(dom, 0.25, 2.0, 0.5)
        mask = np.ones(16, dtype=np.bool_)
        mask[3 + 4 * 1] = False                                                              # a seam-side cell (row 1, col 3)
        got, total = row_of(build_map(dom, [0.25], [2.0], [0.5], mask=mask))
        self.assertNotIn(7, got)
        self.assertAlmostEqual(total, sum(a for c, a in full.items() if c != 7), delta=1e-13)   # denominator = represented valid support
        mask[:] = False
        with self.assertRaises(pp.PlantPetError) as cm:
            build_map(dom, [0.25], [2.0], [0.5], mask=mask)
        self.assertIn('ZERO', str(cm.exception))

    def test_mixed_axes_clip_only_the_nonperiodic_axis(self):
        dom = state.Domain(4, 4, 1.0, periodic=TF)
        got, total = row_of(build_map(dom, [0.25], [0.25], [0.5]))
        self.assertAreas(got, image_fragment_areas(dom, 0.25, 0.25, 0.5))
        self.assertEqual(sorted(got), [0, 3])                                                # x wraps (cols 3 and 0); y is clipped at the south edge
        self.assertLess(total, math.pi * 0.25 * 0.99)                                        # the part below y = 0 is dropped, not wrapped
        domy = state.Domain(4, 4, 1.0, periodic=FT)
        g2, t2 = row_of(build_map(domy, [0.25], [0.25], [0.5]))
        self.assertAreas(g2, image_fragment_areas(domy, 0.25, 0.25, 0.5))
        self.assertEqual(sorted(g2), [0, 12])                                                # y wraps (rows 3 and 0); x is clipped at the west edge
        g3, t3 = row_of(build_map(dom, [0.25], [1.0], [0.5]))                                # away from the y edge: the full circle, 4 cells
        self.assertEqual(sorted(g3), [0, 3, 4, 7])
        self.assertAlmostEqual(t3, math.pi * 0.25, delta=1e-13)

    def test_translation_and_cell_rotation_equivariance(self):
        a = state.Domain(4, 4, 1.0, periodic=TT)
        b = state.Domain(4, 4, 1.0, origin_x_m=100.0, origin_y_m=-50.0, periodic=TT)
        ga, ta = row_of(build_map(a, [0.25], [2.0], [0.5]))
        gb, tb = row_of(build_map(b, [100.25], [-48.0], [0.5]))
        self.assertAreas(gb, ga)
        self.assertAlmostEqual(ta, tb, delta=1e-13)
        gr, tr = row_of(build_map(a, [2.25], [2.0], [0.5]))                                  # shifted by exactly 2 cells in x
        rotated = {(c // 4) * 4 + ((c % 4) + 2) % 4: v for c, v in ga.items()}
        self.assertAreas(gr, rotated)

    def test_self_overlapping_canopy_uses_the_union_not_the_double_counted_sum(self):
        dom = state.Domain(4, 4, 1.0, periodic=TT)
        for r in (2.25, 2.5, 3.0):                                                           # 2 r > the 4 m span
            got, total = row_of(build_map(dom, [1.5], [2.5], [r]))
            want = grid_union_areas(dom, 1.5, 2.5, r)
            for c in set(got) | set(want):                                                   # tiny corner areas may be missed by the grid
                self.assertAlmostEqual(got.get(c, 0.0), want.get(c, 0.0), delta=0.03, msg=f'r={r} cell {c}')
            for a in got.values():
                self.assertLessEqual(a, 1.0 + 1e-12)                                         # never more than the cell
            self.assertLess(total, math.pi * r * r)
            self.assertLessEqual(total, 16.0 + 1e-12)
            # independent exact total: the circle clipped to its nearest-image rectangle |dx|, |dy| <= 2 (the existing integrator)
            self.assertAlmostEqual(total, pp.circle_rectangle_area(r, -2.0, 2.0, -2.0, 2.0), delta=1e-10)
        got, total = row_of(build_map(dom, [1.5], [2.5], [7.0]))                              # covers the whole torus
        self.assertEqual(len(got), 16)
        self.assertAlmostEqual(total, 16.0, delta=1e-11)
        for a in got.values():
            self.assertAlmostEqual(a, 1.0, delta=1e-12)

    def test_exact_tangent_circle_rectangle_area_python_and_compiled(self):
        for compiled in (False,) + ((True,) if HAVE_NUMBA else ()):
            area = lambda *a: pp.circle_rectangle_area(*a, compiled=compiled)
            self.assertAlmostEqual(area(0.5, -0.5, 0.5, -0.5, 0.5), math.pi / 4, delta=1e-15)        # was 1.0 (side-product) before
            self.assertAlmostEqual(area(0.25, -0.25, 0.25, -0.25, 0.25), math.pi / 16, delta=1e-16)  # independent value 0.19634954084936207
            self.assertAlmostEqual(area(1.0, -1.0, 1.0, 0.0, 1.0), math.pi / 2, delta=1e-15)          # tangent on one side only
            self.assertAlmostEqual(area(1.0, -1.0, 1.0, -1.0, 0.0), math.pi / 2, delta=1e-15)
            self.assertAlmostEqual(area(1.0, -2.0, 2.0, -1.0, 1.0), math.pi, delta=1e-15)
            self.assertAlmostEqual(area(1.0, 0.0, 1.0, 0.0, 1.0), math.pi / 4, delta=1e-15)           # unchanged ordinary case
            self.assertEqual(area(1.0, 2.0, 3.0, 2.0, 3.0), 0.0)

    def test_circle_tangent_to_its_cell_sides_in_nonperiodic_and_periodic_canopy_maps(self):
        for builder in ('numpy',) + (('compiled',) if HAVE_NUMBA else ()):
            for per in (FF, TT, TF):
                dom = state.Domain(2, 2, 0.5, periodic=per)
                got, total = row_of(build_map(dom, [0.25], [0.25], [0.25], builder))
                self.assertEqual(list(got), [0], msg=f'{builder} {per}')                          # neighbours are tangent -> excluded
                self.assertAlmostEqual(got[0], math.pi / 16, delta=1e-16)
                self.assertAlmostEqual(total, math.pi / 16, delta=1e-16)
            one = state.Domain(1, 1, 1.0, periodic=TT)                                           # inscribed circle of a one-cell torus
            got, total = row_of(build_map(one, [0.5], [0.5], [0.5], builder))
            self.assertEqual(list(got), [0])
            self.assertAlmostEqual(total, math.pi / 4, delta=1e-15)

    def test_oversized_canopy_keeps_the_nominal_demand_area_while_pet_weights_are_the_union(self):
        cat = cat0()
        dom = state.Domain(4, 4, 1.0, periodic=TT)
        st = state.PlantState.from_records(cat, dom, [rec_row(1, 'grass', 1.5, 2.5, 30.0)])
        r = float(canopy.derive_geometry(st, cat)['radius_m'][0])
        self.assertGreater(2.0 * r, 4.0)                                                     # wider than the 4 m span: overlaps its own image
        for builder in ('numpy',) + (('compiled',) if HAVE_NUMBA else ()):
            g = biocalendar.rebuild_geometry(st, cat, st.agb_dry_kg, st.seasonal_agb_dry_kg, elapsed_days=0.0, max_edges=100_000,
                                             max_bytes=1 << 28, builder=builder, canopy={'max_edges': 100_000, 'max_bytes': 1 << 27})
            rad = np.array([r])
            np.testing.assert_array_equal(g.canopy_area_m2, np.pi * rad * rad)               # the demand law's nominal allometric area
            got, total = row_of(g.canopy_map)
            self.assertLess(total, math.pi * r * r)                                          # PET support: the exact torus union
            self.assertAlmostEqual(total, pp.circle_rectangle_area(r, -2.0, 2.0, -2.0, 2.0), delta=1e-10)
            self.assertAlmostEqual(sum(got.values()), total, delta=1e-12)
        self.assertFalse(hasattr(periodic, 'canopy_areas_m2'))                               # no second demand-area law exists

    @unittest.skipUnless(HAVE_NUMBA, 'numba not installed')
    def test_compiled_kernel_matches_the_python_reference(self):
        dom = state.Domain(12, 9, 1.0, origin_x_m=7.0, origin_y_m=3.0, periodic=TF)
        rng = np.random.default_rng(1)
        x = 7.0 + rng.uniform(0.0, 12.0, 20)
        y = 3.0 + rng.uniform(0.0, 9.0, 20)
        r = rng.uniform(0.2, 5.5, 20)
        a = build_map(dom, x, y, r, 'numpy')
        b = build_map(dom, x, y, r, 'compiled')
        ha, hb = a.host_arrays(), b.host_arrays()
        np.testing.assert_array_equal(ha['offsets'], hb['offsets'])
        np.testing.assert_array_equal(ha['cells'], hb['cells'])
        np.testing.assert_allclose(ha['areas_m2'], hb['areas_m2'], rtol=1e-12)
        field = np.linspace(1.0, 9.0, dom.ncells)
        np.testing.assert_allclose(pp.canopy_pet(a, field, max_bytes=BIG)['pet_mm_per_day'],
                                   pp.canopy_pet(b, field, max_bytes=BIG)['pet_mm_per_day'], rtol=1e-12)

    def test_budgets_refuse_before_large_allocations_and_provenance_names_the_boundary(self):
        dom = state.Domain(4, 4, 1.0, periodic=TT)
        with self.assertRaises(pp.PlantPetBudgetError):
            build_map(dom, [0.25], [2.0], [0.5], max_edges=2)                                # 5+ edges: refused at the count stage
        with self.assertRaises(pp.PlantPetBudgetError):
            build_map(dom, [0.25], [2.0], [0.5], max_bytes=16)
        cm = build_map(dom, [0.25], [2.0], [0.5])
        self.assertEqual(cm.provenance['boundary']['periodic'], [True, True])
        self.assertEqual(cm.provenance['domain']['periodic'], [True, True])
        self.assertNotIn('boundary', build_map(state.Domain(4, 4, 1.0), [2.0], [2.0], [0.5]).provenance)


# ------------------------------------------------------------------ 5. root footprints
def root_state(dom, cat=None):
    cat = cat or cat0()
    ox, oy = dom.origin_x_m, dom.origin_y_m
    recs = [rec_row(11, 'shrub', ox + 0.5, oy + 0.5, 2.5),                    # SW corner: wraps in both axes
            rec_row(12, 'grass', ox + dom.width_m - 0.25, oy + 3.5, 0.4),      # east edge
            rec_row(13, 'grass', ox + dom.width_m, oy + dom.height_m, 0.4),    # the NE corner == the SW corner
            rec_row(14, 'shrub', ox + 6.25, oy + 4.75, 1.5),                   # interior
            rec_row(15, 'grass', ox + 3.5, oy + 2.5, 0.05, 'dead_standing')]
    return state.PlantState.from_records(cat, dom, recs)


class RootTests(Case):
    def builders(self):
        return ('numpy',) + (('compiled',) if HAVE_NUMBA else ())

    def test_toroidal_footprints_equal_the_brute_force_image_union(self):
        cat = cat0()
        for per in (TT, TF, FT, FF):
            for origin in ((0.0, 0.0), (100.0, -50.0)):
                dom = state.Domain(12, 10, 1.0, origin_x_m=origin[0], origin_y_m=origin[1], periodic=per)
                st = root_state(dom, cat)
                want = expected_root_rows(st, cat)
                for builder in self.builders():
                    m = roots.build_root_map(st, cat, max_edges=100_000, max_bytes=BIG, builder=builder)
                    with self.subTest(per=per, origin=origin, builder=builder):
                        self.assertRowsEqual(map_rows(m), want)
                        h = m.host_arrays()
                        for p in range(m.n_plants):                                           # unit weight sum over unique cells
                            self.assertAlmostEqual(float(h['weights'][h['offsets'][p]:h['offsets'][p + 1]].sum()), 1.0, places=12)
        # corners: the plant at the NE corner and the one at the SW corner see the same four wrapped cells
        dom = state.Domain(12, 10, 1.0, periodic=TT)
        rows = map_rows(roots.build_root_map(root_state(dom, cat), cat, max_edges=100_000, max_bytes=BIG, builder='numpy'))
        cells13 = {c % dom.ncells for c in rows[13]}
        self.assertTrue({0, 11, 9 * 12, 9 * 12 + 11} <= cells13)

    def test_both_sides_of_a_seam_and_roots_of_different_plants_compete(self):
        cat = cat0()
        dom = state.Domain(12, 10, 1.0, periodic=TT)
        st = state.PlantState.from_records(cat, dom, [rec_row(1, 'grass', 0.1, 3.5, 0.1), rec_row(2, 'grass', 11.9, 3.5, 0.1)])
        rows = map_rows(roots.build_root_map(st, cat, max_edges=1000, max_bytes=BIG, builder='numpy'))
        c1 = {r % dom.ncells for r in rows[1]}
        c2 = {r % dom.ncells for r in rows[2]}
        self.assertIn(3 * 12 + 11, c1)                                                       # plant 1 (x = 0.1) reaches column 11 across the seam
        self.assertIn(3 * 12 + 0, c2)
        self.assertTrue(c1 & c2)                                                             # roots of different plants may overlap
        self.assertEqual(len(c1), len(set(c1)))
        nonper = state.PlantState.from_records(cat, state.Domain(12, 10, 1.0), [rec_row(1, 'grass', 0.1, 3.5, 0.1), rec_row(2, 'grass', 11.9, 3.5, 0.1)])
        rows_np = map_rows(roots.build_root_map(nonper, cat, max_edges=1000, max_bytes=BIG, builder='numpy'))
        self.assertNotIn(3 * 12 + 11, {r % 120 for r in rows_np[1]})                         # the finite domain has no seam

    def test_species_and_individual_layer_fractions(self):
        cat = cat0()
        dom = state.Domain(12, 10, 1.0, periodic=TT)
        st = root_state(dom, cat)
        ov = {11: [1.0, 0.0], 12: [0.1, 0.9]}
        m = roots.build_root_map(st, cat, max_edges=100_000, max_bytes=BIG, builder='numpy', layer_fraction_overrides=ov)
        self.assertRowsEqual(map_rows(m), expected_root_rows(st, cat, ov))
        self.assertTrue(all(r < dom.ncells for r in map_rows(m)[11]))                        # layer 0 only
        self.assertEqual(m.provenance['n_overrides'], 2)
        base = roots.build_root_map(st, cat, max_edges=100_000, max_bytes=BIG, builder='numpy')
        rows = map_rows(base)
        for pid, sp in ((12, 'grass'), (14, 'shrub')):
            frac = cat.root_layer_table()[cat.index(sp)]
            for layer in range(cat.root_layers):
                self.assertAlmostEqual(sum(w for r, w in rows[pid].items() if r // dom.ncells == layer), float(frac[layer]), places=12)

    def test_tangent_exclusion_and_unique_cells_for_roots_wider_than_the_domain(self):
        dom = state.Domain(8, 8, 1.0, periodic=TT)
        h = roots.horizontal_edges_periodic(np.array([0.5]), np.array([0.5]), np.array([0.5]), 8, 8, 1.0, TT, builder='numpy',
                                            max_edges=100, max_bytes=BIG)
        self.assertEqual((h['cells'].tolist(), h['weights'].tolist()), ([0], [1.0]))        # wrapped neighbours are tangent -> excluded
        h = roots.horizontal_edges_periodic(np.array([0.5]), np.array([0.5]), np.array([1000.0]), 8, 8, 1.0, TT, builder='numpy',
                                            max_edges=100, max_bytes=BIG)
        self.assertEqual(sorted(h['cells'].tolist()), list(range(64)))                       # saturates: every physical cell ONCE
        np.testing.assert_allclose(h['weights'], 1.0 / 64.0, rtol=1e-15)
        for nx, ny, per in ((1, 1, TT), (1, 5, TT), (7, 1, TT), (1, 5, TF), (7, 1, FT)):
            d = state.Domain(nx, ny, 1.0, periodic=per)
            for r in (0.3, 1.5, 40.0):
                hh = roots.horizontal_edges_periodic(np.array([0.5 * nx]), np.array([0.5 * ny]), np.array([r]), nx, ny, 1.0, per,
                                                     builder='numpy', max_edges=10_000, max_bytes=BIG)
                self.assertEqual(sorted(hh['cells'].tolist()), sorted(reached_cells(d, 0.5 * nx, 0.5 * ny, r)), msg=f'{nx}x{ny} {per} r={r}')
                self.assertEqual(len(set(hh['cells'].tolist())), int(hh['n_edges']))
                self.assertAlmostEqual(float(hh['weights'].sum()), 1.0, places=12)
        cat = cat0()
        big = state.PlantState.from_records(cat, state.Domain(12, 10, 1.0, periodic=TT), [rec_row(1, 'shrub', 6.0, 5.0, 500.0)])
        m = roots.build_root_map(big, cat, max_edges=10_000, max_bytes=BIG, builder='numpy')
        rows = map_rows(m)[1]
        self.assertEqual(len(rows), 120 * cat.root_layers)                                    # a root system far wider than the domain
        np.testing.assert_allclose(sorted(set(round(w, 12) for w in rows.values())),
                                   sorted(set(round(f / 120.0, 12) for f in cat.root_layer_table()[cat.index('shrub')])), rtol=1e-9)

    @unittest.skipUnless(HAVE_NUMBA, 'numba not installed')
    def test_compiled_and_python_builders_and_numba_backend_agree_exactly(self):
        cat = cat0()
        dom = state.Domain(12, 10, 1.0, origin_x_m=5.0, origin_y_m=-3.0, periodic=TT)
        st = root_state(dom, cat)
        a = roots.build_root_map(st, cat, max_edges=100_000, max_bytes=BIG, builder='numpy')
        b = roots.build_root_map(st, cat, max_edges=100_000, max_bytes=BIG, builder='compiled')
        c = roots.build_root_map(st, cat, max_edges=100_000, max_bytes=BIG, builder='compiled', backend='numba')
        ha = a.host_arrays()
        for other in (b, c):
            ho = other.host_arrays()
            for k in ('ids', 'offsets', 'resources'):
                np.testing.assert_array_equal(ha[k], ho[k])
            np.testing.assert_array_equal(ha['weights'], ho['weights'])

    def test_limits_refuse_before_edges_and_never_allocate_plant_by_grid(self):
        cat = cat0()
        dom = state.Domain(12, 10, 1.0, periodic=TT)
        st = root_state(dom, cat)
        with self.assertRaises(uptake.AdmissionError):
            roots.build_root_map(st, cat, max_edges=5, max_bytes=BIG, builder='numpy')
        with self.assertRaises(uptake.AdmissionError):
            roots.build_root_map(st, cat, max_edges=100_000, max_bytes=64, builder='numpy')
        # 2000 plants on a 200 x 200 torus: a dense P x N buffer would be 2000 * 40000 * 8 = 640 MB; the sparse map fits far below
        big_dom = state.Domain(200, 200, 1.0, periodic=TT)
        rng = np.random.default_rng(2)
        recs = [rec_row(i + 1, 'grass', float(rng.uniform(0, 200)), float(rng.uniform(0, 200)), 0.1) for i in range(2000)]
        big = state.PlantState.from_records(cat, big_dom, recs)
        tracemalloc.start()
        try:
            m = roots.build_root_map(big, cat, max_edges=1_000_000, max_bytes=64 << 20, builder='numpy')
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 48 << 20)
        self.assertGreater(m.n_edges, 0)
        self.assertLess(m.n_edges, 2000 * 40 * cat.root_layers)

    def test_provenance_names_the_toroidal_law(self):
        cat = cat0()
        m = roots.build_root_map(root_state(state.Domain(12, 10, 1.0, periodic=TF), cat), cat, max_edges=100_000, max_bytes=BIG, builder='numpy')
        self.assertEqual(m.provenance['horizontal_law'], roots.HORIZONTAL_LAW)               # same law, same name
        self.assertEqual(m.provenance['horizontal_law_label'], roots.HORIZONTAL_LAW_LABEL_PERIODIC)
        self.assertEqual(m.provenance['boundary']['periodic'], [True, False])
        self.assertEqual(m.provenance['domain']['periodic'], [True, False])
        m0 = roots.build_root_map(root_state(state.Domain(12, 10, 1.0), cat), cat, max_edges=100_000, max_bytes=BIG, builder='numpy')
        self.assertEqual(m0.provenance['horizontal_law_label'], roots.HORIZONTAL_LAW_LABEL)
        self.assertNotIn('boundary', m0.provenance)
        with self.assertRaises(roots.RootsError):                                              # non-square cells stay refused
            roots.build_root_map(state.PlantState.from_records(cat, state.Domain(4, 4, 1.0, 2.0, periodic=TT), []), cat,
                                 max_edges=10, max_bytes=BIG, builder='numpy')


# ------------------------------------------------------------------ 6. uptake across a seam (law unchanged)
def oracle_uptake(rows_by_pos, D, A, L, N):
    '''Independent scalar oracle of the water rules: request Q_r = sum_p D_p w_pr, accepted_r = min(A_r, Q_r), delivered_p = sum_r
    D_p w_pr accepted_r / Q_r (proportional, equal fraction of requests, no priority, no compensation).'''
    Q = np.zeros(L * N)
    for p, row in enumerate(rows_by_pos):
        for r, w in row.items():
            Q[r] += D[p] * w
    acc = np.minimum(A.reshape(-1), Q)
    out = np.zeros(len(rows_by_pos))
    for p, row in enumerate(rows_by_pos):
        for r, w in row.items():
            if Q[r] > 0.0:
                out[p] += D[p] * w * acc[r] / Q[r]
    return out, acc, Q


class SeamUptakeTests(Case):
    def setUp(self):
        self.cat = cat0()
        self.dom = state.Domain(12, 10, 1.0, periodic=TT)
        recs = [rec_row(1, 'grass', 0.1, 3.5, 0.1), rec_row(2, 'grass', 11.9, 3.5, 0.1), rec_row(3, 'grass', 6.5, 6.5, 0.1)]
        self.st = state.PlantState.from_records(self.cat, self.dom, recs)
        self.nonper = state.PlantState.from_records(self.cat, state.Domain(12, 10, 1.0), recs)

    def plan(self, st, supply, D, backend='numpy', builder='numpy'):
        m = roots.build_root_map(st, self.cat, max_edges=10_000, max_bytes=BIG, builder=builder, backend=backend)
        return m, uptake.propose(m, D, supply)

    def test_water_only_across_the_seam_is_reached_and_competition_is_proportional(self):
        L, N = self.cat.root_layers, self.dom.ncells
        supply = np.zeros((L, N))
        cell = 3 * 12 + 11                                                                  # east-edge cell: plant 1 (x = 0.1) reaches it ONLY by wrapping
        supply[0, cell] = 1.0e-6
        D = np.array([1.0, 2.0, 5.0])
        m, plan = self.plan(self.st, supply, D)
        rows = map_rows(m)
        pos = [rows[1], rows[2], rows[3]]
        want, acc, Q = oracle_uptake(pos, D, supply, L, N)
        fb = plan.accept_all()
        np.testing.assert_allclose(fb.delivered, want, rtol=1e-12, atol=1e-18)
        self.assertGreater(float(fb.delivered[0]), 0.0)                                      # plant 1 drinks across the seam
        self.assertGreater(float(fb.delivered[1]), 0.0)                                      # cross-seam competition with plant 2 on the same cell
        self.assertEqual(float(fb.delivered[2]), 0.0)                                        # the interior plant cannot reach it
        self.assertAlmostEqual(float(fb.delivered.sum()), float(np.asarray(fb.accepted).sum()), delta=1e-15)   # accepted = delivered in total
        self.assertTrue(fb.closure['closed'])
        np.testing.assert_allclose(float(fb.satisfaction[0]) * D[0] / want[0], 1.0, rtol=1e-12)
        # the SAME fraction of the local request is met for every competing plant (no priority, no compensation to other cells)
        w1, w2 = rows[1][cell], rows[2][cell]
        np.testing.assert_allclose(fb.delivered[0] / (D[0] * w1), fb.delivered[1] / (D[1] * w2), rtol=1e-12)
        # control: the finite domain has no seam: plant 1 gets nothing from that cell
        m0, plan0 = self.plan(self.nonper, supply, D)
        fb0 = plan0.accept_all()
        self.assertEqual(float(fb0.delivered[0]), 0.0)
        self.assertGreater(float(fb0.delivered[1]), 0.0)

    def test_dry_cells_zero_demand_and_one_record_per_plant(self):
        L, N = self.cat.root_layers, self.dom.ncells
        m, plan = self.plan(self.st, np.zeros((L, N)), np.array([1.0, 1.0, 0.0]))
        fb = plan.accept_all()
        self.assertEqual(float(np.asarray(fb.delivered).sum()), 0.0)
        np.testing.assert_allclose(fb.unmet, [1.0, 1.0, 0.0], rtol=1e-15)                   # dry cells: nothing is moved to wetter cells
        self.assertEqual(m.n_plants, 3)                                                      # one row per plant: no image rows
        np.testing.assert_array_equal(m.host_arrays()['ids'], [1, 2, 3])

    @unittest.skipUnless(HAVE_NUMBA, 'numba not installed')
    def test_numba_backend_matches_numpy(self):
        L, N = self.cat.root_layers, self.dom.ncells
        supply = np.zeros((L, N))
        supply[0, 3 * 12 + 11] = 1.0e-6
        supply[1, 3 * 12 + 0] = 2.0e-6
        D = np.array([1.0, 2.0, 5.0])
        _, p0 = self.plan(self.st, supply, D)
        _, p1 = self.plan(self.st, supply, D, backend='numba', builder='compiled')
        np.testing.assert_allclose(np.asarray(p1.proposed), np.asarray(p0.proposed), rtol=1e-12, atol=1e-18)
        np.testing.assert_allclose(p1.accept_all().delivered, p0.accept_all().delivered, rtol=1e-12, atol=1e-18)

    def test_cuda_backend_matches_numpy_under_the_existing_contract(self):
        if not _have_cuda():
            self.skipTest('no CUDA device / CuPy')
        import cupy as cp
        L, N = self.cat.root_layers, self.dom.ncells
        supply = np.zeros((L, N))
        supply[0, 3 * 12 + 11] = 1.0e-6
        D = np.array([1.0, 2.0, 5.0])
        m0, p0 = self.plan(self.st, supply, D)
        mc = roots.build_root_map(self.st, self.cat, max_edges=10_000, max_bytes=BIG, builder='compiled' if HAVE_NUMBA else 'numpy',
                                  backend='cuda')
        np.testing.assert_array_equal(mc.host_arrays()['resources'], m0.host_arrays()['resources'])
        pc_ = uptake.propose(mc, cp.asarray(D), cp.asarray(supply))
        np.testing.assert_allclose(cp.asnumpy(pc_.proposed), np.asarray(p0.proposed), rtol=1e-12, atol=1e-18)


# ------------------------------------------------------------------ 7. session / rebuild / checkpoint / sources
def periodic_session(periodic_flags=TT, backend='numpy', plants=None):
    cat = nf.make_catalog()
    dom = state.Domain(12, 10, 1.0, periodic=periodic_flags)
    plants = plants or [(11, 0.1, 3.5, 0.1), (12, 11.9, 3.5, 0.1), (13, 6.5, 6.5, 0.1)]
    recs = [{'plant_id': pid, 'species': 'grass', 'x_m': x, 'y_m': y, 'establishment_ground_m': 0.0, 'agb_dry_kg': b,
             'seasonal_agb_dry_kg': 0.0, 'age_days': 100.0, 'status': 'live'} for pid, x, y, b in plants]
    st = state.PlantState.from_records(cat, dom, recs)
    return nf.start_session(st, nf.make_config(cat, backend=backend), nf.anchor()), cat


class SessionTests(Case):
    def test_session_geometry_wraps_and_growth_rebuild_keeps_the_flags(self):
        s, cat = periodic_session()
        self.assertEqual(s.boundary()['periodic'], [True, True])
        self.assertEqual(s.export_roots()['boundary']['periodic'], [True, True])
        self.assertEqual(s.export_canopy()['boundary']['periodic'], [True, True])
        a = nf.anchor()
        for step in range(2):
            geo = s.pop.geometry
            self.assertEqual(geo.state.domain.periodic, (True, True))
            self.assertRowsEqual(map_rows(geo.root_map), expected_root_rows(geo.state, cat))
            cm, tot = row_of(geo.canopy_map, 0)
            self.assertIn(3 * 12 + 11, cm)                                                    # plant 11's canopy (r ~ 0.16 at x = 0.1) wraps
            rad = canopy.derive_geometry(geo.state, cat)['radius_m']
            self.assertAlmostEqual(tot, math.pi * float(rad[0]) ** 2, delta=1e-12)
            np.testing.assert_allclose(nf.to_host(geo.canopy_area_m2), math.pi * rad * rad, rtol=1e-14)
            if step == 0:
                for i in range(7):                                                            # the 7-day cadence grows once and rebuilds
                    s = nf.run_day(s, a, i)[0]
                self.assertEqual((s.counters()['growth_updates'], s.pop.generation), (1, 1))
        self.assertEqual(s.snapshot_state().domain.periodic, (True, True))

    def test_cross_seam_water_reaches_the_plant_through_the_real_session_path(self):
        for flags, expect in ((TT, True), (FF, False)):
            s, cat = periodic_session(flags)
            rm = s.pop.geometry.root_map
            supply = np.zeros((int(rm.n_layers), int(rm.n_cells)))
            supply[0, 3 * 12 + 11] = 1.0e-6
            child, plan, prop, st = nf.run_day(s, nf.anchor(), 0, supply=supply, publish=False)
            D = nf.to_host(plan.D)
            rows = map_rows(rm)
            want, acc, Q = oracle_uptake([rows[11], rows[12], rows[13]], D, supply, int(rm.n_layers), int(rm.n_cells))
            np.testing.assert_allclose(nf.to_host(st.feedback.delivered_m3), want, rtol=1e-9, atol=1e-18)
            self.assertEqual(float(nf.to_host(st.feedback.delivered_m3)[0]) > 0.0, expect)
            self.assertTrue(st.feedback.closure['closed'])
            self.assertEqual(sorted(st.feedback.live_ids.tolist()), [11, 12, 13])             # one record per plant

    def test_canopy_pet_demand_reads_across_the_seam(self):
        s, cat = periodic_session()
        n = s.pop.geometry.root_map.n_cells
        field = np.arange(n, dtype=np.float64) + 1.0
        plan = s.prepare_daily(nf.day_ctx(nf.anchor(), 0), terrain_pet=field)
        cm = s.pop.geometry.canopy_map
        pet = pp.canopy_pet(cm, field, max_bytes=BIG)['pet_mm_per_day']
        rows = [row_of(cm, p) for p in range(3)]
        for p in range(3):
            exp = sum(a * field[c] for c, a in rows[p][0].items()) / rows[p][1]
            self.assertAlmostEqual(float(pet[p]), exp, delta=1e-12 * exp)
        self.assertEqual(plan.pet_kind, 'canopy_area_weighted')


class CheckpointBoundaryTests(Case):
    LIM = pc.CheckpointLimits(1 << 26)

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.s0, self.cat = periodic_session()
        self.cfg = self.s0.config
        self.a = nf.anchor()
        s = self.s0
        for i in range(3):
            s = self.day(s, i)
        self.s = s
        self.dir = os.path.join(self.tmp, 'ck')

    def day(self, s, i, factor=1.0):
        kw = nf.next_read_args(s)
        plan = s.prepare_daily(nf.day_ctx(self.a, i), terrain_pet=nf.pet_field(s.pop.geometry.root_map.n_cells, 5.0))
        prop = s.propose_withdrawal(plan, nf.make_read(s, ordinal=i, run=nf.HEX_D, **kw))
        return s.publish(s.stage_day(prop, nf.make_accepted(prop, factor)))

    def load(self, **kw):
        return pc.load_biology(self.dir, self.cfg, bindings=pc.ExternalBindings(kw.pop('external', self.s.resource_cursor), **kw), limits=self.LIM)

    def rewrite_manifest(self, edit):
        p = os.path.join(self.dir, 'manifest.json')
        doc = json.load(open(p))
        doc.pop('manifest_sha256')
        edit(doc)
        doc['manifest_sha256'] = hashlib.sha256(pc._canon(doc)).hexdigest()
        json.dump(doc, open(p, 'w'))

    def test_round_trip_persists_the_flags_and_continues_exactly(self):
        man = pc.save_biology(self.dir, self.s, external=self.s.resource_cursor, limits=self.LIM)
        self.assertEqual(man['boundary']['periodic'], [True, True])
        meta = json.load(open(os.path.join(self.dir, 'plants.metadata.json')))
        self.assertEqual(meta['domain']['periodic'], [True, True])
        r = self.load()
        t = r.session
        self.assertEqual((t.token, t.geometry_sha256, t.pop_sha256), (self.s.token, self.s.geometry_sha256, self.s.pop_sha256))
        self.assertEqual(t.snapshot_state().domain.periodic, (True, True))
        self.assertEqual(t.boundary()['periodic'], [True, True])
        u = self.s
        for i in range(3, 8):                                                                 # crosses the weekly growth rebuild
            u, t = self.day(u, i), self.day(t, i)
        np.testing.assert_array_equal(np.asarray(u.pop.B), np.asarray(t.pop.B))
        self.assertEqual(u.geometry_sha256, t.geometry_sha256)
        self.assertEqual(t.counters()['growth_updates'], 1)

    def test_pending_stage_resumes_on_the_periodic_domain(self):
        plan = self.s.prepare_daily(nf.day_ctx(self.a, 3), terrain_pet=nf.pet_field(self.s.pop.geometry.root_map.n_cells, 5.0))
        prop = self.s.propose_withdrawal(plan, nf.make_read(self.s, ordinal=3, run=nf.HEX_D, **nf.next_read_args(self.s)))
        acc = nf.make_accepted(prop, 0.6)
        man = pc.save_biology(self.dir, self.s, external=acc.after, limits=self.LIM, pending=pc.PendingStage(plan, prop, acc))
        self.assertIsNotNone(man['pending'])
        direct = self.s.publish(self.s.stage_day(prop, acc))
        r = self.load(external=acc.after)
        plan2, prop2, acc2 = pc.rebuild_pending(r.session, r.pending)
        resumed = r.session.publish(r.session.stage_day(prop2, acc2))
        for k in 'DUAT':
            np.testing.assert_array_equal(resumed.pending()[k], direct.pending()[k])
        self.assertEqual(resumed.geometry_sha256, direct.geometry_sha256)

    def test_wrong_boundary_restore_is_refused_in_every_form(self):
        pc.save_biology(self.dir, self.s, external=self.s.resource_cursor, limits=self.LIM)
        with self.assertRaises((pc.PlantCheckpointError, state.StateError)):                  # a caller domain with other flags
            self.load(domain=state.Domain(12, 10, 1.0))
        self.rewrite_manifest(lambda d: d['boundary'].update(periodic=[False, False]))
        with self.assertRaises(pc.PlantCheckpointError) as cm:
            self.load()
        self.assertIn('boundary mismatch', str(cm.exception))
        shutil.rmtree(self.dir)
        pc.save_biology(self.dir, self.s, external=self.s.resource_cursor, limits=self.LIM)
        self.rewrite_manifest(lambda d: d.pop('boundary'))                                    # an "old" manifest means nonperiodic: never reinterpreted
        with self.assertRaises(pc.PlantCheckpointError):
            self.load()
        shutil.rmtree(self.dir)
        pc.save_biology(self.dir, self.s, external=self.s.resource_cursor, limits=self.LIM)
        self.rewrite_manifest(lambda d: d['boundary'].update(periodic=[1, 0]))                # not strict booleans
        with self.assertRaises(pc.PlantCheckpointError):
            self.load()

    def test_nonperiodic_checkpoint_has_no_boundary_block_and_still_restores(self):
        s, cat = periodic_session(FF)
        out = os.path.join(self.tmp, 'np')
        man = pc.save_biology(out, s, external=s.resource_cursor, limits=self.LIM)
        self.assertNotIn('boundary', man)
        meta = json.load(open(os.path.join(out, 'plants.metadata.json')))
        self.assertNotIn('periodic', meta['domain'])
        r = pc.load_biology(out, s.config, bindings=pc.ExternalBindings(s.resource_cursor), limits=self.LIM)
        self.assertEqual((r.session.token, r.session.geometry_sha256), (s.token, s.geometry_sha256))
        self.assertNotEqual(s.geometry_sha256, self.s0.geometry_sha256)                      # periodic and finite geometries are different identities

    def test_population_identity_binds_the_boundary_flags(self):
        a, _ = periodic_session(TT)
        b, _ = periodic_session(FF)
        c, _ = periodic_session(TF)
        self.assertEqual(len({a.pop_sha256, b.pop_sha256, c.pop_sha256}), 3)
        self.assertEqual(len({a.geometry_sha256, b.geometry_sha256}), 2)


class SourceHashTests(Case):
    def test_new_helper_is_in_the_source_hash_sets(self):
        self.assertIn('periodic', pc.SOURCE_MODULES)
        h = pc.source_hashes()
        sha = hashlib.sha256(open(ROOT / 'leaf' / 'periodic.py', 'rb').read()).hexdigest()
        self.assertEqual(h['periodic'], sha)
        try:
            from leaf import checkpoint as ck
        except Exception:                                                                    # pragma: no cover - heavy optional import
            self.skipTest('leaf.checkpoint cannot be imported here')
        self.assertIn('periodic', ck.SOURCE_MODULES)
        self.assertIsNotNone(ck.module_hashes()['periodic'])
        override = pc.SourceOverride(('periodic',), 'reviewed: periodic helper added for the boundary revision')
        self.assertEqual(override.modules, ('periodic',))

    def test_core_modules_do_not_import_numba_or_cupy_at_import_time(self):
        import subprocess
        code = ("import sys; import leaf.periodic, leaf.roots, leaf.plant_pet, leaf.canopy, leaf.initialization, leaf.plant_session; "
                "bad=[m for m in ('cupy','numba','maple_syrup','maple') if m in sys.modules]; sys.exit(1 if bad else 0)")
        r = subprocess.run([sys.executable, '-c', code], cwd=str(ROOT), capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)


# ------------------------------------------------------------------ 8. recruitment daughters
def load_rec_catalog(modify=None):
    with open(PHASE3, 'r', encoding='utf-8') as fh:
        doc = json.load(fh)
    if modify:
        modify(doc)
    return traits.SpeciesCatalog(doc)


def run_rec(cat, dom, rows, *, seed=7, counter=0):
    table = rec.compile_species(cat)
    idx = {sp: i for i, sp in enumerate(cat.species_ids)}
    la = dict(live_ids=np.array([r[0] for r in rows], dtype=np.int64), live_species=np.array([idx[r[1]] for r in rows], dtype=np.int32),
              live_x=np.array([r[2] for r in rows]), live_y=np.array([r[3] for r in rows]), live_B=np.array([r[4] for r in rows]),
              live_E=np.array([r[5] for r in rows]), live_age_days=np.array([r[6] for r in rows]))
    return rec.evaluate(table, rec.PatchConfig(4.0), dom, np.full(dom.shape, 100.0), max_existing_id=max(r[0] for r in rows),
                        n_total_records=len(rows), seed=seed, counter=counter, max_bytes=1 << 30, max_new_plants=100000,
                        max_total_records=1000000, max_attempts=1000000, **la)


class RecruitmentBoundaryTests(Case):
    def setUp(self):
        def always(doc):
            for sp in ('grass', 'forb'):
                doc['species'][sp]['reproduction']['traits']['clonal']['probability'] = 1.0
            for sp in ('shrub', 'forb'):
                doc['species'][sp]['reproduction']['traits']['seed']['establishment_probability'] = 0.0
            c = doc['species']['grass']['reproduction']['traits']['clonal']
            c['max_attempts_per_daughter'] = 1                    # a single draw per daughter: rejection is visible
            c['dispersal_radius_m'] = 0.6
            doc['species']['forb']['reproduction']['traits']['clonal']['probability'] = 0.0
        self.cat = load_rec_catalog(always)
        self.rows = [(100 + i, 'grass', x, y, 0.5, 0.1, 800.0) for i, (x, y) in enumerate(
            [(0.2, 4.0), (7.9, 4.0), (4.0, 0.1), (4.0, 7.95), (0.1, 0.1), (7.9, 7.9)] * 6)]
        self.rows = [(i + 1,) + r[1:] for i, r in enumerate(self.rows)]

    def test_periodic_daughters_wrap_instead_of_hitting_a_hard_wall(self):
        per = run_rec(self.cat, state.Domain(8, 8, 1.0, periodic=TT), self.rows)
        d = per['diagnostics']
        self.assertEqual(d['n_daughters_rejected'], 0)                                        # every single attempt is placed
        self.assertEqual(d['n_daughters_requested'], d['n_daughters_placed'])
        b = per['births']
        dom = state.Domain(8, 8, 1.0, periodic=TT)
        self.assertTrue(np.all(dom.contains(b['x_m'], b['y_m'])))
        self.assertEqual(len(set(b['ids'].tolist())), b['ids'].shape[0])                      # one new ID per daughter
        pos = {r[0]: (r[2], r[3]) for r in self.rows}
        crossed = 0
        for k in range(b['ids'].shape[0]):
            px, py = pos[int(b['parent_id'][k])]
            dx = periodic.min_image(b['x_m'][k] - px, 8.0)
            dy = periodic.min_image(b['y_m'][k] - py, 8.0)
            self.assertLessEqual(math.hypot(float(dx), float(dy)), 0.6 * (1 + 1e-9))          # wrapped distance within the dispersal radius
            crossed += abs(b['x_m'][k] - px) > 0.6 or abs(b['y_m'][k] - py) > 0.6
        self.assertGreater(crossed, 0)                                                         # some daughters really crossed a seam
        # ledger: donors lose exactly the transferred B (0.05 per placed daughter), daughters get it
        self.assertAlmostEqual(float(np.sum(0.5 - per['parent_B_new'])), 0.05 * d['n_daughters_placed'], places=10)
        self.assertTrue(d['ledger']['closed'])
        self.assertAlmostEqual(float(np.sum(b['agb_dry_kg'])), 0.05 * d['n_daughters_placed'], places=12)
        self.assertTrue(any('periodic' in c for c in d['caveats']))
        self.assertFalse(any('never wrapped' in c for c in d['caveats']))

    def test_nonperiodic_daughters_still_reject_at_the_wall_with_unchanged_draws(self):
        dom = state.Domain(8, 8, 1.0)
        res = run_rec(self.cat, dom, self.rows)
        self.assertGreater(res['diagnostics']['n_daughters_rejected'], 0)
        self.assertTrue(np.all(dom.contains(res['births']['x_m'], res['births']['y_m'])))
        again = run_rec(self.cat, state.Domain(8, 8, 1.0, periodic=FF), self.rows)
        for k in ('x_m', 'y_m', 'ids', 'agb_dry_kg'):
            np.testing.assert_array_equal(res['births'][k], again['births'][k])
        self.assertTrue(any('never wrapped' in c for c in res['diagnostics']['caveats']))
        # interior parents: periodic and finite results are identical (no coordinate is changed by the explicit wrap)
        inner = [(i + 1, 'grass', 4.0 + 0.01 * i, 4.0, 0.5, 0.1, 800.0) for i in range(20)]
        a = run_rec(self.cat, state.Domain(8, 8, 1.0), inner)
        b = run_rec(self.cat, state.Domain(8, 8, 1.0, periodic=TT), inner)
        np.testing.assert_array_equal(a['births']['x_m'], b['births']['x_m'])
        np.testing.assert_array_equal(a['births']['y_m'], b['births']['y_m'])

    def test_mixed_axes_wrap_only_the_periodic_axis(self):
        dom = state.Domain(8, 8, 1.0, periodic=TF)
        res = run_rec(self.cat, dom, self.rows)
        self.assertTrue(np.all(dom.contains(res['births']['x_m'], res['births']['y_m'])))
        d = res['diagnostics']
        self.assertGreater(d['n_daughters_rejected'], 0)                                       # y is a wall; x never rejects by itself
        pos = {r[0]: (r[2], r[3]) for r in self.rows}
        for k in range(res['births']['ids'].shape[0]):
            py = pos[int(res['births']['parent_id'][k])][1]
            self.assertLessEqual(abs(res['births']['y_m'][k] - py), 0.6 * (1 + 1e-9))        # no wrap in y


# ------------------------------------------------------------------ 9. optional REAL native N15 case (periodic in both axes)
@unittest.skipUnless(nf.native_configured(), 'native SYRUP environment not configured (LEAF_NATIVE_BUNDLE / LEAF_NATIVE_CASE_DIR)')
class NativePeriodicN15Tests(Case):
    '''A genuine native daily day driven through a PERIODIC LEAF session on the native grid (no fake water state): the visible call-site
    sequence is unchanged; SYRUP owns the water and the debit, LEAF only proposes from wrapped roots and applies the accepted result.'''

    @classmethod
    def setUpClass(cls):
        nf.require_native()                                  # a configured but unusable native source is a setup FAILURE
        from maple_syrup import daily_water as dw
        from maple_syrup import water_lifecycle as wl
        from leaf import lifecycle_bridge as lb
        cls.dw, cls.wl, cls.lb = dw, wl, lb
        cls.tmp = tempfile.mkdtemp()
        cls.lop, cls.boundary, cls.profile = nf.native_daily_harness(cls.tmp)
        cls.src = lb.bind_source(cls.profile.syrup_package_digest)
        cls.xp = cls.lop.xp
        cls.backend = nf.native_backend()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_periodic_leaf_session_on_the_native_periodic_grid_keeps_one_authoritative_debit(self):
        dw, wl, lb = self.dw, self.wl, self.lb
        lop = self.lop
        s0 = dw.start_daily_lineage(lop, self.boundary, expected_parent=wl.expect_parent(self.boundary), run_digest=nf.HEX_D)
        plan = dw.next_plan(s0, 2.0, 10.0, 5.0)
        res = dw.daily_water_action(lop, s0, expected_parent=wl.expect_parent(s0), plan=plan,
                                    pet_terrain_m_s=self.xp.full(lop.n_cells, 1e-8, dtype=np.float64))
        self.assertEqual(res.status, 'complete')
        day = res.state
        ny, nx = lop.shape
        dx = float(lop.topology.dx_m)
        cat = nf.make_catalog(n_layers=int(lop.n_layers))
        dom = state.Domain(nx, ny, dx, periodic=TT)
        recs = [{'plant_id': 100 + k, 'species': 'grass', 'x_m': x, 'y_m': y, 'establishment_ground_m': 0.0, 'agb_dry_kg': 0.1,
                 'seasonal_agb_dry_kg': 0.0, 'age_days': 100.0, 'status': 'live'}
                for k, (x, y) in enumerate([(0.1 * dx, 0.5 * ny * dx), (nx * dx - 0.1 * dx, 0.5 * ny * dx), (0.5 * nx * dx, 0.5 * dx)])]
        dev = int(os.environ.get('LEAF_NATIVE_DEVICE', '0')) if self.backend == 'cuda' else None
        cfg = nf.make_config(cat, backend=self.backend, device=dev)
        sess = ps.PlantSession.start(state.PlantState.from_records(cat, dom, recs), cfg, anchor=nf.anchor(),
                                     day_binding=ps.DayBinding(nf.HEX_D, nf.anchor().date().isoformat()),
                                     initial_resource=lb.resource_identity(lop, s0))
        self.assertEqual(sess.boundary()['periodic'], [True, True])
        self.assertEqual(sess.pop.geometry.root_map.provenance['boundary']['periodic'], [True, True])
        pet = lb.pet_m_s_to_mm_per_day(self.xp.full(lop.n_cells, 1e-8, dtype=np.float64))
        lplan = sess.prepare_daily(nf.day_ctx(nf.anchor(), 0), terrain_pet=pet)
        rd = lb.read_availability(self.src, lop, day, max_bytes=1 << 30)
        prop = sess.propose_withdrawal(lplan, rd)
        depth = lb.request_depth_m(rd, prop.proposed_m3, max_bytes=1 << 30)
        new_day, cand = wl.request_withdrawal(lop, day, **lb.request_args(rd, lop, day), depth_m=depth, identity=prop.identity,
                                              retained_feedback_bytes=prop.retained_feedback_bytes)
        acc = lb.accepted_withdrawal(rd, prop.identity, new_day, cand, lop=lop, max_bytes=1 << 30)
        staged = sess.stage_day(prop, acc)
        child = sess.publish(staged)
        fb = staged.feedback
        self.assertLessEqual(abs(fb.totals['delivered_m3'] - acc.actual_total_m3), float(fb.closure['tolerance_m3']))
        audit = wl.audit_lifecycle(lop, new_day)
        self.assertLessEqual(abs(audit['residual_m3']), audit['bound_m3'])
        self.assertEqual(child.counters()['withdrawals_accepted'], 1)
        self.assertEqual(sorted(fb.live_ids.tolist()), [100, 101, 102])                      # one record per plant


if __name__ == '__main__':
    unittest.main()
