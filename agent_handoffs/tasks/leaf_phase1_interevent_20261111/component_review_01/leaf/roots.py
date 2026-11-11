'''leaf.roots: build a plant-major CSR root map (leaf.uptake.RootMap) from the authoritative PlantState + SpeciesCatalog.

Geometry comes from the ONE shared allometry (leaf.canopy.derive_geometry: canopy radius from the species
power law, root radius = lateral_radius_multiplier * canopy radius); nothing here duplicates it.

Horizontal law (LABEL: existing simple geometric approximation). A plant's root circle is mapped to every
raster cell it POSITIVELY overlaps (closest point of the closed square strictly inside the circle; tangent
contact excluded), clipped to the finite non-periodic domain, with UNIFORM weight 1 / (number of overlapped
cells), normalised once at build time. This is the predicate and normalisation of the existing compiled
builder `benchmarks.spatial_mapping` (reused for builder='compiled'; a NumPy reference with the identical
arithmetic serves builder='numpy'). It is NOT a measured root-density or area-weighted law: a cell barely
touched by the circle gets the same weight as the cell under the stem. Replace by rebuilding the map with
a different law; do not reinterpret these weights as root biomass.

Vertical law. Each plant's horizontal weights are multiplied by its layer fractions (top layer first):
the species `roots.layer_fractions` by default, or an explicitly ID-keyed override row per individual
(`layer_fraction_overrides={plant_id: [L fractions]}`, all layers, finite, nonnegative, unit sum). Edges are
ordered layer-major within a plant; layers with fraction 0 produce no edges.

Fixed weights. Root weights are fixed for the life of the map: uptake never rebalances them; growth-driven
rebuilds (new radii) are a later driver operation that constructs a NEW map. No compensatory renormalisation
for dry / masked / unavailable cells: an unavailable cell gets supply A = 0 in the uptake proposal and the
request mapped to it stays unmet (leaf.uptake); its weight is never redistributed to active or wetter cells.

Selection. Default `statuses=('live',)`: removed and dead_standing plants do not transpire by default; any
explicit status set is mapped consistently (selected plants with zero root radius, e.g. removed plants with
B = 0, get an inactive edge-less row on which positive demand is refused by leaf.uptake). Callers may instead
supply a fully custom general map through leaf.uptake.RootMap.from_arrays (one row per individual, any
non-separable weights, typed arrays) and skip this builder entirely.

Grid support. The shared compiled helper supports SQUARE cell spacing (dx == dy) in a [0, nx*dx] x [0, ny*dy]
domain-local frame; non-square domains are REFUSED explicitly before any allocation (dx is never silently used
for dy). Positions are converted from the landscape frame by subtracting the domain origin.

Ownership. The builder creates FRESH contiguous int64 / float64 arrays that nothing else references and hands
them to `leaf.uptake.adopt_owned_arrays` (internal trusted transfer: validated in place, frozen read-only, no
copy). No caller array (state arrays, catalog tables, override dicts) can become mutable map state.

Memory admission (explicit max_edges / max_bytes). Each stage admits what it is about to allocate TOGETHER
WITH everything still live, through a ledger, and refuses before the allocation (population never truncated,
limits never raised):
  roots.preflight          status mask + derive_geometry bound + catalog tables + map resource scratch lower bound
  roots.select             selected rows + geometry peak + extracted ids / radius / species index / layer table
  roots.overrides          override python/array scratch + local sorted ID lookup (the state's persistent by-ID
                           index cache is deliberately NOT created: no unaccounted resident array)
  roots.horizontal_inputs  active index + local coordinates + radius of plants with roots
  roots.horizontal_*       compiled: the shared helper's own pre/post-count admission under the budget left
                           after the live work (its own estimate also counts a (layers+1)*N grid term it does
                           not allocate: stricter, reported verbatim); numpy: exact bounding-grid scratch
                           (max bbox) BEFORE any per-plant array, then edge arrays after the count
  roots.horizontal_map     per-plant counts / offsets over the selected rows
  roots.expand_count       layer positivity mask + counts used to size the expansion
  roots.expand_layers      expanded owned arrays + expansion scratch + the horizontal arrays still live
  uptake.adopt_owned_arrays  map validation scratch (+ device staging for cuda) over the live owned arrays
SCOPE: live algorithm arrays of this builder. derive_geometry is a black box whose peak is bounded by
192 bytes per plant (118 for its 15 outputs including two object-pointer arrays plus allometry, validation and
position-check temporaries). EXCLUDED: caller-held state / catalog / override containers, interpreter objects,
allocator-pool overhead, numba JIT. Importing this module imports NumPy only; numba (compiled builder) and cupy
(cuda backend) load lazily.
'''
from __future__ import annotations

import math

import numpy as np

from leaf import canopy as _canopy
from leaf import state as _state
from leaf import traits as _traits
from leaf import uptake as _uptake

VERSION = '0.2.0'
BUILDERS = ('compiled', 'numpy')
HORIZONTAL_LAW = 'uniform_positive_overlap_v1'
HORIZONTAL_LAW_LABEL = ('existing simple geometric approximation: uniform weight over every raster cell the root circle '
                        'positively overlaps, clipped to the finite non-periodic domain, normalised once at build time; '
                        'NOT measured root density and NOT area weighting')
VERTICAL_LAW_LABEL = ('species roots.layer_fractions (top first) or explicit ID-keyed per-individual override rows; '
                      'weights = horizontal uniform weight x layer fraction; fixed until a new map is built')
FIXED_WEIGHTS_NOTE = ('root weights are fixed during uptake: no plasticity, no compensation, no renormalisation of '
                      'dry / masked / unavailable cells (their requests stay unmet); growth-driven rebuilds create a '
                      'NEW map in a later driver operation')
MEMORY_SCOPE = ('live builder arrays per stage (ledger); derive_geometry bounded by 192 bytes per plant; excluded: '
                'caller-held state / catalog / override containers, interpreter objects, allocator pools, numba JIT')
GEOMETRY_BYTES_PER_PLANT = 192        # derive_geometry: 118 output + <= 74 temporaries (conservative bound)
MASK_BYTES_PER_PLANT = 3              # status mask + np.isin temporaries
SEGMENT_BYTES = 56                    # per (plant, positive layer) segment arrays during layer expansion
NUMPY_BBOX_BYTES_PER_CELL = 25        # float64 sum 8 + bool 1 (count) ; + int64 grid 8 + selection 8 (fill)
NUMPY_BBOX_VECTOR_BYTES = 128         # per row + column of the bounding box (ii, jj, x0, x1, cx, ddx, squares, ...)


class RootsError(ValueError):
    '''Any root-map construction error raised before the map exists (inputs untouched).'''


# ------------------------------------------------------------------ ledger of simultaneously live builder arrays
class _Ledger:
    '''Tracks live bytes by name and admits each stage's transient allocations together with the live set.'''

    def __init__(self, max_edges, max_bytes):
        self.max_edges, self.max_bytes = int(max_edges), int(max_bytes)
        self.live = {}
        self.reports = []

    @property
    def live_bytes(self):
        return int(sum(self.live.values()))

    def hold(self, name, nbytes):
        self.live[name] = int(nbytes)

    def release(self, *names):
        for n in names:
            self.live.pop(n, None)

    def admit(self, stage, transient, *, n_edges=0, scope=''):
        items = {k: int(v) for k, v in transient.items()}
        est = {'stage': stage, 'n_edges': int(n_edges), 'total_bytes': int(sum(items.values())), 'transient': items,
               'live_items': dict(self.live), 'scope': scope or 'transient allocations of this stage over the live set'}
        rep = _uptake.admit(stage, est, max_edges=self.max_edges, max_bytes=self.max_bytes,
                            resident_bytes=self.live_bytes)
        self.reports.append({'stage': stage, 'n_edges': int(n_edges), 'transient_bytes': est['total_bytes'],
                             'resident_bytes': self.live_bytes, 'admitted_total_bytes': rep['admitted_total_bytes'],
                             'transient': items, 'live_items': dict(self.live)})
        return self.reports[-1]


# ------------------------------------------------------------------ NumPy reference of the compiled circle-overlap builder
def _cell_range(c, r, dx, n):
    lo = math.floor((c - r) / dx)
    hi = math.floor((c + r) / dx)
    if lo < 0.0:
        lo = 0.0
    if hi > n - 1.0:
        hi = n - 1.0
    return int(lo), int(hi)


def _bbox(px, py, r, nx, ny, dx):
    i0, i1 = _cell_range(px, r, dx, nx)
    j0, j1 = _cell_range(py, r, dx, ny)
    return i0, i1, j0, j1


def _overlap_mask(px, py, r, nx, ny, dx):
    '''(ii, jj, mask) of the bounding grid: mask[j, i] is True where the closest point of the closed square to the
    centre is strictly inside the circle (identical predicate / arithmetic to benchmarks.spatial_mapping._overlaps).
    None when the bounding box is empty.'''
    i0, i1, j0, j1 = _bbox(px, py, r, nx, ny, dx)
    if i1 < i0 or j1 < j0:
        return None
    ii = np.arange(i0, i1 + 1, dtype=np.int64)
    jj = np.arange(j0, j1 + 1, dtype=np.int64)
    x0 = ii * dx
    x1 = (ii + 1) * dx
    y0 = jj * dx
    y1 = (jj + 1) * dx
    cx = np.where(px < x0, x0, np.where(px > x1, x1, px))
    cy = np.where(py < y0, y0, np.where(py > y1, y1, py))
    ddx = px - cx
    ddy = py - cy
    mask = (ddx * ddx)[None, :] + (ddy * ddy)[:, None] < r * r
    return ii, jj, mask


def _overlap_count(px, py, r, nx, ny, dx):
    out = _overlap_mask(px, py, r, nx, ny, dx)
    return 0 if out is None else int(np.count_nonzero(out[2]))


def _overlap_cells(px, py, r, nx, ny, dx):
    '''Row-major cell indices (j * nx + i), j outer / i inner (same order as the compiled builder).'''
    out = _overlap_mask(px, py, r, nx, ny, dx)
    if out is None:
        return np.zeros(0, dtype=np.int64)
    ii, jj, mask = out
    return (jj[:, None] * nx + ii[None, :])[mask]


def numpy_bbox_scratch_bytes(x, y, root_radius, nx, ny, dx):
    '''Scalar-only pass (no arrays): (bytes of per-plant scratch for the LARGEST bounding box, max cells, max width,
    max height). The scratch is admitted before any per-plant bounding-grid array exists.'''
    bmax = wmax = hmax = 0
    for p in range(int(x.shape[0])):
        i0, i1, j0, j1 = _bbox(float(x[p]), float(y[p]), float(root_radius[p]), nx, ny, dx)
        w, h = max(i1 - i0 + 1, 0), max(j1 - j0 + 1, 0)
        bmax, wmax, hmax = max(bmax, w * h), max(wmax, w), max(hmax, h)
    return NUMPY_BBOX_BYTES_PER_CELL * bmax + NUMPY_BBOX_VECTOR_BYTES * (wmax + hmax), bmax, wmax, hmax


def horizontal_edges_numpy(x, y, root_radius, nx, ny, dx, *, max_edges, max_bytes, resident_bytes=0):
    '''NumPy reference horizontal CSR (uniform weights 1/count): offsets [P+1] int64, cells [E] int64,
    weights [E] float64, counts [P]. Admission BEFORE allocation: (1) the per-plant scratch of the largest
    bounding box (+ counts) over `resident_bytes`, (2) after the count pass and before any edge array, counts +
    offsets + edge arrays + scratch. O(P) Python iterations with vectorised cell tests (reference / fallback;
    production uses the compiled builder).'''
    P = int(x.shape[0])
    scratch, bmax, wmax, hmax = numpy_bbox_scratch_bytes(x, y, root_radius, nx, ny, dx)
    pre = {'stage': 'horizontal_numpy_scratch', 'n_edges': 0, 'total_bytes': 8 * P + scratch,
           'transient': {'counts': 8 * P, 'bbox_scratch': scratch}, 'max_bbox_cells': bmax,
           'scope': 'counts + scratch of the largest bounding grid (25 B/cell + 128 B/row+column)'}
    _uptake.admit('roots.horizontal_numpy_scratch', pre, max_edges=max_edges, max_bytes=max_bytes,
                  resident_bytes=resident_bytes)
    counts = np.zeros(P, dtype=np.int64)
    for p in range(P):
        counts[p] = _overlap_count(float(x[p]), float(y[p]), float(root_radius[p]), nx, ny, dx)
    E = int(counts.sum())
    est = {'stage': 'horizontal_numpy', 'n_edges': E, 'total_bytes': 8 * P + 8 * (P + 1) + 16 * E + scratch,
           'transient': {'counts': 8 * P, 'offsets': 8 * (P + 1), 'cells': 8 * E, 'weights': 8 * E, 'bbox_scratch': scratch},
           'scope': 'counts + offsets + int64 cells + float64 weights of the horizontal CSR + fill-pass scratch'}
    _uptake.admit('roots.horizontal_numpy', est, max_edges=max_edges, max_bytes=max_bytes, resident_bytes=resident_bytes)
    offsets = np.zeros(P + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    cells = np.empty(E, dtype=np.int64)
    weights = np.empty(E, dtype=np.float64)
    for p in range(P):
        c = _overlap_cells(float(x[p]), float(y[p]), float(root_radius[p]), nx, ny, dx)
        o = int(offsets[p])
        cells[o:o + c.shape[0]] = c
        if c.shape[0]:
            weights[o:o + c.shape[0]] = 1.0 / c.shape[0]
    return {'offsets': offsets, 'cells': cells, 'weights': weights, 'counts': counts, 'n_edges': E,
            'estimate': est, 'builder': 'numpy'}


def horizontal_edges_compiled(x, y, root_radius, nx, ny, dx, *, max_edges, max_bytes):
    '''The existing compiled circle-overlap CSR builder (benchmarks.spatial_mapping.build_root_graph_cpu with
    layers=1, numba). Typed contiguous float64 arrays only; scalar / type / shape / budget preflight runs BEFORE
    any allocation (the ids array included). `max_bytes` is the budget REMAINING after the builder's other live arrays; the helper
    counts its own inputs / graph (and a grid term it does not allocate) against it. Cells stay int32 (no
    conversion copy). Requires at least one plant with a positive radius (callers handle the empty case).'''
    try:
        from benchmarks import spatial_mapping as sm
    except Exception as exc:  # numba (or the benchmarks package) is unavailable
        raise _uptake.BackendUnavailableError(f'compiled root builder unavailable (benchmarks.spatial_mapping import '
                                              f'failed: {exc!r}); use builder="numpy" explicitly') from exc
    # ---- preflight BEFORE any P-sized allocation or shared-helper validation: typed arrays only (no conversion,
    # no materialised sequences), scalar grid check, then admission of the helper's own estimate plus its
    # validation scratch (np.unique sort 8n x2, finite / range comparison masks ~6n) over the budget.
    for name, a in (('x', x), ('y', y), ('root_radius', root_radius)):
        if not isinstance(a, np.ndarray) or a.dtype != np.float64 or a.ndim != 1:
            raise RootsError(f'{name} must be a 1-D float64 numpy.ndarray (no sequence materialisation or cast)')
        if not a.flags.c_contiguous:
            raise RootsError(f'{name} must be C-contiguous (a non-contiguous copy would be an unadmitted allocation)')
    n = int(x.shape[0])
    if int(y.shape[0]) != n or int(root_radius.shape[0]) != n or n < 1:
        raise RootsError('x, y, root_radius must share one length >= 1')
    try:
        grid = sm.Grid(int(nx), int(ny), float(dx))
    except ValueError as exc:
        raise RootsError(f'compiled root builder refused its grid: {exc}') from exc
    est = sm.estimate_bytes(n, grid.ncells, 1)
    pre = {'stage': 'horizontal_compiled_preflight', 'n_edges': 0, 'total_bytes': int(est['total']) + 22 * n,
           'transient': {'helper_inputs_ids_graph_grid': int(est['total']), 'helper_validation_scratch': 22 * n},
           'scope': 'benchmarks.spatial_mapping.estimate_bytes (incl. its ids arange and a grid term not allocated '
                    'here) + validation scratch (sort 16n, masks 6n), admitted before the ids array exists'}
    _uptake.admit('roots.horizontal_compiled', pre, max_edges=max_edges, max_bytes=max(int(max_bytes), 0))
    try:
        inputs = {'x': np.ascontiguousarray(x, dtype=np.float64), 'y': np.ascontiguousarray(y, dtype=np.float64),
                  'canopy_radius': np.ascontiguousarray(root_radius, dtype=np.float64),
                  'root_radius': np.ascontiguousarray(root_radius, dtype=np.float64),
                  'ids': np.arange(n, dtype=np.int64)}
        g = sm.build_root_graph_cpu(inputs, grid, 1, int(max_edges), int(max(max_bytes, 0)))
    except sm.AdmissionRejected as exc:
        raise _uptake.AdmissionError('roots.horizontal_compiled', str(exc), dict(exc.estimate, stage=exc.stage,
                                     scope='benchmarks.spatial_mapping.estimate_bytes: inputs + CSR graph + its '
                                           '(layers+1)*N float64 grid term; reported verbatim')) from exc
    except ValueError as exc:
        raise RootsError(f'compiled root builder refused its inputs: {exc}') from exc
    return {'offsets': g.offsets, 'cells': g.cells, 'weights': g.weights, 'counts': g.counts,
            'n_edges': int(g.cells.shape[0]),
            'estimate': dict(g.estimate, stage='horizontal_compiled',
                             scope='benchmarks.spatial_mapping.estimate_bytes (inputs + CSR + its (layers+1)*N grid term)'),
            'builder': 'compiled', 'timings': dict(g.timings)}


# ------------------------------------------------------------------ layer expansion (vectorised, admitted first)
def expansion_plan(h_counts, layer_table):
    '''(E, n_segments) of the layered map from P-sized inputs only: E = sum_p cells_p * positive_layers_p and
    n_segments = number of (plant, positive layer) pairs.'''
    pos = layer_table > 0.0
    m = np.count_nonzero(pos, axis=1).astype(np.int64)
    return int(np.sum(h_counts.astype(np.int64) * m)), int(np.count_nonzero(pos))


def expanded_edge_count(h_counts, layer_table):
    return expansion_plan(h_counts, layer_table)[0]


def expand_layer_edges(h_offsets, h_cells, h_weights, layer_table, n_cells):
    '''(offsets, resources, weights) of the layered map: per plant, layer-major over positive-fraction layers,
    horizontal cell order within a layer; resource = layer * n_cells + cell; weight = fraction * horizontal.
    Fresh C-contiguous int64 / int64 / float64 arrays. Transient scratch: one int64 edge index and one
    repeat / gather temporary (16 bytes per expanded edge) plus the segment arrays.'''
    P, L = layer_table.shape
    k = np.diff(h_offsets).astype(np.int64)
    pos = layer_table > 0.0
    counts = k * np.count_nonzero(pos, axis=1).astype(np.int64)
    offsets = np.zeros(P + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    del counts
    E = int(offsets[-1])
    seg_plant, seg_layer = np.nonzero(pos)                  # row-major: plant, then layer
    seg_len = k[seg_plant]
    seg_off = np.zeros(seg_len.shape[0] + 1, dtype=np.int64)
    np.cumsum(seg_len, out=seg_off[1:])
    hedge = np.arange(E, dtype=np.int64)
    hedge -= np.repeat(seg_off[:-1], seg_len)
    hedge += np.repeat(h_offsets[:-1][seg_plant], seg_len)
    resources = np.repeat(seg_layer.astype(np.int64) * np.int64(n_cells), seg_len)
    resources += h_cells[hedge]
    weights = np.repeat(layer_table[seg_plant, seg_layer], seg_len)
    weights *= h_weights[hedge]
    return offsets, resources, weights


# ------------------------------------------------------------------ overrides
def _validate_overrides(overrides, ids_sel, L):
    '''{plant_id: [L fractions]} -> (positions in the selection [n], table [n, L]); every ID must be a selected
    plant. Uses a LOCAL sorted lookup over the selected IDs (admitted by the ledger); it never touches the
    state's persistent by-ID index cache, so no unaccounted resident array is created on the caller's state.'''
    if not isinstance(overrides, dict) or not overrides:
        raise RootsError('layer_fraction_overrides must be a non-empty dict {plant_id: [root_layers fractions]} or None')
    ids, rows = [], []
    for pid, row in overrides.items():
        if isinstance(pid, (bool, np.bool_)) or not isinstance(pid, (int, np.integer)):
            raise RootsError(f'layer_fraction_overrides keys must be integer plant IDs, got {pid!r}')
        if isinstance(row, (str, bytes, dict)) or not hasattr(row, '__len__'):
            raise RootsError(f'layer_fraction_overrides[{pid}] must be a sequence of {L} fractions')
        if len(row) != L:
            raise RootsError(f'layer_fraction_overrides[{pid}] has {len(row)} entries, expected root_layers={L} (all layers required)')
        try:
            vals = [_traits.nonnegative(v, f'layer_fraction_overrides[{pid}][{i}]') for i, v in enumerate(row)]
        except _traits.CatalogError as exc:
            raise RootsError(str(exc)) from exc
        tot = math.fsum(vals)
        if abs(tot - 1.0) > _traits.LAYER_FRACTION_TOL:
            raise RootsError(f'layer_fraction_overrides[{pid}] must sum to 1 within {_traits.LAYER_FRACTION_TOL} '
                             f'(got {tot!r}); not normalised silently')
        ids.append(int(pid))
        rows.append(vals)
    q = np.array(ids, dtype=np.int64)
    if np.unique(q).shape[0] != q.shape[0]:
        raise RootsError('layer_fraction_overrides: duplicate plant IDs')
    order = np.argsort(ids_sel, kind='stable')
    sorted_ids = ids_sel[order]
    pos = np.searchsorted(sorted_ids, q)
    pos_c = np.minimum(pos, max(sorted_ids.shape[0] - 1, 0))
    found = (pos < sorted_ids.shape[0]) & (sorted_ids[pos_c] == q) if sorted_ids.shape[0] else np.zeros(q.shape, bool)
    if not found.all():
        raise RootsError(f'layer_fraction_overrides target unknown plants or plants outside the selected statuses: ids '
                         f'{q[~found][:10].tolist()}')
    return order[pos_c], np.array(rows, dtype=np.float64).reshape(len(ids), L)


# ------------------------------------------------------------------ builder
def build_root_map(state, catalog, *, max_edges, max_bytes, backend='numpy', builder='compiled',
                   statuses=('live',), layer_fraction_overrides=None, provenance=None):
    '''RootMap for the selected plants of `state` (see the module docstring). Pure with respect to the state
    and catalog. Returns a leaf.uptake.RootMap whose rows are the selected plants in state row order and whose
    `provenance` records builder, laws, selection, per-stage admission reports and the catalog hash / clock.'''
    if builder not in BUILDERS:
        raise RootsError(f'builder must be one of {BUILDERS}, got {builder!r}')
    if backend not in _uptake.BACKENDS:
        raise RootsError(f'backend must be one of {_uptake.BACKENDS}, got {backend!r}')
    if not isinstance(state, _state.PlantState):
        raise RootsError('state must be a leaf.state.PlantState')
    if not isinstance(catalog, _traits.SpeciesCatalog):
        raise RootsError('catalog must be a leaf.traits.SpeciesCatalog')
    dom = state.domain
    if not dom.square_cells:
        raise RootsError(f'non-square cells (dx_m={dom.dx_m!r}, dy_m={dom.dy_m!r}) are not supported by the shared '
                         'circle-overlap builder (square spacing only); refused before any allocation, dx is never '
                         'silently used for dy')
    try:
        me = _traits.strict_int(max_edges, 'max_edges', 0)
        mb = _traits.strict_int(max_bytes, 'max_bytes', 0)
    except _traits.CatalogError as exc:
        raise RootsError(str(exc)) from exc
    codes = sorted(set(int(c) for c in _state.status_codes(list(statuses))))
    if layer_fraction_overrides is not None and (not isinstance(layer_fraction_overrides, dict)
                                                 or not layer_fraction_overrides):
        raise RootsError('layer_fraction_overrides must be a non-empty dict {plant_id: [root_layers fractions]} or None')
    n_ov = 0 if layer_fraction_overrides is None else len(layer_fraction_overrides)
    L, N = catalog.root_layers, dom.ncells
    P_all, S = state.n, catalog.n_species
    led = _Ledger(me, mb)
    # 1. preflight from scalars only (P, S, L, N): nothing P- or N-sized exists yet
    led.admit('roots.preflight', {
        'status_mask': MASK_BYTES_PER_PLANT * P_all,
        'derive_geometry_bound': GEOMETRY_BYTES_PER_PLANT * P_all,
        'catalog_tables': 8 * S * (L + 8),
        'map_resource_scratch_lower_bound': 8 * L * N},
        scope='necessary lower bound from P, S, L, N before any array exists')
    led.hold('status_mask', P_all)
    mask = np.isin(state.status, codes)
    K = int(np.count_nonzero(mask))
    # 2. selection, geometry (the one shared allometry) and extraction of the selected plants' arrays
    led.admit('roots.select', {
        'selected_rows': 8 * K,
        'derive_geometry_bound': GEOMETRY_BYTES_PER_PLANT * P_all,
        'catalog_tables': 8 * S * (L + 8),
        'extracted_ids': 8 * K, 'extracted_root_radius': 8 * K,
        'species_index_int64': 12 * K, 'layer_table_gather': 8 * K * L,
        'radius_positive_mask': K})
    sel = np.flatnonzero(mask)
    led.release('status_mask')
    del mask
    led.hold('selected_rows', 8 * K)
    geo = _canopy.derive_geometry(state, catalog)
    radius = geo['root_radius_m'][sel]
    del geo
    ids = np.array(state.ids[sel], dtype=np.int64, copy=True)
    sp = state.species_index[sel].astype(np.int64)
    table = catalog.root_layer_table()[sp]
    del sp
    led.hold('ids', 8 * K)
    led.hold('root_radius', 8 * K)
    led.hold('layer_table', 8 * K * L)
    # 3. explicit per-individual layer overrides
    n_ov_applied = 0
    if n_ov:
        led.admit('roots.overrides', {
            'override_python_and_array_scratch': n_ov * (128 + 40 * L) + 8 * n_ov * L,
            'local_sorted_lookup': 16 * K + 40 * n_ov})
        ov_idx, ov_rows = _validate_overrides(layer_fraction_overrides, ids, L)
        table[ov_idx] = ov_rows
        n_ov_applied = int(ov_idx.shape[0])
        del ov_idx, ov_rows
    # 4. horizontal CSR for the plants with a positive root radius (domain-local, clipped to the closed domain)
    Ka = int(np.count_nonzero(radius > 0.0))
    led.admit('roots.horizontal_inputs', {
        'active_index': 8 * Ka, 'active_rows': 8 * Ka, 'x_local': 8 * Ka, 'y_local': 8 * Ka,
        'root_radius_active': 8 * Ka, 'radius_positive_mask': K})
    active = np.flatnonzero(radius > 0.0)
    led.hold('active_index', 8 * Ka)
    if Ka:
        W, H = float(dom.nx) * float(dom.dx_m), float(dom.ny) * float(dom.dx_m)
        sa = sel[active]
        xl = state.x_m[sa]
        xl -= dom.origin_x_m
        np.maximum(xl, 0.0, out=xl)
        np.minimum(xl, W, out=xl)
        yl = state.y_m[sa]
        yl -= dom.origin_y_m
        np.maximum(yl, 0.0, out=yl)
        np.minimum(yl, H, out=yl)
        rr = radius[active]
        del sa
        if builder == 'numpy':
            led.hold('x_local', 8 * Ka)
            led.hold('y_local', 8 * Ka)
            led.hold('root_radius_active', 8 * Ka)
            h = horizontal_edges_numpy(xl, yl, rr, dom.nx, dom.ny, float(dom.dx_m), max_edges=me, max_bytes=mb,
                                       resident_bytes=led.live_bytes)
        else:
            # the compiled helper counts its own copies of x / y / radius: pass the budget left after the other live work
            h = horizontal_edges_compiled(xl, yl, rr, dom.nx, dom.ny, float(dom.dx_m), max_edges=me,
                                          max_bytes=mb - led.live_bytes)
        del xl, yl, rr
        led.release('x_local', 'y_local', 'root_radius_active')
    else:
        h = {'offsets': np.zeros(1, dtype=np.int64), 'cells': np.zeros(0, dtype=np.int64),
             'weights': np.zeros(0, dtype=np.float64), 'counts': np.zeros(0, dtype=np.int64), 'n_edges': 0,
             'estimate': {'stage': 'horizontal_empty', 'n_edges': 0, 'total_bytes': 0, 'scope': 'no plant with roots'},
             'builder': builder}
    h_cells, h_weights, h_counts = h.pop('cells'), h.pop('weights'), h.pop('counts')
    h.pop('offsets')
    h_report = {'builder': h['builder'], 'n_edges': h['n_edges'], 'estimate': dict(h['estimate'])}
    if 'timings' in h:
        h_report['timings'] = dict(h['timings'])
    n_h_edges = int(h['n_edges'])
    del h
    led.hold('horizontal_cells', h_cells.nbytes)
    led.hold('horizontal_weights', h_weights.nbytes)
    led.hold('horizontal_counts_active', h_counts.nbytes)
    led.release('root_radius')
    del radius
    # per-plant counts / offsets over ALL selected rows (inactive rows carry zero edges)
    led.admit('roots.horizontal_map', {'counts_selected': 8 * K, 'offsets_selected': 8 * (K + 1)})
    counts_h = np.zeros(K, dtype=np.int64)
    counts_h[active] = h_counts
    offsets_h = np.zeros(K + 1, dtype=np.int64)
    np.cumsum(counts_h, out=offsets_h[1:])
    del h_counts, active, sel
    led.release('horizontal_counts_active', 'active_index', 'selected_rows')
    led.hold('counts_selected', 8 * K)
    led.hold('offsets_selected', 8 * (K + 1))
    # 5. admit the EXPANDED layered edge count (and its scratch) before any expanded array exists
    led.admit('roots.expand_count', {'layer_positive_mask': K * L, 'layer_counts': 24 * K})
    E, nseg = expansion_plan(counts_h, table)
    rep = led.admit('roots.expand_layers', {
        'offsets_out': 8 * (K + 1), 'resources_out': 8 * E, 'weights_out': 8 * E,
        'edge_index': 8 * E, 'repeat_gather_temporary': 8 * E,
        'segment_arrays': SEGMENT_BYTES * nseg + 8, 'plant_counts_and_mask': 24 * K + K * L},
        n_edges=E, scope='expanded owned arrays + expansion scratch over the live horizontal arrays')
    offsets, resources, weights = expand_layer_edges(offsets_h, h_cells, h_weights, table, N)
    del h_cells, h_weights, counts_h, offsets_h, table
    led.release('horizontal_cells', 'horizontal_weights', 'counts_selected', 'offsets_selected', 'layer_table')
    led.hold('offsets', offsets.nbytes)
    led.hold('resources', resources.nbytes)
    led.hold('weights', weights.nbytes)
    prov = {'builder': builder, 'horizontal_law': HORIZONTAL_LAW, 'horizontal_law_label': HORIZONTAL_LAW_LABEL,
            'vertical_law': VERTICAL_LAW_LABEL, 'fixed_weights': FIXED_WEIGHTS_NOTE,
            'statuses': [_state.STATUS_NAMES[c] for c in codes], 'n_selected': K, 'n_with_roots': Ka,
            'n_inactive_rows': K - Ka, 'n_overrides': n_ov_applied, 'n_horizontal_edges': n_h_edges,
            'domain': dom.to_dict(),
            'coordinate_frame': 'domain-local (landscape position minus origin), row-major cell index row * nx + col, '
                                'resource = layer * nx * ny + cell',
            'catalog_hash': state.catalog_hash, 'time_days': state.time_days, 'root_layers': L,
            'memory_scope': MEMORY_SCOPE,
            'admission': {'stages': led.reports, 'horizontal_builder': h_report, 'expand_layers': rep},
            'leaf_version': _traits.LEAF_VERSION}
    if provenance:
        prov['caller'] = dict(provenance)
    # internal trusted ownership transfer: the 4 fresh arrays are live (counted); the map's own counts array is
    # added to the resident bytes because it is created inside the validation
    return _uptake.adopt_owned_arrays(ids, offsets, resources, weights, n_layers=L, n_cells=N, backend=backend,
                                      max_edges=me, max_bytes=mb, provenance=prov,
                                      resident_bytes=led.live_bytes + 8 * K)


__all__ = ['VERSION', 'BUILDERS', 'HORIZONTAL_LAW', 'HORIZONTAL_LAW_LABEL', 'VERTICAL_LAW_LABEL', 'FIXED_WEIGHTS_NOTE',
           'MEMORY_SCOPE', 'RootsError', 'numpy_bbox_scratch_bytes', 'horizontal_edges_numpy',
           'horizontal_edges_compiled', 'expansion_plan', 'expanded_edge_count', 'expand_layer_edges',
           'build_root_map']
