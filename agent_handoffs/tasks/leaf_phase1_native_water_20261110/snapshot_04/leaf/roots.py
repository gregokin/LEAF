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
non-separable weights) and skip this builder entirely.

Grid support. The shared compiled helper supports SQUARE cell spacing (dx == dy) in a [0, nx*dx] x [0, ny*dy]
domain-local frame; non-square domains are REFUSED explicitly before any allocation (dx is never silently used
for dy). Positions are converted from the landscape frame by subtracting the domain origin.

Memory admission (explicit max_edges / max_bytes, refused BEFORE allocation, population never truncated):
  1. preflight (P-sized geometry/inputs + R-sized map validation scratch) from P, N, L alone;
  2. horizontal build: the compiled helper's own pre-count / post-count admission (its estimate also counts a
     (layers+1)*N float64 grid term it needs for its benchmark; reported verbatim under 'horizontal_builder');
  3. expanded layer edges E = sum_p cells_p * positive_layers_p are admitted BEFORE the expanded arrays exist;
  4. leaf.uptake.RootMap.from_arrays admits its owned arrays + validation scratch (+ device staging for cuda).
Importing this module imports NumPy only; numba (compiled builder) and cupy (cuda backend) load lazily.
'''
from __future__ import annotations

import math

import numpy as np

from leaf import canopy as _canopy
from leaf import state as _state
from leaf import traits as _traits
from leaf import uptake as _uptake

VERSION = '0.1.0'
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
_EXPANSION_SCRATCH_PER_EDGE = 6 * 8   # local/hedge/repeat temporaries (int64/float64) during layer expansion


class RootsError(ValueError):
    '''Any root-map construction error raised before the map exists (inputs untouched).'''


# ------------------------------------------------------------------ NumPy reference of the compiled circle-overlap builder
def _cell_range(c, r, dx, n):
    lo = math.floor((c - r) / dx)
    hi = math.floor((c + r) / dx)
    if lo < 0.0:
        lo = 0.0
    if hi > n - 1.0:
        hi = n - 1.0
    return int(lo), int(hi)


def _overlap_cells(px, py, r, nx, ny, dx):
    '''Row-major cell indices (j * nx + i) positively overlapped by the circle; identical predicate and
    arithmetic to benchmarks.spatial_mapping._overlaps / _fill_edges (j outer, i inner).'''
    i0, i1 = _cell_range(px, r, dx, nx)
    j0, j1 = _cell_range(py, r, dx, ny)
    if i1 < i0 or j1 < j0:
        return np.zeros(0, dtype=np.int64)
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
    return (jj[:, None] * nx + ii[None, :])[mask]


def horizontal_edges_numpy(x, y, root_radius, nx, ny, dx, *, max_edges, max_bytes):
    '''NumPy reference horizontal CSR (uniform weights 1/count): offsets [P+1] int64, cells [E] int64,
    weights [E] float64, counts [P]. Two passes (count, admit, fill); O(P) Python iterations over plants with
    vectorised cell tests (reference / fallback builder, not the production compiled path).'''
    P = int(x.shape[0])
    counts = np.zeros(P, dtype=np.int64)
    for p in range(P):
        counts[p] = _overlap_cells(float(x[p]), float(y[p]), float(root_radius[p]), nx, ny, dx).shape[0]
    E = int(counts.sum())
    est = {'stage': 'horizontal_numpy', 'n_edges': E, 'total_bytes': 8 * (P + 1) + 8 * P + 8 * E + 8 * E,
           'scope': 'offsets + counts + int64 cells + float64 weights of the horizontal CSR'}
    _uptake.admit('roots.horizontal_numpy', est, max_edges=max_edges, max_bytes=max_bytes)
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
    layers=1, numba). Requires at least one plant with a positive radius (callers handle the empty case).'''
    try:
        from benchmarks import spatial_mapping as sm
    except Exception as exc:  # numba (or the benchmarks package) is unavailable
        raise _uptake.BackendUnavailableError(f'compiled root builder unavailable (benchmarks.spatial_mapping import '
                                              f'failed: {exc!r}); use builder="numpy" explicitly') from exc
    grid = sm.Grid(int(nx), int(ny), float(dx))
    n = int(x.shape[0])
    inputs = {'x': np.ascontiguousarray(x, dtype=np.float64), 'y': np.ascontiguousarray(y, dtype=np.float64),
              'canopy_radius': np.ascontiguousarray(root_radius, dtype=np.float64),
              'root_radius': np.ascontiguousarray(root_radius, dtype=np.float64),
              'ids': np.arange(n, dtype=np.int64)}
    try:
        g = sm.build_root_graph_cpu(inputs, grid, 1, int(max_edges), int(max_bytes))
    except sm.AdmissionRejected as exc:
        raise _uptake.AdmissionError('roots.horizontal_compiled', str(exc), dict(exc.estimate, stage=exc.stage,
                                     scope='benchmarks.spatial_mapping.estimate_bytes: inputs + CSR graph + its '
                                           '(layers+1)*N float64 grid term; reported verbatim')) from exc
    except ValueError as exc:
        raise RootsError(f'compiled root builder refused its inputs: {exc}') from exc
    return {'offsets': np.asarray(g.offsets, dtype=np.int64), 'cells': np.asarray(g.cells, dtype=np.int64),
            'weights': np.asarray(g.weights, dtype=np.float64), 'counts': np.asarray(g.counts, dtype=np.int64),
            'n_edges': int(g.cells.shape[0]),
            'estimate': dict(g.estimate, stage='horizontal_compiled',
                             scope='benchmarks.spatial_mapping.estimate_bytes (inputs + CSR + its (layers+1)*N grid term)'),
            'builder': 'compiled', 'timings': dict(g.timings)}


# ------------------------------------------------------------------ layer expansion (vectorised, admitted first)
def expanded_edge_count(h_counts, layer_table):
    '''E = sum_p cells_p * (number of layers with fraction > 0) from P-sized inputs only (no E-sized array).'''
    m = np.count_nonzero(layer_table > 0.0, axis=1)
    return int(np.sum(h_counts.astype(np.int64) * m.astype(np.int64)))


def expand_layer_edges(h_offsets, h_cells, h_weights, layer_table, n_cells):
    '''(offsets, resources, weights) of the layered map: per plant, layer-major over positive-fraction layers,
    horizontal cell order within a layer; resource = layer * n_cells + cell; weight = fraction * horizontal.'''
    P, L = layer_table.shape
    k = np.diff(h_offsets).astype(np.int64)
    pos = layer_table > 0.0
    counts = k * np.count_nonzero(pos, axis=1).astype(np.int64)
    offsets = np.zeros(P + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    E = int(offsets[-1])
    seg_plant, seg_layer = np.nonzero(pos)                  # row-major: plant, then layer
    seg_len = k[seg_plant]
    seg_off = np.zeros(seg_len.shape[0] + 1, dtype=np.int64)
    np.cumsum(seg_len, out=seg_off[1:])
    local = np.arange(E, dtype=np.int64) - np.repeat(seg_off[:-1], seg_len)
    hedge = np.repeat(h_offsets[:-1][seg_plant], seg_len) + local
    resources = np.repeat(seg_layer.astype(np.int64), seg_len) * np.int64(n_cells) + h_cells[hedge]
    weights = np.repeat(layer_table[seg_plant, seg_layer], seg_len) * h_weights[hedge]
    return offsets, resources, weights


# ------------------------------------------------------------------ overrides
def _validate_overrides(overrides, state, selected_rows, L):
    '''{plant_id: [L fractions]} -> (row indices [K], table [K, L]); every ID must exist and be selected.'''
    if overrides is None:
        return np.zeros(0, dtype=np.int64), np.zeros((0, L), dtype=np.float64)
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
    try:
        idx = state.indices_of(np.array(ids, dtype=np.int64))
    except _state.StateError as exc:
        raise RootsError(f'layer_fraction_overrides: {exc}') from exc
    not_selected = ~np.isin(idx, selected_rows)
    if not_selected.any():
        raise RootsError(f'layer_fraction_overrides target plants outside the selected statuses: ids '
                         f'{[ids[i] for i in np.flatnonzero(not_selected)[:10]]}')
    return idx, np.array(rows, dtype=np.float64).reshape(len(ids), L)


# ------------------------------------------------------------------ builder
def build_root_map(state, catalog, *, max_edges, max_bytes, backend='numpy', builder='compiled',
                   statuses=('live',), layer_fraction_overrides=None, provenance=None):
    '''RootMap for the selected plants of `state` (see the module docstring). Pure with respect to the state
    and catalog. Returns a leaf.uptake.RootMap whose rows are the selected plants in state row order and whose
    `provenance` records builder, laws, selection, admission reports and the catalog hash / clock.'''
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
    L, N = catalog.root_layers, dom.ncells
    P_all = state.n
    # 1. preflight from scalars only: P-sized geometry + inputs, S*L table, R-sized map validation scratch
    pre = {'stage': 'roots.preflight', 'n_edges': 0,
           'total_bytes': 8 * 12 * P_all + 8 * catalog.n_species * L + 8 * P_all * L + 8 * L * N,
           'scope': 'derive_geometry O(P) arrays + builder inputs + per-plant layer table [P, L] + R-sized map '
                    'validation scratch; computed from P, N, L before any of it is allocated'}
    _uptake.admit('roots.preflight', pre, max_edges=me, max_bytes=mb)
    geo = _canopy.derive_geometry(state, catalog)      # the one shared allometry; validates state/catalog
    sel = np.flatnonzero(np.isin(geo['status'], codes))
    P = int(sel.shape[0])
    ids = geo['plant_id'][sel]
    radius = geo['root_radius_m'][sel]
    table = catalog.root_layer_table()[geo['species_index'][sel].astype(np.int64)]
    ov_idx, ov_rows = _validate_overrides(layer_fraction_overrides, state, sel, L)
    if ov_idx.shape[0]:
        pos_in_sel = np.searchsorted(sel, ov_idx)          # sel is sorted (flatnonzero)
        table[pos_in_sel] = ov_rows
    # 2. horizontal CSR for plants with a positive root radius (domain-local, clipped to the closed domain)
    active = np.flatnonzero(radius > 0.0)
    W, H = float(dom.nx) * float(dom.dx_m), float(dom.ny) * float(dom.dx_m)
    xl = np.minimum(np.maximum(geo['x_m'][sel][active] - dom.origin_x_m, 0.0), W)
    yl = np.minimum(np.maximum(geo['y_m'][sel][active] - dom.origin_y_m, 0.0), H)
    if active.shape[0]:
        fn = horizontal_edges_compiled if builder == 'compiled' else horizontal_edges_numpy
        h = fn(xl, yl, radius[active], dom.nx, dom.ny, float(dom.dx_m), max_edges=me, max_bytes=mb)
    else:
        h = {'offsets': np.zeros(1, dtype=np.int64), 'cells': np.zeros(0, dtype=np.int64),
             'weights': np.zeros(0, dtype=np.float64), 'counts': np.zeros(0, dtype=np.int64), 'n_edges': 0,
             'estimate': {'stage': 'horizontal_empty', 'n_edges': 0, 'total_bytes': 0, 'scope': 'no plant with roots'},
             'builder': builder}
    counts_h = np.zeros(P, dtype=np.int64)
    counts_h[active] = h['counts']
    offsets_h = np.zeros(P + 1, dtype=np.int64)
    np.cumsum(counts_h, out=offsets_h[1:])
    # 3. admit the EXPANDED layered edge count before the expanded arrays exist
    E = expanded_edge_count(counts_h, table)
    map_est = _uptake.estimate_map_bytes(P, E, L * N, backend, host_source=True)
    horizontal_resident = 8 * (P + 1) + 8 * P + 16 * h['n_edges']
    exp_est = {'stage': 'roots.expand_layers', 'n_edges': E,
               'total_bytes': horizontal_resident + _EXPANSION_SCRATCH_PER_EDGE * E + 16 * E + int(map_est['total_bytes']),
               'scope': 'resident horizontal CSR + layer-expansion temporaries (6 x 8 bytes per expanded edge) + the '
                        'expanded owned arrays + the RootMap admission estimate (owned + validation scratch + staging)'}
    exp_adm = _uptake.admit('roots.expand_layers', exp_est, max_edges=me, max_bytes=mb)
    offsets, resources, weights = expand_layer_edges(offsets_h, h['cells'], h['weights'], table, N)
    h_report = {'builder': h['builder'], 'n_edges': h['n_edges'], 'estimate': dict(h['estimate'])}
    if 'timings' in h:
        h_report['timings'] = dict(h['timings'])
    del h
    prov = {'builder': builder, 'horizontal_law': HORIZONTAL_LAW, 'horizontal_law_label': HORIZONTAL_LAW_LABEL,
            'vertical_law': VERTICAL_LAW_LABEL, 'fixed_weights': FIXED_WEIGHTS_NOTE,
            'statuses': [_state.STATUS_NAMES[c] for c in codes], 'n_selected': P, 'n_with_roots': int(active.shape[0]),
            'n_inactive_rows': P - int(active.shape[0]), 'n_overrides': int(ov_idx.shape[0]),
            'n_horizontal_edges': int(counts_h.sum()), 'domain': dom.to_dict(),
            'coordinate_frame': 'domain-local (landscape position minus origin), row-major cell index row * nx + col, '
                                'resource = layer * nx * ny + cell',
            'catalog_hash': state.catalog_hash, 'time_days': state.time_days, 'root_layers': L,
            'admission': {'preflight': pre, 'horizontal_builder': h_report, 'expand_layers': exp_adm},
            'leaf_version': _traits.LEAF_VERSION}
    if provenance:
        prov['caller'] = dict(provenance)
    return _uptake.RootMap.from_arrays(ids, offsets, resources, weights, n_layers=L, n_cells=N, backend=backend,
                                       max_edges=me, max_bytes=mb, provenance=prov, _take_ownership=True)


__all__ = ['VERSION', 'BUILDERS', 'HORIZONTAL_LAW', 'HORIZONTAL_LAW_LABEL', 'VERTICAL_LAW_LABEL', 'FIXED_WEIGHTS_NOTE',
           'RootsError', 'horizontal_edges_numpy', 'horizontal_edges_compiled', 'expanded_edge_count',
           'expand_layer_edges', 'build_root_map']
