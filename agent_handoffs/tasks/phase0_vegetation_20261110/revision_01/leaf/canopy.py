'''leaf.canopy: derived plant geometry, MAPLE-compatible canopy view and bounded union-cover measurement.

Geometry is DERIVED from the canonical PlantState + species catalog on request; it is never an
independently editable plant authority. Canopies are circles (equal semi-axes) of allometric diameter
centred on (x_m, y_m); roots are circles of radius lateral_radius_multiplier * canopy radius. Both may span
many raster cells.

Heights: `height_m` is the INTACT allometric height above the plant's establishment ground. When a current
surface is supplied, `surface_delta_m = current - establishment` (positive = deposition/burial, negative =
erosion/exposure), `burial_depth_m = max(delta, 0)`, `exposure_depth_m = max(-delta, 0)` and
`exposed_height_m = max(height - burial, 0)` are geometric diagnostics only: no root dislodgement, injury
or mortality is inferred here.

MAPLE's VegetationObjectSet (as inspected by Codex) has center_x_m, center_y_m, semi_major_m, semi_minor_m,
orientation_deg, height_m, porosity, drag_coefficient in domain-local metres (+x east, +y north, cell centres
((col+.5)*dx, (row+.5)*dy)) and cannot represent a canopy base elevated above the current surface (eroded
anchor) or a fully buried obstacle. maple_vegetation_view() therefore handles those cases explicitly via
`unsupported_policy` instead of flattening them. MAPLE's own rasterizer owns raster semantics; LEAF hands
over plain arrays plus stable-ID sidecars and never imports MAPLE.

Cover: measure_cover() rasterises each group's circles onto a (ny*s, nx*s) boolean midpoint raster and
reports the union fraction of the domain. It is a deterministic SAMPLED approximation (resolution dx/s),
never an exact analytic area; the error scale is reported. Memory is bounded and checked before allocation
(one or two rasters, never P x N).
'''
from __future__ import annotations

import math

import numpy as np

from leaf import state as _state
from leaf import traits as _traits

COVER_METHOD = 'midpoint_raster_union'
DEFAULT_MAX_COVER_CELLS = 64_000_000
DEFAULT_MAX_COVER_BYTES = 128_000_000
MAPLE_FIELDS = ('center_x_m', 'center_y_m', 'semi_major_m', 'semi_minor_m', 'orientation_deg', 'height_m',
                'porosity', 'drag_coefficient')
HEIGHT_MODES = ('intact', 'exposed')
UNSUPPORTED_POLICIES = ('error', 'exclude', 'flag')
GEOMETRY_UNITS = {'x_m': 'm', 'y_m': 'm', 'diameter_m': 'm', 'radius_m': 'm', 'height_m': 'm (intact allometric)',
                  'root_radius_m': 'm', 'establishment_ground_m': 'm elevation', 'porosity': 'fraction [0,1]',
                  'current_ground_m': 'm elevation', 'surface_delta_m': 'm (+ deposition, - erosion)',
                  'burial_depth_m': 'm >= 0', 'exposure_depth_m': 'm >= 0', 'exposed_height_m': 'm >= 0'}


class ResourceBoundError(ValueError):
    '''A declared cell/byte bound would be exceeded; raised before any large allocation.'''


def _check_catalog(state, catalog):
    if not isinstance(catalog, _traits.SpeciesCatalog):
        raise _state.StateError('catalog must be a leaf.traits.SpeciesCatalog')
    if catalog.hash != state.catalog_hash:
        raise _state.StateError(f'catalog hash {catalog.hash} differs from the state catalog hash {state.catalog_hash}')
    if tuple(catalog.species_ids) != tuple(state.species_names):
        raise _state.StateError('catalog species table differs from the state species table')


# ------------------------------------------------------------------ geometry
def derive_geometry(state, catalog, *, current_surface=None):
    '''O(P) derived geometry arrays for ALL plants (every status). Pure: the state is not modified.'''
    _check_catalog(state, catalog)
    idx = state.species_index.astype(np.int64)
    B = state.agb_dry_kg
    al = catalog.allometry_arrays()
    diameter = al['a_d'][idx] * np.power(B, al['b_d'][idx])
    height = al['a_h'][idx] * np.power(B, al['b_h'][idx])
    radius = 0.5 * diameter
    out = {'n': state.n, 'plant_id': state.ids.copy(), 'species_index': state.species_index.copy(),
           'species': state.species_of(), 'status': state.status.copy(), 'status_name': state.status_names(),
           'x_m': state.x_m.copy(), 'y_m': state.y_m.copy(), 'diameter_m': diameter, 'radius_m': radius,
           'height_m': height, 'root_radius_m': radius * catalog.root_multiplier_array()[idx],
           'establishment_ground_m': state.establishment_ground_m.copy(),
           'establishment_unresolved': state.establishment_unresolved.copy(),
           'porosity': catalog.porosity_array()[idx], 'agb_dry_kg': B.copy(),
           'catalog_hash': state.catalog_hash, 'time_days': state.time_days, 'domain': state.domain.to_dict(),
           'height_definition': 'intact allometric height above establishment ground (not exposed height)',
           'canopy_shape': 'circle (semi_major == semi_minor == radius_m, orientation 0)',
           'porosity_note': 'constant per-species catalog value; unused by LEAF ecology, USED by native MAPLE wind',
           'units': dict(GEOMETRY_UNITS)}
    if current_surface is not None:
        state.require_resolved('burial/exposure diagnostics')
        cur = state.domain.sample(current_surface, state.x_m, state.y_m, 'current surface')
        delta = cur - state.establishment_ground_m
        burial = np.maximum(delta, 0.0)
        out.update({'current_ground_m': cur, 'surface_delta_m': delta, 'burial_depth_m': burial,
                    'exposure_depth_m': np.maximum(-delta, 0.0),
                    'exposed_height_m': np.maximum(height - burial, 0.0),
                    'fully_buried': burial >= height,
                    'diagnostics_note': 'geometric burial/exposure only; no dislodgement, injury or mortality inferred'})
    return out


def maple_vegetation_view(state, catalog, *, current_surface=None, statuses=('live', 'dead_standing'),
                          height_mode='intact', unsupported_policy='error', drag_coefficient=1.0):
    '''Plain-array view compatible with MAPLE's VegetationObjectSet fields plus stable-ID sidecars.

    Coordinates are DOMAIN-LOCAL metres (origin subtracted) as MAPLE expects. Only plants whose status is in
    `statuses` and whose diameter is > 0 are included. height_mode 'intact' publishes allometric heights;
    'exposed' (requires current_surface) publishes height - burial. Unsupported cases are eroded anchors
    (base above the current surface; only detectable with current_surface) and fully buried plants in
    'exposed' mode: policy 'error' raises, 'exclude' drops them (counted), 'flag' keeps them with the
    documented value (intact or exposed height, base at the current surface) and sets sidecar flags.
    Publishing requires resolved establishment anchors.'''
    state.require_resolved('publishing a MAPLE vegetation view')
    if height_mode not in HEIGHT_MODES:
        raise ValueError(f'height_mode must be one of {HEIGHT_MODES}')
    if unsupported_policy not in UNSUPPORTED_POLICIES:
        raise ValueError(f'unsupported_policy must be one of {UNSUPPORTED_POLICIES}')
    if height_mode == 'exposed' and current_surface is None:
        raise ValueError('height_mode="exposed" requires current_surface')
    drag = _traits.nonnegative(drag_coefficient, 'drag_coefficient')
    codes = set(int(c) for c in _state.status_codes(list(statuses)))
    geo = derive_geometry(state, catalog, current_surface=current_surface)
    sel = np.isin(geo['status'], list(codes)) & (geo['diameter_m'] > 0.0)
    excluded_status = int((~np.isin(geo['status'], list(codes))).sum())
    excluded_zero = int((np.isin(geo['status'], list(codes)) & ~(geo['diameter_m'] > 0.0)).sum())
    elevated = np.zeros(state.n, dtype=bool)
    buried = np.zeros(state.n, dtype=bool)
    if current_surface is not None:
        elevated = geo['exposure_depth_m'] > 0.0
        buried = geo['fully_buried']
    unsupported = sel & (elevated | (buried if height_mode == 'exposed' else np.zeros(state.n, dtype=bool)))
    n_unsupported = int(unsupported.sum())
    if n_unsupported and unsupported_policy == 'error':
        ids = geo['plant_id'][unsupported][:10].tolist()
        raise _state.StateError(f'{n_unsupported} plant(s) cannot be represented by MAPLE in height_mode={height_mode!r} '
                                f'(eroded anchor with elevated canopy base and/or fully buried), e.g. ids {ids}; '
                                "choose unsupported_policy='exclude' or 'flag' explicitly")
    if unsupported_policy == 'exclude':
        sel &= ~unsupported
    height = geo['height_m'] if height_mode == 'intact' else geo['exposed_height_m']
    dom = state.domain
    k = int(sel.sum())
    fields = {'center_x_m': geo['x_m'][sel] - dom.origin_x_m, 'center_y_m': geo['y_m'][sel] - dom.origin_y_m,
              'semi_major_m': geo['radius_m'][sel].copy(), 'semi_minor_m': geo['radius_m'][sel].copy(),
              'orientation_deg': np.zeros(k, dtype=np.float64), 'height_m': height[sel].copy(),
              'porosity': geo['porosity'][sel].copy(), 'drag_coefficient': np.full(k, drag, dtype=np.float64)}
    sidecar = {'plant_id': geo['plant_id'][sel], 'species': geo['species'][sel], 'status': geo['status_name'][sel],
               'establishment_ground_m': geo['establishment_ground_m'][sel], 'root_radius_m': geo['root_radius_m'][sel],
               'intact_height_m': geo['height_m'][sel], 'elevated_base': elevated[sel], 'fully_buried': buried[sel]}
    if current_surface is not None:
        sidecar.update({'burial_depth_m': geo['burial_depth_m'][sel], 'exposure_depth_m': geo['exposure_depth_m'][sel],
                        'current_ground_m': geo['current_ground_m'][sel]})
    meta = {'schema': 'leaf.maple_vegetation_view', 'schema_version': 1, 'n_objects': k, 'fields': list(MAPLE_FIELDS),
            'coordinates': 'domain-local metres: center = plant position - domain origin; +x east, +y north; '
                           f'MAPLE cell centres ((col+0.5)*dx, (row+0.5)*dy) with dx={dom.dx_m}, dy={dom.dy_m}',
            'domain': dom.to_dict(), 'catalog_hash': state.catalog_hash, 'time_days': state.time_days,
            'height_mode': height_mode, 'height_definition': ('intact allometric height' if height_mode == 'intact'
                                                              else 'intact height minus burial depth (base at current surface)'),
            'canopy_base': 'at the current surface (MAPLE cannot represent an elevated base)',
            'statuses_included': sorted(_state.STATUS_NAMES[c] for c in codes), 'excluded_by_status': excluded_status,
            'excluded_zero_diameter': excluded_zero, 'unsupported_policy': unsupported_policy,
            'n_unsupported': n_unsupported, 'n_unsupported_excluded': n_unsupported if unsupported_policy == 'exclude' else 0,
            'porosity_note': 'constant per-species catalog value; native MAPLE wind uses it, LEAF ecology does not',
            'drag_coefficient_note': 'uniform pass-through metadata value, not a LEAF trait',
            'rasterization': 'performed by MAPLE\'s existing rasterizer; LEAF keeps plant_id sidecars beside it'}
    return {'fields': fields, 'sidecar': sidecar, 'meta': meta}


# ------------------------------------------------------------------ cover measurement
def raster_bound(domain, subsamples, total=True):
    '''(cells, bytes) of the boolean midpoint raster(s) for a bounds check before allocation.'''
    s = _traits.strict_int(subsamples, 'subsamples', 1)
    cells = domain.nx * s * domain.ny * s
    return cells, cells * (2 if total else 1)


def measure_cover(domain, x_m, y_m, radius_m, group_index=None, n_groups=1, *, subsamples=4,
                  max_cover_cells=DEFAULT_MAX_COVER_CELLS, max_cover_bytes=DEFAULT_MAX_COVER_BYTES, total=True):
    '''Per-group union ground cover of circles clipped to the domain (fraction of the domain area).

    Raster midpoints are origin + (i + 0.5) * dx / s; a midpoint is covered when strictly inside a circle.
    Same-group overlaps count once; different groups are measured independently (cross-group overlap adds
    to each group's cover). `total_union_fraction` is the union over all groups. Deterministic, O(P) work
    on bounded bounding boxes; memory = one or two boolean rasters checked against the declared bounds.'''
    if not isinstance(domain, _state.Domain):
        raise _state.StateError('domain must be a leaf.state.Domain')
    s = _traits.strict_int(subsamples, 'subsamples', 1)
    G = _traits.strict_int(n_groups, 'n_groups', 1)
    x = _state._coerce_float(x_m, 'x_m')
    n = x.shape[0]
    y = _state._coerce_float(y_m, 'y_m', n)
    r = _state._coerce_float(radius_m, 'radius_m', n)
    if n and (not np.all(np.isfinite(r)) or float(r.min()) < 0.0):
        raise _state.StateError('radius_m must be finite and >= 0')
    if n:
        domain.check_positions(x, y, 'circle centres')
    g = (np.zeros(n, dtype=np.int64) if group_index is None
         else _state._coerce_int(group_index, 'group_index', n, np.int64, 0, G))
    cells, nbytes = raster_bound(domain, s, total)
    mc = _traits.strict_int(max_cover_cells, 'max_cover_cells', 1)
    mb = _traits.strict_int(max_cover_bytes, 'max_cover_bytes', 1)
    if cells > mc or nbytes > mb:
        raise ResourceBoundError(f'cover raster {domain.ny * s}x{domain.nx * s} = {cells} cells / {nbytes} bytes exceeds '
                                 f'max_cover_cells={mc} / max_cover_bytes={mb}; lower subsamples or raise the bound explicitly')
    NX, NY = domain.nx * s, domain.ny * s
    hx, hy = domain.dx_m / s, domain.dy_m / s
    x0, y0 = domain.origin_x_m, domain.origin_y_m
    raster = np.zeros((NY, NX), dtype=bool)
    union_all = np.zeros((NY, NX), dtype=bool) if total else None
    A = domain.area_m2
    groups = []
    for gi in range(G):
        raster[...] = False
        sel = np.flatnonzero(g == gi)
        nominal = 0.0
        for p in sel:
            rp = float(r[p])
            if rp <= 0.0:
                continue
            nominal += math.pi * rp * rp
            px, py = float(x[p]), float(y[p])
            i0 = max(int(math.floor((px - rp - x0) / hx)), 0)
            i1 = min(int(math.floor((px + rp - x0) / hx)), NX - 1)
            j0 = max(int(math.floor((py - rp - y0) / hy)), 0)
            j1 = min(int(math.floor((py + rp - y0) / hy)), NY - 1)
            if i1 < i0 or j1 < j0:
                continue
            xs = x0 + (np.arange(i0, i1 + 1, dtype=np.float64) + 0.5) * hx - px
            ys = y0 + (np.arange(j0, j1 + 1, dtype=np.float64) + 0.5) * hy - py
            mask = (xs[None, :] * xs[None, :] + ys[:, None] * ys[:, None]) < rp * rp
            raster[j0:j1 + 1, i0:i1 + 1] |= mask
        covered = int(np.count_nonzero(raster))
        if union_all is not None:
            union_all |= raster
        rr = r[sel]
        rpos = rr[rr > 0.0]
        groups.append({'group': gi, 'count': int(sel.shape[0]), 'covered_cells': covered,
                       'union_fraction': covered / cells, 'union_area_m2': covered / cells * A,
                       'nominal_area_m2': nominal, 'nominal_fraction': nominal / A,
                       'mean_radius_m': float(rpos.mean()) if rpos.size else None,
                       'relative_error_scale': (float(max(hx, hy) / rpos.mean()) if rpos.size else None)})
    return {'method': COVER_METHOD, 'subsamples': s, 'raster_shape': [NY, NX], 'resolution_m': [hx, hy],
            'raster_cells': cells, 'raster_bytes': nbytes, 'domain_area_m2': A, 'n_circles': n, 'groups': groups,
            'total_union_fraction': (int(np.count_nonzero(union_all)) / cells) if union_all is not None else None,
            'sum_of_group_union_fractions': float(sum(gr['union_fraction'] for gr in groups)),
            'definition': 'per-group projected union ground cover clipped to the finite domain (same-group overlap '
                          'counted once; cross-group overlap counted in each group); nominal = sum of unclipped '
                          'crown areas / domain area',
            'error_basis': 'midpoint sampling at resolution dx/s: each circle boundary is resolved to about one '
                           'raster cell, relative error per circle ~ resolution / radius; sampled, not analytic'}


__all__ = ['COVER_METHOD', 'DEFAULT_MAX_COVER_CELLS', 'DEFAULT_MAX_COVER_BYTES', 'MAPLE_FIELDS', 'HEIGHT_MODES',
           'UNSUPPORTED_POLICIES', 'GEOMETRY_UNITS', 'ResourceBoundError', 'derive_geometry',
           'maple_vegetation_view', 'raster_bound', 'measure_cover']
