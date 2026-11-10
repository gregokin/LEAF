'''leaf.canopy: derived plant geometry, safe MAPLE-compatible canopy publication and bounded union cover.

Geometry is DERIVED from the canonical PlantState + species catalog on request; it is never an
independently editable plant authority. Canopies are circles (equal semi-axes) of allometric diameter
centred on (x_m, y_m); roots are circles of radius lateral_radius_multiplier * canopy radius. Both may span
many raster cells.

Heights. `derive_geometry` returns the INTACT allometric height above the plant's establishment ground and,
when a current surface is supplied, the geometric diagnostics `surface_delta_m = current - establishment`
(+ deposition/burial, - erosion/exposure), `burial_depth_m`, `exposure_depth_m`,
`exposed_height_m = max(height - burial, 0)` and `fully_buried`. No root dislodgement, injury or mortality
is inferred.

MAPLE publication (`maple_vegetation_view`). MAPLE's VegetationObjectSet (as inspected by Codex) has
center_x_m, center_y_m, semi_major_m, semi_minor_m, orientation_deg, height_m, porosity, drag_coefficient in
domain-local metres (+x east, +y north, cell centres ((col+.5)*dx, (row+.5)*dy)); every canopy base sits on
the CURRENT surface. Safe publication therefore REQUIRES the current surface and uses the establishment
anchor: the default height is the EXPOSED height (intact minus burial). Fully buried plants
(exposed height <= 0) and eroded anchors (canopy base above the current surface) cannot be represented and
are unsupported: the default policy refuses, 'exclude' drops them (counted), 'flag' keeps them with sidecar
flags and marks the whole view `native_maple_valid: False`. `height_mode='intact_diagnostic'` publishes
intact heights ignoring burial; it is a labelled DIAGNOSTIC APPROXIMATION, never native-MAPLE-valid.
MAPLE's own rasterizer owns raster semantics; LEAF hands over plain arrays plus stable-ID sidecars and never
imports MAPLE.

Cover (`measure_cover`). Each group's circles are rasterised onto a (ny*s, nx*s) boolean midpoint raster
and the union fraction of the FINITE, CLIPPED, NON-PERIODIC domain is reported. It is a deterministic
SAMPLED approximation (resolution dx/s), never an exact analytic area. Memory is admitted before any
allocation with two precisely scoped terms: raster bytes (one or two boolean rasters, bound
`max_cover_bytes`) and working bytes (row-tiled footprint temporaries + 1-D coordinate scratch, bound
`max_working_bytes`); P-sized index arrays are reported separately. Never P x N or P x P.
'''
from __future__ import annotations

import math

import numpy as np

from leaf import state as _state
from leaf import traits as _traits

COVER_METHOD = 'midpoint_raster_union'
DEFAULT_MAX_COVER_CELLS = 64_000_000
DEFAULT_MAX_COVER_BYTES = 128_000_000
DEFAULT_MAX_WORKING_BYTES = 1 << 20
_CHUNK_BYTES_PER_ELEMENT = 9   # float64 squared-distance temporary + bool comparison per footprint element
MAPLE_FIELDS = ('center_x_m', 'center_y_m', 'semi_major_m', 'semi_minor_m', 'orientation_deg', 'height_m',
                'porosity', 'drag_coefficient')
HEIGHT_MODES = ('exposed', 'intact_diagnostic')
UNSUPPORTED_POLICIES = ('error', 'exclude', 'flag')
BOUNDARY_NOTE = ('finite closed non-periodic domain: canopies are clipped at the domain edge for cover and never '
                 'wrap; exclusion placement treats the edge as a hard boundary. A periodic MAPLE raster '
                 'interpretation would change both cover and overlap and must be requalified by the adapter, not '
                 'assumed equal')
GEOMETRY_UNITS = {'x_m': 'm (landscape frame)', 'y_m': 'm (landscape frame)', 'diameter_m': 'm', 'radius_m': 'm',
                  'height_m': 'm (intact allometric)', 'root_radius_m': 'm', 'establishment_ground_m': 'm elevation',
                  'porosity': 'fraction [0,1]', 'current_ground_m': 'm elevation',
                  'surface_delta_m': 'm (+ deposition, - erosion)', 'burial_depth_m': 'm >= 0',
                  'exposure_depth_m': 'm >= 0', 'exposed_height_m': 'm >= 0'}


class ResourceBoundError(ValueError):
    '''A declared cell/byte bound would be exceeded; raised before any large allocation.'''


def _check_catalog(state, catalog):
    if not isinstance(catalog, _traits.SpeciesCatalog):
        raise _state.StateError('catalog must be a leaf.traits.SpeciesCatalog')
    if not isinstance(state, _state.PlantState):
        raise _state.StateError('state must be a leaf.state.PlantState')
    if catalog.hash != state.catalog_hash:
        raise _state.StateError(f'catalog hash {catalog.hash} differs from the state catalog hash {state.catalog_hash}')
    try:
        catalog.check_species_table(state.species_names)
    except _traits.CatalogError as exc:
        raise _state.StateError(str(exc)) from exc


# ------------------------------------------------------------------ geometry
def derive_geometry(state, catalog, *, current_surface=None):
    '''O(P) derived geometry arrays for ALL plants (every status). Pure: the state is not modified; the
    state and every derived output are validated (finite) before anything is returned.'''
    _check_catalog(state, catalog)
    state.validate()
    idx = state.species_index.astype(np.int64)
    B = state.agb_dry_kg
    al = catalog.allometry_arrays()
    with np.errstate(over='ignore', invalid='ignore'):
        diameter = al['a_d'][idx] * np.power(B, al['b_d'][idx])
        height = al['a_h'][idx] * np.power(B, al['b_h'][idx])
    if not (np.all(np.isfinite(diameter)) and np.all(np.isfinite(height))):
        raise _state.StateError('derived allometric geometry is not finite (overflow); refusing to publish')
    radius = 0.5 * diameter
    out = {'n': state.n, 'plant_id': state.ids.copy(), 'species_index': state.species_index.copy(),
           'species': state.species_of(), 'status': state.status.copy(), 'status_name': state.status_names(),
           'x_m': state.x_m.copy(), 'y_m': state.y_m.copy(), 'diameter_m': diameter, 'radius_m': radius,
           'height_m': height, 'root_radius_m': radius * catalog.root_multiplier_array()[idx],
           'establishment_ground_m': state.establishment_ground_m.copy(),
           'establishment_unresolved': state.establishment_unresolved.copy(),
           'porosity': catalog.porosity_array()[idx], 'agb_dry_kg': B.copy(),
           'catalog_hash': state.catalog_hash, 'time_days': state.time_days, 'domain': state.domain.to_dict(),
           'coordinate_frame': _state.COORDINATE_FRAME,
           'height_definition': 'intact allometric height above establishment ground (not exposed height)',
           'canopy_shape': 'circle (semi_major == semi_minor == radius_m, orientation 0)',
           'porosity_note': 'constant per-species catalog value; unused by LEAF ecology, USED by native MAPLE wind',
           'boundary_note': BOUNDARY_NOTE, 'units': dict(GEOMETRY_UNITS)}
    if current_surface is not None:
        state.require_resolved('burial/exposure diagnostics')
        cur = state.domain.sample(current_surface, state.x_m, state.y_m, 'current surface')
        delta = cur - state.establishment_ground_m
        if not np.all(np.isfinite(delta)):
            raise _state.StateError('surface delta is not finite')
        burial = np.maximum(delta, 0.0)
        out.update({'current_ground_m': cur, 'surface_delta_m': delta, 'burial_depth_m': burial,
                    'exposure_depth_m': np.maximum(-delta, 0.0),
                    'exposed_height_m': np.maximum(height - burial, 0.0),
                    'fully_buried': burial >= height,
                    'diagnostics_note': 'geometric burial/exposure only; no dislodgement, injury or mortality inferred'})
    return out


def maple_vegetation_view(state, catalog, current_surface, *, statuses=('live', 'dead_standing'),
                          height_mode='exposed', unsupported_policy='error', drag_coefficient=1.0):
    '''Plain-array view compatible with MAPLE's VegetationObjectSet fields plus stable-ID sidecars.

    `current_surface` (ny, nx) is REQUIRED: placement is validated against the anchor of every plant.
    Coordinates are DOMAIN-LOCAL metres (origin subtracted). Only plants whose status is in `statuses`
    with diameter > 0 are included. height_mode 'exposed' (safe default): height = intact - burial, canopy
    base on the current surface. Unsupported for native MAPLE: eroded anchors (elevated base) and fully
    buried plants; policy 'error' raises, 'exclude' drops them (counted), 'flag' keeps them (zero/exposed
    height) with sidecar flags and `native_maple_valid: False`. height_mode 'intact_diagnostic' publishes
    intact heights ignoring burial; always labelled and never native-MAPLE-valid.'''
    if current_surface is None:
        raise ValueError('maple_vegetation_view requires the CURRENT surface (ny, nx): publication validates every '
                         'canopy against its establishment anchor (pass the initial surface right after resolution)')
    if height_mode not in HEIGHT_MODES:
        raise ValueError(f'height_mode must be one of {HEIGHT_MODES}')
    if unsupported_policy not in UNSUPPORTED_POLICIES:
        raise ValueError(f'unsupported_policy must be one of {UNSUPPORTED_POLICIES}')
    drag = _traits.nonnegative(drag_coefficient, 'drag_coefficient')
    state.require_resolved('publishing a MAPLE vegetation view')
    codes = sorted(set(int(c) for c in _state.status_codes(list(statuses))))
    geo = derive_geometry(state, catalog, current_surface=current_surface)
    in_status = np.isin(geo['status'], codes)
    sel = in_status & (geo['diameter_m'] > 0.0)
    excluded_status = int((~in_status).sum())
    excluded_zero = int((in_status & ~(geo['diameter_m'] > 0.0)).sum())
    elevated = geo['exposure_depth_m'] > 0.0
    buried = geo['fully_buried']
    unsupported = sel & (elevated | buried)
    n_unsupported = int(unsupported.sum())
    if n_unsupported and unsupported_policy == 'error':
        ids = geo['plant_id'][unsupported][:10].tolist()
        raise _state.StateError(f'{n_unsupported} plant(s) cannot be represented by native MAPLE (eroded anchor with '
                                f'elevated canopy base: {int((sel & elevated).sum())}, fully buried: '
                                f"{int((sel & buried).sum())}), e.g. ids {ids}; choose unsupported_policy='exclude' or "
                                "'flag' explicitly")
    if unsupported_policy == 'exclude':
        sel &= ~unsupported
    flagged_kept = int((sel & unsupported).sum())
    height = geo['exposed_height_m'] if height_mode == 'exposed' else geo['height_m']
    dom = state.domain
    k = int(sel.sum())
    fields = {'center_x_m': geo['x_m'][sel] - dom.origin_x_m, 'center_y_m': geo['y_m'][sel] - dom.origin_y_m,
              'semi_major_m': geo['radius_m'][sel].copy(), 'semi_minor_m': geo['radius_m'][sel].copy(),
              'orientation_deg': np.zeros(k, dtype=np.float64), 'height_m': height[sel].copy(),
              'porosity': geo['porosity'][sel].copy(), 'drag_coefficient': np.full(k, drag, dtype=np.float64)}
    for name, arr in fields.items():
        if not np.all(np.isfinite(arr)):
            raise _state.StateError(f'MAPLE view field {name} is not finite')
    sidecar = {'plant_id': geo['plant_id'][sel], 'species': geo['species'][sel], 'status': geo['status_name'][sel],
               'establishment_ground_m': geo['establishment_ground_m'][sel], 'current_ground_m': geo['current_ground_m'][sel],
               'burial_depth_m': geo['burial_depth_m'][sel], 'exposure_depth_m': geo['exposure_depth_m'][sel],
               'intact_height_m': geo['height_m'][sel], 'exposed_height_m': geo['exposed_height_m'][sel],
               'root_radius_m': geo['root_radius_m'][sel], 'elevated_base': elevated[sel], 'fully_buried': buried[sel],
               'native_maple_valid': ~unsupported[sel] if height_mode == 'exposed' else np.zeros(k, dtype=bool)}
    native_valid = height_mode == 'exposed' and flagged_kept == 0
    meta = {'schema': 'leaf.maple_vegetation_view', 'schema_version': 2, 'n_objects': k, 'fields': list(MAPLE_FIELDS),
            'coordinates': 'domain-local metres: center = landscape position - domain origin; +x east, +y north; '
                           f'MAPLE cell centres ((col+0.5)*dx, (row+0.5)*dy) with dx={dom.dx_m}, dy={dom.dy_m}',
            'domain': dom.to_dict(), 'catalog_hash': state.catalog_hash, 'time_days': state.time_days,
            'height_mode': height_mode,
            'height_definition': ('exposed height = intact allometric height - burial depth; canopy base on the '
                                  'current surface' if height_mode == 'exposed' else
                                  'DIAGNOSTIC APPROXIMATION: intact allometric height ignoring burial/exposure; '
                                  'NOT a native-MAPLE-valid canopy set'),
            'native_maple_valid': native_valid,
            'validity_note': ('every object is representable by native MAPLE (base on the current surface, positive '
                              'exposed height)' if native_valid else
                              'NOT native-MAPLE-valid: contains flagged unsupported objects and/or diagnostic heights; '
                              'do not feed to native wind without explicit handling'),
            'surface_reference': 'current surface sampled at each plant (containing cell) vs stored establishment anchor',
            'statuses_included': [_state.STATUS_NAMES[c] for c in codes], 'excluded_by_status': excluded_status,
            'excluded_zero_diameter': excluded_zero, 'unsupported_policy': unsupported_policy,
            'n_unsupported': n_unsupported,
            'n_unsupported_excluded': n_unsupported if unsupported_policy == 'exclude' else 0,
            'n_unsupported_flagged_kept': flagged_kept,
            'porosity_note': 'constant per-species catalog value; native MAPLE wind uses it, LEAF ecology does not',
            'drag_coefficient_note': 'uniform pass-through metadata value, not a LEAF trait',
            'boundary_note': BOUNDARY_NOTE,
            'rasterization': "performed by MAPLE's existing rasterizer; LEAF keeps plant_id sidecars beside it"}
    return {'fields': fields, 'sidecar': sidecar, 'meta': meta}


# ------------------------------------------------------------------ cover measurement
def cover_admission(domain, subsamples, *, total=True, max_cover_cells=DEFAULT_MAX_COVER_CELLS,
                    max_cover_bytes=DEFAULT_MAX_COVER_BYTES, max_working_bytes=DEFAULT_MAX_WORKING_BYTES):
    '''Precisely scoped memory estimate for measure_cover, checked BEFORE any allocation.

    raster_bytes  = (2 if total else 1) * (ny*s) * (nx*s) boolean cells
    working_bytes = 16*(NX+NY) 1-D coordinate scratch + rows_per_chunk*NX*9 footprint temporaries
    rows_per_chunk is the largest value fitting max_working_bytes (at least one raster row).'''
    if not isinstance(domain, _state.Domain):
        raise _state.StateError('domain must be a leaf.state.Domain')
    s = _traits.strict_int(subsamples, 'subsamples', 1)
    mc = _traits.strict_int(max_cover_cells, 'max_cover_cells', 1)
    mb = _traits.strict_int(max_cover_bytes, 'max_cover_bytes', 1)
    mw = _traits.strict_int(max_working_bytes, 'max_working_bytes', 1)
    NX, NY = domain.nx * s, domain.ny * s
    cells = NX * NY
    raster_bytes = cells * (2 if total else 1)
    coord_bytes = 16 * (NX + NY)
    rows = (mw - coord_bytes) // (_CHUNK_BYTES_PER_ELEMENT * NX)
    if cells > mc or raster_bytes > mb:
        raise ResourceBoundError(f'cover raster {NY}x{NX} = {cells} cells / {raster_bytes} bytes exceeds '
                                 f'max_cover_cells={mc} / max_cover_bytes={mb}; lower subsamples or raise the bound explicitly')
    if rows < 1:
        raise ResourceBoundError(f'max_working_bytes={mw} cannot hold one raster row footprint '
                                 f'({coord_bytes + _CHUNK_BYTES_PER_ELEMENT * NX} bytes needed)')
    rows = int(min(rows, NY))
    working_bytes = coord_bytes + rows * NX * _CHUNK_BYTES_PER_ELEMENT
    return {'subsamples': s, 'raster_shape': [NY, NX], 'raster_cells': cells, 'raster_bytes': raster_bytes,
            'working_bytes': working_bytes, 'rows_per_chunk': rows, 'total_bytes': raster_bytes + working_bytes,
            'max_cover_cells': mc, 'max_cover_bytes': mb, 'max_working_bytes': mw,
            'scope': 'raster_bytes: boolean union raster(s); working_bytes: 1-D coordinate scratch + row-tiled '
                     'footprint temporaries (float64 distance + bool mask). P-sized index arrays are reported as '
                     'index_bytes by measure_cover and scale with the population, not the grid'}


def measure_cover(domain, x_m, y_m, radius_m, group_index=None, n_groups=1, *, subsamples=4,
                  max_cover_cells=DEFAULT_MAX_COVER_CELLS, max_cover_bytes=DEFAULT_MAX_COVER_BYTES,
                  max_working_bytes=DEFAULT_MAX_WORKING_BYTES, total=True):
    '''Per-group union ground cover of circles clipped to the finite domain (fraction of the domain area).

    Raster midpoints are origin + (i + 0.5) * dx / s; a midpoint is covered when strictly inside a circle.
    Same-group overlaps count once; different groups are measured independently (cross-group overlap adds
    to each group's cover). `total_union_fraction` is the union over all groups. Deterministic, O(P) work on
    row-tiled bounding boxes; memory admitted by cover_admission before allocation.'''
    adm = cover_admission(domain, subsamples, total=total, max_cover_cells=max_cover_cells,
                          max_cover_bytes=max_cover_bytes, max_working_bytes=max_working_bytes)
    s = adm['subsamples']
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
    NY, NX = adm['raster_shape']
    rows_per_chunk = adm['rows_per_chunk']
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
            xs2 = x0 + (np.arange(i0, i1 + 1, dtype=np.float64) + 0.5) * hx - px
            xs2 *= xs2
            ys2 = y0 + (np.arange(j0, j1 + 1, dtype=np.float64) + 0.5) * hy - py
            ys2 *= ys2
            r2 = rp * rp
            for ja in range(0, ys2.shape[0], rows_per_chunk):   # row tiling bounds the footprint temporaries
                jb = min(ja + rows_per_chunk, ys2.shape[0])
                raster[j0 + ja:j0 + jb, i0:i1 + 1] |= (xs2[None, :] + ys2[ja:jb, None]) < r2
        covered = int(np.count_nonzero(raster))
        if union_all is not None:
            union_all |= raster
        rr = r[sel]
        rpos = rr[rr > 0.0]
        groups.append({'group': gi, 'count': int(sel.shape[0]), 'covered_cells': covered,
                       'union_fraction': covered / adm['raster_cells'], 'union_area_m2': covered / adm['raster_cells'] * A,
                       'nominal_area_m2': nominal, 'nominal_fraction': nominal / A,
                       'mean_radius_m': float(rpos.mean()) if rpos.size else None,
                       'relative_error_scale': (float(max(hx, hy) / rpos.mean()) if rpos.size else None)})
    return {'method': COVER_METHOD, 'subsamples': s, 'raster_shape': [NY, NX], 'resolution_m': [hx, hy],
            'raster_cells': adm['raster_cells'], 'raster_bytes': adm['raster_bytes'], 'working_bytes': adm['working_bytes'],
            'rows_per_chunk': rows_per_chunk, 'index_bytes': int(8 * n + g.nbytes), 'admission': adm,
            'domain_area_m2': A, 'n_circles': n, 'groups': groups,
            'total_union_fraction': (int(np.count_nonzero(union_all)) / adm['raster_cells']) if union_all is not None else None,
            'sum_of_group_union_fractions': float(sum(gr['union_fraction'] for gr in groups)),
            'definition': 'per-group projected union ground cover clipped to the finite non-periodic domain '
                          '(same-group overlap counted once; cross-group overlap counted in each group); nominal = sum '
                          'of unclipped crown areas / domain area',
            'boundary_note': BOUNDARY_NOTE,
            'error_basis': 'midpoint sampling at resolution dx/s: each circle boundary is resolved to about one '
                           'raster cell, relative error per circle ~ resolution / radius; sampled, not analytic'}


__all__ = ['COVER_METHOD', 'DEFAULT_MAX_COVER_CELLS', 'DEFAULT_MAX_COVER_BYTES', 'DEFAULT_MAX_WORKING_BYTES',
           'MAPLE_FIELDS', 'HEIGHT_MODES', 'UNSUPPORTED_POLICIES', 'BOUNDARY_NOTE', 'GEOMETRY_UNITS',
           'ResourceBoundError', 'derive_geometry', 'maple_vegetation_view', 'cover_admission', 'measure_cover']
