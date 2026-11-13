'''leaf.canopy: derived plant geometry, the shared aboveground canopy view, the thin MAPLE adapter and bounded union cover.

Geometry is DERIVED from the canonical PlantState + species catalog on request; it is never an
independently editable plant authority. Canopies are circles (equal semi-axes) of allometric diameter
centred on (x_m, y_m); roots are circles of radius lateral_radius_multiplier * canopy radius. Both may span
many raster cells. `derive_geometry` is the only allometry / surface-sampling implementation on the canopy
export path: every published view below derives from it.

Heights. `derive_geometry` returns the INTACT allometric height above the plant's establishment ground and,
when a current surface is supplied, the geometric diagnostics `surface_delta_m = current - establishment`
(+ deposition/burial, - erosion/exposure), `burial_depth_m`, `exposure_depth_m`,
`exposed_height_m = max(height - burial, 0)` and `fully_buried`. No root dislodgement, injury or mortality
is inferred.

Shared canopy view (`canopy_view`, schema `leaf.canopy_view` v1). The consumer-neutral aboveground
representation for MAPLE AND SYRUP (and any later consumer): the selected standing population (default
statuses live + dead_standing, positive canopy diameter) as O(P) plain arrays in domain-local metres
(+x east, +y north, cell centres ((col+.5)*dx, (row+.5)*dy)) with fields CANOPY_FIELDS = center_x_m,
center_y_m, semi_major_m == semi_minor_m, orientation_deg (0), height_m, porosity, drag_coefficient, plus
stable-ID sidecars (plant_id, species, status, anchors, burial/exposure, intact/exposed heights, root
radius, informational native-MAPLE flags). It REQUIRES the current surface, is the ONE place where
landscape positions and canopy fields are built, and RETAINS fully buried (exposed height 0) and
elevated-base (eroded anchor) plants with their diagnostics: geometric visibility relative to the current
surface is distinct from plant death or removal, which only `status` expresses. Exposed height is
max(intact - burial, 0): partial burial shortens the canopy and its visible base is the current ground;
erosion leaves the intact height unchanged with the base at the FIXED establishment ground above the current
ground (the gap is exposure_depth_m); a fully buried plant has no visible canopy and keeps its anchor. No
rasterization, physics, state mutation or caching. Because every consumer calls the same code, consumers
that use the SAME plant state, catalog and current surface at matching coupling times obtain consistent
geometry; upstream adoption by MAPLE and SYRUP is pending and no integrated run is implied.

MAPLE adapter (`maple_vegetation_view`, schema `leaf.maple_vegetation_view` v2). MAPLE's VegetationObjectSet
(as inspected by Codex) places every canopy base on the CURRENT surface and cannot hold fully buried or
elevated-base objects. The adapter is a thin compatibility layer OVER canopy_view: it only applies the
native-MAPLE unsupported policy ('error' refuses, 'exclude' drops them with ONE index selection over every
plant array, 'flag' keeps them with sidecar flags and `native_maple_valid: False`) and adds compatibility
metadata. `height_mode='intact_diagnostic'` publishes intact heights ignoring burial; it is a labelled
DIAGNOSTIC APPROXIMATION, never native-MAPLE-valid. MAPLE's own rasterizer owns raster semantics; LEAF hands
over plain arrays plus stable-ID sidecars and never imports MAPLE or SYRUP. SYRUP (or any other consumer)
calls canopy_view directly and does not inherit the MAPLE container restrictions.

Cover (`measure_cover`). Each group's circles are rasterised onto a (ny*s, nx*s) boolean midpoint raster
and the union fraction of the domain is reported. It is a deterministic
SAMPLED approximation (resolution dx/s), never an exact analytic area. Memory is admitted before any
allocation with two precisely scoped terms: raster bytes (one or two boolean rasters, bound
`max_cover_bytes`) and working bytes (row-tiled footprint temporaries + 1-D coordinate scratch, bound
`max_working_bytes`); P-sized index arrays are reported separately. Never P x N or P x P.
Boundaries: on a NONPERIODIC axis (the default) circles are clipped at the domain edge and never wrap (numerically
unchanged); on a PERIODIC axis (`Domain.periodic`) a midpoint is covered when its minimum-image distance to the centre is
inside the circle, so seam fragments (corners included) are rasterised once into the union, however many images reach the
cell. Mixed axes clip only the nonperiodic axis. The nominal sum of pi r^2 is reported separately and is NOT a claim about the
physical torus union for circles wider than half a periodic span (`leaf.periodic`).
The shared canopy view / MAPLE adapter publish ONE object per plant (the centre inside the domain) plus boundary metadata;
MAPLE's own rasterizer wraps the physical footprint across periodic seams. Public periodic centres are canonical half-open
[0, L): the exact stored upper endpoint is exported as 0.
'''
from __future__ import annotations

import math

import numpy as np

from leaf import periodic as _periodic
from leaf import state as _state
from leaf import traits as _traits

COVER_METHOD = 'midpoint_raster_union'
DEFAULT_MAX_COVER_CELLS = 64_000_000
DEFAULT_MAX_COVER_BYTES = 128_000_000
DEFAULT_MAX_WORKING_BYTES = 1 << 20
_CHUNK_BYTES_PER_ELEMENT = 9   # float64 squared-distance temporary + bool comparison per footprint element
CANOPY_FIELDS = ('center_x_m', 'center_y_m', 'semi_major_m', 'semi_minor_m', 'orientation_deg', 'height_m',
                 'porosity', 'drag_coefficient')
MAPLE_FIELDS = CANOPY_FIELDS   # compatibility alias: the native MAPLE object fields ARE the shared canopy fields
CANOPY_VIEW_SCHEMA, CANOPY_VIEW_SCHEMA_VERSION = 'leaf.canopy_view', 1
MAPLE_VIEW_SCHEMA, MAPLE_VIEW_SCHEMA_VERSION = 'leaf.maple_vegetation_view', 2
HEIGHT_MODES = ('exposed', 'intact_diagnostic')
UNSUPPORTED_POLICIES = ('error', 'exclude', 'flag')
CANOPY_FIELD_UNITS = {'center_x_m': 'm, domain-local easting = landscape x_m - origin_x_m',
                      'center_y_m': 'm, domain-local northing = landscape y_m - origin_y_m',
                      'semi_major_m': 'm, canopy radius (== semi_minor_m: circles)',
                      'semi_minor_m': 'm, canopy radius (== semi_major_m: circles)',
                      'orientation_deg': 'degrees, always 0 (circles)',
                      'height_m': 'm above the canopy base; see height_definition / height_anchor',
                      'porosity': 'fraction [0,1], constant per-species catalog value',
                      'drag_coefficient': 'dimensionless uniform pass-through value'}
CANOPY_SIDECAR_UNITS = {'plant_id': 'stable int64 plant id', 'species': 'catalog species id',
                        'status': 'ecological status name (live | dead_standing | removed); NOT a visibility flag',
                        'establishment_ground_m': 'm elevation at establishment (stored anchor)',
                        'current_ground_m': 'm elevation of the current surface (containing cell)',
                        'burial_depth_m': 'm >= 0 (current - establishment when positive)',
                        'exposure_depth_m': 'm >= 0 (establishment - current when positive)',
                        'intact_height_m': 'm, intact allometric height above establishment ground',
                        'exposed_height_m': 'm >= 0, intact height - burial depth',
                        'root_radius_m': 'm', 'elevated_base': 'bool, canopy base above the current surface (eroded anchor)',
                        'fully_buried': 'bool, exposed height <= 0 (geometric visibility, not death)',
                        'native_maple_valid': 'bool, INFORMATIONAL: representable by the native MAPLE object container'}
BOUNDARY_NOTE = ('finite closed non-periodic domain: canopies are clipped at the domain edge for cover and never '
                 'wrap; exclusion placement treats the edge as a hard boundary. A periodic MAPLE raster '
                 'interpretation would change both cover and overlap and must be requalified by the adapter, not '
                 'assumed equal')
COVER_DEFINITION = ('per-group projected union ground cover clipped to the finite non-periodic domain '
                    '(same-group overlap counted once; cross-group overlap counted in each group); nominal = sum '
                    'of unclipped crown areas / domain area')
COVER_DEFINITION_PERIODIC = ('per-group projected union ground cover of the domain with periodic axes wrapped (torus union: a '
                             'midpoint is covered when its minimum-image distance to a centre is inside the circle; same-group '
                             'overlap, including a circle with its own wrapped image, counted once; cross-group overlap counted in '
                             'each group); nonperiodic axes are clipped; nominal = sum of unclipped crown areas / domain area, '
                             'which may exceed the physical union')


def boundary_note(domain):
    '''Boundary note of this domain: the legacy finite-domain text when nothing is periodic, else the periodic description.'''
    return _periodic.BOUNDARY_NOTE_PERIODIC if domain.any_periodic else BOUNDARY_NOTE


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
           'boundary_note': boundary_note(state.domain), 'units': dict(GEOMETRY_UNITS)}
    if state.domain.any_periodic:
        out['boundary'] = _periodic.boundary_record(state.domain)
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


# ------------------------------------------------------------------ shared canopy view (MAPLE and SYRUP)
def canopy_view(state, catalog, current_surface, *, statuses=('live', 'dead_standing'), height_mode='exposed',
                drag_coefficient=1.0):
    '''Shared, consumer-neutral aboveground canopy representation (schema `leaf.canopy_view` v1).

    Returns {'fields', 'sidecar', 'meta'}: O(P) plain arrays for the selected standing population (status in
    `statuses`, canopy diameter > 0), derived on request from the authoritative PlantState through
    `derive_geometry` (the single allometry / surface-sampling implementation of the canopy export path).
    `current_surface` (ny, nx) is REQUIRED so that burial/exposure diagnostics and exposed heights are always
    defined. Coordinates are DOMAIN-LOCAL metres (landscape position minus domain origin, +x east, +y north);
    this function is the ONE place where that conversion and the canopy fields are built. height_mode
    'exposed' (default): height_m = max(intact - burial, 0); the visible canopy base is at
    max(establishment_ground_m, current_ground_m) - the current ground under partial burial, the FIXED
    establishment ground under erosion (the intact height is unchanged and exposure_depth_m is the separate
    gap below the base); a fully buried plant has height 0, no visible canopy and keeps its anchor.
    'intact_diagnostic': intact allometric height ignoring burial (labelled diagnostic). Fully buried and
    elevated-base (eroded anchor) plants are RETAINED with their diagnostics: visibility is geometric,
    `status` alone carries death/removal. Per-plant `native_maple_valid` flags are informational metadata
    for the MAPLE adapter, not a validity judgment of this view. Pure: no state mutation, caching or
    rasterization.'''
    if current_surface is None:
        raise ValueError('canopy_view requires the CURRENT surface (ny, nx): every canopy is placed against its '
                         'establishment anchor (pass the initial surface right after resolution)')
    if height_mode not in HEIGHT_MODES:
        raise ValueError(f'height_mode must be one of {HEIGHT_MODES}')
    drag = _traits.nonnegative(drag_coefficient, 'drag_coefficient')
    state.require_resolved('publishing a canopy view')
    codes = sorted(set(int(c) for c in _state.status_codes(list(statuses))))
    geo = derive_geometry(state, catalog, current_surface=current_surface)
    in_status = np.isin(geo['status'], codes)
    positive = geo['diameter_m'] > 0.0
    sel = in_status & positive
    k = int(sel.sum())
    elevated = geo['exposure_depth_m'][sel] > 0.0
    buried = geo['fully_buried'][sel]
    height = geo['exposed_height_m'] if height_mode == 'exposed' else geo['height_m']
    dom = state.domain
    # the ONE place where landscape positions become domain-local canopy fields
    if dom.any_periodic:
        # canonical PUBLIC centres: half-open [0, L) on a periodic axis (the exact stored upper endpoint is exported as 0, decided by
        # landscape equality, so decimal spacings / origins behave); the stored PlantState record is not changed; <= 2 B / plant masks
        cx, cy = _periodic.localize_inplace(dom, geo['x_m'][sel], geo['y_m'][sel])
    else:
        cx, cy = geo['x_m'][sel] - dom.origin_x_m, geo['y_m'][sel] - dom.origin_y_m
    fields = {'center_x_m': cx, 'center_y_m': cy,
              'semi_major_m': geo['radius_m'][sel].copy(), 'semi_minor_m': geo['radius_m'][sel].copy(),
              'orientation_deg': np.zeros(k, dtype=np.float64), 'height_m': height[sel].copy(),
              'porosity': geo['porosity'][sel].copy(), 'drag_coefficient': np.full(k, drag, dtype=np.float64)}
    for name, arr in fields.items():
        if not np.all(np.isfinite(arr)):
            raise _state.StateError(f'canopy view field {name} is not finite')
    representable = ~(elevated | buried) if height_mode == 'exposed' else np.zeros(k, dtype=bool)
    sidecar = {'plant_id': geo['plant_id'][sel], 'species': geo['species'][sel], 'status': geo['status_name'][sel],
               'establishment_ground_m': geo['establishment_ground_m'][sel], 'current_ground_m': geo['current_ground_m'][sel],
               'burial_depth_m': geo['burial_depth_m'][sel], 'exposure_depth_m': geo['exposure_depth_m'][sel],
               'intact_height_m': geo['height_m'][sel], 'exposed_height_m': geo['exposed_height_m'][sel],
               'root_radius_m': geo['root_radius_m'][sel], 'elevated_base': elevated, 'fully_buried': buried,
               'native_maple_valid': representable}
    n_buried, n_elevated = int(buried.sum()), int(elevated.sum())
    n_unsupported = int((elevated | buried).sum())
    meta = {'schema': CANOPY_VIEW_SCHEMA, 'schema_version': CANOPY_VIEW_SCHEMA_VERSION, 'n_objects': k,
            'fields': list(CANOPY_FIELDS), 'sidecar_fields': list(sidecar),
            'coordinates': 'domain-local metres: center = landscape position - domain origin; +x east, +y north; '
                           f'raster cell centres ((col+0.5)*dx, (row+0.5)*dy) with dx={dom.dx_m}, dy={dom.dy_m}',
            'coordinate_frame': 'domain-local (landscape frame minus origin_x_m / origin_y_m); PlantState keeps '
                                'the landscape frame',
            'units': dict(CANOPY_FIELD_UNITS), 'sidecar_units': dict(CANOPY_SIDECAR_UNITS),
            'domain': dom.to_dict(), 'catalog_hash': state.catalog_hash, 'time_days': state.time_days,
            'height_mode': height_mode,
            'height_definition': ('exposed height = max(intact allometric height - burial depth, 0); 0 for fully '
                                  'buried plants; erosion does not change it (the intact height stays, the exposed '
                                  'gap is exposure_depth_m)' if height_mode == 'exposed' else
                                  'DIAGNOSTIC APPROXIMATION: intact allometric height above the establishment ground, '
                                  'ignoring burial/exposure'),
            'height_anchor': ('visible canopy base at max(establishment_ground_m, current_ground_m): the current '
                              'ground under partial burial, the FIXED establishment ground under erosion (the '
                              'establishment anchor stays above the current ground; exposure_depth_m is the separate '
                              'gap below the base); fully buried plants have exposed height 0, no visible canopy and '
                              'retain their establishment anchor' if height_mode == 'exposed' else
                              'canopy base at the ESTABLISHMENT ground (establishment_ground_m)'),
            'canopy_shape': 'circle (semi_major_m == semi_minor_m == canopy radius, orientation 0)',
            'surface_reference': 'current surface sampled at each plant (containing cell) vs stored establishment anchor',
            'statuses_included': [_state.STATUS_NAMES[c] for c in codes],
            'excluded_by_status': int((~in_status).sum()), 'excluded_zero_diameter': int((in_status & ~positive).sum()),
            'n_fully_buried': n_buried, 'n_elevated_base': n_elevated,
            'visibility_note': 'fully_buried, elevated_base, burial/exposure depths and exposed_height_m describe '
                               'GEOMETRIC visibility relative to the current surface; the ecological state is `status` '
                               'alone. Fully buried and elevated-base plants are RETAINED in this view; nothing here '
                               'implies death, dislodgement or removal',
            'native_maple_compatibility': {'n_unsupported': n_unsupported, 'n_elevated_base': n_elevated,
                                           'n_fully_buried': n_buried, 'all_representable': n_unsupported == 0 and height_mode == 'exposed',
                                           'note': 'INFORMATIONAL per-plant flags (sidecar native_maple_valid): the native MAPLE '
                                                   'object container needs the canopy base on the current surface and a '
                                                   'positive exposed height; this is a container limitation, not a physical '
                                                   'validity judgment of the shared view. maple_vegetation_view applies the '
                                                   'MAPLE policies'},
            'authority': 'derived on request from the authoritative PlantState + species catalog; not an editable '
                         'population; recompute after any state or surface change',
            'porosity_note': 'constant per-species catalog value; unused by LEAF ecology; native MAPLE wind uses it; '
                             'no hydrological interpretation is defined here',
            'drag_coefficient_note': 'uniform pass-through metadata value, not a LEAF trait',
            'boundary_note': boundary_note(dom),
            'rasterization': 'none here (O(P) arrays); a consumer that needs a raster derives it from these arrays '
                             'with its own rasterizer and keeps plant_id beside it',
            'consumers': 'shared representation for MAPLE and SYRUP (and any later consumer); no consumer-specific '
                         'filtering applied; MAPLE uses maple_vegetation_view for its container policy'}
    if dom.any_periodic:
        # ONE object per plant, centre inside the closed domain; the consumer wraps the footprint itself (no image records)
        meta['boundary'] = _periodic.boundary_record(dom)
        meta['boundary']['export'] = ('one object per plant (no wrapped copies, no image rows); centres are the stored positions in '
                                      'domain-local metres, CANONICAL half-open [0, L) on a periodic axis: the exact stored upper '
                                      'endpoint is exported as 0 (the MAPLE object container refuses upper centres); the consumer '
                                      'wraps the physical canopy footprint across the periodic seams itself')
    return {'fields': fields, 'sidecar': sidecar, 'meta': meta}


# ------------------------------------------------------------------ thin MAPLE compatibility adapter
def maple_vegetation_view(state, catalog, current_surface, *, statuses=('live', 'dead_standing'),
                          height_mode='exposed', unsupported_policy='error', drag_coefficient=1.0):
    '''Plain-array view compatible with MAPLE's VegetationObjectSet fields plus stable-ID sidecars
    (schema `leaf.maple_vegetation_view` v2).

    Thin compatibility adapter OVER `canopy_view`: no allometry, surface sampling, canopy geometry or
    coordinate conversion happens here. It only applies the native-MAPLE policy for plants the MAPLE object
    container cannot hold (eroded anchors with an elevated base, fully buried plants): 'error' raises with
    the offending IDs, 'exclude' drops them (counted) using ONE index selection over every field and
    sidecar array, 'flag' keeps them with sidecar flags and `native_maple_valid: False`. height_mode
    'intact_diagnostic' is labelled and never native-MAPLE-valid. `current_surface` is REQUIRED (by the
    shared view). Signature, defaults and output schema are unchanged.'''
    if unsupported_policy not in UNSUPPORTED_POLICIES:
        raise ValueError(f'unsupported_policy must be one of {UNSUPPORTED_POLICIES}')
    shared = canopy_view(state, catalog, current_surface, statuses=statuses, height_mode=height_mode,
                         drag_coefficient=drag_coefficient)
    sc, sm = shared['sidecar'], shared['meta']
    unsupported = sc['elevated_base'] | sc['fully_buried']
    n_unsupported = int(unsupported.sum())
    if n_unsupported and unsupported_policy == 'error':
        ids = sc['plant_id'][unsupported][:10].tolist()
        raise _state.StateError(f'{n_unsupported} plant(s) cannot be represented by native MAPLE (eroded anchor with '
                                f"elevated canopy base: {int(sc['elevated_base'].sum())}, fully buried: "
                                f"{int(sc['fully_buried'].sum())}), e.g. ids {ids}; choose unsupported_policy='exclude' or "
                                "'flag' explicitly (the shared canopy_view retains these plants)")
    if unsupported_policy == 'exclude':
        keep = np.flatnonzero(~unsupported)   # the SAME index selection for every plant array (stable ID association)
        fields = {name: arr[keep] for name, arr in shared['fields'].items()}
        sidecar = {name: arr[keep] for name, arr in sc.items()}
    else:
        fields, sidecar = dict(shared['fields']), dict(sc)
    k = int(sidecar['plant_id'].shape[0])
    flagged_kept = int((sidecar['elevated_base'] | sidecar['fully_buried']).sum())
    native_valid = height_mode == 'exposed' and flagged_kept == 0
    dom = state.domain
    meta = {'schema': MAPLE_VIEW_SCHEMA, 'schema_version': MAPLE_VIEW_SCHEMA_VERSION, 'n_objects': k,
            'fields': list(MAPLE_FIELDS),
            'coordinates': 'domain-local metres: center = landscape position - domain origin; +x east, +y north; '
                           f'MAPLE cell centres ((col+0.5)*dx, (row+0.5)*dy) with dx={dom.dx_m}, dy={dom.dy_m}',
            'domain': sm['domain'], 'catalog_hash': sm['catalog_hash'], 'time_days': sm['time_days'],
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
            'surface_reference': sm['surface_reference'],
            'statuses_included': sm['statuses_included'], 'excluded_by_status': sm['excluded_by_status'],
            'excluded_zero_diameter': sm['excluded_zero_diameter'], 'unsupported_policy': unsupported_policy,
            'n_unsupported': n_unsupported,
            'n_unsupported_excluded': n_unsupported if unsupported_policy == 'exclude' else 0,
            'n_unsupported_flagged_kept': flagged_kept,
            'porosity_note': 'constant per-species catalog value; native MAPLE wind uses it, LEAF ecology does not',
            'drag_coefficient_note': 'uniform pass-through metadata value, not a LEAF trait',
            'boundary_note': boundary_note(dom),
            'rasterization': "performed by MAPLE's existing rasterizer; LEAF keeps plant_id sidecars beside it",
            'derived_from': {'schema': CANOPY_VIEW_SCHEMA, 'schema_version': CANOPY_VIEW_SCHEMA_VERSION,
                             'n_objects': sm['n_objects'],
                             'note': 'MAPLE-specific projection of the shared canopy_view: same geometry by plant_id; '
                                     'only the native-MAPLE unsupported policy and compatibility metadata are added here'}}
    if dom.any_periodic:
        meta['boundary'] = dict(sm['boundary'])
    return {'fields': fields, 'sidecar': sidecar, 'meta': meta}


# ------------------------------------------------------------------ cover measurement
def cover_admission(domain, subsamples, *, total=True, max_cover_cells=DEFAULT_MAX_COVER_CELLS,
                    max_cover_bytes=DEFAULT_MAX_COVER_BYTES, max_working_bytes=DEFAULT_MAX_WORKING_BYTES):
    '''Precisely scoped memory estimate for measure_cover, checked BEFORE any allocation.

    raster_bytes  = (2 if total else 1) * (ny*s) * (nx*s) boolean cells
    working_bytes = 16*(NX+NY) 1-D coordinate scratch + rows_per_chunk*NX*9 footprint temporaries
                    (a domain with a periodic axis counts 32*(NX+NY): index, offset, rint and distance scratch)
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
    coord_bytes = (32 if domain.any_periodic else 16) * (NX + NY)
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


def _cover_axis(p, r, origin, h, n, length, periodic):
    '''One axis of a circle footprint on the midpoint raster: ([(raster_start, k0, k1), ...] contiguous runs, squared distances d2[k]
    of the candidate midpoints to the (nearest image of the) centre), or (None, None) when empty. Nonperiodic: the legacy clipped
    range and arithmetic. Periodic: the unwrapped range (all n columns when it spans the axis) mapped modulo n with the
    minimum-image offset, so no column is listed twice.'''
    if not periodic:
        i0 = max(int(math.floor((p - r - origin) / h)), 0)
        i1 = min(int(math.floor((p + r - origin) / h)), n - 1)
        if i1 < i0:
            return None, None
        d = origin + (np.arange(i0, i1 + 1, dtype=np.float64) + 0.5) * h - p
        return [(i0, 0, i1 - i0 + 1)], d * d
    i0 = int(math.floor((p - r - origin) / h))
    i1 = int(math.floor((p + r - origin) / h))
    idx = np.arange(n, dtype=np.int64) if i1 - i0 + 1 >= n else np.arange(i0, i1 + 1, dtype=np.int64) % n
    d = origin + (idx.astype(np.float64) + 0.5) * h - p
    d -= length * np.rint(d / length)
    brk = np.flatnonzero(np.diff(idx) != 1) + 1
    starts = np.concatenate(([0], brk))
    ends = np.concatenate((brk, [idx.shape[0]]))
    return [(int(idx[a]), int(a), int(b)) for a, b in zip(starts, ends)], d * d


def _paint_wrapped(raster, px, py, rp, x0, y0, hx, hy, NX, NY, Lx, Ly, per, rows_per_chunk):
    '''OR the midpoints of one circle (and its periodic images) into the boolean raster: row tiles bound the footprint temporaries;
    every raster cell is addressed by at most one segment pair per axis, so a cell is covered once however many images reach it.'''
    xsegs, xd2 = _cover_axis(px, rp, x0, hx, NX, Lx, per[0])
    ysegs, yd2 = _cover_axis(py, rp, y0, hy, NY, Ly, per[1])
    if xd2 is None or yd2 is None:
        return
    r2 = rp * rp
    for ja in range(0, yd2.shape[0], rows_per_chunk):
        jb = min(ja + rows_per_chunk, yd2.shape[0])
        mask = (xd2[None, :] + yd2[ja:jb, None]) < r2
        for rs, k0, k1 in ysegs:
            a, b = max(k0, ja), min(k1, jb)
            if b <= a:
                continue
            for cs, c0, c1 in xsegs:
                raster[rs + (a - k0):rs + (b - k0), cs:cs + (c1 - c0)] |= mask[a - ja:b - ja, c0:c1]
        del mask        # release the tile BEFORE the next tile's float sum + mask are built (else 10, not the admitted 9, bytes / element)


def measure_cover(domain, x_m, y_m, radius_m, group_index=None, n_groups=1, *, subsamples=4,
                  max_cover_cells=DEFAULT_MAX_COVER_CELLS, max_cover_bytes=DEFAULT_MAX_COVER_BYTES,
                  max_working_bytes=DEFAULT_MAX_WORKING_BYTES, total=True):
    '''Per-group union ground cover of circles (fraction of the domain area): clipped at a nonperiodic edge, WRAPPED across a
    periodic axis of `domain.periodic` (torus union, see the module docstring; the centre stays the single stored position).

    Raster midpoints are origin + (i + 0.5) * dx / s; a midpoint is covered when strictly inside a circle (on a periodic axis:
    strictly inside by its minimum-image distance).
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
    wrapped = domain.any_periodic              # False: the legacy clipped path below runs unchanged
    per = domain.periodic
    Lx, Ly = float(domain.width_m), float(domain.height_m)
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
            if wrapped:
                _paint_wrapped(raster, px, py, rp, x0, y0, hx, hy, NX, NY, Lx, Ly, per, rows_per_chunk)
                continue
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
    out = {'method': COVER_METHOD, 'subsamples': s, 'raster_shape': [NY, NX], 'resolution_m': [hx, hy],
           'raster_cells': adm['raster_cells'], 'raster_bytes': adm['raster_bytes'], 'working_bytes': adm['working_bytes'],
           'rows_per_chunk': rows_per_chunk, 'index_bytes': int(8 * n + g.nbytes), 'admission': adm,
           'domain_area_m2': A, 'n_circles': n, 'groups': groups,
           'total_union_fraction': (int(np.count_nonzero(union_all)) / adm['raster_cells']) if union_all is not None else None,
           'sum_of_group_union_fractions': float(sum(gr['union_fraction'] for gr in groups)),
           'definition': COVER_DEFINITION_PERIODIC if wrapped else COVER_DEFINITION,
           'boundary_note': boundary_note(domain),
           'error_basis': 'midpoint sampling at resolution dx/s: each circle boundary is resolved to about one '
                          'raster cell, relative error per circle ~ resolution / radius; sampled, not analytic'}
    if wrapped:                                   # only a periodic measurement adds this key (the legacy key set is unchanged)
        out['periodic'] = [bool(per[0]), bool(per[1])]
    return out


__all__ = ['COVER_METHOD', 'DEFAULT_MAX_COVER_CELLS', 'DEFAULT_MAX_COVER_BYTES', 'DEFAULT_MAX_WORKING_BYTES',
           'CANOPY_FIELDS', 'MAPLE_FIELDS', 'CANOPY_VIEW_SCHEMA', 'CANOPY_VIEW_SCHEMA_VERSION', 'MAPLE_VIEW_SCHEMA',
           'MAPLE_VIEW_SCHEMA_VERSION', 'CANOPY_FIELD_UNITS', 'CANOPY_SIDECAR_UNITS', 'HEIGHT_MODES',
           'UNSUPPORTED_POLICIES', 'BOUNDARY_NOTE', 'GEOMETRY_UNITS', 'ResourceBoundError', 'derive_geometry',
           'canopy_view', 'maple_vegetation_view', 'cover_admission', 'measure_cover']
