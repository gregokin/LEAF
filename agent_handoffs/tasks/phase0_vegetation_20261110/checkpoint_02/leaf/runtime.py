'''leaf.runtime: narrow bridge between the canonical LEAF population and the existing growth drivers.

While a legacy driver (benchmarks.daily_water_growth.Simulation or leaf.coupling.CoupledSimulation) is
attached, ITS per-plant B (and STEP seasonal E) are the numerical authority. LEAF keeps the canonical
record (identity, species, anchors, ages, clock) and creates the record VIEW only at requested output by
calling capture_snapshot. There are never two independently edited copies, and Phase 0 event proposals do
NOT reach a running legacy driver (no such coupling exists yet).

capture_snapshot(population, sim, *, seasonal_policy='refuse')
    Copies the CURRENT per-plant B (and STEP E) out of the driver into the population BY STABLE ID. Species
    (root profile name), establishment anchors, ages and the clock are preserved; the driver never owns
    them. Missing live plants, unknown/duplicate/non-integer driver IDs (incl. unsigned overflow), non-live
    population plants present in the driver and mismatched profile assignments are refused before any
    change. A legacy NON-STEP driver has no E: if the population holds E > 0 for captured plants the
    default refuses (so pre-existing E is never erased silently); 'keep' keeps E (must stay <= B), 'zero'
    resets it explicitly.

legacy_inputs_from_state(population, catalog, *, biomass_policy='refuse', seasonal_policy='refuse', ...)
    Builds the `inputs` the existing driver accepts (x, y, canopy_radius, root_radius, ids,
    root_profile_id) for the LIVE plants. The unmodified driver RE-COMPUTES initial biomass from canopy
    area, B = pi r^2 * rho with rho = 1.0 kg m^-2 (its fixture) or the STEP trait
    canopy_biomass_density_kg_m2, and keeps that square-root law during growth. LEAF's species law is
    consistent with it only if diameter = a_d * B**0.5 with a_d = 2 / sqrt(pi * rho); an arbitrary exponent
    that happens to match B at one size diverges after growth, so the SPECIES LAW is checked, not one
    initial B. Default 'refuse'; 'accept_driver_recompute' documents an approximation (recorded, labelled).
    Seasonal E: the fresh driver starts E = 0; nonzero population E is refused unless
    seasonal_policy='restore' (the caller must construct in STEP mode and call restore_seasonal_mass) or
    'acknowledge_reset' (documented loss, population untouched). One root_multiplier, strictly positive
    layer fractions, a [0, W]x[0, H] square-cell grid: legacy constraints, checked here.
    This is vegetation STATE REPLAY/handoff; a full coupled restart (pending weekly uptake, native soil
    water, STEP calendar/temperature integral, RNG) is NOT provided. The driver's science is unchanged.
    Static MAPLE initialization (leaf.canopy.maple_vegetation_view) is unaffected by that legacy law.
'''
from __future__ import annotations

import math

import numpy as np

from leaf import state as _state
from leaf import traits as _traits

LEGACY_FIXTURE_BIOMASS_DENSITY_KG_M2 = 1.0  # benchmarks.daily_water_growth.FIXTURE['biomass_density_kg_m2'] (fixed)
LEGACY_DIAMETER_EXPONENT = 0.5
BIOMASS_POLICIES = ('refuse', 'accept_driver_recompute')
SEASONAL_INPUT_POLICIES = ('refuse', 'restore', 'acknowledge_reset')
SEASONAL_CAPTURE_POLICIES = ('refuse', 'keep', 'zero')


class BridgeError(ValueError):
    pass


def legacy_diameter_coefficient(density_kg_m2):
    '''a_d such that diameter = a_d * B**0.5 is the legacy area law B = pi r^2 * density.'''
    return 2.0 / math.sqrt(math.pi * float(density_kg_m2))


def _to_host(a):
    if a is None:
        return None
    if not isinstance(a, np.ndarray) and hasattr(a, 'get'):
        a = a.get()  # CuPy device array -> host (no cupy import)
    return np.asarray(a)


def _vegetation(sim):
    veg = getattr(sim, 'vegetation', None)  # CoupledSimulation composes a vegetation Simulation
    return sim if veg is None else veg


def _int_array(a, name, lo=None, hi=None):
    a = np.asarray(a)
    if a.dtype.kind not in 'iu' or a.ndim != 1:
        raise BridgeError(f'driver {name} must be a 1-D integer array (not bool/float), got dtype {a.dtype}, ndim {a.ndim}')
    if a.size and (int(a.min()) < _state.INT64_MIN or int(a.max()) > _state.INT64_MAX):
        raise BridgeError(f'driver {name} values exceed the int64 range (unsigned overflow)')
    out = a.astype(np.int64)
    if out.size and lo is not None and int(out.min()) < lo:
        raise BridgeError(f'driver {name} values must be >= {lo}')
    if out.size and hi is not None and int(out.max()) >= hi:
        raise BridgeError(f'driver {name} values must be < {hi}')
    return out


def _float_array(a, name, n):
    a = _to_host(a)
    if a.dtype.kind not in 'fiu' or a.ndim != 1 or a.shape[0] != n:
        raise BridgeError(f'driver {name} must be a real 1-D array of length {n}')
    return a.astype(np.float64, copy=True)


def read_driver_arrays(sim):
    '''Host copies of the driver's ids, B, E (zeros without STEP), profile ids and profile names, with exact
    integer type/range checks.'''
    veg = _vegetation(sim)
    for attr in ('host_inputs', 'B', 'host_profile_ids', 'profile_names'):
        if not hasattr(veg, attr):
            raise BridgeError(f'object has no {attr!r}; expected a daily_water_growth.Simulation or CoupledSimulation')
    ids = _int_array(veg.host_inputs['ids'], 'ids')
    n = int(ids.shape[0])
    if n and np.unique(ids).shape[0] != n:
        raise BridgeError('driver ids are not unique')
    names = [str(s) for s in veg.profile_names]
    if len(set(names)) != len(names):
        raise BridgeError('driver profile names are not unique')
    pid = _int_array(veg.host_profile_ids, 'root_profile_ids', 0, len(names))
    if pid.shape[0] != n:
        raise BridgeError('driver profile ids length differs from ids')
    B = _float_array(veg.B, 'B', n)
    sw = getattr(veg, 'stepwat', None)
    E = _float_array(sw.E, 'stepwat.E', n) if sw is not None else np.zeros(n, dtype=np.float64)
    if not np.all(np.isfinite(B)) or (n and float(B.min()) <= 0.0):
        raise BridgeError('driver biomass B must be finite and > 0')
    if not np.all(np.isfinite(E)) or (n and (float(E.min()) < 0.0 or float((E - B).max()) > 0.0)):
        raise BridgeError('driver seasonal mass E must be finite with 0 <= E <= B')
    return {'ids': ids, 'B': B, 'E': E, 'profile_ids': pid, 'profile_names': names, 'stepwat': sw is not None,
            'day': int(getattr(veg, 'day', 0))}


def capture_snapshot(population, sim, *, in_place=True, seasonal_policy='refuse'):
    '''Update the canonical population's B/E from the driver by ID (see module docstring). Returns the
    updated population (the same object when in_place, else a copy); clock and ages are untouched.'''
    if not isinstance(population, _state.PlantState):
        raise BridgeError('population must be a leaf.state.PlantState')
    if seasonal_policy not in SEASONAL_CAPTURE_POLICIES:
        raise BridgeError(f'seasonal_policy must be one of {SEASONAL_CAPTURE_POLICIES}')
    population.validate()
    d = read_driver_arrays(sim)
    ids, B, E, pid, names = d['ids'], d['B'], d['E'], d['profile_ids'], d['profile_names']
    try:
        idx = population.indices_of(ids)
    except _state.StateError as exc:
        raise BridgeError(f'driver ids do not resolve in the population: {exc}') from exc
    live_idx = np.flatnonzero(population.live_mask())
    missing = np.setdiff1d(live_idx, idx, assume_unique=True)
    if missing.size:
        raise BridgeError(f'{missing.size} live population plant(s) absent from the driver, e.g. ids '
                          f'{population.ids[missing][:10].tolist()}')
    not_live = population.status[idx] != _state.LIVE
    if not_live.any():
        raise BridgeError(f'driver holds {int(not_live.sum())} plant(s) that are not live in the population, e.g. ids '
                          f'{ids[not_live][:10].tolist()}')
    pop_species = population.species_of(idx)
    drv_species = np.array(names, dtype=object)[pid] if pid.size else np.zeros(0, dtype=object)
    mism = pop_species != drv_species
    if mism.any():
        raise BridgeError(f'{int(mism.sum())} plant(s) have a driver root profile differing from the population species, '
                          f'e.g. id {int(ids[np.flatnonzero(mism)[0]])}: population {pop_species[mism][0]!r}, '
                          f'driver {drv_species[mism][0]!r}')
    seasonal_note = 'stepwat.E (driver authority)'
    if not d['stepwat']:
        E_pop = population.seasonal_agb_dry_kg[idx]
        if E_pop.size and float(E_pop.max()) > 0.0:
            if seasonal_policy == 'refuse':
                raise BridgeError(f'the population holds nonzero seasonal mass for {int((E_pop > 0).sum())} captured plant(s) '
                                  'but the legacy non-STEP driver has no seasonal pool; refusing to erase it silently '
                                  "(seasonal_policy='keep' retains E <= B, 'zero' resets it explicitly)")
            if seasonal_policy == 'keep':
                E = E_pop.copy()
                seasonal_note = 'kept from the population (legacy driver has no E); must satisfy E <= new B'
            else:
                seasonal_note = 'explicitly reset to 0 (legacy driver has no E)'
        else:
            seasonal_note = 'zeros (legacy driver has no E; population E was already 0)'
    target = population if in_place else population.copy()
    if ids.size:
        try:
            target.update_plants(ids, agb_dry_kg=B, seasonal_agb_dry_kg=E)
        except _state.StateError as exc:
            raise BridgeError(f'captured values rejected ({exc}); population unchanged') from exc
    target.provenance['bridge'] = {'captured_from': 'CoupledSimulation' if hasattr(sim, 'vegetation') else 'Simulation',
                                   'driver_day': d['day'], 'stepwat': d['stepwat'], 'n_captured': int(ids.size),
                                   'seasonal_source': seasonal_note,
                                   'clock_note': 'population time_days/ages are caller-owned (advance_time once per '
                                                 'elapsed interval); not changed by capture'}
    return target


def legacy_inputs_from_state(population, catalog, *, profile_names=None, biomass_policy='refuse',
                             seasonal_policy='refuse', stepwat_traits=None, rtol=1e-9):
    '''Legacy driver inputs for the live plants (see module docstring). Returns a dict with `inputs`,
    `root_layer_fractions`, `profile_names`, `case_fragment`, `excluded_ids`, `biomass_check`,
    `seasonal_agb_dry_kg` (by id, for restore_seasonal_mass) and `notes`.'''
    if not isinstance(population, _state.PlantState):
        raise BridgeError('population must be a leaf.state.PlantState')
    if not isinstance(catalog, _traits.SpeciesCatalog):
        raise BridgeError('catalog must be a leaf.traits.SpeciesCatalog')
    if biomass_policy not in BIOMASS_POLICIES:
        raise BridgeError(f'biomass_policy must be one of {BIOMASS_POLICIES}')
    if seasonal_policy not in SEASONAL_INPUT_POLICIES:
        raise BridgeError(f'seasonal_policy must be one of {SEASONAL_INPUT_POLICIES}')
    try:
        rtol = _traits.nonnegative(rtol, 'rtol')
    except _traits.CatalogError as exc:
        raise BridgeError(str(exc)) from exc
    population.validate()
    if catalog.hash != population.catalog_hash:
        raise BridgeError('catalog hash differs from the population catalog hash')
    try:
        catalog.check_species_table(population.species_names)
    except _traits.CatalogError as exc:
        raise BridgeError(str(exc)) from exc
    dom = population.domain
    if dom.origin_x_m != 0.0 or dom.origin_y_m != 0.0 or not dom.square_cells:
        raise BridgeError('the legacy driver grid is [0, nx*dx] x [0, ny*dx] with square cells and origin 0; '
                          f'population domain is {dom!r}')
    live = population.live_mask()
    idx = np.flatnonzero(live)
    if idx.size == 0:
        raise BridgeError('the legacy driver needs a positive live population')
    excluded = population.ids[~live]
    sp = population.species_index[idx].astype(np.int64)
    B = population.agb_dry_kg[idx]
    E = population.seasonal_agb_dry_kg[idx]
    names = list(catalog.species_ids) if profile_names is None else [str(s) for s in profile_names]
    if len(set(names)) != len(names) or not names:
        raise BridgeError('profile_names must be a non-empty list of unique names')
    unknown = [nm for nm in names if not catalog.has(nm)]
    if unknown:
        raise BridgeError(f'requested profile names {unknown} are not catalog species {list(catalog.species_ids)}')
    used = sorted(set(int(s) for s in np.unique(sp)))
    for s in used:
        if catalog.species_ids[s] not in names:
            raise BridgeError(f'live species {catalog.species_ids[s]!r} is not in profile_names {names}')
    mult = catalog.root_multiplier_array()
    mults = sorted(set(float(mult[s]) for s in used))
    if len(mults) != 1:
        raise BridgeError(f'the legacy driver has one case root_multiplier; live species use {mults}')
    # seasonal mass: the fresh driver starts E = 0
    e_pos = int((E > 0.0).sum())
    if e_pos and seasonal_policy == 'refuse':
        raise BridgeError(f'{e_pos} live plant(s) carry nonzero seasonal mass E, which a fresh legacy driver resets to 0; '
                          "pass seasonal_policy='restore' (STEP mode + restore_seasonal_mass after construction) or "
                          "'acknowledge_reset' (documented loss)")
    # species law check against the driver's area law
    al = catalog.allometry_arrays()
    law = {}
    for s in used:
        sid = catalog.species_ids[s]
        if stepwat_traits is not None:
            row = stepwat_traits.get(sid) if isinstance(stepwat_traits, dict) else None
            if not isinstance(row, dict):
                raise BridgeError(f'stepwat_traits[{sid!r}] row is required')
            try:
                dens = _traits.positive(row.get('canopy_biomass_density_kg_m2'), f'stepwat_traits[{sid!r}].canopy_biomass_density_kg_m2')
            except _traits.CatalogError as exc:
                raise BridgeError(str(exc)) from exc
            law_label = 'STEP: B = pi r^2 * canopy_biomass_density_kg_m2[profile]'
        else:
            dens = LEGACY_FIXTURE_BIOMASS_DENSITY_KG_M2
            law_label = f'legacy fixture: B = pi r^2 * {LEGACY_FIXTURE_BIOMASS_DENSITY_KG_M2} kg m^-2'
        a_req = legacy_diameter_coefficient(dens)
        a_d, b_d = float(al['a_d'][s]), float(al['b_d'][s])
        ok = abs(b_d - LEGACY_DIAMETER_EXPONENT) <= rtol and abs(a_d - a_req) <= rtol * a_req
        law[sid] = {'driver_law': law_label, 'density_kg_m2': dens, 'required_a_d': a_req,
                    'required_b_d': LEGACY_DIAMETER_EXPONENT, 'catalog_a_d': a_d, 'catalog_b_d': b_d, 'consistent': ok}
    canopy_radius = 0.5 * catalog.diameter_from_biomass(sp, B)
    dens_arr = np.array([law[catalog.species_ids[int(s)]]['density_kg_m2'] for s in sp], dtype=np.float64)
    B_driver = math.pi * canopy_radius * canopy_radius * dens_arr
    max_rel = float(np.max(np.abs(B_driver - B) / B))
    consistent = all(v['consistent'] for v in law.values())
    check = {'species_law': law, 'law_consistent': consistent, 'initial_biomass_max_relative_deviation': max_rel,
             'rtol': rtol, 'policy': biomass_policy,
             'note': 'consistency is judged on the SPECIES LAW (b_d = 0.5, a_d = 2/sqrt(pi*rho)); the initial deviation '
                     'alone cannot prove consistency because an arbitrary exponent may match B at one size only'}
    if not consistent and biomass_policy == 'refuse':
        bad = [k for k, v in law.items() if not v['consistent']]
        raise BridgeError(f'species {bad} do not follow the legacy square-root area law (diameter = 2/sqrt(pi*rho) * B^0.5); '
                          f'the unmodified driver would recompute initial biomass (max relative deviation {max_rel:.3e}) and '
                          "diverge during growth. Refusing. Pass biomass_policy='accept_driver_recompute' to acknowledge "
                          'the documented approximation explicitly.')
    root_radius = canopy_radius * mults[0]
    table = {}
    for name in names:
        fr = catalog.root_layer_fractions(name)
        if any(f <= 0.0 for f in fr):
            raise BridgeError(f'species {name!r} has a zero layer fraction; the legacy driver requires strictly positive fractions')
        table[name] = fr
    pos = {name: i for i, name in enumerate(names)}
    root_profile_id = np.array([pos[catalog.species_ids[int(s)]] for s in sp], dtype=np.int64)
    inputs = {'x': np.array(population.x_m[idx], dtype=np.float64, copy=True),
              'y': np.array(population.y_m[idx], dtype=np.float64, copy=True),
              'canopy_radius': np.ascontiguousarray(canopy_radius, dtype=np.float64),
              'root_radius': np.ascontiguousarray(root_radius, dtype=np.float64),
              'ids': np.array(population.ids[idx], dtype=np.int64, copy=True),
              'root_profile_id': root_profile_id}
    return {'inputs': inputs, 'root_layer_fractions': table, 'profile_names': names,
            'case_fragment': {'nx': dom.nx, 'ny': dom.ny, 'dx': dom.dx_m, 'root_multiplier': mults[0],
                              'layers': catalog.root_layers},
            'excluded_ids': excluded.tolist(), 'biomass_check': check,
            'seasonal_agb_dry_kg': {'ids': inputs['ids'].copy(), 'values': np.array(E, copy=True), 'policy': seasonal_policy,
                                    'n_nonzero': e_pos,
                                    'note': ('restore with restore_seasonal_mass after constructing the STEP driver' if seasonal_policy == 'restore'
                                             else 'fresh driver starts E = 0' + (' (acknowledged loss of nonzero E)' if e_pos else ''))},
            'notes': ['vegetation state replay/handoff only: pending weekly uptake, soil water, STEP period clock and '
                      'RNG state are not restored (no full coupled restart)',
                      'the driver owns B/E while attached; capture_snapshot() creates the record view by ID at requested output',
                      'clock and ages are caller-owned: advance_time once per elapsed interval; never changed by the bridge',
                      'dead_standing/removed plants are excluded (the driver requires positive live biomass)']}


def restore_seasonal_mass(population, sim):
    '''Write the population's seasonal mass E into a freshly constructed STEP-mode driver BY ID (the explicit
    completion of seasonal_policy="restore"). Requires the driver to be in STEP mode with 0 <= E <= driver B.'''
    d = read_driver_arrays(sim)
    if not d['stepwat']:
        raise BridgeError('restore_seasonal_mass requires a STEP-mode driver (stepwat_traits); the legacy driver has no E')
    veg = _vegetation(sim)
    try:
        idx = population.indices_of(d['ids'])
    except _state.StateError as exc:
        raise BridgeError(f'driver ids do not resolve in the population: {exc}') from exc
    E = np.array(population.seasonal_agb_dry_kg[idx], dtype=np.float64, copy=True)
    if E.size and (float(E.min()) < 0.0 or float((E - d['B']).max()) > 0.0):
        raise BridgeError('population seasonal mass exceeds the driver biomass for at least one plant; nothing restored')
    target = veg.stepwat.E
    if isinstance(target, np.ndarray):
        target[...] = E
    elif hasattr(target, 'set'):
        target.set(E)
    else:
        raise BridgeError('unsupported driver seasonal array type')
    return int(E.shape[0])


__all__ = ['LEGACY_FIXTURE_BIOMASS_DENSITY_KG_M2', 'LEGACY_DIAMETER_EXPONENT', 'BIOMASS_POLICIES',
           'SEASONAL_INPUT_POLICIES', 'SEASONAL_CAPTURE_POLICIES', 'BridgeError', 'legacy_diameter_coefficient',
           'read_driver_arrays', 'capture_snapshot', 'legacy_inputs_from_state', 'restore_seasonal_mass']
