'''leaf.runtime: narrow bridge between the canonical LEAF population and the existing growth drivers.

Two directions, both by stable plant ID (never by array position):

capture_snapshot(population, sim)
    Copies the CURRENT per-plant aboveground biomass B (and the STEP seasonal mass E when the driver runs in
    stepwat mode) out of a benchmarks.daily_water_growth.Simulation or a leaf.coupling.CoupledSimulation into
    the canonical population, matching sim ids (host_inputs['ids']) to population IDs. Species (root profile
    name), establishment anchors and ages of the population are preserved; the driver never owns them.
    Missing live plants, unknown or duplicate sim IDs, non-live population plants present in the driver and
    mismatched profile assignments are refused before any change. Call it only when a snapshot is requested
    (the writer cadence), not daily; the driver is never imported here (duck typing, host copies only).

legacy_inputs_from_state(population, catalog, ...)
    Builds the `inputs` dict the existing driver accepts (x, y, canopy_radius, root_radius, ids,
    root_profile_id) for the LIVE plants. The legacy driver RE-COMPUTES initial biomass from canopy area
    (B = pi r^2 * density: fixture 1.0 kg m^-2, or the STEP trait canopy_biomass_density_kg_m2) and uses one
    root_multiplier and strictly positive layer fractions. A LEAF power allometry generally disagrees with
    that area law, so by default (biomass_policy='refuse') a mismatch beyond rtol is an error; passing
    biomass_policy='accept_driver_recompute' documents that the driver's own initial biomass will differ.
    The driver's science is not changed here. This is VEGETATION STATE REPLAY/handoff; a full coupled restart
    (pending weekly uptake, native soil water, STEP calendar/temperature integral, RNG) is NOT provided.
    Static MAPLE initialization (leaf.canopy.maple_vegetation_view) is unaffected by this legacy law.
'''
from __future__ import annotations

import math

import numpy as np

from leaf import state as _state
from leaf import traits as _traits

LEGACY_FIXTURE_BIOMASS_DENSITY_KG_M2 = 1.0  # mirrors benchmarks.daily_water_growth.FIXTURE['biomass_density_kg_m2']
BIOMASS_POLICIES = ('refuse', 'accept_driver_recompute')


class BridgeError(ValueError):
    pass


def _to_host(a):
    if a is None:
        return None
    if not isinstance(a, np.ndarray) and hasattr(a, 'get'):
        a = a.get()  # CuPy device array -> host (no cupy import)
    return np.asarray(a)


def _vegetation(sim):
    veg = getattr(sim, 'vegetation', None)  # CoupledSimulation composes a vegetation Simulation
    return sim if veg is None else veg


def read_driver_arrays(sim):
    '''Host copies of the driver's ids, B, E (zeros without STEP), profile ids and profile names.'''
    veg = _vegetation(sim)
    for attr in ('host_inputs', 'B', 'host_profile_ids', 'profile_names'):
        if not hasattr(veg, attr):
            raise BridgeError(f'object has no {attr!r}; expected a daily_water_growth.Simulation or CoupledSimulation')
    ids = np.asarray(veg.host_inputs['ids'])
    if ids.dtype.kind not in 'iu' or ids.ndim != 1:
        raise BridgeError('driver ids must be a 1-D integer array')
    ids = ids.astype(np.int64)
    B = _to_host(veg.B).astype(np.float64, copy=True)
    sw = getattr(veg, 'stepwat', None)
    E = _to_host(sw.E).astype(np.float64, copy=True) if sw is not None else np.zeros(B.shape[0], dtype=np.float64)
    pid = np.asarray(veg.host_profile_ids).astype(np.int64)
    names = [str(s) for s in veg.profile_names]
    n = ids.shape[0]
    if B.shape != (n,) or E.shape != (n,) or pid.shape != (n,):
        raise BridgeError('driver arrays ids/B/E/profile ids have inconsistent lengths')
    if n and np.unique(ids).shape[0] != n:
        raise BridgeError('driver ids are not unique')
    return {'ids': ids, 'B': B, 'E': E, 'profile_ids': pid, 'profile_names': names, 'stepwat': sw is not None,
            'day': int(getattr(veg, 'day', 0))}


def capture_snapshot(population, sim, *, in_place=True):
    '''Update the canonical population's B/E from the driver by ID (see module docstring). Returns the
    updated population (the same object when in_place, else a copy); time/age are untouched.'''
    if not isinstance(population, _state.PlantState):
        raise BridgeError('population must be a leaf.state.PlantState')
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
    if pid.size and (int(pid.min()) < 0 or int(pid.max()) >= len(names)):
        raise BridgeError('driver profile ids outside the profile name table')
    pop_species = population.species_of(idx)
    drv_species = np.array(names, dtype=object)[pid] if pid.size else np.zeros(0, dtype=object)
    mism = pop_species != drv_species
    if mism.any():
        raise BridgeError(f'{int(mism.sum())} plant(s) have a driver root profile differing from the population species, '
                          f'e.g. id {int(ids[np.flatnonzero(mism)[0]])}: population {pop_species[mism][0]!r}, '
                          f'driver {drv_species[mism][0]!r}')
    if not np.all(np.isfinite(B)) or (B.size and float(B.min()) <= 0.0):
        raise BridgeError('driver biomass B must be finite and > 0')
    if not np.all(np.isfinite(E)) or (E.size and (float(E.min()) < 0.0 or float((E - B).max()) > 0.0)):
        raise BridgeError('driver seasonal mass E must be finite with 0 <= E <= B')
    target = population if in_place else population.copy()
    if ids.size:
        target.update_plants(ids, agb_dry_kg=B, seasonal_agb_dry_kg=E)
    target.provenance['bridge'] = {'captured_from': 'CoupledSimulation' if hasattr(sim, 'vegetation') else 'Simulation',
                                   'driver_day': d['day'], 'stepwat': d['stepwat'], 'n_captured': int(ids.size),
                                   'seasonal_source': 'stepwat.E' if d['stepwat'] else 'zeros (legacy growth has no E)'}
    return target


def legacy_inputs_from_state(population, catalog, *, profile_names=None, biomass_policy='refuse',
                             stepwat_traits=None, driver_biomass_density_kg_m2=LEGACY_FIXTURE_BIOMASS_DENSITY_KG_M2,
                             rtol=1e-9):
    '''Legacy driver inputs for the live plants (see module docstring). Returns a dict with `inputs`,
    `root_layer_fractions`, `profile_names`, `case_fragment`, `excluded_ids`, `biomass_check` and `notes`.'''
    if not isinstance(population, _state.PlantState):
        raise BridgeError('population must be a leaf.state.PlantState')
    if catalog.hash != population.catalog_hash:
        raise BridgeError('catalog hash differs from the population catalog hash')
    if biomass_policy not in BIOMASS_POLICIES:
        raise BridgeError(f'biomass_policy must be one of {BIOMASS_POLICIES}')
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
    names = list(catalog.species_ids) if profile_names is None else [str(s) for s in profile_names]
    if len(set(names)) != len(names):
        raise BridgeError('profile_names must be unique')
    used = sorted(set(int(s) for s in np.unique(sp)))
    for s in used:
        if catalog.species_ids[s] not in names:
            raise BridgeError(f'species {catalog.species_ids[s]!r} is not in profile_names {names}')
    mult = catalog.root_multiplier_array()
    mults = sorted(set(float(mult[s]) for s in used))
    if len(mults) != 1:
        raise BridgeError(f'the legacy driver has one case root_multiplier; live species use {mults}')
    canopy_radius = 0.5 * catalog.diameter_from_biomass(sp, B)
    root_radius = canopy_radius * mults[0]
    table = {}
    for name in names:
        if catalog.has(name):
            fr = catalog.root_layer_fractions(name)
            if any(f <= 0.0 for f in fr):
                raise BridgeError(f'species {name!r} has a zero layer fraction; the legacy driver requires strictly positive fractions')
            table[name] = fr
    pos = {name: i for i, name in enumerate(names)}
    root_profile_id = np.array([pos[catalog.species_ids[s]] for s in sp], dtype=np.int64)
    if stepwat_traits is not None:
        dens = np.empty(idx.size, dtype=np.float64)
        for s in used:
            sid = catalog.species_ids[s]
            row = stepwat_traits.get(sid) if isinstance(stepwat_traits, dict) else None
            if not isinstance(row, dict) or not _traits.is_real(row.get('canopy_biomass_density_kg_m2')):
                raise BridgeError(f'stepwat_traits[{sid!r}].canopy_biomass_density_kg_m2 is required for the biomass check')
            dens[sp == s] = float(row['canopy_biomass_density_kg_m2'])
        law = 'STEP: B = pi r^2 * canopy_biomass_density_kg_m2[profile]'
    else:
        dens = np.full(idx.size, _traits.positive(driver_biomass_density_kg_m2, 'driver_biomass_density_kg_m2'))
        law = f'legacy fixture: B = pi r^2 * {LEGACY_FIXTURE_BIOMASS_DENSITY_KG_M2} kg m^-2'
    B_driver = math.pi * canopy_radius * canopy_radius * dens
    rel = np.abs(B_driver - B) / B
    max_rel = float(rel.max())
    check = {'driver_law': law, 'max_relative_deviation': max_rel, 'rtol': float(rtol), 'consistent': max_rel <= rtol,
             'policy': biomass_policy}
    if max_rel > rtol and biomass_policy == 'refuse':
        raise BridgeError(f'the legacy driver would recompute initial biomass by {law} and deviate from the LEAF '
                          f'allometric biomass by up to {max_rel:.3e} (relative); refusing to feed it silently. '
                          "Pass biomass_policy='accept_driver_recompute' to acknowledge the different initial state.")
    inputs = {'x': np.ascontiguousarray(population.x_m[idx], dtype=np.float64),
              'y': np.ascontiguousarray(population.y_m[idx], dtype=np.float64),
              'canopy_radius': np.ascontiguousarray(canopy_radius, dtype=np.float64),
              'root_radius': np.ascontiguousarray(root_radius, dtype=np.float64),
              'ids': np.ascontiguousarray(population.ids[idx], dtype=np.int64),
              'root_profile_id': root_profile_id}
    return {'inputs': inputs, 'root_layer_fractions': table, 'profile_names': names,
            'case_fragment': {'nx': dom.nx, 'ny': dom.ny, 'dx': dom.dx_m, 'root_multiplier': mults[0],
                              'layers': catalog.root_layers},
            'excluded_ids': excluded.tolist(), 'biomass_check': check,
            'notes': ['vegetation state replay/handoff only: pending weekly uptake, soil water, STEP period clock and '
                      'RNG state are not restored (no full coupled restart)',
                      'the driver owns B afterwards; capture_snapshot() brings it back by ID',
                      'dead_standing/removed plants are excluded (the driver requires positive live biomass)']}


__all__ = ['LEGACY_FIXTURE_BIOMASS_DENSITY_KG_M2', 'BIOMASS_POLICIES', 'BridgeError', 'read_driver_arrays',
           'capture_snapshot', 'legacy_inputs_from_state']
