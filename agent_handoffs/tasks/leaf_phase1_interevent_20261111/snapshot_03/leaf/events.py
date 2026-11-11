'''leaf.events: versioned, stable-ID keyed surface/water event diagnostics and the ecological response socket.

A SurfaceEventDiagnostics record describes what MAPLE (wind, shared surface) or SYRUP (water) observed at
individual plants during one explicit interval [time_start_days, time_end_days]:
  erosion_depth_m      >= 0  physical surface LOWERING at the plant (metres)
  deposition_depth_m   >= 0  physical surface RAISING / burial at the plant (metres)
  abrasion_dose        >= 0  exposure dose in the declared `abrasion_unit` (e.g. kg m^-1 of saltation flux
                             passing the plant); it is NOT converted into injury here
  transport_flux       >= 0  optional transport flux in the declared `flux_unit`; distinct from depth changes
  net_surface_delta_m  = deposition_depth_m - erosion_depth_m (signed; + raised, - lowered)
Depth lowering/raising and flux are distinct quantities; LEAF invents no conversion between them.

Records are IMMUTABLE: every array is an owned read-only copy and attributes cannot be rebound, so a
handler can neither redirect the IDs nor smuggle negative depths after validation.

apply_event_diagnostics(state, event, handler=None) resolves the IDs ONCE (unknown or duplicate IDs, bad
times/units/values fail before any mutation) and returns diagnostics. The DEFAULT is no ecological
response. A species-law `handler(view, event, indices) -> proposals` receives a ReadOnlyView (owned
read-only copies; the immutable Domain) and may propose agb_dry_kg / seasonal_agb_dry_kg / status for the
affected plants; proposals are applied to the IDs resolved before the handler ran, validated on a
candidate copy and committed all-or-nothing. A raising handler changes nothing. No threshold mortality is
ever applied here, no surface remapping happens, and MAPLE/SYRUP are never imported or modified.
'''
from __future__ import annotations

import numpy as np

from leaf import state as _state
from leaf import traits as _traits

EVENT_SCHEMA = 'leaf.surface_event_diagnostics'
EVENT_SCHEMA_VERSION = 1
SOURCES = ('maple_wind', 'syrup_water', 'other')
DEPTH_UNIT = 'm'
TIME_UNIT = 'days'
PROPOSAL_KEYS = ('agb_dry_kg', 'seasonal_agb_dry_kg', 'status')


class EventError(ValueError):
    pass


def _frozen_array(a):
    a.setflags(write=False)
    return a


def _depth_array(v, name, n):
    if v is None:
        return _frozen_array(np.zeros(n, dtype=np.float64))
    try:
        a = _state._coerce_float(v, name, n)   # owned copy
    except _state.StateError as exc:
        raise EventError(str(exc)) from exc
    if not np.all(np.isfinite(a)) or (n and float(a.min()) < 0.0):
        raise EventError(f'{name} must be finite and >= 0 ({DEPTH_UNIT})')
    return _frozen_array(a)


class SurfaceEventDiagnostics:
    '''Validated, immutable event record (arrays are owned read-only copies).'''

    def __init__(self, plant_id, time_start_days, time_end_days, source, *, erosion_depth_m=None,
                 deposition_depth_m=None, abrasion_dose=None, abrasion_unit=None, transport_flux=None,
                 flux_unit=None, event_id=None, notes=None):
        s = object.__setattr__
        try:
            ids = _state._coerce_int(plant_id, 'plant_id', None, np.int64)
        except _state.StateError as exc:
            raise EventError(str(exc)) from exc
        n = int(ids.shape[0])
        if n and np.unique(ids).shape[0] != n:
            u, c = np.unique(ids, return_counts=True)
            raise EventError(f'duplicate (ambiguous) plant IDs in event: {u[c > 1][:10].tolist()}')
        try:
            t0 = _traits.real(time_start_days, 'time_start_days')
            t1 = _traits.real(time_end_days, 'time_end_days')
        except _traits.CatalogError as exc:
            raise EventError(str(exc)) from exc
        if t1 < t0:
            raise EventError('time_end_days must be >= time_start_days')
        if source not in SOURCES:
            raise EventError(f'source must be one of {SOURCES}, got {source!r}')
        s(self, 'plant_id', _frozen_array(ids))
        s(self, 'time_start_days', t0)
        s(self, 'time_end_days', t1)
        s(self, 'source', source)
        s(self, 'erosion_depth_m', _depth_array(erosion_depth_m, 'erosion_depth_m', n))
        s(self, 'deposition_depth_m', _depth_array(deposition_depth_m, 'deposition_depth_m', n))
        s(self, 'abrasion_dose', None)
        s(self, 'abrasion_unit', None)
        if abrasion_dose is not None:
            if not isinstance(abrasion_unit, str) or not abrasion_unit:
                raise EventError('abrasion_dose requires an explicit non-empty abrasion_unit string')
            s(self, 'abrasion_dose', _depth_array(abrasion_dose, 'abrasion_dose', n))
            s(self, 'abrasion_unit', abrasion_unit)
        elif abrasion_unit is not None:
            raise EventError('abrasion_unit given without abrasion_dose')
        s(self, 'transport_flux', None)
        s(self, 'flux_unit', None)
        if transport_flux is not None:
            if not isinstance(flux_unit, str) or not flux_unit:
                raise EventError('transport_flux requires an explicit non-empty flux_unit string')
            s(self, 'transport_flux', _depth_array(transport_flux, 'transport_flux', n))
            s(self, 'flux_unit', flux_unit)
        elif flux_unit is not None:
            raise EventError('flux_unit given without transport_flux')
        s(self, 'event_id', None if event_id is None else str(event_id))
        s(self, 'notes', None if notes is None else str(notes))
        s(self, '_frozen', True)

    def __setattr__(self, name, value):
        raise AttributeError('SurfaceEventDiagnostics is immutable')

    def __delattr__(self, name):
        raise AttributeError('SurfaceEventDiagnostics is immutable')

    @property
    def n(self):
        return int(self.plant_id.shape[0])

    @property
    def net_surface_delta_m(self):
        return self.deposition_depth_m - self.erosion_depth_m

    @property
    def duration_days(self):
        return self.time_end_days - self.time_start_days

    def revalidate(self):
        '''Defensive re-check of the frozen content (used inside the apply transaction).'''
        for name in ('plant_id', 'erosion_depth_m', 'deposition_depth_m', 'abrasion_dose', 'transport_flux'):
            a = getattr(self, name)
            if a is None:
                continue
            if a.flags.writeable or a.shape != (self.n,):
                raise EventError(f'event {name} integrity violated')
            if name != 'plant_id' and (not np.all(np.isfinite(a)) or (a.size and float(a.min()) < 0.0)):
                raise EventError(f'event {name} integrity violated (negative/non-finite)')
        if self.n and np.unique(self.plant_id).shape[0] != self.n:
            raise EventError('event plant_id integrity violated')
        return self

    def to_dict(self):
        d = {'schema': EVENT_SCHEMA, 'schema_version': EVENT_SCHEMA_VERSION, 'event_id': self.event_id,
             'source': self.source, 'time_unit': TIME_UNIT, 'time_start_days': self.time_start_days,
             'time_end_days': self.time_end_days, 'depth_unit': DEPTH_UNIT, 'plant_id': self.plant_id.tolist(),
             'erosion_depth_m': self.erosion_depth_m.tolist(), 'deposition_depth_m': self.deposition_depth_m.tolist(),
             'sign_convention': 'depths are >= 0; net_surface_delta_m = deposition - erosion (+ raised, - lowered)',
             'notes': self.notes}
        if self.abrasion_dose is not None:
            d.update({'abrasion_dose': self.abrasion_dose.tolist(), 'abrasion_unit': self.abrasion_unit})
        if self.transport_flux is not None:
            d.update({'transport_flux': self.transport_flux.tolist(), 'flux_unit': self.flux_unit})
        return d

    @classmethod
    def from_dict(cls, d):
        if not isinstance(d, dict):
            raise EventError('event must be a dict / JSON object')
        if d.get('schema') != EVENT_SCHEMA:
            raise EventError(f'event schema must be {EVENT_SCHEMA!r}, got {d.get("schema")!r}')
        if d.get('schema_version') != EVENT_SCHEMA_VERSION or isinstance(d.get('schema_version'), bool):
            raise EventError(f'unsupported event schema_version {d.get("schema_version")!r}')
        if d.get('time_unit', TIME_UNIT) != TIME_UNIT or d.get('depth_unit', DEPTH_UNIT) != DEPTH_UNIT:
            raise EventError(f'event units must be time {TIME_UNIT!r} and depth {DEPTH_UNIT!r}')
        known = {'schema', 'schema_version', 'event_id', 'source', 'time_unit', 'time_start_days', 'time_end_days',
                 'depth_unit', 'plant_id', 'erosion_depth_m', 'deposition_depth_m', 'abrasion_dose', 'abrasion_unit',
                 'transport_flux', 'flux_unit', 'sign_convention', 'notes'}
        unknown = sorted(set(d) - known)
        if unknown:
            raise EventError(f'event: unknown keys {unknown}')
        for k in ('plant_id', 'time_start_days', 'time_end_days', 'source'):
            if k not in d:
                raise EventError(f'event requires {k}')
        ids = d['plant_id']
        if not isinstance(ids, (list, tuple, np.ndarray)):
            raise EventError('plant_id must be a list of integers')
        if isinstance(ids, (list, tuple)):
            for v in ids:
                if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
                    raise EventError(f'plant_id entries must be integers, got {v!r}')
            ids = np.array(ids, dtype=np.int64) if ids else np.zeros(0, np.int64)

        def arr(key):
            v = d.get(key)
            if v is None:
                return None
            if isinstance(v, (list, tuple)):
                for e in v:
                    if isinstance(e, bool) or not _traits.is_real(e):
                        raise EventError(f'{key} entries must be real numbers, got {e!r}')
                return np.array(v, dtype=np.float64) if v else np.zeros(0, np.float64)
            return v

        return cls(ids, d['time_start_days'], d['time_end_days'], d['source'], erosion_depth_m=arr('erosion_depth_m'),
                   deposition_depth_m=arr('deposition_depth_m'), abrasion_dose=arr('abrasion_dose'),
                   abrasion_unit=d.get('abrasion_unit'), transport_flux=arr('transport_flux'), flux_unit=d.get('flux_unit'),
                   event_id=d.get('event_id'), notes=d.get('notes'))


class ReadOnlyView:
    '''What a response handler may see: owned read-only copies of the arrays plus the immutable Domain,
    the species table, catalog hash and clock. Nothing here reaches the canonical state.'''

    def __init__(self, state):
        s = object.__setattr__
        for k in state.ARRAY_NAMES:
            s(self, k, _frozen_array(getattr(state, k).copy()))
        s(self, 'species_names', tuple(state.species_names))
        s(self, 'catalog_hash', state.catalog_hash)
        s(self, 'time_days', float(state.time_days))
        s(self, 'domain', state.domain)   # Domain is immutable
        s(self, 'n', state.n)

    def __setattr__(self, name, value):
        raise AttributeError('ReadOnlyView is immutable')

    def species_of(self, index=None):
        names = np.array(self.species_names, dtype=object)
        return names[self.species_index if index is None else self.species_index[index]]


def resolve_event(state, event, *, require_elapsed=True):
    '''Validate an event against the population; returns row indices (no mutation).'''
    if not isinstance(event, SurfaceEventDiagnostics):
        raise EventError('event must be a SurfaceEventDiagnostics (use SurfaceEventDiagnostics.from_dict for JSON)')
    if not isinstance(state, _state.PlantState):
        raise EventError('state must be a leaf.state.PlantState')
    event.revalidate()
    if require_elapsed and event.time_end_days > state.time_days + 1e-9:
        raise EventError(f'event ends at {event.time_end_days} days but the population clock is at {state.time_days}; '
                         'advance_time() through the event interval first (or pass require_elapsed=False)')
    try:
        return state.indices_of(event.plant_id)
    except _state.StateError as exc:
        raise EventError(f'event plant IDs do not resolve: {exc}') from exc


def apply_event_diagnostics(state, event, *, handler=None, require_elapsed=True):
    '''Diagnostics by default; with a handler, validated all-or-nothing response. Returns a report dict.'''
    idx = resolve_event(state, event, require_elapsed=require_elapsed)
    ids = event.plant_id.copy()          # the IDs used through the WHOLE transaction (not re-read later)
    idx_saved = idx.copy()
    n = int(idx.shape[0])
    report = {'schema': EVENT_SCHEMA, 'event_id': event.event_id, 'source': event.source, 'n_plants': n,
              'time_start_days': event.time_start_days, 'time_end_days': event.time_end_days,
              'indices': idx_saved.copy(), 'plant_id': ids.copy(), 'status_before': state.status[idx_saved].copy(),
              'erosion_depth_m': event.erosion_depth_m.copy(), 'deposition_depth_m': event.deposition_depth_m.copy(),
              'net_surface_delta_m': event.net_surface_delta_m,
              'abrasion_dose': None if event.abrasion_dose is None else event.abrasion_dose.copy(),
              'abrasion_unit': event.abrasion_unit, 'transport_flux': None if event.transport_flux is None
              else event.transport_flux.copy(), 'flux_unit': event.flux_unit,
              'response': 'none (diagnostics only; no ecological response law selected)', 'changed': 0}
    if handler is None:
        return report
    if not callable(handler):
        raise EventError('handler must be callable(view, event, indices) -> dict of proposals or None')
    view = ReadOnlyView(state)
    try:
        proposals = handler(view, event, idx_saved.copy())
    except Exception as exc:
        raise EventError(f'response handler failed ({type(exc).__name__}: {exc}); vegetation state unchanged') from exc
    event.revalidate()
    if not np.array_equal(event.plant_id, ids):
        raise EventError('event IDs changed during the handler; vegetation state unchanged')
    if proposals is None:
        report['response'] = 'handler proposed no change'
        return report
    if not isinstance(proposals, dict) or any(k not in PROPOSAL_KEYS for k in proposals):
        raise EventError(f'handler proposals must be a dict with keys from {PROPOSAL_KEYS}; state unchanged')
    if not proposals or all(v is None for v in proposals.values()):
        report['response'] = 'handler proposed no change'
        return report
    candidate = state.copy()
    try:
        candidate.update_plants(ids, agb_dry_kg=proposals.get('agb_dry_kg'),
                                seasonal_agb_dry_kg=proposals.get('seasonal_agb_dry_kg'), status=proposals.get('status'))
    except _state.StateError as exc:
        raise EventError(f'handler proposals rejected ({exc}); vegetation state unchanged') from exc
    changed = 0
    for k in ('agb_dry_kg', 'seasonal_agb_dry_kg', 'status'):
        changed += int(np.count_nonzero(getattr(candidate, k)[idx_saved] != getattr(state, k)[idx_saved]))
    state.adopt(candidate)
    report.update({'response': 'handler proposals applied (validated all-or-nothing)', 'changed': changed,
                   'status_after': state.status[idx_saved].copy()})
    return report


__all__ = ['EVENT_SCHEMA', 'EVENT_SCHEMA_VERSION', 'SOURCES', 'DEPTH_UNIT', 'TIME_UNIT', 'PROPOSAL_KEYS', 'EventError',
           'SurfaceEventDiagnostics', 'ReadOnlyView', 'resolve_event', 'apply_event_diagnostics']
