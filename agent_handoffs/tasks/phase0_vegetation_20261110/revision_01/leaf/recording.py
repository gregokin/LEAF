'''leaf.recording: cadence-driven plant-state snapshot writer (no P history in memory).

The CALLER owns the clock (PlantState.advance_time). The writer only decides whether a snapshot is due:
  - write_initial(state): explicit initial snapshot; fixes the cadence origin at state.time_days.
  - maybe_write(state): writes exactly one snapshot if state.time_days >= next due time, else nothing.
    If the caller skipped past several due times, the intermediate states are NOT fabricated or
    interpolated: one snapshot of the CURRENT state is written and the missed due times are recorded in
    the manifest (`skipped_due_times_days`). The next due time is the first origin + k*cadence strictly
    greater than the current time.
  - write_final(state): explicit final snapshot regardless of cadence.
Every write validates first and writes atomically; a failed validation/write leaves the writer's due-time
bookkeeping and existing files untouched. Existing files are never overwritten unless overwrite=True.
'''
from __future__ import annotations

import json

from leaf import state as _state
from leaf import traits as _traits

MANIFEST_SCHEMA = 'leaf.snapshot_manifest'
MANIFEST_SCHEMA_VERSION = 1
_TIME_TOL = 1e-9


class RecordingError(ValueError):
    pass


class SnapshotWriter:
    '''path_template: str.format template using {time_days} (float), {index} (int) and {kind} (str),
    e.g. "out/veg_{index:04d}_{time_days:.3f}.csv". Templates without a varying field would overwrite
    themselves and are refused unless overwrite=True.'''

    def __init__(self, path_template, *, cadence_days=None, start_time_days=None, overwrite=False, provenance=None):
        if not isinstance(path_template, str) or not path_template:
            raise RecordingError('path_template must be a non-empty string')
        self.path_template = path_template
        self.overwrite = bool(overwrite)
        if not overwrite and not any(tok in path_template for tok in ('{time_days', '{index', '{kind')):
            raise RecordingError('path_template must contain {time_days}, {index} or {kind} so snapshots do not '
                                 'overwrite each other (or pass overwrite=True explicitly)')
        self.cadence_days = None
        if cadence_days is not None:
            c = _traits.real(cadence_days, 'cadence_days')
            if c <= 0.0:
                raise RecordingError(f'cadence_days must be > 0, got {c}')
            self.cadence_days = c
        self.start_time_days = None if start_time_days is None else _traits.real(start_time_days, 'start_time_days')
        self.provenance = dict(provenance or {})
        self.records = []        # small dicts, one per written snapshot (never P-sized)
        # index k of the next due time: next_due = start + k * cadence (integer k, no float accumulation)
        self._k = 1 if (self.start_time_days is not None and self.cadence_days is not None) else None
        self.last_time_days = None
        self.skipped_total = 0

    # ---- scheduling
    @property
    def next_due_days(self):
        if self._k is None:
            return None
        return self.start_time_days + self._k * self.cadence_days

    def due(self, time_days):
        '''True when a cadence snapshot is due at time_days (never true without cadence/origin).'''
        if self._k is None:
            return False
        return float(time_days) >= self.next_due_days - _TIME_TOL

    def _advance_schedule(self, time_days):
        '''Consume every due time <= time_days. Those not equal to time_days were MISSED (no state existed
        at them) and are returned as skipped; the next due time becomes the first one > time_days.'''
        skipped = []
        if self._k is None:
            return skipped
        t0, c, k = self.start_time_days, self.cadence_days, self._k
        while t0 + k * c <= time_days + _TIME_TOL:
            t = t0 + k * c
            if abs(t - time_days) > _TIME_TOL:
                skipped.append(t)
            k += 1
            if k - self._k > 1_000_000:
                raise RecordingError('unreasonable number of skipped due times')
        self._k = k
        return skipped

    # ---- writing
    def _path(self, kind, time_days):
        try:
            return self.path_template.format(time_days=time_days, index=len(self.records), kind=kind)
        except (KeyError, IndexError, ValueError) as exc:
            raise RecordingError(f'bad path_template {self.path_template!r}: {exc}') from exc

    def _write(self, state, kind, skipped):
        if not isinstance(state, _state.PlantState):
            raise RecordingError('state must be a leaf.state.PlantState')
        t = state.time_days
        if self.last_time_days is not None and t < self.last_time_days - _TIME_TOL:
            raise RecordingError(f'state time {t} is earlier than the last written snapshot time {self.last_time_days}')
        state.require_resolved('recording a snapshot')
        path = self._path(kind, t)
        prov = dict(state.provenance)
        prov.update(self.provenance)
        prov.update({'snapshot_kind': kind, 'snapshot_index': len(self.records), 'cadence_days': self.cadence_days,
                     'skipped_due_times_days': list(skipped)})
        # validate + serialise + atomic write BEFORE any bookkeeping changes
        _state.write_plant_csv(state, path, overwrite=self.overwrite, provenance=prov)
        self.records.append({'kind': kind, 'index': len(self.records), 'time_days': t, 'path': path,
                             'n_plants': state.n, 'skipped_due_times_days': list(skipped)})
        self.skipped_total += len(skipped)
        self.last_time_days = t
        return path

    def write_initial(self, state):
        '''Explicit initial snapshot; sets the cadence origin to the state time if none was given.'''
        if self.start_time_days is None:
            self.start_time_days = float(state.time_days)
            if self.cadence_days is not None:
                self._k = 1
        return self._write(state, 'initial', [])

    def maybe_write(self, state):
        '''Write one cadence snapshot if due; returns the path or None. Never fabricates skipped states.'''
        if self.cadence_days is None:
            return None
        if self._k is None:
            raise RecordingError('cadence snapshots need an origin: pass start_time_days or call write_initial() first')
        t = float(state.time_days)
        if not self.due(t):
            return None
        saved = (self._k, self.skipped_total, self.last_time_days, len(self.records))
        skipped = self._advance_schedule(t)
        try:
            return self._write(state, 'cadence', skipped)
        except BaseException:
            self._k, self.skipped_total, self.last_time_days = saved[0], saved[1], saved[2]
            del self.records[saved[3]:]
            raise

    def write_final(self, state):
        return self._write(state, 'final', [])

    # ---- manifest
    def manifest(self):
        return {'schema': MANIFEST_SCHEMA, 'schema_version': MANIFEST_SCHEMA_VERSION, 'path_template': self.path_template,
                'cadence_days': self.cadence_days, 'start_time_days': self.start_time_days,
                'next_due_days': self.next_due_days, 'n_snapshots': len(self.records), 'skipped_total': self.skipped_total,
                'policy': 'write only when due; skipped due times are listed, never fabricated or interpolated',
                'records': [dict(r) for r in self.records]}

    def write_manifest(self, path, *, overwrite=None):
        ow = self.overwrite if overwrite is None else bool(overwrite)
        return _state.atomic_write_text(path, json.dumps(self.manifest(), indent=1, allow_nan=False) + '\n', overwrite=ow)


__all__ = ['MANIFEST_SCHEMA', 'MANIFEST_SCHEMA_VERSION', 'RecordingError', 'SnapshotWriter']
