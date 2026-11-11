'''leaf.recording: cadence-driven plant-state snapshot writer (no P history in memory).

The CALLER owns the clock (PlantState.advance_time). The writer only decides whether a snapshot is due:
  - write_initial(state): explicit initial snapshot; fixes the cadence origin at state.time_days (unless an
    explicit start_time_days was given).
  - maybe_write(state): writes exactly one snapshot if state.time_days has reached a due time, else nothing.
    Every due time <= the current time is consumed; those not equal to the current time were MISSED (no
    state existed at them) and are recorded as `skipped_due_times_days`, never fabricated or interpolated.
    Example: cadence 10 from 0, snapshots at 10 then 35 -> the 35-snapshot lists [20, 30]; a snapshot at
    30 instead would list [20]. The next due time is the first one strictly after the current time.
  - write_final(state): explicit final snapshot regardless of cadence.
Every write validates and serialises first, then writes atomically; ALL bookkeeping (origin, due index,
record list, skipped totals, last time) is staged and committed only after the file exists. Failures leave
the writer and existing files untouched. Existing files are never overwritten unless overwrite=True.
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

    def _consume_due(self, time_days):
        '''Pure: (new_k, skipped) consuming every due time <= time_days; no state change.'''
        skipped = []
        t0, c, k = self.start_time_days, self.cadence_days, self._k
        while t0 + k * c <= time_days + _TIME_TOL:
            t = t0 + k * c
            if abs(t - time_days) > _TIME_TOL:
                skipped.append(t)
            k += 1
            if k - self._k > 1_000_000:
                raise RecordingError('unreasonable number of skipped due times')
        return k, skipped

    # ---- writing
    def _path(self, kind, time_days):
        try:
            return self.path_template.format(time_days=time_days, index=len(self.records), kind=kind)
        except (KeyError, IndexError, ValueError) as exc:
            raise RecordingError(f'bad path_template {self.path_template!r}: {exc}') from exc

    def _write(self, state, kind, skipped):
        '''Validate, serialise and atomically write; returns (path, record). NO bookkeeping change.'''
        if not isinstance(state, _state.PlantState):
            raise RecordingError('state must be a leaf.state.PlantState')
        t = _state._finite_float(state.time_days, 'state.time_days')
        if self.last_time_days is not None and t < self.last_time_days - _TIME_TOL:
            raise RecordingError(f'state time {t} is earlier than the last written snapshot time {self.last_time_days}')
        state.require_resolved('recording a snapshot')
        path = self._path(kind, t)
        prov = dict(state.provenance)
        prov.update(self.provenance)
        prov.update({'snapshot_kind': kind, 'snapshot_index': len(self.records), 'cadence_days': self.cadence_days,
                     'skipped_due_times_days': list(skipped)})
        _state.write_plant_csv(state, path, overwrite=self.overwrite, provenance=prov)
        record = {'kind': kind, 'index': len(self.records), 'time_days': t, 'path': path, 'n_plants': state.n,
                  'skipped_due_times_days': list(skipped)}
        return path, record

    def _commit(self, record, skipped):
        self.records.append(record)
        self.skipped_total += len(skipped)
        self.last_time_days = record['time_days']

    def write_initial(self, state):
        '''Explicit initial snapshot; the cadence origin becomes the state time (if none was given) only
        after the file has been written successfully.'''
        path, record = self._write(state, 'initial', [])
        if self.start_time_days is None:
            self.start_time_days = record['time_days']
            if self.cadence_days is not None:
                self._k = 1
        self._commit(record, [])
        return path

    def maybe_write(self, state):
        '''Write one cadence snapshot if due; returns the path or None. Never fabricates skipped states.'''
        if self.cadence_days is None:
            return None
        if self._k is None:
            raise RecordingError('cadence snapshots need an origin: pass start_time_days or call write_initial() first')
        t = float(state.time_days)
        if not self.due(t):
            return None
        new_k, skipped = self._consume_due(t)
        path, record = self._write(state, 'cadence', skipped)   # raises -> nothing below runs
        self._k = new_k
        self._commit(record, skipped)
        return path

    def write_final(self, state):
        path, record = self._write(state, 'final', [])
        self._commit(record, [])
        return path

    # ---- manifest
    def manifest(self):
        return {'schema': MANIFEST_SCHEMA, 'schema_version': MANIFEST_SCHEMA_VERSION, 'path_template': self.path_template,
                'cadence_days': self.cadence_days, 'start_time_days': self.start_time_days,
                'next_due_days': self.next_due_days, 'n_snapshots': len(self.records), 'skipped_total': self.skipped_total,
                'policy': 'write only when due; every missed due time is listed, never fabricated or interpolated',
                'records': [dict(r) for r in self.records]}

    def write_manifest(self, path, *, overwrite=None):
        ow = self.overwrite if overwrite is None else bool(overwrite)
        return _state.atomic_write_text(path, json.dumps(self.manifest(), indent=1, allow_nan=False) + '\n', overwrite=ow)


__all__ = ['MANIFEST_SCHEMA', 'MANIFEST_SCHEMA_VERSION', 'RecordingError', 'SnapshotWriter']
