'''leaf.forcing: strict, neutral timestep records for scheduler-driven interevent steps (stdlib only).

A `TimestepRecord` describes ONE interevent timestep handed to LEAF by the event scheduler (the future
driver; the real MASY coordinator is not changed here):
  start, end        offset-aware datetimes with a FIXED UTC offset (datetime.timezone); both carry the same
                    offset; end > start (positive finite dt). No naive datetimes, no DST-aware zones (an
                    offset that may change inside a step is refused).
  temperature_c     the ACTUAL mean air temperature of the step, degC (REQUIRED even when PET is given).
  pet_mm_per_day    optional shared potential-evapotranspiration DEPTH RATE in mm/day (None when the
                    scheduler does not provide it). It is carried, never computed: no PET physics here.
  process_kind      explicit process kind; only 'interevent' is admitted (wind / rain / storm / water event /
                    recession steps are refused: LEAF does not catch up over them).
  gate              explicit process gate: 'open' (the interevent process runs over this step) or 'closed'
                    (the step is recorded but the process is switched off, e.g. a scheduler-declared pause).
  label             optional free text.
No synthetic weather, no synthetic dates, no independent clock starting at 1 January: every record is built
from scheduler-provided times. Exact arithmetic: dt_seconds_exact / dt_days_exact are stdlib Fractions
(microsecond-exact), so leap days, year boundaries and partial days are handled by the calendar itself.
Helpers split a record at local calendar midnights (the fixed offset defines the local calendar) and at
configured growth-update boundaries, validate UTC-offset consistency and contiguous progression (no gaps,
no overlaps, no catch-up), and convert scheduler weather mapping rows (`date` YYYY-MM-DD, `tmean_c` degC) into
whole-day records. No temperature-response or demand/growth formula lives here (pending the science choice).

Precision scope. Instants are Python datetimes, hence MICROSECOND-exact; fixed UTC offsets may carry whole
microseconds and are preserved exactly in every metadata export (`utc_offset_seconds_exact`, an exact Fraction
text, next to the float `utc_offset_seconds`; nothing is truncated to whole seconds). This part does NOT accept
MASY's rational SUBMICROSECOND instants: a future adapter must either preserve them (carrying its own exact
rational instants beside the record) or refuse them BEFORE any state mutation; `exact_microseconds` is the
refusing conversion (a rational number of seconds that is not a whole number of microseconds raises).
Dates are strict zero-padded Gregorian `YYYY-MM-DD` (years 0001-9999, via date.isoformat).
'''
from __future__ import annotations

import datetime as _dt
import math
import numbers
import re
from dataclasses import dataclass
from fractions import Fraction

VERSION = '0.2.0'
_DATE_RE = re.compile(r'[0-9]{4}-[0-9]{2}-[0-9]{2}')
PROCESS_KINDS = ('interevent',)
REFUSED_KIND_TOKENS = ('wind', 'rain', 'storm', 'event', 'water', 'recession', 'runoff')
GATES = ('open', 'closed')
SECONDS_PER_DAY = 86400
MIN_TEMPERATURE_C = -273.15
DATE_FORMAT = 'YYYY-MM-DD (zero padded, Gregorian, date.isoformat)'
UNITS = {'start': 'offset-aware datetime (fixed UTC offset)', 'end': 'offset-aware datetime (fixed UTC offset)',
         'temperature_c': 'degC, actual mean air temperature of the step',
         'pet_mm_per_day': 'mm/day potential evapotranspiration depth rate (optional, scheduler-provided)',
         'dt_seconds': 's (exact Fraction available)', 'dt_days': 'days (exact Fraction available)'}


class ForcingError(ValueError):
    '''Any timestep record / sequence validation error.'''


# ------------------------------------------------------------------ scalar validators
def _is_real(v):
    return isinstance(v, numbers.Real) and not isinstance(v, bool) and type(v).__name__ != 'bool_'


def _finite(v, name):
    if not _is_real(v):
        raise ForcingError(f'{name} must be a real number (not bool/str/None), got {v!r}')
    v = float(v)
    if not math.isfinite(v):
        raise ForcingError(f'{name} must be finite, got {v}')
    return v


def check_process_kind(kind):
    if not isinstance(kind, str):
        raise ForcingError('process_kind must be a string')
    k = kind.lower()
    if k in PROCESS_KINDS:
        return k
    if any(tok in k for tok in REFUSED_KIND_TOKENS):
        raise ForcingError(f'process_kind {kind!r} is a wind/rain/water-event/recession step; LEAF interevent records '
                           'never catch up over such steps (only interevent steps are admitted)')
    raise ForcingError(f'unknown process_kind {kind!r}; admitted: {PROCESS_KINDS}')


def check_gate(gate):
    if not isinstance(gate, str) or gate not in GATES:
        raise ForcingError(f'gate must be one of {GATES}, got {gate!r}')
    return gate


def check_fixed_offset_datetime(value, name='datetime'):
    '''An offset-aware datetime whose tzinfo is a FIXED offset (datetime.timezone); naive datetimes, dates
    and DST-capable zones are refused.'''
    if not isinstance(value, _dt.datetime):
        raise ForcingError(f'{name} must be a datetime.datetime (a date alone has no clock), got {type(value).__name__}')
    if value.tzinfo is None or value.utcoffset() is None:
        raise ForcingError(f'{name} must be offset-aware (naive datetime refused)')
    if not isinstance(value.tzinfo, _dt.timezone):
        raise ForcingError(f'{name} must carry a FIXED UTC offset (datetime.timezone); zone {value.tzinfo!r} may change '
                           'its offset inside a step and is refused')
    return value


def fixed_offset(offset):
    '''datetime.timezone from a timezone or a timedelta (whole microseconds; |offset| < 24 h).'''
    if isinstance(offset, _dt.timezone):
        return offset
    if isinstance(offset, _dt.timedelta):
        try:
            return _dt.timezone(offset)
        except ValueError as exc:
            raise ForcingError(f'invalid UTC offset {offset!r}: {exc}') from exc
    raise ForcingError(f'utc_offset must be a datetime.timezone or datetime.timedelta, got {offset!r}')


def exact_seconds(delta):
    '''Exact seconds of a timedelta as a Fraction (microsecond resolution preserved).'''
    if not isinstance(delta, _dt.timedelta):
        raise ForcingError('exact_seconds expects a datetime.timedelta')
    return Fraction(delta.days * SECONDS_PER_DAY + delta.seconds) + Fraction(delta.microseconds, 10 ** 6)


def exact_microseconds(seconds):
    '''Whole microseconds (int) of an exact RATIONAL number of seconds (int / Fraction). A value that is not a
    whole number of microseconds (e.g. MASY's rational submicrosecond instants) is REFUSED, never rounded or
    truncated; floats and bools are refused because they carry no exactness guarantee. Future adapters call this
    (or carry their own exact instants) before any state mutation.'''
    if isinstance(seconds, bool) or not isinstance(seconds, numbers.Rational):
        raise ForcingError(f'seconds must be an exact rational number (int / Fraction), got {seconds!r}')
    us = Fraction(seconds) * 10 ** 6
    if us.denominator != 1:
        raise ForcingError(f'{seconds} s is not a whole number of microseconds; submicrosecond instants are not '
                           'supported by this record (preserve them in the adapter or refuse them before any mutation)')
    return int(us)


def offset_seconds_exact(offset):
    '''Exact Fraction of seconds of a timedelta / timezone UTC offset (no truncation to whole seconds).'''
    if isinstance(offset, _dt.timezone):
        offset = offset.utcoffset(None)
    return exact_seconds(offset)


# ------------------------------------------------------------------ record
@dataclass(frozen=True)
class TimestepRecord:
    start: _dt.datetime
    end: _dt.datetime
    temperature_c: float
    pet_mm_per_day: object = None
    process_kind: str = 'interevent'
    gate: str = 'open'
    label: object = None

    def __post_init__(self):
        check_fixed_offset_datetime(self.start, 'start')
        check_fixed_offset_datetime(self.end, 'end')
        if self.start.utcoffset() != self.end.utcoffset():
            raise ForcingError(f'start offset {self.start.utcoffset()} differs from end offset {self.end.utcoffset()}; '
                               'one fixed offset per step')
        if not self.end > self.start:
            raise ForcingError(f'end {self.end.isoformat()} must be after start {self.start.isoformat()} (positive dt)')
        t = _finite(self.temperature_c, 'temperature_c')
        if t < MIN_TEMPERATURE_C:
            raise ForcingError(f'temperature_c {t} is below absolute zero')
        object.__setattr__(self, 'temperature_c', t)
        if self.pet_mm_per_day is not None:
            pet = _finite(self.pet_mm_per_day, 'pet_mm_per_day')
            if pet < 0.0:
                raise ForcingError(f'pet_mm_per_day must be >= 0, got {pet}')
            object.__setattr__(self, 'pet_mm_per_day', pet)
        object.__setattr__(self, 'process_kind', check_process_kind(self.process_kind))
        object.__setattr__(self, 'gate', check_gate(self.gate))
        if self.label is not None and not isinstance(self.label, str):
            raise ForcingError('label must be a string or None')

    # ---- durations (exact)
    @property
    def duration(self):
        return self.end - self.start

    @property
    def dt_seconds_exact(self):
        return exact_seconds(self.duration)

    @property
    def dt_seconds(self):
        return float(self.dt_seconds_exact)

    @property
    def dt_days_exact(self):
        return self.dt_seconds_exact / SECONDS_PER_DAY

    @property
    def dt_days(self):
        return float(self.dt_days_exact)

    # ---- calendar (local = the fixed offset's clock)
    @property
    def utc_offset(self):
        return self.start.utcoffset()

    @property
    def start_date(self):
        return self.start.date()

    @property
    def end_date(self):
        '''Local calendar date containing the last instant of the step (an end exactly at midnight belongs
        to the previous day).'''
        return (self.end - _dt.timedelta(microseconds=1)).date()

    @property
    def crosses_local_midnight(self):
        return self.start_date != self.end_date

    @property
    def is_whole_local_days(self):
        '''True when the step starts at local midnight and spans an integer number of local days.'''
        return (self.start.time() == _dt.time(0) and self.end.time() == _dt.time(0))

    @property
    def active(self):
        return self.gate == 'open'

    # ---- derived records
    def with_gate(self, gate):
        return TimestepRecord(self.start, self.end, self.temperature_c, self.pet_mm_per_day, self.process_kind,
                              check_gate(gate), self.label)

    def split_at(self, boundaries):
        return split_record(self, boundaries)

    def split_at_local_midnight(self):
        return split_record(self, local_midnights_between(self.start, self.end))

    def to_dict(self):
        off = offset_seconds_exact(self.utc_offset)
        return {'start': self.start.isoformat(), 'end': self.end.isoformat(),
                'utc_offset_seconds_exact': str(off), 'utc_offset_seconds': float(off),
                'temperature_c': self.temperature_c, 'pet_mm_per_day': self.pet_mm_per_day,
                'process_kind': self.process_kind, 'gate': self.gate, 'label': self.label,
                'dt_seconds_exact': str(self.dt_seconds_exact), 'dt_seconds': self.dt_seconds,
                'dt_days_exact': str(self.dt_days_exact), 'dt_days': self.dt_days,
                'crosses_local_midnight': self.crosses_local_midnight, 'units': dict(UNITS)}


# ------------------------------------------------------------------ splitting
def local_midnights_between(start, end):
    '''Local-calendar midnights strictly inside (start, end) in the records' fixed offset.'''
    check_fixed_offset_datetime(start, 'start')
    check_fixed_offset_datetime(end, 'end')
    if start.utcoffset() != end.utcoffset():
        raise ForcingError('start and end must share one fixed offset')
    out = []
    try:
        m = _dt.datetime.combine(start.date() + _dt.timedelta(days=1), _dt.time(0), tzinfo=start.tzinfo)
        while m < end:
            out.append(m)
            m += _dt.timedelta(days=1)
    except OverflowError:   # past datetime.max: no further midnight can exist inside (start, end)
        pass
    return out


def split_record(record, boundaries):
    '''Split one record at the given fixed-offset datetimes (growth-update boundaries, midnights, ...).
    Boundaries outside the open interval (start, end) are ignored (no zero-length pieces); every piece
    inherits temperature, PET, kind, gate and label (the values describe the whole interval).'''
    if not isinstance(record, TimestepRecord):
        raise ForcingError('split_record expects a TimestepRecord')
    inside = set()
    for b in boundaries:
        check_fixed_offset_datetime(b, 'boundary')
        if b.utcoffset() != record.utc_offset:
            raise ForcingError(f'boundary offset {b.utcoffset()} differs from the record offset {record.utc_offset}')
        if record.start < b < record.end:
            inside.add(b)
    cuts = [record.start] + sorted(inside) + [record.end]
    return [TimestepRecord(cuts[i], cuts[i + 1], record.temperature_c, record.pet_mm_per_day, record.process_kind,
                           record.gate, record.label) for i in range(len(cuts) - 1)]


# ------------------------------------------------------------------ sequences
def check_contiguous(records):
    '''Validate a non-empty sequence of records: fixed offset shared by all, strictly contiguous progression
    (each end == next start; no gap, no overlap, no reordering). Returns a scalar summary.'''
    recs = list(records)
    if not recs:
        raise ForcingError('at least one timestep record is required')
    for i, r in enumerate(recs):
        if not isinstance(r, TimestepRecord):
            raise ForcingError(f'record {i} is not a TimestepRecord')
    off = recs[0].utc_offset
    total = Fraction(0)
    closed = 0
    for i, r in enumerate(recs):
        if r.utc_offset != off:
            raise ForcingError(f'record {i} has UTC offset {r.utc_offset}, expected {off} (one fixed offset per sequence)')
        if i and r.start != recs[i - 1].end:
            gap = exact_seconds(r.start - recs[i - 1].end)
            kind = 'gap' if gap > 0 else 'overlap'
            raise ForcingError(f'records {i - 1} and {i} are not contiguous ({kind} of {gap} s: {recs[i - 1].end.isoformat()} '
                               f'-> {r.start.isoformat()}); no catch-up or overlap is admitted')
        total += r.dt_seconds_exact
        closed += 0 if r.active else 1
    return {'n_records': len(recs), 'start': recs[0].start.isoformat(), 'end': recs[-1].end.isoformat(),
            'utc_offset_seconds_exact': str(offset_seconds_exact(off)), 'utc_offset_seconds': float(offset_seconds_exact(off)),
            'total_seconds_exact': str(total),
            'total_days_exact': str(total / SECONDS_PER_DAY), 'total_days': float(total / SECONDS_PER_DAY),
            'n_gate_closed': closed, 'contiguous': True}


def parse_date(text, name='date'):
    '''Strict zero-padded ASCII YYYY-MM-DD (real Gregorian calendar date incl. leap days, years 0001-9999, so
    '0099-03-01' is valid) -> datetime.date. Uses date.isoformat for the round-trip check (no strftime('%Y'),
    whose padding is platform dependent below year 1000) and a fixed-shape pattern (no 'YYYYMMDD' / week dates /
    signs / non-ASCII digits that newer date.fromisoformat versions would accept).'''
    if not isinstance(text, str):
        raise ForcingError(f'{name} must be a YYYY-MM-DD string, got {text!r}')
    if not _DATE_RE.fullmatch(text):
        raise ForcingError(f'{name} {text!r} must be zero-padded ASCII YYYY-MM-DD')
    try:
        d = _dt.date(int(text[0:4]), int(text[5:7]), int(text[8:10]))
    except ValueError as exc:
        raise ForcingError(f'{name} {text!r} is not a valid Gregorian calendar date: {exc}') from exc
    if d.isoformat() != text:
        raise ForcingError(f'{name} {text!r} does not round-trip through date.isoformat')
    return d


def records_from_weather_rows(rows, utc_offset, *, date_key='date', temperature_key='tmean_c', pet_key=None,
                              gate='open', label_key=None):
    '''Whole-day records from scheduler-provided weather mapping rows (`date` YYYY-MM-DD, `tmean_c` degC,
    optional PET column named by `pet_key`). Each row becomes [local midnight, next local midnight) in the given
    fixed offset; rows must be strictly consecutive calendar days (a gap, duplicate or reordering is refused:
    nothing is synthesised or caught up). No MASY weather/PET physics is imported or copied.'''
    tz = fixed_offset(utc_offset)
    g = check_gate(gate)
    out = []
    prev = None
    for i, row in enumerate(rows):
        if not hasattr(row, 'keys') or not hasattr(row, '__getitem__'):
            raise ForcingError(f'row {i} must be a mapping with {date_key!r} and {temperature_key!r}')
        if date_key not in row or temperature_key not in row:
            raise ForcingError(f'row {i} must contain {date_key!r} and {temperature_key!r} (temperature is required '
                               'even when PET is provided)')
        d = parse_date(row[date_key], f'row {i} {date_key}')
        if prev is not None and d != prev + _dt.timedelta(days=1):
            raise ForcingError(f'row {i} date {d.isoformat()} does not follow {prev.isoformat()} by one day '
                               '(gap / duplicate / reordering refused; no catch-up)')
        pet = None
        if pet_key is not None:
            if pet_key not in row:
                raise ForcingError(f'row {i} lacks the requested PET column {pet_key!r}')
            pet = row[pet_key]
        label = None
        if label_key is not None and label_key in row and row[label_key] is not None:
            label = str(row[label_key])
        try:
            start = _dt.datetime.combine(d, _dt.time(0), tzinfo=tz)
            end = start + _dt.timedelta(days=1)
        except OverflowError as exc:
            raise ForcingError(f'row {i} date {d.isoformat()} is outside the representable datetime range: {exc}') from exc
        out.append(TimestepRecord(start, end, row[temperature_key], pet, 'interevent', g, label))
        prev = d
    if not out:
        raise ForcingError('no weather rows given')
    return out


__all__ = ['VERSION', 'PROCESS_KINDS', 'REFUSED_KIND_TOKENS', 'GATES', 'SECONDS_PER_DAY', 'MIN_TEMPERATURE_C',
           'DATE_FORMAT', 'UNITS', 'ForcingError', 'check_process_kind', 'check_gate', 'check_fixed_offset_datetime',
           'fixed_offset', 'exact_seconds', 'exact_microseconds', 'offset_seconds_exact', 'TimestepRecord', 'local_midnights_between', 'split_record',
           'check_contiguous', 'parse_date', 'records_from_weather_rows']
