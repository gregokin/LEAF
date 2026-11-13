'''leaf.biocalendar: PURE LEAF biology-calendar functions (UNQUALIFIED, UNCALIBRATED; no water, no native SYRUP).

The schedule / season / transport / flush rules of `leaf.interevent.IntereventSession` as pure functions of an immutable
`Schedule`, so an external scheduler (MASY) can sequence dated fragments without constructing an IntereventSession and
without copying any biology law. Nothing here touches water, evaporation, transpiration or plant state; the caller applies
the returned decisions with `leaf.growth` and the adapter that owns the accepted water boundary.

Rules (identical to the session when `growth_enabled=True`, equality-tested against the real session):
  * growth boundaries are a FIXED GRID `next_boundary + k * cadence` (cadence <= 7 days); records are split at local
    midnights and grid points (new years are midnights); boundaries crossed by transport are skipped (no catch-up);
  * a year end inside transport sets `season_due`; the turnover is applied before the next OPEN interval's water step
    (zero water, no event duration);
  * transport (wind / rain / recession / closed-gate pause) advances the calendar only and requires NO pending growth: the
    integrator flushes the partial period at the LAST OPEN interevent endpoint before the event (`needs_flush`,
    `flush_step`); the look-ahead that knows an event follows is the scheduler's;
  * never any E / T / growth during transport.

Approved growth-disabled semantics (explicit opt-in `growth_enabled=False`; the old session never uses it): transpiration
may still occur in that period, but growth is OFF: no biomass growth there, including the flush of earlier earned credit. The period
adds NO pending demand / delivery / activity / temperature / day credit (`credit_increment` returns the SAME arrays, `open_step`
leaves `pending_days` unchanged); a previous enabled pending credit is preserved, not erased; age / calendar still advance; no growth
or season turnover fires while disabled (the turnover waits for an enabled open step) and the grid moves on (no catch-up). The
integrator's look-ahead flushes pending credit at the LAST GROWTH-ENABLED endpoint before a disabled period or a transport event
(`flush_required`); `flush_step(schedule, growth_enabled=False)` with pending credit is refused. Credit across disabled spans
integrates ENABLED time only and stays inside the weekly cadence envelope.

Geometry: `materialize_state` / `rebuild_geometry` compose ONLY public functions (`PlantState`, `canopy.derive_geometry`,
`roots.build_root_map`) exactly like `IntereventSession._materialize/_build_geometry`; optionally the sparse canopy CSR of
`leaf.plant_pet` is built in the same cold rebuild. Phase 2 / 3 annual laws are NOT part of this module.
'''
from __future__ import annotations

import calendar
import dataclasses
import math
import datetime as _dt
from fractions import Fraction

import numpy as np

from leaf import canopy as _canopy
from leaf import forcing as _forcing
from leaf import growth as _growth
from leaf import roots as _roots
from leaf import state as _state
from leaf import traits as _traits

VERSION = '0.1.0'
MAX_CADENCE = _dt.timedelta(days=7)
SECONDS_PER_DAY = 86400
TRANSPORT_KINDS = ('wind', 'rain', 'recession', 'closed_gate')
# Structural per-plant byte constants of the session ledger (identical values; equality-tested against interevent).
ROW_BYTES_PER_PLANT = 62
MATERIALIZE_BYTES_PER_PLANT = 62 + 62 + 17 + 16
VALIDATE_SCRATCH_BYTES_PER_PLANT = 33
GEOMETRY_BYTES_PER_PLANT = _roots.GEOMETRY_BYTES_PER_PLANT
BACKENDS = ('numpy', 'numba', 'cuda')


class BioCalendarError(ValueError):
    '''Any biology-calendar validation error (raised before a result exists; inputs are immutable).'''


class BioCalendarStateError(BioCalendarError):
    '''Wrong cursor (gap / overlap / replay) or pending growth before transport.'''


class BioCalendarBoundaryError(BioCalendarError):
    '''A record crosses a local midnight or growth boundary and must be split first.'''


class BioCalendarAdmissionError(BioCalendarError):
    '''A byte ledger exceeds max_bytes; raised BEFORE the stage allocates.'''

    def __init__(self, stage, total, max_bytes):
        super().__init__(f'{stage}: estimated {total} bytes exceed max_bytes={max_bytes}; refused before allocation')
        self.stage, self.total, self.max_bytes = stage, int(total), int(max_bytes)


# ------------------------------------------------------------------ date arithmetic (the session delegates here)
def next_new_year(t):
    '''First instant of the next local year (same fixed offset) or None past datetime.max.'''
    try:
        return _dt.datetime(t.year + 1, 1, 1, tzinfo=t.tzinfo)
    except (OverflowError, ValueError):
        return None


def year_days(t):
    '''Actual year length (365 / 366) of the local year of `t`.'''
    return 366 if calendar.isleap(t.year) else 365


def advance_boundary(next_boundary, end, cadence):
    '''Smallest grid point `next_boundary + k * cadence` strictly after `end` (no move when `end < next_boundary`).'''
    if end >= next_boundary:
        k = (end - next_boundary) // cadence + 1
        next_boundary = next_boundary + k * cadence
    return next_boundary


def elapsed_days(anchor, cursor):
    '''Exact elapsed days (Fraction) from the schedule anchor to `cursor`.'''
    return _forcing.exact_seconds(cursor - anchor) / SECONDS_PER_DAY


def _cadence_days(cadence):
    return Fraction(_forcing.exact_seconds(cadence), SECONDS_PER_DAY)


# ------------------------------------------------------------------ schedule
@dataclasses.dataclass(frozen=True)
class Schedule:
    '''Immutable biology calendar state: exact fixed-offset instants, the growth grid, the deferred-season flag and the
    pending ENABLED growth time (Fraction of days, <= cadence).'''
    anchor: _dt.datetime
    cadence: _dt.timedelta
    cursor: _dt.datetime
    next_boundary: _dt.datetime
    season_due: bool = False
    pending_days: Fraction = Fraction(0)

    def __post_init__(self):
        _forcing.check_fixed_offset_datetime(self.anchor, 'anchor')
        _forcing.check_fixed_offset_datetime(self.cursor, 'cursor')
        _forcing.check_fixed_offset_datetime(self.next_boundary, 'next_boundary')
        if not isinstance(self.cadence, _dt.timedelta) or not _dt.timedelta(0) < self.cadence <= MAX_CADENCE:
            raise BioCalendarError(f'cadence must be a datetime.timedelta in (0, 7 days], got {self.cadence!r}')
        off = self.anchor.utcoffset()
        if self.cursor.utcoffset() != off or self.next_boundary.utcoffset() != off:
            raise BioCalendarError('anchor, cursor and next_boundary must share one fixed offset')
        if self.cursor < self.anchor:
            raise BioCalendarError('cursor must not precede the anchor')
        if not self.next_boundary > self.cursor:
            raise BioCalendarError('next_boundary must lie strictly after the cursor')
        if not isinstance(self.season_due, bool):
            raise BioCalendarError('season_due must be a bool')
        pd = self.pending_days
        if isinstance(pd, bool) or not isinstance(pd, (Fraction, int)):
            raise BioCalendarError('pending_days must be an exact Fraction of days')
        object.__setattr__(self, 'pending_days', Fraction(pd))
        if self.pending_days < 0 or self.pending_days > _cadence_days(self.cadence):
            raise BioCalendarError('pending_days must lie in [0, cadence]')

    @property
    def has_pending_growth(self):
        return self.pending_days > 0

    def to_dict(self):
        return {'version': VERSION, 'anchor': self.anchor.isoformat(), 'cursor': self.cursor.isoformat(),
                'next_boundary': self.next_boundary.isoformat(), 'cadence_seconds_exact': str(_forcing.exact_seconds(self.cadence)),
                'season_due': self.season_due, 'pending_days': str(self.pending_days)}


def new_schedule(anchor, cadence):
    '''The start state of a session: cursor = anchor, first grid point `anchor + cadence`, nothing pending.'''
    _forcing.check_fixed_offset_datetime(anchor, 'anchor')
    if not isinstance(cadence, _dt.timedelta) or not _dt.timedelta(0) < cadence <= MAX_CADENCE:
        raise BioCalendarError(f'cadence must be a datetime.timedelta in (0, 7 days], got {cadence!r}')
    return Schedule(anchor, cadence, anchor, anchor + cadence)


def boundaries(schedule):
    '''Boundaries the caller must split at, from the cursor (same dict as `IntereventSession.boundaries()`).'''
    _check_schedule(schedule)
    c = schedule.cursor
    nxt_mid = _dt.datetime.combine(c.date() + _dt.timedelta(days=1), _dt.time(0), tzinfo=schedule.anchor.tzinfo)
    return {'cursor': c, 'next_local_midnight': nxt_mid, 'next_growth_boundary': schedule.next_boundary,
            'next_new_year': next_new_year(c), 'growth_cadence': schedule.cadence,
            'next_forced': min(x for x in (nxt_mid, schedule.next_boundary, next_new_year(c)) if x is not None),
            'rule': 'split at every local midnight, every growth-grid point (next_growth_boundary + k * cadence) and '
                    'every new year; closed-gate and transport intervals need no splitting'}


def _check_schedule(schedule):
    if not isinstance(schedule, Schedule):
        raise BioCalendarError('schedule must be a leaf.biocalendar.Schedule')


def split_record(schedule, record):
    '''Pieces of an OPEN record split at local midnights and growth-grid points (a non-open record is returned whole).'''
    _check_schedule(schedule)
    if not isinstance(record, _forcing.TimestepRecord):
        raise BioCalendarError('record must be a leaf.forcing.TimestepRecord')
    if not record.active:
        return [record]
    cuts = list(_forcing.local_midnights_between(record.start, record.end))
    g = schedule.next_boundary
    while g < record.end:
        cuts.append(g)
        g = g + schedule.cadence
    return _forcing.split_record(record, cuts)


def unsplit_reason(schedule, record):
    '''None when the record needs no splitting, else the refusal text (same wording as the session).'''
    _check_schedule(schedule)
    mids = _forcing.local_midnights_between(record.start, record.end)
    if mids:
        return (f'record {record.start.isoformat()}..{record.end.isoformat()} crosses local midnight '
                f'{mids[0].isoformat()}; split it (split_record) before submission')
    if schedule.next_boundary < record.end:
        return (f'record crosses the growth boundary {schedule.next_boundary.isoformat()}; split it (split_record) '
                'before submission')
    return None


def check_interval(schedule, start, end):
    '''Offsets, positive duration and `start == cursor` (replay / overlap / gap are typed state errors).'''
    _check_schedule(schedule)
    _forcing.check_fixed_offset_datetime(start, 'start')
    _forcing.check_fixed_offset_datetime(end, 'end')
    off = schedule.anchor.utcoffset()
    if start.utcoffset() != off or end.utcoffset() != off:
        raise BioCalendarError(f'offsets must equal the schedule offset {off}')
    if not end > start:
        raise BioCalendarError('end must be after start')
    if start != schedule.cursor:
        if end <= schedule.cursor:
            why = 'replay (this period was already accounted)'
        elif start < schedule.cursor:
            why = 'overlap with the accounted period'
        else:
            why = 'gap before the interval (no catch-up over missing time)'
        raise BioCalendarStateError(f'interval start {start.isoformat()} != cursor {schedule.cursor.isoformat()}: {why}')


# ------------------------------------------------------------------ open interval
@dataclasses.dataclass(frozen=True)
class OpenStep:
    '''Decisions of one OPEN interevent interval (the caller applies them in this order):
    1. `season_first`: grow with `year_end=True` on the current pending credit (`season_period_days`, normally 0),
       zero water, BEFORE the interval's water step; the credit is consumed;
    2. the interval's water step / demand / delivery (SYRUP + adapter), credit accumulated only if `credit_days > 0`;
    3. `fire_growth`: grow with `year_end` on the pending credit (`period_days`) after the credit.'''
    schedule: Schedule
    season_first: bool
    season_period_days: Fraction
    fire_growth: bool
    year_end: bool
    credit_days: Fraction
    period_days: Fraction


def open_step(schedule, record, *, growth_enabled=True, has_plants=True):
    '''Calendar decision of one open interevent record (pure; the previous schedule is untouched).'''
    _check_schedule(schedule)
    if not isinstance(record, _forcing.TimestepRecord):
        raise BioCalendarError('record must be a leaf.forcing.TimestepRecord')
    for name, v in (('growth_enabled', growth_enabled), ('has_plants', has_plants)):
        if not isinstance(v, bool):
            raise BioCalendarError(f'{name} must be a bool, got {v!r}')
    if not record.active:
        raise BioCalendarError('a closed-gate record is a pause: use pause_step')
    check_interval(schedule, record.start, record.end)
    reason = unsplit_reason(schedule, record)
    if reason:
        raise BioCalendarBoundaryError(reason)
    end = record.end
    ny = next_new_year(record.start)
    year_end = ny is not None and end == ny
    season_first = bool(schedule.season_due and has_plants and growth_enabled)
    season_days = schedule.pending_days if season_first else Fraction(0)
    pending = Fraction(0) if season_first else schedule.pending_days
    credit = record.dt_days_exact if growth_enabled and has_plants else Fraction(0)
    pending1 = pending + credit
    if pending1 > _cadence_days(schedule.cadence):
        raise BioCalendarStateError('pending duration would exceed the growth cadence (weekly Euler envelope); flush the '
                                    'pending enabled period before accumulating more')
    fire = bool(growth_enabled and has_plants and (end == schedule.next_boundary or year_end))
    pending2 = Fraction(0) if fire else pending1
    if not has_plants:
        season_due = False
    elif growth_enabled:
        season_due = False
    else:
        season_due = bool(schedule.season_due or year_end)
    nb = advance_boundary(schedule.next_boundary, end, schedule.cadence)
    new = Schedule(schedule.anchor, schedule.cadence, end, nb, season_due, pending2)
    return OpenStep(new, season_first, season_days, fire, bool(year_end), credit, pending1)


# ------------------------------------------------------------------ transport / pause / flush
@dataclasses.dataclass(frozen=True)
class TransportStep:
    schedule: Schedule
    skipped_boundaries: int
    season_deferred: bool


def transport_step(schedule, start, end):
    '''Advance the calendar over wind / rain / recession / a closed-gate pause: no E, T, growth or catch-up. Refused while
    growth is pending (flush at the last open interevent endpoint first).'''
    check_interval(schedule, start, end)
    if schedule.pending_days != 0:
        raise BioCalendarStateError('pending growth must be flushed (flush_step) before a transport / paused interval')
    ny = next_new_year(start)
    season = ny is not None and ny <= end
    nb, skipped = schedule.next_boundary, 0
    if end >= nb:
        nb2 = advance_boundary(nb, end, schedule.cadence)
        skipped = int((nb2 - nb) // schedule.cadence)
        nb = nb2
    new = Schedule(schedule.anchor, schedule.cadence, end, nb, bool(schedule.season_due or season), Fraction(0))
    return TransportStep(new, skipped, bool(season))


pause_step = transport_step


def needs_flush(schedule):
    '''True when a partial enabled growth period is pending (it must be flushed before a transport interval).'''
    _check_schedule(schedule)
    return schedule.pending_days > 0


def flush_required(schedule, *, next_growth_enabled, next_is_transport):
    '''Look-ahead helper: True when pending enabled credit exists and the NEXT period is a transport event or growth-disabled, i.e. the
    integrator must flush at the current LAST growth-enabled endpoint (never during the disabled period).'''
    _check_schedule(schedule)
    for name, v in (('next_growth_enabled', next_growth_enabled), ('next_is_transport', next_is_transport)):
        if not isinstance(v, bool):
            raise BioCalendarError(f'{name} must be a bool, got {v!r}')
    return bool(schedule.pending_days > 0 and (next_is_transport or not next_growth_enabled))


def flush_step(schedule, *, growth_enabled):
    '''Schedule after finishing the partial period AT THE CURSOR (grid, clock and season flag unchanged, pending -> 0). `growth_enabled`
    (strict bool, REQUIRED) is the growth flag of the endpoint being flushed: with growth OFF there is no growth, so a flush that would
    convert pending credit is refused (flush at the last growth-enabled endpoint). A no-op (same schedule) when nothing is pending.'''
    _check_schedule(schedule)
    if not isinstance(growth_enabled, bool):
        raise BioCalendarError(f'growth_enabled must be a bool, got {growth_enabled!r}')
    if schedule.pending_days == 0:
        return schedule
    if not growth_enabled:
        raise BioCalendarStateError('growth is disabled at this endpoint: pending credit must be flushed at the LAST growth-enabled '
                                    'endpoint (see flush_required), never during a disabled period')
    return dataclasses.replace(schedule, pending_days=Fraction(0))


# ------------------------------------------------------------------ growth credit contract
def _cap(v, name):
    try:
        return _traits.strict_int(v, name, 0)
    except _traits.CatalogError as exc:
        raise BioCalendarError(str(exc)) from exc


class BioCalendarDeviceError(BioCalendarError):
    '''A CuPy credit array lives on another device than the others / than the CURRENT device (devices are never switched).'''


FAMILIES = ('numpy', 'cupy')
PACKET_BYTES = 8                          # one device scalar (and one D2H) per CuPy reduction
VALIDATION_REDUCTIONS = {True: 16, False: 8}      # min + max of 8 (enabled) / 4 (disabled) arrays
RESULT_REDUCTIONS = 6                     # Dn/Un/An/Tn max (finite) + the two `.any()` relation checks (enabled only)
_CUPY = {}


def _cupy_ndarray():
    '''The ACTUAL optional `cupy.ndarray` class (through `leaf.growth._cupy`), or None when CuPy is unavailable (cached).'''
    if 'cls' not in _CUPY:
        try:
            _CUPY['cls'] = _growth._cupy().ndarray
        except _growth.GrowthError:
            _CUPY['cls'] = None
    return _CUPY['cls']


def _kind(a):
    ''''numpy' for an exact numpy.ndarray, 'cupy' for an exact cupy.ndarray of the real optional package, else None
    (subclasses and look-alike classes are refused).'''
    if type(a) is np.ndarray:
        return 'numpy'
    cls = _cupy_ndarray()
    if cls is not None and type(a) is cls:
        return 'cupy'
    return None


def estimate_credit_bytes(n_plants, n_species, *, backend='numpy', growth_enabled=True):
    '''Phase peaks of one `credit_increment` for ARBITRARY P and S (arrays on the backend; nothing else is allocated):
    validate: reductions only (CuPy: fixed device packets); compute + check: the outputs `D+d, U+u` (16P) and `A+a, T+t` (16S)
    stay live while the P-side check temporaries (`Dn*(1+rtol)` 8P + bool P) and then the S-side one (`Tn > An`, bool S) exist one
    after the other: peak = 16P + 16S + max(9P, S). Disabled: no P / S-sized allocation. CuPy adds the fixed reduction packets
    (8 B device + 8 B D2H each; CuPy-internal workspace and pool rounding excluded). `total_bytes` = max over the phases.'''
    P = _traits.strict_int(n_plants, 'n_plants', 0)
    S = _traits.strict_int(n_species, 'n_species', 0)
    if backend not in FAMILIES:
        raise BioCalendarError(f'backend must be one of {FAMILIES}')
    if not isinstance(growth_enabled, bool):
        raise BioCalendarError('growth_enabled must be a bool')
    cu = backend == 'cupy'
    nv = VALIDATION_REDUCTIONS[growth_enabled]
    nr = RESULT_REDUCTIONS if growth_enabled else 0
    pk_v = PACKET_BYTES * nv if cu else 0
    pk_r = PACKET_BYTES * nr if cu else 0
    out = 16 * P + 16 * S if growth_enabled else 0
    tmp = max(9 * P, S) if growth_enabled else 0
    phases = {'validate': pk_v, 'compute_and_check': out + tmp + pk_r}
    return {'items': {'D_U_outputs': 16 * P if growth_enabled else 0, 'A_T_outputs': 16 * S if growth_enabled else 0,
                      'check_temporary_max(9P, S)': tmp, 'device_packets': max(pk_v, pk_r)},
            'phases': phases, 'total_bytes': int(max(phases.values())), 'validation_reductions': nv + nr if cu else 0,
            'device_packet_bytes': int(max(pk_v, pk_r)), 'd2h_bytes': int(PACKET_BYTES * (nv + nr)) if cu else 0,
            'exclusions': 'CuPy-internal reduction workspace, memory-pool rounding and Python objects are not counted'}


def credit_increment(D, U, A, T, d_inc, u_inc, a_inc, t_inc, *, growth_enabled, max_bytes, reserve_bytes=0,
                     overdelivery_rtol=_growth.OVERDELIVERY_RTOL, pending_days=None):
    '''Validated, admitted pending-growth credit after one interval (pure: nothing is mutated).
    Order: strict flags / integers -> ONE array family (exact numpy.ndarray or the real cupy.ndarray; no subclasses, look-alikes or
    mixing) with 1-D float64 and matched shapes (D / U / d / u: P; A / T / a / t: S) -> for CuPy the arrays' device must be the
    CURRENT device (never switched) -> ADMISSION `reserve_bytes + estimate_credit_bytes(...)['total_bytes'] <= max_bytes` (before
    any validation reduction, which on CuPy allocate / transfer the counted fixed packets) -> finite non-negative inputs by
    reductions -> the adds -> result checks with the ACTUAL downstream restrictions (`growth._check_integrals`: finite, `T <= A`
    EXACT, `A <= MAX_ACTIVITY_INTEGRAL (1 + ACTIVITY_INTEGRAL_SLACK)`; `U <= D (1 + rtol)`; optional session bound
    `A <= pending_days / 365 (1 + 1e-9)`). Disabled: the SAME input arrays are returned (earned credit preserved, none added) after
    the stored credit passes the reduction-only validation (no P / S-sized allocation or transfer). A failure returns nothing.'''
    if not isinstance(growth_enabled, bool):
        raise BioCalendarError(f'growth_enabled must be a bool, got {growth_enabled!r}')
    mb, rb = _cap(max_bytes, 'max_bytes'), _cap(reserve_bytes, 'reserve_bytes')
    rtol = float(overdelivery_rtol)
    if not 0.0 <= rtol <= _growth.MAX_OVERDELIVERY_RTOL:
        raise BioCalendarError(f'overdelivery_rtol must lie in [0, {_growth.MAX_OVERDELIVERY_RTOL}]')
    arrs = (D, U, d_inc, u_inc, A, T, a_inc, t_inc)
    kinds = {_kind(a) for a in arrs}
    if None in kinds or len(kinds) != 1:
        raise BioCalendarError('all eight arrays must be exact numpy.ndarray OR all the real cupy.ndarray (no subclasses, '
                               'look-alike classes, mixing or implicit transfer)')
    fam = next(iter(kinds))
    for a in arrs:
        if a.ndim != 1 or a.dtype != np.float64:
            raise BioCalendarError('credit arrays must be 1-D float64')
    P, S = int(D.shape[0]), int(A.shape[0])
    if not (int(U.shape[0]) == int(d_inc.shape[0]) == int(u_inc.shape[0]) == P and
            int(T.shape[0]) == int(a_inc.shape[0]) == int(t_inc.shape[0]) == S):
        raise BioCalendarError('shape mismatch: D / U / d / u must share P and A / T / a / t must share S')
    if fam == 'cupy':
        devs = {int(a.device.id) for a in arrs}
        cur = int(_growth._current_device_id())
        if len(devs) != 1 or devs != {cur}:
            raise BioCalendarDeviceError(f'CuPy credit arrays live on device(s) {sorted(devs)} but the current device is {cur}; '
                                         'devices are never switched')
    est = estimate_credit_bytes(P, S, backend=fam, growth_enabled=growth_enabled)
    if rb + est['total_bytes'] > mb:
        raise BioCalendarAdmissionError('credit_increment', rb + est['total_bytes'], mb)
    checked = (('D', D), ('U', U), ('A', A), ('T', T)) if not growth_enabled else tuple(zip(
        ('D', 'U', 'd_inc', 'u_inc', 'A', 'T', 'a_inc', 't_inc'), arrs))
    for name, a in checked:
        if a.shape[0] and not (float(a.min()) >= 0.0 and math.isfinite(float(a.max()))):
            raise BioCalendarError(f'{name} must be finite and non-negative')
    if not growth_enabled:
        return D, U, A, T
    Dn, Un, An, Tn = D + d_inc, U + u_inc, A + a_inc, T + t_inc
    if P:
        if not (math.isfinite(float(Dn.max())) and math.isfinite(float(Un.max()))):
            raise BioCalendarError('pending requested / delivered water would become non-finite')
        if bool((Un > Dn * (1.0 + rtol)).any()):
            raise BioCalendarError(f'delivered water would exceed requested water beyond the allocator bound {rtol!r}')
    if S:
        amax = float(An.max())
        if not (math.isfinite(amax) and math.isfinite(float(Tn.max()))):
            raise BioCalendarError('pending activity / temperature integrals would become non-finite')
        if bool((Tn > An).any()):                                       # growth._check_integrals: T <= A EXACT
            raise BioCalendarError('temperature integral exceeds the activity integral (growth requires T <= A exactly)')
        if amax > _growth.MAX_ACTIVITY_INTEGRAL * (1.0 + _growth.ACTIVITY_INTEGRAL_SLACK):
            raise BioCalendarError('activity integral exceeds 7/365 (growth is a weekly-or-shorter Euler adaptation)')
        if pending_days is not None:
            pdv = Fraction(pending_days)
            if pdv < 0 or amax > float(pdv) / 365.0 * (1.0 + 1e-9):
                raise BioCalendarError('activity integral is inconsistent with the pending duration')
    return Dn, Un, An, Tn


# ------------------------------------------------------------------ geometry
def _admit(stage, total, max_bytes):
    if int(total) > int(max_bytes):
        raise BioCalendarAdmissionError(stage, total, max_bytes)


def materialize_state(base, live_rows, B_live, E_live, *, elapsed_days, status_live=None):
    '''Canonical PlantState at `base.time_days + elapsed_days`: base arrays copied, B / E of the live rows replaced and
    their ages advanced by `elapsed_days` (`IntereventSession._materialize` through the public constructor).'''
    if not isinstance(base, _state.PlantState):
        raise BioCalendarError('base must be a leaf.state.PlantState')
    ed = float(elapsed_days)
    rows = np.asarray(live_rows, dtype=np.int64)
    a = {k: np.array(getattr(base, k), copy=True) for k in _state.PlantState.ARRAY_NAMES}
    if rows.shape[0]:
        a['agb_dry_kg'][rows] = B_live
        a['seasonal_agb_dry_kg'][rows] = E_live
        a['age_days'][rows] = a['age_days'][rows] + ed
        if status_live is not None:
            a['status'][rows] = status_live
    return _state.PlantState(base.domain, base.species_names, base.catalog_hash, ids=a['ids'],
                             species_index=a['species_index'], x_m=a['x_m'], y_m=a['y_m'],
                             establishment_ground_m=a['establishment_ground_m'],
                             establishment_unresolved=a['establishment_unresolved'], agb_dry_kg=a['agb_dry_kg'],
                             seasonal_agb_dry_kg=a['seasonal_agb_dry_kg'], age_days=a['age_days'], status=a['status'],
                             time_days=float(base.time_days) + ed, provenance=dict(base.provenance))


@dataclasses.dataclass
class GeometryRebuild:
    root_map: object
    canopy_area_m2: object
    state: object
    canopy_map: object
    memory: dict


def rebuild_geometry(base, catalog, B_live, E_live, *, elapsed_days, max_edges, max_bytes, reserve_bytes=0,
                     backend='numpy', builder='compiled', layer_fraction_overrides=None, live_rows=None,
                     canopy=None, provenance=None):
    '''Fixed-weight root map + canopy areas for candidate B / E of the LIVE rows (composition of public functions only:
    `derive_geometry` radius -> `pi r^2` -> `build_root_map(statuses=('live',))`; rows follow the state row order).
    `reserve_bytes` = every other live byte of the caller; stages are admitted against `max_bytes` BEFORE allocating and the
    root builder receives the remaining budget. `canopy` (dict: `max_edges`, `max_bytes`, optional `valid_cell_mask`,
    `builder`) additionally builds the sparse canopy CSR of `leaf.plant_pet` in the same cold rebuild. Pure: the inputs
    are not modified; failure returns nothing.'''
    if backend not in BACKENDS:
        raise BioCalendarError(f'backend must be one of {BACKENDS}, got {backend!r}')
    for name, v in (('max_edges', max_edges), ('max_bytes', max_bytes), ('reserve_bytes', reserve_bytes)):
        try:
            _traits.strict_int(v, name, 0)
        except _traits.CatalogError as exc:
            raise BioCalendarError(str(exc)) from exc
    if not isinstance(base, _state.PlantState):
        raise BioCalendarError('base must be a leaf.state.PlantState')
    rows = (np.flatnonzero(base.status == _state.LIVE).astype(np.int64) if live_rows is None
            else np.asarray(live_rows, dtype=np.int64))
    Pl, Pa = int(rows.shape[0]), int(base.n)
    cuda = backend == 'cuda'
    mem = {}
    peak = MATERIALIZE_BYTES_PER_PLANT * Pa + 16 * Pl + (16 * Pl if cuda else 0)
    mem['materialize'] = int(reserve_bytes) + peak
    _admit('materialize', mem['materialize'], max_bytes)
    cand_bytes = ROW_BYTES_PER_PLANT * Pa + (16 * Pl if cuda else 0)
    mem['geometry'] = (int(reserve_bytes) + cand_bytes + GEOMETRY_BYTES_PER_PLANT * Pa
                       + VALIDATE_SCRATCH_BYTES_PER_PLANT * Pa + 16 * Pl)
    _admit('geometry', mem['geometry'], max_bytes)
    external = int(reserve_bytes) + cand_bytes + 8 * Pl
    budget = int(max_bytes) - external
    if budget < 0:
        raise BioCalendarAdmissionError('builder', external, max_bytes)
    mem['builder_budget'], mem['builder_external_bytes'] = int(budget), int(external)
    cand = materialize_state(base, rows, B_live, E_live, elapsed_days=elapsed_days)
    geo = _canopy.derive_geometry(cand, catalog)
    radius = np.ascontiguousarray(geo['radius_m'][rows], dtype=np.float64)
    del geo
    area_h = np.ascontiguousarray(np.pi * radius * radius, dtype=np.float64)
    if cuda:
        import cupy as cp
        area = cp.asarray(area_h)
        del area_h
    else:
        area = area_h
    prov = {'biocalendar_version': VERSION}
    if provenance:
        prov['caller'] = dict(provenance)
    rm = _roots.build_root_map(cand, catalog, max_edges=int(max_edges), max_bytes=int(budget), backend=backend,
                               builder=builder, statuses=('live',),
                               layer_fraction_overrides=None if layer_fraction_overrides is None
                               else dict(layer_fraction_overrides), provenance=prov)
    ids = rm.host_arrays()['ids']
    if not np.array_equal(ids, np.ascontiguousarray(cand.ids[rows], dtype=np.int64)):
        raise BioCalendarError('root map rows do not match the live-plant row mapping (ids differ)')
    cmap = None
    if canopy is not None:
        from leaf import plant_pet as _pp
        if not isinstance(canopy, dict):
            raise BioCalendarError('canopy must be a dict of canopy-map options or None')
        unknown = set(canopy) - {'max_edges', 'max_bytes', 'valid_cell_mask', 'builder'}
        if unknown:
            raise BioCalendarError(f'unknown canopy options {sorted(unknown)}')
        # MEMORY-ACCOUNTING EXTENSION of the C1 file (revision 3; no scientific kernel changed): the canopy builder shares the GLOBAL
        # `max_bytes` with everything already live: the caller reserve, the candidate PlantState, the just-built root map and the
        # canopy-area array. Its own cap is min(requested canopy cap, global cap) and its reserve is that live total, so both the
        # requested canopy limit AND the remaining global limit are enforced.
        live_canopy = int(reserve_bytes) + cand_bytes + int(rm.resident_bytes) + 8 * Pl
        mem['canopy_reserve_bytes'] = int(live_canopy)
        cmap = _pp.canopy_map_from_state(cand, catalog, live_rows=rows, backend=backend,
                                         max_edges=canopy['max_edges'], max_bytes=min(int(canopy['max_bytes']), int(max_bytes)),
                                         valid_cell_mask=canopy.get('valid_cell_mask'),
                                         builder=canopy.get('builder', builder), reserve_bytes=live_canopy)
    return GeometryRebuild(rm, area, cand, cmap, mem)


__all__ = ['VERSION', 'MAX_CADENCE', 'TRANSPORT_KINDS', 'BioCalendarError', 'BioCalendarStateError',
           'BioCalendarBoundaryError', 'BioCalendarAdmissionError', 'next_new_year', 'year_days', 'advance_boundary',
           'elapsed_days', 'Schedule', 'new_schedule', 'boundaries', 'split_record', 'unsplit_reason',
           'check_interval', 'OpenStep', 'open_step', 'TransportStep', 'transport_step', 'pause_step', 'needs_flush',
           'flush_required', 'flush_step', 'BioCalendarDeviceError', 'estimate_credit_bytes', 'credit_increment', 'materialize_state', 'GeometryRebuild', 'rebuild_geometry']
