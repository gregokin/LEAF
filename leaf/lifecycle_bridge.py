'''leaf.lifecycle_bridge: THIN read/validate bridge between the LEAF daily plant session and the SYRUP Phase 1.5 lifecycle.

Scope (UNQUALIFIED, UNCALIBRATED). SYRUP owns ALL physical water. This module never debits, advances, copies or stores water and never
calls any hydrology, evaporation, standalone withdrawal (`syrup_water.withdraw_plants`, native `withdraw_transpiration`) or daily
operation. Every physical call (`dw.daily_water_action` / `dw.finish_daily_storm`, `wl.request_withdrawal`) stays VISIBLE at the MASY call site:

  1. `read_availability(source, lop, parent, max_bytes=...)`  a pure read (`lifecycle_availability.layer_availability_m`) converted ONCE to m3
     with the native `lop.column.cell_area_m2`. The read carries a LIVE capability: a weak reference to the exact parent INSTANCE and the
     `ExpectedParent` minted from it at read time. These are never serialized.
  2. `request_args(read, lop, parent)`  the keyword arguments MASY passes to its visible `wl.request_withdrawal(lop, parent, **args, depth_m=..,
     identity=.., retained_feedback_bytes=..)`: the capability tied to the READ, so SYRUP itself refuses another physical branch (two real
     daily transitions from one parent have equal digests but different instances).
  3. `accepted_withdrawal(read, identity, new_state, candidate, lop=..., max_bytes=...)` validates the ACTUAL result: the candidate is the exact
     native result of THAT successor (`new_state.payload is candidate.state`), the successor's unchanged leaves (surface, faces, wetting history,
     bed) are the read parent's own arrays, the native receipt echoes the parent token / revision / sequence / identity, the clock and hydrology
     count are unchanged, the request sequence and T count advanced once, and actual / unmet depth are finite non-negative typed arrays whose volume
     (depth * area, once) closes against the native budget bound. Nothing here debits anything.

Units. availability / actual / unmet depth: m of water per cell area; volumes: m3 = depth * `cell_area_m2`; `t_s` is native physical seconds and
is NEVER a biological date. Shapes: L layers, N cells (flat `row * nx + col`), R = L * N layer-major. Arrays live in `lop.xp` (NumPy for numpy /
numba, CuPy on the column's device for CUDA); nothing is transferred. Conversion buffers (`estimate_conversion_bytes`) are admitted against an
explicit `max_bytes` (+ `reserve_bytes`, the caller's other live bytes) BEFORE the arrays exist.

Run / ordinal identity. `ResourceIdentity` records the daily run digest, the LAST consumed daily ordinal (`cursor.ordinal - 1`; None before the first
occurrence) and whether a storm is pending, so a session can map a nominal date to a SYRUP occurrence (`origin_date + ordinal days`) and refuse a
read from another day or run. Native seconds never enter that mapping.

Source selection. `bind_source()` COLD-verifies the imported SYRUP package against the qualified Phase 1.5 digest with the package's own
`provenance.source_tree_digest` (once; never per day). The old `syrup_water` pin (c76c603) stays valid for the standalone drivers.
'''
from __future__ import annotations

import dataclasses
import math
import os
import weakref

import numpy as np

VERSION = '0.2.0'
QUALIFIED_SYRUP_PACKAGE_SHA256 = '82e9c0111b9d883f857fd2f28a21f1d8bd8f13e22d473796e736ca4204f30107'
QUALIFIED_SYRUP_COMMIT = 'a5b8b18358e0d014fe13c9f48d00eb582710655a'
QUALIFIED_SYRUP_BRANCH = 'feature/subsurface-daily-recharge'
MAX_IDENTITY_CHARS = 128
FP64_EPS = float(np.finfo(np.float64).eps)
REDUCTION_ALLOWANCE_BYTES = 4096          # declared allowance for backend reduction scratch (an estimate, like SYRUP's)
_HEX = frozenset('0123456789abcdef')
UNCHANGED_LEAVES = ('surface_m', 'qx_m2_s', 'qy_m2_s', 'history', 'bed')


class BridgeError(ValueError):
    '''Any bridge validation error (raised before a result exists; nothing is mutated).'''


class BridgeSourceError(BridgeError):
    '''The SYRUP source is not the qualified Phase 1.5 package.'''


class BridgeResultError(BridgeError):
    '''The actual withdrawal result is not the exact, single, successor transition of the read parent.'''


class BridgeAdmissionError(BridgeError):
    '''A conversion buffer would exceed the explicit byte cap; raised before any array exists.'''


def _hex64(v, name):
    if type(v) is not str or len(v) != 64 or not set(v) <= _HEX:
        raise BridgeError(f'{name} must be a lowercase 64-hex SHA-256 string')
    return v


def _int(v, name, lo=0):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or int(v) < lo:
        raise BridgeError(f'{name} must be an integer >= {lo}, got {v!r}')
    return int(v)


def _real(v, name, *, positive=False, nonneg=False):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
        raise BridgeError(f'{name} must be a real number, got {v!r}')
    f = float(v)
    if not math.isfinite(f) or (positive and not f > 0.0) or (nonneg and f < 0.0):
        raise BridgeError(f'{name} is not an admissible finite real: {v!r}')
    return f


def debit_identity(label):
    '''Validated SYRUP request identity: printable ASCII, 1..128 characters (an opaque bounded label, not provenance).'''
    if type(label) is not str or not 1 <= len(label) <= MAX_IDENTITY_CHARS or not label.isascii() or not label.isprintable():
        raise BridgeError(f'identity must be a printable ASCII string of 1..{MAX_IDENTITY_CHARS} characters, got {label!r}')
    return label


def estimate_conversion_bytes(n_layers, n_cells, *, operation):
    '''Fresh bytes of one bridge conversion (pure arithmetic; the borrowed inputs are not counted). `read`: the native availability helper's
    own peak (`lifecycle_availability.required_availability_bytes`, caller-added) is separate; here the native depth (8LN) and the volume
    (8LN) coexist. `request`: the (L,N) depth output 8LN. `accept`: actual and unmet volumes 16LN. Plus the declared reduction allowance.'''
    ln = 8 * _int(n_layers, 'n_layers') * _int(n_cells, 'n_cells')
    fresh = {'read': 2 * ln, 'request': ln, 'accept': 2 * ln}
    if operation not in fresh:
        raise BridgeError(f'operation must be one of {sorted(fresh)}')
    return {'operation': operation, 'fresh_bytes': fresh[operation], 'allowance_bytes': REDUCTION_ALLOWANCE_BYTES,
            'total_bytes': fresh[operation] + REDUCTION_ALLOWANCE_BYTES}


def _admit(need, max_bytes, reserve_bytes, what):
    mb, rb = _int(max_bytes, 'max_bytes'), _int(reserve_bytes, 'reserve_bytes')
    if rb + int(need) > mb:
        raise BridgeAdmissionError(f'{what}: reserve {rb} + estimated {int(need)} bytes exceed max_bytes={mb}; refused before allocation')


# ------------------------------------------------------------------ typed value objects
@dataclasses.dataclass(frozen=True)
class ResourceIdentity:
    '''Serializable scalar identity of one accepted SYRUP lifecycle state (NOT a live capability).'''
    lop_sha256: str
    parent_digest: str
    revision: int
    request_sequence: int
    hydro_count: int
    n_requests: int
    t_s: float
    role: str = 'interevent'
    daily_ordinal: object = None          # the LAST consumed daily ordinal (cursor.ordinal - 1); None before the first occurrence / no cursor
    daily_run: object = None              # daily run digest (64 hex) or None
    daily_pending: bool = False

    def __post_init__(self):
        _hex64(self.lop_sha256, 'lop_sha256')
        _hex64(self.parent_digest, 'parent_digest')
        for n in ('revision', 'request_sequence', 'hydro_count', 'n_requests'):
            _int(getattr(self, n), n)
        object.__setattr__(self, 't_s', _real(self.t_s, 't_s', nonneg=True))
        if self.role not in ('boundary', 'storm', 'interevent'):
            raise BridgeError(f'role must be boundary / storm / interevent, got {self.role!r}')
        if self.daily_ordinal is not None:
            _int(self.daily_ordinal, 'daily_ordinal')
        if self.daily_run is not None:
            _hex64(self.daily_run, 'daily_run')
        if not isinstance(self.daily_pending, bool):
            raise BridgeError('daily_pending must be a bool')

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d):
        if not isinstance(d, dict) or set(d) != {f.name for f in dataclasses.fields(cls)}:
            raise BridgeError('resource identity record keys mismatch (closed key set)')
        return cls(**d)


@dataclasses.dataclass(frozen=True)
class AvailabilityRead:
    '''ONE fresh availability read (volume form): `supply_m3` (L, N) = native availability depth * `cell_area_m2` (the single conversion).
    `_parent_ref` / `_expected` / `_lop` are LIVE capabilities (never serialized, excluded from equality); a read rebuilt from a checkpoint
    has none and can never be used to validate a NEW withdrawal.'''
    identity: ResourceIdentity
    supply_m3: object
    cell_area_m2: float
    n_layers: int
    n_cells: int
    backend: str
    _parent_ref: object = dataclasses.field(default=None, compare=False, repr=False)
    _expected: object = dataclasses.field(default=None, compare=False, repr=False)
    _lop: object = dataclasses.field(default=None, compare=False, repr=False)

    @property
    def is_live(self):
        return self._parent_ref is not None and self._expected is not None and self._parent_ref() is not None


@dataclasses.dataclass(frozen=True)
class AcceptedWithdrawal:
    '''Validated actual result of ONE `wl.request_withdrawal`: `actual_m3` / `unmet_m3` (L, N) = native depth * the same area (single conversion).'''
    identity: str
    before: ResourceIdentity
    after: ResourceIdentity
    actual_m3: object
    unmet_m3: object
    requested_m3: float
    actual_total_m3: float
    unmet_total_m3: float
    cell_area_m2: float


# ------------------------------------------------------------------ source selection (cold, once)
@dataclasses.dataclass(frozen=True)
class BridgeSource:
    '''Record of the COLD verification of the imported SYRUP package (directory + tree digest).'''
    package_sha256: str
    commit: str
    branch: str
    package_dir: str
    module_files: tuple


def _modules():
    try:
        from maple_syrup import lifecycle_availability as lal
        from maple_syrup import water_lifecycle as wl
    except Exception as exc:                                   # pragma: no cover - environment dependent
        raise BridgeSourceError(f'the SYRUP Phase 1.5 package is not importable: {exc}') from exc
    return wl, lal


def bind_source(evidence_sha256=None):
    '''Cold-check the IMPORTED SYRUP package: `maple_syrup.provenance.source_tree_digest` over the package directory must equal
    `QUALIFIED_SYRUP_PACKAGE_SHA256` (this is the package's own existing helper and the algorithm of `verify_phase15_source.py`). The optional
    `evidence_sha256` (e.g. `profile.syrup_package_digest`) must agree. Run it ONCE per process / run, never per day.'''
    if evidence_sha256 is not None:
        _hex64(evidence_sha256, 'evidence_sha256')
        if evidence_sha256 != QUALIFIED_SYRUP_PACKAGE_SHA256:
            raise BridgeSourceError(f'SYRUP package evidence {evidence_sha256} is not the qualified Phase 1.5 package')
    wl, lal = _modules()
    try:
        import maple_syrup
        from maple_syrup.provenance import source_tree_digest
    except Exception as exc:                                   # pragma: no cover
        raise BridgeSourceError(f'the SYRUP provenance helper is unavailable: {exc}') from exc
    pdir = os.path.dirname(os.path.realpath(maple_syrup.__file__))
    digest = source_tree_digest(pdir).digest_sha256
    if digest != QUALIFIED_SYRUP_PACKAGE_SHA256:
        raise BridgeSourceError(f'the imported SYRUP package ({pdir}) has tree digest {digest}, not the qualified Phase 1.5 '
                                f'{QUALIFIED_SYRUP_PACKAGE_SHA256}; the old c76c603 pin is not rebound by this bridge')
    for mod, names in ((wl, ('expect_parent', 'request_withdrawal', 'LifecycleState')), (lal, ('layer_availability_m',))):
        missing = [n for n in names if not hasattr(mod, n)]
        if missing:
            raise BridgeSourceError(f'{mod.__name__} lacks {missing}')
    return BridgeSource(digest, QUALIFIED_SYRUP_COMMIT, QUALIFIED_SYRUP_BRANCH, pdir,
                        (os.path.realpath(wl.__file__), os.path.realpath(lal.__file__)))


def _check_source(source):
    '''Cheap per-call guard (no hashing): a verified BridgeSource whose modules are still the imported files.'''
    if not isinstance(source, BridgeSource) or source.package_sha256 != QUALIFIED_SYRUP_PACKAGE_SHA256:
        raise BridgeSourceError('source must be the BridgeSource returned by bind_source()')
    wl, lal = _modules()
    if (os.path.realpath(wl.__file__), os.path.realpath(lal.__file__)) != source.module_files:
        raise BridgeSourceError('the imported SYRUP modules are not the files verified by bind_source(); refused')


# ------------------------------------------------------------------ identity / read
def resource_identity(lop, parent):
    '''Scalars of an accepted lifecycle state (metadata only; no array read). `lop` supplies the operator hash.'''
    wl, _ = _modules()
    if type(parent) is not wl.LifecycleState:
        raise BridgeError(f'parent must be a SYRUP LifecycleState, got {type(parent).__name__}')
    if type(lop) is not wl.LifecycleOperator:
        raise BridgeError(f'lop must be a genuine SYRUP LifecycleOperator, got {type(lop).__name__}')
    wl.expect_parent(parent)                    # genuine minted state (SYRUP's own check)
    if parent.lop_sha256 != lop.sha256:
        raise BridgeError('the state is bound to another lifecycle operator than the passed lop (operator association refused)')
    d = getattr(parent, 'daily', None)
    return ResourceIdentity(parent.lop_sha256, parent.digest, int(parent.revision), int(parent.request_sequence), int(parent.hydro_count),
                            int(parent.n_requests), float(parent.t_s), str(parent.role),
                            None if d is None or int(d.ordinal) == 0 else int(d.ordinal) - 1,
                            None if d is None else str(d.run_digest), bool(d is not None and d.pending is not None))


def read_availability(source, lop, parent, *, max_bytes, reserve_bytes=0, extra_reserve_bytes=0):
    '''PURE read: fresh availability of the accepted interevent `parent` as volumes (`depth * lop.column.cell_area_m2`, once). The native helper's
    own plan plus the conversion buffers are admitted against `max_bytes` (+ `reserve_bytes`) BEFORE the read. Requires an interevent parent that
    already passed the MASY-visible daily action; reads no time, creates no state; mints the LIVE capability bound to this exact instance.'''
    _check_source(source)
    wl, lal = _modules()
    if parent.role != 'interevent':
        raise BridgeError(f'availability needs an interevent parent, got role {parent.role!r}')
    L, N = int(lop.n_layers), int(lop.n_cells)
    plan = lal.required_availability_bytes(lop, extra_reserve_bytes=extra_reserve_bytes)
    conv = estimate_conversion_bytes(L, N, operation='read')
    _admit(max(int(plan.required_bytes), conv['total_bytes']), max_bytes, reserve_bytes, 'availability read')
    ident = resource_identity(lop, parent)
    d = getattr(parent, 'daily', None)
    if d is None or d.pending is not None or int(d.ordinal) < 1:
        raise BridgeError('a water-connected read needs a state of a genuine COMPLETED daily lineage (a daily cursor with at least one consumed '
                          'occurrence and no pending storm); legacy / lineage-free interevent states are refused')
    expected = wl.expect_parent(parent)
    depth = lal.layer_availability_m(lop, parent, expected_parent=expected, extra_reserve_bytes=extra_reserve_bytes)
    area = _real(lop.column.cell_area_m2, 'cell_area_m2', positive=True)
    supply = depth * area                                  # the single depth -> volume conversion (fresh array)
    del depth
    return AvailabilityRead(ident, supply, area, L, N, str(lop.backend), weakref.ref(parent), expected, lop)


def request_args(read, lop, parent):
    '''Keyword arguments for the MASY-visible `wl.request_withdrawal(lop, parent, **request_args(read, lop, parent), depth_m=..., identity=...,
    retained_feedback_bytes=...)`: the capability minted at READ time. Refused unless `parent` is the very instance that was read and the read
    belongs to this operator, so a request cannot be aimed at another physical branch by minting a new capability.'''
    if not isinstance(read, AvailabilityRead) or not read.is_live:
        raise BridgeError('read must be a live AvailabilityRead of this process (a restored / dead read has no capability)')
    if read._parent_ref() is not parent or read._expected._ref() is not parent:
        raise BridgeResultError('the parent is not the instance that was read (another physical branch); refused')
    if read._lop is not lop:
        raise BridgeResultError('the read belongs to another lifecycle operator; refused')
    return {'expected_parent': read._expected}


def request_depth_m(read, proposed_m3, *, max_bytes, reserve_bytes=0):
    '''LEAF proposal volumes [(R,) or (L, N)] -> the (L, N) depth for `wl.request_withdrawal` (divided by the SAME area as the read, once; one
    fresh C-contiguous float64 array in the proposal's namespace, admitted first).'''
    if not isinstance(read, AvailabilityRead):
        raise BridgeError('read must be an AvailabilityRead')
    a = proposed_m3
    if tuple(a.shape) not in ((read.n_layers * read.n_cells,), (read.n_layers, read.n_cells)):
        raise BridgeError(f'proposal shape {tuple(a.shape)} does not match (L, N) = ({read.n_layers}, {read.n_cells})')
    _admit(estimate_conversion_bytes(read.n_layers, read.n_cells, operation='request')['total_bytes'], max_bytes, reserve_bytes,
           'request depth conversion')
    return (a / read.cell_area_m2).reshape(read.n_layers, read.n_cells)


# ------------------------------------------------------------------ result validation
def _scalar_min_max(a):
    return float(a.min()), float(a.max())


def accepted_withdrawal(read, identity, new_state, candidate, *, lop, max_bytes, reserve_bytes=0):
    '''Validate the ACTUAL result `(new_state, candidate) = wl.request_withdrawal(...)` against the LIVE read it was prepared from (see module
    docstring) and return the volume feedback. Refused (typed `BridgeResultError`, nothing mutated) unless every check passes:
      * the read is live and its parent instance is still the object the capability names;
      * `new_state.payload is candidate.state` and the unchanged physical leaves are the parent's own arrays;
      * the native receipt echoes the debit identity, the parent token (payload digest), revisions and request sequence;
      * same operator / role, ZERO time (clock and hydrology count unchanged), request sequence and T count advanced exactly once, new digest;
      * actual / unmet are finite non-negative float64 (L, N) arrays and actual + unmet closes against the receipt and the native budget bound.'''
    if not isinstance(read, AvailabilityRead):
        raise BridgeError('read must be an AvailabilityRead')
    if not read.is_live:
        raise BridgeResultError('the read has no live capability (restored or its parent is gone): cannot validate a new withdrawal')
    identity = debit_identity(identity)
    parent = read._parent_ref()
    if read._expected._ref() is not parent or read._lop is not lop:
        raise BridgeResultError('the read capability does not name its parent / operator; refused')
    if resource_identity(lop, parent) != read.identity:
        raise BridgeResultError('the read parent changed since the read (scalars differ); refused')
    pl, cs = getattr(new_state, 'payload', None), getattr(candidate, 'state', None)
    if pl is None or pl is not cs:
        raise BridgeResultError('the candidate is not the native result of this successor state (new_state.payload is not candidate.state): '
                                'feedback of another withdrawal cannot be paired with this state')
    ppl = parent.payload
    for name in UNCHANGED_LEAVES:
        if getattr(pl, name, None) is not getattr(ppl, name, None):
            raise BridgeResultError(f'the successor does not share the read parent\'s unchanged {name!r} array: it descends from another '
                                    'physical branch; refused')
    after, b = resource_identity(lop, new_state), read.identity
    if after.lop_sha256 != b.lop_sha256 or after.role != 'interevent':
        raise BridgeResultError('the result belongs to another operator or is not an interevent state')
    if after.t_s != b.t_s or after.hydro_count != b.hydro_count:
        raise BridgeResultError(f'the withdrawal must be ZERO time: clock {b.t_s!r} -> {after.t_s!r}, hydro count '
                                f'{b.hydro_count} -> {after.hydro_count}')
    if after.request_sequence != b.request_sequence + 1 or after.n_requests != b.n_requests + 1:
        raise BridgeResultError(f'T request sequence/count must advance exactly once: ({b.request_sequence}, {b.n_requests}) -> '
                                f'({after.request_sequence}, {after.n_requests})')
    if after.revision <= b.revision or after.parent_digest == b.parent_digest:
        raise BridgeResultError('the result is not a new revision of the read parent')
    if (after.daily_run, after.daily_ordinal, after.daily_pending) != (b.daily_run, b.daily_ordinal, b.daily_pending):
        raise BridgeResultError('the daily occurrence cursor must be unchanged by a withdrawal')
    rec = getattr(candidate, 'receipt', None)
    if (rec is None or rec.identity != identity or int(rec.sequence) != b.request_sequence
            or rec.parent_token != ppl.digest or int(rec.revision_before) != int(ppl.revision)
            or int(rec.revision_after) != int(cs.revision) or float(rec.t_s) != b.t_s):
        raise BridgeResultError('the native receipt does not echo this debit identity / parent token / revision / read request sequence')
    actual, unmet = candidate.actual_m, candidate.unmet_m
    shp = (read.n_layers, read.n_cells)
    for name, arr in (('actual_m', actual), ('unmet_m', unmet)):
        if tuple(arr.shape) != shp or arr.dtype != np.float64 or type(arr).__module__.split('.')[0] != type(read.supply_m3).__module__.split('.')[0]:
            raise BridgeResultError(f'{name} must be a float64 {shp} array of the read namespace')
        lo, hi = _scalar_min_max(arr)
        if not (math.isfinite(lo) and math.isfinite(hi) and lo >= 0.0):
            raise BridgeResultError(f'{name} must be finite and non-negative')
    _admit(estimate_conversion_bytes(read.n_layers, read.n_cells, operation='accept')['total_bytes'], max_bytes, reserve_bytes,
           'accepted volume conversion')
    actual_m3 = actual * read.cell_area_m2                 # the single volume conversion of the native actual depth
    unmet_m3 = unmet * read.cell_area_m2
    at, ut = float(actual_m3.sum()), float(unmet_m3.sum())
    req = float(rec.requested_m3)
    bound = float(getattr(getattr(candidate, 'budget', None), 'bound_m3', 0.0)) + 8.0 * FP64_EPS * max(abs(req), abs(at), abs(ut), 1e-300) \
        * max(read.n_layers * read.n_cells, 1)
    if not (math.isfinite(at) and math.isfinite(ut) and abs((at + ut) - req) <= bound and abs(at - float(rec.actual_m3)) <= bound):
        raise BridgeResultError('actual + unmet (or actual) does not close against the native receipt within the native budget bound')
    return AcceptedWithdrawal(identity, b, after, actual_m3, unmet_m3, req, at, ut, read.cell_area_m2)


def pet_m_s_to_mm_per_day(pet_m_s):
    '''Fresh `mm / day` array from a SYRUP terrain PET `m / s` field (a pure unit conversion owned by the caller; the SAME field feeds SYRUP).'''
    return pet_m_s * (86400.0 * 1000.0)


__all__ = ['VERSION', 'QUALIFIED_SYRUP_PACKAGE_SHA256', 'QUALIFIED_SYRUP_COMMIT', 'QUALIFIED_SYRUP_BRANCH', 'BridgeError',
           'BridgeSourceError', 'BridgeResultError', 'BridgeAdmissionError', 'ResourceIdentity', 'AvailabilityRead',
           'AcceptedWithdrawal', 'BridgeSource', 'debit_identity', 'estimate_conversion_bytes', 'bind_source', 'resource_identity',
           'read_availability', 'request_args', 'request_depth_m', 'accepted_withdrawal', 'pet_m_s_to_mm_per_day']
