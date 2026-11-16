"""Source-bound Dense CUDA bridge from genuine floating daily water to PlantSession.

AUTHOR-WRITTEN, NOT RUN ON A DEVICE: no GPU, CuPy or native package was executed by
the author; Root runs the real CPU/GPU multi-day controls. Importing this module never
imports CuPy; CuPy is reached only through the genuine native probes
(`floating_soil_layers_cuda._rc().require_cuda()`) inside bound calls.

Same protocol and API shape as `leaf.floating_lifecycle_bridge` (the strict default CPU
bridge, which is unchanged and refuses every record of this module): bind_source ->
connect -> begin_day -> read_availability -> request_args / request_depth_m -> the
MASY-visible native `request_floating_withdrawal` -> accepted_withdrawal -> PlantSession
stage/publish -> the MASY-visible native close/hand_back -> commit_day. The shared
protocol, registries and genuine LEAF resource types (`lifecycle_bridge.AvailabilityRead`,
`AcceptedWithdrawal`, `ResourceIdentity`) are those of the CPU module; only the device
pieces differ:

  * Source: a genuine `NativeSourceProfile` WITH device_runtime_capability=True naming
    the explicitly supplied committed Source pair (no digest is hardcoded here); the CPU
    proof roster plus the owning native CUDA modules (`floating_daily_cuda`,
    `floating_transpiration_cuda`, `floating_soil_layers_cuda`,
    `floating_event_daily_bridge_cuda`, `spatial_transpiration_cuda`,
    `soil_water_remap_cuda`) by exact origin/function code/object identity.
  * Residency: availability, request depth and actual/unmet volumes are CuPy arrays on
    the native column's device. No `np.asarray`, `.get()`, `asnumpy`, full-W download
    or host mirror occurs in any bridge path; per-resource validation runs as resident
    row algebra and the genuine native active-sum kernel (`floating_soil_layers_cuda.
    storage_depth`), and the host receives only bounded 8-byte scalar packets through
    the native counted reader (`floating_event_daily_bridge_cuda.scalar`).
  * Admission: TWO separate pools (`CudaPools`): LEAF's explicit host cap/reserve
    (`lifecycle_bridge._admit`) and the genuine native two-pool admission
    (`floating_daily_cuda.admit`: device reader/budget/reserve, host reader/budget/
    reserve). Pools are never summed and device bytes are never counted as host bytes.
    Caps are explicit finite integers; readers are required real callables.
  * Order of every device call: current-device gate FIRST (`floating_daily_cuda.gate`),
    metadata (`floating_soil_layers._check_vec`: exact cupy type/dtype/contiguity/
    shape/device), admission, then the work.

The comparison law is unchanged: the native `conservation.volume_roundoff_bound_m3`
with the owning withdrawal term count and the genuine before/request reservoir
operands. No epsilon, arithmetic law, soil store, second debit or W0 conversion is
introduced. Combined LEAF/wind/water persisted restart remains explicitly unqualified.
"""
from __future__ import annotations

import dataclasses
import math
import types

import numpy as np

from . import floating_lifecycle_bridge as fb
from . import lifecycle_bridge as lb

VERSION = '1'
PROFILE = 'leaf.floating_lifecycle_bridge.cuda.v1'
BACKEND = 'cuda'
VALIDATION_ROW_DOUBLES = 6   # declared (N,) float64 row temporaries live during one layer of resident validation
VALIDATION_ROW_FLAGS = 6     # declared (N,) bool row temporaries (finite/sign/inactive masks) of one layer
FLAG_WORDS = 8               # resident refusal flags (bool) packed into ONE int64 scalar read
SCALAR_READS_PER_CALL = 16   # upper bound of counted 8-byte device-to-host scalar reads of one bridge call
HOST_PACKET_BYTES = 8 * SCALAR_READS_PER_CALL

FloatingBridgeError = fb.FloatingBridgeError
FloatingSourceError = fb.FloatingSourceError
FloatingCursorError = fb.FloatingCursorError
BridgeCursor = fb.BridgeCursor
COLD_SOURCE_ALLOWANCE_BYTES = fb.COLD_SOURCE_ALLOWANCE_BYTES

_CUDA_MODULE_NAMES = ('floating_daily_cuda', 'floating_transpiration_cuda', 'floating_soil_layers_cuda',
                      'floating_event_daily_bridge_cuda', 'spatial_transpiration_cuda', 'soil_water_remap_cuda')
_CUDA_FUNCTION_NAMES = {
    'floating_soil_layers': fb._FUNCTION_NAMES['floating_soil_layers'] + ('_ns_type', '_check_water', '_gate'),
    'subsurface_transpiration': fb._FUNCTION_NAMES['subsurface_transpiration'] + ('render_withdrawal_source',),
    'floating_daily_cuda': ('gate', 'quota', 'component_quota', 'admit', 'start_plan', 'day_plan',
                            'withdrawal_plan', 'audit_plan', 'check_state_resident', 'zeros'),
    'floating_transpiration_cuda': ('render_source', 'kernel_source', 'kernel_provenance', 'load_kernels',
                                    'n_blocks', 'check_launch_ranges', 'operation_terms', '_map_args',
                                    '_retarget', 'withdraw', 'availability'),
    'floating_soil_layers_cuda': ('_rc', 'current_device', 'device_reader', 'n_blocks', 'check_launch_ranges',
                                  'load_kernels', 'operation_terms', '_read_words', 'state_check',
                                  'storage_depth', 'gate_device', '_count', 'admit', 'raise_flags'),
    'floating_event_daily_bridge_cuda': ('_bridge', 'scalar', 'sum_cuda', 'inputs_ok_cuda', 'depth_ok_cuda',
                                         'wet_quota', 'wet_plan_cuda', 'admit_wet', 'open_plan_cuda', 'open_day_cuda'),
    'spatial_transpiration_cuda': ('n_blocks', 'check_launch_ranges', 'operation_terms'),
    'soil_water_remap_cuda': ('require_cuda', 'current_device_id', 'effective_free_device_bytes',
                              'default_device_available_bytes_fn'),
}
_CUDA_OBJECT_NAMES = {
    'subsurface_transpiration': fb._OBJECT_NAMES['subsurface_transpiration'] + ('_CUDA_BODY', '_FLAG_TEXT'),
    'floating_daily_cuda': ('STORAGE_PACKET_BYTES',),
    'floating_transpiration_cuda': ('FLOATING_READERS', 'FLOATING_HEAD', 'FLOATING_TAIL', '_FUNCTIONS',
                                    'COMPILE_OPTIONS', 'THREADS', 'FINAL_THREADS', 'PACKET_BYTES', 'ROWS'),
    'floating_soil_layers_cuda': ('_FUNCTIONS', 'QUOTA_DEFAULTS', 'PY_ALLOWANCE_BYTES', 'STATE_FLAG_TEXT', 'THREADS'),
    'spatial_transpiration_cuda': ('SPATIAL_HEAD', 'SPATIAL_TAIL', 'COMPILE_OPTIONS', 'THREADS', 'FINAL_THREADS',
                                   '_MAP_PARAMS', '_STRIDES'),
}
_CUDA_MODULE_ALIASES = (('floating_daily', 'fdc', 'floating_daily_cuda'),
                        ('floating_daily_cuda', 'fcu', 'floating_soil_layers_cuda'),
                        ('floating_daily_cuda', 'ftc', 'floating_transpiration_cuda'),
                        ('floating_daily_cuda', 'fsl', 'floating_soil_layers'),
                        ('floating_transpiration_cuda', 'stc', 'spatial_transpiration_cuda'),
                        ('floating_transpiration_cuda', 'st', 'subsurface_transpiration'),
                        ('floating_event_daily_bridge_cuda', 'fdc', 'floating_daily_cuda'),
                        ('floating_event_daily_bridge_cuda', 'fd', 'floating_daily'),
                        ('floating_event_daily_bridge_cuda', 'fsl', 'floating_soil_layers'))


class CudaFloatingSource(fb.FloatingSource):
    """Genuine CUDA-capability NativeSourceProfile binding; refused by the CPU module."""
    __slots__ = ()


class CudaFloatingConnection(fb.FloatingConnection):
    __slots__ = ()


@dataclasses.dataclass(frozen=True)
class CudaPools:
    """Explicit two-pool admission inputs of one CUDA bridge call (never summed).

    device_max_bytes / device_reserve_bytes / device_available_bytes_fn: the DEVICE pool
    (native `memory_budget_bytes` / `reserve_bytes` / `available_bytes_fn`).
    host_max_bytes / host_reserve_bytes / host_available_bytes_fn: the HOST pool (LEAF's
    explicit cap/reserve of `lifecycle_bridge._admit` AND the native
    `host_memory_budget_bytes` / `host_reserve_bytes` / `host_available_bytes_fn`).
    The reserves price every OTHER live byte of the caller in that pool (parents,
    proposals, native request outputs, plant arrays); a reader returns that pool's
    actually available bytes. Readers are required (no default pool, no fallback).
    """
    device_max_bytes: int
    device_available_bytes_fn: object
    host_max_bytes: int
    host_available_bytes_fn: object
    device_reserve_bytes: int = 0
    host_reserve_bytes: int = 0

    def __post_init__(self):
        for name in ('device_max_bytes', 'host_max_bytes'):
            lb._int(getattr(self, name), name, 1)
        for name in ('device_reserve_bytes', 'host_reserve_bytes'):
            lb._int(getattr(self, name), name)
        for name in ('device_available_bytes_fn', 'host_available_bytes_fn'):
            if not callable(getattr(self, name)):
                raise FloatingBridgeError(f'{name} must be a real callable available-bytes reader (no default pool)')

    def quota(self, fdc):
        """The genuine native strict two-pool quota dict (`floating_daily_cuda.quota`)."""
        return fdc.quota(self.device_available_bytes_fn, self.device_reserve_bytes, self.host_available_bytes_fn,
                         self.host_reserve_bytes, self.device_max_bytes, self.host_max_bytes)


def _pools(pools):
    if type(pools) is not CudaPools:
        raise FloatingBridgeError('pools must be a CudaPools record (separate explicit device and host pools)')
    return pools


def estimate_bridge_bytes(n_layers, n_cells, *, operation):
    """Separate DEVICE and HOST payload of one CUDA bridge call (never summed).

    device: the fresh resident conversion output(s) of `lifecycle_bridge.
    estimate_conversion_bytes` (read: supply; request: depth; accept: actual + unmet)
    plus the bounded per-layer validation rows and the flag words. host: the counted
    scalar packets plus LEAF's declared reduction allowance. The native availability /
    withdrawal payloads and the resident active-sum reductions admit separately through
    their own native terms; every other coexisting array belongs in the caller reserves.
    """
    e = lb.estimate_conversion_bytes(n_layers, n_cells, operation=operation)
    N = int(n_cells)
    rows = VALIDATION_ROW_DOUBLES * 8 * N + VALIDATION_ROW_FLAGS * N + 8 * FLAG_WORDS + 64
    return {'operation': operation, 'device_fresh_bytes': e['fresh_bytes'], 'device_validation_bytes': rows,
            'device_bytes': e['fresh_bytes'] + rows, 'host_bytes': HOST_PACKET_BYTES + lb.REDUCTION_ALLOWANCE_BYTES,
            'scope': 'declared payload of this bridge call per pool; native kernels, allocator rounding, context and interpreter excluded'}


# ---------------------------------------------------------------- device helpers (gate first)
def _cupy(m, column):
    """The CURRENT device must be the column's device BEFORE any reader, metadata read,
    copy or kernel (the genuine native gate); returns the CuPy module."""
    try:
        m['floating_daily_cuda'].gate(column)
        return m['floating_soil_layers_cuda']._rc().require_cuda()
    except (ValueError, RuntimeError) as exc:
        raise FloatingBridgeError(f'the CUDA device gate refused: {exc}') from exc


def _vec(m, column, name, a, shape):
    """Genuine native device metadata check: exact cupy.ndarray of the column's device,
    float64, C-contiguous, exact shape. No coercion, no upload, no value read."""
    fsl = m['floating_soil_layers']
    try:
        fsl._check_vec(column, name, a, tuple(shape))
    except fsl.FloatingSoilError as exc:
        raise FloatingBridgeError(f'{name}: {exc}') from exc


def _column(m, column, *, shape=None):
    fsl = m['floating_soil_layers']
    if type(column) is not fsl.FloatingSoilColumn or column.backend != BACKEND:
        raise FloatingBridgeError('this bridge serves genuine Dense CUDA floating columns only; no backend fallback')
    dev = column.device_id
    if isinstance(dev, bool) or not isinstance(dev, int) or dev < 0:
        raise FloatingBridgeError('a CUDA floating column must carry its integer device id')
    cp = _cupy(m, column)
    L, N = int(column.n_layers), int(column.n_cells)
    area = fb._finite(column.cell_area_m2, 'cell_area_m2')
    if L < 1 or N < 1 or area <= 0 or (shape is not None and math.prod(shape) != N):
        raise FloatingBridgeError('native geometry/layer/area metadata is inconsistent')
    mask = column._maps['active']
    if type(mask) is not cp.ndarray or mask.dtype != np.bool_ or tuple(mask.shape) != (N,) or not mask.flags.c_contiguous or int(mask.device.id) != dev:
        raise FloatingBridgeError('the native active mask must be a resident bool [N] CuPy array of the column device')
    return L, N, area


def _root(m, root):
    if root.host_active is None or root.edges_host is None:
        raise FloatingCursorError('a CUDA boundary root must carry its sealed host mask and edges (a boundary minted by the CUDA lifecycle)')


def _admit(m, column, L, N, operation, pools, what):
    """Two separate pools, never summed: LEAF's explicit host cap first (pure arithmetic,
    before any reader), then the genuine native two-pool admission with the real device
    and host readers/budgets/reserves. Native resource refusals keep their own classes
    (`FloatingMemoryError` is a MemoryError; a bad reader value is a FloatingSoilError)."""
    e = estimate_bridge_bytes(L, N, operation=operation)
    lb._admit(e['host_bytes'], pools.host_max_bytes, pools.host_reserve_bytes, what)
    fdc = m['floating_daily_cuda']
    q = pools.quota(fdc)
    fdc.admit(column, e['device_bytes'], e['host_bytes'], q, what)
    return q


def _flag(flags, i, cond):
    """Device-side flag accumulation: flags[i] |= cond (cond a 0-d device bool); no host read."""
    flags[i] = flags[i] | cond


def _finite_rows(cp, flags, bit, a):
    """Finite and non-negative on every value, layer by layer ((N,) row temporaries only)."""
    for l in range(int(a.shape[0])):
        row = a[l]
        ok = cp.isfinite(row)
        ok &= row >= 0.0
        _flag(flags, bit, ~cp.all(ok))


def _code(m, cp, flags):
    """ONE counted 8-byte read of the packed refusal flags (bit i = flags[i])."""
    weights = cp.left_shift(cp.int64(1), cp.arange(int(flags.shape[0]), dtype=cp.int64))
    return int(m['floating_event_daily_bridge_cuda'].scalar(cp.sum(flags.astype(cp.int64) * weights)))


def _finite_or_refuse(m, cp, column, name, a, shape):
    """Metadata, then resident finite/non-negative flags over (L, N) rows ((R,) arrays
    are viewed layer-major as (L, N); no copy), ONE packed scalar read."""
    _vec(m, column, name, a, shape)
    flags = cp.zeros(FLAG_WORDS, dtype=cp.bool_)
    _finite_rows(cp, flags, 0, a.reshape(int(column.n_layers), int(column.n_cells)))
    if _code(m, cp, flags):
        raise FloatingBridgeError(f'{name} has nonfinite/negative values')


def _active_sum_m3(m, cp, column, a, q, area, what):
    """Resident sum over ACTIVE cells through the genuine native storage kernel (block
    tree + one-block finalize + ONE 8-byte read), admitted on both pools first."""
    fcu, fdc = m['floating_soil_layers_cuda'], m['floating_daily_cuda']
    sto = fcu.operation_terms('storage', int(column.n_layers), int(column.n_cells))
    fdc.admit(column, sto['device'], sto['host'], q, what)
    return float(fcu.storage_depth(cp, column, a)) * area


# ---------------------------------------------------------------- backend ops
def _availability(m, ctx, water, L, N, area, pools):
    col = ctx.column
    cp = _cupy(m, col)
    q = _admit(m, col, L, N, 'read', pools, 'floating availability read (CUDA)')
    # Genuine native resident availability: its own gate, metadata, two-pool admission,
    # kernel and ONE 8-byte diagnostic; a fresh (L, N) device array.
    depth = m['floating_transpiration'].floating_layer_availability_m(col, water, **q)
    _vec(m, col, 'availability', depth, (L, N))
    supply = depth * area  # the ONE depth -> volume conversion, resident
    del depth
    _finite_or_refuse(m, cp, col, 'availability volume', supply, (L, N))
    return supply


def _check_proposal(m, column, read, proposed_m3, pools):
    L, N = read.n_layers, read.n_cells
    cp = _cupy(m, column)
    if type(proposed_m3) is not cp.ndarray or tuple(proposed_m3.shape) not in ((L * N,), (L, N)):
        raise FloatingBridgeError('proposal must be the layer-major aggregate volume CuPy array of the native device')
    _vec(m, column, 'proposal', proposed_m3, tuple(proposed_m3.shape))
    _admit(m, column, L, N, 'request', pools, 'floating request conversion (CUDA)')
    _finite_or_refuse(m, cp, column, 'proposal', proposed_m3, tuple(proposed_m3.shape))


def _to_depth(m, column, read, proposed_m3):
    cp = _cupy(m, column)
    request = (proposed_m3 / read.cell_area_m2).reshape(read.n_layers, read.n_cells)  # the ONE volume -> depth conversion, resident
    _finite_or_refuse(m, cp, column, 'request depth', request, (read.n_layers, read.n_cells))
    return request


def _admit_accept(m, column, read, pools):
    _cupy(m, column)
    _admit(m, column, read.n_layers, read.n_cells, 'accept', pools, 'floating accepted feedback (CUDA)')


def _check_proposed(m, column, read, proposed):
    cp = _cupy(m, column)
    _finite_or_refuse(m, cp, column, 'LEAF aggregate proposal', proposed, (read.n_layers * read.n_cells,))


def _root_map(m, rm, read, ctx):
    if (rm.n_layers, rm.n_cells, rm.backend) != (read.n_layers, read.n_cells, BACKEND):
        raise FloatingCursorError('the actual proposal root map differs from the native resource shape/backend')
    if rm.device_id != ctx.column.device_id:
        raise FloatingCursorError('the proposal root map lives on another device than the native column')


_FLAG_TEXT = ((1, FloatingBridgeError, 'a native/LEAF feedback array has nonfinite/negative values'),
              (2, FloatingCursorError, 'actual/unmet or W debit does not close per resource'),
              (4, FloatingCursorError, "the native request differs from LEAF's actual proposal"),
              (8, FloatingCursorError, 'withdrawal violates scale/request or inactive-cell identity'),
              (16, FloatingCursorError, 'the native scale does not reproduce actual withdrawal'))


def _validate_feedback(m, parent, outcome, request, proposed, area, pools):
    """Shared scalar producer/ledger proof, then RESIDENT per-resource and global
    comparisons under the ONLY law: native volume_roundoff_bound_m3 with the owning
    term count and the genuine before/request operands. Row algebra per layer, flags
    accumulated on the device, ONE packed scalar read; active sums through the genuine
    native storage kernel (one 8-byte read each). Nothing of W is downloaded."""
    child, terms = fb._feedback_proof(m, parent, outcome)
    ctx = parent.ctx
    col = ctx.column
    L, N = ctx.n_layers, ctx.n_cells
    shape = (L, N)
    cp = _cupy(m, col)
    w0, w1 = parent.water.water_m, child.water.water_m
    for name, a in (('actual', outcome.actual_m), ('unmet', outcome.unmet_m), ('scale', outcome.scale),
                    ('before W', w0), ('after W', w1), ('request depth', request)):
        _vec(m, col, name, a, shape)
    _vec(m, col, 'LEAF aggregate proposal', proposed, (L * N,))
    q = pools.quota(m['floating_daily_cuda'])
    before = _active_sum_m3(m, cp, col, w0, q, area, 'floating feedback resident W sum (CUDA)')
    scale = before + float(outcome.requested_m3)
    comparison = m['conservation'].volume_roundoff_bound_m3(terms, scale)
    if not math.isfinite(comparison):
        raise FloatingCursorError('native comparison bound is not representable')
    inactive = ~col._maps['active']
    flags = cp.zeros(FLAG_WORDS, dtype=cp.bool_)
    proposed2 = proposed.reshape(shape)
    for l in range(L):
        a, u, qd, sc = outcome.actual_m[l], outcome.unmet_m[l], request[l], outcome.scale[l]
        old, new, pr = w0[l], w1[l], proposed2[l]
        for row in (a, u, qd, sc, old, new):
            ok = cp.isfinite(row)
            ok &= row >= 0.0
            _flag(flags, 0, ~cp.all(ok))
        # Each resource uses the same Native global-scale bound; there is no tighter per-cell law.
        t = a + u
        t -= qd
        t *= area
        cp.abs(t, out=t)
        _flag(flags, 1, cp.any(t > comparison))
        t = new + a
        t -= old
        t *= area
        cp.abs(t, out=t)
        _flag(flags, 1, cp.any(t > comparison))
        t = qd * area
        t -= pr
        cp.abs(t, out=t)
        _flag(flags, 2, cp.any(t > comparison))
        _flag(flags, 3, cp.any(sc > 1.0))
        _flag(flags, 3, cp.any(a > qd))
        _flag(flags, 3, cp.any((a != 0.0) & inactive))
        _flag(flags, 3, cp.any((u != qd) & inactive))
        _flag(flags, 3, cp.any((new != old) & inactive))
        t = sc * qd
        t -= a
        t *= area
        cp.abs(t, out=t)
        _flag(flags, 4, cp.any(t > comparison))
    del t, inactive
    code = _code(m, cp, flags)
    for bit, kind, text in _FLAG_TEXT:
        if code & bit:
            raise kind(text)
    act = _active_sum_m3(m, cp, col, outcome.actual_m, q, area, 'floating feedback resident actual sum (CUDA)')
    unm = _active_sum_m3(m, cp, col, outcome.unmet_m, q, area, 'floating feedback resident unmet sum (CUDA)')
    req = _active_sum_m3(m, cp, col, request, q, area, 'floating feedback resident request sum (CUDA)')
    after = _active_sum_m3(m, cp, col, w1, q, area, 'floating feedback resident W sum (CUDA)')
    if any(abs(x - y) > comparison for x, y in ((act, outcome.actual_m3), (unm, outcome.unmet_m3), (req, outcome.requested_m3),
                                                (act + unm, req), (after + act, before))):
        raise FloatingCursorError('native outcome/global W/T/request closure mismatch')


def _accepted_volumes(m, column, outcome, area):
    cp = _cupy(m, column)
    shape = (int(column.n_layers), int(column.n_cells))
    actual = outcome.actual_m * area  # the single depth -> volume conversion of the native actual, resident
    unmet = outcome.unmet_m * area
    _finite_or_refuse(m, cp, column, 'actual volume', actual, shape)
    _finite_or_refuse(m, cp, column, 'unmet volume', unmet, shape)
    return actual, unmet  # CuPy arrays carry no read-only flag; the bridge never writes them


_OPS = types.SimpleNamespace(column=_column, root=_root, availability=_availability, check_proposal=_check_proposal,
                             to_depth=_to_depth, admit_accept=_admit_accept, check_proposed=_check_proposed,
                             root_map=_root_map, validate_feedback=_validate_feedback, accepted_volumes=_accepted_volumes)
_ROSTER = fb._Roster(fb._MODULE_NAMES + _CUDA_MODULE_NAMES, {**fb._FUNCTION_NAMES, **_CUDA_FUNCTION_NAMES},
                     dict(fb._CLASS_NAMES), {**fb._OBJECT_NAMES, **_CUDA_OBJECT_NAMES}, dict(fb._METHOD_NAMES),
                     fb._MODULE_ALIASES + _CUDA_MODULE_ALIASES)
_CUDA = fb._Backend(BACKEND, PROFILE, CudaFloatingSource, CudaFloatingConnection, True, _ROSTER, _OPS,
                    {'backend': BACKEND, 'device_runtime_capability': True,
                     'residency': 'availability/request/feedback volumes stay on the native column device; the host receives bounded scalar packets only'})


# ---------------------------------------------------------------- public API (same shape as the CPU module)
def bind_source(profile, *, expected_syrup_digest, expected_syrup_commit, expected_maple_commit, max_bytes, reserve_bytes=0):
    """Cold Source binding for the Dense CUDA bridge: the EXPLICIT GPU opt-in.

    Requires a genuine `NativeSourceProfile` whose device_runtime_capability is True
    (the explicit current-commit CUDA source capability Root pins externally) and the
    explicitly supplied committed expected digest/commits; nothing is hardcoded here.
    Host-only cold work admitted against `max_bytes` (+ `reserve_bytes`). No device is
    touched: the device gate runs at connect/read time against the actual column.
    """
    return fb._bind_source(_CUDA, profile, expected_syrup_digest=expected_syrup_digest,
                           expected_syrup_commit=expected_syrup_commit, expected_maple_commit=expected_maple_commit,
                           max_bytes=max_bytes, reserve_bytes=reserve_bytes)


def connect(source, lineage, initial_boundary, *, day_binding):
    """Connect once at a genuine fresh CUDA native accepted boundary (no restore).

    Gate first, then metadata: the root's column must be a Dense CUDA floating column
    on the current device carrying its sealed host mask/edges. Use
    connection.resource_cursor as PlantSession.start(initial_resource=...) of a
    PlantSession whose SessionConfig backend is 'cuda' on the same device.
    """
    return fb._connect(_CUDA, source, lineage, initial_boundary, day_binding=day_binding)


def begin_day(connection, daily_boundary, *, leaf_resource_cursor):
    """Check the genuine CUDA day opening (see the CPU module); metadata and gate only."""
    return fb._begin_day(_CUDA, connection, daily_boundary, leaf_resource_cursor=leaf_resource_cursor)


def read_availability(connection, parent, *, leaf_resource_cursor, pools):
    """Read native resident availability once; return the genuine LEAF AvailabilityRead
    with backend 'cuda' and a resident (L, N) CuPy supply volume for PlantSession."""
    return fb._read_availability(_CUDA, connection, parent, leaf_resource_cursor=leaf_resource_cursor, resources=_pools(pools))


def request_args(read, connection, parent):
    """Native request keywords, bound to the exact read parent and native seq=0. MASY adds
    its own device/host readers and reserves to `request_floating_withdrawal`."""
    return fb._request_args(_CUDA, read, connection, parent)


def request_depth_m(read, proposed_m3, *, pools):
    """One resident volume->depth conversion of the CuPy proposal; the exact request is
    retained for the feedback proof. A host array or another device is refused."""
    return fb._request_depth_m(_CUDA, read, proposed_m3, resources=_pools(pools))


def accepted_withdrawal(connection, proposal, outcome, *, pools):
    """Validate ONE real native CUDA debit; return the genuine LEAF AcceptedWithdrawal
    with resident actual/unmet volumes (see the CPU module for the staging contract)."""
    return fb._accepted_withdrawal(_CUDA, connection, proposal, outcome, resources=_pools(pools))


def commit_day(connection, interevent, closed_daily, native_boundary, *, session):
    """Consume bridge accounting LAST after the native close/hand-back (MASY passes its
    device/host readers to those) and the LEAF publication; identical protocol."""
    return fb._commit_day(_CUDA, connection, interevent, closed_daily, native_boundary, session=session)


__all__ = ['VERSION', 'PROFILE', 'BACKEND', 'FloatingBridgeError', 'FloatingSourceError', 'FloatingCursorError',
           'CudaFloatingSource', 'BridgeCursor', 'CudaFloatingConnection', 'CudaPools', 'bind_source', 'connect',
           'begin_day', 'estimate_bridge_bytes', 'read_availability', 'request_args',
           'request_depth_m', 'accepted_withdrawal', 'commit_day']
