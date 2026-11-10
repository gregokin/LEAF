'''Explicit transpiration withdrawal for the native layered soil column (maple_syrup.subsurface).

Opt-in extension: the existing `subsurface.step` (intake -> capacity return -> drainage -> soil
evaporation -> diffusion -> closure) is reused unchanged; transpiration is a SEPARATE, explicit
withdrawal applied to the hydro result (first-order hydro -> T operator splitting).

Units: water is water-equivalent DEPTH in m on the [L, N] column grid; `requested_water_m` is the
depth demanded over the ALREADY DEFINED interval (the caller fixes dt; this module never scales a rate).
Volumes are depth * column.cell_area_m2 (multiplied exactly once, in the budget).

Withdrawal law (per active layer/cell): available = max(w - residual_m[l], 0); actual = min(requested,
available); water' = w - actual; unmet = requested - actual; scale = actual / requested (0 for zero
demand). No wetter-layer compensation, no species wilting law, no clipping after subtraction.
'''
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from maple_syrup import subsurface as ss
from maple_syrup.conservation import volume_roundoff_bound_m3

try:  # GPU host transfer helper (single small packet); CuPy's own copy is used only if the helper is absent
    from maple.core.backend import to_host as _to_host
except ImportError:  # pragma: no cover - environment dependent
    def _to_host(a):
        return np.asarray(a.get() if hasattr(a, 'get') else a)

SubsurfaceError = ss.SubsurfaceError
SubsurfaceMemoryError = ss.SubsurfaceMemoryError
SubsurfaceBackendUnavailableError = ss.SubsurfaceBackendUnavailableError

VERSION = '0.1.0'
THREADS = 256
_F8 = 8
_VALIDATION_PACKET = 4  # scalar slots in front of the per-layer maxima (see _validate_packet)
_RESULT_PACKET = 6


# ------------------------------------------------------------------ memory estimates / admission
@dataclass(frozen=True)
class TranspirationMemory:
    '''Byte estimate for one withdrawal (prepare parameters excluded; see TranspirationOperator.prepare_bytes).'''
    backend: str
    n_layers: int
    n_cells: int
    parameter_bytes: int      # per-layer residual copy on the compute side
    state_bytes: int          # caller soil water [L,N] that stays live during the operation
    input_bytes: int          # requested_water_m [L,N]
    output_bytes: int         # water' + actual + unmet + scale, all [L,N]
    scratch_bytes: int        # backend temporaries + host/device reduction packets
    host_bytes: int
    device_bytes: int
    reserve_bytes: int        # column.reserve_bytes
    extra_reserve_bytes: int  # caller-owned simultaneously live bytes (e.g. vegetation arrays)
    required_bytes: int       # admitted side (host for numpy/numba, device for cuda) + reserves
    total_bytes: int

    def as_dict(self):
        return dict(self.__dict__)


@dataclass(frozen=True)
class CombinedMemory:
    '''Combined admission for step_with_transpiration: original state + forcing + hydro outputs still
    live + T inputs/outputs/parameters/scratch + column reserve + caller reserve.'''
    backend: str
    hydro: object
    transpiration: TranspirationMemory
    state_bytes: int
    host_bytes: int
    device_bytes: int
    reserve_bytes: int
    extra_reserve_bytes: int
    required_bytes: int
    available_bytes: int
    peak_bytes: int

    def as_dict(self):
        d = dict(self.__dict__)
        d['hydro'] = dict(getattr(self.hydro, '__dict__', {})) or repr(self.hydro)
        d['transpiration'] = self.transpiration.as_dict()
        return d


def _check_backend(backend):
    if backend not in ('numpy', 'numba', 'cuda'):
        raise SubsurfaceError(f'unknown backend {backend!r}')
    return backend


def estimate_transpiration_memory(n_layers, n_cells, *, backend, reserve_bytes=0, extra_reserve_bytes=0,
                                  include_state=True):
    '''Scalar byte estimate of a withdrawal (no allocation). numpy reference needs two [L,N]-sized
    temporaries (available depth, positive-demand mask); numba/cuda kernels need none. Packets: one
    validation packet (4 + L doubles plus two N-sized per-cell maxima on the compute side) and one
    result packet (6 doubles).'''
    L, N = int(n_layers), int(n_cells)
    if L < 1 or N < 1:
        raise SubsurfaceError('n_layers and n_cells must be positive')
    _check_backend(backend)
    LN = _F8 * L * N
    param, inp, out = _F8 * L, LN, 4 * LN
    state = LN if include_state else 0
    scratch = (2 * LN if backend == 'numpy' else 0) + 2 * _F8 * N + _F8 * (_VALIDATION_PACKET + L + _RESULT_PACKET)
    compute = param + state + inp + out + scratch
    host_packet = _F8 * (_VALIDATION_PACKET + L + _RESULT_PACKET)
    host = compute if backend != 'cuda' else host_packet
    device = compute if backend == 'cuda' else 0
    side = device if backend == 'cuda' else host
    required = side + int(reserve_bytes) + int(extra_reserve_bytes)
    return TranspirationMemory(backend, L, N, param, state, inp, out, scratch, host, device, int(reserve_bytes),
                               int(extra_reserve_bytes), required, host + device)


def _available(column):
    fn = getattr(column, 'available_bytes_fn', None)
    if fn is None:
        return None
    v = fn()
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
        raise SubsurfaceError('available_bytes_fn must return an integer byte count')
    return int(v)


def _admit(column, required_bytes, stage):
    '''Scalar admission BEFORE any bulk allocation; refuses without mutating anything or raising budgets.'''
    avail = _available(column)
    if avail is not None and int(required_bytes) > avail:
        raise SubsurfaceMemoryError(f'{stage}: required {int(required_bytes)} bytes exceed available {avail} bytes '
                                    f'(column reserve {int(column.reserve_bytes)} bytes included); nothing allocated')
    return avail


# ------------------------------------------------------------------ kernels
def _withdraw_numpy(w, q, residual, out_w, actual, unmet, scale):
    avail = w - residual[:, None]
    np.maximum(avail, 0.0, out=avail)
    np.minimum(q, avail, out=actual)
    np.subtract(w, actual, out=out_w)
    np.subtract(q, actual, out=unmet)
    pos = q > 0.0
    scale.fill(0.0)
    np.divide(actual, q, out=scale, where=pos)


try:
    from numba import njit as _njit

    @_njit(cache=True)
    def _withdraw_numba(w, q, residual, out_w, actual, unmet, scale):
        L, N = w.shape
        for l in range(L):
            r = residual[l]
            for j in range(N):
                wv = w[l, j]
                qv = q[l, j]
                a = wv - r
                if a < 0.0:
                    a = 0.0
                t = qv if qv < a else a
                out_w[l, j] = wv - t
                actual[l, j] = t
                unmet[l, j] = qv - t
                scale[l, j] = t / qv if qv > 0.0 else 0.0
except ImportError:  # pragma: no cover
    _withdraw_numba = None

_CUDA_SRC = r'''
extern "C" __global__ void withdraw(const double* __restrict__ w, const double* __restrict__ q,
                                    const double* __restrict__ residual, double* __restrict__ out_w,
                                    double* __restrict__ actual, double* __restrict__ unmet,
                                    double* __restrict__ scale, const long long N, const long long M) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    long long l = i / N;
    double wv = w[i];
    double qv = q[i];
    double a = wv - residual[l];
    if (a < 0.0) a = 0.0;
    double t = (qv < a) ? qv : a;
    out_w[i] = wv - t;
    actual[i] = t;
    unmet[i] = qv - t;
    scale[i] = (qv > 0.0) ? (t / qv) : 0.0;
}
'''


# ------------------------------------------------------------------ operator
@dataclass(frozen=True, eq=False)
class TranspirationOperator:
    '''Prepared (compiled) withdrawal operator bound to one native column. Holds a small per-layer
    parameter copy on the compute side; never re-uploads parameters in daily work.'''
    column: object
    backend: str
    xp: object
    n_layers: int
    n_cells: int
    residual_xp: object       # [L] FP64 residual depth per layer (compute side)
    saturation_host: object   # [L] host saturation + capacity tolerance (state validity bound)
    kernel: object            # compiled callable (numba dispatcher / CUDA RawKernel) or None (numpy)
    memory: TranspirationMemory
    prepare_bytes: int

    def check_device(self):
        '''CUDA: the current device must be the column's device and the column preflight (if any) must pass.'''
        if self.backend == 'cuda':
            dev = int(self.xp.cuda.Device().id)
            if self.column.device_id is not None and dev != int(self.column.device_id):
                raise SubsurfaceBackendUnavailableError(
                    f'current CUDA device {dev} differs from column device {self.column.device_id}')
        pf = getattr(self.column, 'preflight', None)
        if callable(pf) and pf() is False:
            raise SubsurfaceBackendUnavailableError('column preflight failed')


def prepare_transpiration(column, *, extra_reserve_bytes=0):
    '''Compile/prepare outside daily work. One scalar admission precedes the parameter copy and the
    compiled warm-up launch. Columns are never mutated.'''
    ss._check_column(column)
    backend = _check_backend(column.backend)
    L, N = int(column.n_layers), int(column.n_cells)
    layers = column.layers
    residual = np.ascontiguousarray(np.asarray(layers.residual_m, dtype=np.float64))
    sat = np.asarray(layers.saturation_m, dtype=np.float64) + np.asarray(layers.capacity_tolerance_m, dtype=np.float64)
    if residual.shape != (L,) or sat.shape != (L,):
        raise SubsurfaceError('layer residual/saturation arrays must have shape [n_layers]')
    if not (np.all(np.isfinite(residual)) and np.all(residual >= 0.0) and np.all(np.isfinite(sat))):
        raise SubsurfaceError('layer residual_m must be finite and non-negative, saturation_m finite')
    xp = column.xp
    if backend == 'numba' and _withdraw_numba is None:
        raise SubsurfaceBackendUnavailableError('numba backend requested but numba is not importable')
    mem = estimate_transpiration_memory(L, N, backend=backend, reserve_bytes=column.reserve_bytes,
                                        extra_reserve_bytes=extra_reserve_bytes)
    warm = estimate_transpiration_memory(1, 1, backend=backend)  # tiny compiled warm-up launch
    prepare_bytes = _F8 * L + warm.total_bytes
    _admit(column, prepare_bytes + int(column.reserve_bytes) + int(extra_reserve_bytes), 'prepare_transpiration')
    kernel = None
    if backend == 'cuda':
        dev = int(xp.cuda.Device().id)
        if column.device_id is not None and dev != int(column.device_id):
            raise SubsurfaceBackendUnavailableError(
                f'current CUDA device {dev} differs from column device {column.device_id}')
        kernel = xp.RawKernel(_CUDA_SRC, 'withdraw', options=('-std=c++11',))
        kernel.compile()
    elif backend == 'numba':
        kernel = _withdraw_numba
    residual_xp = xp.ascontiguousarray(xp.asarray(residual, dtype=xp.float64))
    op = TranspirationOperator(column, backend, xp, L, N, residual_xp, sat, kernel, mem, prepare_bytes)
    # compiled warm-up on a (1, 1) problem (admitted above): compile/cache load outside daily work
    one = xp.ones((1, 1), dtype=xp.float64)
    outs = [xp.empty((1, 1), dtype=xp.float64) for _ in range(4)]
    _launch(op, one, xp.zeros((1, 1), dtype=xp.float64), residual_xp[:1], *outs)
    if backend == 'cuda':
        xp.cuda.Device().synchronize()
    return op


def _launch(op, w, q, residual, out_w, actual, unmet, scale):
    if op.backend == 'numpy':
        _withdraw_numpy(w, q, residual, out_w, actual, unmet, scale)
    elif op.backend == 'numba':
        op.kernel(w, q, residual, out_w, actual, unmet, scale)
    else:
        M = int(w.shape[0]) * int(w.shape[1])
        op.kernel(((M + THREADS - 1) // THREADS,), (THREADS,),
                  (w, q, residual, out_w, actual, unmet, scale, np.int64(w.shape[1]), np.int64(M)))


# ------------------------------------------------------------------ validation (one small packet)
def _validate_packet(op, state, requested):
    '''Metadata via the pinned native helpers; values via ONE host packet
    [w_min, q_min, q_max, max inactive request, w_max per layer...] (no [L,N] temporaries, no full reads).'''
    column, xp = op.column, op.xp
    t_s = ss._check_state(column, state)
    ss._check_dynamic(column, 'requested_water_m', requested, (op.n_layers, op.n_cells))
    w = state.water_m
    qmax_cell = requested.max(axis=0)                       # [N]
    inactive_req = xp.where(column.active, 0.0, qmax_cell).max()
    head = xp.stack([w.min(), requested.min(), requested.max(), inactive_req]).astype(xp.float64)
    packet = np.asarray(_to_host(xp.concatenate([head, w.max(axis=1).astype(xp.float64)])), dtype=np.float64)
    wmin, qmin, qmax, inact = (float(v) for v in packet[:_VALIDATION_PACKET])
    wmax_l = packet[_VALIDATION_PACKET:]
    if not (math.isfinite(wmin) and np.all(np.isfinite(wmax_l))) or wmin < 0.0:
        raise SubsurfaceError('soil water must be finite and non-negative on every cell (inactive included)')
    if np.any(wmax_l > op.saturation_host):
        raise SubsurfaceError('soil water above saturation_m + capacity_tolerance_m in at least one layer')
    if not (math.isfinite(qmin) and math.isfinite(qmax)) or qmin < 0.0:
        raise SubsurfaceError('requested_water_m must be finite and non-negative on every cell (inactive included)')
    if inact > 0.0 or not math.isfinite(inact):
        raise SubsurfaceError('requested_water_m must be zero on inactive cells')
    return t_s, qmax


# ------------------------------------------------------------------ withdrawal
@dataclass(frozen=True)
class TranspirationBudget:
    storage_before_m3: float
    storage_after_m3: float
    requested_m3: float
    actual_m3: float
    unmet_m3: float
    residual_m3: float          # after - before + actual (closure of the water removed)
    request_residual_m3: float  # requested - actual - unmet
    bound_m3: float             # MAPLE FP64 summation policy bound (volume_roundoff_bound_m3)
    n_terms: int
    min_water_after_m: float

    def as_dict(self):
        return dict(self.__dict__)


@dataclass(frozen=True, eq=False)
class TranspirationResult:
    state: object        # fresh authoritative SoilColumnState (time preserved)
    actual_m: object     # [L,N] depth actually withdrawn
    unmet_m: object      # [L,N] requested - actual
    scale: object        # [L,N] actual / requested (0 for zero demand)
    budget: TranspirationBudget
    memory: TranspirationMemory


def withdraw_transpiration(operator, state, requested_water_m, *, extra_reserve_bytes=0):
    '''Pure native withdrawal: validates state/forcing on all cells, admits memory, runs the compiled
    kernel, reduces ONE result packet, checks closure (no clipping) and returns a fresh state with the
    same t_s. Inputs are never modified; on any rejection nothing is published.'''
    op = operator
    if not isinstance(op, TranspirationOperator):
        raise SubsurfaceError('operator must come from prepare_transpiration')
    op.check_device()
    column, xp = op.column, op.xp
    t_s, _ = _validate_packet(op, state, requested_water_m)
    mem = estimate_transpiration_memory(op.n_layers, op.n_cells, backend=op.backend,
                                        reserve_bytes=column.reserve_bytes, extra_reserve_bytes=extra_reserve_bytes)
    _admit(column, mem.required_bytes, 'withdraw_transpiration')
    shape = (op.n_layers, op.n_cells)
    out_w, actual, unmet, scale = (xp.empty(shape, dtype=xp.float64) for _ in range(4))
    w = state.water_m
    _launch(op, w, requested_water_m, op.residual_xp, out_w, actual, unmet, scale)
    packet = np.asarray(_to_host(xp.stack([w.sum(), out_w.sum(), actual.sum(), unmet.sum(),
                                            requested_water_m.sum(), out_w.min()]).astype(xp.float64)),
                        dtype=np.float64)
    area = float(column.cell_area_m2)
    before, after, act, unm, req = (float(v) * area for v in packet[:5])
    wmin = float(packet[5])
    n_terms = 3 * op.n_layers * op.n_cells + 4
    bound = float(volume_roundoff_bound_m3(n_terms, max(before + req, 0.0)))
    budget = TranspirationBudget(before, after, req, act, unm, after - before + act, req - act - unm, bound,
                                 n_terms, wmin)
    if not (math.isfinite(after) and math.isfinite(act) and math.isfinite(unm)) or wmin < 0.0:
        raise SubsurfaceError(f'transpiration withdrawal produced non-finite or negative water: {budget}')
    if abs(budget.residual_m3) > bound or abs(budget.request_residual_m3) > bound:
        raise SubsurfaceError(f'transpiration closure violated beyond the FP64 bound: {budget}')
    return TranspirationResult(ss.SoilColumnState(out_w, t_s), actual, unmet, scale, budget, mem)


# ------------------------------------------------------------------ combined step
@dataclass(frozen=True)
class CoupledBudget:
    '''after - before = added - surface_return - drainage - evaporation - transpiration (E and T distinct).
    storage_after is the hydro storage definition carried through the T change; residual = hydro residual
    + T residual; bound = hydro bound + T bound (both MAPLE-derived, no new epsilon).'''
    storage_before_m3: float
    storage_after_m3: float
    added_m3: float
    surface_return_m3: float
    drainage_m3: float
    evaporation_m3: float
    evaporation_unmet_m3: float
    transpiration_m3: float
    transpiration_unmet_m3: float
    transpiration_requested_m3: float
    residual_m3: float
    bound_m3: float
    n_terms: int

    def as_dict(self):
        return dict(self.__dict__)


@dataclass(frozen=True, eq=False)
class CoupledStep:
    '''Final authoritative state plus hydro diagnostics and T outputs. The intermediate (post-hydro,
    pre-T) water is NOT retained.'''
    state: object
    dt_s: float
    surface_return_m: object
    drainage_m: object
    evaporation_m: object
    evaporation_unmet_m: object
    percolation_m: object
    diffusion_m: object
    hydro_budget: object
    actual_m: object
    unmet_m: object
    scale: object
    transpiration_budget: TranspirationBudget
    budget: CoupledBudget
    memory: CombinedMemory

    def diagnostic_bytes(self):
        return int(sum(int(a.nbytes) for a in (self.surface_return_m, self.drainage_m, self.evaporation_m,
                                                self.evaporation_unmet_m, self.percolation_m, self.diffusion_m,
                                                self.actual_m, self.unmet_m, self.scale)))


def estimate_combined_memory(column, *, extra_reserve_bytes=0):
    '''Scalar combined estimate (no allocation): hydro step estimate + T estimate (state counted once)
    + column reserve + caller reserve.'''
    L, N = int(column.n_layers), int(column.n_cells)
    backend = column.backend
    hm = ss.estimate_step_memory(L, N, backend=backend, reserve_bytes=0)
    tm = estimate_transpiration_memory(L, N, backend=backend, reserve_bytes=0, include_state=False)
    state_bytes = _F8 * L * N
    host = int(hm.host_bytes) + tm.host_bytes + (0 if backend == 'cuda' else state_bytes)
    device = int(hm.device_bytes) + tm.device_bytes + (state_bytes if backend == 'cuda' else 0)
    side = device if backend == 'cuda' else host
    required = side + int(column.reserve_bytes) + int(extra_reserve_bytes)
    avail = _available(column)
    return CombinedMemory(backend, hm, tm, state_bytes, host, device, int(column.reserve_bytes),
                          int(extra_reserve_bytes), required, -1 if avail is None else avail, required)


def step_with_transpiration(operator, state, dt_s, *, added_water_m, potential_evaporation_m_s,
                            requested_transpiration_m, extra_reserve_bytes=0):
    '''First-order splitting: native hydro `step` (advances t_s by dt once) followed by the explicit
    withdrawal (time preserved). dt=0 is an identity and requires zero requested transpiration.
    Combined admission (original state + forcing + still-live hydro outputs + T work) precedes any work.'''
    op = operator
    if not isinstance(op, TranspirationOperator):
        raise SubsurfaceError('operator must come from prepare_transpiration')
    op.check_device()
    column = op.column
    dt = ss._check_dt(dt_s)
    _, qmax = _validate_packet(op, state, requested_transpiration_m)
    if dt == 0.0 and qmax > 0.0:
        raise SubsurfaceError('dt_s=0 is an identity step and requires zero requested transpiration')
    mem = estimate_combined_memory(column, extra_reserve_bytes=extra_reserve_bytes)
    _admit(column, mem.required_bytes, 'step_with_transpiration')
    hydro = ss.step(column, state, dt, added_water_m=added_water_m, potential_evaporation_m_s=potential_evaporation_m_s)
    tr = withdraw_transpiration(op, hydro.state, requested_transpiration_m, extra_reserve_bytes=extra_reserve_bytes)
    hb, tb = hydro.budget, tr.budget
    before = float(hb.storage_before_m3)
    after = float(hb.storage_after_m3) + (tb.storage_after_m3 - tb.storage_before_m3)
    residual = float(hb.residual_m3) + tb.residual_m3
    bound = float(hb.bound_m3) + tb.bound_m3
    budget = CoupledBudget(before, after, float(hb.added_m3), float(hb.surface_return_m3), float(hb.drainage_m3),
                           float(hb.evaporation_m3), float(hb.evaporation_unmet_m3), tb.actual_m3, tb.unmet_m3,
                           tb.requested_m3, residual, bound, int(hb.n_terms) + tb.n_terms)
    if abs(residual) > bound:
        raise SubsurfaceError(f'combined hydro+transpiration closure violated beyond the FP64 bound: {budget}')
    return CoupledStep(tr.state, dt, hydro.surface_return_m, hydro.drainage_m, hydro.evaporation_m,
                       hydro.evaporation_unmet_m, hydro.percolation_m, hydro.diffusion_m, hb,
                       tr.actual_m, tr.unmet_m, tr.scale, tb, budget, mem)


__all__ = ['prepare_transpiration', 'withdraw_transpiration', 'step_with_transpiration',
           'estimate_transpiration_memory', 'estimate_combined_memory', 'TranspirationOperator',
           'TranspirationResult', 'TranspirationBudget', 'TranspirationMemory', 'CombinedMemory',
           'CoupledStep', 'CoupledBudget', 'SubsurfaceError', 'SubsurfaceMemoryError',
           'SubsurfaceBackendUnavailableError', 'VERSION']
