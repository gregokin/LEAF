'''leaf.uptake: stateless sparse water-uptake allocator (proposal + accepted feedback) over a plant-major CSR root map.

Water contract (LEAF Phase 1 part A, revision 2). Every amount is m3 of water for ONE timestep; depth/area
conversion belongs to the native SYRUP side (downstream), never to this module.

    R[p, r]     = D[p] * w[p, r]                 per-edge request; resource r = layer * n_cells + cell
    Q[r]        = sum_p R[p, r]                  local request on a resource (scatter)
    f[r]        = min(1, A[r] / Q[r])            shared fraction (0 when Q[r] == 0): every plant on a resource
                                                 receives the SAME fraction, no species / size / order priority
    proposed[r] = min(A[r], Q[r])
    SYRUP validates and accepts ONCE:            accepted[r] <= proposed[r]  (+ honest FP64 roundoff, see below)
    delivered[p] = sum_e D[p] * w[e] * accepted[r_e] / Q[r_e]      (0 where Q[r_e] == 0)   (gather)
    S[p]        = delivered[p] / D[p] for D[p] > 0, else 0           explicit zero-demand policy (no invented growth)
    unmet[p]    = D[p] - delivered[p]

No root plasticity or compensation: a missed request on a dry / masked / unavailable resource (A = 0) stays
unmet and is never moved to another cell or layer. Water nobody requested stays with SYRUP (deeper residual
water is unused). SYRUP owns the sole water inventory and the accepted debit: nothing here holds, decrements
or caches a water store; `propose` and `accept` are pure functions of their inputs.

Accepted water is the native debit. `accept` keeps SYRUP's accepted amounts UNCHANGED as the authoritative
input: delivered water and the closure are computed against them. An excess over the proposal that is within
`acceptance_tolerance_rel` (FP64 conversion / summation roundoff) is reported (`n_roundoff_excess`,
`max_roundoff_excess_m3`) but never clipped; a material excess, water on a resource nobody requested, negative
or non-finite values are refused. Per plant, delivered water may then exceed the demand by at most
`overdelivery_bound_rel` (row-sum tolerance + roundoff); S is capped at 1 and unmet at 0 only within that bound
(`n_plants_clamped`, `max_overdelivery_m3` record it); beyond it the result is an implementation failure.

Root map (`RootMap`). Validated, immutable, plant-major CSR: stable int64 plant IDs [P], int64 offsets [P+1],
int64 flattened resource indices [E] (layer * n_cells + cell), FP64 nonnegative weights [E]. Weights are fully
general (NON-separable). Rows with edges must sum to 1 within WEIGHT_SUM_TOL; rows without edges are inactive
roots (allowed) on which positive demand is REFUSED. Inputs must be TYPED ARRAYS (numpy.ndarray, or
cupy.ndarray for the cuda backend): Python sequences are refused without being materialised, so no type
information is lost in a conversion (a mixed [True, 2] list never becomes an integer vector here). Bool / float
/ unsigned-overflow IDs and indices are refused before any cast; offsets are range- and order-checked BEFORE
any subtraction, repeat or kernel (no wrap-around through diff). The map owns copies (host copies frozen
read-only); failed construction leaves the caller's arrays untouched.

Backends. `numpy` (vectorised reference), `numba` (compiled CPU kernels) and `cuda` (CuPy RawKernel, FP64
atomics, block reductions) implement ONE algorithm: the per-edge arithmetic and all per-plant / per-resource
diagnostics are written once in the plain Python kernels `_k_*` below (the authoritative definition, compiled
by numba, mirrored line by line in CUDA C, vectorised by the NumPy reference). Diagnostics and validation
flags are accumulated into ONE small fixed-size FP64 packet per operation; on cuda the packet is read back by a
single explicit synchronisation (`_sync_packet`), never a P/E/R-sized array. No dense P x L x N tensor, no
P x P matrix, no per-edge daily history. A cuda map is BOUND TO THE DEVICE it was built on (`device_id`): every
supplied CuPy array and the current device are checked against it before any work; kernels are compiled per
device; there is no device switch, peer access or fallback. Plans / feedback live on the backend of their map;
cross-namespace arrays are refused (no implicit host/device transfers).

Finite results only. Every computed array, aggregate, bound and the closure are checked finite; overflowing
demand / request / totals (e.g. two 1e308 demands on one resource) refuse the plan or feedback instead of
returning inf / NaN, so nothing non-finite can reach a native authority.

Memory admission. `max_edges` and `max_bytes` are explicit for map construction; operations reuse the map's
budget, and an operation-level `max_bytes` may only TIGHTEN it. Every allocating public path estimates its
OWNED arrays, its SCRATCH (validation temporaries, casts, boolean masks, sort / repeat / bincount buffers,
old/new coexistence during conversions) and TRANSFER STAGING before anything is allocated and refuses with
`AdmissionError` (carrying the itemised estimate) beforehand; preflight happens before any copy, cast or
transfer. Scope: live algorithm arrays of this module only. EXCLUDED: Python interpreter objects, NumPy / CuPy
allocator-pool overhead, numba JIT / CUDA context and compiled-kernel memory, and the caller's own input
arrays (the caller already holds them). Bounds are never raised and individuals never discarded to pass.

Native authority boundary. `propose` produces a proposal; `accept` turns SYRUP's single accepted answer into
delivered / unmet / S. Publishing receipts or state is the job of the later driver and the native water
authority. Importing this module imports NumPy only; numba / cupy are imported lazily per backend.
'''
from __future__ import annotations

import math

import numpy as np

from leaf import traits as _traits

VERSION = '0.2.0'
BACKENDS = ('numpy', 'numba', 'cuda')
HOST_BACKENDS = ('numpy', 'numba')
FP64_EPS = float(np.finfo(np.float64).eps)
WEIGHT_SUM_TOL = _traits.LAYER_FRACTION_TOL       # 1e-9: rows with edges must sum to 1 within this (not renormalised)
INT64_MAX = 2 ** 63 - 1
THREADS = 256
MAX_BLOCKS = 65535
ID_PREVIEW = 10
UNITS = {'demand': 'm3 per timestep', 'supply': 'm3 available per (layer, cell) for the timestep',
         'request': 'm3', 'proposed': 'm3', 'accepted': 'm3 (SYRUP debit, unchanged)', 'delivered': 'm3',
         'unmet': 'm3', 'satisfaction': 'fraction [0, 1]', 'fraction': 'fraction [0, 1]',
         'weights': 'dimensionless (row sum 1)'}
ALGORITHM = ('plant-major CSR: scatter Q[r] += D[p]*w[e]; proposed[r] = min(A[r], Q[r]); '
             'scale[r] = accepted[r]/Q[r] (0 if Q == 0); gather delivered[p] += D[p]*w[e]*scale[r]')
ZERO_DEMAND_POLICY = 'S[p] = 0 when D[p] == 0 (explicit: zero demand yields zero satisfaction, no invented growth)'
NO_PLASTICITY_NOTE = ('no root plasticity / compensation: a missed request on a dry or unavailable resource (A = 0) '
                      'stays unmet; weights are never renormalised or redistributed to wetter cells or layers')
ACCEPTED_POLICY = ('accepted amounts are the native debit and are never modified: roundoff-level excess over the '
                   'proposal is reported, material excess is refused; delivered water and the closure use the '
                   'accepted amounts as supplied')
MEMORY_SCOPE = ('live algorithm arrays of leaf.uptake only (owned + scratch + staging); excluded: interpreter objects, '
                'NumPy/CuPy allocator-pool overhead, numba JIT / CUDA context / compiled kernels, caller input arrays')
# packet layouts (FP64 counters / totals; one small packet per operation, one synchronisation on cuda)
PROPOSE_PACKET = ('demand_nonfinite', 'demand_negative', 'n_plants_demanding', 'demand_total_m3', 'n_positive_on_empty',
                  'supply_nonfinite', 'supply_negative', 'request_nonfinite', 'supply_total_m3', 'request_total_m3',
                  'proposed_total_m3', 'n_resources_requested', 'n_resources_limited')
ACCEPT_PACKET = ('accepted_nonfinite', 'accepted_negative', 'n_roundoff_excess', 'max_roundoff_excess_m3',
                 'n_material_excess', 'accepted_total_m3', 'n_unrequested_water', 'delivered_total_m3',
                 'unmet_total_m3', 'n_over_bound', 'n_plants_clamped', 'max_overdelivery_m3', 'n_plants_unsatisfied',
                 'n_plants_zero_demand', 'delivered_nonfinite')
_PP = {k: i for i, k in enumerate(PROPOSE_PACKET)}
_AP = {k: i for i, k in enumerate(ACCEPT_PACKET)}
NP_PACKET, NA_PACKET = len(PROPOSE_PACKET), len(ACCEPT_PACKET)


class UptakeError(ValueError):
    '''Any root-map / plan / feedback validation error or allocator implementation failure.'''


class AdmissionError(UptakeError):
    '''A declared edge/byte bound would be exceeded; raised BEFORE any large allocation, cast or transfer.'''

    def __init__(self, stage, reason, estimate):
        super().__init__(f'{stage}: {reason}')
        self.stage = stage
        self.reason = reason
        self.estimate = estimate


class BackendUnavailableError(UptakeError):
    '''The requested backend cannot be used on this machine (missing numba / cupy / CUDA device).'''


# ------------------------------------------------------------------ authoritative per-edge arithmetic (plain Python)
# Packet indices are spelled out as literals so that the same source compiles under numba unchanged.
# PROPOSE: 0 demand_nonfinite 1 demand_negative 2 n_demanding 3 sum_demand 4 n_positive_on_empty 5 supply_nonfinite
#          6 supply_negative 7 request_nonfinite 8 sum_supply 9 sum_request 10 sum_proposed 11 n_requested 12 n_limited
# ACCEPT:  0 acc_nonfinite 1 acc_negative 2 n_roundoff_excess 3 max_excess 4 n_material_excess 5 sum_accepted
#          6 n_unrequested 7 sum_delivered 8 sum_unmet 9 n_over_bound 10 n_clamped 11 max_overdelivery
#          12 n_unsatisfied 13 n_zero_demand 14 delivered_nonfinite
def _k_check_demand(counts, demand, packet):
    '''Demand validation + totals. Positive demand on an edge-less row is counted (refused by the caller).'''
    for p in range(demand.shape[0]):
        d = demand[p]
        if not math.isfinite(d):
            packet[0] += 1.0
        elif d < 0.0:
            packet[1] += 1.0
        elif d > 0.0:
            packet[2] += 1.0
            packet[3] += d
            if counts[p] == 0:
                packet[4] += 1.0


def _k_scatter(offsets, resources, weights, demand, request):
    '''request[r] = sum over edges of demand[p] * weights[e] (request pre-zeroed by the caller).'''
    for p in range(demand.shape[0]):
        d = demand[p]
        if d == 0.0:
            continue
        for e in range(offsets[p], offsets[p + 1]):
            request[resources[e]] += d * weights[e]


def _k_proposal(supply, request, proposed, fraction, packet):
    '''proposed[r] = min(A, Q); fraction[r] = proposed / Q (both 0 when Q == 0); validation + totals.'''
    for r in range(request.shape[0]):
        a = supply[r]
        q = request[r]
        if not math.isfinite(a):
            packet[5] += 1.0
        elif a < 0.0:
            packet[6] += 1.0
        else:
            packet[8] += a
        if not math.isfinite(q):
            packet[7] += 1.0
            proposed[r] = 0.0
            fraction[r] = 0.0
            continue
        packet[9] += q
        if q > 0.0:
            packet[11] += 1.0
            pr = a if a < q else q
            if q > a:
                packet[12] += 1.0
            proposed[r] = pr
            fraction[r] = pr / q
            packet[10] += pr
        else:
            proposed[r] = 0.0
            fraction[r] = 0.0


def _k_scale(accepted, request, proposed, tol_rel, scale, packet):
    '''scale[r] = accepted[r] / Q[r] (0 when Q == 0) from the UNMODIFIED accepted amounts; excess diagnostics.'''
    for r in range(request.shape[0]):
        acc = accepted[r]
        q = request[r]
        if not math.isfinite(acc):
            packet[0] += 1.0
            scale[r] = 0.0
            continue
        if acc < 0.0:
            packet[1] += 1.0
            scale[r] = 0.0
            continue
        packet[5] += acc
        ex = acc - proposed[r]
        if ex > 0.0:
            packet[2] += 1.0
            if ex > packet[3]:
                packet[3] = ex
            if ex > tol_rel * proposed[r]:
                packet[4] += 1.0
        if q > 0.0:
            scale[r] = acc / q
        else:
            scale[r] = 0.0
            if acc > 0.0:
                packet[6] += 1.0


def _k_gather(offsets, resources, weights, demand, scale, over_rel, delivered, unmet, satisfaction, packet):
    '''delivered[p] = sum over edges of demand[p] * weights[e] * scale[resources[e]]; unmet / S / diagnostics.'''
    for p in range(demand.shape[0]):
        d = demand[p]
        u = 0.0
        if d != 0.0:
            for e in range(offsets[p], offsets[p + 1]):
                u += d * weights[e] * scale[resources[e]]
        delivered[p] = u
        if not math.isfinite(u):
            packet[14] += 1.0
            unmet[p] = 0.0
            satisfaction[p] = 0.0
            continue
        packet[7] += u
        if d > 0.0:
            if u > d * (1.0 + over_rel):
                packet[9] += 1.0
            if u > d:
                packet[10] += 1.0
                ov = u - d
                if ov > packet[11]:
                    packet[11] = ov
                unmet[p] = 0.0
                satisfaction[p] = 1.0
            else:
                unmet[p] = d - u
                satisfaction[p] = u / d
                if u < d:
                    packet[12] += 1.0
        else:
            unmet[p] = 0.0
            satisfaction[p] = 0.0
            packet[13] += 1.0
        packet[8] += unmet[p]


_CUDA_SRC = r'''
extern "C" {
#define NT 256

__device__ __forceinline__ void block_add(double v, double* target, double* shm) {
    shm[threadIdx.x] = v;
    __syncthreads();
    for (int s = NT / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shm[threadIdx.x] += shm[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0 && shm[0] != 0.0) atomicAdd(target, shm[0]);
    __syncthreads();
}

__device__ __forceinline__ void atomic_max_nonneg(double* target, double v) {
    /* non-negative doubles order like their unsigned bit patterns */
    if (v > 0.0) atomicMax((unsigned long long*)target, (unsigned long long)__double_as_longlong(v));
}

__global__ void check_demand(const long long* counts, const double* demand, double* packet, const long long n) {
    __shared__ double shm[NT];
    long long stride = (long long)gridDim.x * blockDim.x;
    double c0 = 0.0, c1 = 0.0, c2 = 0.0, c3 = 0.0, c4 = 0.0;
    for (long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x; p < n; p += stride) {
        double d = demand[p];
        if (!isfinite(d)) c0 += 1.0;
        else if (d < 0.0) c1 += 1.0;
        else if (d > 0.0) { c2 += 1.0; c3 += d; if (counts[p] == 0) c4 += 1.0; }
    }
    block_add(c0, packet + 0, shm); block_add(c1, packet + 1, shm); block_add(c2, packet + 2, shm);
    block_add(c3, packet + 3, shm); block_add(c4, packet + 4, shm);
}

__global__ void scatter(const long long* offsets, const long long* resources, const double* weights,
                        const double* demand, double* request, const long long n_plants) {
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x; p < n_plants; p += stride) {
        double d = demand[p];
        if (d == 0.0) continue;
        for (long long e = offsets[p]; e < offsets[p + 1]; ++e) atomicAdd(&request[resources[e]], d * weights[e]);
    }
}

__global__ void proposal(const double* supply, const double* request, double* proposed, double* fraction,
                         double* packet, const long long n) {
    __shared__ double shm[NT];
    long long stride = (long long)gridDim.x * blockDim.x;
    double c5 = 0.0, c6 = 0.0, c7 = 0.0, c8 = 0.0, c9 = 0.0, c10 = 0.0, c11 = 0.0, c12 = 0.0;
    for (long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x; r < n; r += stride) {
        double a = supply[r];
        double q = request[r];
        if (!isfinite(a)) c5 += 1.0;
        else if (a < 0.0) c6 += 1.0;
        else c8 += a;
        if (!isfinite(q)) { c7 += 1.0; proposed[r] = 0.0; fraction[r] = 0.0; continue; }
        c9 += q;
        if (q > 0.0) {
            c11 += 1.0;
            double pr = a < q ? a : q;
            if (q > a) c12 += 1.0;
            proposed[r] = pr;
            fraction[r] = pr / q;
            c10 += pr;
        } else {
            proposed[r] = 0.0;
            fraction[r] = 0.0;
        }
    }
    block_add(c5, packet + 5, shm); block_add(c6, packet + 6, shm); block_add(c7, packet + 7, shm);
    block_add(c8, packet + 8, shm); block_add(c9, packet + 9, shm); block_add(c10, packet + 10, shm);
    block_add(c11, packet + 11, shm); block_add(c12, packet + 12, shm);
}

__global__ void scale_kernel(const double* accepted, const double* request, const double* proposed,
                             const double tol_rel, double* scale, double* packet, const long long n) {
    __shared__ double shm[NT];
    long long stride = (long long)gridDim.x * blockDim.x;
    double c0 = 0.0, c1 = 0.0, c2 = 0.0, c4 = 0.0, c5 = 0.0, c6 = 0.0, mx = 0.0;
    for (long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x; r < n; r += stride) {
        double acc = accepted[r];
        double q = request[r];
        if (!isfinite(acc)) { c0 += 1.0; scale[r] = 0.0; continue; }
        if (acc < 0.0) { c1 += 1.0; scale[r] = 0.0; continue; }
        c5 += acc;
        double ex = acc - proposed[r];
        if (ex > 0.0) {
            c2 += 1.0;
            if (ex > mx) mx = ex;
            if (ex > tol_rel * proposed[r]) c4 += 1.0;
        }
        if (q > 0.0) scale[r] = acc / q;
        else { scale[r] = 0.0; if (acc > 0.0) c6 += 1.0; }
    }
    block_add(c0, packet + 0, shm); block_add(c1, packet + 1, shm); block_add(c2, packet + 2, shm);
    block_add(c4, packet + 4, shm); block_add(c5, packet + 5, shm); block_add(c6, packet + 6, shm);
    atomic_max_nonneg(packet + 3, mx);
}

__global__ void gather(const long long* offsets, const long long* resources, const double* weights,
                       const double* demand, const double* scale, const double over_rel, double* delivered,
                       double* unmet, double* satisfaction, double* packet, const long long n_plants) {
    __shared__ double shm[NT];
    long long stride = (long long)gridDim.x * blockDim.x;
    double c7 = 0.0, c8 = 0.0, c9 = 0.0, c10 = 0.0, c12 = 0.0, c13 = 0.0, c14 = 0.0, mx = 0.0;
    for (long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x; p < n_plants; p += stride) {
        double d = demand[p];
        double u = 0.0;
        if (d != 0.0) {
            for (long long e = offsets[p]; e < offsets[p + 1]; ++e) u += d * weights[e] * scale[resources[e]];
        }
        delivered[p] = u;
        if (!isfinite(u)) { c14 += 1.0; unmet[p] = 0.0; satisfaction[p] = 0.0; continue; }
        c7 += u;
        if (d > 0.0) {
            if (u > d * (1.0 + over_rel)) c9 += 1.0;
            if (u > d) {
                c10 += 1.0;
                double ov = u - d;
                if (ov > mx) mx = ov;
                unmet[p] = 0.0;
                satisfaction[p] = 1.0;
            } else {
                unmet[p] = d - u;
                satisfaction[p] = u / d;
                if (u < d) c12 += 1.0;
            }
        } else {
            unmet[p] = 0.0;
            satisfaction[p] = 0.0;
            c13 += 1.0;
        }
        c8 += unmet[p];
    }
    block_add(c7, packet + 7, shm); block_add(c8, packet + 8, shm); block_add(c9, packet + 9, shm);
    block_add(c10, packet + 10, shm); block_add(c12, packet + 12, shm); block_add(c13, packet + 13, shm);
    block_add(c14, packet + 14, shm);
    atomic_max_nonneg(packet + 11, mx);
}

}
'''
_CUDA_KERNELS = ('check_demand', 'scatter', 'proposal', 'scale_kernel', 'gather')


# ------------------------------------------------------------------ backends (lazy, device keyed)
_BACKEND = {}
_COMPILED = {}


def backend_status(backend):
    '''Probe a backend once per process: {'backend', 'available', 'reason', ...}. Never raises for an
    unavailable backend (tests may skip on it); an unknown backend name raises UptakeError. For cuda the
    CURRENT device at probe time is reported; maps bind to the current device at build time.'''
    if backend not in BACKENDS:
        raise UptakeError(f'backend must be one of {BACKENDS}, got {backend!r}')
    st = _BACKEND.get(backend)
    if st is None:
        if backend == 'numpy':
            st = {'backend': 'numpy', 'available': True, 'reason': None, 'numpy_version': np.__version__}
        elif backend == 'numba':
            try:
                import numba
                st = {'backend': 'numba', 'available': True, 'reason': None, 'numba_version': numba.__version__}
            except Exception as exc:  # pragma: no cover - machine dependent
                st = {'backend': 'numba', 'available': False, 'reason': f'numba import failed: {exc!r}'}
        else:
            try:
                import cupy as cp
                if int(cp.cuda.runtime.getDeviceCount()) < 1:
                    raise RuntimeError('no CUDA device visible')
                dev = cp.cuda.Device()
                cc = int(dev.compute_capability)
                if cc < 60:
                    raise RuntimeError(f'compute capability {cc} < 60 required for FP64 atomicAdd')
                st = {'backend': 'cuda', 'available': True, 'reason': None, 'cupy_version': cp.__version__,
                      'compute_capability': cc, 'device_id': int(dev.id),
                      'note': 'maps bind to the current device at build time; no device switch or peer access'}
            except Exception as exc:  # pragma: no cover - machine dependent
                st = {'backend': 'cuda', 'available': False, 'reason': f'cupy/CUDA unavailable: {exc!r}'}
        _BACKEND[backend] = st
    return dict(st)


def _require_backend(backend):
    if backend not in BACKENDS:
        raise UptakeError(f'backend must be one of {BACKENDS}, got {backend!r}')
    st = backend_status(backend)
    if not st['available']:
        raise BackendUnavailableError(f'backend {backend!r} is not available: {st["reason"]} (no implicit fallback)')
    return st


def _xp(backend):
    if backend == 'cuda':
        import cupy
        return cupy
    return np


def _cupy():
    try:
        import cupy
        return cupy
    except Exception:
        return None


def _current_device():
    import cupy
    return int(cupy.cuda.Device().id)


def _require_current_device(device_id, what):
    cur = _current_device()
    if cur != device_id:
        raise UptakeError(f'{what}: current CUDA device {cur} differs from the map device {device_id}; the map is bound '
                          'to one device and never switches devices')


def _check_device_array(a, name, device_id):
    did = int(a.device.id)
    if did != device_id:
        raise UptakeError(f'{name}: CuPy array lives on device {did}, map is bound to device {device_id}; refused before '
                          'any allocation or launch (no peer access)')


def _numba_kernels():
    if 'numba' not in _COMPILED:
        from numba import njit
        _COMPILED['numba'] = {name: njit(cache=True)(fn) for name, fn in (
            ('check_demand', _k_check_demand), ('scatter', _k_scatter), ('proposal', _k_proposal),
            ('scale', _k_scale), ('gather', _k_gather))}
    return _COMPILED['numba']


def _cuda_kernels(device_id):
    key = ('cuda', int(device_id))
    if key not in _COMPILED:
        import cupy as cp
        _require_current_device(device_id, 'kernel compilation')
        mod = cp.RawModule(code=_CUDA_SRC, options=('-fmad=false',))
        mod.compile()
        _COMPILED[key] = {name: mod.get_function(name) for name in _CUDA_KERNELS}
        _COMPILED[key]['module'] = mod
    return _COMPILED[key]


def _blocks(n):
    return max(1, min((int(n) + THREADS - 1) // THREADS, MAX_BLOCKS))


def _is_device(a):
    return hasattr(a, '__cuda_array_interface__')


def _sync_packet(packet):
    '''The ONE explicit host readback of an operation: a small FP64 packet (never a P/E/R array). On the
    host backends this is a no-op view.'''
    if _is_device(packet):
        return np.asarray(packet.get(), dtype=np.float64)   # explicit synchronisation boundary
    return np.asarray(packet, dtype=np.float64)


# ------------------------------------------------------------------ memory estimates (pure, itemised)
def _int_bound(v, name, lo=0):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
        raise UptakeError(f'{name} must be an integer, got {v!r}')
    v = int(v)
    if v < lo:
        raise UptakeError(f'{name} must be >= {lo}, got {v}')
    return v


def _estimate(stage, P, E, R, backend, owned, scratch, staging, scope):
    return {'stage': stage, 'n_plants': P, 'n_edges': E, 'n_resources': R, 'backend': backend,
            'owned_bytes': int(sum(owned.values())), 'scratch_bytes': int(sum(scratch.values())),
            'staging_bytes': int(sum(staging.values())), 'mirror_bytes': 0,
            'total_bytes': int(sum(owned.values()) + sum(scratch.values()) + sum(staging.values())),
            'owned': {k: int(v) for k, v in owned.items()}, 'scratch': {k: int(v) for k, v in scratch.items()},
            'staging': {k: int(v) for k, v in staging.items()},
            'resident_on': 'device' if backend == 'cuda' else 'host', 'scope': scope,
            'exclusions': MEMORY_SCOPE}


def estimate_map_bytes(n_plants, n_edges, n_resources, backend='numpy', *, host_source=True, owned_resident=False):
    '''Bytes to construct a RootMap. owned = ids[P] + offsets[P+1] + counts[P] (int64) + resources[E] (int64) +
    weights[E] (float64). scratch (validation on the input namespace, freed afterwards, summed as a conservative
    upper bound of the peak): offsets order mask [P] bool, id uniqueness sort [P] int64 + mask, weight finite mask
    [E] bool, plant-of-edge [E] int64, float64 weight view for bincount [E], row sums [P] + |row - 1| [P] + bad mask
    [P], resource edge counts [R] int64. staging = a second owned set on the device when a cuda map is loaded from
    host arrays. With owned_resident=True (internal adoption of freshly built arrays already counted by the
    caller) the owned set is reported but not added to the total.'''
    P = _int_bound(n_plants, 'n_plants')
    E = _int_bound(n_edges, 'n_edges')
    R = _int_bound(n_resources, 'n_resources')
    owned = {'ids': 8 * P, 'offsets': 8 * (P + 1), 'counts': 8 * P, 'resources': 8 * E, 'weights': 8 * E}
    scratch = {'offsets_order_mask': P, 'id_unique_sort': 8 * P + P, 'weight_finite_mask': E, 'plant_of_edge': 8 * E,
               'weights_f64_view': 8 * E, 'row_sums': 8 * P, 'row_deviation': 8 * P, 'row_bad_mask': P,
               'resource_edge_counts': 8 * R}
    staging = {'device_owned_during_host_load': sum(owned.values())} if (backend == 'cuda' and host_source) else {}
    est = _estimate('map', P, E, R, backend, owned, scratch, staging,
                    'owned map arrays + validation scratch (freed) + transfer staging; caller inputs excluded; no mirrors')
    if owned_resident:
        est['owned_already_resident'] = True
        est['total_bytes'] = est['scratch_bytes'] + est['staging_bytes']
    return est


def estimate_plan_bytes(n_plants, n_edges, n_resources, backend='numpy'):
    '''Bytes for `propose`. owned = demand[P] + request[R] + proposed[R] + fraction[R] + packet. scratch =
    supply float64 copy [R] (freed) and, for the numpy reference only, its vectorised temporaries (per-edge
    demand + term [2 x 8E], demand masks / subset [6P bool + 8P], supply/request masks and temporaries
    [6R bool + 5 x 8R]). numba / cuda kernels allocate nothing beyond the owned arrays.'''
    P = _int_bound(n_plants, 'n_plants')
    E = _int_bound(n_edges, 'n_edges')
    R = _int_bound(n_resources, 'n_resources')
    owned = {'demand': 8 * P, 'request': 8 * R, 'proposed': 8 * R, 'fraction': 8 * R, 'packet': 8 * NP_PACKET}
    scratch = {'supply_f64_copy': 8 * R}
    if backend == 'numpy':
        scratch.update({'reference_edge_temporaries': 16 * E, 'reference_demand_temporaries': 6 * P + 8 * P,
                        'reference_resource_temporaries': 6 * R + 40 * R})
    return _estimate('plan', P, E, R, backend, owned, scratch, {},
                     'owned plan arrays + supply copy + numpy-reference temporaries; the resident map is added by admit()')


def estimate_feedback_bytes(n_plants, n_edges, n_resources, backend='numpy'):
    '''Bytes for `accept`. owned = accepted copy [R] + scale [R] + delivered / unmet / satisfaction [3 x 8P] +
    packet. scratch: numba / cuda kernels fuse validation, gather and the per-plant outputs (nothing beyond the
    owned arrays); the numpy reference adds per-edge temporaries [4 x 8E], per-resource temporaries [4 x 8R +
    6R bool] and per-plant temporaries [3 x 8P + 6P bool].'''
    P = _int_bound(n_plants, 'n_plants')
    E = _int_bound(n_edges, 'n_edges')
    R = _int_bound(n_resources, 'n_resources')
    owned = {'accepted': 8 * R, 'scale': 8 * R, 'delivered': 8 * P, 'unmet': 8 * P, 'satisfaction': 8 * P,
             'packet': 8 * NA_PACKET}
    scratch = {}
    if backend == 'numpy':
        scratch = {'reference_edge_temporaries': 32 * E, 'reference_resource_temporaries': 32 * R + 6 * R,
                   'reference_plant_temporaries': 24 * P + 6 * P}
    return _estimate('feedback', P, E, R, backend, owned, scratch, {},
                     'owned feedback arrays + numpy-reference temporaries; resident map and plan are added by admit()')


def estimate_conversion_bytes(n_plants, n_edges, n_resources, source_backend, target_backend):
    '''Bytes for RootMap.with_backend: the candidate's owned set (the source map stays resident and is passed
    as resident_bytes); host -> cuda transfers from the resident host arrays, cuda -> host reads back into the
    candidate's host arrays, cuda -> cuda (same device) copies on the device; no extra staging.'''
    est = estimate_map_bytes(n_plants, n_edges, n_resources, target_backend, host_source=False)
    est['stage'] = 'conversion'
    est['scratch'] = {}
    est['scratch_bytes'] = 0
    est['staging'] = {}
    est['staging_bytes'] = 0
    est['total_bytes'] = est['owned_bytes']
    est['scope'] = (f'candidate owned arrays on {target_backend!r} coexisting with the resident {source_backend!r} map '
                    '(resident_bytes); no validation scratch (values are already validated), no extra staging')
    return est


def admit(stage, estimate, *, max_edges, max_bytes, resident_bytes=0):
    '''Refuse BEFORE allocation when n_edges > max_edges or resident + estimated total > max_bytes.'''
    me = _int_bound(max_edges, 'max_edges')
    mb = _int_bound(max_bytes, 'max_bytes')
    total = int(resident_bytes) + int(estimate['total_bytes'])
    report = dict(estimate, resident_bytes=int(resident_bytes), admitted_total_bytes=total,
                  max_edges=me, max_bytes=mb)
    if int(estimate['n_edges']) > me:
        raise AdmissionError(stage, f'{estimate["n_edges"]} edges exceed max_edges={me}; nothing allocated', report)
    if total > mb:
        raise AdmissionError(stage, f'resident {int(resident_bytes)} + estimated {int(estimate["total_bytes"])} bytes '
                                    f'exceed max_bytes={mb}; nothing allocated', report)
    return report


def _budget(cap, max_bytes, what):
    '''An operation-level max_bytes may only tighten the map cap.'''
    if max_bytes is None:
        return int(cap)
    mb = _int_bound(max_bytes, 'max_bytes')
    if mb > int(cap):
        raise UptakeError(f'{what}: max_bytes={mb} exceeds the map cap {int(cap)}; an operation budget may only tighten it')
    return mb


def closure_tolerance_m3(total_m3, n_plants, n_edges, n_resources):
    '''FP64 summation tolerance for sum(delivered) == sum(accepted): eps * (2E + R + P + 8) * |total|. The
    factor counts every rounding in the chain (per-edge scatter adds and gather products, per-resource ratio /
    proposal, the sequential or pairwise reductions over R and P); proportional to machine epsilon only.'''
    return FP64_EPS * (2 * int(n_edges) + int(n_resources) + int(n_plants) + 8) * abs(float(total_m3))


# ------------------------------------------------------------------ typed-array admission helpers (no materialisation)
def _typed_array(v, name, backend, *, allow_host_load=False, device_id=None):
    '''Return (array, is_device) WITHOUT copying or converting. Only numpy.ndarray (host) and cupy.ndarray
    (device) are accepted: sequences / scalars / foreign objects are refused without being materialised.
    Host backends refuse device arrays; the cuda backend refuses host arrays unless `allow_host_load` (the
    explicit map load) and checks device arrays against `device_id`.'''
    if isinstance(v, np.ndarray):
        if backend == 'cuda' and not allow_host_load:
            raise UptakeError(f'{name}: cuda backend received a host array; no implicit host->device transfer '
                              '(pass a CuPy array on the map device; RootMap.from_arrays(backend="cuda") / '
                              'with_backend("cuda") are the explicit loads)')
        if v.dtype == object:
            raise UptakeError(f'{name}: object arrays are refused')
        return v, False
    if _is_device(v):
        cp = _cupy()
        if cp is None or not isinstance(v, cp.ndarray):
            raise UptakeError(f'{name}: device arrays must be cupy.ndarray instances')
        if backend != 'cuda':
            raise UptakeError(f'{name}: host backend {backend!r} received a device array; no implicit device->host transfer')
        if device_id is not None:
            _check_device_array(v, name, device_id)
        return v, True
    raise UptakeError(f'{name}: a typed array (numpy.ndarray or cupy.ndarray) is required; Python sequences and '
                      f'scalars are refused without conversion (got {type(v).__name__})')


def _check_int_meta(a, name):
    if a.ndim != 1:
        raise UptakeError(f'{name} must be 1-D, got ndim {a.ndim}')
    k = a.dtype.kind
    if k == 'b':
        raise UptakeError(f'{name} must be an integer array, got bool')
    if k == 'f':
        raise UptakeError(f'{name} must be an integer array, got float dtype {a.dtype} (fractional/lossy casts refused)')
    if k not in 'iu':
        raise UptakeError(f'{name} must be an integer array, got dtype {a.dtype}')


def _check_real_meta(a, name, shape=None):
    k = a.dtype.kind
    if k == 'b' or k not in 'fiu':
        raise UptakeError(f'{name} must be a real numeric array (not bool/complex/object), got dtype {a.dtype}')
    if shape is not None and tuple(int(s) for s in a.shape) != tuple(shape):
        raise UptakeError(f'{name} must have shape {tuple(shape)}, got {tuple(a.shape)}')


def _check_int64_range(a, name):
    '''Unsigned values above int64 are refused BEFORE any cast (a scalar reduction).'''
    if a.dtype.kind == 'u' and a.size and int(a.max()) > INT64_MAX:
        raise UptakeError(f'{name} holds values above the int64 range (unsigned overflow refused before cast)')


def _freeze(a):
    if isinstance(a, np.ndarray):
        a.setflags(write=False)
    return a


def _preview_ids(ids, mask, xp):
    idx = xp.flatnonzero(mask)[:ID_PREVIEW]
    sel = ids[idx]
    return [int(v) for v in np.asarray(sel.get() if _is_device(sel) else sel)]


def _finite_scalars(values, what):
    for k, v in values.items():
        if not math.isfinite(float(v)):
            raise UptakeError(f'{what}: aggregate {k!r} is not finite ({v!r}); refusing to return a non-finite result')


# ------------------------------------------------------------------ root map
class RootMap:
    '''Immutable validated plant-major CSR root map (see the module docstring). Construct with `from_arrays`.'''

    _FIELDS = ('backend', 'device_id', 'xp', 'ids', 'offsets', 'counts', 'resources', 'weights', 'n_plants', 'n_edges',
               'n_layers', 'n_cells', 'n_resources', 'max_edges', 'max_bytes', 'estimate', 'admission',
               'max_edges_per_plant', 'max_edges_per_resource', 'n_inactive_plants', 'provenance')

    def __init__(self, **kw):
        for k in self._FIELDS:
            object.__setattr__(self, k, kw[k])
        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name, value):
        raise AttributeError('RootMap is immutable (build a new map; growth-driven rebuilds are a later driver operation)')

    def __delattr__(self, name):
        raise AttributeError('RootMap is immutable')

    @classmethod
    def from_arrays(cls, ids, offsets, resources, weights, *, n_layers, n_cells, backend='numpy', max_edges,
                    max_bytes, provenance=None):
        '''Validate (metadata -> admission -> content on the inputs -> owned copies) and build a frozen map.

        ids [P] int64 stable plant IDs (unique); offsets [P+1] int64 with offsets[0] == 0, offsets[-1] == E,
        every value in [0, E] and nondecreasing (checked by direct comparison before any subtraction);
        resources [E] int64 in [0, n_layers * n_cells); weights [E] FP64 finite >= 0, each non-empty row
        summing to 1 within WEIGHT_SUM_TOL. Typed arrays only (numpy.ndarray for every backend; cupy.ndarray
        on the current device for `cuda`). For `cuda` host inputs are validated on the host and transferred
        ONCE after admission (the explicit load); device inputs are validated on the device with scalar
        reductions. Nothing is copied, cast or transferred before admission passes.'''
        return _build_map(ids, offsets, resources, weights, n_layers=n_layers, n_cells=n_cells, backend=backend,
                          max_edges=max_edges, max_bytes=max_bytes, provenance=provenance, adopt=False,
                          resident_bytes=0)

    # ---- properties
    @property
    def resident_bytes(self):
        '''Bytes of the owned map arrays resident on the map backend.'''
        return int(self.estimate['owned_bytes'])

    def __repr__(self):
        dev = '' if self.device_id is None else f', device={self.device_id}'
        return (f'RootMap(P={self.n_plants}, E={self.n_edges}, L={self.n_layers}, N={self.n_cells}, '
                f'backend={self.backend!r}{dev})')

    def describe(self):
        return {'n_plants': self.n_plants, 'n_edges': self.n_edges, 'n_layers': self.n_layers, 'n_cells': self.n_cells,
                'n_resources': self.n_resources, 'backend': self.backend, 'device_id': self.device_id,
                'n_inactive_plants': self.n_inactive_plants, 'max_edges_per_plant': self.max_edges_per_plant,
                'max_edges_per_resource': self.max_edges_per_resource, 'resident_bytes': self.resident_bytes,
                'max_edges': self.max_edges, 'max_bytes': self.max_bytes, 'weight_sum_tol': WEIGHT_SUM_TOL,
                'resource_index': 'layer * n_cells + cell (row-major cell index)', 'units': dict(UNITS),
                'algorithm': ALGORITHM, 'memory_scope': MEMORY_SCOPE, 'provenance': dict(self.provenance),
                'admission': dict(self.admission)}

    def host_arrays(self):
        '''DIAGNOSTIC / test helper: host copies of the map arrays (an explicit device->host transfer for cuda
        maps; never used by the per-step path).'''
        out = {}
        for k in ('ids', 'offsets', 'counts', 'resources', 'weights'):
            a = getattr(self, k)
            out[k] = np.array(a.get() if _is_device(a) else a, copy=True)
        return out

    def with_backend(self, backend, *, max_bytes=None):
        '''Rebuild this (already validated) map on another backend. Admission BEFORE any copy or transfer:
        candidate owned arrays + the resident source map; `max_bytes` may only tighten the map cap. cuda ->
        cuda copies on the bound device (no host staging); host <-> cuda is the one explicit transfer.'''
        _require_backend(backend)
        budget = _budget(self.max_bytes, max_bytes, 'with_backend')
        est = estimate_conversion_bytes(self.n_plants, self.n_edges, self.n_resources, self.backend, backend)
        adm = admit('uptake.RootMap.with_backend', est, max_edges=self.max_edges, max_bytes=budget,
                    resident_bytes=self.resident_bytes)
        device_id = None
        if backend == 'cuda':
            device_id = _current_device()
            if self.backend == 'cuda' and self.device_id != device_id:
                raise UptakeError(f'with_backend: current device {device_id} differs from the source map device '
                                  f'{self.device_id}; no device switch or peer copy')
        elif self.backend == 'cuda':
            _require_current_device(self.device_id, 'with_backend')
        xp = _xp(backend)
        arrays = {}
        for k in ('ids', 'offsets', 'counts', 'resources', 'weights'):
            a = getattr(self, k)
            if backend == 'cuda':
                arrays[k] = xp.array(a, copy=True) if _is_device(a) else xp.asarray(a)   # device copy / explicit H2D
            else:
                arrays[k] = np.array(a.get() if _is_device(a) else a, copy=True)        # explicit D2H / host copy
            _freeze(arrays[k])
        new_est = estimate_map_bytes(self.n_plants, self.n_edges, self.n_resources, backend, host_source=False)
        return RootMap(backend=backend, device_id=device_id, xp=xp, n_plants=self.n_plants, n_edges=self.n_edges,
                       n_layers=self.n_layers, n_cells=self.n_cells, n_resources=self.n_resources,
                       max_edges=self.max_edges, max_bytes=budget, estimate=new_est, admission=adm,
                       max_edges_per_plant=self.max_edges_per_plant, max_edges_per_resource=self.max_edges_per_resource,
                       n_inactive_plants=self.n_inactive_plants,
                       provenance=dict(self.provenance, converted_from=self.backend), **arrays)


def _build_map(ids, offsets, resources, weights, *, n_layers, n_cells, backend, max_edges, max_bytes, provenance,
               adopt, resident_bytes):
    '''Shared validation / construction. adopt=False: public path, owned copies are made after validation.
    adopt=True: internal trusted transfer of FRESH int64/int64/int64/float64 C-contiguous host arrays whose only
    references are dropped by the caller (leaf.roots); they are frozen in place (no copy), and the caller has
    already counted them as resident.'''
    _require_backend(backend)
    xp = _xp(backend)
    L = _int_bound(n_layers, 'n_layers', 1)
    N = _int_bound(n_cells, 'n_cells', 1)
    if L > INT64_MAX // N:
        raise UptakeError(f'n_layers * n_cells = {L} * {N} overflows the int64 resource index')
    R = L * N
    device_id = _current_device() if backend == 'cuda' else None
    ids_a, dev = _typed_array(ids, 'ids', backend, allow_host_load=True, device_id=device_id)
    off_a, d2 = _typed_array(offsets, 'offsets', backend, allow_host_load=True, device_id=device_id)
    res_a, d3 = _typed_array(resources, 'resources', backend, allow_host_load=True, device_id=device_id)
    w_a, d4 = _typed_array(weights, 'weights', backend, allow_host_load=True, device_id=device_id)
    if len({dev, d2, d3, d4}) != 1:
        raise UptakeError('ids/offsets/resources/weights must all be host arrays or all be device arrays')
    if adopt:
        if dev:
            raise UptakeError('internal adoption expects host arrays')
        for a, dt, name in ((ids_a, np.int64, 'ids'), (off_a, np.int64, 'offsets'), (res_a, np.int64, 'resources'),
                            (w_a, np.float64, 'weights')):
            if a.dtype != dt or not a.flags.c_contiguous or not a.flags.owndata:
                raise UptakeError(f'internal adoption requires fresh C-contiguous {np.dtype(dt)} {name}')
    # ---- metadata (no data read, no allocation)
    for a, name in ((ids_a, 'ids'), (off_a, 'offsets'), (res_a, 'resources')):
        _check_int_meta(a, name)
    if w_a.ndim != 1:
        raise UptakeError(f'weights must be 1-D, got ndim {w_a.ndim}')
    _check_real_meta(w_a, 'weights')
    P = int(ids_a.shape[0])
    E = int(res_a.shape[0])
    if int(off_a.shape[0]) != P + 1:
        raise UptakeError(f'offsets must have length n_plants + 1 = {P + 1}, got {int(off_a.shape[0])}')
    if int(w_a.shape[0]) != E:
        raise UptakeError(f'weights length {int(w_a.shape[0])} differs from resources length {E}')
    # ---- admission BEFORE any cast / copy / scratch / transfer
    est = estimate_map_bytes(P, E, R, backend, host_source=not dev, owned_resident=adopt)
    adm = admit('uptake.RootMap.from_arrays' if not adopt else 'uptake.adopt_owned_arrays', est, max_edges=max_edges,
                max_bytes=max_bytes, resident_bytes=resident_bytes)
    vxp = xp if dev else np   # validation namespace: where the inputs live
    # ---- content validation ON THE INPUTS (scalar reductions only); no subtraction before the bounds are known
    for a, name in ((ids_a, 'ids'), (off_a, 'offsets'), (res_a, 'resources')):
        _check_int64_range(a, name)
    if int(off_a[0]) != 0 or int(off_a[-1]) != E:
        raise UptakeError(f'offsets must start at 0 and end at the edge count {E} (got {int(off_a[0])}, {int(off_a[-1])})')
    if int(off_a.min()) < 0 or int(off_a.max()) > E:
        raise UptakeError(f'every offset must lie in [0, {E}] (got range [{int(off_a.min())}, {int(off_a.max())}])')
    if P and not bool((off_a[1:] >= off_a[:-1]).all()):
        raise UptakeError('offsets must be nondecreasing')
    if P and int(vxp.unique(ids_a).shape[0]) != P:
        raise UptakeError('duplicate plant IDs in the root map are refused (one row per individual)')
    if E:
        if int(res_a.min()) < 0 or int(res_a.max()) >= R:
            raise UptakeError(f'resource indices must lie in [0, n_layers * n_cells = {R})')
        if not bool(vxp.isfinite(w_a).all()):
            raise UptakeError('weights must be finite')
        if float(w_a.min()) < 0.0:
            raise UptakeError('weights must be >= 0')
    # offsets are now known to be in range and ordered: counts cannot wrap
    off64 = off_a if (adopt or off_a.dtype == np.int64) and not dev else vxp.array(off_a, dtype=vxp.int64, copy=True)
    counts = vxp.diff(off64)
    if E:
        plant_of_edge = vxp.repeat(vxp.arange(P, dtype=vxp.int64), counts)
        row = vxp.bincount(plant_of_edge, weights=w_a, minlength=P)
        del plant_of_edge
        bad = (counts > 0) & (vxp.abs(row - 1.0) > WEIGHT_SUM_TOL)
        if bool(bad.any()):
            raise UptakeError(f'root rows must sum to 1 within {WEIGHT_SUM_TOL} (not renormalised); offending plant ids '
                              f'{_preview_ids(ids_a, bad, vxp)}')
        del row, bad
        max_res = int(vxp.bincount(res_a, minlength=R).max())
    else:
        max_res = 0
    max_plant = int(counts.max()) if P else 0
    n_inactive = int(vxp.count_nonzero(counts == 0)) if P else 0
    # ---- ownership: casts / copies after validation; cuda-from-host performs the one explicit transfer
    if adopt:
        ids64, res64, w64 = ids_a, res_a, w_a
        counts = np.array(counts, dtype=np.int64, copy=True)
    else:
        ids64 = vxp.array(ids_a, dtype=vxp.int64, copy=True)
        res64 = vxp.array(res_a, dtype=vxp.int64, copy=True)
        w64 = vxp.array(w_a, dtype=vxp.float64, copy=True)
        if off64 is off_a:
            off64 = vxp.array(off_a, dtype=vxp.int64, copy=True)
        counts = vxp.array(counts, dtype=vxp.int64, copy=True)
    if backend == 'cuda' and not dev:
        ids64, off64, res64, w64, counts = (xp.asarray(a) for a in (ids64, off64, res64, w64, counts))
    for a in (ids64, off64, res64, w64, counts):
        _freeze(a)
    prov = dict(provenance) if provenance else {}
    return RootMap(backend=backend, device_id=device_id, xp=xp, ids=ids64, offsets=off64, counts=counts, resources=res64,
                   weights=w64, n_plants=P, n_edges=E, n_layers=L, n_cells=N, n_resources=R, max_edges=int(max_edges),
                   max_bytes=int(max_bytes), estimate=est, admission=adm, max_edges_per_plant=max_plant,
                   max_edges_per_resource=max_res, n_inactive_plants=n_inactive, provenance=prov)


def adopt_owned_arrays(ids, offsets, resources, weights, *, n_layers, n_cells, backend, max_edges, max_bytes,
                       provenance=None, resident_bytes=0):
    '''INTERNAL trusted ownership transfer (used by leaf.roots): validate FRESH int64/int64/int64/float64
    C-contiguous host arrays in place, freeze them and build the map without copying. The caller must hold no
    other references (it has counted the arrays as `resident_bytes`); the frozen arrays therefore leave no
    caller-writable alias. Public callers use RootMap.from_arrays.'''
    return _build_map(ids, offsets, resources, weights, n_layers=n_layers, n_cells=n_cells, backend=backend,
                      max_edges=max_edges, max_bytes=max_bytes, provenance=provenance, adopt=True,
                      resident_bytes=resident_bytes)


# ------------------------------------------------------------------ backend dispatch (one algorithm, three paths)
def _np_check_demand(counts, D, packet):
    fin = np.isfinite(D)
    packet[0] = np.count_nonzero(~fin)
    packet[1] = np.count_nonzero(fin & (D < 0.0))
    pos = fin & (D > 0.0)
    packet[2] = np.count_nonzero(pos)
    packet[3] = float(D[pos].sum()) if pos.any() else 0.0
    packet[4] = np.count_nonzero(pos & (counts == 0))


def _np_scatter(m, D):
    if m.n_edges == 0:
        return np.zeros(m.n_resources, dtype=np.float64)
    term = np.repeat(D, m.counts)
    term *= m.weights
    return np.bincount(m.resources, weights=term, minlength=m.n_resources)


def _np_proposal(A, Q, proposed, fraction, packet):
    fa = np.isfinite(A)
    packet[5] = np.count_nonzero(~fa)
    packet[6] = np.count_nonzero(fa & (A < 0.0))
    ok = fa & (A >= 0.0)
    packet[8] = float(A[ok].sum()) if ok.any() else 0.0
    fq = np.isfinite(Q)
    packet[7] = np.count_nonzero(~fq)
    packet[9] = float(Q[fq].sum()) if fq.any() else 0.0
    pos = fq & (Q > 0.0)
    packet[11] = np.count_nonzero(pos)
    packet[12] = np.count_nonzero(pos & (Q > A))
    proposed[...] = 0.0
    fraction[...] = 0.0
    proposed[pos] = np.minimum(A[pos], Q[pos])
    fraction[pos] = proposed[pos] / Q[pos]
    packet[10] = float(proposed.sum())


def _np_scale(acc, Q, proposed, tol_rel, scale, packet):
    fa = np.isfinite(acc)
    packet[0] = np.count_nonzero(~fa)
    neg = fa & (acc < 0.0)
    packet[1] = np.count_nonzero(neg)
    ok = fa & ~neg
    packet[5] = float(acc[ok].sum()) if ok.any() else 0.0
    ex = acc - proposed
    posx = ok & (ex > 0.0)
    packet[2] = np.count_nonzero(posx)
    packet[3] = float(ex[posx].max()) if posx.any() else 0.0
    packet[4] = np.count_nonzero(posx & (ex > tol_rel * proposed))
    pos = Q > 0.0
    scale[...] = 0.0
    scale[pos & ok] = acc[pos & ok] / Q[pos & ok]
    packet[6] = np.count_nonzero(ok & ~pos & (acc > 0.0))


def _np_gather(m, D, scale):
    if m.n_plants == 0:
        return np.zeros(0, dtype=np.float64)
    if m.n_edges == 0:
        return np.zeros(m.n_plants, dtype=np.float64)
    term = np.repeat(D, m.counts)
    term *= m.weights
    term *= scale[m.resources]
    plant_of_edge = np.repeat(np.arange(m.n_plants, dtype=np.int64), m.counts)
    return np.bincount(plant_of_edge, weights=term, minlength=m.n_plants)


def _np_post(D, over_rel, delivered, unmet, satisfaction, packet):
    fin = np.isfinite(delivered)
    packet[14] = np.count_nonzero(~fin)
    packet[7] = float(delivered[fin].sum()) if fin.any() else 0.0
    pos = fin & (D > 0.0)
    packet[9] = np.count_nonzero(pos & (delivered > D * (1.0 + over_rel)))
    clamp = pos & (delivered > D)
    packet[10] = np.count_nonzero(clamp)
    packet[11] = float((delivered[clamp] - D[clamp]).max()) if clamp.any() else 0.0
    np.subtract(D, delivered, out=unmet)
    np.maximum(unmet, 0.0, out=unmet)
    unmet[~pos] = 0.0
    satisfaction[...] = 0.0
    np.divide(delivered, D, out=satisfaction, where=pos)
    np.minimum(satisfaction, 1.0, out=satisfaction)
    packet[12] = np.count_nonzero(pos & (delivered < D))
    packet[13] = np.count_nonzero(fin & (D == 0.0))
    packet[8] = float(unmet.sum())


def _run_propose(m, D, A, request, proposed, fraction, packet):
    if m.backend == 'numpy':
        _np_check_demand(m.counts, D, packet)
        request[...] = _np_scatter(m, D)
        _np_proposal(A, request, proposed, fraction, packet)
    elif m.backend == 'numba':
        k = _numba_kernels()
        k['check_demand'](m.counts, D, packet)
        k['scatter'](m.offsets, m.resources, m.weights, D, request)
        k['proposal'](A, request, proposed, fraction, packet)
    else:
        k = _cuda_kernels(m.device_id)
        bp, br = (_blocks(m.n_plants),), (_blocks(m.n_resources),)
        k['check_demand'](bp, (THREADS,), (m.counts, D, packet, np.int64(m.n_plants)))
        k['scatter'](bp, (THREADS,), (m.offsets, m.resources, m.weights, D, request, np.int64(m.n_plants)))
        k['proposal'](br, (THREADS,), (A, request, proposed, fraction, packet, np.int64(m.n_resources)))


def _run_accept(m, D, acc, request, proposed, tol_rel, over_rel, scale, delivered, unmet, satisfaction, packet):
    if m.backend == 'numpy':
        _np_scale(acc, request, proposed, tol_rel, scale, packet)
        delivered[...] = _np_gather(m, D, scale)
        _np_post(D, over_rel, delivered, unmet, satisfaction, packet)
    elif m.backend == 'numba':
        k = _numba_kernels()
        k['scale'](acc, request, proposed, tol_rel, scale, packet)
        k['gather'](m.offsets, m.resources, m.weights, D, scale, over_rel, delivered, unmet, satisfaction, packet)
    else:
        k = _cuda_kernels(m.device_id)
        k['scale_kernel']((_blocks(m.n_resources),), (THREADS,),
                          (acc, request, proposed, np.float64(tol_rel), scale, packet, np.int64(m.n_resources)))
        k['gather']((_blocks(m.n_plants),), (THREADS,),
                    (m.offsets, m.resources, m.weights, D, scale, np.float64(over_rel), delivered, unmet, satisfaction,
                     packet, np.int64(m.n_plants)))


def _resource_array(v, name, m):
    '''(L, N) or (R,) typed real array on the map backend (no copy here).'''
    a, _ = _typed_array(v, name, m.backend, device_id=m.device_id)
    _check_real_meta(a, name)
    shape = tuple(int(s) for s in a.shape)
    if shape not in ((m.n_layers, m.n_cells), (m.n_resources,)):
        raise UptakeError(f'{name} must have shape (n_layers, n_cells) = {(m.n_layers, m.n_cells)} or ({m.n_resources},), got {shape}')
    return a


# ------------------------------------------------------------------ plan (proposal)
class UptakePlan:
    '''Pure proposal bound to one RootMap and one owned demand vector. Arrays live on the map backend:
    demand [P], request Q [R], proposed [R], fraction f [R] (R = n_layers * n_cells, layer-major). `summary`
    holds scalar totals (from the operation packet) only.'''

    def __init__(self, root_map, demand, request, proposed, fraction, summary):
        object.__setattr__(self, 'root_map', root_map)
        object.__setattr__(self, 'demand', _freeze(demand))
        object.__setattr__(self, 'request', _freeze(request))
        object.__setattr__(self, 'proposed', _freeze(proposed))
        object.__setattr__(self, 'fraction', _freeze(fraction))
        object.__setattr__(self, 'summary', summary)

    def __setattr__(self, name, value):
        raise AttributeError('UptakePlan is immutable')

    @property
    def backend(self):
        return self.root_map.backend

    @property
    def resident_bytes(self):
        return int(self.summary['admission']['owned_bytes'])

    def proposed_grid(self):
        '''proposed reshaped to (n_layers, n_cells) (a view on the owned array).'''
        return self.proposed.reshape(self.root_map.n_layers, self.root_map.n_cells)

    def accept(self, accepted, *, max_bytes=None):
        return accept(self, accepted, max_bytes=max_bytes)

    def accept_all(self, *, max_bytes=None):
        '''Feedback for full acceptance (accepted == proposed): the no-shortfall case.'''
        return accept(self, self.proposed, max_bytes=max_bytes)


def propose(root_map, demand, supply, *, max_bytes=None):
    '''Proposal for one timestep: demand D [P] m3 and supply A [(L, N) or (R,)] m3 -> UptakePlan with Q,
    proposed = min(A, Q) and f. Pure: no state, no water debit. Typed arrays on the map backend only.
    Refused: non-finite / negative demand or supply, positive demand on an inactive (edge-less) root row,
    non-finite request (overflow) or totals. One packet readback (cuda: one synchronisation).'''
    if not isinstance(root_map, RootMap):
        raise UptakeError('root_map must be a leaf.uptake.RootMap')
    m, xp = root_map, root_map.xp
    d_in, _ = _typed_array(demand, 'demand', m.backend, device_id=m.device_id)
    _check_real_meta(d_in, 'demand', (m.n_plants,))
    a_in = _resource_array(supply, 'supply', m)
    budget = _budget(m.max_bytes, max_bytes, 'propose')
    est = estimate_plan_bytes(m.n_plants, m.n_edges, m.n_resources, m.backend)
    adm = admit('uptake.propose', est, max_edges=m.max_edges, max_bytes=budget, resident_bytes=m.resident_bytes)
    if m.backend == 'cuda':
        _require_current_device(m.device_id, 'propose')
    D = xp.array(d_in, dtype=xp.float64, copy=True)
    A = xp.array(a_in, dtype=xp.float64, copy=True).reshape(-1)
    packet = xp.zeros(NP_PACKET, dtype=xp.float64)
    request = xp.zeros(m.n_resources, dtype=xp.float64)
    proposed = xp.empty(m.n_resources, dtype=xp.float64)
    fraction = xp.empty(m.n_resources, dtype=xp.float64)
    _run_propose(m, D, A, request, proposed, fraction, packet)
    del A
    pk = _sync_packet(packet)
    if not np.all(np.isfinite(pk)):
        raise UptakeError('propose: non-finite aggregate packet (overflowing demand / supply / request totals); refused')
    g = lambda k: float(pk[_PP[k]])
    if g('demand_nonfinite') or g('demand_negative'):
        raise UptakeError('demand must be finite and >= 0')
    if g('supply_nonfinite') or g('supply_negative'):
        raise UptakeError('supply must be finite and >= 0')
    if g('n_positive_on_empty'):
        bad = (m.counts == 0) & (D > 0.0)
        raise UptakeError(f'positive demand on root rows without edges (inactive roots): plant ids '
                          f'{_preview_ids(m.ids, bad, xp)}; map the roots or set their demand to 0')
    if g('request_nonfinite'):
        raise UptakeError('request Q overflowed to a non-finite value on at least one resource; refusing the proposal')
    totals = {k: g(k) for k in ('demand_total_m3', 'supply_total_m3', 'request_total_m3', 'proposed_total_m3')}
    _finite_scalars(totals, 'propose')
    summary = {'n_plants': m.n_plants, 'n_edges': m.n_edges, 'n_resources': m.n_resources, 'backend': m.backend,
               'device_id': m.device_id, **totals,
               'n_plants_demanding': int(g('n_plants_demanding')), 'n_resources_requested': int(g('n_resources_requested')),
               'n_resources_limited': int(g('n_resources_limited')), 'admission': adm, 'units': dict(UNITS),
               'zero_demand_policy': ZERO_DEMAND_POLICY, 'no_plasticity': NO_PLASTICITY_NOTE,
               'synchronisation': 'one packet readback' if m.backend == 'cuda' else 'host'}
    return UptakePlan(m, D, request, proposed, fraction, summary)


# ------------------------------------------------------------------ feedback (accepted water)
class UptakeFeedback:
    '''Result of SYRUP's single acceptance: accepted (UNMODIFIED copy of the native debit) [R], scale
    accepted/Q [R], delivered [P], unmet [P], satisfaction S [P]; `closure` holds the FP64 water-closure check
    and `summary` the scalar diagnostics (roundoff excess, clamping, totals).'''

    def __init__(self, plan, accepted, scale, delivered, unmet, satisfaction, closure, summary):
        object.__setattr__(self, 'plan', plan)
        object.__setattr__(self, 'root_map', plan.root_map)
        object.__setattr__(self, 'accepted', _freeze(accepted))
        object.__setattr__(self, 'scale', _freeze(scale))
        object.__setattr__(self, 'delivered', _freeze(delivered))
        object.__setattr__(self, 'unmet', _freeze(unmet))
        object.__setattr__(self, 'satisfaction', _freeze(satisfaction))
        object.__setattr__(self, 'closure', closure)
        object.__setattr__(self, 'summary', summary)

    def __setattr__(self, name, value):
        raise AttributeError('UptakeFeedback is immutable')

    @property
    def backend(self):
        return self.plan.backend


def acceptance_tolerance_rel(root_map):
    '''Relative FP64 roundoff by which accepted[r] may exceed proposed[r] before being refused as material:
    eps * (max edges on one resource + 2) (the roundings of the Q sum, the min and a conversion).'''
    return FP64_EPS * (int(root_map.max_edges_per_resource) + 2)


def overdelivery_bound_rel(root_map):
    '''Documented relative bound by which delivered[p] may exceed D[p] through honest roundoff: the row-sum
    tolerance + the acceptance roundoff + eps * (max edges per plant + 4). S is capped at 1 and unmet at 0 only
    inside this bound; beyond it `accept` fails.'''
    return WEIGHT_SUM_TOL + acceptance_tolerance_rel(root_map) + FP64_EPS * (int(root_map.max_edges_per_plant) + 4)


def accept(plan, accepted, *, max_bytes=None):
    '''Turn SYRUP's accepted withdrawals [(L, N) or (R,)] m3 into delivered / unmet / S against the accepted
    amounts AS SUPPLIED (never modified). Refused: wrong backend / device / shape, non-finite or negative values,
    water on resources nobody requested, material excess over the proposal (beyond acceptance_tolerance_rel;
    roundoff-level excess is reported, not clipped), non-finite delivered water or totals, overdelivery beyond
    overdelivery_bound_rel, and any closure failure (sum(delivered) == sum(accepted) within
    closure_tolerance_m3). One packet readback (cuda: one synchronisation). Pure; nothing is published.'''
    if not isinstance(plan, UptakePlan):
        raise UptakeError('plan must be a leaf.uptake.UptakePlan produced by propose()')
    m, xp = plan.root_map, plan.root_map.xp
    a_in = _resource_array(accepted, 'accepted', m)
    budget = _budget(m.max_bytes, max_bytes, 'accept')
    est = estimate_feedback_bytes(m.n_plants, m.n_edges, m.n_resources, m.backend)
    adm = admit('uptake.accept', est, max_edges=m.max_edges, max_bytes=budget,
                resident_bytes=m.resident_bytes + plan.resident_bytes)
    if m.backend == 'cuda':
        _require_current_device(m.device_id, 'accept')
    acc = xp.array(a_in, dtype=xp.float64, copy=True).reshape(-1)     # the native debit, kept unchanged
    scale = xp.empty(m.n_resources, dtype=xp.float64)
    delivered = xp.empty(m.n_plants, dtype=xp.float64)
    unmet = xp.empty(m.n_plants, dtype=xp.float64)
    satisfaction = xp.empty(m.n_plants, dtype=xp.float64)
    packet = xp.zeros(NA_PACKET, dtype=xp.float64)
    tol_rel = acceptance_tolerance_rel(m)
    over_rel = overdelivery_bound_rel(m)
    _run_accept(m, plan.demand, acc, plan.request, plan.proposed, tol_rel, over_rel, scale, delivered, unmet,
                satisfaction, packet)
    pk = _sync_packet(packet)
    if not np.all(np.isfinite(pk)):
        raise UptakeError('accept: non-finite aggregate packet (overflowing accepted / delivered totals); refused')
    g = lambda k: float(pk[_AP[k]])
    if g('accepted_nonfinite') or g('accepted_negative'):
        raise UptakeError('accepted must be finite and >= 0')
    if g('n_unrequested_water'):
        raise UptakeError(f'accepted water on {int(g("n_unrequested_water"))} resource(s) nobody requested (Q == 0); refused')
    if g('n_material_excess'):
        raise UptakeError(f'accepted exceeds the proposal on {int(g("n_material_excess"))} resource(s) beyond the FP64 '
                          f'roundoff tolerance {tol_rel:.3e} (max excess {g("max_roundoff_excess_m3")!r} m3); material '
                          'excess is refused, nothing is clipped')
    if g('delivered_nonfinite'):
        raise UptakeError('delivered water is non-finite for at least one plant (overflow); refused')
    if g('n_over_bound'):
        raise UptakeError(f'implementation failure: delivered water exceeds demand beyond the documented bound '
                          f'{over_rel:.3e} on {int(g("n_over_bound"))} plant(s)')
    totals = {k: g(k) for k in ('accepted_total_m3', 'delivered_total_m3', 'unmet_total_m3', 'max_roundoff_excess_m3',
                                'max_overdelivery_m3')}
    _finite_scalars(totals, 'accept')
    acc_total, del_total = totals['accepted_total_m3'], totals['delivered_total_m3']
    residual = del_total - acc_total
    tol = closure_tolerance_m3(max(abs(acc_total), abs(del_total)), m.n_plants, m.n_edges, m.n_resources)
    closure = {'accepted_total_m3': acc_total, 'delivered_total_m3': del_total, 'residual_m3': residual,
               'tolerance_m3': tol, 'closed': bool(math.isfinite(residual) and math.isfinite(tol) and abs(residual) <= tol),
               'definition': 'sum(delivered) == sum(accepted as supplied) within closure_tolerance_m3 (FP64 summation '
                             'tolerance); accepted amounts are never modified'}
    if not closure['closed']:
        raise UptakeError(f'water closure violated: sum(delivered) - sum(accepted) = {residual!r} m3 exceeds the FP64 '
                          f'tolerance {tol!r} m3 (implementation failure, not a scientific result)')
    summary = {'n_plants': m.n_plants, 'backend': m.backend, 'device_id': m.device_id, **totals,
               'demand_total_m3': plan.summary['demand_total_m3'], 'proposed_total_m3': plan.summary['proposed_total_m3'],
               'shortfall_total_m3': plan.summary['proposed_total_m3'] - acc_total,
               'n_roundoff_excess': int(g('n_roundoff_excess')), 'acceptance_tolerance_rel': tol_rel,
               'n_plants_clamped': int(g('n_plants_clamped')), 'overdelivery_bound_rel': over_rel,
               'n_plants_unsatisfied': int(g('n_plants_unsatisfied')), 'n_plants_zero_demand': int(g('n_plants_zero_demand')),
               'admission': adm, 'units': dict(UNITS), 'zero_demand_policy': ZERO_DEMAND_POLICY,
               'accepted_policy': ACCEPTED_POLICY,
               'synchronisation': 'one packet readback' if m.backend == 'cuda' else 'host',
               'authority': 'stateless allocator result; state/receipt publication belongs to the later driver and '
                            'the native water authority'}
    return UptakeFeedback(plan, acc, scale, delivered, unmet, satisfaction, closure, summary)


__all__ = ['VERSION', 'BACKENDS', 'HOST_BACKENDS', 'FP64_EPS', 'WEIGHT_SUM_TOL', 'UNITS', 'ALGORITHM',
           'ZERO_DEMAND_POLICY', 'NO_PLASTICITY_NOTE', 'ACCEPTED_POLICY', 'MEMORY_SCOPE', 'PROPOSE_PACKET',
           'ACCEPT_PACKET', 'UptakeError', 'AdmissionError', 'BackendUnavailableError', 'backend_status',
           'estimate_map_bytes', 'estimate_plan_bytes', 'estimate_feedback_bytes', 'estimate_conversion_bytes', 'admit',
           'closure_tolerance_m3', 'acceptance_tolerance_rel', 'overdelivery_bound_rel', 'RootMap', 'adopt_owned_arrays',
           'UptakePlan', 'UptakeFeedback', 'propose', 'accept']
