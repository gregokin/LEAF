'''leaf.uptake: stateless sparse water-uptake allocator (proposal + accepted feedback) over a plant-major CSR root map.

Water contract (LEAF Phase 1 part A). Every amount is m3 of water for ONE timestep; depth/area conversion
belongs to the native SYRUP side (downstream), never to this module.

    R[p, r]     = D[p] * w[p, r]                 per-edge request; resource r = layer * n_cells + cell
    Q[r]        = sum_p R[p, r]                  local request on a resource (scatter)
    f[r]        = min(1, A[r] / Q[r])            shared fraction (0 when Q[r] == 0): every plant on a resource
                                                 receives the SAME fraction, no species / size / order priority
    proposed[r] = min(A[r], Q[r])
    SYRUP validates and accepts ONCE:            accepted[r] <= proposed[r]
    delivered[p] = sum_e D[p] * w[e] * accepted[r_e] / Q[r_e]      (0 where Q[r_e] == 0)   (gather)
    S[p]        = delivered[p] / D[p] for D[p] > 0, else 0           explicit zero-demand policy (no invented growth)
    unmet[p]    = D[p] - delivered[p]

No root plasticity or compensation: a missed request on a dry / masked / unavailable resource (A = 0) stays
unmet and is never moved to another cell or layer. Water that nobody requested stays with SYRUP (deeper
residual water is unused). SYRUP owns the sole water inventory and the accepted debit: nothing here holds,
decrements or caches a water store; `propose` and `accept` are pure functions of their inputs.

Root map (`RootMap`). Validated, immutable, plant-major CSR: stable int64 plant IDs [P], int64 offsets [P+1],
int64 flattened resource indices [E] (layer * n_cells + cell), FP64 nonnegative weights [E]. Weights are fully
general (NON-separable: each individual may put a different fraction in every layer/cell pair). Rows with edges
must sum to 1 within WEIGHT_SUM_TOL; rows without edges are inactive roots (allowed) on which positive demand is
REFUSED. Duplicate IDs, bool / float / out-of-range (unsigned overflow) IDs and indices, index/shape bounds and
non-finite / negative weights are refused before any lossy cast. The map owns copies of the accepted arrays
(host copies are frozen read-only), so callers cannot mutate accepted root geometry afterwards; failed
construction leaves the caller's arrays untouched.

Backends. `numpy` (vectorised reference, bincount scatter/gather), `numba` (compiled CPU kernels) and `cuda`
(CuPy RawKernel FP64 atomics) implement ONE algorithm: the per-edge arithmetic is written once in the plain
Python kernels `_k_scatter` / `_k_proposal` / `_k_scale` / `_k_gather` below (the authoritative definition,
compiled by numba, mirrored line by line in CUDA C and vectorised by the NumPy reference). No dense P x L x N
tensor, no P x P matrix, no per-edge daily history: arrays are bounded by P, E and R = L * N. Production CPU/GPU
paths contain no Python per-edge loops (the plain Python kernels are executed only through numba, or by tests
on tiny problems). Plans / feedback live on the backend of their map: a `cuda` map takes and returns CuPy
arrays, host backends take and return NumPy arrays; cross-namespace arrays are refused (no implicit host/device
population transfers; the GPU path reads back only scalar reductions). No implicit GPU fallback or device switch.

Memory admission. `max_edges` and `max_bytes` are explicit for map construction; operations reuse the map's
budget unless a tighter one is passed. Every allocating operation estimates its OWNED arrays (map, plan,
feedback), its VALIDATION / REFERENCE SCRATCH and any TRANSFER STAGING before anything is allocated and refuses
with `AdmissionError` (carrying the estimate) beforehand. Estimates never include the caller's own input arrays
(the caller already holds them) and no host/device mirrors are kept. Bounds are never raised and individuals
are never discarded to pass.

Native authority boundary. `propose` produces a proposal; `accept` turns SYRUP's single accepted answer into
delivered / unmet / S. Publishing receipts or state (biomass, soil water) is the job of the later driver and the
native water authority, not of this stateless allocator. Importing this module imports NumPy only; numba / cupy
are imported lazily when a backend is first requested.
'''
from __future__ import annotations

import math

import numpy as np

from leaf import traits as _traits

VERSION = '0.1.0'
BACKENDS = ('numpy', 'numba', 'cuda')
HOST_BACKENDS = ('numpy', 'numba')
FP64_EPS = float(np.finfo(np.float64).eps)
WEIGHT_SUM_TOL = _traits.LAYER_FRACTION_TOL       # 1e-9: rows with edges must sum to 1 within this (not renormalised)
INT64_MAX = 2 ** 63 - 1
THREADS = 256
MAX_BLOCKS = 65535
ID_PREVIEW = 10
UNITS = {'demand': 'm3 per timestep', 'supply': 'm3 available per (layer, cell) for the timestep',
         'request': 'm3', 'proposed': 'm3', 'accepted': 'm3', 'delivered': 'm3', 'unmet': 'm3',
         'satisfaction': 'fraction [0, 1]', 'fraction': 'fraction [0, 1]', 'weights': 'dimensionless (row sum 1)'}
ALGORITHM = ('plant-major CSR: scatter Q[r] += D[p]*w[e]; proposed[r] = min(A[r], Q[r]); '
             'scale[r] = accepted[r]/Q[r] (0 if Q == 0); gather delivered[p] += D[p]*w[e]*scale[r]')
ZERO_DEMAND_POLICY = 'S[p] = 0 when D[p] == 0 (explicit: zero demand yields zero satisfaction, no invented growth)'
NO_PLASTICITY_NOTE = ('no root plasticity / compensation: a missed request on a dry or unavailable resource (A = 0) '
                      'stays unmet; weights are never renormalised or redistributed to wetter cells or layers')


class UptakeError(ValueError):
    '''Any root-map / plan / feedback validation error or allocator implementation failure.'''


class AdmissionError(UptakeError):
    '''A declared edge/byte bound would be exceeded; raised BEFORE any large allocation or transfer.'''

    def __init__(self, stage, reason, estimate):
        super().__init__(f'{stage}: {reason}')
        self.stage = stage
        self.reason = reason
        self.estimate = estimate


class BackendUnavailableError(UptakeError):
    '''The requested backend cannot be used on this machine (missing numba / cupy / CUDA device).'''


# ------------------------------------------------------------------ authoritative per-edge arithmetic (plain Python)
def _k_scatter(offsets, resources, weights, demand, request):
    '''request[r] = sum over edges of demand[p] * weights[e]. Authoritative definition of the scatter.'''
    for r in range(request.shape[0]):
        request[r] = 0.0
    for p in range(demand.shape[0]):
        d = demand[p]
        if d == 0.0:
            continue
        for e in range(offsets[p], offsets[p + 1]):
            request[resources[e]] += d * weights[e]


def _k_proposal(supply, request, proposed, fraction):
    '''proposed[r] = min(A, Q); fraction[r] = proposed / Q (both 0 when Q == 0).'''
    for r in range(request.shape[0]):
        q = request[r]
        if q > 0.0:
            a = supply[r]
            pr = a if a < q else q
            proposed[r] = pr
            fraction[r] = pr / q
        else:
            proposed[r] = 0.0
            fraction[r] = 0.0


def _k_scale(accepted, request, scale):
    '''scale[r] = accepted[r] / Q[r] (0 when Q == 0); accepted is already clipped to the proposal.'''
    for r in range(request.shape[0]):
        q = request[r]
        if q > 0.0:
            scale[r] = accepted[r] / q
        else:
            scale[r] = 0.0


def _k_gather(offsets, resources, weights, demand, scale, delivered):
    '''delivered[p] = sum over edges of demand[p] * weights[e] * scale[resources[e]].'''
    for p in range(demand.shape[0]):
        d = demand[p]
        u = 0.0
        if d != 0.0:
            for e in range(offsets[p], offsets[p + 1]):
                u += d * weights[e] * scale[resources[e]]
        delivered[p] = u


_CUDA_SRC = r'''
extern "C" {

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
                         const long long n) {
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x; r < n; r += stride) {
        double q = request[r];
        if (q > 0.0) {
            double a = supply[r];
            double pr = a < q ? a : q;
            proposed[r] = pr;
            fraction[r] = pr / q;
        } else {
            proposed[r] = 0.0;
            fraction[r] = 0.0;
        }
    }
}

__global__ void scale_kernel(const double* accepted, const double* request, double* scale, const long long n) {
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x; r < n; r += stride) {
        double q = request[r];
        scale[r] = (q > 0.0) ? accepted[r] / q : 0.0;
    }
}

__global__ void gather(const long long* offsets, const long long* resources, const double* weights,
                       const double* demand, const double* scale, double* delivered, const long long n_plants) {
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x; p < n_plants; p += stride) {
        double d = demand[p];
        double u = 0.0;
        if (d != 0.0) {
            for (long long e = offsets[p]; e < offsets[p + 1]; ++e) u += d * weights[e] * scale[resources[e]];
        }
        delivered[p] = u;
    }
}

}
'''
_CUDA_KERNELS = ('scatter', 'proposal', 'scale_kernel', 'gather')


# ------------------------------------------------------------------ backends (lazy)
_BACKEND = {}
_COMPILED = {}


def backend_status(backend):
    '''Probe a backend once per process: {'backend', 'available', 'reason', ...}. Never raises for an
    unavailable backend (tests may skip on it); an unknown backend name raises UptakeError.'''
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
                cc = int(cp.cuda.Device().compute_capability)
                if cc < 60:
                    raise RuntimeError(f'compute capability {cc} < 60 required for FP64 atomicAdd')
                st = {'backend': 'cuda', 'available': True, 'reason': None, 'cupy_version': cp.__version__,
                      'compute_capability': cc, 'device_id': int(cp.cuda.Device().id)}
            except Exception as exc:  # pragma: no cover - machine dependent
                st = {'backend': 'cuda', 'available': False, 'reason': f'cupy/CUDA unavailable: {exc!r}'}
        _BACKEND[backend] = st
    return dict(st)


def _require_backend(backend):
    st = backend_status(backend)
    if not st['available']:
        raise BackendUnavailableError(f'backend {backend!r} is not available: {st["reason"]} (no implicit fallback)')
    return st


def _xp(backend):
    if backend == 'cuda':
        import cupy
        return cupy
    return np


def _numba_kernels():
    if 'numba' not in _COMPILED:
        from numba import njit
        _COMPILED['numba'] = {'scatter': njit(cache=True)(_k_scatter), 'proposal': njit(cache=True)(_k_proposal),
                              'scale': njit(cache=True)(_k_scale), 'gather': njit(cache=True)(_k_gather)}
    return _COMPILED['numba']


def _cuda_kernels():
    if 'cuda' not in _COMPILED:
        import cupy as cp
        mod = cp.RawModule(code=_CUDA_SRC, options=('-fmad=false',))
        mod.compile()
        _COMPILED['cuda'] = {name: mod.get_function(name) for name in _CUDA_KERNELS}
        _COMPILED['cuda']['module'] = mod
    return _COMPILED['cuda']


def _blocks(n):
    return max(1, min((int(n) + THREADS - 1) // THREADS, MAX_BLOCKS))


def _is_device(a):
    return hasattr(a, '__cuda_array_interface__')


# ------------------------------------------------------------------ memory estimates (pure)
def _int_bound(v, name, lo=0):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
        raise UptakeError(f'{name} must be an integer, got {v!r}')
    v = int(v)
    if v < lo:
        raise UptakeError(f'{name} must be >= {lo}, got {v}')
    return v


def estimate_map_bytes(n_plants, n_edges, n_resources, backend='numpy', *, host_source=True):
    '''Bytes needed to construct a RootMap: owned arrays + validation scratch + transfer staging. Scope:
    owned   = ids[P] + offsets[P+1] + counts[P] (int64) + resources[E] (int64) + weights[E] (float64)
    scratch = plant-of-edge[E] + cast copies (int64 resources[E], float64 weights[E]) + row sums[P] +
              id uniqueness sort[P] + resource edge counts[R] (int64), freed after validation
    staging = owned (device copy coexisting with host-validated arrays) when a cuda map is loaded from host
    Caller input arrays are NOT counted (the caller already holds them); no host/device mirrors are kept.'''
    P = _int_bound(n_plants, 'n_plants')
    E = _int_bound(n_edges, 'n_edges')
    R = _int_bound(n_resources, 'n_resources')
    owned = 8 * P + 8 * (P + 1) + 8 * P + 8 * E + 8 * E
    scratch = 8 * E + 8 * E + 8 * E + 8 * P + 8 * P + 8 * R
    staging = owned if (backend == 'cuda' and host_source) else 0
    return {'stage': 'map', 'n_plants': P, 'n_edges': E, 'n_resources': R, 'backend': backend,
            'owned_bytes': owned, 'scratch_bytes': scratch, 'staging_bytes': staging, 'mirror_bytes': 0,
            'total_bytes': owned + scratch + staging,
            'resident_on': 'device' if backend == 'cuda' else 'host',
            'scratch_on': 'host' if (backend != 'cuda' or host_source) else 'device',
            'scope': 'owned map arrays + validation scratch (freed) + transfer staging; caller inputs excluded; '
                     'no mirrors'}


def estimate_plan_bytes(n_plants, n_edges, n_resources, backend='numpy'):
    '''Bytes for `propose`: owned = demand[P] + request[R] + proposed[R] + fraction[R] (float64);
    scratch = supply float64 copy[R] + (numpy reference only) per-edge demand[E] and term[E].'''
    P = _int_bound(n_plants, 'n_plants')
    E = _int_bound(n_edges, 'n_edges')
    R = _int_bound(n_resources, 'n_resources')
    owned = 8 * P + 3 * 8 * R
    scratch = 8 * R + (16 * E if backend == 'numpy' else 0)
    return {'stage': 'plan', 'n_plants': P, 'n_edges': E, 'n_resources': R, 'backend': backend,
            'owned_bytes': owned, 'scratch_bytes': scratch, 'staging_bytes': 0, 'mirror_bytes': 0,
            'total_bytes': owned + scratch, 'resident_on': 'device' if backend == 'cuda' else 'host',
            'scope': 'owned plan arrays + supply cast copy + numpy-reference per-edge scratch; the resident map '
                     'is counted separately by the admission total'}


def estimate_feedback_bytes(n_plants, n_edges, n_resources, backend='numpy'):
    '''Bytes for `accept`: owned = accepted (clipped copy)[R] + scale[R] + delivered/unmet/satisfaction[3P];
    scratch = accepted float64 cast[R] + comparison mask[R] + (numpy reference only) 4 per-edge arrays[E].'''
    P = _int_bound(n_plants, 'n_plants')
    E = _int_bound(n_edges, 'n_edges')
    R = _int_bound(n_resources, 'n_resources')
    owned = 2 * 8 * R + 3 * 8 * P
    scratch = 9 * R + (32 * E if backend == 'numpy' else 0)
    return {'stage': 'feedback', 'n_plants': P, 'n_edges': E, 'n_resources': R, 'backend': backend,
            'owned_bytes': owned, 'scratch_bytes': scratch, 'staging_bytes': 0, 'mirror_bytes': 0,
            'total_bytes': owned + scratch, 'resident_on': 'device' if backend == 'cuda' else 'host',
            'scope': 'owned feedback arrays + cast/mask scratch + numpy-reference per-edge scratch; resident '
                     'map and plan are counted separately by the admission total'}


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


def closure_tolerance_m3(total_m3, n_plants, n_edges, n_resources):
    '''FP64 summation tolerance for sum(delivered) == sum(accepted): eps * (2E + R + P + 8) * |total|. The
    factor counts every rounding in the chain (per-edge scatter adds and gather products, per-resource ratio /
    proposal, and the two reductions); it is proportional to machine epsilon, never an enlarged tolerance.'''
    return FP64_EPS * (2 * int(n_edges) + int(n_resources) + int(n_plants) + 8) * abs(float(total_m3))


# ------------------------------------------------------------------ array admission helpers
def _reject_scalar_types(v, name):
    if v is None or isinstance(v, (bool, np.bool_, str, bytes, complex)):
        raise UptakeError(f'{name} must be a numeric array, got {type(v).__name__}')


def _array_for(v, name, backend, *, allow_host_load=False):
    '''Return an array in the backend namespace WITHOUT copying. Host backends refuse device arrays; the cuda
    backend refuses host arrays unless `allow_host_load` (explicit map load) is set, in which case the host
    array is returned for host-side validation before one explicit transfer.'''
    _reject_scalar_types(v, name)
    if backend == 'cuda':
        if _is_device(v):
            import cupy
            if not isinstance(v, cupy.ndarray):
                raise UptakeError(f'{name}: cuda backend requires a cupy.ndarray')
            return v, True
        if not allow_host_load:
            raise UptakeError(f'{name}: cuda backend received a host array; no implicit host->device transfer '
                              '(pass a CuPy array; RootMap.from_arrays(backend="cuda") is the explicit map load)')
        return np.asarray(v), False
    if _is_device(v):
        raise UptakeError(f'{name}: host backend {backend!r} received a device array; no implicit device->host transfer')
    return np.asarray(v), False


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


def _int64_copy(a, name, xp, *, take=False):
    '''int64 copy (or owned cast) after the unsigned-overflow check; metadata already validated.'''
    if a.dtype.kind == 'u' and a.size and int(a.max()) > INT64_MAX:
        raise UptakeError(f'{name} holds values above the int64 range (unsigned overflow refused before cast)')
    if take and a.dtype == np.int64 and getattr(a.flags, 'c_contiguous', True):
        return a
    return xp.array(a, dtype=xp.int64, copy=True)


def _f64_copy(a, xp, *, take=False):
    if take and a.dtype == np.float64 and getattr(a.flags, 'c_contiguous', True):
        return a
    return xp.array(a, dtype=xp.float64, copy=True)


def _freeze(a):
    if isinstance(a, np.ndarray):
        a.setflags(write=False)
    return a


def _total(a, xp):
    return float(xp.sum(a)) if a.size else 0.0


def _preview(ids, mask, xp):
    idx = xp.flatnonzero(mask)[:ID_PREVIEW]
    return [int(v) for v in np.asarray(ids[idx].get() if _is_device(ids) else ids[idx])]


# ------------------------------------------------------------------ root map
class RootMap:
    '''Immutable validated plant-major CSR root map (see the module docstring). Construct with `from_arrays`.'''

    _FIELDS = ('backend', 'xp', 'ids', 'offsets', 'counts', 'resources', 'weights', 'n_plants', 'n_edges', 'n_layers',
               'n_cells', 'n_resources', 'max_edges', 'max_bytes', 'estimate', 'admission', 'max_edges_per_plant',
               'max_edges_per_resource', 'n_inactive_plants', 'provenance', '_frozen')

    def __init__(self, **kw):
        for k in self._FIELDS:
            if k == '_frozen':
                continue
            object.__setattr__(self, k, kw[k])
        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name, value):
        raise AttributeError('RootMap is immutable (build a new map; growth-driven rebuilds are a later driver operation)')

    def __delattr__(self, name):
        raise AttributeError('RootMap is immutable')

    @classmethod
    def from_arrays(cls, ids, offsets, resources, weights, *, n_layers, n_cells, backend='numpy', max_edges,
                    max_bytes, provenance=None, _take_ownership=False):
        '''Validate (metadata -> admission -> content) and build an owned, frozen map.

        ids [P] int64 stable plant IDs (unique); offsets [P+1] int64 nondecreasing, offsets[0] == 0,
        offsets[-1] == E; resources [E] int64 in [0, n_layers * n_cells); weights [E] FP64 finite >= 0, each
        non-empty row summing to 1 within WEIGHT_SUM_TOL. Host arrays are accepted for every backend (for
        `cuda` this is the one explicit host->device load, after host-side validation); CuPy arrays are accepted
        for `cuda` (validated on device with scalar reductions only). Bool / float / unsigned-overflow integer
        inputs are refused before any cast. Sequences are converted with np.asarray first.'''
        if backend not in BACKENDS:
            raise UptakeError(f'backend must be one of {BACKENDS}, got {backend!r}')
        _require_backend(backend)
        xp = _xp(backend)
        L = _traits.strict_int(n_layers, 'n_layers', 1)
        N = _traits.strict_int(n_cells, 'n_cells', 1)
        if L > INT64_MAX // N:
            raise UptakeError(f'n_layers * n_cells = {L} * {N} overflows the int64 resource index')
        R = L * N
        ids_a, dev = _array_for(ids, 'ids', backend, allow_host_load=True)
        off_a, d2 = _array_for(offsets, 'offsets', backend, allow_host_load=True)
        res_a, d3 = _array_for(resources, 'resources', backend, allow_host_load=True)
        w_a, d4 = _array_for(weights, 'weights', backend, allow_host_load=True)
        if len({dev, d2, d3, d4}) != 1:
            raise UptakeError('ids/offsets/resources/weights must all be host arrays or all be device arrays')
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
        est = estimate_map_bytes(P, E, R, backend, host_source=not dev)
        adm = admit('uptake.RootMap.from_arrays', est, max_edges=max_edges, max_bytes=max_bytes)
        vxp = xp if dev else np  # validation namespace: where the inputs live
        # ---- content validation (scalar reductions only; nothing of size P/E/R leaves the device)
        ids64 = _int64_copy(ids_a, 'ids', vxp, take=_take_ownership)
        off64 = _int64_copy(off_a, 'offsets', vxp, take=_take_ownership)
        res64 = _int64_copy(res_a, 'resources', vxp, take=_take_ownership)
        w64 = _f64_copy(w_a, vxp, take=_take_ownership)
        if int(off64[0]) != 0 or int(off64[-1]) != E:
            raise UptakeError(f'offsets must start at 0 and end at the edge count {E} (got {int(off64[0])}, {int(off64[-1])})')
        counts = vxp.diff(off64)
        if P and int(counts.min()) < 0:
            raise UptakeError('offsets must be nondecreasing')
        if P and int(vxp.unique(ids64).shape[0]) != P:
            raise UptakeError('duplicate plant IDs in the root map are refused (one row per individual)')
        if E:
            if int(res64.min()) < 0 or int(res64.max()) >= R:
                raise UptakeError(f'resource indices must lie in [0, n_layers * n_cells = {R})')
            if not bool(vxp.isfinite(w64).all()):
                raise UptakeError('weights must be finite')
            if float(w64.min()) < 0.0:
                raise UptakeError('weights must be >= 0')
            plant_of_edge = vxp.repeat(vxp.arange(P, dtype=vxp.int64), counts)
            row = vxp.bincount(plant_of_edge, weights=w64, minlength=P)
            bad = (counts > 0) & (vxp.abs(row - 1.0) > WEIGHT_SUM_TOL)
            if bool(bad.any()):
                raise UptakeError(f'root rows must sum to 1 within {WEIGHT_SUM_TOL} (not renormalised); offending plant ids '
                                  f'{_preview(ids64, bad, vxp)}')
            max_res = int(vxp.bincount(res64, minlength=R).max())
            del plant_of_edge, row
        else:
            max_res = 0
        max_plant = int(counts.max()) if P else 0
        n_inactive = int(vxp.count_nonzero(counts == 0)) if P else 0
        # ---- ownership: cast copies become the owned arrays; cuda-from-host performs the one explicit transfer
        if backend == 'cuda' and not dev:
            ids64, off64, res64, w64, counts = (xp.asarray(a) for a in (ids64, off64, res64, w64, counts))
        else:
            counts = vxp.array(counts, dtype=vxp.int64, copy=True)
        for a in (ids64, off64, res64, w64, counts):
            _freeze(a)
        prov = dict(provenance) if provenance else {}
        return cls(backend=backend, xp=xp, ids=ids64, offsets=off64, counts=counts, resources=res64, weights=w64,
                   n_plants=P, n_edges=E, n_layers=L, n_cells=N, n_resources=R, max_edges=int(max_edges),
                   max_bytes=int(max_bytes), estimate=est, admission=adm, max_edges_per_plant=max_plant,
                   max_edges_per_resource=max_res, n_inactive_plants=n_inactive, provenance=prov)

    # ---- properties
    @property
    def resident_bytes(self):
        '''Bytes of the owned map arrays currently resident on the map backend.'''
        return int(self.estimate['owned_bytes'])

    def __repr__(self):
        return (f'RootMap(P={self.n_plants}, E={self.n_edges}, L={self.n_layers}, N={self.n_cells}, '
                f'backend={self.backend!r})')

    def describe(self):
        return {'n_plants': self.n_plants, 'n_edges': self.n_edges, 'n_layers': self.n_layers, 'n_cells': self.n_cells,
                'n_resources': self.n_resources, 'backend': self.backend, 'n_inactive_plants': self.n_inactive_plants,
                'max_edges_per_plant': self.max_edges_per_plant, 'max_edges_per_resource': self.max_edges_per_resource,
                'resident_bytes': self.resident_bytes, 'max_edges': self.max_edges, 'max_bytes': self.max_bytes,
                'weight_sum_tol': WEIGHT_SUM_TOL, 'resource_index': 'layer * n_cells + cell (row-major cell index)',
                'units': dict(UNITS), 'algorithm': ALGORITHM, 'provenance': dict(self.provenance),
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
        '''Rebuild this map on another backend (explicit transfer; same validation and admission).'''
        a = self.host_arrays()
        return RootMap.from_arrays(a['ids'], a['offsets'], a['resources'], a['weights'], n_layers=self.n_layers,
                                   n_cells=self.n_cells, backend=backend, max_edges=self.max_edges,
                                   max_bytes=self.max_bytes if max_bytes is None else max_bytes,
                                   provenance=dict(self.provenance, converted_from=self.backend))


# ------------------------------------------------------------------ backend operations (one algorithm, three paths)
def _scatter(m, demand):
    xp = m.xp
    if m.backend == 'numpy':
        if m.n_edges == 0:
            return np.zeros(m.n_resources, dtype=np.float64)
        term = np.repeat(demand, m.counts) * m.weights
        return np.bincount(m.resources, weights=term, minlength=m.n_resources)
    request = xp.zeros(m.n_resources, dtype=xp.float64)
    if m.backend == 'numba':
        _numba_kernels()['scatter'](m.offsets, m.resources, m.weights, demand, request)
    else:
        _cuda_kernels()['scatter']((_blocks(m.n_plants),), (THREADS,),
                                   (m.offsets, m.resources, m.weights, demand, request, np.int64(m.n_plants)))
    return request


def _proposal(m, supply, request):
    xp = m.xp
    proposed = xp.empty(m.n_resources, dtype=xp.float64)
    fraction = xp.empty(m.n_resources, dtype=xp.float64)
    if m.backend == 'numpy':
        pos = request > 0.0
        proposed[...] = np.where(pos, np.minimum(supply, request), 0.0)
        fraction[...] = 0.0
        fraction[pos] = proposed[pos] / request[pos]
    elif m.backend == 'numba':
        _numba_kernels()['proposal'](supply, request, proposed, fraction)
    else:
        _cuda_kernels()['proposal']((_blocks(m.n_resources),), (THREADS,),
                                    (supply, request, proposed, fraction, np.int64(m.n_resources)))
    return proposed, fraction


def _scale(m, accepted, request):
    xp = m.xp
    scale = xp.empty(m.n_resources, dtype=xp.float64)
    if m.backend == 'numpy':
        pos = request > 0.0
        scale[...] = 0.0
        scale[pos] = accepted[pos] / request[pos]
    elif m.backend == 'numba':
        _numba_kernels()['scale'](accepted, request, scale)
    else:
        _cuda_kernels()['scale_kernel']((_blocks(m.n_resources),), (THREADS,),
                                        (accepted, request, scale, np.int64(m.n_resources)))
    return scale


def _gather(m, demand, scale):
    xp = m.xp
    if m.backend == 'numpy':
        if m.n_plants == 0:
            return np.zeros(0, dtype=np.float64)
        if m.n_edges == 0:
            return np.zeros(m.n_plants, dtype=np.float64)
        term = np.repeat(demand, m.counts) * m.weights * scale[m.resources]
        plant_of_edge = np.repeat(np.arange(m.n_plants, dtype=np.int64), m.counts)
        return np.bincount(plant_of_edge, weights=term, minlength=m.n_plants)
    delivered = xp.empty(m.n_plants, dtype=xp.float64)
    if m.backend == 'numba':
        _numba_kernels()['gather'](m.offsets, m.resources, m.weights, demand, scale, delivered)
    else:
        _cuda_kernels()['gather']((_blocks(m.n_plants),), (THREADS,),
                                  (m.offsets, m.resources, m.weights, demand, scale, delivered, np.int64(m.n_plants)))
    return delivered


def _resource_array(v, name, m):
    '''Validated float64 copy [R] of a (L, N) or (R,) supply/accepted array on the map backend.'''
    a, _ = _array_for(v, name, m.backend)
    _check_real_meta(a, name)
    shape = tuple(int(s) for s in a.shape)
    if shape not in ((m.n_layers, m.n_cells), (m.n_resources,)):
        raise UptakeError(f'{name} must have shape (n_layers, n_cells) = {(m.n_layers, m.n_cells)} or ({m.n_resources},), got {shape}')
    return a


def _finite_nonneg(a, name, xp):
    if a.size:
        if not bool(xp.isfinite(a).all()):
            raise UptakeError(f'{name} must be finite')
        if float(a.min()) < 0.0:
            raise UptakeError(f'{name} must be >= 0')


# ------------------------------------------------------------------ plan (proposal)
class UptakePlan:
    '''Pure proposal bound to one RootMap and one owned demand vector. Arrays live on the map backend:
    demand [P], request Q [R], proposed [R], fraction f [R] (R = n_layers * n_cells, flattened layer-major, i.e.
    reshape(n_layers, n_cells) gives A[l, c] layout). `summary` holds scalar totals only.'''

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
    proposed = min(A, Q) and f. Pure: no state, no water debit. Positive demand on an inactive (edge-less) root
    row is refused; zero demand anywhere is allowed (S will be 0 there). Arrays must live on the map backend.'''
    if not isinstance(root_map, RootMap):
        raise UptakeError('root_map must be a leaf.uptake.RootMap')
    m, xp = root_map, root_map.xp
    d_in, _ = _array_for(demand, 'demand', m.backend)
    _check_real_meta(d_in, 'demand', (m.n_plants,))
    a_in = _resource_array(supply, 'supply', m)
    est = estimate_plan_bytes(m.n_plants, m.n_edges, m.n_resources, m.backend)
    adm = admit('uptake.propose', est, max_edges=m.max_edges,
                max_bytes=m.max_bytes if max_bytes is None else max_bytes, resident_bytes=m.resident_bytes)
    D = xp.array(d_in, dtype=xp.float64, copy=True)
    A = xp.array(a_in, dtype=xp.float64, copy=True).reshape(-1)
    _finite_nonneg(D, 'demand', xp)
    _finite_nonneg(A, 'supply', xp)
    if m.n_plants:
        bad = (m.counts == 0) & (D > 0.0)
        if bool(bad.any()):
            raise UptakeError(f'positive demand on root rows without edges (inactive roots): plant ids '
                              f'{_preview(m.ids, bad, xp)}; map the roots or set their demand to 0')
    request = _scatter(m, D)
    proposed, fraction = _proposal(m, A, request)
    summary = {'n_plants': m.n_plants, 'n_edges': m.n_edges, 'n_resources': m.n_resources, 'backend': m.backend,
               'demand_total_m3': _total(D, xp), 'request_total_m3': _total(request, xp),
               'supply_total_m3': _total(A, xp), 'proposed_total_m3': _total(proposed, xp),
               'n_plants_demanding': int(xp.count_nonzero(D > 0.0)) if m.n_plants else 0,
               'n_resources_requested': int(xp.count_nonzero(request > 0.0)),
               'n_resources_limited': int(xp.count_nonzero(request > A)),
               'admission': adm, 'units': dict(UNITS), 'zero_demand_policy': ZERO_DEMAND_POLICY,
               'no_plasticity': NO_PLASTICITY_NOTE}
    return UptakePlan(m, D, request, proposed, fraction, summary)


# ------------------------------------------------------------------ feedback (accepted water)
class UptakeFeedback:
    '''Result of SYRUP's single acceptance: accepted (clipped copy) [R], scale accepted/Q [R], delivered [P],
    unmet [P], satisfaction S [P]; `closure` holds the FP64 water-closure check and `summary` scalar totals.'''

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
    '''Relative FP64 tolerance by which accepted[r] may exceed proposed[r] before being refused:
    eps * (max edges on one resource + 2) (the roundings of the Q sum and the min / ratio).'''
    return FP64_EPS * (int(root_map.max_edges_per_resource) + 2)


def accept(plan, accepted, *, max_bytes=None):
    '''Turn SYRUP's accepted withdrawals [(L, N) or (R,)] m3 into delivered / unmet / S. Validates: same
    backend, shape, finite, >= 0, accepted <= proposed within acceptance_tolerance_rel (material excess is
    refused, rounding-level excess is clipped and COUNTED, never silent), zero on resources nobody requested.
    Checks finite water closure sum(delivered) == sum(accepted) within closure_tolerance_m3 (an implementation
    failure otherwise). Pure: the plan and map are untouched; nothing is published.'''
    if not isinstance(plan, UptakePlan):
        raise UptakeError('plan must be a leaf.uptake.UptakePlan produced by propose()')
    m, xp = plan.root_map, plan.root_map.xp
    a_in = _resource_array(accepted, 'accepted', m)
    est = estimate_feedback_bytes(m.n_plants, m.n_edges, m.n_resources, m.backend)
    resident = m.resident_bytes + int(plan.summary['admission']['owned_bytes'])
    adm = admit('uptake.accept', est, max_edges=m.max_edges,
                max_bytes=m.max_bytes if max_bytes is None else max_bytes, resident_bytes=resident)
    acc = xp.array(a_in, dtype=xp.float64, copy=True).reshape(-1)
    _finite_nonneg(acc, 'accepted', xp)
    tol_rel = acceptance_tolerance_rel(m)
    excess = acc - plan.proposed
    over = excess > tol_rel * plan.proposed
    if bool(over.any()):
        worst = float(excess.max())
        raise UptakeError(f'accepted exceeds the proposal on {int(xp.count_nonzero(over))} resource(s) beyond the FP64 '
                          f'tolerance {tol_rel:.3e} (max excess {worst!r} m3); material clipping is refused')
    n_clip = int(xp.count_nonzero(excess > 0.0))
    max_clip = float(excess.max()) if (n_clip and acc.size) else 0.0
    acc_eff = xp.minimum(acc, plan.proposed)
    del excess, over
    scale = _scale(m, acc_eff, plan.request)
    D = plan.demand
    delivered = _gather(m, D, scale)
    if m.n_plants:
        limit = D * (1.0 + WEIGHT_SUM_TOL + FP64_EPS * (m.max_edges_per_plant + 4))
        if bool((delivered > limit).any()):
            raise UptakeError('implementation failure: delivered water exceeds demand beyond the row-sum / FP64 tolerance')
        unmet = xp.maximum(D - delivered, 0.0)
        pos = D > 0.0
        satisfaction = xp.where(pos, xp.minimum(delivered / xp.where(pos, D, 1.0), 1.0), 0.0)
    else:
        unmet = xp.zeros(0, dtype=xp.float64)
        satisfaction = xp.zeros(0, dtype=xp.float64)
    acc_total = _total(acc_eff, xp)
    del_total = _total(delivered, xp)
    residual = del_total - acc_total
    tol = closure_tolerance_m3(max(abs(acc_total), abs(del_total)), m.n_plants, m.n_edges, m.n_resources)
    closure = {'accepted_total_m3': acc_total, 'delivered_total_m3': del_total, 'residual_m3': residual,
               'tolerance_m3': tol, 'closed': abs(residual) <= tol,
               'definition': 'sum(delivered) == sum(accepted) within closure_tolerance_m3 (FP64 summation tolerance)'}
    if abs(residual) > tol:
        raise UptakeError(f'water closure violated: sum(delivered) - sum(accepted) = {residual!r} m3 exceeds the FP64 '
                          f'tolerance {tol!r} m3 (implementation failure, not a scientific result)')
    summary = {'n_plants': m.n_plants, 'backend': m.backend, 'accepted_total_m3': acc_total,
               'delivered_total_m3': del_total, 'unmet_total_m3': _total(unmet, xp),
               'demand_total_m3': plan.summary['demand_total_m3'], 'proposed_total_m3': plan.summary['proposed_total_m3'],
               'shortfall_total_m3': plan.summary['proposed_total_m3'] - acc_total,
               'n_rounding_clipped': n_clip, 'max_rounding_excess_m3': max_clip, 'acceptance_tolerance_rel': tol_rel,
               'n_plants_unsatisfied': int(xp.count_nonzero((D > 0.0) & (satisfaction < 1.0))) if m.n_plants else 0,
               'n_plants_zero_demand': int(xp.count_nonzero(D == 0.0)) if m.n_plants else 0,
               'admission': adm, 'units': dict(UNITS), 'zero_demand_policy': ZERO_DEMAND_POLICY,
               'authority': 'stateless allocator result; state/receipt publication belongs to the later driver and '
                            'the native water authority'}
    return UptakeFeedback(plan, acc_eff, scale, delivered, unmet, satisfaction, closure, summary)


__all__ = ['VERSION', 'BACKENDS', 'HOST_BACKENDS', 'FP64_EPS', 'WEIGHT_SUM_TOL', 'UNITS', 'ALGORITHM',
           'ZERO_DEMAND_POLICY', 'NO_PLASTICITY_NOTE', 'UptakeError', 'AdmissionError', 'BackendUnavailableError',
           'backend_status', 'estimate_map_bytes', 'estimate_plan_bytes', 'estimate_feedback_bytes', 'admit',
           'closure_tolerance_m3', 'acceptance_tolerance_rel', 'RootMap', 'UptakePlan', 'UptakeFeedback', 'propose',
           'accept']
