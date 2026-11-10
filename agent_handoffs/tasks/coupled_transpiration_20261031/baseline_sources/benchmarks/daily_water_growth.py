'''Bounded generic long-horizon water/growth benchmark (daily soil water + weekly growth).

Every ecological rule here is a BENCHMARK FIXTURE: synthetic, uncalibrated, chosen only to give a
bounded, auditable, naturally growing workload on top of benchmarks.spatial_mapping (sm, unmodified).
Not calibrated ecology; no production-model or measured-speedup claims.
'''
import argparse
import json
import math
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np
from numba import njit

_T_IMPORT = time.perf_counter()
try:
    from benchmarks import spatial_mapping as sm
except ImportError:  # plain script run from inside benchmarks/
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import spatial_mapping as sm

VERSION = '0.2.0'  # 0.2.0: configurable per-profile root layer fractions; Rh map removed (memory/time not comparable to 0.1.0)
PROFILE_TOL = 1e-12  # input normalization tolerance for root profile rows
WEEK = 7
PI = math.pi
FIXTURE = {
    'label': 'synthetic benchmark fixture, not calibrated ecology',
    'storage_m': 0.15, 'initial_fill_fraction': 0.5, 'recharge_m': 0.010, 'recharge_period_days': 14,
    'demand_m_per_day': 0.003, 'biomass_density_kg_m2': 1.0, 'water_density_kg_m3': 1000.0,
    'wue_grass_kg_per_kg': 0.002, 'wue_shrub_kg_per_kg': 0.0015, 'turnover_rate_per_day': 0.003,
    'week_days': WEEK, 'days_per_year': 365, 'default_years': 30,
}
AUDIT_TOL_REL = 1e-9
AUDIT_TOL_ABS = 1e-12


# ------------------------------------------------------------ CPU kernels (numba, cached)
@njit(cache=True)
def _demand_scatter_cpu(offsets, counts, cells, weights, Dp, pid, frac, R):
    '''R[l, j] += Dp[p] * w[p, j] * frac[pid[p], l], scattered directly into the resident [L, N] cache.'''
    L = R.shape[0]
    R[:, :] = 0.0
    for p in range(Dp.shape[0]):
        o = offsets[p]
        d = Dp[p]
        s = pid[p]
        for k in range(counts[p]):
            j = cells[o + k]
            dw = d * weights[o + k]
            for l in range(L):
                R[l, j] += dw * frac[s, l]


@njit(cache=True)
def _block_cpu(W, R, scale, removed, overflow, cap, rech, period, day0, ndays,
               offsets, counts, cells, weights, Dp, pid, frac, uptake_week, uptake_cum):
    L, N = W.shape
    for d in range(ndays):
        do_rech = ((day0 + d) % period) == 0
        for l in range(L):
            for j in range(N):
                w = W[l, j]
                if do_rech:
                    w += rech
                    if w > cap:
                        overflow[l, j] += w - cap
                        w = cap
                r = R[l, j]
                s = 0.0
                if r > 0.0:
                    acc = w if w < r else r
                    w -= acc
                    removed[l, j] += acc
                    s = acc / r
                scale[l, j] = s
                W[l, j] = w
        for p in range(Dp.shape[0]):
            o = offsets[p]
            d = Dp[p]
            s = pid[p]
            u = 0.0
            for k in range(counts[p]):
                j = cells[o + k]
                dw = d * weights[o + k]
                for l in range(L):  # exactly the requested piece times the same cell/layer acceptance scale
                    u += dw * frac[s, l] * scale[l, j]
            uptake_week[p] += u
            uptake_cum[p] += u


@njit(cache=True)
def _weight_sums_cpu(offsets, counts, weights, out):
    for p in range(out.shape[0]):
        o = offsets[p]
        s = 0.0
        for k in range(counts[p]):
            s += weights[o + k]
        out[p] = s


# ------------------------------------------------------------ GPU kernels (CuPy RawModule, FP64)
_CUDA_SRC = r'''
extern "C" {
__global__ void day_step(double* W, const double* R, double* scale, double* removed, double* overflow,
                         const double cap, const double rech, const int do_rech, const long long M) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    double w = W[i];
    if (do_rech) { w += rech; if (w > cap) { overflow[i] += w - cap; w = cap; } }
    double r = R[i];
    double s = 0.0;
    if (r > 0.0) { double acc = (w < r) ? w : r; w -= acc; removed[i] += acc; s = acc / r; }
    scale[i] = s;
    W[i] = w;
}
__global__ void gather(const long long* offsets, const long long* counts, const int* cells,
                       const double* weights, const double* Dp, const int* pid, const double* frac,
                       const double* scale, const int L, const long long N, double* uw, double* uc,
                       const long long P) {
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    long long o = offsets[p];
    long long c = counts[p];
    double d = Dp[p];
    const double* f = frac + (long long)pid[p] * L;
    double u = 0.0;
    for (long long k = 0; k < c; ++k) {
        long long j = cells[o + k];
        double dw = d * weights[o + k];
        for (int l = 0; l < L; ++l) u += dw * f[l] * scale[(long long)l * N + j];
    }
    uw[p] += u;
    uc[p] += u;
}
__global__ void demand_scatter(const long long* offsets, const long long* counts, const int* cells,
                               const double* weights, const double* Dp, const int* pid, const double* frac,
                               double* R, const int L, const long long N, const long long P) {
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    long long o = offsets[p];
    long long c = counts[p];
    double d = Dp[p];
    const double* f = frac + (long long)pid[p] * L;
    for (long long k = 0; k < c; ++k) {
        long long j = cells[o + k];
        double dw = d * weights[o + k];
        for (int l = 0; l < L; ++l) atomicAdd(&R[(long long)l * N + j], dw * f[l]);
    }
}
__global__ void weight_sums(const long long* offsets, const long long* counts, const double* weights,
                            double* out, const long long P) {
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    long long o = offsets[p];
    long long c = counts[p];
    double s = 0.0;
    for (long long k = 0; k < c; ++k) s += weights[o + k];
    out[p] = s;
}
}
'''
_GPU_CACHE = {}


def _gpu_kernels():
    if 'mod' not in _GPU_CACHE:
        mod = sm.cp.RawModule(code=_CUDA_SRC, options=('-std=c++11',))
        _GPU_CACHE['mod'] = mod
        for name in ('day_step', 'gather', 'demand_scatter', 'weight_sums'):
            _GPU_CACHE[name] = mod.get_function(name)
    return _GPU_CACHE


def gpu_available():
    '''True only with CuPy, an available device per sm.gpu_status and CC >= 6.0 (FP64 atomicAdd).'''
    cp = sm.cp
    if cp is None:
        return False
    try:
        st = sm.gpu_status()
        ok = st.get('available', st.get('availability', False)) if isinstance(st, dict) else bool(st)
        return bool(ok) and int(cp.cuda.Device().compute_capability) >= 60
    except Exception:
        return False


def environment_info(backend):
    info = {'python': platform.python_version(), 'platform': platform.platform(),
            'numpy': np.__version__, 'numba': None, 'cupy': None, 'device': None}
    try:
        import numba
        info['numba'] = numba.__version__
    except Exception:
        pass
    cp = sm.cp
    if cp is not None:
        info['cupy'] = getattr(cp, '__version__', None)
        if backend == 'gpu' and gpu_available():
            try:
                dev = cp.cuda.Device()
                name = cp.cuda.runtime.getDeviceProperties(dev.id).get('name')
                info['device'] = {'id': int(dev.id), 'name': name.decode() if isinstance(name, bytes) else str(name),
                                  'compute_capability': str(dev.compute_capability)}
            except Exception as exc:
                info['device'] = {'error': repr(exc)}
    try:
        info['gpu_status'] = sm.gpu_status()
    except Exception as exc:
        info['gpu_status'] = {'error': repr(exc)}
    return info


# Pool-peak profiling is delegated to an EXTERNAL CuPy MemoryHook passed as run_case(memory_hook=...)
# (entered once around the whole run) or installed by the caller around warm-up, initialization,
# loop and exports. No built-in hook: partial-lifetime counts are never reported as a true peak.


# ------------------------------------------------------------ validation / sizing
def _as_int(v, name, lo=None):
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
        raise ValueError(f'{name} must be an integer, got {v!r}')
    v = int(v)
    if lo is not None and v < lo:
        raise ValueError(f'{name} must be >= {lo}, got {v}')
    return v


def _as_float(v, name):
    if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name} must be a number, got {v!r}')
    v = float(v)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f'{name} must be finite and positive, got {v}')
    return v


_DEFAULT_ROOT_FRACTIONS = {1: {'grass': [1.0], 'shrub': [1.0]},
                           2: {'grass': [0.80, 0.20], 'shrub': [0.25, 0.75]},
                           3: {'grass': [0.70, 0.20, 0.10], 'shrub': [0.15, 0.25, 0.60]}}
_DEFAULT_PROFILE_NAMES = ('grass', 'shrub')


def _geometry_case(case):
    '''Project a validated case onto sm's original geometry fields (sm.generate_inputs validates strictly).'''
    fields = getattr(sm, 'CASE_FIELDS', None)
    if fields is None:
        return {k: v for k, v in case.items() if k != 'root_layer_fractions'}
    return {k: case[k] for k in fields if k in case}


def default_root_layer_fractions(L, max_bytes=None):
    '''Configurable UNCALIBRATED benchmark defaults (top to bottom). L=1/2/3 use the fixed tables above;
    other L use positive linear weights (grass descending, shrub ascending) normalized to one.
    Bytes of the 2*L FP64 default table are checked against max_bytes before construction.'''
    L = _as_int(L, 'layers', 1)
    if max_bytes is not None and 2 * L * 8 > _as_int(max_bytes, 'max_bytes', 1):
        raise _reject('daily_water_growth.init', 'default root profile table exceeds max_bytes',
                      {'profile_table_bytes': 16 * L, 'max_bytes': int(max_bytes)})
    if L in _DEFAULT_ROOT_FRACTIONS:
        return {k: list(v) for k, v in _DEFAULT_ROOT_FRACTIONS[L].items()}
    w = np.arange(1, L + 1, dtype=np.float64)
    w /= w.sum()
    return {'grass': w[::-1].tolist(), 'shrub': w.tolist()}


def _check_fraction_value(name, v):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
        raise ValueError(f'root profile {name!r} entries must be numeric (not bool), got {v!r}')
    v = float(v)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f'root profile {name!r} entries must be finite and > 0, got {v}')
    return v


def validate_root_layer_fractions(mapping, L):
    '''PURE check of an insertion-ordered name -> L finite positive numeric fractions (no bool/NaN/inf/<=0)
    mapping whose rows sum to one within PROFILE_TOL. Inspects caller-owned rows without copying them or
    allocating a table; returns the ordered profile names. Invalid rows are rejected, never normalized.'''
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError('root_layer_fractions must be a non-empty mapping of profile name -> list of layer fractions')
    names = []
    for name, row in mapping.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f'root profile names must be non-empty strings, got {name!r}')
        if isinstance(row, (str, bytes, dict)) or not hasattr(row, '__len__'):
            raise ValueError(f'root profile {name!r} must be a sequence of {L} fractions')
        if len(row) != L:
            raise ValueError(f'root profile {name!r} has {len(row)} entries, expected layers={L}')
        tot = math.fsum(_check_fraction_value(name, v) for v in row)
        if abs(tot - 1.0) > PROFILE_TOL:
            raise ValueError(f'root profile {name!r} must sum to 1 within {PROFILE_TOL} (got {tot!r}); '
                             'not normalized silently')
        names.append(name)
    return names


def resolve_root_profile_names(case):
    '''(names, source) WITHOUT materializing any table: default names need no allocation; custom rows are
    checked purely (see validate_root_layer_fractions).'''
    custom = case.get('root_layer_fractions')
    if custom is None:
        return list(_DEFAULT_PROFILE_NAMES), 'default'
    return validate_root_layer_fractions(custom, case['layers']), 'case'


def build_root_profile_table(case, names):
    '''Materialize the contiguous FP64 [S, L] table. Called only AFTER the aggregate admission preflight.'''
    L, mb = case['layers'], case['max_bytes']
    custom = case.get('root_layer_fractions')
    if 8 * len(names) * L > mb:
        raise _reject('daily_water_growth.init', 'root profile table exceeds max_bytes',
                      {'profile_table_bytes': 8 * len(names) * L, 'max_bytes': int(mb)})
    rows = default_root_layer_fractions(L, mb) if custom is None else custom
    table = np.empty((len(names), L), dtype=np.float64)
    for i, n in enumerate(names):
        table[i, :] = np.asarray(rows[n], dtype=np.float64)
    return table


def _require_profile_classes(names, need):
    missing = [k for k in need if k not in names]
    if missing:
        raise ValueError(f'root_layer_fractions must name the default class row(s) {missing} used by the '
                         'initial class assignment (or supply explicit inputs[root_profile_id])')


def check_profile_ids(v, P, S):
    '''PURE validation of explicit profile IDs: integer (not bool/float), 1-D, length P, 0 <= id < S.
    No conversion, copy or allocation; called BEFORE the aggregate admission preflight.'''
    if isinstance(v, np.ndarray):
        if v.dtype.kind not in 'iu':
            raise ValueError(f'root_profile_id must have an integer dtype (not bool/float), got {v.dtype}')
        if v.ndim != 1 or int(v.shape[0]) != P:
            raise ValueError(f'root_profile_id must have shape ({P},), got {tuple(v.shape)}')
        if int(v.min()) < 0 or int(v.max()) >= S:
            raise ValueError(f'root_profile_id values must lie in [0, {S}) (named profile table), '
                             f'got range [{int(v.min())}, {int(v.max())}]')
        return
    if isinstance(v, (str, bytes, dict)) or not hasattr(v, '__len__'):
        raise ValueError('root_profile_id must be a 1-D integer sequence or array')
    if len(v) != P:
        raise ValueError(f'root_profile_id has {len(v)} entries, expected population {P}')
    for e in v:
        if isinstance(e, (bool, np.bool_)) or not isinstance(e, (int, np.integer)):
            raise ValueError(f'root_profile_id entries must be integers (not bool/float), got {e!r}')
        if int(e) < 0 or int(e) >= S:
            raise ValueError(f'root_profile_id value {int(e)} outside [0, {S})')


def materialize_profile_ids(v):
    '''int32 [P] conversion of already-checked IDs; called only AFTER the aggregate admission preflight.'''
    return np.ascontiguousarray(np.asarray(v), dtype=np.int32)


def validate_case(case):
    if not isinstance(case, dict):
        raise TypeError('case must be a dict')
    out = dict(case)
    out['name'] = str(case.get('name', 'unnamed'))
    for k in ('nx', 'ny'):
        out[k] = _as_int(case.get(k), k, 1)
    for k in ('grass_count', 'shrub_count'):
        out[k] = _as_int(case.get(k), k, 0)
    out['layers'] = _as_int(case.get('layers'), 'layers', 1)
    out['max_edges'] = _as_int(case.get('max_edges'), 'max_edges', 1)
    out['max_bytes'] = _as_int(case.get('max_bytes'), 'max_bytes', 1)
    out['seed'] = _as_int(case.get('seed', 0), 'seed')
    for k in ('dx', 'grass_diameter', 'shrub_diameter', 'root_multiplier'):
        out[k] = _as_float(case.get(k), k)
    if out['grass_count'] + out['shrub_count'] < 1:
        raise ValueError('population (grass_count + shrub_count) must be positive')
    out['pattern'] = case.get('pattern', 'uniform')
    if out['pattern'] not in ('uniform', 'clumped'):
        raise ValueError('pattern must be uniform or clumped')
    rlf = case.get('root_layer_fractions')
    out.pop('root_layer_fractions', None)
    if rlf is not None:
        validate_root_layer_fractions(rlf, out['layers'])  # pure check; caller-owned rows are not copied
        out['root_layer_fractions'] = rlf
    return out


def resolve_days(days=None, years=None):
    if days is not None:
        return _as_int(days, 'days', 1)
    years = FIXTURE['default_years'] if years is None else _as_int(years, 'years', 1)
    return years * FIXTURE['days_per_year']


def calendar(days):
    '''Continuous weekly clock: 10950 days -> 1564 weekly updates, 2 pending days, 1565 maps.'''
    days = _as_int(days, 'days', 0)
    return {'days': days, 'weekly_updates': days // WEEK, 'pending_days': days % WEEK, 'maps_built': days // WEEK + 1}


_N_SOIL_MAPS, _N_PLANT_ARRAYS, _N_INPUT_ARRAYS, _ID_BYTES = 5, 16, 5, 4


def state_bytes_estimate(P, N, L, n_profiles=2):
    '''Resident state: 5 FP64 [L,N] maps, 21 FP64 P-vectors (plants + inputs), int32 profile IDs [P] and
    the FP64 profile table [S,L]. No global Rh[N] map (removed in 0.2.0).'''
    return (8 * (_N_SOIL_MAPS * L * N + (_N_PLANT_ARRAYS + _N_INPUT_ARRAYS) * P)
            + _ID_BYTES * P + 8 * int(n_profiles) * L)


def graph_bytes_estimate(P, n_edges):
    return int(n_edges) * (4 + 8) + 2 * (P + 1) * 8


def _edge_upper_bound(root_radius, dx, ncells, xp):
    n = xp.floor(2.0 * root_radius / dx) + 2.0
    return float(xp.minimum(n * n, float(ncells)).sum())


def _bbox_cells(root_radius, dx, ncells):
    n = math.floor(2.0 * float(root_radius) / float(dx)) + 2
    return min(int(ncells), n * n)


def initial_edge_bound(case, dx, ncells, inputs=None):
    '''Conservative pre-generation edge bound: each trait counted with its OWN circle bbox
    (grass_count * grass bound + shrub_count * shrub bound), or per-plant bounds for supplied inputs.'''
    if inputs is not None:
        return int(_edge_upper_bound(np.asarray(inputs['root_radius'], dtype=np.float64), dx, ncells, np))
    m = case['root_multiplier']
    return (case['grass_count'] * _bbox_cells(0.5 * case['grass_diameter'] * m, dx, ncells)
            + case['shrub_count'] * _bbox_cells(0.5 * case['shrub_diameter'] * m, dx, ncells))


def scratch_allowance(P, N, L, n_profiles=2):
    '''Documented bounded scratch allowance: P-sized temporaries (edge bound, growth, audits),
    builder count/scan buffers, one [L,N]-sized temporary, profile audit temporaries (S*L booleans and
    up to four overlapping S-sized FP64 row-sum/subtract/abs temporaries = 32*S bytes), plus 64 KiB fixed.'''
    S = int(n_profiles)
    return 64 * P + 8 * L * N + S * L + 32 * S + 65536


def admission_ledger(P, N, L, old_graph_bytes, n_edges, host_mirror_bytes=0, n_profiles=2):
    '''Bytes ledger used for every admission decision (state + old graph + new graph + scratch + mirrors).'''
    led = {'population': int(P), 'ncells': int(N), 'layers': int(L), 'n_edges_bound': int(n_edges),
           'n_profiles': int(n_profiles), 'profile_table_bytes': 8 * int(n_profiles) * int(L),
           'profile_ids_bytes': _ID_BYTES * int(P),
           'state_bytes': state_bytes_estimate(P, N, L, n_profiles), 'old_graph_bytes': int(old_graph_bytes),
           'new_graph_bytes': graph_bytes_estimate(P, n_edges),
           'scratch_allowance_bytes': scratch_allowance(P, N, L, n_profiles),
           'host_mirror_bytes': int(host_mirror_bytes)}
    led['total_bytes'] = (led['state_bytes'] + led['old_graph_bytes'] + led['new_graph_bytes']
                          + led['scratch_allowance_bytes'] + led['host_mirror_bytes'])
    return led


def _reject(stage, reason, estimate):
    '''sm.AdmissionRejected(stage, reason, estimate_dict).'''
    return sm.AdmissionRejected(stage, reason, dict(estimate))


def _gf(graph, key):
    try:
        return graph[key]
    except (TypeError, KeyError, IndexError):
        return getattr(graph, key)


def _ru_maxrss_bytes():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r) * (1 if platform.system() == 'Darwin' else 1024)


def warm_backend(backend):
    '''Load/compile cached kernels on tiny arrays (startup phase, timed separately).'''
    sm.warm_cpu()
    o, c = np.zeros(1, np.int64), np.ones(1, np.int64)
    cells, w = np.zeros(1, np.int32), np.ones(1, np.float64)
    pid, frac = np.zeros(1, np.int32), np.array([[0.75, 0.25]])
    m2 = np.ones((2, 1))
    _demand_scatter_cpu(o, c, cells, w, np.ones(1), pid, frac, m2.copy())
    _block_cpu(m2.copy(), m2.copy(), m2.copy(), m2.copy(), m2.copy(), 1.0, 0.0, 14, 0, 1,
               o, c, cells, w, np.ones(1), pid, frac, np.zeros(1), np.zeros(1))
    _weight_sums_cpu(o, c, w, np.zeros(1))
    if backend == 'gpu':
        sm._gpu_module()
        _gpu_kernels()
        if not _GPU_CACHE.get('warmed'):
            tiny = {'name': 'warm', 'nx': 6, 'ny': 6, 'dx': 1.0, 'grass_count': 2, 'shrub_count': 1,
                    'grass_diameter': 0.5, 'shrub_diameter': 1.0, 'root_multiplier': 3.0, 'pattern': 'uniform',
                    'seed': 7, 'layers': 3, 'max_edges': 10000, 'max_bytes': 50_000_000,
                    'root_layer_fractions': {'grass': [0.7, 0.2, 0.1], 'shrub': [0.15, 0.25, 0.6],
                                             'deep': [0.1, 0.3, 0.6]}}
            inp = dict(sm.generate_inputs(_geometry_case(validate_case(tiny))))
            inp['root_profile_id'] = np.arange(len(inp['x']), dtype=np.int64) % 3  # explicit IDs, 3 profiles
            sim = Simulation(tiny, 'gpu', inputs=inp)  # geometry, daily kernels, growth, scatter, audits
            sim.step_days(WEEK)
            sim.summary()
            del sim
            _GPU_CACHE['warmed'] = True


# ------------------------------------------------------------ simulation state
_LIVE_NAMES = ('W', 'R', 'scale', 'removed_cum', 'overflow_cum', 'B', 'canopy', 'root', 'Dp',
               'uptake_week', 'uptake_cum', 'gain_cum', 'loss_cum', 'wue', 'wsum',
               '_Bc', '_cc', '_rc', '_Dc', '_gc', '_lc', 'root_profile_ids', 'root_layer_fractions')
_INPUT_NAMES = ('x', 'y', 'canopy_radius', 'root_radius', 'ids')


class Simulation:
    '''Resident state for one case/backend. Growth is transactional: candidate, admit, build, swap.'''

    def __init__(self, case, backend='cpu', inputs=None):
        self.case = case = validate_case(case)
        if backend not in ('cpu', 'gpu'):
            raise ValueError('backend must be cpu or gpu')
        self.gpu = backend == 'gpu'
        if self.gpu and not gpu_available():
            raise RuntimeError('requested GPU backend unavailable (CuPy, device and CC>=6.0 required)')
        xp = self.xp = sm.cp if self.gpu else np
        L = self.L = case['layers']
        dx = self.dx = case['dx']
        N = self.N = case['nx'] * case['ny']
        P = int(len(inputs['x'])) if inputs is not None else case['grass_count'] + case['shrub_count']
        if P < 1:
            raise ValueError('population must be positive')
        self.mult = case['root_multiplier']
        self.cap = FIXTURE['storage_m'] / L * dx * dx
        self.rech = FIXTURE['recharge_m'] / L * dx * dx
        self.period = FIXTURE['recharge_period_days']
        self.loss_frac = 1.0 - math.exp(-FIXTURE['turnover_rate_per_day'] * WEEK)
        self.k_gain = FIXTURE['water_density_kg_m3']
        mb = case['max_bytes']
        self.failed, self.last_ledger = None, None
        explicit_ids = None
        if inputs is not None:
            if not isinstance(inputs, dict):
                raise TypeError('inputs must be a dict')
            explicit_ids = inputs.get('root_profile_id')
        # 1) PURE validation of profile rows / explicit IDs (caller-owned; no copies, no table, no int32 vector)
        names, self.profile_source = resolve_root_profile_names(case)
        self.profile_names = names
        S = self.S = len(names)
        if explicit_ids is not None:
            check_profile_ids(explicit_ids, P, S)
        elif inputs is None:  # generated classes follow the counts: refuse missing class rows before generation
            _require_profile_classes(names, [c for c, n in (('grass', case['grass_count']),
                                                            ('shrub', case['shrub_count'])) if n > 0])
        # 2) aggregate admission preflight BEFORE any materialization (table, int32 IDs, generation, transfer);
        #    mirrors here: is_shrub [P] bool + host IDs [P] int32 + host table [S,L] FP64
        mirrors0 = P + _ID_BYTES * P + 8 * S * L
        ledger = admission_ledger(P, N, L, 0, 0, mirrors0, n_profiles=S)  # scalar lower bound: zero edges
        ledger['stage'] = 'init_scalar_bound'
        if ledger['total_bytes'] > mb:
            raise _reject('daily_water_growth.init',
                          f'projected state + scratch (zero edges) already exceeds max_bytes {mb}; '
                          'no edge-bound temporaries computed', ledger)
        est_edges = initial_edge_bound(case, dx, N, inputs)  # P-sized temporaries only after the scalar guard
        ledger = admission_ledger(P, N, L, 0, est_edges, mirrors0, n_profiles=S)
        ledger['stage'] = 'init_bound'
        if ledger['total_bytes'] > mb:
            raise _reject('daily_water_growth.init',
                          f'projected state + initial graph bound + scratch exceeds max_bytes {mb}', ledger)
        sm.check_admission('daily_water_growth.init', P, N, L, case['max_edges'], mb)
        # 3) materialize the small profile table / IDs only after preflight admission
        table = self.host_table = build_root_profile_table(case, names)
        explicit_ids = None if explicit_ids is None else materialize_profile_ids(explicit_ids)
        self.grid = grid = sm.Grid(case['nx'], case['ny'], dx)
        if int(grid.ncells) != N:
            raise ValueError('grid ncells differs from nx*ny')
        if inputs is None:
            inputs = sm.generate_inputs(_geometry_case(case))  # only sm's original geometry fields
        else:
            inputs = {k: np.ascontiguousarray(inputs[k], dtype=(np.int64 if k == 'ids' else np.float64))
                      for k in _INPUT_NAMES}
        sm.validate_inputs(inputs, grid)
        P = self.P = int(len(inputs['x']))
        mirror = sum(int(inputs[k].nbytes) for k in _INPUT_NAMES) + P + _ID_BYTES * P + int(table.nbytes)
        ledger = admission_ledger(P, N, L, 0, initial_edge_bound(case, dx, N, inputs), mirror, n_profiles=S)
        ledger['stage'] = 'init_generated'
        if ledger['total_bytes'] > mb:
            raise _reject('daily_water_growth.init',
                          f'generated population bound exceeds max_bytes {mb} before allocation', ledger)
        self.last_ledger = ledger
        self.host_inputs = inputs
        # trait assignment (fixture): nearest initial diameter; ties -> index < grass_count is grass
        cr = np.asarray(inputs['canopy_radius'], dtype=np.float64)
        rg, rs = 0.5 * case['grass_diameter'], 0.5 * case['shrub_diameter']
        if abs(rg - rs) > 1e-12 * max(rg, rs):
            is_shrub = np.abs(cr - rs) < np.abs(cr - rg)
        else:
            is_shrub = np.arange(P) >= case['grass_count']
        self.is_shrub = is_shrub
        wue_host = np.where(is_shrub, FIXTURE['wue_shrub_kg_per_kg'], FIXTURE['wue_grass_kg_per_kg'])
        # root profile IDs (fixed across weekly growth): explicit caller IDs are authoritative, else the
        # grass/shrub row matching the initial class assignment
        if explicit_ids is None:
            _require_profile_classes(names, [c for c, m in (('grass', ~is_shrub), ('shrub', is_shrub))
                                             if bool(m.any())])  # only classes actually present are required
            gi = names.index('grass') if 'grass' in names else -1
            si = names.index('shrub') if 'shrub' in names else -1
            pid_host = np.where(is_shrub, si, gi).astype(np.int32)
            self.profile_assignment = 'default class row (grass/shrub) from initial canopy-diameter class'
        else:
            if int(explicit_ids.shape[0]) != P:
                raise ValueError('root_profile_id length differs from population')
            pid_host = explicit_ids
            self.profile_assignment = 'explicit inputs[root_profile_id] (authoritative)'
        self.host_profile_ids = np.ascontiguousarray(pid_host, dtype=np.int32)
        self.transfer_record, self.dev_inputs, self.profile_transfer_record = None, None, None
        if self.gpu:
            self.dev_inputs, self.transfer_record = sm.to_device(inputs)
            # separately timed/synchronized H2D copy of the profile IDs/table (setup only; no daily readback)
            t0 = time.perf_counter()
            pid_dev = xp.asarray(self.host_profile_ids, dtype=xp.int32)
            tab_dev = xp.asarray(self.host_table, dtype=xp.float64)
            sm.cp.cuda.Device().synchronize()
            self.profile_transfer_record = {
                'arrays': {'root_profile_ids': int(self.host_profile_ids.nbytes),
                           'root_layer_fractions': int(self.host_table.nbytes)},
                'bytes': int(self.host_profile_ids.nbytes) + int(self.host_table.nbytes),
                'wall_s': time.perf_counter() - t0,
                'label': 'profile IDs/table H2D copy, timed separately from sm.to_device (synchronized once at setup)'}
        else:
            pid_dev, tab_dev = self.host_profile_ids, self.host_table
        src = self.dev_inputs if self.gpu else inputs
        # resident state (all FP64; counted in state_bytes_estimate)
        self.canopy = xp.array(src['canopy_radius'], dtype=xp.float64)
        self.root = xp.array(src['root_radius'], dtype=xp.float64)
        area = PI * self.canopy * self.canopy
        self.B = area * FIXTURE['biomass_density_kg_m2']
        self.Dp = area * FIXTURE['demand_m_per_day']
        self.wue = xp.asarray(wue_host, dtype=xp.float64)
        self.W = xp.full((L, N), FIXTURE['initial_fill_fraction'] * self.cap, dtype=xp.float64)
        for name in ('R', 'scale', 'removed_cum', 'overflow_cum'):
            setattr(self, name, xp.zeros((L, N), dtype=xp.float64))
        self.root_profile_ids = xp.ascontiguousarray(pid_dev)
        self.root_layer_fractions = xp.ascontiguousarray(tab_dev)
        for name in ('uptake_week', 'uptake_cum', 'gain_cum', 'loss_cum', 'wsum',
                     '_Bc', '_cc', '_rc', '_Dc', '_gc', '_lc'):
            setattr(self, name, xp.zeros(P, dtype=xp.float64))
        self.W0 = FIXTURE['initial_fill_fraction'] * self.cap * L * N
        self.B0 = float(self.B.sum())
        self.day = self.weeks_done = self.maps_built = self.demand_map_builds = 0
        self.t_block = self.t_weekly = self.t_audit = 0.0
        self.graph, self.graph_bytes = None, 0
        self._commit_graph(self._build_graph(self.canopy, self.root))
        self.rebuild_peak_bytes = int(self.last_ledger['total_bytes'])  # admitted init ledger estimate
        self._sync()

    # ---- helpers
    @staticmethod
    def _nb(n):
        return (int(n) + 255) // 256

    @staticmethod
    def _gbytes(g):
        return int(sum(int(a.nbytes) for a in g.values()))

    def _sync(self):
        if self.gpu:
            sm.cp.cuda.Device().synchronize()

    def live_bytes(self):
        tot = sum(int(getattr(self, n).nbytes) for n in _LIVE_NAMES)
        src = self.dev_inputs if self.gpu else self.host_inputs
        return tot + sum(int(src[k].nbytes) for k in _INPUT_NAMES)

    def array_records(self):
        where = 'device' if self.gpu else 'host'
        items = [(n, getattr(self, n)) for n in _LIVE_NAMES]
        items += [('graph.' + k, a) for k, a in self.graph.items()]
        src = self.dev_inputs if self.gpu else self.host_inputs
        items += [('inputs.' + k, src[k]) for k in _INPUT_NAMES]
        return [{'name': n, 'dtype': str(a.dtype), 'shape': [int(s) for s in a.shape],
                 'nbytes': int(a.nbytes), 'where': where} for n, a in items]

    # ---- geometry (reuses sm builders; no duplicated geometry)
    def _build_graph(self, canopy, root):
        case = self.case
        d = dict(self.dev_inputs if self.gpu else self.host_inputs)
        d['canopy_radius'], d['root_radius'] = canopy, root
        build = sm.build_root_graph_gpu if self.gpu else sm.build_root_graph_cpu
        raw = build(d, self.grid, self.L, case['max_edges'], case['max_bytes'])
        g = {k: _gf(raw, k) for k in ('offsets', 'counts', 'cells', 'weights')}
        if self.gpu:
            for k, dt in (('offsets', 'int64'), ('counts', 'int64'), ('cells', 'int32'), ('weights', 'float64')):
                if str(g[k].dtype) != dt:
                    raise TypeError(f'GPU graph array {k} has dtype {g[k].dtype}, expected {dt}')
        if int(g['counts'].shape[0]) != self.P:
            raise ValueError('graph population differs from plant population')
        return g

    def _commit_graph(self, g):
        self.graph = g
        self.graph_bytes = self._gbytes(g)
        self.maps_built += 1
        self._rebuild_demand_map()

    def _rebuild_demand_map(self):
        '''Scatter R[l, j] = sum_p Dp*w_pj*f[id_p, l] directly into the resident [L, N] cache (no Rh map).'''
        g = self.graph
        if self.gpu:
            self.R.fill(0.0)
            _gpu_kernels()['demand_scatter']((self._nb(self.P),), (256,), (
                g['offsets'], g['counts'], g['cells'], g['weights'], self.Dp, self.root_profile_ids,
                self.root_layer_fractions, self.R, np.int32(self.L), np.int64(self.N), np.int64(self.P)))
        else:
            _demand_scatter_cpu(g['offsets'], g['counts'], g['cells'], g['weights'], self.Dp,
                                self.root_profile_ids, self.root_layer_fractions, self.R)
        self.demand_map_builds += 1

    def _host_mirror_bytes(self):
        host = int(self.is_shrub.nbytes)
        if self.gpu:
            host += sum(int(self.host_inputs[k].nbytes) for k in _INPUT_NAMES)
            host += int(self.host_profile_ids.nbytes) + int(self.host_table.nbytes)
        return host

    def _weight_sums(self, g=None):
        g = self.graph if g is None else g
        if self.gpu:
            _gpu_kernels()['weight_sums']((self._nb(self.P),), (256,), (
                g['offsets'], g['counts'], g['weights'], self.wsum, np.int64(self.P)))
        else:
            _weight_sums_cpu(g['offsets'], g['counts'], g['weights'], self.wsum)

    # ---- daily blocks
    def _run_block(self, ndays):
        g = self.graph
        if self.gpu:
            k = _gpu_kernels()
            M = self.L * self.N
            gM, gP = (self._nb(M),), (self._nb(self.P),)
            cap, rech = np.float64(self.cap), np.float64(self.rech)
            L, N, P, M64 = np.int32(self.L), np.int64(self.N), np.int64(self.P), np.int64(M)
            for d in range(ndays):
                do_rech = np.int32(1 if (self.day + d) % self.period == 0 else 0)
                k['day_step'](gM, (256,), (self.W, self.R, self.scale, self.removed_cum, self.overflow_cum,
                                            cap, rech, do_rech, M64))
                k['gather'](gP, (256,), (g['offsets'], g['counts'], g['cells'], g['weights'], self.Dp,
                                          self.root_profile_ids, self.root_layer_fractions, self.scale,
                                          L, N, self.uptake_week, self.uptake_cum, P))
        else:
            _block_cpu(self.W, self.R, self.scale, self.removed_cum, self.overflow_cum, self.cap, self.rech,
                       self.period, self.day, ndays, g['offsets'], g['counts'], g['cells'], g['weights'],
                       self.Dp, self.root_profile_ids, self.root_layer_fractions, self.uptake_week,
                       self.uptake_cum)

    def step_days(self, ndays):
        if self.failed is not None:
            raise _reject('daily_water_growth.latched',
                          f'weekly update at day {self.day} was not committed ({self.failed}); '
                          'call retry_weekly_update() before any further daily uptake', {'day': self.day})
        end = self.day + _as_int(ndays, 'ndays', 0)
        while self.day < end:
            n = min(WEEK - self.day % WEEK, end - self.day)
            t0 = time.perf_counter()
            self._run_block(n)
            self.day += n
            self._sync()
            self.t_block += time.perf_counter() - t0
            if self.day % WEEK == 0:
                self._guarded_weekly_update()

    def _guarded_weekly_update(self):
        try:
            self._weekly_update()
        except Exception as exc:
            self.failed = f'{type(exc).__name__}: {exc}'  # latch: prior committed state preserved
            raise

    def retry_weekly_update(self):
        '''Explicitly retry the same pending (uncommitted) weekly boundary; no daily uptake in between.'''
        if self.failed is None or self.day % WEEK != 0 or self.weeks_done == self.day // WEEK:
            raise ValueError('no pending uncommitted weekly update')
        self.failed = None
        self._guarded_weekly_update()

    # ---- weekly transactional growth
    def _growth_candidate(self):
        xp = self.xp
        xp.multiply(self.wue, self.uptake_week, out=self._gc)
        self._gc *= self.k_gain
        xp.multiply(self.B, self.loss_frac, out=self._lc)
        xp.subtract(self.B, self._lc, out=self._Bc)
        self._Bc += self._gc
        xp.multiply(self._Bc, 1.0 / (PI * FIXTURE['biomass_density_kg_m2']), out=self._cc)
        xp.sqrt(self._cc, out=self._cc)
        xp.multiply(self._cc, self.mult, out=self._rc)
        xp.multiply(self._cc, self._cc, out=self._Dc)
        self._Dc *= FIXTURE['demand_m_per_day'] * PI

    def _weekly_update(self):
        t0 = time.perf_counter()
        case, P, mb, xp = self.case, self.P, self.case['max_bytes'], self.xp
        self._growth_candidate()
        for name in ('_Bc', '_cc', '_rc', '_Dc'):  # candidate invariants: weekly scalar readbacks only
            a = getattr(self, name)
            if not bool(xp.isfinite(a).all()) or float(a.min()) <= 0.0:
                raise RuntimeError(f'growth candidate {name} not finite/positive at day {self.day}; state preserved')
        est_edges = _edge_upper_bound(self._rc, self.dx, self.N, xp)
        ledger = admission_ledger(P, self.N, self.L, self.graph_bytes, est_edges, self._host_mirror_bytes(),
                                  n_profiles=self.S)
        ledger['stage'], ledger['day'] = 'rebuild_bound', self.day
        if ledger['total_bytes'] > mb:
            raise _reject('daily_water_growth.rebuild',
                          f'day {self.day}: state + old graph + new graph bound + scratch exceeds max_bytes {mb}; '
                          'prior state preserved', ledger)
        sm.check_admission('daily_water_growth.rebuild', P, self.N, self.L, case['max_edges'], mb)
        g = self._build_graph(self._cc, self._rc)  # may raise AdmissionRejected; state untouched
        ledger = admission_ledger(P, self.N, self.L, self.graph_bytes, int(g['cells'].shape[0]),
                                  self._host_mirror_bytes(), n_profiles=self.S)
        ledger['stage'], ledger['day'] = 'rebuild_actual', self.day
        peak = ledger['total_bytes']
        if peak > mb:
            del g
            raise _reject('daily_water_growth.rebuild',
                          f'day {self.day}: old+new graph peak exceeds max_bytes {mb}; prior state preserved', ledger)
        self._weight_sums(g)  # visits every weight; nonfinite weights give nonfinite sums (no per-edge temporary)
        if (int(g['counts'].min()) <= 0 or not bool(xp.isfinite(self.wsum).all())
                or float(xp.abs(self.wsum - 1.0).max()) > 1e-9):
            del g
            raise RuntimeError(f'candidate graph invariants violated at day {self.day}; state preserved')
        self.last_ledger = ledger
        self.rebuild_peak_bytes = max(self.rebuild_peak_bytes, peak)
        # commit: swap candidate/state buffers, accumulate budget terms, replace graph, rebuild cache
        self.B, self._Bc = self._Bc, self.B
        self.canopy, self._cc = self._cc, self.canopy
        self.root, self._rc = self._rc, self.root
        self.Dp, self._Dc = self._Dc, self.Dp
        self.gain_cum += self._gc
        self.loss_cum += self._lc
        self._commit_graph(g)  # old graph dropped here
        self.uptake_week.fill(0.0)
        self.weeks_done += 1
        self._sync()
        self.t_weekly += time.perf_counter() - t0

    # ---- audits / reporting
    def counters(self):
        return {'day': self.day, 'weeks_done': self.weeks_done, 'pending_days': self.day % WEEK,
                'maps_built': self.maps_built, 'demand_map_builds': self.demand_map_builds,
                'recharge_events': (self.day + self.period - 1) // self.period, 'population': self.P,
                'n_edges': int(self.graph['cells'].shape[0]), 'latched_failure': self.failed}

    def summary(self):
        t0 = time.perf_counter()
        xp, f, day = self.xp, float, self.day
        events = (day + self.period - 1) // self.period
        water_in = events * self.rech * self.L * self.N
        W_total, overflow, removed = f(self.W.sum()), f(self.overflow_cum.sum()), f(self.removed_cum.sum())
        uptake, pending = f(self.uptake_cum.sum()), f(self.uptake_week.sum())
        B_total, gain, loss = f(self.B.sum()), f(self.gain_cum.sum()), f(self.loss_cum.sum())
        water_res = W_total - (self.W0 + water_in - overflow - removed)
        flux_res = uptake - removed
        bio_res = B_total - (self.B0 + gain - loss)
        tol_w = AUDIT_TOL_REL * (self.W0 + water_in) + AUDIT_TOL_ABS
        tol_f = AUDIT_TOL_REL * max(uptake, removed) + AUDIT_TOL_ABS
        tol_b = AUDIT_TOL_REL * (self.B0 + gain) + AUDIT_TOL_ABS
        self._weight_sums()
        wdev = f(xp.abs(self.wsum - 1.0).max())
        rdev = f(xp.abs(self.root / self.canopy - self.mult).max())
        cmin = int(self.graph['counts'].min())
        wmin, wmax, bmin = f(self.W.min()), f(self.W.max()), f(self.B.min())
        # root profile audits (bounded reductions over the S*L table, P ids and the [L,N] cache)
        tab, pid = self.root_layer_fractions, self.root_profile_ids
        pdev = f(xp.abs(tab.sum(axis=1) - 1.0).max())
        pmin, pmax = int(pid.min()), int(pid.max())
        prof_ok = (bool(xp.isfinite(tab).all()) and f(tab.min()) > 0.0 and pdev <= 1e-9
                   and int(tab.shape[0]) == self.S and int(tab.shape[1]) == self.L
                   and int(pid.shape[0]) == self.P and pmin >= 0 and pmax < self.S)
        R_total, D_total = f(self.R.sum()), f(self.Dp.sum())
        req_res = R_total - D_total  # requested demand equals potential demand (weights and fractions sum to 1)
        tol_r = 10.0 * AUDIT_TOL_REL * D_total + AUDIT_TOL_ABS
        checks = {
            'water_finite': bool(xp.isfinite(self.W).all()), 'water_nonnegative': wmin >= 0.0,
            'water_le_capacity': wmax <= self.cap * (1.0 + 1e-12),
            'biomass_finite_positive': bool(xp.isfinite(self.B).all()) and bmin > 0.0,
            'population_retained': int(self.B.shape[0]) == self.P and int(self.graph['counts'].shape[0]) == self.P,
            'positive_root_counts': cmin > 0, 'weights_normalized': wdev <= 1e-9,
            'root_canopy_ratio': rdev <= 1e-9 * self.mult,
            'calendar': (self.weeks_done == day // WEEK and self.maps_built == self.weeks_done + 1
                         and self.demand_map_builds == self.maps_built),
            'water_budget': abs(water_res) <= tol_w, 'plant_grid_flux': abs(flux_res) <= tol_f,
            'biomass_budget': abs(bio_res) <= tol_b,
            'root_profiles_valid': prof_ok, 'requested_demand_total': abs(req_res) <= tol_r,
        }
        out = {'day': day, 'ok': all(checks.values()), 'checks': checks,
               'water': {'initial': self.W0, 'recharge_events': events, 'input': water_in, 'overflow': overflow,
                         'grid_removed': removed, 'plant_uptake': uptake, 'pending_week_uptake': pending,
                         'total': W_total, 'residual': water_res, 'flux_residual': flux_res, 'tolerance': tol_w,
                         'flux_tolerance': tol_f, 'min': wmin, 'max': wmax, 'capacity_per_layer_cell': self.cap},
               'biomass': {'initial': self.B0, 'growth': gain, 'turnover': loss, 'total': B_total,
                           'residual': bio_res, 'tolerance': tol_b, 'min': bmin},
               'graph': {'n_edges': int(self.graph['cells'].shape[0]), 'min_count': cmin,
                         'max_weight_sum_deviation': wdev, 'max_root_canopy_ratio_deviation': rdev},
               'root_profiles': {'names': list(self.profile_names), 'n_profiles': self.S, 'id_min': pmin,
                                 'id_max': pmax, 'max_row_sum_deviation': pdev, 'requested_total': R_total,
                                 'potential_total': D_total, 'requested_residual': req_res,
                                 'requested_tolerance': tol_r},
               'calendar': self.counters()}
        self.t_audit += time.perf_counter() - t0
        return out

    def export_arrays(self):
        conv = sm.cp.asnumpy if self.gpu else np.array
        names = ('W', 'R', 'B', 'canopy', 'root', 'Dp', 'uptake_week', 'uptake_cum', 'gain_cum', 'loss_cum',
                 'removed_cum', 'overflow_cum', 'wue', 'root_profile_ids', 'root_layer_fractions')
        out = {n: conv(getattr(self, n)) for n in names}
        out['graph'] = {k: conv(v) for k, v in self.graph.items()}
        out['is_shrub'] = np.array(self.is_shrub)
        return out


# ------------------------------------------------------------ run_case / CLI
class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _memory_report(sim, hook):
    rep = {'live_arrays': sim.array_records(), 'live_bytes': sim.live_bytes(), 'graph_bytes': sim.graph_bytes,
           'rebuild_peak_bytes': sim.rebuild_peak_bytes,
           'rebuild_peak_label': 'ESTIMATED admission peak: state + old graph + new graph + reserved scratch '
                                 'allowance + host mirrors (max over init and weekly ledgers); not an observed '
                                 'allocator peak (use an external profiler for measured live peak)',
           'max_bytes': sim.case['max_bytes'], 'admission_ledger': sim.last_ledger,
           'scratch_allowance_bytes': scratch_allowance(sim.P, sim.N, sim.L, sim.S),
           'host_mirror_bytes': sim._host_mirror_bytes(), 'process_peak_rss_bytes': _ru_maxrss_bytes(),
           'process_peak_rss_label': 'ru_maxrss whole-process peak (not per array)'}
    if sim.gpu:
        pool = sm.cp.get_default_memory_pool()
        rep.update({'cupy_pool_used_bytes': int(pool.used_bytes()), 'cupy_pool_total_bytes': int(pool.total_bytes()),
                    'cupy_pool_label': 'sampled at end of run'})
        if hook is not None and hasattr(hook, 'report'):
            rep['pool_hook'] = hook.report()
    return rep


def _progress_line(sim, s, elapsed):
    line = {'event': 'progress', 'case': sim.case['name'], 'backend': 'gpu' if sim.gpu else 'cpu',
            'day': sim.day, 'year': sim.day / FIXTURE['days_per_year'], 'water_total_m3': s['water']['total'],
            'water_residual': s['water']['residual'], 'flux_residual': s['water']['flux_residual'],
            'biomass_total_kg': s['biomass']['total'], 'biomass_residual': s['biomass']['residual'],
            'pending_uptake_m3': s['water']['pending_week_uptake'], 'n_edges': s['graph']['n_edges'],
            'audit_ok': s['ok'], 'elapsed_s': elapsed, 'block_wall_s': sim.t_block, 'weekly_wall_s': sim.t_weekly}
    if sim.gpu:
        pool = sm.cp.get_default_memory_pool()
        line['pool_used_bytes'], line['pool_total_bytes'] = int(pool.used_bytes()), int(pool.total_bytes())
    return line


def run_case(case, backend='cpu', days=None, years=None, report_every=365, return_arrays=True,
             inputs=None, memory_hook=None, progress=None):
    '''Run one case on one backend. Scalar validation errors raise before allocation; admission,
    audit and runtime failures are returned as result status (prior valid state preserved).'''
    if memory_hook is not None:
        # enter the external hook exactly once around warm-up, initialization, loop, exports and cleanup
        with memory_hook:
            result = run_case(case, backend, days=days, years=years, report_every=report_every,
                              return_arrays=return_arrays, inputs=inputs, memory_hook=None, progress=progress)
        if hasattr(memory_hook, 'report') and isinstance(result.get('memory'), dict):
            try:
                result['memory']['pool_hook'] = memory_hook.report()
            except Exception as exc:
                result['memory']['pool_hook'] = {'error': repr(exc)}
        return result
    t_start = time.perf_counter()
    case = validate_case(case)
    days = resolve_days(days, years)
    report_every = _as_int(report_every, 'report_every', 1)
    if backend not in ('cpu', 'gpu'):
        raise ValueError('backend must be cpu or gpu')
    result = {'benchmark': 'daily_water_growth', 'version': VERSION, 'status': 'ok', 'case': case,
              'backend': backend, 'days_requested': days, 'report_every': report_every, 'fixture': FIXTURE,
              'environment': environment_info(backend),
              'source': {'geometry_module': sm.__name__, 'geometry_file': getattr(sm, '__file__', None),
                         'geometry_scope_limitations': getattr(sm, 'SCOPE_LIMITATIONS', None)}}
    if backend == 'gpu' and not gpu_available():
        result['status'] = 'gpu_unavailable'
        result['error'] = 'requested GPU backend unavailable (CuPy missing, no device, or CC < 6.0)'
        return result
    timing = {'startup_cache_load_s': None, 'initialization_s': None}
    sim, last, t_loop, progress_lines = None, None, None, []
    try:
        t0 = time.perf_counter()
        warm_backend(backend)
        timing['startup_cache_load_s'] = time.perf_counter() - t0
        t0 = time.perf_counter()
        sim = Simulation(case, backend, inputs=inputs)
        timing['initialization_s'] = time.perf_counter() - t0
        t_loop = time.perf_counter()
        with _NullCtx():  # external hook (if any) already entered at the top of run_case
            while sim.day < days:
                sim.step_days(min(report_every, days - sim.day))
                last = sim.summary()
                line = _progress_line(sim, last, time.perf_counter() - t_loop)
                progress_lines.append(line)
                if progress is not None:
                    progress(line)
                if not last['ok']:
                    result['status'] = 'audit_failed'
                    result['error'] = 'audit failed: ' + ', '.join(k for k, v in last['checks'].items() if not v)
                    break
    except sm.AdmissionRejected as exc:
        result['status'], result['error'] = 'rejected', f'AdmissionRejected: {exc}'
    except Exception as exc:
        result['status'], result['error'] = 'error', f'{type(exc).__name__}: {exc}'
    loop = (time.perf_counter() - t_loop) if t_loop is not None else 0.0
    if sim is not None:
        try:
            if last is None or last['day'] != sim.day:
                last = sim.summary()
        except Exception as exc:
            result['status'] = 'error' if result['status'] == 'ok' else result['status']
            result['error'] = result.get('error') or f'summary failed: {exc}'
        audit = sim.t_audit
        integ = max(loop - audit, 0.0)
        timing.update({'loop_wall_s': loop, 'audit_report_wall_s': audit, 'integration_wall_s': integ,
                       'daily_block_wall_s': sim.t_block, 'weekly_update_wall_s': sim.t_weekly,
                       'end_to_end_wall_s': (timing['initialization_s'] or 0.0) + loop,
                       'note': 'wall clock; GPU synchronized at weekly and report boundaries; '
                               'no asynchronous dispatch times are summed'})
        d = sim.day
        if integ > 0.0 and d > 0:
            result['throughput'] = {'model_days_completed': d, 'model_days_per_wall_s': d / integ,
                                    'model_years_per_wall_min': (d / FIXTURE['days_per_year']) / (integ / 60.0),
                                    'simulated_seconds_per_wall_s': d * 86400.0 / integ}
        result['counters'] = sim.counters()
        result['audit'] = last
        result['memory'] = _memory_report(sim, memory_hook)
        result['transfer_record'] = sim.transfer_record  # sm.to_device: the 5 original input arrays only
        result['profile_transfer_record'] = sim.profile_transfer_record  # IDs/table H2D, timed separately (GPU)
        if sim.gpu:
            in_bytes = sum(int(sim.host_inputs[k].nbytes) for k in _INPUT_NAMES)
            trait_bytes = int(sim.wue.nbytes)  # WUE vector H2D (8*P), timed only within initialization_s
            result['transfer_summary'] = {
                'input_arrays_bytes': in_bytes,
                'profile_bytes': sim.profile_transfer_record['bytes'],
                'trait_bytes': trait_bytes,
                'total_setup_h2d_bytes': in_bytes + sim.profile_transfer_record['bytes'] + trait_bytes,
                'profile_wall_s': sim.profile_transfer_record['wall_s'],
                'label': 'setup transfers: 5 input arrays (sm.to_device record has its own geometry-copy timing) '
                         '+ profile IDs/table (profile_wall_s times only this copy) + WUE traits (included only '
                         'in total initialization_s); no daily transfers or readbacks'}
        result['traits'] = {'n_grass': int((~sim.is_shrub).sum()), 'n_shrub': int(sim.is_shrub.sum()),
                            'assignment': 'nearest initial canopy diameter (ties: index < grass_count)'}
        result['root_profiles'] = {
            'names': list(sim.profile_names), 'n_profiles': sim.S, 'layers': sim.L,
            'table': sim.host_table.tolist(), 'table_dtype': 'float64', 'table_bytes': int(sim.host_table.nbytes),
            'ids_dtype': 'int32', 'ids_bytes': int(sim.host_profile_ids.nbytes),
            'source': sim.profile_source, 'id_assignment': sim.profile_assignment,
            'fixed_across_weekly_growth': True,
            'label': 'uncalibrated benchmark fixture; fractions are potential-uptake weights per layer, '
                     'not volumetric root density or guaranteed actual uptake'}
        if return_arrays:
            result['arrays'] = sim.export_arrays()
    timing['run_case_wall_s'] = time.perf_counter() - t_start
    timing['since_module_import_wall_s'] = time.perf_counter() - _T_IMPORT
    result['timing'], result['progress'] = timing, progress_lines
    return result


def _print_progress(line):
    print(json.dumps(line, default=str), file=sys.stderr)


def _run_isolated(case, backend, args):
    with tempfile.TemporaryDirectory() as td:
        cin, cout = os.path.join(td, 'case.json'), os.path.join(td, 'out.json')
        with open(cin, 'w') as fh:
            json.dump([case], fh)
        cmd = [sys.executable, os.path.abspath(__file__), '--cases-json', cin, '--output', cout, '--backend',
               backend, '--years', str(args.years), '--report-every', str(args.report_every)]
        if args.days is not None:
            cmd += ['--days', str(args.days)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        try:
            with open(cout) as fh:
                r = json.load(fh)['results'][0]
        except Exception as exc:
            r = {'status': 'error', 'error': f'isolated run produced no result: {exc}', 'case': case, 'backend': backend}
        r['isolated'] = {'returncode': proc.returncode, 'stderr_tail': proc.stderr[-4000:]}
        return r


def main(argv=None):
    ap = argparse.ArgumentParser(description='bounded generic daily water / weekly growth benchmark')
    ap.add_argument('--cases-json', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--backend', choices=('cpu', 'gpu', 'both'), default='cpu')
    ap.add_argument('--years', type=int, default=FIXTURE['default_years'])
    ap.add_argument('--days', type=int, default=None)
    ap.add_argument('--report-every', type=int, default=365)
    ap.add_argument('--isolate', action='store_true', help='one subprocess per case/backend (per-case RSS)')
    args = ap.parse_args(argv)
    try:
        resolve_days(args.days, args.years)
        _as_int(args.report_every, 'report_every', 1)
        with open(args.cases_json) as fh:
            cases = json.load(fh)
        if not isinstance(cases, list) or not cases:
            raise ValueError('--cases-json must hold a non-empty list of case dicts')
    except Exception as exc:
        print(json.dumps({'event': 'fatal', 'error': f'{type(exc).__name__}: {exc}'}), file=sys.stderr)
        return 2
    backends = ('cpu', 'gpu') if args.backend == 'both' else (args.backend,)
    results = []
    for case in cases:
        for b in backends:
            try:
                if args.isolate:
                    r = _run_isolated(case, b, args)
                else:
                    r = run_case(case, b, days=args.days, years=args.years, report_every=args.report_every,
                                 return_arrays=False, memory_hook=None, progress=_print_progress)
            except Exception as exc:
                r = {'status': 'error', 'error': f'{type(exc).__name__}: {exc}', 'case': case, 'backend': b}
            results.append(r)
            c = r.get('case')
            print(json.dumps({'event': 'case_done', 'case': c.get('name') if isinstance(c, dict) else None,
                              'backend': b, 'status': r['status'], 'error': r.get('error')}), file=sys.stderr)
    ok = all(r['status'] == 'ok' for r in results)
    out = {'benchmark': 'daily_water_growth', 'version': VERSION, 'status': 'ok' if ok else 'failed',
           'n_results': len(results), 'results': results,
           'process': {'peak_rss_bytes': _ru_maxrss_bytes(), 'label': 'ru_maxrss whole process',
                       'wall_since_import_s': time.perf_counter() - _T_IMPORT}}
    with open(args.output, 'w') as fh:
        json.dump(out, fh, indent=1, default=str)
    return 0 if ok else 2


if __name__ == '__main__':
    sys.exit(main())
