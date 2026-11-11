'''Synthetic sparse circle-to-square-grid geometry benchmark.

Plant footprints (circles) are mapped to the square grid cells they POSITIVELY
overlap, producing a plant-major CSR graph with synthetic UNIFORM weights.  A
layered reduction scatters a unit demand per plant over that graph, and a
canopy projection samples circle cover with fixed 8x8 midpoints per cell.

This is a generic numerical benchmark (graph build, scatter contention,
sampling).  It is NOT an ecological, hydrological or physical model.  See
SCOPE_LIMITATIONS and benchmarks/README.md.
'''
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import platform
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field

import numpy as np
import numba
from numba import njit

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX
    resource = None

try:
    import cupy as cp
    _CUPY_IMPORT_ERROR = None
except Exception as _exc:  # pragma: no cover - machine dependent
    cp = None
    _CUPY_IMPORT_ERROR = repr(_exc)

CANOPY_SUBSAMPLES = 8
INT32_CELL_LIMIT = 2 ** 31 - 1
DEMAND_TOTAL_RTOL = 1e-9
PARITY_RTOL = 1e-9
PARITY_ATOL = 1e-12
WEIGHT_RTOL = 1e-12
PATTERNS = ('uniform', 'clumped')
THREADS = 256
MAX_BLOCKS = 65535
REQUIRED_INPUTS = ('x', 'y', 'canopy_radius', 'root_radius', 'ids')

SCOPE_LIMITATIONS = [
    'Generic synthetic benchmark; no ecological, hydrological, growth, mortality, '
    'wind or water process is modelled.',
    'Root edge weights are UNIFORM over the cells a root circle positively overlaps '
    '(sum to 1 per plant). They are NOT area weighted and encode no scientific '
    'allocation law.',
    'Vertical layer weights are factorized as a uniform 1/L per layer (separability '
    'assumption); no dense plant-by-layer-by-cell tensor is formed.',
    'The layered reduction scatters a synthetic unit demand per plant; it measures '
    'graph traversal and atomic contention, not water competition.',
    'Canopy cover is an 8x8 midpoint sample per cell capped at 1; it is coverage '
    'only, not an exact circle union and not area conserving. Small circles can be '
    'missed entirely (aliasing); missed plants are counted and reported.',
    'Domain is finite, rectangular, non-periodic with square cells; circles are '
    'clipped at the boundary and root weights renormalize over retained cells.',
    'Plant positions (uniform or clumped) are synthetic seeded samples, not an '
    'ecological spatial pattern.',
    'CPU compiled loops are single-threaded; GPU atomics are order-nondeterministic, '
    'so CPU/GPU agreement is checked with tolerances, never bitwise.',
    'ru_maxrss is a cumulative process peak; CuPy pool numbers are sampled at stage '
    'boundaries and are not a true allocation peak.',
]


# --------------------------------------------------------------------------- domain and input validation

@dataclass(frozen=True)
class Grid:
    '''Finite non-periodic domain [0, nx*dx] x [0, ny*dx] of square cells, row-major cell index j*nx+i.'''
    nx: int
    ny: int
    dx: float

    def __post_init__(self):
        for name in ('nx', 'ny'):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or int(v) < 1:
                raise ValueError('%s must be a positive integer, got %r' % (name, v))
        dx = self.dx
        if isinstance(dx, bool) or not isinstance(dx, (int, float, np.integer, np.floating)):
            raise ValueError('dx must be numeric, got %r' % (dx,))
        if not (math.isfinite(float(dx)) and float(dx) > 0.0):
            raise ValueError('dx must be positive and finite, got %r' % (dx,))
        if int(self.nx) * int(self.ny) > INT32_CELL_LIMIT:
            raise ValueError('nx*ny exceeds the int32 cell index guard')

    @property
    def ncells(self) -> int:
        return int(self.nx) * int(self.ny)

    @property
    def width(self) -> float:
        return float(self.nx) * float(self.dx)

    @property
    def height(self) -> float:
        return float(self.ny) * float(self.dx)


def validate_inputs(inputs, grid):
    '''Validate structure-of-arrays plant input; returns the plant count.'''
    for k in REQUIRED_INPUTS:
        if k not in inputs:
            raise ValueError('missing input array %r' % k)
    x, y, rc, rr, ids = (np.asarray(inputs[k]) for k in REQUIRED_INPUTS)
    n = int(x.shape[0]) if x.ndim == 1 else -1
    for name, a, dt in (('x', x, np.float64), ('y', y, np.float64),
                        ('canopy_radius', rc, np.float64), ('root_radius', rr, np.float64),
                        ('ids', ids, np.int64)):
        if a.ndim != 1 or a.shape[0] != n:
            raise ValueError('%s must be 1-D with length %d' % (name, n))
        if a.dtype != np.dtype(dt):
            raise ValueError('%s must have dtype %s, got %s' % (name, np.dtype(dt), a.dtype))
    if n < 1:
        raise ValueError('at least one plant is required')
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError('coordinates must be finite')
    if np.any(x < 0.0) or np.any(x > grid.width) or np.any(y < 0.0) or np.any(y > grid.height):
        raise ValueError('coordinates must lie inside the domain [0,%g]x[0,%g]' % (grid.width, grid.height))
    for name, a in (('canopy_radius', rc), ('root_radius', rr)):
        if not np.all(np.isfinite(a)) or np.any(a <= 0.0):
            raise ValueError('%s must be positive and finite' % name)
    if np.unique(ids).shape[0] != n:
        raise ValueError('ids must be unique')
    return n


def _check_layers(layers):
    if isinstance(layers, bool) or not isinstance(layers, (int, np.integer)) or int(layers) < 1:
        raise ValueError('layers must be a positive integer, got %r' % (layers,))
    return int(layers)


def _check_repeats(repeats):
    if isinstance(repeats, bool) or not isinstance(repeats, (int, np.integer)) or int(repeats) < 1:
        raise ValueError('repeats must be a positive integer, got %r' % (repeats,))
    return int(repeats)


# --------------------------------------------------------------------------- admission / capacity

class AdmissionRejected(Exception):
    '''Raised BEFORE any large edge/grid allocation when declared limits would be exceeded.'''

    def __init__(self, stage, reason, estimate):
        super().__init__('%s: %s' % (stage, reason))
        self.stage = stage
        self.reason = reason
        self.estimate = estimate

    def to_dict(self):
        return {'accepted': False, 'stage': self.stage, 'reason': self.reason, 'estimate': self.estimate}


def estimate_bytes(n, ncells, layers, n_edges=None):
    '''Byte estimate of inputs + CSR graph + grid outputs (reduction[L,ncells], cover, hit).'''
    inputs = 4 * n * 8 + n * 8
    graph = (n + 1) * 8 + n * 8 + (0 if n_edges is None else n_edges * (4 + 8))
    grid_arrays = ncells * 8 * (layers + 1) + n * 4
    return {'inputs': int(inputs), 'graph': int(graph), 'grid': int(grid_arrays),
            'n_edges': None if n_edges is None else int(n_edges),
            'total': int(inputs + graph + grid_arrays)}


def check_admission(stage, n, ncells, layers, max_edges, max_bytes, n_edges=None):
    est = estimate_bytes(n, ncells, layers, n_edges)
    if n_edges is not None and n_edges > max_edges:
        raise AdmissionRejected(stage, 'edge count %d exceeds max_edges %d' % (n_edges, max_edges), est)
    if est['total'] > max_bytes:
        raise AdmissionRejected(stage, 'estimated %d bytes exceeds max_bytes %d' % (est['total'], max_bytes), est)
    return est


# --------------------------------------------------------------------------- compiled CPU kernels

@njit(cache=True)
def _cell_range(c, r, dx, n):
    lo = math.floor((c - r) / dx)
    hi = math.floor((c + r) / dx)
    if lo < 0.0:
        lo = 0.0
    if hi > n - 1.0:
        hi = n - 1.0
    return int(lo), int(hi)


@njit(cache=True)
def _overlaps(px, py, r, i, j, dx):
    # Positive overlap: squared distance from centre to the closest point of the
    # closed square is strictly less than r*r.  Tangent contact is excluded.
    x0 = i * dx
    x1 = (i + 1) * dx
    y0 = j * dx
    y1 = (j + 1) * dx
    cx = px
    if cx < x0:
        cx = x0
    elif cx > x1:
        cx = x1
    cy = py
    if cy < y0:
        cy = y0
    elif cy > y1:
        cy = y1
    ddx = px - cx
    ddy = py - cy
    return ddx * ddx + ddy * ddy < r * r


@njit(cache=True)
def _count_edges(x, y, rr, nx, ny, dx, counts):
    for p in range(x.shape[0]):
        i0, i1 = _cell_range(x[p], rr[p], dx, nx)
        j0, j1 = _cell_range(y[p], rr[p], dx, ny)
        c = 0
        for j in range(j0, j1 + 1):
            for i in range(i0, i1 + 1):
                if _overlaps(x[p], y[p], rr[p], i, j, dx):
                    c += 1
        counts[p] = c


@njit(cache=True)
def _fill_edges(x, y, rr, nx, ny, dx, offsets, cells, weights):
    for p in range(x.shape[0]):
        k = offsets[p]
        cnt = offsets[p + 1] - k
        if cnt == 0:
            continue
        w = 1.0 / cnt
        i0, i1 = _cell_range(x[p], rr[p], dx, nx)
        j0, j1 = _cell_range(y[p], rr[p], dx, ny)
        for j in range(j0, j1 + 1):
            for i in range(i0, i1 + 1):
                if _overlaps(x[p], y[p], rr[p], i, j, dx):
                    cells[k] = j * nx + i
                    weights[k] = w
                    k += 1


@njit(cache=True)
def _layered_reduction(offsets, cells, weights, layers, out):
    out[:, :] = 0.0
    inv = 1.0 / layers
    for p in range(offsets.shape[0] - 1):
        for e in range(offsets[p], offsets[p + 1]):
            v = weights[e] * inv
            c = cells[e]
            for l in range(layers):
                out[l, c] += v


@njit(cache=True)
def _canopy_cover(x, y, rc, nx, ny, dx, cover, hit):
    cover[:] = 0.0
    s = CANOPY_SUBSAMPLES
    step = 1.0 / s
    inv = 1.0 / (s * s)
    for p in range(x.shape[0]):
        i0, i1 = _cell_range(x[p], rc[p], dx, nx)
        j0, j1 = _cell_range(y[p], rc[p], dx, ny)
        r2 = rc[p] * rc[p]
        total = 0
        for j in range(j0, j1 + 1):
            for i in range(i0, i1 + 1):
                cnt = 0
                for b in range(s):
                    sy = (j + (b + 0.5) * step) * dx
                    ddy = sy - y[p]
                    for a in range(s):
                        sx = (i + (a + 0.5) * step) * dx
                        ddx = sx - x[p]
                        if ddx * ddx + ddy * ddy < r2:
                            cnt += 1
                if cnt > 0:
                    cover[j * nx + i] += cnt * inv
                    total += cnt
        hit[p] = 1 if total > 0 else 0


@njit(cache=True)
def _cap_unit(cover):
    for c in range(cover.shape[0]):
        if cover[c] > 1.0:
            cover[c] = 1.0


# --------------------------------------------------------------------------- CPU API

@dataclass
class RootGraph:
    '''Plant-major CSR horizontal graph plus factorized uniform vertical weight 1/layers.'''
    offsets: object
    counts: object
    cells: object
    weights: object
    layers: int
    vertical_weight: float
    n_plants: int
    ncells: int
    estimate: dict
    timings: dict = field(default_factory=dict)
    weight_note: str = ('uniform synthetic weights over positively overlapped cells; '
                        'NOT area weighted; vertical weights factorized (separability assumption)')


def build_root_graph_cpu(inputs, grid, layers=2, max_edges=10_000_000, max_bytes=1 << 31):
    n = validate_inputs(inputs, grid)
    layers = _check_layers(layers)
    check_admission('pre-count', n, grid.ncells, layers, max_edges, max_bytes)
    x, y, rr = inputs['x'], inputs['y'], inputs['root_radius']
    dx = float(grid.dx)
    t0 = time.perf_counter()
    counts = np.empty(n, np.int64)
    _count_edges(x, y, rr, int(grid.nx), int(grid.ny), dx, counts)
    t1 = time.perf_counter()
    n_edges = int(counts.sum())
    est = check_admission('post-count', n, grid.ncells, layers, max_edges, max_bytes, n_edges)
    offsets = np.zeros(n + 1, np.int64)
    np.cumsum(counts, out=offsets[1:])
    t2 = time.perf_counter()
    cells = np.empty(n_edges, np.int32)
    weights = np.empty(n_edges, np.float64)
    _fill_edges(x, y, rr, int(grid.nx), int(grid.ny), dx, offsets, cells, weights)
    t3 = time.perf_counter()
    timings = {'count_seconds': t1 - t0, 'scan_seconds': t2 - t1,
               'fill_seconds': t3 - t2, 'total_seconds': t3 - t0}
    return RootGraph(offsets, counts, cells, weights, layers, 1.0 / layers, n, grid.ncells, est, timings)


def layered_reduction_cpu(graph, out=None):
    '''Scatter a unit demand per plant equally over its edges and layers into float64[L, ncells].'''
    if out is None:
        out = np.empty((graph.layers, graph.ncells), np.float64)
    _layered_reduction(graph.offsets, graph.cells, graph.weights, graph.layers, out)
    return out


@dataclass
class CanopyResult:
    cover: object
    hit: object
    uncapped_sum: float
    capped_cells: int
    missed_plants: int


def _canopy_pass(inputs, grid, cover, hit):
    _canopy_cover(inputs['x'], inputs['y'], inputs['canopy_radius'],
                  int(grid.nx), int(grid.ny), float(grid.dx), cover, hit)
    _cap_unit(cover)


def canopy_cover_cpu(inputs, grid, cover=None, hit=None):
    '''8x8 midpoint-sampled circle cover per cell, capped at 1; reports plants missed by sampling.'''
    n = validate_inputs(inputs, grid)
    if cover is None:
        cover = np.empty(grid.ncells, np.float64)
    if hit is None:
        hit = np.empty(n, np.int32)
    _canopy_cover(inputs['x'], inputs['y'], inputs['canopy_radius'],
                  int(grid.nx), int(grid.ny), float(grid.dx), cover, hit)
    uncapped = float(cover.sum())
    capped_cells = int(np.count_nonzero(cover > 1.0))
    _cap_unit(cover)
    return CanopyResult(cover, hit, uncapped, capped_cells, int(n - int(hit.sum())))


_NJIT_WARM = {'done': False, 'seconds': None}


def warm_cpu():
    '''Run every compiled kernel once on a tiny problem; returns seconds spent (0.0 if already warm).'''
    if _NJIT_WARM['done']:
        return 0.0
    t0 = time.perf_counter()
    grid = Grid(2, 2, 1.0)
    tiny = {'x': np.array([0.5]), 'y': np.array([0.5]), 'canopy_radius': np.array([0.2]),
            'root_radius': np.array([0.2]), 'ids': np.array([0], dtype=np.int64)}
    g = build_root_graph_cpu(tiny, grid, 2, 1000, 1 << 24)
    layered_reduction_cpu(g)
    canopy_cover_cpu(tiny, grid)
    s = time.perf_counter() - t0
    _NJIT_WARM['done'] = True
    _NJIT_WARM['seconds'] = s
    return s


# --------------------------------------------------------------------------- audits / bookkeeping

def audit_reduction(out, counts, layers):
    '''Expected demand is the FULL plant count; zero-edge plants are reported and fail the audit.'''
    out = np.asarray(out)
    counts = np.asarray(counts)
    n = int(counts.shape[0])
    zero_edge = int(np.count_nonzero(counts <= 0))
    shape_ok = bool(out.ndim == 2 and out.shape[0] == int(layers))
    finite = bool(np.all(np.isfinite(out)))
    total = float(out.sum()) if finite else None
    tol = DEMAND_TOTAL_RTOL * max(1.0, float(n))
    layer_sums = out.sum(axis=1) if (shape_ok and finite) else np.zeros(0)
    layer_ok = bool(shape_ok and finite and
                    np.allclose(layer_sums, n / int(layers), rtol=DEMAND_TOTAL_RTOL, atol=1e-12))
    total_ok = bool(finite and abs(total - n) <= tol)
    return {'total_allocated_demand': total, 'expected_demand': n, 'n_plants': n,
            'zero_edge_plants': zero_edge, 'all_plants_have_edges': zero_edge == 0,
            'output_finite': finite, 'shape_consistent': shape_ok,
            'abs_error': None if total is None else abs(total - n),
            'tolerance_abs': tol, 'per_layer_sums': [float(v) for v in layer_sums],
            'passed': bool(zero_edge == 0 and shape_ok and finite and total_ok and layer_ok),
            'note': 'unit synthetic demand per plant; checks conservation of the scatter, not physics'}


def ledger(arrays):
    return {k: {'dtype': str(a.dtype), 'shape': [int(s) for s in a.shape], 'nbytes': int(a.nbytes)}
            for k, a in arrays.items()}


def _ru_maxrss():
    if resource is None:
        return {'ru_maxrss': None, 'label': 'unavailable on this platform'}
    return {'ru_maxrss': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            'unit': 'KiB on Linux (bytes on macOS)',
            'label': 'cumulative process peak RSS, not stage specific'}


def _timing_block(times, note):
    return {'repeats_seconds': [float(t) for t in times], 'median_seconds': float(statistics.median(times)),
            'n_repeats': len(times), 'warmup_excluded': True, 'note': note}


def _canopy_stats(res, inputs, grid):
    rc = inputs['canopy_radius']
    dx = float(grid.dx)
    return {'missed_plants': int(res.missed_plants), 'n_plants': int(rc.shape[0]),
            'cells_capped_at_one': int(res.capped_cells),
            'uncapped_sampled_cover_sum': float(res.uncapped_sum),
            'capped_cover_sum': float(res.cover.sum()) if hasattr(res.cover, 'sum') else None,
            'sampled_area_uncapped': float(res.uncapped_sum) * dx * dx,
            'analytic_unclipped_disc_area_sum': float(np.sum(math.pi * rc * rc)),
            'subsamples_per_axis': CANOPY_SUBSAMPLES,
            'note': 'coverage only; 8x8 midpoint sampling is not area conserving and misses small '
                    'circles in coarse cells (aliasing); missed plants are reported, not discarded'}


def _close(a, b, rtol, atol):
    a = np.asarray(a)
    b = np.asarray(b)
    return bool(a.shape == b.shape and np.allclose(a, b, rtol=rtol, atol=atol))


def compare_backends(a, b):
    '''Tolerance-based CPU/GPU agreement; graph structure must match exactly.'''
    checks = {
        'offsets_exact': bool(np.array_equal(a['offsets'], b['offsets'])),
        'cells_exact': bool(np.array_equal(a['cells'], b['cells'])),
        'weights': _close(a['weights'], b['weights'], WEIGHT_RTOL, 0.0),
        'reduction': _close(a['reduction'], b['reduction'], PARITY_RTOL, PARITY_ATOL),
        'cover': _close(a['cover'], b['cover'], PARITY_RTOL, PARITY_ATOL),
        'hit_exact': bool(np.array_equal(a['hit'], b['hit'])),
    }
    diffs = {}
    for k in ('reduction', 'cover'):
        if np.asarray(a[k]).shape == np.asarray(b[k]).shape:
            diffs[k] = float(np.max(np.abs(np.asarray(a[k]) - np.asarray(b[k]))))
    return {'passed': all(checks.values()), 'checks': checks, 'max_abs_diff': diffs,
            'tolerances': {'weights_rtol': WEIGHT_RTOL, 'rtol': PARITY_RTOL, 'atol': PARITY_ATOL},
            'note': 'GPU atomics are order nondeterministic; bitwise equality is not expected'}


# --------------------------------------------------------------------------- CPU benchmark driver

def run_cpu(inputs, grid, layers, max_edges, max_bytes, repeats):
    repeats = _check_repeats(repeats)
    layers = _check_layers(layers)
    rec = {'backend': 'cpu', 'status': 'ok',
           'note': 'single-threaded Numba njit(cache=True) loops timed with perf_counter'}
    warm = warm_cpu()
    rec['startup'] = {'njit_warm_seconds_this_call': warm,
                      'njit_warm_seconds_first_in_process': _NJIT_WARM['seconds'],
                      'note': 'tiny-problem warm-up including compile or cache load; excluded from repeats'}
    try:
        graph = build_root_graph_cpu(inputs, grid, layers, max_edges, max_bytes)
    except AdmissionRejected as exc:
        rec['status'] = 'rejected'
        rec['admission'] = exc.to_dict()
        return rec
    rec['admission'] = {'accepted': True, 'estimate': graph.estimate}
    rec['build'] = dict(graph.timings, n_edges=int(graph.cells.shape[0]), synchronized_wall=True)
    out = np.empty((layers, grid.ncells), np.float64)
    layered_reduction_cpu(graph, out)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        layered_reduction_cpu(graph, out)
        times.append(time.perf_counter() - t0)
    rec['reduction'] = _timing_block(times, 'includes explicit zero-init of the [L,ncells] output')
    cover = np.empty(grid.ncells, np.float64)
    hit = np.empty(graph.n_plants, np.int32)
    res = canopy_cover_cpu(inputs, grid, cover, hit)
    ctimes = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _canopy_pass(inputs, grid, cover, hit)
        ctimes.append(time.perf_counter() - t0)
    rec['canopy'] = dict(_timing_block(ctimes, 'sampling kernel plus cap pass'), **_canopy_stats(res, inputs, grid))
    rec['audit'] = audit_reduction(out, graph.counts, layers)
    rec['ledger'] = ledger({'x': inputs['x'], 'y': inputs['y'], 'canopy_radius': inputs['canopy_radius'],
                            'root_radius': inputs['root_radius'], 'ids': inputs['ids'],
                            'offsets': graph.offsets, 'counts': graph.counts, 'cells': graph.cells,
                            'weights': graph.weights, 'reduction': out, 'cover': cover, 'hit': hit})
    rec['ru_maxrss'] = _ru_maxrss()
    rec['arrays'] = {'offsets': graph.offsets, 'counts': graph.counts, 'cells': graph.cells,
                     'weights': graph.weights, 'reduction': out, 'cover': cover, 'hit': hit}
    if not rec['audit']['passed']:
        rec['status'] = 'failed'
    return rec


# --------------------------------------------------------------------------- GPU (CuPy raw CUDA) path

_CUDA_SRC = r'''
extern "C" {

__device__ __forceinline__ void cell_range(double c, double r, double dx, int n, int* lo, int* hi) {
    double l = floor((c - r) / dx);
    double h = floor((c + r) / dx);
    if (l < 0.0) l = 0.0;
    if (h > n - 1.0) h = n - 1.0;
    *lo = (int)l;
    *hi = (int)h;
}

__device__ __forceinline__ bool overlaps(double px, double py, double r, int i, int j, double dx) {
    double x0 = i * dx;
    double x1 = (i + 1) * dx;
    double y0 = j * dx;
    double y1 = (j + 1) * dx;
    double cx = px < x0 ? x0 : (px > x1 ? x1 : px);
    double cy = py < y0 ? y0 : (py > y1 ? y1 : py);
    double ddx = px - cx;
    double ddy = py - cy;
    return ddx * ddx + ddy * ddy < r * r;
}

__global__ void count_edges(const double* x, const double* y, const double* rr,
                            int n, int nx, int ny, double dx, long long* counts) {
    int stride = gridDim.x * blockDim.x;
    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < n; p += stride) {
        int i0, i1, j0, j1;
        cell_range(x[p], rr[p], dx, nx, &i0, &i1);
        cell_range(y[p], rr[p], dx, ny, &j0, &j1);
        long long c = 0;
        for (int j = j0; j <= j1; ++j)
            for (int i = i0; i <= i1; ++i)
                if (overlaps(x[p], y[p], rr[p], i, j, dx)) ++c;
        counts[p] = c;
    }
}

__global__ void fill_edges(const double* x, const double* y, const double* rr,
                           int n, int nx, int ny, double dx, const long long* offsets,
                           int* cells, double* weights) {
    int stride = gridDim.x * blockDim.x;
    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < n; p += stride) {
        long long k = offsets[p];
        long long cnt = offsets[p + 1] - k;
        if (cnt == 0) continue;
        double w = 1.0 / cnt;
        int i0, i1, j0, j1;
        cell_range(x[p], rr[p], dx, nx, &i0, &i1);
        cell_range(y[p], rr[p], dx, ny, &j0, &j1);
        for (int j = j0; j <= j1; ++j)
            for (int i = i0; i <= i1; ++i)
                if (overlaps(x[p], y[p], rr[p], i, j, dx)) {
                    cells[k] = j * nx + i;
                    weights[k] = w;
                    ++k;
                }
    }
}

__global__ void layered_reduction(const long long* offsets, const int* cells, const double* weights,
                                  int n, int layers, long long ncells, double* out) {
    int stride = gridDim.x * blockDim.x;
    double inv = 1.0 / layers;
    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < n; p += stride) {
        for (long long e = offsets[p]; e < offsets[p + 1]; ++e) {
            double v = weights[e] * inv;
            long long c = cells[e];
            for (int l = 0; l < layers; ++l) atomicAdd(&out[l * ncells + c], v);
        }
    }
}

__global__ void canopy_cover(const double* x, const double* y, const double* rc,
                             int n, int nx, int ny, double dx, double* cover, int* hit) {
    const int S = 8;
    const double step = 1.0 / S;
    const double inv = 1.0 / (S * S);
    int stride = gridDim.x * blockDim.x;
    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < n; p += stride) {
        int i0, i1, j0, j1;
        cell_range(x[p], rc[p], dx, nx, &i0, &i1);
        cell_range(y[p], rc[p], dx, ny, &j0, &j1);
        double r2 = rc[p] * rc[p];
        long long total = 0;
        for (int j = j0; j <= j1; ++j)
            for (int i = i0; i <= i1; ++i) {
                int cnt = 0;
                for (int b = 0; b < S; ++b) {
                    double sy = (j + (b + 0.5) * step) * dx;
                    double ddy = sy - y[p];
                    for (int a = 0; a < S; ++a) {
                        double sx = (i + (a + 0.5) * step) * dx;
                        double ddx = sx - x[p];
                        if (ddx * ddx + ddy * ddy < r2) ++cnt;
                    }
                }
                if (cnt > 0) {
                    atomicAdd(&cover[(long long)j * nx + i], cnt * inv);
                    total += cnt;
                }
            }
        hit[p] = total > 0 ? 1 : 0;
    }
}

__global__ void cap_unit(double* cover, long long ncells) {
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x; c < ncells; c += stride)
        if (cover[c] > 1.0) cover[c] = 1.0;
}

}
'''

_KERNEL_NAMES = ('count_edges', 'fill_edges', 'layered_reduction', 'canopy_cover', 'cap_unit')
_GPU = {'status': None, 'module': None, 'compile_seconds': None}


def gpu_status():
    '''Probe CuPy/CUDA once per process; records the exact exception when unavailable.'''
    if _GPU['status'] is not None:
        return _GPU['status']
    if cp is None:
        st = {'available': False, 'reason': 'cupy import failed: %s' % _CUPY_IMPORT_ERROR}
    else:
        try:
            t0 = time.perf_counter()
            dev = cp.cuda.Device()
            cc = int(dev.compute_capability)
            cp.cuda.runtime.deviceSynchronize()
            ctx_s = time.perf_counter() - t0
            props = cp.cuda.runtime.getDeviceProperties(dev.id)
            name = props.get('name', b'')
            name = name.decode(errors='replace') if isinstance(name, bytes) else str(name)
            if cc < 60:
                st = {'available': False,
                      'reason': 'compute capability %d < 60 required for FP64 atomicAdd' % cc}
            else:
                st = {'available': True, 'device_name': name, 'compute_capability': cc,
                      'context_init_seconds': ctx_s, 'cupy_version': cp.__version__,
                      'cuda_runtime_version': int(cp.cuda.runtime.runtimeGetVersion())}
        except Exception as exc:
            st = {'available': False, 'reason': repr(exc)}
    _GPU['status'] = st
    return st


def _gpu_module():
    if _GPU['module'] is None:
        t0 = time.perf_counter()
        mod = cp.RawModule(code=_CUDA_SRC, options=('-fmad=false',))
        mod.compile()
        for name in _KERNEL_NAMES:
            mod.get_function(name)
        cp.cuda.runtime.deviceSynchronize()
        _GPU['module'] = mod
        _GPU['compile_seconds'] = time.perf_counter() - t0
    return _GPU['module']


def _blocks(n):
    return max(1, min((int(n) + THREADS - 1) // THREADS, MAX_BLOCKS))


def _pool_sample(pool):
    return {'used_bytes': int(pool.used_bytes()), 'total_bytes': int(pool.total_bytes()),
            'label': 'sampled at stage boundary; not a true allocation peak'}


def _event_seconds(fn):
    s = cp.cuda.Event()
    e = cp.cuda.Event()
    s.record()
    fn()
    e.record()
    e.synchronize()
    return float(cp.cuda.get_elapsed_time(s, e)) / 1000.0


def to_device(inputs):
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    d = {k: cp.asarray(inputs[k]) for k in REQUIRED_INPUTS}
    cp.cuda.runtime.deviceSynchronize()
    t1 = time.perf_counter()
    return d, {'seconds': t1 - t0, 'bytes': int(sum(a.nbytes for a in d.values()))}


def build_root_graph_gpu(d, grid, layers=2, max_edges=10_000_000, max_bytes=1 << 31):
    '''Same predicate, cell order and normalization as the CPU build; one thread per plant.'''
    layers = _check_layers(layers)
    n = int(d['x'].shape[0])
    check_admission('pre-count', n, grid.ncells, layers, max_edges, max_bytes)
    mod = _gpu_module()
    nx, ny, dx = np.int32(grid.nx), np.int32(grid.ny), np.float64(grid.dx)
    blocks = _blocks(n)
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    counts = cp.empty(n, cp.int64)
    mod.get_function('count_edges')((blocks,), (THREADS,),
                                    (d['x'], d['y'], d['root_radius'], np.int32(n), nx, ny, dx, counts))
    offsets = cp.zeros(n + 1, cp.int64)
    offsets[1:] = cp.cumsum(counts)
    n_edges = int(offsets[-1].get())  # single scalar D2H: capacity guard before edge allocation
    est = check_admission('post-count', n, grid.ncells, layers, max_edges, max_bytes, n_edges)
    cells = cp.empty(n_edges, cp.int32)
    weights = cp.empty(n_edges, cp.float64)
    mod.get_function('fill_edges')((blocks,), (THREADS,),
                                   (d['x'], d['y'], d['root_radius'], np.int32(n), nx, ny, dx,
                                    offsets, cells, weights))
    cp.cuda.runtime.deviceSynchronize()
    t1 = time.perf_counter()
    timings = {'total_seconds': t1 - t0, 'note': 'count + device scan + guard scalar D2H + fill, synchronized'}
    return RootGraph(offsets, counts, cells, weights, layers, 1.0 / layers, n, grid.ncells, est, timings)


def layered_reduction_gpu(graph, out):
    out.fill(0.0)
    _gpu_module().get_function('layered_reduction')(
        (_blocks(graph.n_plants),), (THREADS,),
        (graph.offsets, graph.cells, graph.weights, np.int32(graph.n_plants),
         np.int32(graph.layers), np.int64(graph.ncells), out))
    return out


def canopy_cover_gpu(d, grid, cover, hit, cap=True):
    n = int(d['x'].shape[0])
    mod = _gpu_module()
    cover.fill(0.0)
    mod.get_function('canopy_cover')((_blocks(n),), (THREADS,),
                                     (d['x'], d['y'], d['canopy_radius'], np.int32(n), np.int32(grid.nx),
                                      np.int32(grid.ny), np.float64(grid.dx), cover, hit))
    if cap:
        mod.get_function('cap_unit')((_blocks(grid.ncells),), (THREADS,), (cover, np.int64(grid.ncells)))
    return cover


_GPU_WARM = {'done': False, 'seconds': None}


def warm_gpu():
    '''Once per process: full H2D -> CSR build (incl. device scan) -> reduction -> canopy -> D2H on a
    fixed multi-plant fixture.  Returns seconds spent (0.0 if already warm).  Not a true cold start.'''
    if _GPU_WARM['done']:
        return 0.0
    t0 = time.perf_counter()
    grid = Grid(4, 3, 1.0)
    m = 1024
    fixture = {'x': np.linspace(0.25, 3.75, m), 'y': np.linspace(0.2, 2.8, m),
               'canopy_radius': np.full(m, 0.6), 'root_radius': np.full(m, 0.6),
               'ids': np.arange(m, dtype=np.int64)}
    d, _ = to_device(fixture)
    g = build_root_graph_gpu(d, grid, 2, 10_000, 1 << 24)
    out = cp.empty((2, grid.ncells), cp.float64)
    layered_reduction_gpu(g, out)
    cover = cp.empty(grid.ncells, cp.float64)
    hit = cp.empty(m, cp.int32)
    canopy_cover_gpu(d, grid, cover, hit)
    out.get()
    cover.get()
    hit.get()
    cp.cuda.runtime.deviceSynchronize()
    s = time.perf_counter() - t0
    _GPU_WARM['done'] = True
    _GPU_WARM['seconds'] = s
    return s


def run_gpu(inputs, grid, layers, max_edges, max_bytes, repeats):
    repeats = _check_repeats(repeats)
    layers = _check_layers(layers)
    validate_inputs(inputs, grid)
    st = gpu_status()
    rec = {'backend': 'gpu', 'device': st}
    if not st['available']:
        rec['status'] = 'unavailable'
        rec['reason'] = st.get('reason')
        return rec
    try:
        try:  # cheap scalar guard BEFORE any device transfer, compile or warm-up
            check_admission('pre-count', int(inputs['x'].shape[0]), grid.ncells, layers, max_edges, max_bytes)
        except AdmissionRejected as exc:
            rec['status'] = 'rejected'
            rec['admission'] = exc.to_dict()
            return rec
        compiled_now = _GPU['module'] is None
        _gpu_module()
        warmed_now = not _GPU_WARM['done']
        warm = warm_gpu()
        rec['startup'] = {'context_init_seconds': st.get('context_init_seconds'),
                          'kernel_compile_seconds': _GPU['compile_seconds'],
                          'compiled_in_this_call': compiled_now,
                          'gpu_warmup_seconds_this_call': warm,
                          'gpu_warmup_seconds_first_in_process': _GPU_WARM['seconds'],
                          'warmed_in_this_call': warmed_now,
                          'note': 'compile (possibly served from the CuPy kernel cache) and a fixed '
                                  'small-fixture H2D->build->reduction->canopy->D2H warm-up are reported '
                                  'separately and excluded from timed build/repeats; not a true cold start'}
        pool = cp.get_default_memory_pool()
        samples = {'before_h2d': _pool_sample(pool)}
        d, h2d = to_device(inputs)
        rec['h2d'] = h2d
        try:
            graph = build_root_graph_gpu(d, grid, layers, max_edges, max_bytes)
        except AdmissionRejected as exc:
            rec['status'] = 'rejected'
            rec['admission'] = exc.to_dict()
            rec['pool_samples'] = samples
            return rec
        rec['admission'] = {'accepted': True, 'estimate': graph.estimate}
        rec['build'] = dict(graph.timings, n_edges=int(graph.cells.shape[0]), synchronized_wall=True)
        samples['after_build'] = _pool_sample(pool)
        out = cp.empty((layers, grid.ncells), cp.float64)
        layered_reduction_gpu(graph, out)
        cp.cuda.runtime.deviceSynchronize()
        times = [_event_seconds(lambda: layered_reduction_gpu(graph, out)) for _ in range(repeats)]
        rec['reduction'] = _timing_block(times, 'CUDA events; includes explicit device zero-init; graph resident on device')
        cover = cp.empty(grid.ncells, cp.float64)
        hit = cp.empty(graph.n_plants, cp.int32)
        canopy_cover_gpu(d, grid, cover, hit, cap=False)
        uncapped = float(cover.sum().get())
        capped_cells = int(cp.count_nonzero(cover > 1.0).get())
        _gpu_module().get_function('cap_unit')((_blocks(grid.ncells),), (THREADS,), (cover, np.int64(grid.ncells)))
        cp.cuda.runtime.deviceSynchronize()
        ctimes = [_event_seconds(lambda: canopy_cover_gpu(d, grid, cover, hit)) for _ in range(repeats)]
        samples['after_kernels'] = _pool_sample(pool)
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        host = {'offsets': graph.offsets.get(), 'counts': graph.counts.get(), 'cells': graph.cells.get(),
                'weights': graph.weights.get(), 'reduction': out.get(), 'cover': cover.get(), 'hit': hit.get()}
        cp.cuda.runtime.deviceSynchronize()
        rec['d2h'] = {'seconds': time.perf_counter() - t0,
                      'bytes': int(sum(a.nbytes for a in host.values())),
                      'note': 'validation copies outside kernel timing'}
        res = CanopyResult(host['cover'], host['hit'], uncapped, capped_cells,
                           int(graph.n_plants - int(host['hit'].sum())))
        rec['canopy'] = dict(_timing_block(ctimes, 'CUDA events; sampling kernel plus cap kernel'),
                             **_canopy_stats(res, inputs, grid))
        rec['audit'] = audit_reduction(host['reduction'], host['counts'], layers)
        rec['ledger'] = ledger(dict(d, offsets=graph.offsets, counts=graph.counts, cells=graph.cells,
                                    weights=graph.weights, reduction=out, cover=cover, hit=hit))
        rec['pool_samples'] = samples
        rec['ru_maxrss'] = _ru_maxrss()
        rec['arrays'] = host
        rec['status'] = 'failed' if not rec['audit']['passed'] else 'ok'
        return rec
    except Exception as exc:
        rec['status'] = 'error'
        rec['exception'] = repr(exc)
        rec['traceback'] = traceback.format_exc()
        return rec


# --------------------------------------------------------------------------- cases and generation

CASE_FIELDS = {'name': str, 'nx': int, 'ny': int, 'dx': float, 'grass_count': int, 'shrub_count': int,
               'grass_diameter': float, 'shrub_diameter': float, 'root_multiplier': float,
               'pattern': str, 'seed': int, 'layers': int, 'max_edges': int, 'max_bytes': int}
CASE_DEFAULTS = {'layers': 2}


def validate_case(case):
    if not isinstance(case, dict):
        raise ValueError('case must be a JSON object')
    unknown = sorted(set(case) - set(CASE_FIELDS))
    if unknown:
        raise ValueError('unknown case fields: %s' % unknown)
    out = {}
    for k, typ in CASE_FIELDS.items():
        if k not in case:
            if k in CASE_DEFAULTS:
                out[k] = CASE_DEFAULTS[k]
                continue
            raise ValueError('case missing required field %r' % k)
        v = case[k]
        if isinstance(v, bool):
            raise ValueError('field %r must not be boolean' % k)
        if typ is int:
            if not isinstance(v, int):
                raise ValueError('field %r must be an integer' % k)
            out[k] = int(v)
        elif typ is float:
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError('field %r must be a finite number' % k)
            out[k] = float(v)
        else:
            if not isinstance(v, str) or not v:
                raise ValueError('field %r must be a non-empty string' % k)
            out[k] = v
    if out['nx'] < 1 or out['ny'] < 1 or out['dx'] <= 0.0:
        raise ValueError('nx, ny must be >= 1 and dx > 0')
    Grid(out['nx'], out['ny'], out['dx'])
    if out['grass_count'] < 0 or out['shrub_count'] < 0 or out['grass_count'] + out['shrub_count'] < 1:
        raise ValueError('plant counts must be non-negative with at least one plant')
    if out['grass_diameter'] <= 0.0 or out['shrub_diameter'] <= 0.0 or out['root_multiplier'] <= 0.0:
        raise ValueError('diameters and root_multiplier must be positive')
    if out['pattern'] not in PATTERNS:
        raise ValueError('pattern must be one of %s' % (PATTERNS,))
    if out['seed'] < 0 or out['layers'] < 1 or out['max_edges'] < 0 or out['max_bytes'] < 0:
        raise ValueError('seed, max_edges, max_bytes must be >= 0 and layers >= 1')
    return out


MAX_CLUSTERS = 8
CLUMP_SIGMA_FRACTION = 0.02


def clump_parameters(n, grid):
    '''Deliberate synthetic contention fixture (not an ecological pattern): at most MAX_CLUSTERS
    cluster centres and gaussian sigma = CLUMP_SIGMA_FRACTION * min(width, height).'''
    return min(MAX_CLUSTERS, max(1, n // 64)), CLUMP_SIGMA_FRACTION * min(grid.width, grid.height)


def generate_inputs(case):
    '''Deterministic seeded host input; identical arrays are used for CPU and GPU runs.'''
    case = validate_case(case)
    grid = Grid(case['nx'], case['ny'], case['dx'])
    g, s = case['grass_count'], case['shrub_count']
    n = g + s
    rng = np.random.default_rng(case['seed'])
    w, h = grid.width, grid.height
    if case['pattern'] == 'uniform':
        x = rng.uniform(0.0, w, n)
        y = rng.uniform(0.0, h, n)
    else:  # synthetic contention fixture: <= 8 seeded gaussian clusters, clipped to the domain boundary
        k, sigma = clump_parameters(n, grid)
        cx = rng.uniform(0.0, w, k)
        cy = rng.uniform(0.0, h, k)
        a = rng.integers(0, k, n)
        x = np.clip(cx[a] + rng.normal(0.0, sigma, n), 0.0, w)
        y = np.clip(cy[a] + rng.normal(0.0, sigma, n), 0.0, h)
    canopy = np.concatenate([np.full(g, 0.5 * case['grass_diameter']), np.full(s, 0.5 * case['shrub_diameter'])])
    canopy = np.ascontiguousarray(canopy, dtype=np.float64)
    root = np.ascontiguousarray(canopy * case['root_multiplier'], dtype=np.float64)
    return {'x': np.ascontiguousarray(x, dtype=np.float64), 'y': np.ascontiguousarray(y, dtype=np.float64),
            'canopy_radius': canopy, 'root_radius': root, 'ids': np.arange(n, dtype=np.int64)}


def run_case(case, backend='cpu', repeats=3):
    case = validate_case(case)
    if backend not in ('cpu', 'gpu', 'both'):
        raise ValueError('backend must be cpu, gpu or both, got %r' % (backend,))
    repeats = _check_repeats(repeats)
    t_case = time.perf_counter()
    rec = {'name': case['name'], 'case': case, 'status': 'ok', 'backend': backend}
    try:
        grid = Grid(case['nx'], case['ny'], case['dx'])
        n = case['grass_count'] + case['shrub_count']
        rec['n_plants'] = n
        rec['ncells'] = grid.ncells
        rec['admission_limits'] = {'max_edges': case['max_edges'], 'max_bytes': case['max_bytes'],
                                   'pre_count_estimate': estimate_bytes(n, grid.ncells, case['layers'])}
        # cheap scalar guard BEFORE any host array generation (never truncates the population)
        check_admission('pre-count', n, grid.ncells, case['layers'], case['max_edges'], case['max_bytes'])
        k, sigma = clump_parameters(n, grid)
        clumped = case['pattern'] == 'clumped'
        rec['generation'] = {'pattern': case['pattern'], 'seed': case['seed'],
                             'cluster_count': k if clumped else None,
                             'cluster_sigma': sigma if clumped else None,
                             'note': 'synthetic seeded positions; clumped is a deliberate contention '
                                     'fixture (<= 8 gaussian clusters clipped to the domain boundary), '
                                     'not an ecological pattern; root_radius = canopy_radius * root_multiplier'}
        inputs = generate_inputs(case)
        args = (inputs, grid, case['layers'], case['max_edges'], case['max_bytes'], repeats)
        if backend in ('cpu', 'both'):
            rec['cpu'] = run_cpu(*args)
        if backend in ('gpu', 'both'):
            rec['gpu'] = run_gpu(*args)
        if 'cpu' in rec and 'gpu' in rec and rec['cpu']['status'] == 'ok' and rec['gpu']['status'] == 'ok':
            rec['parity'] = compare_backends(rec['cpu']['arrays'], rec['gpu']['arrays'])
            if not rec['parity']['passed']:
                rec['gpu']['status'] = 'failed'
        for k in ('cpu', 'gpu'):
            if k in rec:
                rec[k].pop('arrays', None)
        sub = [rec[k]['status'] for k in ('cpu', 'gpu') if k in rec]
        if 'error' in sub:
            rec['status'] = 'error'
        elif 'failed' in sub:
            rec['status'] = 'failed'
        elif 'rejected' in sub:
            rec['status'] = 'rejected'
        elif 'unavailable' in sub:
            rec['status'] = 'partial' if any(s == 'ok' for s in sub) else 'unavailable'
    except AdmissionRejected as exc:
        rec['status'] = 'rejected'
        rec['admission'] = exc.to_dict()
    except Exception as exc:
        rec['status'] = 'error'
        rec['exception'] = repr(exc)
        rec['traceback'] = traceback.format_exc()
    rec['case_wall_seconds'] = time.perf_counter() - t_case
    rec['case_wall_note'] = ('host generation/validation plus every requested backend including startup '
                             'and warm-up; distinct from kernel timings. Single-thread CPU vs GPU kernel '
                             'medians are a kernel comparison, not a scientific speedup claim')
    rec['ru_maxrss'] = _ru_maxrss()
    return rec


def environment_info(query_gpu):
    return {'timestamp_utc': _dt.datetime.now(_dt.timezone.utc).isoformat(),
            'python': platform.python_version(), 'numpy': np.__version__, 'numba': numba.__version__,
            'cupy': None if cp is None else cp.__version__, 'cupy_import_error': _CUPY_IMPORT_ERROR,
            'platform': platform.platform(), 'machine': platform.machine(), 'cpu_count': os.cpu_count(),
            'gpu': gpu_status() if query_gpu else {'queried': False}}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError('not JSON serializable: %r' % type(o))


def main(argv=None):
    ap = argparse.ArgumentParser(description='Synthetic sparse circle-grid geometry benchmark (not an ecological model).')
    ap.add_argument('--cases-json', required=True, help='path to a JSON list of case objects')
    ap.add_argument('--output', required=True, help='path for the JSON result')
    ap.add_argument('--backend', choices=('cpu', 'gpu', 'both'), default='cpu')
    ap.add_argument('--repeats', type=int, default=3)
    args = ap.parse_args(argv)
    if args.repeats < 1:
        ap.error('--repeats must be >= 1')
    with open(args.cases_json, 'r', encoding='utf-8') as f:
        cases = json.load(f)
    if not isinstance(cases, list) or not cases:
        ap.error('--cases-json must contain a non-empty JSON list')
    validated = []
    for idx, raw in enumerate(cases):
        try:
            validated.append((validate_case(raw), None))
        except ValueError as exc:
            validated.append((raw, str(exc)))
    results = []
    for idx, (case, err) in enumerate(validated):
        if err is not None:
            name = case.get('name', 'case_%d' % idx) if isinstance(case, dict) else 'case_%d' % idx
            results.append({'name': name, 'status': 'invalid', 'error': err})
            continue
        results.append(run_case(case, args.backend, args.repeats))
    statuses = [r['status'] for r in results]
    report = {'metadata': environment_info(args.backend != 'cpu'), 'backend': args.backend,
              'repeats': args.repeats, 'scope_limitations': SCOPE_LIMITATIONS,
              'tolerances': {'demand_total_rtol': DEMAND_TOTAL_RTOL, 'parity_rtol': PARITY_RTOL,
                             'parity_atol': PARITY_ATOL, 'weight_rtol': WEIGHT_RTOL},
              'summary': {s: statuses.count(s) for s in sorted(set(statuses))}, 'results': results}
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=_json_default)
    if 'invalid' in statuses:
        return 2
    if 'error' in statuses or 'failed' in statuses:
        return 1
    if 'partial' in statuses or 'unavailable' in statuses:
        return 3  # requested backend unavailable; CPU records (if any) are retained in the output
    return 0


if __name__ == '__main__':
    sys.exit(main())
