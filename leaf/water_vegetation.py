'''leaf.water_vegetation: bounded projection of the accepted LEAF canopy onto per-cell hydrology / rain-impact vegetation inputs (Phase 2.5 P1).

LEAF computes DERIVED vegetation geometry only: canopy cover fractions, a top-layer root index and per-species preferential-infiltration contribution inputs. It owns no
water: the Tietjen matrix factor g = (a1 + R1) / (a2 + R1), the preferential rate / receipt and all physical water stay in SYRUP (never applied here, never doubled).
Pure stdlib + NumPy (+ optional Numba / CuPy for the explicit compiled / CUDA reductions).

APPROVED RULES (user decisions 2026-11-14; nothing here is a new physics choice)
  * Eligibility / visibility. Splash cover: VISIBLE live + dead_standing canopies. Evaporation shading and infiltration inputs: VISIBLE LIVE only. `removed` never counts. A
    fully buried plant (burial >= intact height) has no visible cover (it contributes to no output); a partially buried plant keeps the SAME circle; an elevated-base canopy
    stays projected at its X, Y. Optical `porosity` is unused; no canopy-density factor. Roots are the fixed species `roots.layer_fractions`.
  * Aggregation (explicit MAHLERAN-style add-and-cap; NOT a geometric or random-overlap union): per species and cell the eligible plants' exact overlap areas ADD and are
    divided by the cell area, f_s,c = min(sum_p area_p,c / A_c, 1); total cover = min(sum_s f_s,c, 1). Caps are recorded (`diagnostics`).
  * Matrix top-root index R1_c = sum_s f_live_s,c * rootfrac_s[0] (uncapped, ALL participating live species whatever their receiving weights).
  * Preferential contribution of a species with an ENABLED (positive-sum) receiving vector: c_s,c = f_live_s,c * sum(rootfrac_s[1:]) (0 for one layer). An all-zero vector
    disables only that contribution. Weights are the RAW schema-2 trait (SYRUP normalises). A participating ACTIVE live species without the trait REFUSES.

GEOMETRY / MASKS. `canopy.derive_geometry` (the single allometry behind `canopy_view`) gives radius, status and burial; the actual landscape positions feed the existing exact
circle-cell builder `plant_pet.build_canopy_map` (nonzero origin, periodic seams, wide circles via its torus union), reused unchanged. Cell index = row * nx + col;
`active_mask` (ny, nx) bool, True = active. ACTIVE-FIRST: before any footprint is built, the existing `plant_pet` count kernels (compiled / NumPy / periodic) count each plant's
ACTIVE-cell overlap; a plant with none is omitted from THIS projection only (ids and counts recorded; the plant is untouched) and the public builder is then called with the
mask and only the surviving rows, so inactive footprints are never built and `max_edges` bounds the ACTIVE edges. Zero active work: an empty / all-inactive case builds no edge.

STORAGE (bounded; no P x N, no persisted S x N tensor). Sorted cell-major sparse entries (cell, species) with summed eligible areas (`A_all` live + dead, `A_live` live) in stable
(species id, plant id) order inside every entry; O(N) fields; O(S L) tables. NumPy, Numba and the CUDA one-thread-per-cell kernel add species in ascending order per cell with the
same arithmetic (division by the cell area, min(f, 1), no fused multiply-add) and are bit identical.

MEMORY. Every stage is admitted from METADATA (N, P, S, L, E, M) BEFORE its owned allocation, with the caller's reserve counted once per pool and an optional host reader:
validate (chunked finite check / hashes, cold package digest) -> snapshot (session) -> select -> count -> build -> aggregate -> reduce -> seal. Streamed contributions take an explicit
quota and admit each fresh (N,) array before allocating it (one species at a time, nothing retained). CUDA: explicit device, the CURRENT device must equal it (no switching), separate
device quota / reader / reserve admitted before any device allocation; only the cold sealed entry arrays go H2D once and the three O(N) results come back in one counted D2H; the
canonical outputs are HOST arrays (documented interface, not resident GPU maps). The CUDA path is AUTHORED, NOT RUN by the author.

IMMUTABILITY / IDENTITY. Outputs are read-only arrays whose base chain ends in an immutable `bytes` object (setflags(write=True) is impossible anywhere in the chain); metadata is
deeply frozen; construction is private (a thread-local minting window) with a registry fingerprint (type, data pointers, shapes, dtypes, bases, frozen metadata identities) checked
on every access in O(arrays) without rehashing; subclasses and uninitialised records are refused. Identity records the accepted session token / sequence / pop / geometry hashes (or
the labelled initial-state hash), catalog, domain, surface and mask hashes, the FULL LEAF package source-tree digest (SYRUP `source_tree_digest` algorithm: sorted relative path NUL
file-sha256 LF, `__pycache__` / .pyc / .pyo excluded, chunked, bounded, cold once per process; file count and algorithm recorded) and a content hash of the outputs. The state / catalog /
session inputs are never mutated; exact genuine types are required (subclasses refused).
'''
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import stat
import threading
import types
import weakref
from contextlib import contextmanager

import numpy as np

from leaf import canopy as _canopy
from leaf import periodic as _periodic
from leaf import plant_pet as _pet
from leaf import state as _state
from leaf import traits as _traits

SCHEMA = 'leaf.water_vegetation_projection'
VERSION = 3                                    # v3: the content hash covers EVERY derived table (root / lower / receiving weights / participation / omitted ids / species / scalars)
RULE = ('species cover f=min(sum eligible plant overlap area/cell area,1); total=min(sum_species f,1); matrix R1=sum_s f_live_s*rootfrac_s[0]; '
        'preferential c_s=f_live_s*sum(rootfrac_s[1:]) for enabled receiving weights; splash=visible live+dead, shading/infiltration=visible live')
BACKENDS = ('numpy', 'numba', 'cuda')
ELIGIBILITY = ('live', 'all')
CUDA_THREADS = 128
HASH_CHUNK_BYTES = 1 << 20
SOURCE_DIGEST_ALGORITHM = 'sha256 over sorted "relative/path\\0<file sha256 hex>\\n"; __pycache__ and .pyc/.pyo excluded; regular files only'
SOURCE_MAX_FILES = 4096
SOURCE_MAX_BYTES = 1 << 26
SNAPSHOT_RESIDENT_PER_PLANT = 64               # PlantState rows of a session snapshot (measured 63.5 B / row)
SNAPSHOT_PEAK_PER_PLANT = 168                  # transient peak of `PlantSession.snapshot_state()` (measured 160 B / row)
GEOMETRY_BYTES_PER_PLANT = 176                 # `canopy.derive_geometry`: 21 arrays (160 B nbytes / row; measured 143 resident, 158 peak) + power temporaries
SELECT_ARRAYS_PER_PLANT = 41                   # ids 8, species 8, x 8, y 8, radius 8, live 1 (the borrowed-by-the-builder selection arrays)
SELECT_TEMP_PER_PLANT = 40                     # flatnonzero rows 8, lexsort keys / order 16, gathers 16 (+ 4 bool rows per STATE row, counted separately)
BOOL_ROWS_PER_STATE_ROW = 5                    # live, dead, visible, eligible, ~fully_buried masks
AGGREGATE_BYTES_PER_EDGE = 48                  # sort key / argsort / reorder / group-id workspace per active edge (measured 46.4 B per edge incl. the entries; traced controls)
AGGREGATE_BYTES_PER_ENTRY = 40                 # first-index + the four entry arrays
OVERHEAD_SLACK_BYTES = 1 << 16                 # documented small Python / dict / metadata overhead the stage arithmetic does not itemise (never used for O(N/P/E) terms)
_MINT = threading.local()
_REGISTRY = weakref.WeakKeyDictionary()


class VegetationProjectionError(ValueError):
    '''Any projection validation error (nothing is returned or published).'''


class VegetationBudgetError(VegetationProjectionError):
    '''A declared edge / byte bound (or the caller's memory reader) would be exceeded; raised BEFORE the stage allocates.'''

    def __init__(self, stage, reason):
        super().__init__(f'{stage}: {reason}; nothing allocated')
        self.stage, self.reason = stage, reason


class VegetationBackendError(VegetationProjectionError):
    '''The requested backend (numba / CuPy / a CUDA device) cannot be used; there is no silent fallback.'''


class StaleProjectionError(VegetationProjectionError):
    '''The projection does not belong to the given accepted session / surface / mask (stale or mismatched generation).'''


# ------------------------------------------------------------------ admission (one pool = one quota; the reserve is counted once)
class _Quota:
    __slots__ = ('max_bytes', 'reserve', 'reader', 'pool')

    def __init__(self, max_bytes, reserve, reader, pool='host'):
        self.max_bytes, self.reserve, self.reader, self.pool = max_bytes, reserve, reader, pool

    def admit(self, stage, need):
        if self.reserve + need > self.max_bytes:
            raise VegetationBudgetError(stage, f'{self.pool}: reserve {self.reserve} + {need} bytes exceed max_bytes={self.max_bytes}')
        if self.reader is not None:
            avail = int(self.reader())
            if avail < self.reserve + need:
                raise VegetationBudgetError(stage, f'{self.pool}: reader reports {avail} bytes available, {self.reserve + need} needed')


def _quota(max_bytes, reserve_bytes, reader, pool='host'):
    mb, rb = _strict_cap(max_bytes, 'max_bytes'), _strict_cap(reserve_bytes, 'reserve_bytes')
    if reader is not None and not callable(reader):
        raise VegetationProjectionError('available_bytes_fn must be callable or None')
    return _Quota(mb, rb, reader, pool)


def _strict_cap(v, name):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or int(v) < 0:
        raise VegetationProjectionError(f'{name} must be an integer >= 0, got {v!r}')
    return int(v)


def estimate_projection_bytes(n_plants_all, n_plants, n_edges, n_cells, n_species, n_layers, *, n_entries=None, builder='numpy', from_session=False, source_cold=False,
                              bbox_scratch_bytes=0, n_omitted=0, n_eligible=None, backend='numpy'):
    '''LITERAL host byte model of one projection: the arrays that are simultaneously live in each stage (the max stage is `peak_bytes`, never a padded sum) and the RESIDENT result.
    A = state rows; Pe = eligible rows (`n_eligible`, default `n_plants`); P = eligible ACTIVE rows (`n_plants`); E = ACTIVE circle-cell edges; M = (cell, species) entries <= E;
    Om = omitted ids. Every array an owned stage allocates is listed (original / filtered / concatenated / cast / immutable copies); references are released as soon as they are unused
    (a session snapshot is dropped after the selection; raw outputs are released one by one while they are sealed). Excluded: interpreter, Numba / NVRTC caches, CUDA context and a
    documented `OVERHEAD_SLACK_BYTES` for Python / dict / metadata objects (never used for an O(N), O(P), O(E) or O(S L) term).
    Stages: validate; snapshot; select; count_pre; count; filter; build; aggregate; reduce; seal. `backend` MUST match the projection: the CUDA host result is one shared (3, N)
    backing whose lifetime adds exactly 16 N bytes at the seal peak (the NumPy / Numba model is unchanged).'''
    A, P, E, N, S, L, Om = (_traits.strict_int(v, k, 0) for v, k in ((n_plants_all, 'n_plants_all'), (n_plants, 'n_plants'), (n_edges, 'n_edges'), (n_cells, 'n_cells'),
                                                                     (n_species, 'n_species'), (n_layers, 'n_layers'), (n_omitted, 'n_omitted')))
    Pe = P if n_eligible is None else _traits.strict_int(n_eligible, 'n_eligible', 0)
    Pe = max(Pe, P)
    M = E if n_entries is None else _traits.strict_int(n_entries, 'n_entries', 0)
    tables = 2 * 8 * S * L + 16 * S + S                                  # root, weights (+ lower, has_w, species flags)
    sealed_tables = tables + 8 * Om
    entry = 8 + 4 + 8 + 8                                                # entry_cell int64, entry_species int32, A_all, A_live
    cold = (HASH_CHUNK_BYTES + 4096 * 64) if source_cold else 0
    snap_res = SNAPSHOT_RESIDENT_PER_PLANT * A if from_session else 0
    snap_peak = SNAPSHOT_PEAK_PER_PLANT * A if from_session else 0
    sel_e = SELECT_ARRAYS_PER_PLANT * Pe
    sel_p = SELECT_ARRAYS_PER_PLANT * P
    builder_peak = _pet.estimate_canopy_map_bytes(P, E, 'numpy', builder=builder, bbox_scratch_bytes=bbox_scratch_bytes)
    cr = min(N, M)                                                       # cells holding an entry at one rank pass
    shared_backing = 16 * N if backend == 'cuda' else 0                    # sealed rows alive while the shared (3, N) host base cannot yet be freed
    out_arrays = 3 * 8 * N                                               # splash, shade, r1 (the raw outputs of the reduction)
    off = 8 * (N + 1)
    resident = out_arrays + off + M * entry + 16 * S * L + 9 * S + 8 * Om        # exactly the sealed arrays: 3 N + (N + 1) doubles, entries, root / weights / lower / has_w, omitted ids
    if backend == 'cuda':
        reduce_extra = 0                                                 # the (3, N) host result below replaces the three separate raw outputs
    else:
        reduce_extra = 8 * N                                             # np.diff(off) counts during the rank loop (the bincount temporary is released first)
    stages = {
        'validate': HASH_CHUNK_BYTES + cold,
        'snapshot': snap_peak,
        'select': snap_res + GEOMETRY_BYTES_PER_PLANT * A + BOOL_ROWS_PER_STATE_ROW * A + sel_e + SELECT_TEMP_PER_PLANT * Pe,
        'count_pre': sel_e + 32 * Pe,                                    # x - origin, y - origin for the NumPy bounding scratch
        'count': sel_e + 16 * Pe + 8 * Pe + 2 * Pe + bbox_scratch_bytes,  # local coordinates + counts + periodic masks
        'filter': sel_e + sel_p + Pe + 8 * Pe + 8 * P,                   # keep mask, counts, filtered copies next to the originals
        'build': sel_p + tables + int(builder_peak['build_total_bytes']),
        'aggregate': 9 * P + tables + int(builder_peak['resident_host_bytes']) + AGGREGATE_BYTES_PER_EDGE * E + AGGREGATE_BYTES_PER_ENTRY * M,
        # bincount temporary (released before the outputs exist) | outputs + np.diff counts + the per-rank bool mask `counts > r` (N) and gathers (rows, k, fa, fl + 3 temporaries ~ 72 B per cell of that rank) | caps / diagnostics
        'reduce': M * entry + off + max(8 * N, out_arrays + reduce_extra + (N if M else 0) + 72 * cr, out_arrays + 18 * M + 2 * N) + tables + 8 * Om,
        # every raw array is replaced by its immutable copy one at a time: raw + sealed stays the raw total, plus ONE temporary copy of the largest array. CUDA: the three O(N) results
        # are views of ONE contiguous (3, N) host array (a single D2H), so its 24 N backing is released only when the LAST view has been sealed: while the third row is copied the base
        # (24 N) AND the two already sealed rows (16 N) coexist, i.e. exactly 16 N more than the independent-array CPU path (`shared_backing`)
        'seal': (out_arrays + off + M * entry + 8 * Om + tables) + max(8 * N, 8 * M, 8 * (N + 1)) + shared_backing,
    }
    stages = {k: int(v) + (OVERHEAD_SLACK_BYTES if (k != 'validate' and v > 0) else 0) for k, v in stages.items()}
    return {'stages': stages, 'peak_bytes': int(max(stages.values())), 'resident_bytes': int(resident), 'cold_source_digest_bytes': int(cold),
            'scope': 'literal host arithmetic of the live arrays per stage (see the docstring); excludes interpreter / JIT / CUDA context; CUDA adds a separate device pool (estimate_device_bytes)'}


def estimate_device_bytes(n_cells, n_entries, n_species):
    '''Declared DEVICE pool of the CUDA reduction and its transfers: entries (offsets, species, A_all, A_live) + root0 + three O(N) outputs; cold H2D = the entry arrays + root0,
    one D2H = the three O(N) outputs. The canonical outputs are HOST arrays afterwards (no resident GPU maps).'''
    N, M, S = (_traits.strict_int(v, k, 0) for v, k in ((n_cells, 'n_cells'), (n_entries, 'n_entries'), (n_species, 'n_species')))
    h2d = 8 * (N + 1) + M * (4 + 8 + 8) + 8 * S
    d2h = 3 * 8 * N
    return {'device_bytes': int(h2d + d2h), 'h2d_bytes': int(h2d), 'd2h_bytes': int(d2h), 'scope': 'declared; excludes CUDA context and pool rounding'}


# ------------------------------------------------------------------ source package digest (full package, cold once)
_SOURCE_CACHE = {}


def package_source_record():
    '''SYRUP `source_tree_digest` algorithm over the WHOLE LEAF package directory (every regular non-bytecode file, symlinks refused, bounded, chunked); cold once per process.'''
    if 'rec' not in _SOURCE_CACHE:
        root = pathlib.Path(__file__).resolve().parent
        entries, total = [], 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d != '__pycache__')
            for name in sorted(filenames):
                path = pathlib.Path(dirpath) / name
                if path.suffix in ('.pyc', '.pyo'):
                    continue
                info = os.lstat(path)
                if not stat.S_ISREG(info.st_mode):
                    raise VegetationProjectionError(f'{path} is not a regular file (symlinks / special files are refused in the source digest scope)')
                if len(entries) >= SOURCE_MAX_FILES or total + info.st_size > SOURCE_MAX_BYTES:
                    raise VegetationProjectionError('the LEAF source scope exceeds the declared digest bounds')
                h = hashlib.sha256()
                with open(path, 'rb') as fh:
                    while True:
                        chunk = fh.read(HASH_CHUNK_BYTES)
                        if not chunk:
                            break
                        h.update(chunk)
                total += info.st_size
                entries.append((path.relative_to(root).as_posix(), h.hexdigest()))
        entries.sort()
        top = hashlib.sha256()
        for rel, fh_ in entries:
            top.update(rel.encode('utf-8') + b'\0' + fh_.encode('ascii') + b'\n')
        _SOURCE_CACHE['rec'] = {'digest': 'sha256:' + top.hexdigest(), 'algorithm': SOURCE_DIGEST_ALGORITHM, 'file_count': len(entries), 'total_bytes': int(total)}
    return dict(_SOURCE_CACHE['rec'])


_EXTERNAL_MODULES = ('benchmarks.spatial_mapping',)


def external_source_record():
    '''The EXTERNAL helper modules outside the `leaf` package that the reused compiled `PlantSession` / canopy / root builders import (`benchmarks/spatial_mapping.py`):
    located without importing them (`importlib.util.find_spec`), each file hashed once per process (bounded, chunked) and recorded separately from the 28-file LEAF package digest. A
    missing helper refuses. The LEAF package hash is NOT extended to cover them (no pretending); the producer metadata simply records their exact bytes so a changed helper is observed.'''
    if 'ext' not in _SOURCE_CACHE:
        import importlib.util
        recs = []
        for name in _EXTERNAL_MODULES:
            try:
                spec = importlib.util.find_spec(name)
            except (ImportError, ValueError) as exc:
                raise VegetationProjectionError(f'required helper module {name!r} cannot be located: {exc}') from exc
            if spec is None or not spec.origin or not os.path.isfile(spec.origin):
                raise VegetationProjectionError(f'required helper module {name!r} cannot be located (a compiled builder needs it)')
            info = os.lstat(spec.origin)
            if not stat.S_ISREG(info.st_mode) or info.st_size > SOURCE_MAX_BYTES:
                raise VegetationProjectionError(f'helper {name!r} is not a bounded regular file')
            h = hashlib.sha256()
            with open(spec.origin, 'rb') as fh:
                while True:
                    chunk = fh.read(HASH_CHUNK_BYTES)
                    if not chunk:
                        break
                    h.update(chunk)
            recs.append({'module': name, 'file': os.path.basename(spec.origin), 'sha256': 'sha256:' + h.hexdigest(), 'bytes': int(info.st_size)})
        _SOURCE_CACHE['ext'] = {'helpers': recs, 'scope': 'external to the 28-file LEAF package digest; observed once per process'}
    return {'helpers': [dict(r) for r in _SOURCE_CACHE['ext']['helpers']], 'scope': _SOURCE_CACHE['ext']['scope']}


def package_sha256():
    '''The FULL LEAF package digest string ("sha256:<hex>"), cold once per process.'''
    return package_source_record()['digest']


# ------------------------------------------------------------------ chunked hashing / checks (no whole-array copies)
def _hash_array(a, h=None):
    h = hashlib.sha256() if h is None else h
    a = a if a.flags.c_contiguous else np.ascontiguousarray(a)
    mv = memoryview(a).cast('B') if a.size else memoryview(b'')
    for lo in range(0, len(mv), HASH_CHUNK_BYTES):
        h.update(mv[lo:lo + HASH_CHUNK_BYTES])
    return h


def _ones_sha(n):
    h = hashlib.sha256()
    chunk = b'\x01' * min(HASH_CHUNK_BYTES, max(n, 1))
    for lo in range(0, n, HASH_CHUNK_BYTES):
        h.update(chunk[:min(HASH_CHUNK_BYTES, n - lo)])
    return h.hexdigest()


def _all_finite(a, chunk=1 << 16):
    flat = a.reshape(-1)
    for lo in range(0, flat.shape[0], chunk):
        if not np.isfinite(flat[lo:lo + chunk]).all():
            return False
    return True


def _state_sha256(st):
    h = hashlib.sha256()
    for name in st.ARRAY_NAMES:
        a = getattr(st, name)
        h.update(name.encode() + b'\0' + str(a.dtype).encode() + str(a.shape).encode())
        _hash_array(a, h)
    h.update(st.catalog_hash.encode() + json.dumps(st.domain.to_dict(), sort_keys=True).encode())
    return 'sha256:' + h.hexdigest()


def _prep_surface_mask(domain, surface, active_mask):
    '''METADATA-only validation (no allocation, no coercion).'''
    if type(surface) is not np.ndarray or surface.dtype != np.float64 or surface.shape != domain.shape or not surface.flags.c_contiguous:
        raise VegetationProjectionError(f'current_surface must be a C-contiguous float64 numpy array of shape {domain.shape}')
    if active_mask is not None and (type(active_mask) is not np.ndarray or active_mask.dtype != np.bool_ or active_mask.shape != domain.shape or not active_mask.flags.c_contiguous):
        raise VegetationProjectionError(f'active_mask must be a C-contiguous bool numpy array of shape {domain.shape} (True = active) or None')
    return surface, (None if active_mask is None else active_mask.reshape(-1))


# ------------------------------------------------------------------ reductions (one arithmetic, three implementations)
_NUMBA = {}


def _numba_kernels():
    if 'k' in _NUMBA:
        return _NUMBA['k']
    try:
        import numba
    except Exception as exc:       # noqa: BLE001
        raise VegetationBackendError(f'backend "numba" is unavailable: {exc}') from exc

    @numba.njit(cache=False)
    def agg(gid, area, live, a_all, a_live):
        for i in range(gid.shape[0]):
            g = gid[i]
            a_all[g] += area[i]
            if live[i]:
                a_live[g] += area[i]

    @numba.njit(cache=False)
    def reduce(off, sp, a_all, a_live, root0, cell_area, splash_raw, shade_raw, r1):
        for c in range(off.shape[0] - 1):
            ta = 0.0
            tl = 0.0
            rr = 0.0
            for k in range(off[c], off[c + 1]):
                fa = a_all[k] / cell_area
                if fa > 1.0:
                    fa = 1.0
                fl = a_live[k] / cell_area
                if fl > 1.0:
                    fl = 1.0
                ta = ta + fa
                tl = tl + fl
                rr = rr + fl * root0[sp[k]]
            splash_raw[c] = ta
            shade_raw[c] = tl
            r1[c] = rr

    _NUMBA['k'] = (agg, reduce)
    return _NUMBA['k']


def _aggregate_numpy(gid, area, live, M):
    a_all, a_live = np.zeros(M), np.zeros(M)
    start = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1]]) if gid.shape[0] else np.zeros(0, np.int64)
    rank = np.arange(gid.shape[0]) - start[gid] if gid.shape[0] else np.zeros(0, np.int64)
    for r in range(int(rank.max()) + 1 if rank.shape[0] else 0):
        i = np.flatnonzero(rank == r)
        a_all[gid[i]] += area[i]
        j = i[live[i]]
        a_live[gid[j]] += area[j]
    return a_all, a_live


def _reduce_numpy(off, sp, a_all, a_live, root0, cell_area, N):
    splash, shade, r1 = np.zeros(N), np.zeros(N), np.zeros(N)
    counts = np.diff(off)
    for r in range(int(counts.max()) if N and counts.shape[0] else 0):
        rows = np.flatnonzero(counts > r)
        k = off[rows] + r
        fa = np.minimum(a_all[k] / cell_area, 1.0)
        fl = np.minimum(a_live[k] / cell_area, 1.0)
        splash[rows] = splash[rows] + fa
        shade[rows] = shade[rows] + fl
        r1[rows] = r1[rows] + fl * root0[sp[k]]
    return splash, shade, r1


_CUDA_SRC = r"""
extern "C" __global__ void veg_reduce(const long long* off, const int* sp, const double* a_all, const double* a_live, const double* root0,
                                      const double cell_area, const long long n_cells, double* out) {      // out = [splash_raw (N) | shade_raw (N) | r1 (N)]
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;      // ONE THREAD PER CELL; entries visited in ascending species order
    if (c >= n_cells) return;
    double ta = 0.0, tl = 0.0, rr = 0.0;
    for (long long k = off[c]; k < off[c + 1]; ++k) {
        double fa = __ddiv_rn(a_all[k], cell_area);
        if (fa > 1.0) fa = 1.0;
        double fl = __ddiv_rn(a_live[k], cell_area);
        if (fl > 1.0) fl = 1.0;
        ta = __dadd_rn(ta, fa);
        tl = __dadd_rn(tl, fl);
        rr = __dadd_rn(rr, __dmul_rn(fl, root0[sp[k]]));
    }
    out[c] = ta; out[n_cells + c] = tl; out[2 * n_cells + c] = rr;
}
"""


def _reduce_cuda(device, dq, off, sp, a_all, a_live, root0, cell_area, N):
    '''Explicit CUDA reduction (AUTHORED, not run by the author). Order: CuPy import -> CURRENT-device guard (no switching) -> device admission -> cold H2D of the sealed entry arrays
    -> one kernel -> ONE counted D2H of the three O(N) results (host arrays). No fallback, no per-plant or per-step transfer.'''
    try:
        import cupy as cp
    except Exception as exc:       # noqa: BLE001
        raise VegetationBackendError(f'backend "cuda" is unavailable: {exc}') from exc
    dev = _traits.strict_int(device, 'device', 0)
    cur = int(cp.cuda.runtime.getDevice())
    if cur != dev:
        raise VegetationBackendError(f'the CURRENT CUDA device is {cur}, not the requested device {dev}; select it explicitly (no implicit switching)')
    dq.admit('cuda_reduce', estimate_device_bytes(N, int(sp.shape[0]), int(root0.shape[0]))['device_bytes'])
    d_off, d_sp = cp.asarray(off), cp.asarray(sp)
    d_aa, d_al, d_r0 = cp.asarray(a_all), cp.asarray(a_live), cp.asarray(root0)
    d_out = cp.zeros(3 * N, dtype=cp.float64)                                 # ONE contiguous device result: rows [splash_raw | shade_raw | r1]
    kern = cp.RawKernel(_CUDA_SRC, 'veg_reduce')
    kern(((N + CUDA_THREADS - 1) // CUDA_THREADS,), (CUDA_THREADS,), (d_off, d_sp, d_aa, d_al, d_r0, np.float64(cell_area), np.int64(N), d_out))
    host = d_out.get()                                                        # the ONE device-to-host call: one fresh host allocation, no second copy
    cp.cuda.Stream.null.synchronize()
    return host.reshape(3, N)


STREAM_FIXED_BYTES = 8192                         # the O(1) Python objects of one stream (the estimate dict, ndarray / tuple / list headers, closures, the generator frame, the weakref): measured 7.6 KB peak / 5 KB retained at n <= 2 with tracemalloc; never an O(S) term


def estimate_stream_bytes(n_cells, n_entries, n_enabled, n_layers, *, previous_held=False):
    '''PUBLIC memory model of the preferential contribution stream: the ONE model both `VegetationProjection.preferential_inputs` and `preferential_inputs_from_compact` admit (no
    private surcharge, no mirroring). With n = enabled species, L layers, N cells, M entries:
    * `weights_bytes` 8 n L: the raw receiving weights (n, L) RETURNED to the caller;
    * `index_bytes` 8 n: the typed intp enabled-species index array kept for the iterator lifetime (no Python-int list is ever built);
    * `ids_bytes` 8 n: the returned id tuple (pointers; the species-id strings are borrowed, never copied);
    * `weights_stage_bytes` (the admission BEFORE anything is built) = weights + index + the transient 8 n pointer list used to build the tuple (16 n peak with the tuple) + `STREAM_FIXED_BYTES`;
    * `retained_metadata_bytes` = index + ids + `STREAM_FIXED_BYTES`: what stays live for the iterator / caller after the weights stage;
    * `result_bytes` 8 N + `scratch_bytes` (the species mask M and four k-sized float arrays, k <= M: M + 32 M) = `species_cover_bytes`;
    * `per_next_bytes` = result + scratch + the PREVIOUS yielded array 8 N when the caller still holds it + weights + retained metadata.
    The caller's reserve is counted ONCE by the quota (never inside these numbers). Only the most recent yielded array is tracked; a consumer that retains several arrays must reserve them
    itself (the producer stores no history). `weights_bytes` is the PAYLOAD (what the caller keeps); `weights_stage_bytes` is what the call itself needs at its peak.'''
    N, M, n, L = (_traits.strict_int(v, k, 0) for v, k in ((n_cells, 'n_cells'), (n_entries, 'n_entries'), (n_enabled, 'n_enabled'), (n_layers, 'n_layers')))
    weights, result, scratch = 8 * n * L, 8 * N, M + 32 * M
    index, ids = 8 * n, 8 * n
    retained = index + ids + STREAM_FIXED_BYTES
    prev = result if previous_held else 0
    return {'weights_bytes': weights, 'index_bytes': index, 'ids_bytes': ids, 'weights_stage_bytes': weights + index + 16 * n + STREAM_FIXED_BYTES, 'retained_metadata_bytes': retained,
            'result_bytes': result, 'scratch_bytes': scratch, 'previous_held_bytes': prev, 'species_cover_bytes': result + scratch,
            'per_next_bytes': result + scratch + prev + weights + retained,
            'scope': 'live arrays of one next(); caller reserve counted once by the quota; only the most recent yielded array is tracked'}


def _content_header(species_ids, shape, cell_area, n_cells, n_entries, n_plants, root_layers):
    '''The canonical header of the V3 content hash as an iterator of byte pieces whose concatenation is EXACTLY `json.dumps({...}, sort_keys=True).encode()` (the producer and
    `compact_content_sha256` use this ONE function). The species ids are streamed one JSON string at a time (default `", "` separator, `ensure_ascii`), so no list / text / bytes copy of
    all the identifiers ever exists: the transient is a few copies of the longest id (`estimate_compact_validation_bytes(max_id_chars=...)`).'''
    text = json.dumps({'schema': SCHEMA, 'version': VERSION, 'species_ids': [], 'shape': [int(v) for v in shape], 'cell_area_m2': float(cell_area).hex(), 'n_cells': int(n_cells),
                       'n_entries': int(n_entries), 'n_plants': int(n_plants), 'root_layers': int(root_layers)}, sort_keys=True)
    head, mark, tail = text.partition('"species_ids": []')
    yield (head + '"species_ids": [').encode()
    for i, sid in enumerate(species_ids):
        yield ((', ' if i else '') + json.dumps(sid)).encode()
    yield (']' + tail).encode()
    del mark


def _content_digest(header, arrays, tables):
    '''"sha256:<hex>" over the header pieces, then `_ARRAYS`, then `_TABLES` (name, dtype, shape, chunked array digest). `arrays` / `tables` map name -> C-contiguous ndarray.'''
    content = hashlib.sha256()
    for piece in header:
        content.update(piece)
    for k in _ARRAYS:
        content.update(k.encode() + str(arrays[k].dtype).encode() + str(arrays[k].shape).encode() + _hash_array(arrays[k]).digest())
    for k in _TABLES:
        content.update(k.encode() + str(tables[k].dtype).encode() + str(tables[k].shape).encode() + _hash_array(tables[k]).digest())
    return 'sha256:' + content.hexdigest()


def _species_cover(entry_species, area, entry_cell, cell_area, n_cells, s, scale=None):
    '''ONE arithmetic of the per-species cover f_s = min(area / cell area, 1) (optionally * scale) on the entries of species `s`; a FRESH (N,) array (shared by the projection
    methods and the compact helpers; same float operation order).'''
    sel = np.flatnonzero(entry_species == s)
    out = np.zeros(n_cells)
    vals = area[sel]
    vals /= cell_area
    np.minimum(vals, 1.0, out=vals)
    if scale is not None:
        vals *= scale                                     # same arithmetic as cover * scale, in place on the k-sized values
    out[entry_cell[sel]] = vals
    return out


def _count_enabled(has_w, weights):
    '''Number of enabled species WITHOUT allocating anything (a pure counting loop).'''
    return sum(1 for i in range(weights.shape[0]) if has_w[i] and float(weights[i].max()) > 0.0)


def _enabled_index(has_w, weights, n):
    '''The enabled species as ONE typed intp array of the already counted size n, filled by a cold loop (no Python-int list).'''
    idx = np.empty(n, dtype=np.intp)
    k = 0
    for i in range(weights.shape[0]):
        if has_w[i] and float(weights[i].max()) > 0.0:
            idx[k] = i
            k += 1
    return idx


def _preferential_stream(q, n_cells, n_entries, species_ids, weights_tbl, lower, has_w, cover):
    '''The ONE preferential stream (projection and compact helpers): `cover(i, 'live', scale)` returns the FRESH (N,) array of species index i. The enabled species are COUNTED first (no
    allocation) and `estimate_stream_bytes(...)['weights_stage_bytes']` is admitted BEFORE the typed index array, the weights copy or the id tuple exist; each `next()` admits `per_next_bytes`.'''
    import weakref as _wr
    L = weights_tbl.shape[1]
    n = _count_enabled(has_w, weights_tbl)
    q.admit('preferential_weights', estimate_stream_bytes(n_cells, n_entries, n, L)['weights_stage_bytes'])
    idx = _enabled_index(has_w, weights_tbl, n)
    w = weights_tbl[idx]                                      # one fresh (n, L) copy (typed advanced index: no implicit index copy)
    pieces = [None] * n
    for k in range(n):
        pieces[k] = species_ids[int(idx[k])]
    ids = tuple(pieces)
    del pieces
    prev = [None]

    def gen():
        for i in idx:
            i = int(i)
            held = prev[0] is not None and prev[0]() is not None
            q.admit('preferential_contribution', estimate_stream_bytes(n_cells, n_entries, n, L, previous_held=held)['per_next_bytes'])
            out = cover(i, 'live', float(lower[i]))
            prev[0] = _wr.ref(out)
            yield out
            del out

    return ids, w, gen()


# ------------------------------------------------------------------ immutable sealing
def _immutable(a):
    '''A read-only 1-D array whose base chain ends in an immutable bytes object (setflags(write=True) is impossible anywhere in the chain). One declared copy.'''
    a = np.ascontiguousarray(a)
    out = np.frombuffer(a.tobytes(), dtype=a.dtype)
    return out


def _freeze(obj):
    if isinstance(obj, dict):
        return types.MappingProxyType({k: _freeze(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(v) for v in obj)
    return obj


def _thaw(obj):
    if isinstance(obj, (dict, types.MappingProxyType)):
        return {k: _thaw(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [_thaw(v) for v in obj]
    return obj


@contextmanager
def _minting():
    _MINT.ok = True
    try:
        yield
    finally:
        _MINT.ok = False


_ARRAYS = ('splash_cover', 'shading_cover', 'matrix_root_index', 'cell_offsets', 'entry_cell', 'entry_species', 'entry_area_all_m2', 'entry_area_live_m2')
_TABLES = ('root', 'lower', 'weights', 'has_w', 'omitted_ids')


def _fingerprint(p):
    out = []
    for group in (p._a, p._t):
        for k in sorted(group):
            a = group[k]
            out.append((k, int(a.__array_interface__['data'][0]), tuple(a.shape), a.dtype.str, type(a.base).__name__, bool(a.flags.writeable)))
    # every scalar that affects arithmetic, shape, counts, the memory model or the identity export is fingerprinted by VALUE (a replaced scalar is refused, no hot rehash)
    out.append((id(p.identity), id(p.diagnostics), tuple(p.species_ids), type(p).__name__, float(p.cell_area_m2).hex(), tuple(int(v) for v in p.shape), int(p.n_cells), int(p.n_entries),
                int(p.n_plants), int(p.resident_bytes)))
    return tuple(out)


class VegetationProjection:
    '''Sealed immutable projection (build with `project_from_plant_session` / `project_from_initial_state`; a private minting window and a registry fingerprint are the
    producer proof). Fields are read-only views whose base chain ends in immutable bytes; metadata is deeply frozen; `preferential_inputs()` / `species_cell_cover()` return FRESH
    arrays owned by the caller under an explicit quota. Never rehashed on a read (the content hash is computed once at build).'''

    __slots__ = ('__weakref__', '_a', '_t', 'identity', 'diagnostics', 'species_ids', 'n_cells', 'shape', 'cell_area_m2', 'n_entries', 'n_plants', 'resident_bytes')

    def __init_subclass__(cls, **kw):
        raise TypeError('VegetationProjection cannot be subclassed (sealed class)')

    def __init__(self, **kw):
        if not getattr(_MINT, 'ok', False):
            raise VegetationProjectionError('VegetationProjection is produced only by project_from_plant_session / project_from_initial_state')
        for k, v in kw.items():
            object.__setattr__(self, k, v)
        _REGISTRY[self] = _fingerprint(self)

    def __setattr__(self, name, value):
        raise AttributeError('VegetationProjection is immutable')

    def __delattr__(self, name):
        raise AttributeError('VegetationProjection is immutable')

    def __repr__(self):
        return f'VegetationProjection(N={self.n_cells}, S={len(self.species_ids)}, entries={self.n_entries})'

    def _check(self):
        fp = _REGISTRY.get(self) if type(self) is VegetationProjection else None
        if fp is None:
            raise VegetationProjectionError('not a genuine produced VegetationProjection (forged, copied or uninitialised record)')
        try:
            now = _fingerprint(self)
        except (AttributeError, KeyError, TypeError) as exc:
            raise VegetationProjectionError('the projection record is incomplete or replaced') from exc
        if now != fp:
            raise VegetationProjectionError('the projection record was altered (array / metadata replaced after sealing)')

    def _view(self, name):
        self._check()
        return self._a[name].view()

    @property
    def splash_cover(self):
        '''(N,) total visible live + dead_standing cover in [0, 1] (rain-impact `vegetation_cover_fraction`).'''
        return self._view('splash_cover')

    @property
    def shading_cover(self):
        '''(N,) total visible LIVE cover in [0, 1] (evaporation `cover`).'''
        return self._view('shading_cover')

    @property
    def matrix_root_index(self):
        '''(N,) R1 = sum_s f_live_s * rootfrac_s[0] (uncapped; SYRUP forms g = (a1 + R1) / (a2 + R1)).'''
        return self._view('matrix_root_index')

    @property
    def omitted_ids(self):
        '''Plant ids omitted from THIS projection (no active overlap); the plants themselves are untouched.'''
        self._check()
        return self._t['omitted_ids'].view()

    def identity_record(self):
        '''A fresh plain (mutable, detached) deep copy of the identity.'''
        self._check()
        return _thaw(self.identity)

    # ---- per-species streamed inputs (explicit quota; fresh arrays; one species at a time)
    def stream_estimate(self, *, previous_held=False):
        '''The public stream memory model (`estimate_stream_bytes`) for THIS projection: the enabled species are only COUNTED (allocation-free loop; no index list is built).'''
        self._check()
        return estimate_stream_bytes(self.n_cells, self.n_entries, _count_enabled(self._t['has_w'], self._t['weights']), self._t['weights'].shape[1], previous_held=previous_held)

    def _cover(self, s, eligibility, scale=None):
        return _species_cover(self._a['entry_species'], self._a['entry_area_live_m2' if eligibility == 'live' else 'entry_area_all_m2'], self._a['entry_cell'], self.cell_area_m2,
                              self.n_cells, s, scale)

    def species_cell_cover(self, species_id, eligibility='live', *, max_bytes, reserve_bytes=0, available_bytes_fn=None):
        '''FRESH (N,) f_s = min(sum eligible areas / cell area, 1) of one species ("live" or "all" = live + dead_standing). The result and its scratch are admitted first.'''
        self._check()
        if eligibility not in ELIGIBILITY:
            raise VegetationProjectionError(f'eligibility must be one of {ELIGIBILITY}')
        if not isinstance(species_id, str) or species_id not in self.species_ids:
            raise VegetationProjectionError(f'unknown species {species_id!r}')
        _quota(max_bytes, reserve_bytes, available_bytes_fn).admit('species_cover', estimate_stream_bytes(self.n_cells, self.n_entries, 0, self._t['weights'].shape[1])['species_cover_bytes'])
        return self._cover(self.species_ids.index(species_id), eligibility)

    def preferential_inputs(self, *, max_bytes, reserve_bytes=0, available_bytes_fn=None):
        '''`(species_ids, receiver_weights (n, L) float64 RAW, contributions)` for the species with an ENABLED (positive max) receiving vector. `contributions` is a one-shot iterator
        yielding one FRESH (N,) array c_s = f_live_s * sum(rootfrac_s[1:]) per species in the same order, nothing retained (no S x N). CONTRACT: each `next()` admits, under THIS quota
        (the caller's reserve counted once, no implicit exemption), the simultaneously live arrays: the new (N,) result + its species scratch (`_scratch_need`) + the PREVIOUS
        yielded array when the caller still holds it (tracked by a weak reference) + the raw weights (n, L) this call returned. The weights are the species trait as specified
        (SYRUP normalises); an all-zero species is disabled and absent (its id is in `diagnostics`). COMPOSITION for a consumer (S0): the producer accounts ONLY its own arrays (the raw
        weights it returned, the new result + scratch and, when still held, the previous yielded array). Put every OTHER native buffer in the caller reserve, counted once. A consumer that
        already counts a held array in its own reserve must RELEASE it before `next()` (otherwise the producer counts it again); releasing is the way to avoid the previous-held term.'''
        self._check()
        return _preferential_stream(_quota(max_bytes, reserve_bytes, available_bytes_fn), self.n_cells, self.n_entries, self.species_ids, self._t['weights'], self._t['lower'],
                                    self._t['has_w'], self._cover)

    def export_compact(self):
        '''PUBLIC read-only compact export of everything a held record needs, with NO copy and no S x N / P x N array: immutable views of the canonical arrays and tables in the
        canonical order `field_order` (the order the content hash uses): the O(N) fields splash_cover / shading_cover / matrix_root_index, the cell-major entries (cell_offsets,
        entry_cell, entry_species, entry_area_all_m2, entry_area_live_m2), the species tables root (S, L), lower, weights (S, L) RAW, has_w (participation of a receiving-weight trait,
        uint8) and omitted_ids, plus the scalars and the producer identity. Validated by the producer fingerprint (metadata only, no rehash). This is an EXPORT: it creates no
        projection and is not a rehydration factory (a later restart record carries its own outer seal).'''
        self._check()
        arrays = types.MappingProxyType({k: (self._a[k] if k in self._a else self._t[k]).view() for k in _ARRAYS + _TABLES})
        return types.MappingProxyType({'schema': SCHEMA, 'version': VERSION, 'field_order': _ARRAYS + _TABLES, 'arrays': arrays, 'species_ids': tuple(self.species_ids),
                                       'n_cells': self.n_cells, 'shape': tuple(self.shape), 'cell_area_m2': self.cell_area_m2, 'n_entries': self.n_entries, 'n_plants': self.n_plants,
                                       'root_layers': int(self._t['weights'].shape[1]), 'content_sha256': self.identity['content_sha256'], 'identity': self.identity})

    # ---- identity / staleness
    def check_session(self, session, *, current_surface=None, active_mask=None):
        '''Raise `StaleProjectionError` unless this projection was built from THIS accepted session generation (token, sequence, pop and geometry hashes, session id, lineage, catalog)
        and, when given, THIS surface / active mask (shape / dtype / contiguity validated BEFORE the chunked memoryview hash; no coercion, no whole-N copy, no owned allocation). Returns a read-only dict saying what was checked; with the surface and mask omitted it is explicitly BIOLOGY-ONLY. A genuine `PlantSession` is required (exact type).'''
        from leaf import plant_session as _ps
        self._check()
        if type(session) is not _ps.PlantSession:
            raise VegetationProjectionError('session must be a genuine leaf.plant_session.PlantSession (exact type)')
        src = self.identity['source']
        if src['kind'] != 'plant_session':
            raise StaleProjectionError('this projection was built from an initial state, not from an accepted session generation')
        live = {'token': session.token, 'sequence': int(session.sequence), 'pop_sha256': session.pop_sha256, 'geometry_sha256': session.geometry_sha256,
                'session_id': session.session_id, 'lineage': session.lineage, 'catalog_hash': session.config.catalog.hash}
        bad = sorted(k for k, v in live.items() if src[k] != v)
        if bad:
            raise StaleProjectionError(f'the projection is stale / mismatched for the session: {bad}')
        checked = {'biology': True, 'surface': False, 'mask': False}
        if current_surface is not None:
            if (type(current_surface) is not np.ndarray or current_surface.dtype != np.float64 or tuple(current_surface.shape) != tuple(self.shape)
                    or not current_surface.flags.c_contiguous):
                raise VegetationProjectionError(f'current_surface must be a C-contiguous float64 numpy array of shape {tuple(self.shape)} (no coercion)')
            if _hash_array(current_surface).hexdigest() != self.identity['surface_sha256']:      # chunked memoryview hash: no copy, no owned allocation
                raise StaleProjectionError('the current surface differs from the projection surface')
            checked['surface'] = True
        if active_mask is not None:
            if (type(active_mask) is not np.ndarray or active_mask.dtype != np.bool_ or tuple(active_mask.shape) != tuple(self.shape) or not active_mask.flags.c_contiguous):
                raise VegetationProjectionError(f'active_mask must be a C-contiguous bool numpy array of shape {tuple(self.shape)} (no coercion)')
            if _hash_array(active_mask.reshape(-1).view(np.uint8)).hexdigest() != self.identity['active_mask_sha256']:
                raise StaleProjectionError('the active mask differs from the projection mask')
            checked['mask'] = True
        return types.MappingProxyType(dict(checked, scope='the session generation (biology) is always verified; the surface / mask only when passed. With both omitted this is a BIOLOGY-ONLY '
                                                          'check: it does NOT validate the current landscape (SYRUP binding always supplies the exact surface and mask)'))


# ------------------------------------------------------------------ active-first counting (reuses the existing plant_pet count kernels; no integrator is copied)
def _active_counts(domain, x, y, r, mask_u8, use_mask, builder):
    '''Per-plant count of ACTIVE-cell overlaps with the existing `plant_pet` count kernels (compiled / NumPy / periodic), on the same local coordinates the public builder uses.'''
    P = int(x.shape[0])
    xl, yl = x - domain.origin_x_m, y - domain.origin_y_m
    W, H = float(domain.nx) * float(domain.dx_m), float(domain.ny) * float(domain.dx_m)
    if domain.any_periodic:
        _periodic.canonicalize_local(domain, x, y, xl, yl)
        if not domain.periodic[0]:
            np.minimum(xl, W, out=xl)
        if not domain.periodic[1]:
            np.minimum(yl, H, out=yl)
    elif P and (float(xl.min()) < 0.0 or float(xl.max()) > W or float(yl.min()) < 0.0 or float(yl.max()) > H):
        raise VegetationProjectionError('plant positions must lie inside the closed domain')
    counts = np.zeros(P, dtype=np.int64)
    if P:
        core = _pet._core('numba' if builder == 'compiled' else 'python')
        mk = mask_u8 if use_mask else np.zeros(1, np.uint8)
        dx = float(domain.dx_m)
        if domain.any_periodic:
            pk = _periodic.kernels('numba' if builder == 'compiled' else 'python', core['area'])
            pk['count'](xl, yl, r, domain.nx, domain.ny, dx, bool(domain.periodic[0]), bool(domain.periodic[1]), mk, bool(use_mask), counts)
        elif builder == 'compiled':
            core['count'](xl, yl, r, domain.nx, domain.ny, dx, mk, bool(use_mask), counts)
        else:
            _pet._numpy_counts(xl, yl, r, domain, mask_u8, use_mask, counts)
    return counts


# ------------------------------------------------------------------ core build
def _seal_arrays(raw):
    '''Seal `raw` (name -> ndarray) one array at a time, RELEASING each raw array as soon as its immutable copy exists (the dict is emptied).'''
    out = {}
    for k in list(raw):
        out[k] = _immutable(raw.pop(k))
    return out


def _build(st, catalog, surface, active_mask, *, source_fn, session_dims, max_edges, quota, dq, backend, builder, device, require_receiver_traits):
    if type(catalog) is not _traits.SpeciesCatalog:
        raise VegetationProjectionError('catalog must be a genuine leaf.traits.SpeciesCatalog (exact type)')
    domain = session_dims['domain'] if st is None else st.domain
    if type(domain) is not _state.Domain:
        raise VegetationProjectionError('the plant domain must be a genuine leaf.state.Domain')
    if not domain.square_cells:
        raise VegetationProjectionError('non-square cells are not supported; refused before any allocation')
    surf, mask = _prep_surface_mask(domain, surface, active_mask)
    N, S, L = domain.ncells, catalog.n_species, catalog.root_layers
    A = session_dims['n_rows'] if st is None else int(st.n)           # a session snapshot materialises EVERY base row (live, dead and removed)
    cell_area = float(domain.dx_m) * float(domain.dx_m)
    # ---- gates from METADATA before any owned allocation: validate (+ cold source digests), then the snapshot and select stages (P <= A bounds the selection)
    est = estimate_projection_bytes(A, A, 0, N, S, L, builder=builder, from_session=st is None, source_cold=('rec' not in _SOURCE_CACHE or 'ext' not in _SOURCE_CACHE), backend=backend)
    quota.admit('validate', est['stages']['validate'])
    if st is None:
        quota.admit('snapshot', est['stages']['snapshot'])
    quota.admit('select', est['stages']['select'])
    # ---- bounded validation (chunked, no whole-array temporaries), cold source digests, input hashes
    pkg = package_source_record()
    ext = external_source_record()
    if not _all_finite(surf):
        raise VegetationProjectionError('current_surface must be finite')
    surf_sha = _hash_array(surf).hexdigest()
    mask_u8 = None if mask is None else mask.view(np.uint8)
    mask_sha = _ones_sha(N) if mask_u8 is None else _hash_array(mask_u8).hexdigest()               # a None mask hashes exactly like an all-True mask (no N-byte temporary)
    source, st = source_fn(st)                                     # session: snapshot_state() runs only AFTER the admission above
    if type(st) is not _state.PlantState:
        raise VegetationProjectionError('state must be a genuine leaf.state.PlantState (exact type)')
    if st.catalog_hash != catalog.hash:
        raise VegetationProjectionError('the state catalog hash differs from the catalog (state and catalog must match)')
    if st.domain is not domain and st.domain.to_dict() != domain.to_dict():
        raise VegetationProjectionError('the state domain differs from the planned domain')
    n_state = int(st.n)
    if st is not None and source['kind'] == 'plant_session' and n_state > A:
        raise VegetationProjectionError(f'the session snapshot has {n_state} rows, above the admitted bound {A} (base rows)')
    geo = _canopy.derive_geometry(st, catalog, current_surface=surf)
    if source['kind'] == 'plant_session':
        del st                                                     # the session snapshot is OUR array set: release it as soon as the geometry exists
    status = geo['status']
    live_all = status == _state.LIVE
    dead_all = status == _state.DEAD_STANDING
    buried = geo['fully_buried']
    pos = geo['diameter_m'] > 0.0
    excl = {'removed': int((status == _state.REMOVED).sum()), 'fully_buried': int(((live_all | dead_all) & buried).sum()),
            'zero_diameter': int(((live_all | dead_all) & ~pos).sum())}
    rows = np.flatnonzero((live_all | dead_all) & ~buried & pos)
    del live_all, dead_all, buried, pos, status
    rows = rows[np.lexsort((geo['plant_id'][rows], geo['species_index'][rows]))]          # stable (species id, plant id) order
    ids = np.ascontiguousarray(geo['plant_id'][rows], dtype=np.int64)
    sidx = np.ascontiguousarray(geo['species_index'][rows], dtype=np.int64)
    is_live = geo['status'][rows] == _state.LIVE
    x = np.ascontiguousarray(geo['x_m'][rows], dtype=np.float64)
    y = np.ascontiguousarray(geo['y_m'][rows], dtype=np.float64)
    rad = np.ascontiguousarray(geo['radius_m'][rows], dtype=np.float64)
    del geo, rows
    P_elig = int(ids.shape[0])
    # ---- ACTIVE-FIRST count, every temporary admitted before it exists
    bbox = 0
    if builder == 'numpy' and P_elig and not domain.any_periodic:
        quota.admit('count_pre', estimate_projection_bytes(A, P_elig, 0, N, S, L, builder=builder, backend=backend)['stages']['count_pre'])
        from leaf import roots as _roots
        bbox = int(_roots.numpy_bbox_scratch_bytes(x - domain.origin_x_m, y - domain.origin_y_m, rad, domain.nx, domain.ny, float(domain.dx_m))[0])
    quota.admit('count', estimate_projection_bytes(A, P_elig, 0, N, S, L, builder=builder, bbox_scratch_bytes=bbox, backend=backend)['stages']['count'])
    counts = _active_counts(domain, x, y, rad, mask_u8, mask_u8 is not None, builder)
    keep_p = counts > 0
    P = int(keep_p.sum())
    quota.admit('filter', estimate_projection_bytes(A, P, 0, N, S, L, builder=builder, n_eligible=P_elig, backend=backend)['stages']['filter'])
    omitted = np.array(ids[~keep_p], dtype=np.int64)
    ids, sidx, is_live, x, y, rad, counts = ids[keep_p], sidx[keep_p], is_live[keep_p], x[keep_p], y[keep_p], rad[keep_p], counts[keep_p]
    E = int(counts.sum())
    del keep_p, counts
    if E > max_edges:
        raise VegetationBudgetError('canopy_edges', f'{E} ACTIVE edges exceed max_edges={max_edges}; nothing allocated')
    # ---- the literal composed admission of EVERY remaining stage (tables included) BEFORE the tables or the builder allocate
    full = estimate_projection_bytes(A, P, E, N, S, L, builder=builder, bbox_scratch_bytes=bbox, n_omitted=int(omitted.shape[0]), n_eligible=P_elig, backend=backend)
    for stage in ('build', 'aggregate', 'reduce', 'seal'):
        quota.admit(stage, full['stages'][stage])
    # ---- species tables (O(S L)) and the participating ACTIVE live species / required trait
    root = np.array(catalog.root_layer_table(), dtype=np.float64)
    if not np.isfinite(root).all() or (root < 0.0).any():
        raise VegetationProjectionError('the catalog root layer fractions must be finite and >= 0')
    live_species = sorted(set(int(v) for v in sidx[is_live]))
    has_w = np.array([bool(catalog.has_preferential_receiver_weights(sid)) for sid in catalog.species_ids], dtype=bool)
    if require_receiver_traits:
        missing = [catalog.species_ids[v] for v in live_species if not has_w[v]]
        if missing:
            raise VegetationProjectionError(f'participating live species {missing} have no preferential_infiltration_receiver_weights trait (catalog schema_version '
                                            f'{catalog.schema_version}); nothing is defaulted (pass require_receiver_traits=False for splash/shading/R1 only)')
    weights = np.zeros((S, L))
    for v in range(S):
        if has_w[v]:
            w_ = catalog.preferential_receiver_weights(catalog.species_ids[v])
            if not np.isfinite(w_).all() or (w_ < 0.0).any():
                raise VegetationProjectionError(f'species {catalog.species_ids[v]!r}: receiving weights must be finite and >= 0')
            weights[v] = w_
    lower = np.array([float(sum(float(t) for t in root[v, 1:])) for v in range(S)])
    n_entries = 0
    if P:
        try:
            cmap = _pet.build_canopy_map(domain, ids, x, y, rad, max_edges=max_edges, max_bytes=quota.max_bytes, backend='numpy', valid_cell_mask=mask, builder=builder,
                                         reserve_bytes=quota.reserve + SELECT_ARRAYS_PER_PLANT * P + (int(weights.nbytes) + int(root.nbytes)))
        except _pet.PlantPetBudgetError as exc:
            raise VegetationBudgetError(f'canopy_build/{exc.stage}', exc.reason) from exc
        except _pet.PlantPetBackendError as exc:
            raise VegetationBackendError(str(exc)) from exc
        except _pet.PlantPetError as exc:
            raise VegetationProjectionError(f'canopy geometry refused: {exc}') from exc
        if int(cmap.n_edges) != E:
            raise VegetationProjectionError('internal mismatch between the active-first count and the builder edge count')
        del ids, x, y, rad                                          # the selection arrays are no longer needed once the footprints exist
        offs, e_cell, e_area = cmap.view('offsets'), cmap.view('cells'), cmap.view('areas_m2')        # read-only views of the sealed CSR (no copy)
        e_plant = np.repeat(np.arange(P, dtype=np.int64), np.diff(offs))
        del offs
        e_sp = sidx[e_plant].astype(np.int32)
        e_live = is_live[e_plant]
        del e_plant, cmap, sidx, is_live
        key = e_cell * np.int64(S)
        key += e_sp
        order = np.argsort(key, kind='stable')                    # stable: plant order (species id, plant id) is kept inside an entry
        key = key[order]
        e_cell, e_sp, e_area, e_live = e_cell[order], e_sp[order], e_area[order], e_live[order]
        del order
        newg = np.empty(key.shape[0], dtype=bool)
        newg[0] = True
        np.not_equal(key[1:], key[:-1], out=newg[1:])
        del key
        gid = np.cumsum(newg)
        gid -= 1
        first = np.flatnonzero(newg)
        del newg
        n_entries = int(first.shape[0])
        ent_cell, ent_sp = e_cell[first], e_sp[first]
        del first
        if backend == 'numpy':
            a_all, a_live = _aggregate_numpy(gid, e_area, e_live, n_entries)
        else:
            agg, _ = _numba_kernels()
            a_all, a_live = np.zeros(n_entries), np.zeros(n_entries)
            agg(gid, e_area, e_live, a_all, a_live)
        del e_cell, e_sp, e_area, e_live, gid
    else:
        ent_cell, ent_sp, a_all, a_live = np.zeros(0, np.int64), np.zeros(0, np.int32), np.zeros(0), np.zeros(0)
    off = np.zeros(N + 1, dtype=np.int64)
    off[1:] = np.bincount(ent_cell, minlength=N)
    np.cumsum(off, out=off)
    root0 = np.ascontiguousarray(root[:, 0])
    if backend == 'numpy':
        splash_raw, shade_raw, r1 = _reduce_numpy(off, ent_sp, a_all, a_live, root0, cell_area, N)
    elif backend == 'numba':
        _, red = _numba_kernels()
        splash_raw, shade_raw, r1 = np.zeros(N), np.zeros(N), np.zeros(N)
        red(off, ent_sp, a_all, a_live, root0, cell_area, splash_raw, shade_raw, r1)
    else:
        three = _reduce_cuda(device, dq, off, ent_sp, a_all, a_live, root0, cell_area, N)          # ONE (3, N) host array from ONE device-to-host call
        splash_raw, shade_raw, r1 = three[0], three[1], three[2]
    fa = a_all / cell_area
    fl = a_live / cell_area
    capped_all, capped_live, max_fa = int((fa > 1.0).sum()), int((fl > 1.0).sum()), float(fa.max(initial=0.0))
    del fa, fl
    diag = {'rule': RULE, 'n_plants_state': n_state, 'n_plants_eligible': P_elig, 'n_plants_projected': P, 'n_omitted_no_active_overlap': int(omitted.shape[0]),
            'excluded_by_reason': excl, 'n_species_cell_entries': int(n_entries), 'n_species_cell_capped_all': capped_all, 'n_species_cell_capped_live': capped_live,
            'n_cells_total_capped_splash': int((splash_raw > 1.0).sum()), 'n_cells_total_capped_shading': int((shade_raw > 1.0).sum()),
            'max_species_fraction_uncapped': max_fa, 'max_total_uncapped_splash': float(splash_raw.max(initial=0.0)),
            'max_total_uncapped_shading': float(shade_raw.max(initial=0.0)), 'n_inactive_cells': 0 if mask is None else int((~mask).sum()),
            'participating_live_species': [catalog.species_ids[t] for t in live_species],
            'receiver_trait_disabled_species': [catalog.species_ids[t] for t in range(S) if has_w[t] and not float(weights[t].max()) > 0.0],
            'max_edges': int(max_edges), 'n_edges_active': E, 'declared_peak_bytes': full['peak_bytes'],
            'declared_resident_bytes': estimate_projection_bytes(A, P, E, N, S, L, n_entries=n_entries, n_omitted=int(omitted.shape[0]), backend=backend)['resident_bytes']}
    np.minimum(splash_raw, 1.0, out=splash_raw)                     # the caps are applied IN PLACE (no second array)
    np.minimum(shade_raw, 1.0, out=shade_raw)
    raw = {'splash_cover': splash_raw, 'shading_cover': shade_raw, 'matrix_root_index': r1, 'cell_offsets': off, 'entry_cell': ent_cell, 'entry_species': ent_sp,
           'entry_area_all_m2': a_all, 'entry_area_live_m2': a_live}
    del splash_raw, shade_raw, r1, off, ent_cell, ent_sp, a_all, a_live
    if backend == 'cuda':
        del three
    arrays = _seal_arrays(raw)                                      # each raw array is released as soon as its immutable copy exists
    tab_raw = {'root': root.reshape(-1), 'lower': lower, 'weights': weights.reshape(-1), 'has_w': has_w.astype(np.uint8), 'omitted_ids': omitted}
    tables = _seal_arrays(tab_raw)
    tables['weights'] = tables['weights'].reshape(S, L)             # a view of the immutable bytes array (the chain still ends in bytes)
    tables['root'] = tables['root'].reshape(S, L)
    content_sha = _content_digest(_content_header(catalog.species_ids, domain.shape, cell_area, N, n_entries, P, L), arrays, tables)      # the ONE canonical hash (also H1)
    ident = {'schema': SCHEMA, 'version': VERSION, 'rule': RULE, 'source': source, 'catalog_hash': catalog.hash, 'domain': domain.to_dict(), 'surface_sha256': surf_sha,
             'active_mask_sha256': mask_sha, 'package': pkg, 'external_helpers': ext, 'backend': backend, 'builder': builder, 'device': device, 'species_ids': list(catalog.species_ids),
             'root_layers': L, 'content_sha256': content_sha, 'cap': {'species_cell': 1.0, 'total': 1.0}, 'max_edges': int(max_edges), 'max_bytes': quota.max_bytes,
             'transfers': None if backend != 'cuda' else dict(estimate_device_bytes(N, n_entries, S), h2d_calls=5, d2h_calls=1),
             'canonical_outputs': 'host arrays (a cuda reduction returns them by ONE device-to-host call; no resident GPU maps)'}
    resident = sum(a.nbytes for a in arrays.values()) + sum(a.nbytes for a in tables.values())
    with _minting():
        return VegetationProjection(_a=arrays, _t=tables, identity=_freeze(ident), diagnostics=_freeze(diag), species_ids=tuple(catalog.species_ids), n_cells=N,
                                    shape=tuple(domain.shape), cell_area_m2=cell_area, n_entries=int(n_entries), n_plants=P, resident_bytes=int(resident))


def _prep_common(max_edges, max_bytes, reserve_bytes, available_bytes_fn, backend, builder, device, available_device_bytes_fn, max_device_bytes, device_reserve_bytes):
    if backend not in BACKENDS:
        raise VegetationProjectionError(f'backend must be one of {BACKENDS}, got {backend!r}')
    me = _strict_cap(max_edges, 'max_edges')
    quota = _quota(max_bytes, reserve_bytes, available_bytes_fn)
    dq = None
    if backend == 'cuda':
        if device is None:
            raise VegetationBackendError('backend "cuda" requires an explicit integer device (no fallback)')
        _traits.strict_int(device, 'device', 0)
        if max_device_bytes is None:
            raise VegetationProjectionError('backend "cuda" requires an explicit max_device_bytes (separate device pool)')
        dq = _quota(max_device_bytes, device_reserve_bytes, available_device_bytes_fn, 'device')
    elif device is not None or max_device_bytes is not None or available_device_bytes_fn is not None or device_reserve_bytes:
        raise VegetationProjectionError('device / device-pool arguments are only meaningful for backend "cuda"')
    if builder is None:
        builder = 'numpy' if backend == 'numpy' else 'compiled'
    if builder not in _pet.BUILDERS:
        raise VegetationProjectionError(f'builder must be one of {_pet.BUILDERS}')
    return me, quota, dq, builder


def project_from_plant_session(session, current_surface, *, active_mask=None, max_edges, max_bytes, reserve_bytes=0, backend='numpy', builder=None, device=None,
                               available_bytes_fn=None, max_device_bytes=None, device_reserve_bytes=0, available_device_bytes_fn=None, require_receiver_traits=True):
    '''Projection of an ACCEPTED `PlantSession` generation (exact type). The session's metadata (live count, domain, catalog) is admitted BEFORE `snapshot_state()` allocates the
    state; the identity records the session token, sequence, pop and geometry hashes and `check_session` later detects a stale one. Nothing of the session is mutated.'''
    from leaf import plant_session as _ps
    if type(session) is not _ps.PlantSession:
        raise VegetationProjectionError('session must be a genuine leaf.plant_session.PlantSession (exact type)')
    me, quota, dq, builder = _prep_common(max_edges, max_bytes, reserve_bytes, available_bytes_fn, backend, builder, device, available_device_bytes_fn, max_device_bytes,
                                          device_reserve_bytes)
    cat = session.config.catalog
    dims = {'n_rows': int(session.base.n), 'domain': session.base.domain}

    def source_fn(_none):
        st = session.snapshot_state()
        return ({'kind': 'plant_session', 'token': session.token, 'sequence': int(session.sequence), 'pop_sha256': session.pop_sha256, 'geometry_sha256': session.geometry_sha256,
                 'session_id': session.session_id, 'lineage': session.lineage, 'catalog_hash': cat.hash,
                 'note': 'accepted session generation; the state is the session snapshot at its cursor'}, st)

    return _build(None, cat, current_surface, active_mask, source_fn=source_fn, session_dims=dims, max_edges=me, quota=quota, dq=dq, backend=backend, builder=builder,
                  device=device, require_receiver_traits=require_receiver_traits)


def project_from_initial_state(st, catalog, current_surface, *, active_mask=None, max_edges, max_bytes, reserve_bytes=0, backend='numpy', builder=None, device=None,
                               available_bytes_fn=None, max_device_bytes=None, device_reserve_bytes=0, available_device_bytes_fn=None, require_receiver_traits=True):
    '''Projection of an INITIAL `PlantState` (exact type; no accepted session generation exists yet). Labelled `initial_state` with a content hash of the state: NOT equivalent to a
    session receipt, and `check_session` refuses it. The state is borrowed and never mutated.'''
    if type(st) is not _state.PlantState:
        raise VegetationProjectionError('state must be a genuine leaf.state.PlantState (exact type)')
    me, quota, dq, builder = _prep_common(max_edges, max_bytes, reserve_bytes, available_bytes_fn, backend, builder, device, available_device_bytes_fn, max_device_bytes,
                                          device_reserve_bytes)

    def source_fn(state_):
        return ({'kind': 'initial_state', 'state_sha256': _state_sha256(state_), 'catalog_hash': state_.catalog_hash,
                 'note': 'initial state projection: not an accepted PlantSession generation and no session identity is claimed'}, state_)

    return _build(st, catalog, current_surface, active_mask, source_fn=source_fn, session_dims=None, max_edges=me, quota=quota, dq=dq, backend=backend, builder=builder,
                  device=device, require_receiver_traits=require_receiver_traits)


# ------------------------------------------------------------------ PUBLIC PURE COMPACT HELPERS (H1 hash / H2 streams / H3 fields)
# They operate on the mapping of `VegetationProjection.export_compact()` (or an equal mapping a restart record reads back) and are PURE derived calculations: they create no
# projection, no session receipt and no producer proof. Source identity (the actual LEAF package digest / external helper record) is NOT a content-generation hash: the CALLER
# (the owning SYRUP record) compares the actual imported package with its external expected pins. COLD (once per loaded record): `compact_content_sha256` validates every structural
# invariant and hashes; `rederive_fields_from_compact` re-validates the structure and recomputes the O(N) fields. PER CALL: the stream / cover helpers check metadata only
# (types, dtypes, shapes, counts) in O(1); they RELY on the caller having passed the cold validation of the same arrays in this process (an unvalidated negative entry index would wrap).
COMPACT_CHUNK = 1 << 16                           # elements per validation chunk (bounds every validation temporary)
_COMPACT_REQUIRED = frozenset(('schema', 'version', 'field_order', 'arrays', 'species_ids', 'n_cells', 'shape', 'cell_area_m2', 'n_entries', 'n_plants', 'root_layers'))
_COMPACT_OPTIONAL = frozenset(('content_sha256', 'identity'))
_COMPACT_DTYPES = {'splash_cover': np.float64, 'shading_cover': np.float64, 'matrix_root_index': np.float64, 'cell_offsets': np.int64, 'entry_cell': np.int64,
                   'entry_species': np.int32, 'entry_area_all_m2': np.float64, 'entry_area_live_m2': np.float64, 'root': np.float64, 'lower': np.float64, 'weights': np.float64,
                   'has_w': np.uint8, 'omitted_ids': np.int64}


class CompactContentError(VegetationProjectionError):
    '''A compact mapping is malformed, inconsistent or its recorded content hash differs from the recomputed one (never repaired or normalised).'''


def estimate_compact_validation_bytes(n_cells, n_entries, n_species, n_layers, n_omitted=0, *, with_mask=False, max_id_chars=0):
    '''LITERAL bound of the transient bytes of the cold structural validation (`compact_content_sha256`, and the validation inside `rederive_fields_from_compact`): 6 chunk-sized
    8-byte temporaries (a chunk is `COMPACT_CHUNK` elements of the cell or entry dimension) + the unique-id sort of the omitted ids 24 K + the species-id uniqueness sort 12 S (a sorted
    pointer list 8 S plus the merge buffer <= 4 S) + the species-table row chunks (R = max(1, chunk // L) rows: 4 R L bytes of masks + 48 R for the lower-sum list / array) + the
    streamed content-header transient (3 copies of the longest identifier: 3 * (12 * max_id_chars + 4); pass `max_id_chars=0` for a caller that does not hash the header) + the
    documented small Python overhead `OVERHEAD_SLACK_BYTES` (never an O(N) / O(S) term); the mask check adds one bool chunk. Nothing scales with N or M beyond the chunk and nothing
    with S beyond the 12 S sort. The identifiers themselves are caller-owned (this is not a whole-process bound).'''
    N, M, S, L, K, C = (_traits.strict_int(v, k, 0) for v, k in ((n_cells, 'n_cells'), (n_entries, 'n_entries'), (n_species, 'n_species'), (n_layers, 'n_layers'), (n_omitted, 'n_omitted'),
                                                                (max_id_chars, 'max_id_chars')))
    chunk = min(COMPACT_CHUNK, max(N + 1, M, 1))
    rows = min(max(S, 1), max(1, COMPACT_CHUNK // max(L, 1)))
    return 6 * 8 * chunk + 24 * K + 12 * S + 4 * rows * L + 48 * rows + 3 * (12 * C + 4) + (chunk if with_mask else 0) + OVERHEAD_SLACK_BYTES


def estimate_rederive_bytes(n_cells, n_species, *, backend='numpy'):
    '''LITERAL model of `rederive_fields_from_compact`: the three fresh (N,) results 24 N plus the reduction temporaries (numpy: the (N,) counts 8 N, the bool mask N and, at the
    worst rank, rows / k / fa / fl 32 N plus the add temporaries r1[rows], the species gather and the product 28 N; numba: no temporaries) and the (S,) top-layer copy 8 S.
    The caller's reserve is counted once by the quota, never inside these numbers.'''
    N, S = _traits.strict_int(n_cells, 'n_cells', 0), _traits.strict_int(n_species, 'n_species', 0)
    if backend not in ('numpy', 'numba'):
        raise VegetationBackendError(f'compact re-derivation supports the CPU backends only, got {backend!r}')
    outputs = 24 * N
    scratch = (8 * N + N + 32 * N + 28 * N if backend == 'numpy' else 0) + 8 * S
    return {'outputs_bytes': outputs, 'scratch_bytes': scratch, 'peak_bytes': outputs + scratch}


def _compact_meta(compact):
    '''METADATA-only validation (types, dtypes, shapes, counts, key sets; no array read): returns a plain dict of the validated pieces. NOTHING O(S) is allocated here (no set, list
    or tuple copy of the identifiers: only counting / type loops), so the caller's admission precedes every owned metadata expansion; the species-id UNIQUENESS (a sort) is part of
    the admitted cold validation. Unknown / missing keys or fields, a wrong / non-integer version, order, dtype, shape or count are refused (nothing is coerced or repaired).'''
    from collections.abc import Mapping
    if not isinstance(compact, Mapping):
        raise CompactContentError('compact must be a mapping (the export_compact mapping)')
    if len(compact) > len(_COMPACT_REQUIRED) + len(_COMPACT_OPTIONAL):
        raise CompactContentError('compact has unknown keys')
    for k in compact:
        if k not in _COMPACT_REQUIRED and k not in _COMPACT_OPTIONAL:
            raise CompactContentError(f'compact has an unknown key {k!r}')
    for k in _COMPACT_REQUIRED:
        if k not in compact:
            raise CompactContentError(f'compact is missing {k!r}')
    try:
        version = _traits.strict_int(compact['version'], 'version')
    except _traits.CatalogError as exc:
        raise CompactContentError(f'compact version must be an integer: {exc}') from exc
    if compact['schema'] != SCHEMA or version != VERSION:
        raise CompactContentError(f'compact must be {SCHEMA} version {VERSION}')
    fo = compact['field_order']
    if not isinstance(fo, (tuple, list)) or len(fo) != len(_ARRAYS) + len(_TABLES) or any(a != b for a, b in zip(fo, _ARRAYS + _TABLES)):
        raise CompactContentError('compact field_order is not the canonical order')
    ids = compact['species_ids']
    if not isinstance(ids, (tuple, list)) or not ids:
        raise CompactContentError('species_ids must be a non-empty sequence of strings')
    max_id = 0
    for v in ids:
        if not isinstance(v, str) or not v:
            raise CompactContentError('species_ids must be non-empty strings')
        max_id = max(max_id, len(v))
    try:
        N, M, P, L = (_traits.strict_int(compact[k], k, lo) for k, lo in (('n_cells', 1), ('n_entries', 0), ('n_plants', 0), ('root_layers', 1)))
    except _traits.CatalogError as exc:
        raise CompactContentError(str(exc)) from exc
    shape = compact['shape']
    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
        raise CompactContentError('shape must be (ny, nx)')
    try:
        shape = tuple(_traits.strict_int(v, 'shape', 1) for v in shape)
    except _traits.CatalogError as exc:
        raise CompactContentError(str(exc)) from exc
    area = compact['cell_area_m2']
    if isinstance(area, (bool, np.bool_)) or not isinstance(area, (float, np.floating)) or not np.isfinite(area) or not float(area) > 0.0:
        raise CompactContentError('cell_area_m2 must be a finite float > 0')
    S = len(ids)
    if shape[0] * shape[1] != N:
        raise CompactContentError('shape does not match n_cells')
    if M > N * S or (P == 0 and M != 0):
        raise CompactContentError('n_entries is inconsistent with n_cells, the species count and n_plants')
    rec = compact.get('content_sha256')
    if rec is not None and not (isinstance(rec, str) and len(rec) == 71 and rec.startswith('sha256:') and all(c in '0123456789abcdef' for c in rec[7:])):
        raise CompactContentError('content_sha256 must be "sha256:" + 64 lowercase hex')
    arrs = compact['arrays']
    if not isinstance(arrs, Mapping) or len(arrs) != len(_ARRAYS) + len(_TABLES) or any(k not in arrs for k in _ARRAYS + _TABLES):
        raise CompactContentError('arrays must hold exactly the canonical fields')
    shapes = {'splash_cover': (N,), 'shading_cover': (N,), 'matrix_root_index': (N,), 'cell_offsets': (N + 1,), 'entry_cell': (M,), 'entry_species': (M,), 'entry_area_all_m2': (M,),
              'entry_area_live_m2': (M,), 'root': (S, L), 'lower': (S,), 'weights': (S, L), 'has_w': (S,)}
    for k in _ARRAYS + _TABLES:
        a = arrs[k]
        if type(a) is not np.ndarray or a.dtype != np.dtype(_COMPACT_DTYPES[k]) or not a.flags.c_contiguous or a.ndim not in (1, 2):
            raise CompactContentError(f'{k}: must be a C-contiguous numpy array of dtype {np.dtype(_COMPACT_DTYPES[k])}')
        if k in shapes and tuple(a.shape) != shapes[k]:
            raise CompactContentError(f'{k}: shape {tuple(a.shape)} != {shapes[k]}')
        if k == 'omitted_ids' and a.ndim != 1:
            raise CompactContentError('omitted_ids must be 1-D')
    return {'ids': ids, 'max_id': max_id, 'N': N, 'M': M, 'P': P, 'L': L, 'S': S, 'shape': shape, 'area': float(area), 'arrays': arrs, 'recorded': rec}


def _chunks(n, size=COMPACT_CHUNK):
    for lo in range(0, n, size):
        yield lo, min(lo + size, n)


def _fail(msg):
    raise CompactContentError(msg)


def _validate_compact_content(m, *, active_mask=None):
    '''Chunked structural validation of the VALUES (no whole-array temporaries; no repair): finiteness and ranges, CSR consistency, strictly increasing (cell, species) entries,
    area ordering, table invariants. With `active_mask` (a bool flat array) every entry must lie in an active cell.'''
    a, N, M, S = m['arrays'], m['N'], m['M'], m['S']
    for k in ('splash_cover', 'shading_cover'):
        for lo, hi in _chunks(N):
            v = a[k][lo:hi]
            if not (np.isfinite(v).all() and (v >= 0.0).all() and (v <= 1.0).all()):
                _fail(f'{k} must be finite and within [0, 1]')
    for lo, hi in _chunks(N):
        v = a['matrix_root_index'][lo:hi]
        if not (np.isfinite(v).all() and (v >= 0.0).all()):
            _fail('matrix_root_index must be finite and >= 0')
    off = a['cell_offsets']
    if int(off[0]) != 0 or int(off[N]) != M:
        _fail('cell_offsets must start at 0 and end at n_entries')
    for lo, hi in _chunks(N):
        if (np.diff(off[lo:hi + 1]) < 0).any():
            _fail('cell_offsets must be non-decreasing')
    ec, es, aa, al = a['entry_cell'], a['entry_species'], a['entry_area_all_m2'], a['entry_area_live_m2']
    prev_key = -1
    for lo, hi in _chunks(M):
        c, sp = ec[lo:hi], es[lo:hi]
        if (c < 0).any() or (c >= N).any() or (sp < 0).any() or (sp >= S).any():
            _fail('entry_cell / entry_species out of range')
        if not np.array_equal(np.searchsorted(off, np.arange(lo, hi, dtype=np.int64), side='right') - 1, c):
            _fail('entry_cell is inconsistent with cell_offsets')
        key = c * np.int64(S) + sp
        if int(key[0]) <= prev_key or (np.diff(key) <= 0).any():
            _fail('entries must be strictly increasing in (cell, species)')
        prev_key = int(key[-1])
        x, y = aa[lo:hi], al[lo:hi]
        if not (np.isfinite(x).all() and np.isfinite(y).all() and (y >= 0.0).all() and (x >= y).all()):
            _fail('entry areas must be finite with 0 <= live <= all')
        if active_mask is not None and not active_mask[c].all():
            _fail('an entry lies in an inactive cell')
    ids = sorted(m['ids'])                                    # the admitted 12 S uniqueness sort (the identifiers are not copied)
    if any(x == y for x, y in zip(ids, ids[1:])):
        _fail('species_ids must be distinct')
    del ids
    root, lower, w, has = a['root'], a['lower'], a['weights'], a['has_w']
    L = m['L']
    rows = max(1, COMPACT_CHUNK // L)
    for lo in range(0, S, rows):                              # species rows in chunks: no S-sized temporary
        hi = min(lo + rows, S)
        r, wr, hr = root[lo:hi], w[lo:hi], has[lo:hi]
        if not (np.isfinite(r).all() and (r >= 0.0).all()):
            _fail('root layer fractions must be finite and >= 0')
        want = np.array([float(sum(float(t) for t in r[v, 1:])) for v in range(hi - lo)])          # the producer's exact left-fold
        if not np.array_equal(want, lower[lo:hi]):
            _fail('lower does not equal the sum of the lower root layers')
        if not (((hr == 0) | (hr == 1)).all() and np.isfinite(wr).all() and (wr >= 0.0).all()):
            _fail('has_w must be 0 / 1 and the receiving weights finite and >= 0')
        if (wr[hr == 0] != 0.0).any():
            _fail('a species without a receiving-weight trait must have all-zero weights')
    om = a['omitted_ids']
    if om.shape[0] != np.unique(om).shape[0]:
        _fail('omitted_ids must be distinct')


def compact_content_sha256(compact, *, max_bytes, reserve_bytes=0, available_bytes_fn=None):
    '''H1. The EXACT V3 content hash ("sha256:<hex>") of a compact mapping, by the SAME private header / digest functions the producer uses. Metadata is validated first (O(1) beyond
    the S x L tables), the chunked structural validation is admitted under the quota (`estimate_compact_validation_bytes`) BEFORE it allocates, then the arrays are hashed in place
    (no copy). Nothing is coerced, normalised or repaired; a malformed mapping raises `CompactContentError`; a mapping carrying its own `content_sha256` that differs from the
    recomputed hash is refused. Pure: no projection, session receipt or producer proof is created, and the LEAF source identity is NOT part of this hash.'''
    q = _quota(max_bytes, reserve_bytes, available_bytes_fn)
    m = _compact_meta(compact)
    q.admit('compact_validate', estimate_compact_validation_bytes(m['N'], m['M'], m['S'], m['L'], m['arrays']['omitted_ids'].shape[0], max_id_chars=m['max_id']))
    _validate_compact_content(m)
    arrays, tables = ({k: m['arrays'][k] for k in grp} for grp in (_ARRAYS, _TABLES))
    got = _content_digest(_content_header(m['ids'], m['shape'], m['area'], m['N'], m['M'], m['P'], m['L']), arrays, tables)
    if m['recorded'] is not None and m['recorded'] != got:
        _fail(f'recorded content_sha256 {m["recorded"]} differs from the recomputed {got}')
    return got


def species_cover_from_compact(compact, species_id, eligibility='live', *, max_bytes, reserve_bytes=0, available_bytes_fn=None):
    '''H2 (single species). FRESH (N,) f_s = min(sum eligible areas / cell area, 1) of one species from the compact entries: the SAME `_species_cover` arithmetic as
    `VegetationProjection.species_cell_cover`. Per-call metadata check only (see the section note); result and scratch are admitted first.'''
    m = _compact_meta(compact)
    if eligibility not in ELIGIBILITY:
        raise VegetationProjectionError(f'eligibility must be one of {ELIGIBILITY}')
    if not isinstance(species_id, str) or species_id not in m['ids']:
        raise VegetationProjectionError(f'unknown species {species_id!r}')
    _quota(max_bytes, reserve_bytes, available_bytes_fn).admit('species_cover', estimate_stream_bytes(m['N'], m['M'], 0, m['L'])['species_cover_bytes'])
    a = m['arrays']
    return _species_cover(a['entry_species'], a['entry_area_live_m2' if eligibility == 'live' else 'entry_area_all_m2'], a['entry_cell'], m['area'], m['N'], m['ids'].index(species_id))


def species_contribution_from_compact(compact, species_id, *, max_bytes, reserve_bytes=0, available_bytes_fn=None):
    '''H2 (one contribution). FRESH (N,) c_s = f_live_s * lower_s (the preferential contribution of one species) with the same arithmetic and float order as the stream.'''
    m = _compact_meta(compact)
    if not isinstance(species_id, str) or species_id not in m['ids']:
        raise VegetationProjectionError(f'unknown species {species_id!r}')
    _quota(max_bytes, reserve_bytes, available_bytes_fn).admit('species_cover', estimate_stream_bytes(m['N'], m['M'], 0, m['L'])['species_cover_bytes'])
    a, i = m['arrays'], m['ids'].index(species_id)
    return _species_cover(a['entry_species'], a['entry_area_live_m2'], a['entry_cell'], m['area'], m['N'], i, float(a['lower'][i]))


def preferential_inputs_from_compact(compact, *, max_bytes, reserve_bytes=0, available_bytes_fn=None):
    '''H2 (stream). `(species_ids, raw receiver weights (n, L), contributions)` exactly like `VegetationProjection.preferential_inputs` (same `_preferential_stream`, same per-next
    admission, one FRESH (N,) array per species in order, nothing retained, no S x N / P x N): species with an ENABLED positive receiving vector only. The result is a DERIVED
    calculation, not a PlantSession receipt.'''
    m = _compact_meta(compact)
    a = m['arrays']
    return _preferential_stream(_quota(max_bytes, reserve_bytes, available_bytes_fn), m['N'], m['M'], m['ids'], a['weights'], a['lower'], a['has_w'],
                                lambda i, elig, scale: _species_cover(a['entry_species'], a['entry_area_live_m2'], a['entry_cell'], m['area'], m['N'], i, scale))


class RederivedFields:
    '''Typed receipt of `rederive_fields_from_compact`: three FRESH writable (N,) arrays owned by the caller plus what was and was not checked. It is a DERIVED calculation: it is
    never a `VegetationProjection`, carries no producer proof and no source identity (`genuine_projection` is always False).'''
    __slots__ = ('active_mask_checked', 'backend', 'matrix_root_index', 'n_cells', 'n_entries', 'shading_cover', 'splash_cover')
    genuine_projection = False

    def __init__(self, **kw):
        for k, v in kw.items():
            object.__setattr__(self, k, v)

    def __setattr__(self, name, value):
        raise AttributeError('RederivedFields is immutable')

    def __repr__(self):
        return f'RederivedFields(N={self.n_cells}, entries={self.n_entries}, backend={self.backend}, active_mask_checked={self.active_mask_checked})'


def rederive_fields_from_compact(compact, *, backend='numpy', active_mask=None, expected_active_mask_sha256=None, max_bytes, reserve_bytes=0, available_bytes_fn=None):
    '''H3. Recompute splash / shading cover and R1 from the compact entries and the top-layer root fractions through the SAME reduction the producer uses (`_reduce_numpy`, or the
    numba kernel; the caps are applied in place exactly as in the build). The CPU backends only (`cuda` is refused: a GPU-prepared law is checked downstream). The structure is
    re-validated first (cold, chunked), the literal need (`estimate_rederive_bytes`, `estimate_compact_validation_bytes`) is admitted BEFORE any array is allocated, and the caller then
    compares every returned array with its sealed cached values. The ACTIVE MASK is checked only when passed: it must come WITH its known sha256 hex (`expected_active_mask_sha256`),
    its hash must equal it, and every entry must lie in an active cell; it is never guessed from missing entries or the current surface. Pure and CPU only.'''
    if backend not in ('numpy', 'numba'):
        raise VegetationBackendError(f'compact re-derivation supports the CPU backends only, got {backend!r}')
    q = _quota(max_bytes, reserve_bytes, available_bytes_fn)
    m = _compact_meta(compact)
    N, S = m['N'], m['S']
    mask_flat = None
    if active_mask is not None:
        if type(active_mask) is not np.ndarray or active_mask.dtype != np.bool_ or tuple(active_mask.shape) != m['shape'] or not active_mask.flags.c_contiguous:
            raise VegetationProjectionError(f'active_mask must be a C-contiguous bool numpy array of shape {m["shape"]}')
        if not isinstance(expected_active_mask_sha256, str):
            raise VegetationProjectionError('an active_mask must come with its known expected_active_mask_sha256 (hex)')
        mask_flat = active_mask.reshape(-1)
        if _hash_array(mask_flat.view(np.uint8)).hexdigest() != expected_active_mask_sha256:
            raise StaleProjectionError('the active mask does not match its recorded sha256')
    elif expected_active_mask_sha256 is not None:
        raise VegetationProjectionError('expected_active_mask_sha256 given without an active_mask')
    est = estimate_rederive_bytes(N, S, backend=backend)
    q.admit('compact_rederive', max(est['peak_bytes'], estimate_compact_validation_bytes(N, m['M'], S, m['L'], m['arrays']['omitted_ids'].shape[0], with_mask=mask_flat is not None)))
    _validate_compact_content(m, active_mask=mask_flat)
    a = m['arrays']
    root0 = np.ascontiguousarray(a['root'][:, 0])
    if backend == 'numpy':
        splash, shade, r1 = _reduce_numpy(a['cell_offsets'], a['entry_species'], a['entry_area_all_m2'], a['entry_area_live_m2'], root0, m['area'], N)
    else:
        _, red = _numba_kernels()
        splash, shade, r1 = np.zeros(N), np.zeros(N), np.zeros(N)
        red(a['cell_offsets'], a['entry_species'], a['entry_area_all_m2'], a['entry_area_live_m2'], root0, m['area'], splash, shade, r1)
    np.minimum(splash, 1.0, out=splash)                       # the caps in place, as in the build
    np.minimum(shade, 1.0, out=shade)
    return RederivedFields(splash_cover=splash, shading_cover=shade, matrix_root_index=r1, backend=backend, active_mask_checked=mask_flat is not None, n_cells=N,
                           n_entries=m['M'])



__all__ = ['SCHEMA', 'VERSION', 'RULE', 'BACKENDS', 'ELIGIBILITY', 'VegetationProjection', 'VegetationProjectionError', 'VegetationBudgetError', 'VegetationBackendError',
           'StaleProjectionError', 'estimate_projection_bytes', 'estimate_device_bytes', 'estimate_stream_bytes', 'package_source_record', 'package_sha256', 'project_from_plant_session',
           'project_from_initial_state', 'COMPACT_CHUNK', 'STREAM_FIXED_BYTES', 'CompactContentError', 'RederivedFields', 'estimate_compact_validation_bytes', 'estimate_rederive_bytes',
           'compact_content_sha256', 'species_cover_from_compact', 'species_contribution_from_compact', 'preferential_inputs_from_compact', 'rederive_fields_from_compact']
