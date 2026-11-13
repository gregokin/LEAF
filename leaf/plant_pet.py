'''leaf.plant_pet: canopy-area-weighted per-plant PET (UNQUALIFIED, UNCALIBRATED; no water, no SciPy).

Ownership. The terrain PET field `pet[N]` (mm / day, flat cell order `row * nx + col`) belongs to SYRUP / the caller (the
conversion from m / s is the caller's). LEAF averages it over each plant's CANOPY: PET_p = sum_c A_pc * pet_c / sum_c A_pc,
with A_pc the exact area of (canopy circle of plant p) intersected with (cell c). The result is ONE per-plant PET array
for `leaf.growth.demand_per_plant`; root weights and the allocation (`leaf.uptake`) are unchanged and act afterwards. No
canopy-area rescaling trick, no duplicated demand law, no hidden fallback.

Geometry (exact, no sampling). For a circle of radius r at the origin and the rectangle [x0, x1] x [y0, y1] the overlap is
the integral over x in [x0, x1] ∩ [-r, r] of the chord length l(x) = max(0, min(y1, h) - max(y0, -h)), h = sqrt(r^2 - x^2).
l is piecewise one of four closed forms whose breakpoints are x = ±sqrt(r^2 - y0^2), ±sqrt(r^2 - y1^2); every piece is
integrated with the antiderivative I(x) = (x sqrt(r^2 - x^2) + r^2 asin(x / r)) / 2. Accuracy is FP64 roundoff.
Cells whose four corners lie strictly inside the circle take exactly dx * dy. Edges are the cells whose closest point is
strictly inside the circle (the root-builder predicate), restricted to the valid-cell mask.

Sparse canopy CSR (`CanopyMap`): offsets [P + 1], cells [E], areas_m2 [E], area_total_m2 [P]; never P x N or P x L x N.
`max_edges` / `max_bytes` are explicit strict integers and are checked before any array of that stage exists.

Zero-overlap policy (declared): a plant whose in-domain VALID canopy area is zero (centre inside the closed domain but the
circle covers only masked cells) makes the build REFUSE with the plant count and first ids; there is no site-mean or
uniform fallback. `valid_cell_mask=None` means the WHOLE domain is valid (initial rule for an ambiguous mask). The part of
a canopy outside the closed domain is dropped from numerator and denominator.

Backends: 'numpy' (reference: vectorised cell predicate + the scalar area), 'numba' (compiled count / fill / reduce) and
'cuda' (host-built CSR retained on the device; device reduction kernel). CUDA code is authored for Root GPU0 and was not run by
the author.
'''
from __future__ import annotations

import hashlib
import math

import numpy as np

from leaf import canopy as _canopy
from leaf import roots as _roots
from leaf import state as _state
from leaf import traits as _traits

VERSION = '0.1.0'
LAW = 'circle_cell_exact_area_v1'
LAW_LABEL = ('area-weighted mean of the terrain PET over the exact intersection of each plant canopy circle with the '
             'raster cells; weights are geometric overlap areas, NOT uniform touched-cell weights')
ZERO_OVERLAP_POLICIES = ('refuse',)
BACKENDS = ('numpy', 'numba', 'cuda')
BUILDERS = ('compiled', 'numpy')
CUDA_THREADS = 256
CUDA_MAX_BLOCKS = 1024
EXCLUSIONS = 'Python interpreter, Numba / NVRTC JIT caches, CUDA context and memory-pool rounding are not counted'


class PlantPetError(ValueError):
    '''Any canopy-map / per-plant PET validation error (nothing is returned or published).'''


class PlantPetBudgetError(PlantPetError):
    '''A declared edge / byte bound would be exceeded; raised BEFORE the stage allocates.'''

    def __init__(self, stage, reason):
        super().__init__(f'{stage}: {reason}; nothing allocated')
        self.stage, self.reason = stage, reason


class PlantPetBackendError(PlantPetError):
    '''The requested backend (numba / CuPy / a CUDA device) cannot be used; there is no silent fallback.'''


# ------------------------------------------------------------------ geometry core (one source: Python reference + Numba)
def _make_core(jit):
    @jit
    def anti(x, r):
        '''integral_0^x sqrt(r^2 - t^2) dt for |x| <= r.'''
        s = r * r - x * x
        if s < 0.0:
            s = 0.0
        t = x / r
        if t > 1.0:
            t = 1.0
        if t < -1.0:
            t = -1.0
        return 0.5 * (x * math.sqrt(s) + r * r * math.asin(t))

    @jit
    def area(r, x0, x1, y0, y1):
        '''Area of {x^2 + y^2 <= r^2} ∩ [x0, x1] x [y0, y1] (rectangle in circle-centred coordinates).'''
        if not (r > 0.0) or not (x1 > x0) or not (y1 > y0):
            return 0.0
        if y0 >= r or y1 <= -r or x0 >= r or x1 <= -r:
            return 0.0
        a = x0
        if a < -r:
            a = -r
        b = x1
        if b > r:
            b = r
        bp = np.empty(6)
        n = 1
        bp[0] = a
        for k in range(4):                 # breakpoints -c0, +c0 (from y0) and -c1, +c1 (from y1)
            yy = y0 if k < 2 else y1
            c = -1.0
            if yy > -r and yy < r:
                c = math.sqrt(r * r - yy * yy)
            if c >= 0.0:
                v = c if (k % 2 == 1) else -c
                if v > a and v < b:
                    bp[n] = v
                    n += 1
        bp[n] = b
        n += 1
        for i in range(1, n):          # insertion sort of <= 6 values
            v = bp[i]
            j = i - 1
            while j >= 0 and bp[j] > v:
                bp[j + 1] = bp[j]
                j -= 1
            bp[j + 1] = v
        total = 0.0
        for k in range(n - 1):
            lo = bp[k]
            hi = bp[k + 1]
            if not (hi > lo):
                continue
            mid = 0.5 * (lo + hi)
            s = r * r - mid * mid
            if s < 0.0:
                s = 0.0
            h = math.sqrt(s)
            up_y1 = h >= y1
            low_y0 = -h <= y0
            up = y1 if up_y1 else h
            low = y0 if low_y0 else -h
            if not (up > low):
                continue
            w = hi - lo
            ih = anti(hi, r) - anti(lo, r)
            val = 0.0
            val += y1 * w if up_y1 else ih
            val += -(y0 * w) if low_y0 else ih
            total += val
        return total

    @jit
    def cell_range(c, r, dx, n):
        lo = math.floor((c - r) / dx)
        hi = math.floor((c + r) / dx)
        if lo < 0.0:
            lo = 0.0
        if hi > n - 1.0:
            hi = n - 1.0
        return int(lo), int(hi)

    @jit
    def count_pass(x, y, rad, nx, ny, dx, valid, use_mask, counts):
        for p in range(x.shape[0]):
            px = x[p]
            py = y[p]
            rr = rad[p]
            i0, i1 = cell_range(px, rr, dx, nx)
            j0, j1 = cell_range(py, rr, dx, ny)
            c = 0
            for j in range(j0, j1 + 1):
                ya = j * dx
                yb = (j + 1) * dx
                cy = ya if py < ya else (yb if py > yb else py)
                ddy = py - cy
                for i in range(i0, i1 + 1):
                    if use_mask and valid[j * nx + i] == 0:
                        continue
                    xa = i * dx
                    xb = (i + 1) * dx
                    cx = xa if px < xa else (xb if px > xb else px)
                    ddx = px - cx
                    if ddx * ddx + ddy * ddy < rr * rr:
                        c += 1
            counts[p] = c

    @jit
    def fill_pass(x, y, rad, nx, ny, dx, valid, use_mask, offsets, cells, areas, totals):
        for p in range(x.shape[0]):
            px = x[p]
            py = y[p]
            rr = rad[p]
            i0, i1 = cell_range(px, rr, dx, nx)
            j0, j1 = cell_range(py, rr, dx, ny)
            o = offsets[p]
            tot = 0.0
            for j in range(j0, j1 + 1):
                ya = j * dx
                yb = (j + 1) * dx
                cy = ya if py < ya else (yb if py > yb else py)
                ddy = py - cy
                for i in range(i0, i1 + 1):
                    if use_mask and valid[j * nx + i] == 0:
                        continue
                    xa = i * dx
                    xb = (i + 1) * dx
                    cx = xa if px < xa else (xb if px > xb else px)
                    ddx = px - cx
                    if ddx * ddx + ddy * ddy < rr * rr:
                        rx0 = xa - px
                        rx1 = xb - px
                        ry0 = ya - py
                        ry1 = yb - py
                        fx = max(abs(rx0), abs(rx1))
                        fy = max(abs(ry0), abs(ry1))
                        if fx * fx + fy * fy < rr * rr:
                            av = dx * dx
                        else:
                            av = area(rr, rx0, rx1, ry0, ry1)
                        cells[o] = j * nx + i
                        areas[o] = av
                        tot += av
                        o += 1
            totals[p] = tot

    @jit
    def reduce_pass(offsets, cells, areas, totals, pet, out, packet):
        for p in range(out.shape[0]):
            num = 0.0
            pmin = 1.7976931348623157e308
            pmax = -1.7976931348623157e308
            for e in range(offsets[p], offsets[p + 1]):
                v = pet[cells[e]]
                if (not math.isfinite(v)) or v < 0.0:
                    packet[0] += 1.0
                num += areas[e] * v
                if v < pmin:
                    pmin = v
                if v > pmax:
                    pmax = v
            m = num / totals[p]
            if m < pmin:
                m = pmin
            if m > pmax:
                m = pmax
            out[p] = m
            packet[1] += 1.0

    return {'anti': anti, 'area': area, 'cell_range': cell_range, 'count': count_pass, 'fill': fill_pass,
            'reduce': reduce_pass}


_CORE = {}


def _core(kind):
    '''kind 'python' (reference, identity decorator) or 'numba' (compiled, built lazily; no disk cache).'''
    if kind not in _CORE:
        if kind == 'python':
            _CORE[kind] = _make_core(lambda f: f)
        else:
            try:
                import numba
            except Exception as exc:                      # pragma: no cover - environment dependent
                raise PlantPetBackendError(f'builder / backend "numba" needs the numba package: {exc}') from exc
            _CORE[kind] = _make_core(numba.njit(cache=False))
    return _CORE[kind]


def circle_rectangle_area(r, x0, x1, y0, y1, *, compiled=False):
    '''Exact area of a circle (radius r, centre at the origin) intersected with [x0, x1] x [y0, y1]. Finite reals only.'''
    vals = []
    for name, v in (('r', r), ('x0', x0), ('x1', x1), ('y0', y0), ('y1', y1)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)) \
                or not math.isfinite(float(v)):
            raise PlantPetError(f'{name} must be a finite real, got {v!r}')
        vals.append(float(v))
    if vals[0] <= 0.0:
        raise PlantPetError('r must be > 0')
    return float(_core('numba' if compiled else 'python')['area'](*vals))


# ------------------------------------------------------------------ memory model (pure arithmetic, per-stage peaks)
VALIDATION_ITEMS = ('ids_unique 24P', 'offsets_increasing_mask P', 'row_sum_recompute 8P', 'row_sum_compare 17P')
EXPORT_NOTE = ('host_arrays() is the EXPLICIT full export: it copies every CSR array (resident_bytes on the host; on a CUDA map the '
               'same bytes are copied device -> host); ids / views are the cheap accessors')


def _bbox(builder, bbox_scratch_bytes):
    return int(bbox_scratch_bytes) if builder == 'numpy' else 0


def estimate_canopy_map_bytes(n_plants, n_edges, backend='numpy', *, builder='numpy', bbox_scratch_bytes=0):
    '''Per-stage live bytes of building a canopy map; `build_total_bytes` is the MAX stage (peak), never a padded sum. Inputs
    are borrowed (caller-owned, not counted). Stages: `validate` (x, y subtraction 16P + 3P temporaries), `count` (x, y 16P + counts
    8P + the NumPy builder's largest bounding-box scratch), `offsets` (x, y + counts + offsets), `fill` (x, y + offsets + cells / areas
    16E + totals 8P + scratch), `adopt_validate` (resident arrays + the content-validation temporaries 50P), `upload` (CUDA: host +
    device copies). Resident = ids 8P + offsets 8(P+1) + totals 8P + cells 8E + areas 8E.'''
    P = _traits.strict_int(n_plants, 'n_plants', 0)
    E = _traits.strict_int(n_edges, 'n_edges', 0)
    if backend not in BACKENDS:
        raise PlantPetError(f'backend must be one of {BACKENDS}')
    bb = _bbox(builder, bbox_scratch_bytes)
    resident = 8 * P + 8 * (P + 1) + 8 * P + 16 * E
    stages = {'validate': 19 * P,
              'count': 24 * P + bb,
              'offsets': 16 * P + 8 * P + 8 * (P + 1),
              'fill': 16 * P + 8 * (P + 1) + 16 * E + 8 * P + bb,
              'adopt_validate': resident + 50 * P}
    if backend == 'cuda':
        stages['upload'] = 2 * resident
    return {'stages': stages, 'build_total_bytes': int(max(stages.values())), 'resident_host_bytes': int(resident),
            'device_bytes': int(resident) if backend == 'cuda' else 0, 'export_bytes': int(resident), 'export_note': EXPORT_NOTE,
            'exclusions': EXCLUSIONS}


def estimate_pet_bytes(n_plants, n_edges, backend='numpy'):
    '''Peak bytes of ONE `canopy_pet` call (output included). numpy: gathered values 8E + pmin / pmax / numerator 8P each (the
    numerator buffer IS the returned array; product, division and clamp run in place) + 8P because `ufunc.reduceat` COPIES the
    read-only int64 start-index view while it runs (measured; the sealed offsets are read-only): 8E + 32P. numba: output 8P + packet 16.
    cuda: output 8P + bad counter 4 (device).'''
    P = _traits.strict_int(n_plants, 'n_plants', 0)
    E = _traits.strict_int(n_edges, 'n_edges', 0)
    if backend not in BACKENDS:
        raise PlantPetError(f'backend must be one of {BACKENDS}')
    if backend == 'numpy':
        items = {'gathered_values': 8 * E, 'pmin': 8 * P, 'pmax': 8 * P, 'numerator_is_output': 8 * P,
                 'reduceat_index_copy': 8 * P}
    elif backend == 'numba':
        items = {'output': 8 * P, 'packet': 16}
    else:
        items = {'output': 8 * P, 'bad_counter': 4}
    return {'items': items, 'total_bytes': int(sum(items.values())), 'exclusions': EXCLUSIONS}


# ------------------------------------------------------------------ helpers
def _strict_cap(v, name):
    try:
        return _traits.strict_int(v, name, 0)
    except _traits.CatalogError as exc:
        raise PlantPetError(str(exc)) from exc


def _cupy():
    try:
        import cupy
    except Exception as exc:                              # pragma: no cover - environment dependent
        raise PlantPetBackendError(f'backend "cuda" needs the CuPy package: {exc}') from exc
    return cupy


def _mask_meta(valid_cell_mask, N):
    '''Metadata only (borrowed, no copy, no hash). Returns (uint8 view or None, use_mask).'''
    if valid_cell_mask is None:
        return None, 0
    m = valid_cell_mask
    if type(m) is not np.ndarray or m.dtype != np.bool_ or m.shape != (N,):
        raise PlantPetError(f'valid_cell_mask must be a numpy bool array of shape ({N},) or None (whole domain valid)')
    if not m.flags.c_contiguous:
        raise PlantPetError('valid_cell_mask must be C-contiguous')
    return m.view(np.uint8), 1


def _mask_sha256(mask_u8, chunk=1 << 20):
    '''sha256 of the borrowed mask buffer in 1 MiB chunks (no N-byte duplicate); called only AFTER the lower admission.'''
    h = hashlib.sha256()
    mv = memoryview(mask_u8)
    for lo in range(0, mask_u8.shape[0], chunk):
        h.update(mv[lo:lo + chunk])
    return h.hexdigest()


def _ptr(a):
    return int(a.data.ptr) if hasattr(a, 'data') and hasattr(a.data, 'ptr') else int(a.ctypes.data)


def _fingerprint(arrs):
    return tuple((_ptr(a), tuple(a.shape), a.dtype.str) for a in arrs)


_NAMES = ('ids', 'offsets', 'cells', 'areas_m2', 'area_total_m2')
_DTYPES = (np.int64, np.int64, np.int64, np.float64, np.float64)


class CanopyMap:
    '''Sealed read-only sparse canopy CSR. The constructor needs a module-private seal: build it with `build_canopy_map`,
    `canopy_map_from_state` or the validating `CanopyMap.from_arrays`. Attributes are immutable, subclasses are refused, the
    arrays are private (host arrays read-only; CUDA arrays on the bound device) and `canopy_pet` authenticates the map in O(1)
    (exact type, seal, metadata / dtype / shape / pointer fingerprint, backend / device) before any allocation or kernel; it
    never rehashes the O(E + P) content (immutability, not rehashing, protects it).
    Accessors: `ids` (a copy of ONLY the ids; CUDA: a P-int64 device -> host copy of the ids only), `view(name)` (read-only host view,
    host maps), `host_arrays(max_bytes=None)` (EXPLICIT full export, cost `export_bytes` = every array; see EXPORT_NOTE).'''

    __slots__ = ('_seal', 'domain', 'backend', 'device_id', 'n_cells', 'n_plants', 'n_edges', 'provenance',
                 'resident_bytes', 'cell_area_m2', '_a', '_fp')

    def __init_subclass__(cls, **kw):
        raise TypeError('CanopyMap cannot be subclassed (sealed class)')

    def __init__(self, seal, **kw):
        if seal is not _SEAL:
            raise PlantPetError('CanopyMap cannot be constructed directly: use build_canopy_map, canopy_map_from_state or '
                                'CanopyMap.from_arrays (validated)')
        for k, v in kw.items():
            object.__setattr__(self, k, v)
        object.__setattr__(self, '_seal', _SEAL)

    def __setattr__(self, name, value):
        raise AttributeError('CanopyMap is immutable (build a new map)')

    def __delattr__(self, name):
        raise AttributeError('CanopyMap is immutable')

    def __repr__(self):
        return f'CanopyMap(P={self.n_plants}, E={self.n_edges}, N={self.n_cells}, backend={self.backend!r})'

    @property
    def export_bytes(self):
        return int(self.resident_bytes)

    @property
    def ids(self):
        '''Copy of ONLY the plant ids (no other CSR array is touched or transferred).'''
        _authenticate(self)
        a = self._a[0]
        return np.array(a.get() if self.backend == 'cuda' else a, copy=True)

    def view(self, name):
        '''Read-only host view of one array (host maps only; no copy).'''
        _authenticate(self)
        if self.backend == 'cuda':
            raise PlantPetError('device arrays are not exposed: use host_arrays() (explicit device -> host export)')
        if name not in _NAMES:
            raise PlantPetError(f'unknown array {name!r}; expected one of {_NAMES}')
        return self._a[_NAMES.index(name)].view()

    def host_arrays(self, *, max_bytes=None):
        '''EXPLICIT full export of every CSR array as host copies; cost = `export_bytes` (+ the device -> host transfer of the same
        bytes on a CUDA map). With `max_bytes` the export is admitted first (refused when `export_bytes > max_bytes`).'''
        _authenticate(self)
        if max_bytes is not None and self.export_bytes > _strict_cap(max_bytes, 'max_bytes'):
            raise PlantPetBudgetError('canopy_export', f'{self.export_bytes} bytes exceed max_bytes={max_bytes}')
        return {n: np.array(a.get() if self.backend == 'cuda' else a, copy=True) for n, a in zip(_NAMES, self._a)}

    def to_dict(self):
        return {'law': LAW, 'law_label': LAW_LABEL, 'backend': self.backend, 'n_plants': self.n_plants,
                'n_edges': self.n_edges, 'n_cells': self.n_cells, 'resident_bytes': self.resident_bytes,
                'provenance': dict(self.provenance)}

    def to_device(self, *, device, max_bytes, reserve_bytes=0):
        '''Counted upload of the validated CSR (host + device copies both live during the upload: admitted); returns a NEW sealed map.'''
        _authenticate(self)
        if self.backend == 'cuda':
            raise PlantPetError('the canopy map already lives on a CUDA device')
        mb, rb = _strict_cap(max_bytes, 'max_bytes'), _strict_cap(reserve_bytes, 'reserve_bytes')
        dev_id = _traits.strict_int(device, 'device', 0)
        need = 2 * self.resident_bytes
        if rb + need > mb:
            raise PlantPetBudgetError('canopy_to_device', f'reserve {rb} + host and device copies {need} bytes exceed '
                                                           f'max_bytes={mb}')
        cp = _cupy()
        if int(cp.cuda.runtime.getDevice()) != dev_id:
            raise PlantPetBackendError(f'device {dev_id} is not the current CUDA device; this module never switches devices')
        dev = tuple(cp.asarray(a) for a in self._a)
        return _seal_map(self.domain, dev, 'cuda', dev_id, dict(self.provenance, device=dev_id), self.resident_bytes)

    @classmethod
    def from_arrays(cls, domain, ids, offsets, cells, areas_m2, area_total_m2, *, max_edges, max_bytes, backend='numpy',
                    reserve_bytes=0, device=None, valid_mask_sha256=None, provenance=None):
        '''Validated construction from host arrays (metadata -> caps / admission -> content -> owned copies -> seal).
        Content rules: offsets[0] == 0, strictly increasing (every plant has >= 1 edge), offsets[-1] == E; cells in [0, N); areas
        finite, > 0 and <= the cell area; totals finite, > 0 and equal to the row sums (rtol 1e-12); ids unique.'''
        if not isinstance(domain, _state.Domain):
            raise PlantPetError('domain must be a leaf.state.Domain')
        me, mb, rb = _strict_cap(max_edges, 'max_edges'), _strict_cap(max_bytes, 'max_bytes'), _strict_cap(reserve_bytes, 'reserve_bytes')
        if backend not in BACKENDS:
            raise PlantPetError(f'backend must be one of {BACKENDS}, got {backend!r}')
        if not domain.square_cells:
            raise PlantPetError('non-square cells are not supported')
        if backend == 'cuda':
            if device is None:
                raise PlantPetError('backend "cuda" requires an explicit integer device')
            _traits.strict_int(device, 'device', 0)
        elif device is not None:
            raise PlantPetError('device is only meaningful for backend "cuda"')
        arrs = _check_meta(ids, offsets, cells, areas_m2, area_total_m2)
        P, E = int(arrs[0].shape[0]), int(arrs[2].shape[0])
        if int(arrs[1].shape[0]) != P + 1 or int(arrs[4].shape[0]) != P:
            raise PlantPetError('offsets must have P + 1 entries and area_total_m2 P entries')
        if E > me:
            raise PlantPetBudgetError('canopy_edges', f'{E} edges exceed max_edges={me}')
        est = estimate_canopy_map_bytes(P, E, backend)
        need = max(50 * P + est['resident_host_bytes'], est['stages'].get('upload', 0))
        if rb + need > mb:
            raise PlantPetBudgetError('canopy_from_arrays', f'reserve {rb} + estimated {need} bytes exceed max_bytes={mb}')
        _validate_csr(domain, arrs)
        owned = tuple(np.array(a, copy=True) for a in arrs)
        for a in owned:
            a.flags.writeable = False
        prov = {'law': LAW, 'source': 'from_arrays', 'domain': domain.to_dict(), 'valid_mask_sha256': valid_mask_sha256,
                'n_edges': E, 'max_edges': me, 'max_bytes': mb, 'version': VERSION}
        if provenance:
            prov['caller'] = dict(provenance)
        cm = _seal_map(domain, owned, 'numba' if backend == 'numba' else 'numpy', None, prov, est['resident_host_bytes'])
        if backend == 'cuda':
            cm = cm.to_device(device=int(device), max_bytes=mb, reserve_bytes=rb)
        return cm


_SEAL = object()


def _seal_map(domain, arrs, backend, device, provenance, resident):
    return CanopyMap(_SEAL, domain=domain, backend=backend, device_id=device, n_cells=domain.ncells,
                     n_plants=int(arrs[0].shape[0]), n_edges=int(arrs[2].shape[0]), provenance=provenance,
                     resident_bytes=int(resident), cell_area_m2=float(domain.dx_m) * float(domain.dx_m), _a=tuple(arrs),
                     _fp=_fingerprint(arrs))


def _check_meta(*arrs):
    for name, a, dt in zip(_NAMES, arrs, _DTYPES):
        if type(a) is not np.ndarray or a.dtype != np.dtype(dt) or a.ndim != 1:
            raise PlantPetError(f'{name} must be a 1-D {np.dtype(dt)} numpy.ndarray (no cast, no sequence)')
        if not a.flags.c_contiguous:
            raise PlantPetError(f'{name} must be C-contiguous')
    return arrs


def _validate_csr(domain, arrs):
    ids, off, cells, areas, tot = arrs
    P, E = int(ids.shape[0]), int(cells.shape[0])
    if int(off[0]) != 0 or int(off[-1]) != E:
        raise PlantPetError('offsets must start at 0 and end at the edge count')
    if P:
        if not bool(np.all(off[1:] > off[:-1])):
            raise PlantPetError('offsets must be strictly increasing (every plant needs at least one canopy edge)')
        if np.unique(ids).shape[0] != P:
            raise PlantPetError('plant ids must be unique')
        if not (0 <= int(cells.min()) and int(cells.max()) < domain.ncells):
            raise PlantPetError(f'cell indices must lie in [0, {domain.ncells})')
        cell_area = float(domain.dx_m) * float(domain.dx_m)
        if not (float(areas.min()) > 0.0 and float(areas.max()) <= cell_area * (1 + 1e-12)):
            raise PlantPetError('overlap areas must be finite, > 0 and <= the cell area')
        if not (float(tot.min()) > 0.0 and math.isfinite(float(tot.max()))):
            raise PlantPetError('canopy totals must be finite and > 0')
        s = np.add.reduceat(areas, off[:-1])
        if not bool(np.all(np.abs(s - tot) <= 1e-12 * tot)):
            raise PlantPetError('canopy totals differ from the row sums of the overlap areas')


def _authenticate(c):
    '''O(1) integrity gate: exact class, seal, metadata / dtype / shape / pointer fingerprint, read-only host arrays, device
    binding. Raises before any allocation, kernel or transfer.'''
    if type(c) is not CanopyMap:
        raise PlantPetError(f'a sealed leaf.plant_pet.CanopyMap is required, got {type(c).__name__}')
    try:
        ok = c._seal is _SEAL
        arrs, fp = c._a, c._fp
    except AttributeError:
        ok = False
    if not ok:
        raise PlantPetError('the canopy map is not sealed (forged / uninitialised object refused)')
    if not isinstance(arrs, tuple) or len(arrs) != 5 or _fingerprint(arrs) != fp:
        raise PlantPetError('canopy map arrays do not match their sealed fingerprint (replaced / resized); refused')
    P, E = c.n_plants, c.n_edges
    shapes = ((P,), (P + 1,), (E,), (E,), (P,))
    cuda = c.backend == 'cuda'
    for a, shp, dt in zip(arrs, shapes, _DTYPES):
        if tuple(a.shape) != shp or a.dtype != np.dtype(dt):
            raise PlantPetError('canopy map array shape / dtype differs from its metadata; refused')
        if cuda:
            if type(a) is not _cupy().ndarray:
                raise PlantPetError('a CUDA canopy map must hold CuPy arrays')
            if int(a.device.id) != int(c.device_id):
                raise PlantPetBackendError('canopy map array lives on another CUDA device; refused')
        elif type(a) is not np.ndarray or a.flags.writeable:
            raise PlantPetError('a host canopy map must hold read-only NumPy arrays')
    if c.backend not in BACKENDS or (cuda and c.device_id is None):
        raise PlantPetError('canopy map backend metadata is inconsistent')
    return arrs


# ------------------------------------------------------------------ build
def build_canopy_map(domain, ids, x_m, y_m, canopy_radius_m, *, max_edges, max_bytes, backend='numpy',
                     valid_cell_mask=None, zero_overlap='refuse', builder='compiled', reserve_bytes=0, device=None):
    '''Sealed canopy CSR of circles (x_m / y_m are landscape positions; the domain origin is subtracted). Inputs are borrowed.
    Order: scalar / metadata checks -> lower admission from P -> value checks -> (NumPy builder) bounding-box scratch -> count pass ->
    edge cap and full per-stage admission -> zero-overlap refusal -> mask hash (after admission, chunked) -> fill -> content
    validation -> seal (-> counted upload for CUDA). `reserve_bytes` is the caller's component budget.'''
    if not isinstance(domain, _state.Domain):
        raise PlantPetError('domain must be a leaf.state.Domain')
    me, mb = _strict_cap(max_edges, 'max_edges'), _strict_cap(max_bytes, 'max_bytes')
    rb = _strict_cap(reserve_bytes, 'reserve_bytes')
    if backend not in BACKENDS:
        raise PlantPetError(f'backend must be one of {BACKENDS}, got {backend!r}')
    if builder not in BUILDERS:
        raise PlantPetError(f'builder must be one of {BUILDERS}, got {builder!r}')
    if zero_overlap not in ZERO_OVERLAP_POLICIES:
        raise PlantPetError(f'zero_overlap must be one of {ZERO_OVERLAP_POLICIES} (declared refusal, no fallback), '
                            f'got {zero_overlap!r}')
    if not domain.square_cells:
        raise PlantPetError('non-square cells are not supported (square spacing only); refused before any allocation')
    if backend == 'cuda':
        if device is None:
            raise PlantPetError('backend "cuda" requires an explicit integer device (exact binding, no fallback)')
        _traits.strict_int(device, 'device', 0)
    elif device is not None:
        raise PlantPetError('device is only meaningful for backend "cuda"')
    arrs = {}
    for name, a, dt in (('ids', ids, np.int64), ('x_m', x_m, np.float64), ('y_m', y_m, np.float64),
                        ('canopy_radius_m', canopy_radius_m, np.float64)):
        if type(a) is not np.ndarray or a.dtype != np.dtype(dt) or a.ndim != 1:
            raise PlantPetError(f'{name} must be a 1-D {np.dtype(dt)} numpy.ndarray (no cast, no sequence)')
        if not a.flags.c_contiguous:
            raise PlantPetError(f'{name} must be C-contiguous (a copy would be an unadmitted allocation)')
        arrs[name] = a
    P = int(arrs['ids'].shape[0])
    if any(int(a.shape[0]) != P for a in arrs.values()):
        raise PlantPetError('ids, x_m, y_m and canopy_radius_m must share one length')
    N = domain.ncells
    mask_u8, use_mask = _mask_meta(valid_cell_mask, N)
    low = estimate_canopy_map_bytes(P, 0, backend, builder=builder)
    lb = max(low['stages']['validate'], low['stages']['count'])
    if rb + lb > mb:
        raise PlantPetBudgetError('canopy_preflight', f'reserve {rb} + lower bound {lb} bytes exceed max_bytes={mb}')
    x = arrs['x_m'] - domain.origin_x_m
    y = arrs['y_m'] - domain.origin_y_m
    r = arrs['canopy_radius_m']
    if P:
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.isfinite(r))):
            raise PlantPetError('positions and canopy radii must be finite')
        if float(r.min()) <= 0.0:
            raise PlantPetError('canopy radii must be > 0 (a plant without a canopy has no PET footprint)')
        W, H = float(domain.nx) * float(domain.dx_m), float(domain.ny) * float(domain.dx_m)
        if float(x.min()) < 0.0 or float(x.max()) > W or float(y.min()) < 0.0 or float(y.max()) > H:
            raise PlantPetError('plant positions must lie inside the closed domain')
    dx = float(domain.dx_m)
    bbox = 0
    if builder == 'numpy' and P:
        bbox = int(_roots.numpy_bbox_scratch_bytes(x, y, r, domain.nx, domain.ny, dx)[0])
        pre = estimate_canopy_map_bytes(P, 0, backend, builder=builder, bbox_scratch_bytes=bbox)['stages']['count']
        if rb + pre > mb:
            raise PlantPetBudgetError('canopy_count', f'reserve {rb} + count stage {pre} bytes exceed max_bytes={mb}')
    core = _core('numba' if builder == 'compiled' else 'python')
    counts = np.zeros(P, dtype=np.int64)
    mk = mask_u8 if use_mask else np.zeros(1, np.uint8)
    if P:
        if builder == 'compiled':
            core['count'](x, y, r, domain.nx, domain.ny, dx, mk, bool(use_mask), counts)
        else:
            _numpy_counts(x, y, r, domain, mask_u8, use_mask, counts)
    E = int(counts.sum())
    est = estimate_canopy_map_bytes(P, E, backend, builder=builder, bbox_scratch_bytes=bbox)
    if E > me:
        raise PlantPetBudgetError('canopy_edges', f'{E} edges exceed max_edges={me}')
    if rb + est['build_total_bytes'] > mb:
        raise PlantPetBudgetError('canopy_build', f'reserve {rb} + estimated peak {est["build_total_bytes"]} bytes exceed '
                                                  f'max_bytes={mb}')
    empty = np.flatnonzero(counts == 0)
    if empty.shape[0]:
        raise PlantPetError(f'{int(empty.shape[0])} plant(s) have ZERO valid in-domain canopy overlap (first ids '
                            f'{[int(i) for i in arrs["ids"][empty[:10]]]}); the declared policy is to refuse (no uniform / '
                            'site-mean fallback); fix the geometry or the valid-cell mask')
    mask_sha = _mask_sha256(mask_u8) if use_mask else None            # AFTER the admissions, over the borrowed buffer
    offsets = np.zeros(P + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    del counts
    cells = np.empty(E, dtype=np.int64)
    areas = np.empty(E, dtype=np.float64)
    totals = np.zeros(P, dtype=np.float64)
    if P:
        if builder == 'compiled':
            core['fill'](x, y, r, domain.nx, domain.ny, dx, mk, bool(use_mask), offsets, cells, areas, totals)
        else:
            _numpy_fill(x, y, r, domain, mask_u8, use_mask, offsets, cells, areas, totals, core)
    del x, y
    own = (np.array(arrs['ids'], dtype=np.int64, copy=True), offsets, cells, areas, totals)
    _validate_csr(domain, own)
    for a in own:
        a.flags.writeable = False
    prov = {'law': LAW, 'builder': builder, 'domain': domain.to_dict(), 'valid_mask_sha256': mask_sha,
            'valid_mask': 'whole domain' if mask_sha is None else 'explicit', 'zero_overlap': zero_overlap,
            'max_edges': me, 'max_bytes': mb, 'n_edges': E, 'admission': {'estimate': est, 'reserve_bytes': rb},
            'version': VERSION}
    cmap = _seal_map(domain, own, 'numba' if backend == 'numba' else 'numpy', None, prov, est['resident_host_bytes'])
    if backend == 'cuda':
        cmap = cmap.to_device(device=int(device), max_bytes=mb, reserve_bytes=rb)
    return cmap


def canopy_map_from_state(state, catalog, *, max_edges, max_bytes, backend='numpy', statuses=('live',),
                          live_rows=None, valid_cell_mask=None, zero_overlap='refuse', builder='compiled',
                          reserve_bytes=0, device=None):
    '''Canopy map of the selected plants of a PlantState (default: live, state row order = the root-map row order) with the
    shared allometry radius (`canopy.derive_geometry`, circle canopy).'''
    if not isinstance(state, _state.PlantState):
        raise PlantPetError('state must be a leaf.state.PlantState')
    if not isinstance(catalog, _traits.SpeciesCatalog):
        raise PlantPetError('catalog must be a leaf.traits.SpeciesCatalog')
    if live_rows is None:
        codes = sorted(set(int(c) for c in _state.status_codes(list(statuses))))
        sel = np.flatnonzero(np.isin(state.status, codes)).astype(np.int64)
    else:
        sel = np.asarray(live_rows, dtype=np.int64)
    geo = _canopy.derive_geometry(state, catalog)
    ids = np.ascontiguousarray(state.ids[sel], dtype=np.int64)
    x = np.ascontiguousarray(state.x_m[sel], dtype=np.float64)
    y = np.ascontiguousarray(state.y_m[sel], dtype=np.float64)
    rad = np.ascontiguousarray(geo['radius_m'][sel], dtype=np.float64)
    del geo
    if backend == 'cuda' and device is None:
        device = int(_cupy().cuda.runtime.getDevice())
    return build_canopy_map(state.domain, ids, x, y, rad, max_edges=max_edges, max_bytes=max_bytes, backend=backend,
                            valid_cell_mask=valid_cell_mask, zero_overlap=zero_overlap, builder=builder,
                            reserve_bytes=reserve_bytes, device=device)


def _numpy_selection(px, py, rr, domain, mask_u8, use_mask):
    out = _roots._overlap_mask(px, py, rr, domain.nx, domain.ny, float(domain.dx_m))
    if out is None:
        return None
    ii, jj, m = out
    cells = (jj[:, None] * domain.nx + ii[None, :])
    if use_mask:
        m = m & (mask_u8[cells] != 0)
    return ii, jj, m, cells


def _numpy_counts(x, y, r, domain, mask_u8, use_mask, counts):
    for p in range(x.shape[0]):
        sel = _numpy_selection(float(x[p]), float(y[p]), float(r[p]), domain, mask_u8, use_mask)
        counts[p] = 0 if sel is None else int(np.count_nonzero(sel[2]))


def _numpy_fill(x, y, r, domain, mask_u8, use_mask, offsets, cells, areas, totals, core):
    dx = float(domain.dx_m)
    area = core['area']
    for p in range(x.shape[0]):
        px, py, rr = float(x[p]), float(y[p]), float(r[p])
        sel = _numpy_selection(px, py, rr, domain, mask_u8, use_mask)
        if sel is None:
            continue
        ii, jj, m, cell = sel
        jidx, iidx = np.nonzero(m)               # row-major: j outer, i inner (same order as the compiled builder)
        o = int(offsets[p])
        tot = 0.0
        for j, i in zip(jidx, iidx):
            ci, cj = int(ii[i]), int(jj[j])
            rx0, rx1 = ci * dx - px, (ci + 1) * dx - px
            ry0, ry1 = cj * dx - py, (cj + 1) * dx - py
            fx = max(abs(rx0), abs(rx1))
            fy = max(abs(ry0), abs(ry1))
            av = dx * dx if fx * fx + fy * fy < rr * rr else area(rr, rx0, rx1, ry0, ry1)
            cells[o] = cj * domain.nx + ci
            areas[o] = av
            tot += av
            o += 1
        totals[p] = tot


# ------------------------------------------------------------------ per-plant PET
_CUDA_SRC = r'''
extern "C" {
__device__ __forceinline__ bool fin(double x) { return (x - x) == 0.0; }
__global__ void leaf_canopy_pet(const long long* offsets, const long long* cells, const double* areas,
                                const double* total, const double* pet, double* out, int* bad, const long long P) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x; p < P; p += stride) {
        double num = 0.0, pmin = 1.7976931348623157e308, pmax = -1.7976931348623157e308;
        int nbad = 0;
        for (long long e = offsets[p]; e < offsets[p + 1]; ++e) {
            const double v = pet[cells[e]];
            if (!fin(v) || v < 0.0) nbad += 1;
            num += areas[e] * v;
            if (v < pmin) pmin = v;
            if (v > pmax) pmax = v;
        }
        double m = num / total[p];
        if (m < pmin) m = pmin;
        if (m > pmax) m = pmax;
        out[p] = m;
        if (nbad) atomicAdd(bad, nbad);
    }
}
}
'''
_KERNEL = {}


def _cuda_kernel(device):
    key = int(device)
    if key not in _KERNEL:
        _KERNEL[key] = _cupy().RawKernel(_CUDA_SRC, 'leaf_canopy_pet', options=('-std=c++11', '--fmad=false'))
    return _KERNEL[key]


def canopy_pet(cmap, pet_field, *, max_bytes, reserve_bytes=0):
    '''Per-plant canopy-area-weighted PET (mm / day) [P] on the map's backend plus a small packet and the memory model.
    The map is AUTHENTICATED first (sealed class, fingerprint, device binding; O(1), no content rehash). `pet_field` is the terrain
    field [N] (float64, C-contiguous, plain NumPy for host maps, CuPy on the map's device for CUDA maps; no implicit transfer).
    All metadata, budget and device checks run before any allocation; values on touched cells must be finite and >= 0 (checked by
    reductions on the gathered values on the host; counted by the kernel on CUDA; nothing is returned on failure). The mean is clamped
    to the touched [min, max] so a uniform field is reproduced exactly. `max_bytes` bounds `reserve_bytes` (the caller's other live
    components) + the call peak (`estimate_pet_bytes`).'''
    arrs = _authenticate(cmap)
    mb, rb = _strict_cap(max_bytes, 'max_bytes'), _strict_cap(reserve_bytes, 'reserve_bytes')
    cuda = cmap.backend == 'cuda'
    if cuda:
        cp = _cupy()
        if not isinstance(pet_field, cp.ndarray):
            raise PlantPetError('pet_field must be a cupy.ndarray on the map\'s CUDA device (no implicit host transfer)')
        if int(pet_field.device.id) != int(cmap.device_id):
            raise PlantPetBackendError(f'pet_field lives on CUDA device {int(pet_field.device.id)} but the map is bound to '
                                       f'{cmap.device_id}')
        if int(cp.cuda.runtime.getDevice()) != int(cmap.device_id):
            raise PlantPetBackendError('the current CUDA device differs from the map device; devices are never switched')
    elif type(pet_field) is not np.ndarray:
        raise PlantPetError('pet_field must be a plain numpy.ndarray for a host canopy map (no implicit conversion)')
    if pet_field.dtype != np.float64 or pet_field.ndim != 1 or int(pet_field.shape[0]) != cmap.n_cells:
        raise PlantPetError(f'pet_field must be float64 of shape ({cmap.n_cells},), got {pet_field.dtype} '
                            f'{tuple(pet_field.shape)}')
    if not pet_field.flags.c_contiguous:
        raise PlantPetError('pet_field must be C-contiguous (no implicit copy)')
    be = cmap.backend
    est = estimate_pet_bytes(cmap.n_plants, cmap.n_edges, be)
    if rb + est['total_bytes'] > mb:
        raise PlantPetBudgetError('canopy_pet', f'reserve {rb} + estimated {est["total_bytes"]} bytes exceed max_bytes={mb}')
    ids, off, cells, areas, tot = arrs
    P, E = cmap.n_plants, cmap.n_edges
    nbad = 0
    if cuda:
        out = cp.empty(P, dtype=cp.float64)
        bad = cp.zeros(1, dtype=cp.int32)
        if P:
            nb = max(1, min(-(-P // CUDA_THREADS), CUDA_MAX_BLOCKS))
            _cuda_kernel(cmap.device_id)((nb,), (CUDA_THREADS,), (off, cells, areas, tot, pet_field, out, bad, np.int64(P)))
        nbad = int(bad.get()[0])
    elif be == 'numba':
        out = np.empty(P, dtype=np.float64)
        if P:
            packet = np.zeros(2)
            _core('numba')['reduce'](off, cells, areas, tot, pet_field, out, packet)
            nbad = int(packet[0])
    elif P:
        v = pet_field[cells]                                           # 8E
        lo, hi = float(v.min()), float(v.max())
        if not (lo >= 0.0 and math.isfinite(hi)):                      # NaN fails `lo >= 0`; no temporaries
            nbad = 1
        else:
            starts = off[:-1]
            pmin = np.minimum.reduceat(v, starts)                      # 8P
            pmax = np.maximum.reduceat(v, starts)                      # 8P
            v *= areas                                                 # in place
            out = np.add.reduceat(v, starts)                           # 8P: the numerator buffer is the output (+8P index copy)
            out /= tot
            np.minimum(out, pmax, out=out)
            np.maximum(out, pmin, out=out)
    else:
        out = np.empty(0, dtype=np.float64)
    if nbad:
        raise PlantPetError('touched terrain PET value(s) are non-finite or negative; nothing returned')
    return {'pet_mm_per_day': out, 'packet': {'bad_pet_values': 0, 'n_plants': P}, 'memory': est,
            'law': LAW, 'units': 'mm / day'}


__all__ = ['VERSION', 'LAW', 'ZERO_OVERLAP_POLICIES', 'PlantPetError', 'PlantPetBudgetError', 'PlantPetBackendError',
           'circle_rectangle_area', 'estimate_canopy_map_bytes', 'estimate_pet_bytes', 'CanopyMap',
           'build_canopy_map', 'canopy_map_from_state', 'canopy_pet']
