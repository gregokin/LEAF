'''leaf.periodic: common periodic-boundary geometry helpers (UNQUALIFIED; no water, no ecology, no new law).

One `leaf.state.Domain` carries an immutable per-axis declaration `periodic=(px, py)`. This module holds the geometry that every
consumer shares so that exclusion placement, measured cover, canopy-area PET weights and root footprints agree on what "wrapped"
means. Nothing here creates plants, images or coordinate shifts: a plant is ONE record with ONE centre in the closed domain, and
all wrapping is arithmetic on distances / index ranges.

Conventions (all lengths in the domain-local frame, i.e. landscape position minus the domain origin)
  * a periodic axis of n cells of size h has length L = n * h; the point L is the same place as the point 0. The exact upper
    endpoint is recognised by EQUALITY on the landscape coordinate (`x == Domain.x_max_m`), never by comparing `x - origin` with L
    (which rounds on decimal spacings / origins) and never by a tolerance; `localize_inplace` / `canonicalize_local` map it to 0.0;
  * the shortest separation of two points on such an axis is the MINIMUM-IMAGE distance d - L * rint(d / L); on a nonperiodic axis
    it is the ordinary difference. Separations on the two axes are independent, so the minimum over all 2-D images of the squared
    distance is the sum of the per-axis minimum-image squares (this is why a circle and all its images are handled without
    materialising a single image);
  * the union of a circle with all its images is the circle clipped to the nearest-image (Voronoi) rectangle of its centre,
    |dx| <= Lx / 2 on a periodic x axis and |dy| <= Ly / 2 on a periodic y axis (a nonperiodic axis is not clipped by this
    rule). Overlapped images are therefore never counted twice, for ANY radius.

Footprint kernels (`kernels`): one Python reference implementation compiled lazily with Numba for production (`kind='numba'`) or run
as plain Python (`kind='python'`, exact but slow reference). Cells are SQUARE (dx == dy; the callers refuse anything else). For each
plant the wrapped candidate cells come from the unwrapped index range of the centre +- min(r, L / 2) per periodic axis (nonperiodic
axes: the legacy clipped range). A wrapped cell is REACHED iff the closest point of the closed square, taken from the NEAREST image
of the centre, is strictly inside the circle (the legacy positive-overlap predicate; tangency excluded). Each physical plant-cell
pair is emitted ONCE, in unwrapped footprint order (rows from the south edge of the unwrapped footprint, columns from its west edge;
NOT sorted by cell index across a seam). The canopy fill sums the exact circle-rectangle areas (`plant_pet`'s integrator, passed in)
over every wrapped fragment of that cell, clipped to the nearest-image rectangle when the circle is wider than half a periodic span.
'''
from __future__ import annotations

import math

import numpy as np

VERSION = '0.1.0'
BOUNDARY_NOTE_PERIODIC = ('periodic boundaries (per axis, see Domain.periodic): canopies and roots WRAP across a periodic seam while each '
                          'plant stays ONE record with ONE ID; measured cover and the canopy-PET support (CanopyMap areas / denominator) '
                          'are physical torus unions (overlapped self-images are never counted twice); nonperiodic axes keep the finite '
                          'closed clipped behaviour. The biological DEMAND area stays the nominal allometric pi r^2 (unchanged law), which '
                          'may exceed the physical union for a circle wider than half a periodic span. Public periodic centres are '
                          'half-open [0, L): the exact upper endpoint is exported as 0.')


class PeriodicGeometryError(ValueError):
    '''A periodic-geometry request that cannot be honoured (raised before any allocation / placement).'''


# ------------------------------------------------------------------ small pure helpers
def spans(domain):
    '''(Lx, Ly) in metres: the periodic lengths nx * dx and ny * dy.'''
    return float(domain.width_m), float(domain.height_m)


def min_image(delta, length):
    '''Minimum-image separation of an array / scalar difference on an axis of the given length.'''
    d = np.asarray(delta, dtype=np.float64)
    return d - length * np.rint(d / length)


def boundary_record(domain):
    '''Small JSON-able boundary description used in provenance / reports / exports.'''
    return {'periodic': [bool(domain.periodic[0]), bool(domain.periodic[1])], 'edges': domain.edge_convention,
            'cell_sampling': domain.cell_sampling_convention, 'one_record_per_plant': True,
            'note': BOUNDARY_NOTE_PERIODIC if domain.any_periodic else 'finite closed non-periodic domain (default)'}


def self_overlapping(domain, diameter_m):
    '''Boolean array: the canopy of this diameter overlaps its OWN image on a periodic axis (diameter > span; tangency allowed).'''
    d = np.asarray(diameter_m, dtype=np.float64)
    bad = np.zeros(d.shape, dtype=bool)
    lx, ly = spans(domain)
    if domain.periodic[0]:
        bad |= d > lx
    if domain.periodic[1]:
        bad |= d > ly
    return bad


def check_no_self_overlap(domain, diameter_m, what='canopy'):
    '''Refuse (PeriodicGeometryError) when any diameter exceeds a periodic span: no silent rescale, drop or buffer.'''
    d = np.asarray(diameter_m, dtype=np.float64)
    if d.size and domain.any_periodic:
        bad = self_overlapping(domain, d)
        if bool(bad.any()):
            lx, ly = spans(domain)
            raise PeriodicGeometryError(
                f'{int(bad.sum())} {what} diameter(s) (largest {float(d.max()):.6g} m) exceed a periodic span '
                f'(Lx={lx:.6g} m periodic={domain.periodic[0]}, Ly={ly:.6g} m periodic={domain.periodic[1]}): the {what} would '
                'overlap its own periodic image, which a no-overlap placement cannot represent; nothing is rescaled, dropped or buffered')


def _axes(domain, x, y):
    return ((domain.periodic[0], x, domain.origin_x_m, domain.x_max_m, float(domain.width_m)),
            (domain.periodic[1], y, domain.origin_y_m, domain.y_max_m, float(domain.height_m)))


def _canonical_axis(upper, local, length):
    '''Half-open canonical form of a periodic domain-local coordinate: the exact-upper-endpoint entries (`upper`, decided by equality on
    the LANDSCAPE value) become 0.0; any other entry that `x - origin` rounded to >= L becomes the largest double below L (a real
    just-inside position is never moved onto the seam by a tolerance).'''
    local[upper] = 0.0
    over = local >= length
    if bool(over.any()):
        local[over] = np.nextafter(length, 0.0)


def localize_inplace(domain, x, y):
    '''Convert OWNED float64 landscape copies x, y to domain-local coordinates IN PLACE (x - origin). On a periodic axis the exact upper
    endpoint x == x_max_m is canonicalised to 0.0 (it is the lower edge) and every local value is kept in [0, L); nonperiodic axes are
    the plain subtraction (unchanged). Transient: one boolean mask per periodic axis plus one for the rounding guard (<= 2 bytes per
    plant and axis, to be admitted by the caller). Returns (x, y).'''
    for per, v, origin, vmax, length in _axes(domain, x, y):
        upper = (v == vmax) if per else None
        v -= origin
        if per:
            _canonical_axis(upper, v, length)
    return x, y


def canonicalize_local(domain, x_m, y_m, xl, yl):
    '''Same canonicalisation for local copies `xl`, `yl` (= x_m - origin, modified in place) when the landscape arrays `x_m`, `y_m` are
    still available. Nonperiodic axes are untouched.'''
    for per, m, loc in ((domain.periodic[0], x_m, xl), (domain.periodic[1], y_m, yl)):
        if per:
            vmax, length = (domain.x_max_m, float(domain.width_m)) if loc is xl else (domain.y_max_m, float(domain.height_m))
            _canonical_axis(m == vmax, loc, length)
    return xl, yl


# ------------------------------------------------------------------ exclusion-placement spatial bins
def bin_layout(domain, size):
    '''Per-axis hashed-bin layout for conflict queries between circles whose centre distance can matter only below `size` (>= the
    largest diameter). A nonperiodic axis keeps the legacy bin edge `size`. A periodic axis uses nb = max(1, floor(L / size)) bins of
    EXACT width L / nb (>= size), so the bins tile the span with no short last bin and modular neighbours are exact for any ratio
    of L to size. Returns [(nb or None, width), (nb or None, width)].'''
    lay = []
    for per, length in ((domain.periodic[0], domain.width_m), (domain.periodic[1], domain.height_m)):
        if per:
            nb = max(1, int(length // size))
            lay.append((nb, length / nb))
        else:
            lay.append((None, float(size)))
    return lay


def bin_key(p, origin, axis):
    '''Integer bin of coordinate p on one axis of a `bin_layout`.'''
    nb, width = axis
    k = int((p - origin) // width)
    return k % nb if nb is not None else k


def bin_neighbours(k, axis):
    '''Distinct neighbouring bins (k - 1, k, k + 1; modulo nb on a periodic axis, deduplicated for nb <= 2).'''
    nb, _ = axis
    if nb is None:
        return (k - 1, k, k + 1)
    return tuple(sorted({(k - 1) % nb, k % nb, (k + 1) % nb}))


# ------------------------------------------------------------------ footprint kernels (Python reference + Numba)
def _make_kernels(jit, area):
    @jit
    def fold(p, length):
        q = p - length * math.floor(p / length)
        if q >= length or q < 0.0:
            q = 0.0
        return q

    @jit
    def axis_range(p, r, h, n, per, length):
        '''(lo, hi, dup): unwrapped cell index range of the footprint on one axis; dup is True when it holds n + 1 cells (first and
        last are the same wrapped cell).'''
        if per:
            ext = r
            if ext > 0.5 * length:
                ext = 0.5 * length
            plo = int(math.floor((p - ext) / h))
            phi = int(math.floor((p + ext) / h))
            return plo, phi, (phi - plo + 1) > n
        flo = math.floor((p - r) / h)
        fhi = math.floor((p + r) / h)
        if flo < 0.0:
            flo = 0.0
        if fhi > n - 1.0:
            fhi = n - 1.0
        return int(flo), int(fhi), False

    @jit
    def axis_gap(p, iw, h, per, length):
        '''Signed distance from the (nearest image of the) centre to the closest point of the closed cell interval iw.'''
        q = p
        if per:
            m = math.floor((p - (iw + 0.5) * h) / length + 0.5)
            q = p - m * length
        a = iw * h
        b = (iw + 1) * h
        c = q
        if c < a:
            c = a
        elif c > b:
            c = b
        return q - c

    @jit
    def wrap_index(i, n, per):
        if per:
            return ((i % n) + n) % n
        return i

    @jit
    def count_pass(x, y, rad, nx, ny, dx, pxp, pyp, valid, use_mask, counts):
        lx = nx * dx
        ly = ny * dx
        for p in range(x.shape[0]):
            px = x[p]
            py = y[p]
            if pxp:
                px = fold(px, lx)
            if pyp:
                py = fold(py, ly)
            rr = rad[p]
            i0, i1, dupx = axis_range(px, rr, dx, nx, pxp, lx)
            j0, j1, dupy = axis_range(py, rr, dx, ny, pyp, ly)
            if dupx:
                i1 -= 1
            if dupy:
                j1 -= 1
            c = 0
            for j in range(j0, j1 + 1):
                jw = wrap_index(j, ny, pyp)
                ddy = axis_gap(py, jw, dx, pyp, ly)
                for i in range(i0, i1 + 1):
                    iw = wrap_index(i, nx, pxp)
                    if use_mask and valid[jw * nx + iw] == 0:
                        continue
                    ddx = axis_gap(px, iw, dx, pxp, lx)
                    if ddx * ddx + ddy * ddy < rr * rr:
                        c += 1
            counts[p] = c

    @jit
    def fill_roots(x, y, rad, nx, ny, dx, pxp, pyp, offsets, cells, weights):
        lx = nx * dx
        ly = ny * dx
        for p in range(x.shape[0]):
            k = offsets[p]
            cnt = offsets[p + 1] - k
            if cnt == 0:
                continue
            w = 1.0 / cnt
            px = x[p]
            py = y[p]
            if pxp:
                px = fold(px, lx)
            if pyp:
                py = fold(py, ly)
            rr = rad[p]
            i0, i1, dupx = axis_range(px, rr, dx, nx, pxp, lx)
            j0, j1, dupy = axis_range(py, rr, dx, ny, pyp, ly)
            if dupx:
                i1 -= 1
            if dupy:
                j1 -= 1
            for j in range(j0, j1 + 1):
                jw = wrap_index(j, ny, pyp)
                ddy = axis_gap(py, jw, dx, pyp, ly)
                for i in range(i0, i1 + 1):
                    iw = wrap_index(i, nx, pxp)
                    ddx = axis_gap(px, iw, dx, pxp, lx)
                    if ddx * ddx + ddy * ddy < rr * rr:
                        cells[k] = jw * nx + iw
                        weights[k] = w
                        k += 1

    @jit
    def fragment_area(rr, px, py, i, j, dx, lx, ly, pxp, pyp):
        '''Exact area of the circle within the (unwrapped) cell (i, j), clipped to the nearest-image rectangle on periodic axes when
        the circle is wider than half the span (otherwise unclipped: bit-identical to the legacy fill).'''
        rx0 = i * dx - px
        rx1 = (i + 1) * dx - px
        ry0 = j * dx - py
        ry1 = (j + 1) * dx - py
        clipx = False
        clipy = False
        if pxp and rr > 0.5 * lx:
            hx = 0.5 * lx
            if rx0 < -hx:
                rx0 = -hx
                clipx = True
            if rx1 > hx:
                rx1 = hx
                clipx = True
        if pyp and rr > 0.5 * ly:
            hy = 0.5 * ly
            if ry0 < -hy:
                ry0 = -hy
                clipy = True
            if ry1 > hy:
                ry1 = hy
                clipy = True
        if not (rx1 > rx0) or not (ry1 > ry0):
            return 0.0
        fx = max(abs(rx0), abs(rx1))
        fy = max(abs(ry0), abs(ry1))
        if fx * fx + fy * fy < rr * rr:
            wx = dx
            wy = dx
            if clipx:
                wx = rx1 - rx0
            if clipy:
                wy = ry1 - ry0
            return wx * wy
        return area(rr, rx0, rx1, ry0, ry1)

    @jit
    def fill_canopy(x, y, rad, nx, ny, dx, pxp, pyp, valid, use_mask, offsets, cells, areas, totals):
        lx = nx * dx
        ly = ny * dx
        for p in range(x.shape[0]):
            px = x[p]
            py = y[p]
            if pxp:
                px = fold(px, lx)
            if pyp:
                py = fold(py, ly)
            rr = rad[p]
            i0, i1, dupx = axis_range(px, rr, dx, nx, pxp, lx)
            j0, j1, dupy = axis_range(py, rr, dx, ny, pyp, ly)
            if dupx:
                i1 -= 1
            if dupy:
                j1 -= 1
            o = offsets[p]
            tot = 0.0
            for j in range(j0, j1 + 1):
                jw = wrap_index(j, ny, pyp)
                ddy = axis_gap(py, jw, dx, pyp, ly)
                for i in range(i0, i1 + 1):
                    iw = wrap_index(i, nx, pxp)
                    if use_mask and valid[jw * nx + iw] == 0:
                        continue
                    ddx = axis_gap(px, iw, dx, pxp, lx)
                    if ddx * ddx + ddy * ddy < rr * rr:
                        av = fragment_area(rr, px, py, i, j, dx, lx, ly, pxp, pyp)
                        if dupx and i == i0:                    # the cell is also met at i0 + nx (same wrapped cell)
                            av += fragment_area(rr, px, py, i + nx, j, dx, lx, ly, pxp, pyp)
                            if dupy and j == j0:
                                av += fragment_area(rr, px, py, i + nx, j + ny, dx, lx, ly, pxp, pyp)
                        if dupy and j == j0:
                            av += fragment_area(rr, px, py, i, j + ny, dx, lx, ly, pxp, pyp)
                        cells[o] = jw * nx + iw
                        areas[o] = av
                        tot += av
                        o += 1
            totals[p] = tot

    return {'count': count_pass, 'fill_roots': fill_roots, 'fill_canopy': fill_canopy}


def _no_area(r, x0, x1, y0, y1):
    return 0.0


_KERNELS = {}


def kernels(kind, area=None):
    '''Footprint kernels (dict with count / fill_roots / fill_canopy). kind 'python' = exact pure-Python reference (identity decorator,
    slow), 'numba' = compiled lazily (no disk cache). `area` is the circle-rectangle integrator of the SAME kind (plant_pet's
    core['area']); roots do not need it. The result is cached per (kind, area).'''
    if kind not in ('python', 'numba'):
        raise PeriodicGeometryError(f'kernel kind must be "python" or "numba", got {kind!r}')
    key = (kind, id(area))
    if key not in _KERNELS:
        if kind == 'python':
            _KERNELS[key] = _make_kernels(lambda f: f, _no_area if area is None else area)
        else:
            try:
                import numba
            except Exception as exc:                  # pragma: no cover - environment dependent
                raise PeriodicGeometryError(f'the compiled periodic footprint kernels need the numba package: {exc}') from exc
            ar = numba.njit(cache=False)(_no_area) if area is None else area
            _KERNELS[key] = _make_kernels(numba.njit(cache=False), ar)
        _KERNELS[key]['_area_ref'] = area      # keep the integrator alive so id() stays unique
    return _KERNELS[key]


__all__ = ['VERSION', 'BOUNDARY_NOTE_PERIODIC', 'PeriodicGeometryError', 'spans', 'min_image', 'boundary_record',
           'self_overlapping', 'check_no_self_overlap', 'localize_inplace', 'canonicalize_local', 'bin_layout', 'bin_key', 'bin_neighbours', 'kernels']
