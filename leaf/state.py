'''leaf.state: explicit domain, canonical structure-of-arrays plant population and the plant-state CSV format.

Conventions (LEAF Phase 0)
  Domain      nx, ny cells of dx_m x dy_m metres, origin (origin_x_m, origin_y_m) at the south-west corner
              in the LANDSCAPE frame. Raster axes are (ny, nx): row = y (north), col = x (east). Cell
              centres are origin + (col + 0.5) * dx, origin + (row + 0.5) * dy. The domain is finite,
              closed [x_min, x_max] x [y_min, y_max]. DEFAULT: NON-periodic (nothing wraps). Domain objects are
              immutable (frozen after construction).
  Boundaries  ONE immutable per-axis declaration `Domain(..., periodic=(px, py))` (x then y; strict bools, default
              (False, False)). A periodic axis is a torus axis of length nx*dx (ny*dy): plant geometry (canopy,
              roots, exclusion, cover) wraps across that seam while every plant stays ONE record with ONE ID and ONE
              centre inside the closed domain (no image records, no coordinate shifts). Ordinary position input is still
              validated against the closed domain on every axis (nothing is silently normalised); computed coordinates
              (e.g. new daughter positions) are wrapped explicitly with `Domain.wrap_positions`. Mixed axes are
              supported. A nonperiodic Domain serialises exactly as before (no new key); a periodic one adds
              `periodic: [bool, bool]` and per-axis `edges` / `cell_sampling` texts. See docs/periodic-boundaries.md.
  Positions   x_m, y_m are landscape-frame coordinates in the SAME frame as the origin (NOT origin-relative);
              the MAPLE view subtracts the origin when it publishes domain-local centres.
  Sampling    CELL_SAMPLING_CONVENTION = 'containing_cell_closed_upper': a point samples the cell that
              contains it, col = floor((x - origin_x) / dx) clipped to [0, nx-1], so the upper domain edge
              x == x_max belongs to the last column. No interpolation, no transpose. On a PERIODIC axis the upper
              edge is the SAME place as the lower edge: the EXACT endpoint x == x_max_m (equality on the landscape
              coordinate, robust on decimal spacings / origins) samples column 0 (CELL_SAMPLING_PERIODIC); any other
              position keeps floor((x - origin_x) / dx) clipped to [0, nx-1]; nonperiodic axes keep the closed-upper rule.
  Status      int8 codes 0 live, 1 dead_standing, 2 removed; the CSV carries the names. Removed/dead
              identities are retained forever (never compacted).
  Biomass     agb_dry_kg is THE aboveground dry mass state; seasonal_agb_dry_kg (E) is a subcomponent,
              0 <= E <= B, not an independent tissue pool. Live/dead_standing plants need B > 0; removed
              plants may have B = 0.
  Age         age_days advances for LIVE plants only when the caller advances the calendar
              (advance_time); dead/removed plants keep the age they had when their status changed. Reaching
              a species life expectancy never changes status here. Unresolved initial placements cannot be
              advanced in time (resolve first).
  SURFACE     an initial CSV (handwritten plain list, or a file whose companion metadata declares
              "initial": true) may carry the literal token SURFACE in establishment_ground_m. It is resolved
              ONCE from the caller's initial surface (resolve_establishment) and the concrete elevation is saved
              thereafter; later erosion/deposition never rewrites it. A file whose companion metadata does not
              declare "initial": true must carry concrete anchors (refused BEFORE any surface is sampled).
              Direct construction requires the explicit `establishment_unresolved` mask; a bare NaN anchor is
              an error.

Ownership: the canonical arrays are owned copies and are READ-ONLY; every supported mutation
(resolve_establishment, advance_time, update_plants, adopt) validates a candidate and then replaces whole
arrays, so caller-owned inputs never alias the state and the by-ID index can never go stale.

PLAIN PLANT CSV CONVENTION (project rule; do not embed metadata in a plant file)
  A LEAF plant file is a PLAIN CSV: line 1 is the nine-column header, every following line is one plant with
  exactly nine fields. No "# ..." metadata line, no JSON, no comments, in ANY file LEAF writes (initial input
  written by `python -m leaf init|generate`, snapshots and final outputs, checkpoint plants.csv, preview tables).
  Saved-file metadata lives in a SEPARATE human-readable companion JSON named "<stem>.metadata.json" in the same
  directory (initial_plants.csv -> initial_plants.metadata.json, plants.csv -> plants.metadata.json). There is no
  option that writes embedded metadata.

  header               plant_id,species,x_m,y_m,establishment_ground_m,agb_dry_kg,seasonal_agb_dry_kg,age_days,status
  rows                 floats written with repr() -> exact FP64 round trip. Rows are standard strict CSV
                       (csv.reader(strict=True)): ordinary quoted fields such as "0.12" or "live" are
                       accepted (spreadsheet writers quote freely); malformed/unterminated quotes fail.

  companion metadata   schema "leaf.plant_state_metadata" version 1: time_days, n_plants, catalog_hash,
                       species_names, domain, units, coordinate_frame, initial / establishment_resolved (SURFACE
                       safety), provenance and a binding to the CSV bytes (csv.sha256 = "sha256:<hex>" of the exact
                       file bytes, csv.n_bytes, csv.columns; csv.file is informational only). `read_plant_csv`
                       loads the companion automatically and REFUSES a CSV whose bytes do not match it (stale or
                       partially written pair), a wrong catalog / domain / time / row count, or an undeclared
                       SURFACE token. Without a companion the file is a handwritten plain list: the caller supplies
                       the domain (and optionally time_days / a surface). `plant_csv_text` always returns plain CSV;
                       `plant_metadata(state, csv_text, ...)` builds the matching companion dict, which
                       `parse_plant_csv(..., metadata=dict)` can verify in memory.
  two-file writes      `write_plant_files` validates and serialises BOTH files first, preflights BOTH destinations
                       (an existing CSV or an existing companion is never replaced unless overwrite=True, and an
                       overwrite never replaces a file that is not a LEAF plant-state companion), publishes the
                       companion FIRST and the CSV SECOND (each with its own atomic temp-file + fsync + hard-link /
                       os.replace), and removes / restores the companion if the CSV publication fails. Two renames
                       are NOT one atomic operation: if the process dies between them an orphan companion without
                       a CSV (reads fail: no CSV) or, on overwrite, a new companion beside the old CSV (reads fail:
                       hash mismatch) remains. A reader never silently accepts a mismatched pair.
  legacy reads         a first line "# {json}" (the pre-companion embedded form, schema "leaf.plant_state_csv"
                       version 1) is still READ for old files, with the old checks; it cannot be combined with a
                       companion, and LEAF never writes it.
Reads and updates are all-or-nothing: every row is validated before a state exists or is replaced.
'''
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile

import numpy as np

from leaf import traits as _traits

CSV_SCHEMA = 'leaf.plant_state_csv'          # LEGACY embedded "# {json}" metadata line (read-only compatibility)
CSV_SCHEMA_VERSION = 1
METADATA_SCHEMA = 'leaf.plant_state_metadata'  # companion "<stem>.metadata.json" of a plain plant CSV
METADATA_SCHEMA_VERSION = 1
METADATA_SUFFIX = '.metadata.json'
CSV_COLUMNS = ('plant_id', 'species', 'x_m', 'y_m', 'establishment_ground_m', 'agb_dry_kg',
               'seasonal_agb_dry_kg', 'age_days', 'status')
SURFACE_TOKEN = 'SURFACE'
LIVE, DEAD_STANDING, REMOVED = 0, 1, 2
STATUS_CODES = {'live': LIVE, 'dead_standing': DEAD_STANDING, 'removed': REMOVED}
STATUS_NAMES = {v: k for k, v in STATUS_CODES.items()}
CELL_SAMPLING_CONVENTION = 'containing_cell_closed_upper'
AXES_CONVENTION = '(ny, nx): row = y north, col = x east'
EDGE_CONVENTION = 'finite closed domain, non-periodic'
CELL_SAMPLING_PERIODIC = ('containing_cell_periodic_axes_upper_edge_equals_lower_edge;'
                          'nonperiodic_axes_closed_upper')


def edge_convention(periodic=(False, False)):
    '''Text of the `edges` metadata for a per-axis periodic declaration (legacy text when nothing is periodic).'''
    px, py = periodic
    if not (px or py):
        return EDGE_CONVENTION
    wrapped = '+'.join(a for a, p in (('x', px), ('y', py)) if p)
    fixed = '+'.join(a for a, p in (('x', px), ('y', py)) if not p)
    txt = f'periodic (torus) in {wrapped}'
    if fixed:
        txt += f'; finite closed non-periodic in {fixed}'
    return txt


COORDINATE_FRAME = 'landscape frame (same frame as origin_x_m/origin_y_m; NOT origin-relative)'
UNITS = {'x_m': 'm, landscape-frame easting (same frame as origin_x_m; not origin-relative)',
         'y_m': 'm, landscape-frame northing (same frame as origin_y_m; not origin-relative)',
         'establishment_ground_m': 'm elevation at establishment',
         'agb_dry_kg': 'kg dry aboveground mass per individual',
         'seasonal_agb_dry_kg': 'kg dry aboveground mass (subcomponent of agb_dry_kg, 0 <= E <= B)',
         'age_days': 'days', 'time_days': 'days (model calendar)'}
_FLOAT_RE = re.compile(r'[+-]?(?:\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)')
_INT_RE = re.compile(r'[+-]?\d+')
INT64_MIN, INT64_MAX = -(2 ** 63), 2 ** 63 - 1


class StateError(ValueError):
    '''Any plant-state validation / I/O content error (raised before any mutation).'''


# ------------------------------------------------------------------ scalar helpers
def _finite_float(v, name):
    if not _traits.is_real(v):
        raise StateError(f'{name} must be a real number (not bool/str/None), got {v!r}')
    v = float(v)
    if not math.isfinite(v):
        raise StateError(f'{name} must be finite, got {v}')
    return v


def _positive_float(v, name):
    v = _finite_float(v, name)
    if v <= 0.0:
        raise StateError(f'{name} must be > 0, got {v}')
    return v


def _positive_int(v, name):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
        raise StateError(f'{name} must be an integer, got {v!r}')
    if int(v) < 1:
        raise StateError(f'{name} must be >= 1, got {v}')
    return int(v)


# ------------------------------------------------------------------ domain
def check_periodic(periodic, name='periodic'):
    '''Strict per-axis boundary declaration -> (bool, bool) in x / y order. Only a list / tuple of exactly two real booleans
    (bool or numpy.bool_) is accepted: no truthiness of ints / strings / None, no scalar broadcast.'''
    if not isinstance(periodic, (list, tuple)) or len(periodic) != 2 \
            or any(not isinstance(v, (bool, np.bool_)) for v in periodic):
        raise StateError(f'{name} must be a [x, y] pair of booleans (true = periodic axis), got {periodic!r}')
    return (bool(periodic[0]), bool(periodic[1]))


class Domain:
    '''Finite, closed rectangular raster domain (immutable); each axis is nonperiodic (default) or periodic.
    See the module docstring.'''

    __slots__ = ('nx', 'ny', 'dx_m', 'dy_m', 'origin_x_m', 'origin_y_m', 'periodic')
    _DICT_KEYS = ('nx', 'ny', 'dx_m', 'dy_m', 'origin_x_m', 'origin_y_m')
    _DOC_KEYS = {'axes': AXES_CONVENTION, 'cell_sampling': CELL_SAMPLING_CONVENTION, 'edges': EDGE_CONVENTION,
                 'coordinate_frame': COORDINATE_FRAME}

    def __init__(self, nx, ny, dx_m, dy_m=None, origin_x_m=0.0, origin_y_m=0.0, *, periodic=(False, False)):
        s = object.__setattr__
        s(self, 'nx', _positive_int(nx, 'nx'))
        s(self, 'ny', _positive_int(ny, 'ny'))
        s(self, 'dx_m', _positive_float(dx_m, 'dx_m'))
        s(self, 'dy_m', self.dx_m if dy_m is None else _positive_float(dy_m, 'dy_m'))
        s(self, 'origin_x_m', _finite_float(origin_x_m, 'origin_x_m'))
        s(self, 'origin_y_m', _finite_float(origin_y_m, 'origin_y_m'))
        s(self, 'periodic', check_periodic(periodic))
        for name in ('width_m', 'height_m', 'x_max_m', 'y_max_m', 'area_m2'):
            if not math.isfinite(getattr(self, name)):
                raise StateError(f'domain {name} is not finite')
        if self.area_m2 <= 0.0:
            raise StateError('domain area must be positive')

    def __setattr__(self, name, value):
        raise AttributeError('Domain is immutable')

    def __delattr__(self, name):
        raise AttributeError('Domain is immutable')

    # ---- geometry
    @property
    def shape(self):
        return (self.ny, self.nx)

    @property
    def ncells(self):
        return self.nx * self.ny

    @property
    def width_m(self):
        return self.nx * self.dx_m

    @property
    def height_m(self):
        return self.ny * self.dy_m

    @property
    def x_min_m(self):
        return self.origin_x_m

    @property
    def y_min_m(self):
        return self.origin_y_m

    @property
    def x_max_m(self):
        return self.origin_x_m + self.width_m

    @property
    def y_max_m(self):
        return self.origin_y_m + self.height_m

    @property
    def area_m2(self):
        return self.width_m * self.height_m

    @property
    def square_cells(self):
        return self.dx_m == self.dy_m

    @property
    def periodic_x(self):
        return self.periodic[0]

    @property
    def periodic_y(self):
        return self.periodic[1]

    @property
    def any_periodic(self):
        return bool(self.periodic[0] or self.periodic[1])

    @property
    def edge_convention(self):
        return edge_convention(self.periodic)

    @property
    def cell_sampling_convention(self):
        return CELL_SAMPLING_PERIODIC if self.any_periodic else CELL_SAMPLING_CONVENTION

    def cell_centers(self):
        '''(x_centers[nx], y_centers[ny]) in the landscape frame.'''
        xs = self.origin_x_m + (np.arange(self.nx, dtype=np.float64) + 0.5) * self.dx_m
        ys = self.origin_y_m + (np.arange(self.ny, dtype=np.float64) + 0.5) * self.dy_m
        return xs, ys

    # ---- serialization / equality
    def to_dict(self):
        '''JSON-able description. A fully nonperiodic domain serialises EXACTLY as before (no new key, legacy texts); a domain
        with a periodic axis adds `periodic: [px, py]` and the matching per-axis `edges` / `cell_sampling` texts.'''
        d = {'nx': self.nx, 'ny': self.ny, 'dx_m': self.dx_m, 'dy_m': self.dy_m,
             'origin_x_m': self.origin_x_m, 'origin_y_m': self.origin_y_m}
        d.update(self._DOC_KEYS)
        if self.any_periodic:
            d['cell_sampling'] = self.cell_sampling_convention
            d['edges'] = self.edge_convention
            d['periodic'] = [self.periodic[0], self.periodic[1]]
        return d

    @classmethod
    def from_dict(cls, d):
        '''Strict reader. Optional `periodic: [bool, bool]` (real JSON booleans only). The documentation keys `edges` /
        `cell_sampling`, when present, must match the DECLARED periodicity: a legacy `edges: "periodic"` (or any periodic
        text) without a valid `periodic` declaration is refused, as is a finite `edges` text beside `periodic: [true, ...]`.'''
        if not isinstance(d, dict):
            raise StateError('domain must be a JSON object / dict')
        unknown = sorted(set(d) - set(cls._DICT_KEYS) - set(cls._DOC_KEYS) - {'notes', 'periodic'})
        if unknown:
            raise StateError(f'domain: unknown keys {unknown}')
        for k in ('nx', 'ny', 'dx_m'):
            if k not in d:
                raise StateError(f'domain requires {k}')
        periodic = check_periodic(d['periodic'], 'domain periodic') if 'periodic' in d else (False, False)
        expected_doc = dict(cls._DOC_KEYS)
        if any(periodic):
            expected_doc['cell_sampling'] = CELL_SAMPLING_PERIODIC
            expected_doc['edges'] = edge_convention(periodic)
        for k, expected in expected_doc.items():
            if k in d and d[k] != expected:
                raise StateError(f'domain metadata {k!r} = {d[k]!r} conflicts with the LEAF convention {expected!r} for the '
                                 f'declared periodic={list(periodic)!r} (periodicity is declared ONLY by the explicit boolean '
                                 '"periodic": [x, y] key; a periodic "edges" text without it, or another sampling rule, is refused)')
        return cls(d['nx'], d['ny'], d['dx_m'], d.get('dy_m'), d.get('origin_x_m', 0.0), d.get('origin_y_m', 0.0),
                   periodic=periodic)

    def __eq__(self, other):
        return isinstance(other, Domain) and self.periodic == other.periodic \
            and all(getattr(self, k) == getattr(other, k) for k in self._DICT_KEYS)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((tuple(getattr(self, k) for k in self._DICT_KEYS), self.periodic))

    def __repr__(self):
        base = (f'Domain(nx={self.nx}, ny={self.ny}, dx_m={self.dx_m!r}, dy_m={self.dy_m!r}, '
                f'origin_x_m={self.origin_x_m!r}, origin_y_m={self.origin_y_m!r}')
        return base + (f', periodic={self.periodic!r})' if self.any_periodic else ')')

    # ---- positions and sampling
    def contains(self, x_m, y_m):
        '''Boolean mask of points inside the CLOSED domain (NaN -> False).'''
        x = np.asarray(x_m, dtype=np.float64)
        y = np.asarray(y_m, dtype=np.float64)
        return ((x >= self.x_min_m) & (x <= self.x_max_m) & (y >= self.y_min_m) & (y <= self.y_max_m))

    def check_positions(self, x_m, y_m, label='positions'):
        x = np.asarray(x_m, dtype=np.float64)
        y = np.asarray(y_m, dtype=np.float64)
        if x.shape != y.shape:
            raise StateError(f'{label}: x and y shapes differ')
        bad = ~(np.isfinite(x) & np.isfinite(y))
        if bad.any():
            raise StateError(f'{label}: {int(bad.sum())} non-finite coordinate(s)')
        out = ~self.contains(x, y)
        if out.any():
            i = int(np.flatnonzero(out)[0])
            raise StateError(f'{label}: {int(out.sum())} point(s) outside the closed domain '
                             f'[{self.x_min_m}, {self.x_max_m}] x [{self.y_min_m}, {self.y_max_m}] '
                             f'(first: index {i}, x={x.flat[i]!r}, y={y.flat[i]!r}); no wrapping is applied')
        return x, y

    def wrap_positions(self, x_m, y_m):
        '''(x, y) float64 with every PERIODIC axis wrapped into the half-open [min, max): the EXACT upper endpoint (a value equal to
        `x_max_m` / `y_max_m`, including a wrapped result that rounds onto it) is canonicalised to the lower edge by explicit
        equality, never by a tolerance; a value already inside [min, max) is returned bit-for-bit unchanged, so a genuinely
        just-inside value stays distinct. NONPERIODIC axes are returned unchanged (no
        clipping, no wrapping, no range validation): this is the explicit step for NEWLY COMPUTED coordinates (e.g. daughter
        positions); ordinary input positions are never silently wrapped (`check_positions` keeps refusing them). Non-finite input is
        refused for every domain.'''
        x = np.asarray(x_m, dtype=np.float64)
        y = np.asarray(y_m, dtype=np.float64)
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise StateError('wrap_positions: non-finite coordinate(s)')
        if not self.any_periodic:
            return x, y

        def wrap(v, origin, length, vmax):
            inside = (v >= origin) & (v < vmax)                 # already in the half-open range: returned bit-for-bit unchanged
            w = np.mod(v - origin, length)
            w = np.where(w >= length, 0.0, w)
            res = origin + w
            # the exact upper endpoint is the lower edge: decided on the ORIGINAL input (v - origin may round above the span before the
            # modulo) and, as well, on a result that rounded onto it
            res = np.where((v == vmax) | (res == vmax), origin, res)
            return np.where(inside, v, res)
        if self.periodic[0]:
            x = wrap(x, self.origin_x_m, self.width_m, self.x_max_m)
        if self.periodic[1]:
            y = wrap(y, self.origin_y_m, self.height_m, self.y_max_m)
        return x, y

    def cell_indices(self, x_m, y_m):
        '''(row, col) int64 arrays of the containing cell (`cell_sampling_convention`): closed upper edge on a nonperiodic axis
        (clipped to the last cell). On a periodic axis the EXACT upper endpoint (`x == x_max_m`, decided by equality on the landscape
        coordinate, so it holds on decimal spacings / origins where `x - origin` rounds below the span) samples index 0 (upper edge ==
        lower edge); every other value is clipped to [0, n - 1], so a real just-inside position keeps its own (last) cell.'''
        x, y = self.check_positions(x_m, y_m)
        col = np.floor((x - self.origin_x_m) / self.dx_m).astype(np.int64)
        row = np.floor((y - self.origin_y_m) / self.dy_m).astype(np.int64)
        np.clip(col, 0, self.nx - 1, out=col)
        np.clip(row, 0, self.ny - 1, out=row)
        if self.periodic[0]:
            col[x == self.x_max_m] = 0
        if self.periodic[1]:
            row[y == self.y_max_m] = 0
        return row, col

    def check_field(self, field, name='surface'):
        '''Validated FP64 (ny, nx) field: numeric, exact shape (transposed shapes are refused), all finite.'''
        if isinstance(field, (bool, str, bytes)) or field is None:
            raise StateError(f'{name} must be a numeric (ny, nx) array')
        a = np.asarray(field)
        if a.dtype.kind not in 'fiu':
            raise StateError(f'{name} must have a real numeric dtype, got {a.dtype}')
        if a.shape != self.shape:
            hint = ' (looks transposed: arrays are (ny, nx))' if a.shape == (self.nx, self.ny) else ''
            raise StateError(f'{name} shape {a.shape} differs from domain shape {self.shape}{hint}')
        a = np.asarray(a, dtype=np.float64)
        if not np.all(np.isfinite(a)):
            raise StateError(f'{name} contains {int((~np.isfinite(a)).sum())} non-finite value(s)')
        return a

    def sample(self, field, x_m, y_m, name='surface'):
        '''Containing-cell sample of a validated (ny, nx) field at points inside the domain.'''
        f = self.check_field(field, name)
        row, col = self.cell_indices(x_m, y_m)
        return f[row, col]


def constant_surface(domain, elevation_m):
    return np.full(domain.shape, _finite_float(elevation_m, 'elevation_m'), dtype=np.float64)


def load_surface(path, domain=None):
    '''np.load(allow_pickle=False) of a 2-D numeric .npy surface; checked against the domain if given.'''
    a = np.load(path, allow_pickle=False)
    if not isinstance(a, np.ndarray) or a.ndim != 2:
        raise StateError(f'surface file {path!r} must hold a 2-D numeric array')
    return a if domain is None else domain.check_field(a, f'surface {path!r}')


# ------------------------------------------------------------------ array coercion / validation (pure)
def _coerce_float(v, name, n=None):
    '''Owned float64 copy (never aliases the input); bool/object/complex/str refused.'''
    if isinstance(v, (bool, np.bool_, str, bytes, complex)):
        raise StateError(f'{name} must be a real numeric array, got {type(v).__name__}')
    a = np.asarray(v)
    if a.dtype.kind not in 'fiu':
        raise StateError(f'{name} must be a real numeric array (not bool/complex/object), got dtype {a.dtype}')
    if a.ndim > 1:
        raise StateError(f'{name} must be 1-D')
    a = np.array(a, dtype=np.float64, copy=True).reshape(-1)
    if n is not None and a.shape[0] != n:
        raise StateError(f'{name} has length {a.shape[0]}, expected {n}')
    return a


def _coerce_int(v, name, n=None, dtype=np.int64, lo=None, hi=None):
    '''Owned integer copy; bool/float refused; unsigned values above the target range refused.'''
    if isinstance(v, (bool, np.bool_, str, bytes, float, complex)):
        raise StateError(f'{name} must be an integer array, got {type(v).__name__}')
    a = np.asarray(v)
    if a.dtype.kind not in 'iu':
        raise StateError(f'{name} must be an integer array (not bool/float), got dtype {a.dtype}')
    if a.ndim > 1:
        raise StateError(f'{name} must be 1-D')
    a = a.reshape(-1)
    if n is not None and a.shape[0] != n:
        raise StateError(f'{name} has length {a.shape[0]}, expected {n}')
    if a.size:
        mn, mx = int(a.min()), int(a.max())
        info = np.iinfo(dtype)
        if mn < int(info.min) or mx > int(info.max):
            raise StateError(f'{name} values outside the {np.dtype(dtype)} range (got [{mn}, {mx}])')
        if lo is not None and mn < lo:
            raise StateError(f'{name} values must be >= {lo}, got {mn}')
        if hi is not None and mx >= hi:
            raise StateError(f'{name} values must be < {hi}, got {mx}')
    return np.array(a, dtype=dtype, copy=True)


def _coerce_bool(v, name, n):
    if isinstance(v, (bool, np.bool_, str, bytes)):
        raise StateError(f'{name} must be a bool array of length {n}')
    a = np.asarray(v)
    if a.dtype.kind != 'b' or a.ndim != 1 or a.shape[0] != n:
        raise StateError(f'{name} must be a bool array of length {n}')
    return np.array(a, dtype=bool, copy=True)


def status_codes(status, n=None):
    '''Names/ints (scalar or sequence) -> owned int8 code array; unknown statuses refused.'''
    if isinstance(status, (str, int, np.integer)) and not isinstance(status, (bool, np.bool_)):
        status = [status] * (1 if n is None else n)
    if isinstance(status, np.ndarray) and status.dtype.kind in 'iu':
        return _coerce_int(status, 'status', n, np.int8, 0, len(STATUS_CODES))
    if isinstance(status, np.ndarray) and status.dtype.kind not in 'US' and status.dtype != object:
        raise StateError(f'status array must hold names or integer codes, got dtype {status.dtype}')
    out = []
    for s in status:
        if isinstance(s, (bool, np.bool_)):
            raise StateError('status must not be bool')
        if isinstance(s, (int, np.integer)):
            if int(s) not in STATUS_NAMES:
                raise StateError(f'unsupported status code {int(s)}; known {STATUS_NAMES}')
            out.append(int(s))
        elif isinstance(s, str) and s in STATUS_CODES:
            out.append(STATUS_CODES[s])
        else:
            raise StateError(f'unsupported status {s!r}; known {sorted(STATUS_CODES)}')
    a = np.array(out, dtype=np.int8)
    if n is not None and a.shape[0] != n:
        raise StateError(f'status has length {a.shape[0]}, expected {n}')
    return a


def _validate_arrays(a, domain, n_species):
    '''PURE invariant check of a complete array set (dict). Raises StateError naming the first bad row.'''
    n = int(a['ids'].shape[0])
    for k, v in a.items():
        if v.ndim != 1 or v.shape[0] != n:
            raise StateError(f'array {k} has shape {v.shape}, expected ({n},)')
    if n == 0:
        return
    ids = a['ids']
    if np.unique(ids).shape[0] != n:
        u, c = np.unique(ids, return_counts=True)
        raise StateError(f'duplicate plant IDs {u[c > 1][:10].tolist()}')
    sp = a['species_index']
    if int(sp.min()) < 0 or int(sp.max()) >= n_species:
        raise StateError(f'species_index outside [0, {n_species})')
    domain.check_positions(a['x_m'], a['y_m'], 'plant positions')
    st = a['status']
    if int(st.min()) < 0 or int(st.max()) > REMOVED:
        raise StateError('unsupported status code')
    est, unres = a['establishment_ground_m'], a['establishment_unresolved']
    bad = ~unres & ~np.isfinite(est)
    if bad.any():
        raise StateError(f'establishment_ground_m not finite for resolved plant id {int(ids[np.flatnonzero(bad)[0]])}')
    bad = unres & np.isfinite(est)
    if bad.any():
        raise StateError('unresolved anchors must not carry an elevation')
    B, E, age = a['agb_dry_kg'], a['seasonal_agb_dry_kg'], a['age_days']
    for name, arr in (('agb_dry_kg', B), ('seasonal_agb_dry_kg', E), ('age_days', age)):
        if not np.all(np.isfinite(arr)):
            raise StateError(f'{name} not finite for plant id {int(ids[np.flatnonzero(~np.isfinite(arr))[0]])}')
    standing = st != REMOVED
    bad = standing & (B <= 0.0)
    if bad.any():
        raise StateError(f'agb_dry_kg must be > 0 for live/dead_standing plants (id '
                         f'{int(ids[np.flatnonzero(bad)[0]])}: {B[np.flatnonzero(bad)[0]]!r})')
    bad = B < 0.0
    if bad.any():
        raise StateError(f'agb_dry_kg must be >= 0 (removed plants), id {int(ids[np.flatnonzero(bad)[0]])}')
    bad = (E < 0.0) | (E > B)
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise StateError(f'seasonal_agb_dry_kg must satisfy 0 <= E <= B (id {int(ids[i])}: E={E[i]!r}, B={B[i]!r})')
    bad = age < 0.0
    if bad.any():
        raise StateError(f'age_days must be >= 0 (id {int(ids[np.flatnonzero(bad)[0]])})')


# ------------------------------------------------------------------ plant population
class PlantState:
    '''Canonical structure-of-arrays population (O(P) owned read-only arrays, one species name table,
    stable int64 IDs). Scalars: time_days (model calendar), catalog_hash, domain, species_names,
    provenance (small dict). Row order carries no meaning; by-ID operations use a sorted index.'''

    ARRAY_NAMES = ('ids', 'species_index', 'x_m', 'y_m', 'establishment_ground_m', 'establishment_unresolved',
                   'agb_dry_kg', 'seasonal_agb_dry_kg', 'age_days', 'status')

    def __init__(self, domain, species_names, catalog_hash, *, ids, species_index, x_m, y_m,
                 establishment_ground_m, agb_dry_kg, seasonal_agb_dry_kg=None, age_days, status,
                 establishment_unresolved=None, time_days=0.0, provenance=None):
        if not isinstance(domain, Domain):
            raise StateError('domain must be a leaf.state.Domain')
        names = tuple(species_names)
        if not names or any(not isinstance(s, str) or not s for s in names) or len(set(names)) != len(names):
            raise StateError('species_names must be a non-empty tuple of unique non-empty strings')
        if not isinstance(catalog_hash, str) or not catalog_hash.startswith('sha256:'):
            raise StateError('catalog_hash must be the catalog "sha256:..." hash string')
        self.domain = domain
        self.species_names = names
        self.catalog_hash = catalog_hash
        self.time_days = _finite_float(time_days, 'time_days')
        self.provenance = dict(provenance) if provenance else {}
        ids_a = _coerce_int(ids, 'plant_id', None, np.int64)
        n = int(ids_a.shape[0])
        est = _coerce_float(establishment_ground_m, 'establishment_ground_m', n)
        if establishment_unresolved is None:
            if not np.all(np.isfinite(est)):
                raise StateError('establishment_ground_m must be finite; unresolved SURFACE anchors require the explicit '
                                 'establishment_unresolved mask (or the SURFACE token in an initial CSV); a bare NaN/inf '
                                 'anchor is not a surface request')
            unres = np.zeros(n, dtype=bool)
        else:
            unres = _coerce_bool(establishment_unresolved, 'establishment_unresolved', n)
            bad = ~unres & ~np.isfinite(est)
            if bad.any():
                raise StateError(f'{int(bad.sum())} resolved anchor(s) are not finite')
            est[unres] = np.nan
        arrays = {
            'ids': ids_a,
            'species_index': _coerce_int(species_index, 'species_index', n, np.int32),
            'x_m': _coerce_float(x_m, 'x_m', n), 'y_m': _coerce_float(y_m, 'y_m', n),
            'establishment_ground_m': est, 'establishment_unresolved': unres,
            'agb_dry_kg': _coerce_float(agb_dry_kg, 'agb_dry_kg', n),
            'seasonal_agb_dry_kg': (np.zeros(n, dtype=np.float64) if seasonal_agb_dry_kg is None
                                    else _coerce_float(seasonal_agb_dry_kg, 'seasonal_agb_dry_kg', n)),
            'age_days': _coerce_float(age_days, 'age_days', n),
            'status': status_codes(status, n),
        }
        _validate_arrays(arrays, domain, len(names))
        self._set_arrays(arrays)

    # ---- internals
    def _set_arrays(self, a):
        '''Install OWNED arrays (callers pass fresh copies) and freeze them.'''
        for k in self.ARRAY_NAMES:
            arr = a[k]
            arr.setflags(write=False)
            object.__setattr__(self, k, arr)
        object.__setattr__(self, '_order', None)
        object.__setattr__(self, '_sorted_ids', None)

    def _arrays(self, copy=False):
        return {k: (getattr(self, k).copy() if copy else getattr(self, k)) for k in self.ARRAY_NAMES}

    def _ensure_index(self):
        if self._order is None:
            order = np.argsort(self.ids, kind='stable')
            object.__setattr__(self, '_order', order)
            object.__setattr__(self, '_sorted_ids', self.ids[order])

    def __setattr__(self, name, value):
        if name in self.ARRAY_NAMES:
            raise AttributeError(f'{name} is a canonical read-only array; use update_plants/advance_time/'
                                 'resolve_establishment/adopt')
        if name == 'time_days':
            value = _finite_float(value, 'time_days')
        object.__setattr__(self, name, value)

    def validate(self):
        '''Full invariant check (scalars + arrays); returns self.'''
        _finite_float(self.time_days, 'time_days')
        if not isinstance(self.domain, Domain):
            raise StateError('domain must be a leaf.state.Domain')
        _validate_arrays(self._arrays(), self.domain, len(self.species_names))
        return self

    # ---- basic properties
    @property
    def n(self):
        return int(self.ids.shape[0])

    def __len__(self):
        return self.n

    def __repr__(self):
        return f'PlantState(n={self.n}, time_days={self.time_days!r}, species={list(self.species_names)})'

    @property
    def establishment_resolved(self):
        return not bool(self.establishment_unresolved.any())

    def copy(self):
        new = PlantState.__new__(PlantState)
        object.__setattr__(new, 'domain', self.domain)
        object.__setattr__(new, 'species_names', self.species_names)
        object.__setattr__(new, 'catalog_hash', self.catalog_hash)
        object.__setattr__(new, 'time_days', self.time_days)
        object.__setattr__(new, 'provenance', dict(self.provenance))
        new._set_arrays(self._arrays(copy=True))
        return new

    def species_of(self, index=None):
        '''Object array of species id strings (all plants or the given row indices).'''
        names = np.array(self.species_names, dtype=object)
        sp = self.species_index if index is None else self.species_index[index]
        return names[sp]

    def status_names(self, index=None):
        st = self.status if index is None else self.status[index]
        return np.array([STATUS_NAMES[int(s)] for s in st], dtype=object)

    def live_mask(self):
        return self.status == LIVE

    def standing_mask(self):
        '''live or dead_standing (plants with a canopy); removed excluded.'''
        return self.status != REMOVED

    def counts(self):
        return {name: int(np.count_nonzero(self.status == code)) for name, code in STATUS_CODES.items()}

    # ---- ID resolution
    def indices_of(self, ids):
        '''Row indices of the given IDs (order preserved). Unknown or duplicated query IDs raise.'''
        q = _coerce_int(ids, 'plant_id', None, np.int64)
        if q.size == 0:
            return np.zeros(0, dtype=np.int64)
        if np.unique(q).shape[0] != q.shape[0]:
            u, c = np.unique(q, return_counts=True)
            raise StateError(f'duplicate plant IDs in request: {u[c > 1][:10].tolist()}')
        if self.n == 0:
            raise StateError(f'unknown plant IDs {q[:10].tolist()} (population is empty)')
        self._ensure_index()
        pos = np.searchsorted(self._sorted_ids, q)
        pos_c = np.minimum(pos, self.n - 1)
        found = (pos < self.n) & (self._sorted_ids[pos_c] == q)
        if not found.all():
            raise StateError(f'unknown plant IDs {q[~found][:10].tolist()}')
        return self._order[pos_c]

    # ---- SURFACE resolution (once)
    def resolve_establishment(self, surface):
        '''Resolve every unresolved SURFACE anchor from the caller's INITIAL surface (ny, nx). Already
        resolved anchors are never touched. Returns the number of plants resolved (0 is a no-op).'''
        f = self.domain.check_field(surface, 'initial surface')
        m = self.establishment_unresolved
        k = int(m.sum())
        if k == 0:
            return 0
        est = self.establishment_ground_m.copy()
        est[m] = self.domain.sample(f, self.x_m[m], self.y_m[m])
        a = self._arrays()
        a['establishment_ground_m'] = est
        a['establishment_unresolved'] = np.zeros(self.n, dtype=bool)
        _validate_arrays(a, self.domain, len(self.species_names))
        self._set_arrays(a)
        return k

    def require_resolved(self, what='this operation'):
        if not self.establishment_resolved:
            raise StateError(f'{int(self.establishment_unresolved.sum())} plant(s) still carry the unresolved SURFACE '
                             f'anchor; call resolve_establishment(initial_surface) before {what}')

    # ---- calendar
    def advance_time(self, days):
        '''Advance the model calendar by `days` (>= 0, finite): time_days += days and live plants age by
        `days`. Call once per elapsed interval (wind, water, growth, transport alike); exporting snapshots
        never calls this. The new clock and ages are computed and checked first; nothing changes on failure.
        Unresolved initial placements cannot be advanced (resolve them first).'''
        d = _finite_float(days, 'days')
        if d < 0.0:
            raise StateError(f'days must be >= 0, got {d}')
        self.require_resolved('advancing time')
        new_time = self.time_days + d
        if not math.isfinite(new_time):
            raise StateError('clock overflow: time_days would not be finite')
        age = self.age_days.copy()
        with np.errstate(over='raise', invalid='raise'):
            try:
                age[self.status == LIVE] += d
            except FloatingPointError as exc:
                raise StateError(f'age overflow: {exc}') from exc
        if not np.all(np.isfinite(age)):
            raise StateError('age overflow: age_days would not be finite')
        a = self._arrays()
        a['age_days'] = age
        _validate_arrays(a, self.domain, len(self.species_names))
        self._set_arrays(a)
        object.__setattr__(self, 'time_days', new_time)
        return self.time_days

    # ---- transactional updates by ID
    def update_plants(self, ids, *, agb_dry_kg=None, seasonal_agb_dry_kg=None, status=None):
        '''Replace biomass / seasonal mass / status of the given IDs. Everything is validated on a
        candidate copy first; the state is untouched when anything fails. Status transitions
        removed -> other and dead_standing -> live are refused. Returns the row indices.'''
        idx = self.indices_of(ids)
        n = int(idx.shape[0])
        if agb_dry_kg is None and seasonal_agb_dry_kg is None and status is None:
            raise StateError('update_plants: nothing to update')
        a = self._arrays()
        cand = dict(a)
        if agb_dry_kg is not None:
            B = a['agb_dry_kg'].copy()
            B[idx] = _coerce_float(agb_dry_kg, 'agb_dry_kg', n)
            cand['agb_dry_kg'] = B
        if seasonal_agb_dry_kg is not None:
            E = a['seasonal_agb_dry_kg'].copy()
            E[idx] = _coerce_float(seasonal_agb_dry_kg, 'seasonal_agb_dry_kg', n)
            cand['seasonal_agb_dry_kg'] = E
        if status is not None:
            new = status_codes(status, n)
            old = a['status'][idx]
            bad = (old == REMOVED) & (new != REMOVED)
            if bad.any():
                raise StateError(f'removed plants cannot change status (id {int(self.ids[idx[np.flatnonzero(bad)[0]]])})')
            bad = (old == DEAD_STANDING) & (new == LIVE)
            if bad.any():
                raise StateError(f'dead_standing plants cannot return to live (id {int(self.ids[idx[np.flatnonzero(bad)[0]]])})')
            st = a['status'].copy()
            st[idx] = new
            cand['status'] = st
        _validate_arrays(cand, self.domain, len(self.species_names))
        self._set_arrays(cand)
        return idx

    def adopt(self, other):
        '''Take over all arrays/scalars of a validated candidate state (same domain/catalog/species).'''
        if not isinstance(other, PlantState) or other.domain != self.domain or other.catalog_hash != self.catalog_hash \
                or other.species_names != self.species_names:
            raise StateError('adopt: candidate state has a different domain/catalog/species table')
        other.validate()
        object.__setattr__(self, 'time_days', other.time_days)
        object.__setattr__(self, 'provenance', dict(other.provenance))
        self._set_arrays(other._arrays(copy=True))

    # ---- reporting
    def summary(self):
        live = self.live_mask()
        per_species = {}
        for i, name in enumerate(self.species_names):
            m = self.species_index == i
            per_species[name] = {'total': int(m.sum()), 'live': int((m & live).sum()),
                                 'agb_dry_kg_sum': float(self.agb_dry_kg[m].sum()) if m.any() else 0.0}
        return {'n_plants': self.n, 'time_days': self.time_days, 'status_counts': self.counts(),
                'establishment_resolved': self.establishment_resolved,
                'agb_dry_kg_total': float(self.agb_dry_kg.sum()) if self.n else 0.0,
                'seasonal_agb_dry_kg_total': float(self.seasonal_agb_dry_kg.sum()) if self.n else 0.0,
                'age_days_live_mean': float(self.age_days[live].mean()) if live.any() else None,
                'per_species': per_species, 'catalog_hash': self.catalog_hash, 'domain': self.domain.to_dict()}

    def to_records(self):
        '''List of per-plant dicts with CSV column names (SURFACE token for unresolved anchors).'''
        names = self.species_of()
        out = []
        for i in range(self.n):
            out.append({'plant_id': int(self.ids[i]), 'species': str(names[i]), 'x_m': float(self.x_m[i]),
                        'y_m': float(self.y_m[i]),
                        'establishment_ground_m': (SURFACE_TOKEN if self.establishment_unresolved[i]
                                                   else float(self.establishment_ground_m[i])),
                        'agb_dry_kg': float(self.agb_dry_kg[i]), 'seasonal_agb_dry_kg': float(self.seasonal_agb_dry_kg[i]),
                        'age_days': float(self.age_days[i]), 'status': STATUS_NAMES[int(self.status[i])]})
        return out

    # ---- construction helpers
    @classmethod
    def empty(cls, catalog, domain, time_days=0.0, provenance=None):
        z = np.zeros(0)
        return cls(domain, catalog.species_ids, catalog.hash, ids=np.zeros(0, np.int64), species_index=np.zeros(0, np.int32),
                   x_m=z, y_m=z, establishment_ground_m=z, agb_dry_kg=z, seasonal_agb_dry_kg=z, age_days=z,
                   status=np.zeros(0, np.int8), time_days=time_days, provenance=provenance)

    @classmethod
    def from_records(cls, catalog, domain, records, *, time_days=0.0, surface=None, provenance=None):
        '''Build from dicts keyed by CSV column names (species ids as strings; SURFACE token allowed:
        records are an initial list by definition).'''
        return _build_from_rows(catalog, domain, [_record_to_row(r, i) for i, r in enumerate(records)],
                                time_days=time_days, surface=surface, provenance=provenance, source='records')


# ------------------------------------------------------------------ CSV parsing helpers
def _record_to_row(r, i):
    if not isinstance(r, dict):
        raise StateError(f'record {i} must be a dict')
    missing = [c for c in CSV_COLUMNS if c not in r]
    if missing:
        raise StateError(f'record {i} is missing columns {missing}')
    return [r[c] for c in CSV_COLUMNS]


def _parse_float(tok, col, where):
    if isinstance(tok, (bool, np.bool_)):
        raise StateError(f'{where}: {col} must not be bool')
    if _traits.is_real(tok):
        v = float(tok)
    else:
        s = str(tok).strip()
        if not _FLOAT_RE.fullmatch(s):
            raise StateError(f'{where}: {col} value {tok!r} is not a plain decimal number')
        v = float(s)
    if not math.isfinite(v):
        raise StateError(f'{where}: {col} value {tok!r} is not finite')
    return v


def _parse_int(tok, col, where):
    if isinstance(tok, (bool, np.bool_)):
        raise StateError(f'{where}: {col} must not be bool')
    if isinstance(tok, (int, np.integer)):
        v = int(tok)
    else:
        s = str(tok).strip()
        if not _INT_RE.fullmatch(s):
            raise StateError(f'{where}: {col} value {tok!r} is not an integer')
        v = int(s)
    if v < INT64_MIN or v > INT64_MAX:
        raise StateError(f'{where}: {col} value {v} outside int64')
    return v


def _is_surface_token(tok):
    return isinstance(tok, str) and tok.strip() == SURFACE_TOKEN


def _build_from_rows(catalog, domain, rows, *, time_days, surface, provenance, source):
    n = len(rows)
    ids = np.empty(n, np.int64)
    sp = np.empty(n, np.int32)
    x = np.empty(n)
    y = np.empty(n)
    est = np.empty(n)
    unres = np.zeros(n, bool)
    B = np.empty(n)
    E = np.empty(n)
    age = np.empty(n)
    st = np.empty(n, np.int8)
    for i, row in enumerate(rows):
        where = f'{source} row {i + 1}'
        if len(row) != len(CSV_COLUMNS):
            raise StateError(f'{where}: expected {len(CSV_COLUMNS)} fields, got {len(row)} (malformed/truncated row)')
        ids[i] = _parse_int(row[0], 'plant_id', where)
        species = row[1].strip() if isinstance(row[1], str) else None
        if species is None or not catalog.has(species):
            raise StateError(f'{where}: unknown species {row[1]!r}; catalog species {list(catalog.species_ids)}')
        sp[i] = catalog.index(species)
        x[i] = _parse_float(row[2], 'x_m', where)
        y[i] = _parse_float(row[3], 'y_m', where)
        tok = row[4]
        if _is_surface_token(tok):
            unres[i], est[i] = True, np.nan
        else:
            est[i] = _parse_float(tok, 'establishment_ground_m', where)
        B[i] = _parse_float(row[5], 'agb_dry_kg', where)
        E[i] = _parse_float(row[6], 'seasonal_agb_dry_kg', where)
        age[i] = _parse_float(row[7], 'age_days', where)
        stok = row[8].strip() if isinstance(row[8], str) else row[8]
        if not isinstance(stok, str) or stok not in STATUS_CODES:
            raise StateError(f'{where}: unsupported status {row[8]!r}; known {sorted(STATUS_CODES)}')
        st[i] = STATUS_CODES[stok]
    state = PlantState(domain, catalog.species_ids, catalog.hash, ids=ids, species_index=sp, x_m=x, y_m=y,
                       establishment_ground_m=est, establishment_unresolved=unres, agb_dry_kg=B,
                       seasonal_agb_dry_kg=E, age_days=age, status=st, time_days=time_days, provenance=provenance)
    if surface is not None:
        state.resolve_establishment(surface)
    return state


def _sha256(data):
    return 'sha256:' + hashlib.sha256(data).hexdigest()


def metadata_path(path):
    '''Companion metadata path of a plant CSV: "<stem>.metadata.json" in the same directory, where the stem is the
    file name without its last extension (initial_plants.csv -> initial_plants.metadata.json).'''
    path = os.fspath(path)
    if path.endswith(METADATA_SUFFIX):
        raise StateError(f'{path!r} is a metadata companion name, not a plant CSV path')
    return os.path.splitext(path)[0] + METADATA_SUFFIX


def plant_csv_text(state, *, initial=False):
    '''Serialise a state as PLAIN CSV text (header + one nine-field row per plant; never any metadata, comment or
    JSON). Unresolved SURFACE anchors are only allowed with initial=True. The matching companion metadata is built by
    `plant_metadata` / `plant_files_text`.'''
    state.validate()
    if not state.establishment_resolved and not initial:
        state.require_resolved('writing a non-initial snapshot')
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(CSV_COLUMNS)
    names = state.species_names
    ids, sp, x, y = state.ids, state.species_index, state.x_m, state.y_m
    est, unres, B, E, age, st = (state.establishment_ground_m, state.establishment_unresolved, state.agb_dry_kg,
                                 state.seasonal_agb_dry_kg, state.age_days, state.status)
    for i in range(state.n):
        w.writerow([int(ids[i]), names[sp[i]], repr(float(x[i])), repr(float(y[i])),
                    SURFACE_TOKEN if unres[i] else repr(float(est[i])), repr(float(B[i])), repr(float(E[i])),
                    repr(float(age[i])), STATUS_NAMES[int(st[i])]])
    return buf.getvalue()


def plant_metadata(state, csv_text, *, initial=False, provenance=None, csv_file=None):
    '''Companion metadata dict (schema METADATA_SCHEMA) of a plain plant CSV. `csv_text` (str, or the exact UTF-8
    bytes) is the output of `plant_csv_text` for the SAME state: its SHA-256 and size bind the pair. Unresolved
    SURFACE anchors need initial=True. `csv_file` (a file name) is informational only.'''
    if not state.establishment_resolved and not initial:
        state.require_resolved('describing a non-initial snapshot')
    data = csv_text.encode('utf-8') if isinstance(csv_text, str) else bytes(csv_text)
    csv_block = {'columns': list(CSV_COLUMNS), 'sha256': _sha256(data), 'n_bytes': len(data)}
    if csv_file is not None:
        csv_block['file'] = str(csv_file)
    return {'schema': METADATA_SCHEMA, 'schema_version': METADATA_SCHEMA_VERSION, 'model': 'LEAF',
            'leaf_version': _traits.LEAF_VERSION,
            'description': 'Companion metadata of a plain nine-column LEAF plant CSV; the CSV itself carries no metadata.',
            'csv': csv_block,
            'time_days': state.time_days, 'n_plants': state.n, 'catalog_hash': state.catalog_hash,
            'species_names': list(state.species_names), 'species_table_order': _traits.SPECIES_TABLE_ORDER,
            'domain': state.domain.to_dict(), 'units': dict(UNITS), 'coordinate_frame': COORDINATE_FRAME,
            'status_codes': dict(STATUS_CODES), 'establishment_resolved': state.establishment_resolved,
            'initial': bool(initial), 'cell_sampling': state.domain.cell_sampling_convention,
            'provenance': dict(provenance) if provenance is not None else dict(state.provenance)}


def plant_metadata_text(meta):
    '''Human-readable companion JSON text (sorted keys, indent 2, trailing newline; NaN/Infinity refused).'''
    return json.dumps(meta, sort_keys=True, indent=2, allow_nan=False) + '\n'


METADATA_SIZE_SLACK_BYTES = 512     # scalar fields whose text length depends on the final CSV / clock (see plant_metadata_size)


def plant_metadata_size(state, *, initial=False, provenance=None):
    '''Conservative size in BYTES of the companion JSON text (`plant_metadata_text`) of `state`, computed by a
    STREAMING counting pass (json.JSONEncoder.iterencode with the same options; no full text is ever built, the largest
    transient is one encoded value). Provenance is counted whole (it is unbounded and never truncated). The CSV block
    uses a placeholder with the fixed SHA-256 text length, a 20-digit `n_bytes` and the longest finite float repr for
    `time_days`, so the result does not depend on the clock or the final CSV size and is >= the real text length;
    METADATA_SIZE_SLACK_BYTES is extra margin. The encoder escapes to ASCII, so characters equal bytes. Use it to admit
    the serialized companion BEFORE serialising.'''
    meta = plant_metadata(state, b'', initial=initial, provenance=provenance)
    meta['csv']['n_bytes'] = 10 ** 20 - 1
    meta['time_days'] = -1.7976931348623157e308
    enc = json.JSONEncoder(sort_keys=True, indent=2, allow_nan=False)
    n = 0
    for chunk in enc.iterencode(meta):
        n += len(chunk)
    return n + 1 + METADATA_SIZE_SLACK_BYTES      # + the trailing newline


def plant_files_text(state, *, initial=False, provenance=None, csv_file=None):
    '''(plain CSV text, companion JSON text) of a state, both serialised and validated, nothing written.'''
    csv_text = plant_csv_text(state, initial=initial)
    meta = plant_metadata(state, csv_text, initial=initial, provenance=provenance, csv_file=csv_file)
    return csv_text, plant_metadata_text(meta)


_SHA_RE = re.compile(r'sha256:[0-9a-f]{64}')


def parse_plant_metadata(data, *, source='metadata'):
    '''Strictly parse a companion document (JSON text / bytes / an already parsed dict) and check its structure
    (schema, integer schema_version, required keys, types, SHA-256 form, non-negative integer csv.n_bytes). Catalog /
    domain / time / CSV size + SHA-256 consistency is checked later against the CSV by `parse_plant_csv`. Returns a
    dict (a copy for dict input).'''
    if isinstance(data, (bytes, bytearray, str)):
        try:
            meta = _traits.parse_strict_json(data, what=f'{source} companion metadata')
        except _traits.CatalogError as exc:
            raise StateError(f'{source}: companion metadata is not valid strict JSON: {exc}') from exc
    elif isinstance(data, dict):
        meta = dict(data)
    else:
        raise StateError(f'{source}: companion metadata must be JSON text / bytes or a dict, got {type(data).__name__}')
    if not isinstance(meta, dict):
        raise StateError(f'{source}: companion metadata must be a JSON object')
    if meta.get('schema') != METADATA_SCHEMA:
        raise StateError(f'{source}: metadata schema {meta.get("schema")!r} != {METADATA_SCHEMA!r}')
    ver = meta.get('schema_version')
    if type(ver) is not int or ver != METADATA_SCHEMA_VERSION:
        raise StateError(f'{source}: unsupported plant metadata schema_version {ver!r} (this reader supports exactly '
                         f'integer version {METADATA_SCHEMA_VERSION})')
    required = ('catalog_hash', 'species_names', 'domain', 'time_days', 'n_plants', 'initial', 'establishment_resolved',
                'units', 'coordinate_frame', 'csv')
    missing = [k for k in required if k not in meta]
    if missing:
        raise StateError(f'{source}: companion metadata lacks required keys {missing}')
    blk = meta['csv']
    if not isinstance(blk, dict) or not isinstance(blk.get('sha256'), str) or not _SHA_RE.fullmatch(blk['sha256']):
        raise StateError(f'{source}: companion metadata csv.sha256 must be a "sha256:<64 hex>" string')
    nb = blk.get('n_bytes')
    if isinstance(nb, bool) or not isinstance(nb, int) or nb < 0:
        raise StateError(f'{source}: companion metadata csv.n_bytes must be a non-negative integer (the exact CSV size), '
                         f'got {nb!r}')
    if not isinstance(meta['species_names'], list) or any(not isinstance(s, str) for s in meta['species_names']):
        raise StateError(f'{source}: companion metadata species_names must be a list of strings')
    n = meta['n_plants']
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise StateError(f'{source}: companion metadata n_plants must be a non-negative integer')
    for k in ('initial', 'establishment_resolved'):
        if not isinstance(meta[k], bool):
            raise StateError(f'{source}: metadata {k!r} must be a JSON boolean')
    return meta


def _check_context(meta, catalog, domain, time_days, source):
    '''Shared catalog / table / domain / time / unit checks of legacy and companion metadata. Returns (domain, time_days).'''
    if meta.get('catalog_hash') != catalog.hash:
        raise StateError(f'{source}: catalog hash mismatch (file {meta.get("catalog_hash")!r}, catalog {catalog.hash!r})')
    if 'species_names' in meta:
        names = meta['species_names']
        if not isinstance(names, list) or names != list(catalog.species_ids):
            raise StateError(f'{source}: species name table {names} differs from catalog {list(catalog.species_ids)}')
    if 'domain' in meta:
        md = Domain.from_dict(meta['domain'])
        if domain is not None and md != domain:
            raise StateError(f'{source}: metadata domain {md!r} differs from the supplied domain {domain!r}')
        domain = md
    if 'time_days' in meta:
        mt = _finite_float(meta['time_days'], f'{source} metadata time_days')
        if time_days is not None and float(time_days) != mt:
            raise StateError(f'{source}: metadata time_days {mt!r} differs from the supplied time_days {time_days!r}')
        time_days = mt
    if 'units' in meta and meta['units'] != UNITS:
        raise StateError(f'{source}: metadata units differ from {UNITS}')
    if 'coordinate_frame' in meta and meta['coordinate_frame'] != COORDINATE_FRAME:
        raise StateError(f'{source}: coordinate frame {meta["coordinate_frame"]!r} differs from {COORDINATE_FRAME!r}')
    if 'initial' in meta and not isinstance(meta['initial'], bool):
        raise StateError(f'{source}: metadata "initial" must be a JSON boolean')
    return domain, time_days


def _legacy_metadata(first_line, source):
    '''LEGACY READ ONLY: the embedded "# {json}" first line of pre-companion files.'''
    try:
        meta = _traits.parse_strict_json(first_line[1:].strip(), what=f'{source} metadata line')
    except _traits.CatalogError as exc:
        raise StateError(f'{source}: the first line starts with "#" but is not a legacy JSON metadata line (plain plant '
                         f'CSV files carry no comments or metadata): {exc}') from exc
    if not isinstance(meta, dict):
        raise StateError(f'{source}: metadata line must be a JSON object')
    if meta.get('schema') != CSV_SCHEMA:
        raise StateError(f'{source}: metadata schema {meta.get("schema")!r} != {CSV_SCHEMA!r}')
    if meta.get('schema_version') != CSV_SCHEMA_VERSION or isinstance(meta.get('schema_version'), bool):
        raise StateError(f'{source}: unsupported plant CSV schema_version {meta.get("schema_version")!r}')
    return meta


def _csv_bytes_and_text(text, source):
    if isinstance(text, (bytes, bytearray, memoryview)):
        raw = bytes(text)
        try:
            return raw, raw.decode('utf-8')
        except UnicodeDecodeError as exc:
            raise StateError(f'{source}: plant CSV is not valid UTF-8: {exc}') from exc
    if isinstance(text, str):
        try:
            return text.encode('utf-8'), text
        except UnicodeEncodeError as exc:
            raise StateError(f'{source}: plant CSV text cannot be encoded as UTF-8: {exc}') from exc
    raise StateError(f'{source}: plant CSV must be str or bytes, got {type(text).__name__}')


def parse_plant_csv(text, catalog, *, domain=None, surface=None, time_days=None, provenance=None, source='csv',
                    metadata=None):
    '''All-or-nothing parse of plain plant CSV text / bytes. Optional `metadata` (companion JSON text / bytes or the
    dict of `plant_metadata`) supplies and VERIFIES the file context before anything is built: schema version,
    CSV SHA-256 and row count binding (a stale or mismatched pair is refused), catalog hash and species table,
    domain / time (and any caller-supplied domain / time_days must agree), units and the initial / SURFACE
    declaration. Rows are read with csv.reader(strict=True), so ordinary quoted fields are accepted while
    malformed/unterminated quotes raise. Without metadata the file is a handwritten plain list: the caller must
    supply the domain (time_days defaults to 0) and a SURFACE token is allowed. LEGACY: a first line "# {json}"
    (embedded metadata of old files) is still read with the old checks, but not together with `metadata`, and is
    never written.'''
    raw, text = _csv_bytes_and_text(text, source)
    lines = io.StringIO(text, newline='')
    first = lines.readline()
    explicit = metadata is not None and metadata is not False
    meta, legacy = None, False
    if first.startswith('#'):
        if explicit:
            raise StateError(f'{source}: the file starts with an embedded legacy metadata line AND companion metadata was '
                             'supplied; refusing the ambiguous pair')
        meta, legacy = _legacy_metadata(first, source), True
    else:
        lines.seek(0)
        if explicit:
            meta = parse_plant_metadata(metadata, source=source)
            if meta['csv']['n_bytes'] != len(raw):
                raise StateError(f'{source}: the CSV has {len(raw)} bytes but the companion metadata declares '
                                 f'csv.n_bytes={meta["csv"]["n_bytes"]}: the sizes do not match, so the pair is stale, '
                                 'edited or partially written; refusing to use it')
            if meta['csv']['sha256'] != _sha256(raw):
                raise StateError(f'{source}: the CSV bytes ({_sha256(raw)}) do not match the companion metadata '
                                 f'({meta["csv"]["sha256"]}): the pair is stale, edited or partially written; refusing '
                                 'to use it')
            if 'columns' in meta['csv'] and meta['csv']['columns'] != list(CSV_COLUMNS):
                raise StateError(f'{source}: companion metadata csv.columns {meta["csv"]["columns"]} differ from '
                                 f'{list(CSV_COLUMNS)}')
    if meta is not None:
        domain, time_days = _check_context(meta, catalog, domain, time_days, source)
    if domain is None:
        raise StateError(f'{source}: no domain: a plain CSV without companion metadata ("<stem>{METADATA_SUFFIX}") needs '
                         'a Domain supplied by the caller')
    if time_days is None:
        time_days = 0.0
    reader = csv.reader(lines, strict=True)
    try:
        header = next(reader)
    except StopIteration:
        raise StateError(f'{source}: missing header line') from None
    except csv.Error as exc:
        raise StateError(f'{source}: malformed CSV: {exc}') from exc
    if [h.strip() for h in header] != list(CSV_COLUMNS):
        raise StateError(f'{source}: header {header} must be exactly {list(CSV_COLUMNS)}')
    rows = []
    try:
        for row in reader:
            if len(row) == 0:
                continue  # blank line
            rows.append(row)
    except csv.Error as exc:
        raise StateError(f'{source}: malformed CSV: {exc}') from exc
    if meta is not None and 'n_plants' in meta and meta['n_plants'] != len(rows):
        raise StateError(f'{source}: metadata n_plants {meta["n_plants"]} != {len(rows)} data rows (truncated or extended file)')
    n_tokens = sum(1 for row in rows if len(row) > 4 and _is_surface_token(row[4]))
    if meta is not None and n_tokens:
        # a file with metadata that does not declare "initial": true must carry concrete anchors: refuse BEFORE
        # any surface is sampled, so a later surface can never be mistaken for the establishment surface
        if meta.get('initial') is not True or meta.get('establishment_resolved') is True:
            raise StateError(f'{source}: {n_tokens} SURFACE anchor(s) in a file whose metadata does not declare '
                             '"initial": true (saved snapshots must carry concrete establishment elevations); refused '
                             'before any surface is sampled')
    if meta is not None and not legacy and meta['establishment_resolved'] != (n_tokens == 0):
        raise StateError(f'{source}: companion metadata establishment_resolved={meta["establishment_resolved"]} '
                         f'contradicts the CSV ({n_tokens} SURFACE token(s))')
    prov = dict(provenance) if provenance is not None else (dict(meta.get('provenance') or {}) if meta else {})
    return _build_from_rows(catalog, domain, rows, time_days=time_days, surface=surface, provenance=prov, source=source)


def read_plant_companion(path):
    '''Raw bytes of the companion metadata of a plant CSV path, or None when no companion file exists.'''
    mpath = metadata_path(path)
    if not os.path.lexists(mpath):
        return None
    with open(mpath, 'rb') as fh:
        return fh.read()


def read_plant_csv(path, catalog, *, domain=None, surface=None, time_days=None, provenance=None, metadata=None):
    '''Read a plain plant CSV. The companion "<stem>.metadata.json" beside it is loaded automatically and verified
    against the exact CSV bytes (see `parse_plant_csv`); a mismatched / stale / invalid companion is refused. With no
    companion the file is a handwritten plain list (supply `domain`). `metadata` may be companion JSON text / bytes /
    dict to use instead of the file, or False to read the CSV as a plain list while IGNORING a companion (an explicit
    opt-out: the declared time / SURFACE safety of the companion then no longer protects the read).'''
    path = os.fspath(path)
    with open(path, 'rb') as fh:
        data = fh.read()
    if metadata is None:
        metadata = read_plant_companion(path)
    return parse_plant_csv(data, catalog, domain=domain, surface=surface, time_days=time_days,
                           provenance=provenance, source=path, metadata=metadata)


# ------------------------------------------------------------------ atomic output
def preflight_destination(path, *, overwrite=False):
    '''Check a destination before any work: directory exists, and the file does not exist unless overwrite.'''
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path)) or '.'
    if not os.path.isdir(directory):
        raise FileNotFoundError(f'output directory {directory!r} does not exist')
    if os.path.isdir(path):
        raise IsADirectoryError(f'{path!r} is a directory')
    if not overwrite and os.path.lexists(path):
        raise FileExistsError(f'{path!r} exists; pass overwrite=True to replace it')
    return path


def atomic_write_bytes(path, data, *, overwrite=False):
    '''Write to a temporary file in the target directory, fsync, then publish atomically.
    overwrite=False: publish with os.link, which fails if the destination exists (no check-then-replace
    window). When the filesystem cannot hard-link, the write is REFUSED (temp file removed) rather than
    risking an existing destination; use overwrite=True to replace intentionally via os.replace.'''
    path = preflight_destination(path, overwrite=overwrite)
    directory = os.path.dirname(os.path.abspath(path)) or '.'
    fd, tmp = tempfile.mkstemp(prefix='.leaf-', suffix='.tmp', dir=directory)
    try:
        with os.fdopen(fd, 'wb') as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if overwrite:
            os.replace(tmp, path)
        else:
            try:
                os.link(tmp, path)
            except FileExistsError:
                raise
            except OSError as exc:
                raise OSError(f'atomic no-overwrite creation of {path!r} is not supported on this filesystem '
                              f'(hard link failed: {exc}); nothing was written. Pass overwrite=True to replace '
                              'intentionally') from exc
            os.unlink(tmp)
    except BaseException:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def atomic_write_text(path, text, *, overwrite=False):
    return atomic_write_bytes(path, text.encode('utf-8'), overwrite=overwrite)


def _check_replaceable_companion(meta_path):
    '''overwrite=True may only replace a file that is a LEAF plant-state companion (never unrelated metadata).'''
    try:
        with open(meta_path, 'rb') as fh:
            doc = _traits.parse_strict_json(fh.read(), what=f'existing metadata file {meta_path!r}')
    except (OSError, _traits.CatalogError) as exc:
        raise FileExistsError(f'{meta_path!r} exists and is not a readable LEAF plant-state companion ({exc}); '
                              'refusing to overwrite it') from exc
    if not isinstance(doc, dict) or doc.get('schema') != METADATA_SCHEMA:
        raise FileExistsError(f'{meta_path!r} exists but is not a {METADATA_SCHEMA!r} companion; refusing to overwrite '
                              'unrelated metadata')


def preflight_plant_files(path, *, overwrite=False):
    '''Check BOTH destinations of a plant file pair (CSV and "<stem>.metadata.json") before any work: the directory
    exists and neither file exists unless overwrite=True; an overwrite never replaces a metadata file that is not a
    LEAF plant-state companion. Returns (csv_path, metadata_path); nothing is created.'''
    csv_path = preflight_destination(path, overwrite=overwrite)
    meta_path = metadata_path(csv_path)
    if os.path.isdir(meta_path):
        raise IsADirectoryError(f'{meta_path!r} is a directory')
    if os.path.lexists(meta_path):
        if not overwrite:
            raise FileExistsError(f'metadata companion {meta_path!r} exists (it may belong to another file with the same '
                                  'stem); not overwritten; pass overwrite=True to replace it')
        _check_replaceable_companion(meta_path)
    return csv_path, meta_path


def publish_plant_files(path, csv_bytes, metadata_bytes, *, overwrite=False):
    '''Publish an already serialised pair. Both destinations are preflighted first. The companion is published FIRST
    and the CSV SECOND (each atomically, see atomic_write_bytes); if the CSV cannot be published, a newly created
    companion is removed and a replaced one is restored. The two publications are NOT one atomic operation: if the
    process dies in between, an orphan companion without CSV (overwrite=False) or a new companion beside the old CSV
    (overwrite=True) remains, and every reader refuses it (missing CSV / SHA-256 mismatch) instead of using it.
    Returns (csv_path, metadata_path).'''
    csv_path, meta_path = preflight_plant_files(path, overwrite=overwrite)
    existed = os.path.lexists(meta_path)
    prior = None
    if existed:
        with open(meta_path, 'rb') as fh:
            prior = fh.read()
    atomic_write_bytes(meta_path, metadata_bytes, overwrite=overwrite)
    try:
        atomic_write_bytes(csv_path, csv_bytes, overwrite=overwrite)
    except BaseException:
        try:
            if existed:
                atomic_write_bytes(meta_path, prior, overwrite=True)     # restore the previous companion
            else:
                os.unlink(meta_path)                                       # remove the companion this call created
        except OSError:
            pass    # best effort: a leftover companion fails the hash binding on read, never silently used
        raise
    return csv_path, meta_path


def write_plant_files(state, path, *, overwrite=False, initial=False, provenance=None):
    '''Validate and serialise BOTH files (plain CSV + companion JSON), preflight both destinations, then publish.
    Nothing is written when validation or a preflight fails. Returns (csv_path, metadata_path).'''
    preflight_plant_files(path, overwrite=overwrite)
    csv_text, meta_text = plant_files_text(state, initial=initial, provenance=provenance,
                                           csv_file=os.path.basename(os.fspath(path)))
    return publish_plant_files(path, csv_text.encode('utf-8'), meta_text.encode('utf-8'), overwrite=overwrite)


def write_plant_csv(state, path, *, overwrite=False, initial=False, provenance=None):
    '''Write the plain CSV and its companion metadata (see write_plant_files); returns the CSV path.'''
    return write_plant_files(state, path, overwrite=overwrite, initial=initial, provenance=provenance)[0]


__all__ = ['CSV_SCHEMA', 'CSV_SCHEMA_VERSION', 'METADATA_SCHEMA', 'METADATA_SCHEMA_VERSION', 'METADATA_SUFFIX',
           'CSV_COLUMNS', 'SURFACE_TOKEN', 'LIVE', 'DEAD_STANDING', 'REMOVED',
           'STATUS_CODES', 'STATUS_NAMES', 'CELL_SAMPLING_CONVENTION', 'AXES_CONVENTION', 'EDGE_CONVENTION',
           'CELL_SAMPLING_PERIODIC', 'edge_convention', 'check_periodic',
           'COORDINATE_FRAME', 'UNITS', 'StateError', 'Domain', 'constant_surface', 'load_surface', 'status_codes',
           'PlantState', 'metadata_path', 'plant_csv_text', 'plant_metadata', 'plant_metadata_text', 'plant_metadata_size',
           'METADATA_SIZE_SLACK_BYTES', 'plant_files_text',
           'parse_plant_metadata', 'parse_plant_csv', 'read_plant_companion', 'read_plant_csv',
           'preflight_plant_files', 'publish_plant_files', 'write_plant_files', 'write_plant_csv',
           'preflight_destination', 'atomic_write_bytes', 'atomic_write_text']
