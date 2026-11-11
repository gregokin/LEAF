'''leaf.interevent: dated native-water / growth session (LEAF Phase 1 integration; UNQUALIFIED, UNCALIBRATED).

One `IntereventSession` owns ONE accepted boundary of the coupled interevent state:

    native SYRUP soil water (SoilColumnState, the only water authority)  +  living-plant biomass B and seasonal
    biomass E (backend arrays)  +  pending requested / delivered water per plant (D, U)  +  pending per-species
    activity / temperature integrals (A, T, elapsed years)  +  the current fixed root map and canopy areas  +  the
    exact calendar cursor, the next growth boundary, a deferred seasonal turnover flag and the last receipt.

An interval is ONE `TimestepRecord` (leaf.forcing) whose start equals the cursor. The caller owns weather: PET
(record.pet_mm_per_day), temperature, the already-split native forcing (`added_water_m` [L, N] m and the potential
soil evaporation rate `potential_evaporation_m_s` [N]). LEAF computes no PET, no rainfall / infiltration and no
partition of E between soil and plants (that partition is SYRUP's); this narrow driver composes, per open interval:

    1. REAL native `step` (non-T hydrology) from the accepted state            -> staged native state
    2. `growth.demand` on the EXISTING period geometry (canopy areas of the last rebuild), actual month,
       temperature, shared PET and the ACTIVE interval length               -> D [m3] per live plant
    3. REAL `syrup_water.withdraw_plants` ONCE on the staged state (LEAF requests / competes, SYRUP accepts and
       debits, nothing is subtracted in LEAF)                              -> new native state, ACTUAL delivered U
    4. D, U and the activity / temperature integrals (actual 365 / 366 year length) are ACCUMULATED
    5. at a growth boundary (cadence <= 7 days, a local new year, or an explicit flush) `growth.grow` is applied
       ONCE to the accumulated requested / actual water, B/E are copied out of its shared output buffer, and the
       roots / canopy areas are REBUILT from the shared catalog implementation (species + individual overrides
       preserved; an unmet root is never redirected to wet cells)
    6. everything is PUBLISHED by ONE replacement of the accepted boundary, only after every stage, closure and
       admission passed. Any failure leaves the previous boundary untouched; a rejected interval can be retried.

Boundaries. The caller splits records at local midnight (month / year length), at every growth boundary and at a new
year; `IntereventSession.boundaries()` and `split_record()` expose them and an unsplit record is refused BEFORE any
mutation. Growth boundaries are a FIXED GRID `next_boundary + k * cadence` from the session start; a flush finishes
a partial period without moving the grid; boundaries crossed by transport are skipped (no catch-up).

Gates and transport. `flush_growth()` finishes a partial period at the cursor (so native consumers can export the
current canopy). `advance_transport(kind, start, end, ...)` requires NO pending growth, advances the calendar (hence
live ages, exactly once) and skips growth boundaries; no T, growth, E or catch-up happens for its duration. 'wind':
native water unchanged, native time advanced consistently. 'rain' / 'recession': the caller supplies the external
native owner's resulting SoilColumnState (checked for column binding, native validity and end time); LEAF runs no
storm physics. A year end inside transport sets `season_due`; the turnover (E removed, no accrual of event
duration) is applied at the start of the next OPEN interevent interval. A CLOSED gate (`record.gate == 'closed'`) is
a PAUSE with the same semantics as wind: no native step (mutual exclusion with whichever process owns the time),
no T, no growth; the calendar advances. No mortality / disturbance law lives here.

Row mapping. All persistent per-plant working arrays and the root map contain ONLY LIVE plants, in the state row
order: working row i is plant `plants.ids[live_rows[i]]`, `live_rows = flatnonzero(status == live)` (fixed for the
session; the root map ids are checked against it at every rebuild). Stable IDs, species, X/Y, establishment
ground, age and status stay recoverable for ALL plants (dead / removed included) from the private host base arrays;
`snapshot()` returns a canonical PlantState at the cursor (explicit, counted device->host transfer). The PlantState
passed in is copied and never a competing biomass authority. Ages: live age = age at session start + elapsed days
(exact cursor arithmetic), so a continuous run and any split run agree and snapshots never advance the clock.

Memory. Every allocating stage is admitted against ONE global `max_bytes` (host + device summed; `reserve_bytes`
for the caller's other live state): persistent state (base population, working arrays, pending accumulators, root
map, accepted native state, native column copies) + the stage's transient estimate from the existing component
estimators (native `estimate_step_memory`, `growth.memory_estimate`, `syrup_water.estimate_bridge_bytes`, the
allocator plan / feedback estimates) + old / new coexistence. Components receive the actual remaining budget through
their `extra_reserve_bytes`. Where a component already counts an array that the session also lists, the session
EXCLUDES it explicitly; remaining double counts (e.g. D / U in a flush) are labelled conservative. The root builder
gets `max_bytes - (everything else live, old map included)`; the refused rebuild leaves the old boundary. NOT
counted: interpreter, JIT caches, CUDA context, allocator pools, caller-held arrays outside `reserve_bytes`. The
weekly geometry path (host allometry + compiled / NumPy root builder) is HOST based: on CUDA it transfers B and E to
the host and the new CSR back (explicit and counted); it is not GPU resident.
'''
from __future__ import annotations

import calendar
import dataclasses
import datetime as _dt
import math
from fractions import Fraction

import numpy as np

from leaf import canopy as _canopy
from leaf import forcing as _forcing
from leaf import growth as _growth
from leaf import roots as _roots
from leaf import state as _state
from leaf import syrup_water as _sw
from leaf import traits as _traits
from leaf import uptake as _uptake

VERSION = '0.1.0'
BACKENDS = ('numpy', 'numba', 'cuda')
TRANSPORT_KINDS = ('wind', 'rain', 'recession')
EXTERNAL_WATER_KINDS = ('rain', 'recession')
MAX_CADENCE = _dt.timedelta(days=7)
ROW_BYTES_PER_PLANT = 62          # host base population: ids 8, species 4, x 8, y 8, est 8, unresolved 1, B 8, E 8, age 8, status 1
SECONDS_PER_DAY = 86400
TIME_TOL_ABS_S = 1e-6
TIME_TOL_REL = 1e-9
SCOPE = ('UNQUALIFIED / UNCALIBRATED composition of real native SYRUP water, leaf.growth and leaf.roots; no field '
         'calibration, no storm / wind physics, no mortality; explicit native forcing only')
MEMORY_SCOPE = ('global conservative stage ledger: persistent session state + stage transient estimate (component '
                'estimators, resident arrays the component counts excluded) + old/new coexistence + caller reserve; '
                'excluded: interpreter, JIT / CUDA context, allocator pools, caller-held arrays outside reserve_bytes')


class IntereventError(ValueError):
    '''Any session validation / closure / publication error (the previous accepted boundary is intact).'''


class IntereventStateError(IntereventError):
    '''Wrong cursor (gap / overlap / replay), pending growth before transport, or another state precondition.'''


class IntereventBoundaryError(IntereventError):
    '''A record crosses a midnight, growth boundary or new year and must be split by the caller.'''


class IntereventAdmissionError(IntereventError):
    '''The global byte ledger of a stage exceeds max_bytes; raised BEFORE the stage allocates.'''

    def __init__(self, stage, total, max_bytes, report=None):
        super().__init__(f'{stage}: estimated {total} bytes (global ledger) exceed max_bytes={max_bytes}; refused before '
                         'the stage allocates; the previous accepted boundary is intact')
        self.stage, self.total, self.max_bytes, self.report = stage, int(total), int(max_bytes), report


# ------------------------------------------------------------------ small helpers
def _hostarr(a):
    '''Host NumPy view / copy of a backend array (explicit device->host transfer for CuPy).'''
    if isinstance(a, np.ndarray):
        return a
    return np.asarray(a.get(), dtype=a.dtype)


def _clean(obj):
    '''JSON-safe plain structure (numpy scalars -> Python, tuples -> lists); non-finite floats refused.'''
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if not math.isfinite(f):
            raise IntereventError(f'non-finite value {obj!r} in a receipt / restart record')
        return f
    if obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, Fraction):
        return str(obj)
    return str(obj)


def _real_pos_int(v, name, lo=0):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or int(v) < lo:
        raise IntereventError(f'{name} must be a non-boolean integer >= {lo}, got {v!r}')
    return int(v)


def _next_new_year(t):
    try:
        return _dt.datetime(t.year + 1, 1, 1, tzinfo=t.tzinfo)
    except (OverflowError, ValueError):
        return None


def _year_days(t):
    return 366 if calendar.isleap(t.year) else 365


def _sum_items(items, exclude=()):
    return int(sum(v for k, v in items.items() if k not in exclude))


def native_fingerprint(column):
    '''sha256 over the physical column definition (layers, active mask, area): checkpoints bind to it.'''
    import hashlib
    h = hashlib.sha256()
    lay = column.layers
    h.update(repr((int(column.n_layers), int(column.n_cells), float(column.cell_area_m2), column.backend)).encode())
    for name in ('depth_edges_m', 'theta_residual', 'theta_field_capacity', 'theta_saturation', 'diffusivity_m2_s',
                 'residual_m', 'field_capacity_m', 'saturation_m'):
        h.update(np.ascontiguousarray(np.asarray(getattr(lay, name), dtype=np.float64)).tobytes())
    h.update(repr((float(lay.theta_evaporation_stress), float(lay.evaporation_factor), bool(lay.gravity_drainage))).encode())
    h.update(np.ascontiguousarray(_hostarr(column.active).astype(np.uint8)).tobytes())
    return 'sha256:' + h.hexdigest()


# ------------------------------------------------------------------ accepted boundary
class _Boundary:
    '''The accepted boundary. Arrays are NEVER written in place: a transaction clones the boundary, rebinds fields
    to fresh arrays and the session publishes by replacing its single reference.'''

    __slots__ = ('native_state', 'B', 'E', 'D', 'U', 'A', 'T', 'pending_days', 'root_map', 'area', 'cursor',
                 'next_boundary', 'season_due', 'sequence', 'last_receipt', 'last_growth', 'n_growth', 'n_rebuilds')

    def clone(self):
        new = _Boundary()
        for k in self.__slots__:
            setattr(new, k, getattr(self, k, None))
        return new


class IntereventSession:
    '''See the module docstring. Construct with the initial PlantState + catalog, the prepared native operator and
    state, the exact fixed-offset start instant, the backend and the budgets.'''

    # ------------------------------------------------------------------ construction
    def __init__(self, plants, catalog, *, operator, native_state, start, backend, growth_cadence, max_edges,
                 max_bytes, reserve_bytes=0, root_builder='compiled', layer_fraction_overrides=None, root_map=None,
                 verify_native_sources=True):
        if root_map is not None:
            raise IntereventError('an external root map is refused: the session always builds species-separable roots '
                                  'with the shared builder (custom non-separable maps need an explicit compatible '
                                  'rebuild policy that is not part of this phase)')
        if not isinstance(plants, _state.PlantState):
            raise IntereventError('plants must be a leaf.state.PlantState')
        plants.require_resolved('starting an interevent session')
        plants.validate()
        base = plants.copy()
        self._setup(base=base, catalog=catalog, operator=operator, backend=backend, cadence=growth_cadence,
                    max_edges=max_edges, max_bytes=max_bytes, reserve_bytes=reserve_bytes, root_builder=root_builder,
                    overrides=layer_fraction_overrides, anchor=start, native_anchor_t=None,
                    verify_native=verify_native_sources, native_state=native_state)
        self._init_boundary(native_state)

    # ---- static configuration (shared with checkpoint restore)
    def _setup(self, *, base, catalog, operator, backend, cadence, max_edges, max_bytes, reserve_bytes, root_builder,
               overrides, anchor, native_anchor_t, verify_native, native_state):
        if not isinstance(catalog, _traits.SpeciesCatalog):
            raise IntereventError('catalog must be a leaf.traits.SpeciesCatalog')
        if base.catalog_hash != catalog.hash:
            raise IntereventError(f'plant state catalog hash {base.catalog_hash} differs from the catalog {catalog.hash}')
        catalog.check_species_table(base.species_names)
        if backend not in BACKENDS:
            raise IntereventError(f'backend must be explicitly one of {BACKENDS}, got {backend!r}')
        if root_builder not in _roots.BUILDERS:
            raise IntereventError(f'root_builder must be one of {_roots.BUILDERS}')
        _forcing.check_fixed_offset_datetime(anchor, 'start')
        if not isinstance(cadence, _dt.timedelta) or not _dt.timedelta(0) < cadence <= MAX_CADENCE:
            raise IntereventError(f'growth_cadence must be a datetime.timedelta in (0, 7 days], got {cadence!r}')
        self.max_edges = _real_pos_int(max_edges, 'max_edges', 0)
        self.max_bytes = _real_pos_int(max_bytes, 'max_bytes', 1)
        self.reserve_bytes = _real_pos_int(reserve_bytes, 'reserve_bytes', 0)
        nst = _sw.native_status()
        if not nst['available'] or not nst['supported']:
            raise IntereventError(f'native SYRUP unavailable or unsupported version: {nst}; there is no fallback water authority')
        column, xp, op_backend, L, N, area = _sw.check_operator(operator)
        if op_backend != backend:
            raise IntereventError(f'the native operator runs on backend {op_backend!r} but the session backend is '
                                  f'{backend!r}; no silent backend switch')
        dom = base.domain
        if not dom.square_cells:
            raise IntereventError('the shared root builder requires square cells')
        if dom.ncells != N:
            raise IntereventError(f'domain has {dom.ncells} cells but the native column has {N}; the flat cell order is '
                                  'row * nx + col of the (ny, nx) raster')
        if not math.isclose(dom.dx_m * dom.dy_m, area, rel_tol=1e-12, abs_tol=0.0):
            raise IntereventError(f'domain cell area {dom.dx_m * dom.dy_m} differs from the native cell area {area}')
        if catalog.root_layers != L:
            raise IntereventError(f'catalog root_layers {catalog.root_layers} differ from the native layer count {L}')
        self._provenance = _sw.native_provenance(verify=bool(verify_native))
        if verify_native and not self._provenance.get('all_hashes_match', False):
            raise IntereventError('imported native SYRUP sources do not match the pinned hashes of the bridge '
                                  f'({self._provenance.get("sha256")}); refusing (verify_native_sources=False to override)')
        self._operator, self._column, self._xp = operator, column, xp
        self.backend = backend
        self._L, self._N, self._area = L, N, area
        self._device = int(column.device_id) if backend == 'cuda' else None
        if backend == 'cuda':
            operator.check_device()
        self.catalog = catalog
        self._base = base
        self._Pa = base.n
        self._S = catalog.n_species
        self._live_rows = np.flatnonzero(base.status == _state.LIVE).astype(np.int64)
        self._Pl = int(self._live_rows.shape[0])
        self._has_plants = self._Pl > 0
        self._cadence = cadence
        self._anchor = anchor
        self._offset = anchor.utcoffset()
        self._tz = anchor.tzinfo
        self._time0 = float(base.time_days)
        self.root_builder = root_builder
        self._overrides = None
        if overrides is not None:
            if not isinstance(overrides, dict) or not overrides:
                raise IntereventError('layer_fraction_overrides must be a non-empty dict {plant_id: [L fractions]} or None')
            self._overrides = {int(k): tuple(float(x) for x in v) for k, v in overrides.items()}
        ss, _ = _sw._native()
        est = ss.estimate_step_memory(L, N, backend=backend, reserve_bytes=0)
        self._native_est = est
        self._native_resident_bytes = int(est.column_bytes + est.layer_bytes + est.parameter_bytes)
        self._table_bytes = int(_growth.table_memory(self._S, backend)['total_bytes'])
        # admission of the static + working state BEFORE any allocation
        Pa, Pl, S = self._Pa, self._Pl, self._S
        initial = (ROW_BYTES_PER_PLANT * Pa + 8 * Pl + 16 * S + self._table_bytes + self._native_resident_bytes
                   + 2 * 8 * L * N + 45 * Pl + self.reserve_bytes)
        self._check_stage('initialization', initial, {})
        side_dev = 'cuda' if backend == 'cuda' else None
        self._table = _growth.compile_species(catalog, backend, self.max_bytes, device=self._device if side_dev else None,
                                              extra_reserve_bytes=max(initial - self._table_bytes, 0))
        _growth.check_catalog(self._table, catalog)
        self._sidx = None
        self._live = None
        if self._has_plants:
            sp = np.ascontiguousarray(base.species_index[self._live_rows], dtype=np.int32)
            self._sidx = xp.asarray(sp) if backend == 'cuda' else sp
            lv = np.ones(Pl, dtype=np.bool_)
            self._live = xp.asarray(lv) if backend == 'cuda' else lv
        self._live_ids = np.ascontiguousarray(base.ids[self._live_rows], dtype=np.int64)
        self._native_anchor_t = native_anchor_t
        self._b = None

    def _init_boundary(self, native_state):
        ss, _ = _sw._native()
        column, xp, L, N = self._column, self._xp, self._L, self._N
        t_s = _sw._check_state(column, native_state, L, N)
        self._native_anchor_t = float(t_s)
        b = _Boundary()
        # native validity (finite, >= 0, <= saturation) through the bridge projection, admitted globally
        P0 = self._persistent(None)
        self._check_stage('initial_native_validity', _sum_items(P0, ('root_map',)) + 8 * L * N * 2 + 8 * L * N,
                          {})
        _sw.availability_snapshot(self._operator, native_state, max_bytes=self.max_bytes,
                                  extra_reserve_bytes=_sum_items(P0, ('native_resident',)))
        b.native_state = ss.column_state(column, native_state.water_m, native_state.t_s)   # OWNED copy
        Pl, S = self._Pl, self._S
        b.A, b.T = np.zeros(S), np.zeros(S)
        b.pending_days = Fraction(0)
        b.cursor = self._anchor
        b.next_boundary = self._anchor + self._cadence
        b.season_due = False
        b.sequence = 0
        b.last_receipt = None
        b.last_growth = None
        b.n_growth = b.n_rebuilds = 0
        b.root_map = None
        b.area = None
        if self._has_plants:
            Bh = np.ascontiguousarray(self._base.agb_dry_kg[self._live_rows], dtype=np.float64)
            Eh = np.ascontiguousarray(self._base.seasonal_agb_dry_kg[self._live_rows], dtype=np.float64)
            b.B, b.E = self._to_backend(Bh), self._to_backend(Eh)
            b.D, b.U = xp.zeros(Pl, dtype=np.float64), xp.zeros(Pl, dtype=np.float64)
            tmp = _Boundary()
            tmp.root_map = None
            rm, area = self._build_geometry(Bh, Eh, b.cursor, live_other=_sum_items(P0, ('root_map',)) + 16 * Pl,
                                            mem={}, stage='initial_roots')
            b.root_map, b.area = rm, area
        else:
            b.B = b.E = b.D = b.U = xp.zeros(0, dtype=np.float64)
        self._b = b

    def _to_backend(self, host_arr):
        return self._xp.asarray(host_arr) if self.backend == 'cuda' else np.array(host_arr, copy=True)

    # ------------------------------------------------------------------ properties
    @property
    def cursor(self):
        return self._b.cursor

    @property
    def utc_offset(self):
        return self._offset

    @property
    def native_state(self):
        '''The accepted native SoilColumnState (READ ONLY by contract: never write its array).'''
        return self._b.native_state

    @property
    def root_map(self):
        return self._b.root_map

    @property
    def last_receipt(self):
        return None if self._b.last_receipt is None else dict(self._b.last_receipt)

    @property
    def season_due(self):
        return bool(self._b.season_due)

    @property
    def pending_days(self):
        return self._b.pending_days

    @property
    def has_pending_growth(self):
        return self._b.pending_days > 0

    @property
    def n_live(self):
        return self._Pl

    @property
    def live_rows(self):
        '''int64 state-row index of each working row (copy).'''
        return self._live_rows.copy()

    @property
    def live_ids(self):
        return self._live_ids.copy()

    @property
    def sequence(self):
        return int(self._b.sequence)

    def boundaries(self):
        '''Boundaries the caller must split records at, from the current cursor.'''
        b = self._b
        c = b.cursor
        nxt_mid = _dt.datetime.combine(c.date() + _dt.timedelta(days=1), _dt.time(0), tzinfo=self._tz)
        return {'cursor': c, 'next_local_midnight': nxt_mid, 'next_growth_boundary': b.next_boundary,
                'next_new_year': _next_new_year(c), 'growth_cadence': self._cadence,
                'next_forced': min(x for x in (nxt_mid, b.next_boundary, _next_new_year(c)) if x is not None),
                'rule': 'split at every local midnight, every growth-grid point (next_growth_boundary + k * cadence) and '
                        'every new year; closed-gate and transport intervals need no splitting'}

    def split_record(self, record):
        '''Pieces of an OPEN record split at local midnights and growth-grid points (new years are midnights).'''
        if not isinstance(record, _forcing.TimestepRecord):
            raise IntereventError('record must be a leaf.forcing.TimestepRecord')
        if not record.active:
            return [record]
        cuts = list(_forcing.local_midnights_between(record.start, record.end))
        g = self._b.next_boundary
        while g < record.end:
            cuts.append(g)
            g = g + self._cadence
        return _forcing.split_record(record, cuts)

    def elapsed_days(self, cursor=None):
        '''Exact elapsed days (Fraction) from the session anchor to the cursor.'''
        c = self._b.cursor if cursor is None else cursor
        return _forcing.exact_seconds(c - self._anchor) / SECONDS_PER_DAY

    def describe(self):
        b = self._b
        return {'version': VERSION, 'scope': SCOPE, 'backend': self.backend, 'device': self._device,
                'n_plants_total': self._Pa, 'n_live': self._Pl, 'n_species': self._S, 'n_layers': self._L,
                'n_cells': self._N, 'cursor': b.cursor.isoformat(), 'next_growth_boundary': b.next_boundary.isoformat(),
                'season_due': bool(b.season_due), 'pending_days': str(b.pending_days), 'sequence': int(b.sequence),
                'growth_cadence_us': int(_forcing.exact_microseconds(_forcing.exact_seconds(self._cadence))),
                'root_builder': self.root_builder, 'max_edges': self.max_edges, 'max_bytes': self.max_bytes,
                'reserve_bytes': self.reserve_bytes,
                'n_root_edges': None if b.root_map is None else int(b.root_map.n_edges),
                'n_growth_updates': int(b.n_growth), 'n_rebuilds': int(b.n_rebuilds),
                'row_mapping': 'working row i is plant ids[live_rows[i]]; live plants only; state row order',
                'memory_scope': MEMORY_SCOPE}

    # ------------------------------------------------------------------ memory ledger
    def _persistent(self, cur):
        Pl, Pa, S, L, N = self._Pl, self._Pa, self._S, self._L, self._N
        rm = None if cur is None else cur.root_map
        return {'population_host': ROW_BYTES_PER_PLANT * Pa, 'row_mapping_host': 8 * Pl + 8 * Pl,
                'pending_integrals_host': 16 * S, 'species_tables': self._table_bytes,
                'native_resident': self._native_resident_bytes, 'native_state_accepted': 8 * L * N,
                'B': 8 * Pl, 'E': 8 * Pl, 'D': 8 * Pl, 'U': 8 * Pl, 'species_index': 4 * Pl, 'live': Pl,
                'canopy_area': 8 * Pl, 'root_map': 0 if rm is None else int(rm.resident_bytes),
                'caller_reserve': self.reserve_bytes}

    def _check_stage(self, stage, total, mem):
        mem[stage] = int(total)
        if int(total) > self.max_bytes:
            raise IntereventAdmissionError(stage, total, self.max_bytes, dict(mem))

    def _stages(self, cur, *, fire, carry, forcing_bytes, native_extra, n_edges=None):
        '''Global stage ledgers of one OPEN interval on persistent state `cur` (carry = superseded arrays still
        referenced by the accepted boundary). Returns {stage: {'reserve', 'total'}}.'''
        Pl, S, L, N, be = self._Pl, self._S, self._L, self._N, self.backend
        P0 = self._persistent(cur)
        one = 8 * L * N
        out = {}
        est = self._native_est
        out['native_step'] = {'reserve': 0, 'total': _sum_items(P0, ('native_state_accepted', 'native_resident')) + carry
                              + int(est.total_bytes) - 0}
        if not self._has_plants:
            return out
        R = L * N
        E = int(cur.root_map.n_edges) if n_edges is None else int(n_edges)
        dem = int(_growth.memory_estimate('demand', Pl, S, be, has_canopy_area=True)['total_bytes'])
        r = _sum_items(P0, ('B', 'species_index', 'live', 'canopy_area', 'species_tables')) + carry + forcing_bytes + one
        out['demand'] = {'reserve': r, 'total': r + dem}
        xr = _sum_items(P0, ('root_map', 'native_resident')) + carry + forcing_bytes
        br = _sw.estimate_bridge_bytes(Pl, E, L, N, map_backend=be, native_backend=be, demand_bytes=8 * Pl,
                                       map_resident_bytes=0 if cur.root_map is None else int(cur.root_map.resident_bytes),
                                       column_reserve_bytes=int(self._column.reserve_bytes), extra_reserve_bytes=xr)
        out['bridge'] = {'reserve': xr, 'total': int(br['total_bytes']), 'allocator_peak': int(br['allocator_peak_bytes'])}
        plan = int(_uptake.estimate_plan_bytes(Pl, E, R, be)['total_bytes'])
        fb = int(_uptake.estimate_feedback_bytes(Pl, E, R, be)['total_bytes'])
        out['accumulate'] = {'reserve': 0, 'total': _sum_items(P0) + carry + forcing_bytes + 2 * one + 24 * L * N
                             + plan + fb + 8 * Pl + 16 * Pl}
        if fire:
            out.update(self._grow_stages(cur, carry=carry, forcing_bytes=forcing_bytes, native_extra=native_extra + one))
        return out

    def _grow_stages(self, cur, *, carry, forcing_bytes, native_extra):
        Pl, Pa, S = self._Pl, self._Pa, self._S
        P0 = self._persistent(cur)
        g = int(_growth.memory_estimate('grow', Pl, S, self.backend)['total_bytes'])
        rg = (_sum_items(P0, ('B', 'E', 'species_index', 'live', 'species_tables')) + carry + forcing_bytes + native_extra
              + 16 * Pl + 24 * Pl + 8 * S * 10)
        host_xfer = 16 * Pl if self.backend == 'cuda' else 0
        pre = (_sum_items(P0) + carry + forcing_bytes + native_extra + 16 * Pl + ROW_BYTES_PER_PLANT * Pa + host_xfer
               + 192 * Pa + 16 * Pl)
        return {'grow': {'reserve': rg, 'total': rg + g}, 'rebuild_preflight': {'reserve': 0, 'total': pre}}

    # ------------------------------------------------------------------ geometry / snapshot
    def _materialize(self, B_live, E_live, cursor):
        '''Canonical PlantState at `cursor` from the private base arrays + working B / E (host arrays).'''
        base = self._base
        a = base._arrays(copy=True)
        if self._Pl:
            a['agb_dry_kg'][self._live_rows] = B_live
            a['seasonal_agb_dry_kg'][self._live_rows] = E_live
            a['age_days'][self._live_rows] = a['age_days'][self._live_rows] + float(self.elapsed_days(cursor))
        prov = dict(base.provenance)
        prov['interevent'] = {'version': VERSION, 'cursor': cursor.isoformat(), 'sequence': int(self._b.sequence if self._b else 0)}
        return _state.PlantState(base.domain, base.species_names, base.catalog_hash, ids=a['ids'],
                                 species_index=a['species_index'], x_m=a['x_m'], y_m=a['y_m'],
                                 establishment_ground_m=a['establishment_ground_m'],
                                 establishment_unresolved=a['establishment_unresolved'], agb_dry_kg=a['agb_dry_kg'],
                                 seasonal_agb_dry_kg=a['seasonal_agb_dry_kg'], age_days=a['age_days'],
                                 status=a['status'], time_days=self._time0 + float(self.elapsed_days(cursor)),
                                 provenance=prov)

    def snapshot(self):
        '''Canonical PlantState at the current cursor (explicit counted transfer; the clock is NOT advanced).'''
        b = self._b
        mem = {}
        P0 = self._persistent(b)
        self._check_stage('snapshot', _sum_items(P0) + 2 * ROW_BYTES_PER_PLANT * self._Pa + 16 * self._Pl, mem)
        if not self._has_plants:
            return self._materialize(None, None, b.cursor)
        return self._materialize(_hostarr(b.B), _hostarr(b.E), b.cursor)

    def export_canopy_view(self, current_surface, **kw):
        '''Shared canopy view of the snapshot at the cursor (ONE allometry implementation). Callers should
        flush_growth() first when a partial period is pending so that the exported canopy is current.'''
        return _canopy.canopy_view(self.snapshot(), self.catalog, current_surface, **kw)

    def write_snapshot(self, writer, which='maybe'):
        '''Write through an existing leaf.recording.SnapshotWriter: which in ('initial', 'maybe', 'final').'''
        st = self.snapshot()
        if which == 'initial':
            return writer.write_initial(st)
        if which == 'maybe':
            return writer.maybe_write(st)
        if which == 'final':
            return writer.write_final(st)
        raise IntereventError("which must be 'initial', 'maybe' or 'final'")

    def _build_geometry(self, Bh, Eh, cursor, *, live_other, mem, stage):
        '''Host geometry + root rebuild for the candidate B / E. live_other = every other live byte (old map
        included). Returns (RootMap, canopy_area backend array).'''
        Pl, Pa, xp = self._Pl, self._Pa, self._xp
        cand_bytes = ROW_BYTES_PER_PLANT * Pa + (16 * Pl if self.backend == 'cuda' else 0)
        self._check_stage(stage + '_geometry', live_other + cand_bytes + 192 * Pa + 16 * Pl, mem)
        cand = self._materialize(Bh, Eh, cursor)
        geo = _canopy.derive_geometry(cand, self.catalog)
        radius = np.ascontiguousarray(geo['radius_m'][self._live_rows], dtype=np.float64)
        del geo
        area_h = np.ascontiguousarray(np.pi * radius * radius, dtype=np.float64)
        del radius
        area = xp.asarray(area_h) if self.backend == 'cuda' else area_h
        budget = self.max_bytes - (live_other + cand_bytes + 8 * Pl)
        mem[stage + '_builder_budget'] = int(budget)
        if budget < 0:
            raise IntereventAdmissionError(stage + '_builder', live_other + cand_bytes + 8 * Pl, self.max_bytes, dict(mem))
        rm = _roots.build_root_map(cand, self.catalog, max_edges=self.max_edges, max_bytes=int(budget),
                                   backend=self.backend, builder=self.root_builder, statuses=('live',),
                                   layer_fraction_overrides=None if self._overrides is None else dict(self._overrides),
                                   provenance={'interevent_version': VERSION, 'cursor': cursor.isoformat()})
        del cand
        if not isinstance(rm, _uptake.RootMap) or rm.backend != self.backend or rm.n_plants != Pl \
                or rm.n_layers != self._L or rm.n_cells != self._N:
            raise IntereventError('the root builder returned a map that does not bind to the session')
        ids = _hostarr(rm.ids)
        if not np.array_equal(ids, self._live_ids):
            raise IntereventError('root map rows do not match the live-plant row mapping (ids differ)')
        if self.backend == 'cuda' and int(rm.device_id) != int(self._device):
            raise IntereventError('root map is bound to another CUDA device')
        br = _sw.estimate_bridge_bytes(Pl, rm.n_edges, self._L, self._N, map_backend=self.backend,
                                       native_backend=self.backend, demand_bytes=8 * Pl,
                                       map_resident_bytes=int(rm.resident_bytes),
                                       column_reserve_bytes=int(self._column.reserve_bytes), extra_reserve_bytes=0)
        if int(br['allocator_peak_bytes']) > int(rm.max_bytes):
            raise IntereventAdmissionError(stage + '_allocator_budget', br['allocator_peak_bytes'], rm.max_bytes, dict(mem))
        return rm, area

    # ------------------------------------------------------------------ transactions: common
    def _prepare_ids(self, b, kind, start, interval_id):
        rid = interval_id if interval_id is not None else f'seq{b.sequence + 1}:{kind}:{start.isoformat()}'
        if not isinstance(rid, str) or not rid:
            raise IntereventError('interval_id must be a non-empty string')
        if b.last_receipt is not None and b.last_receipt.get('interval_id') == rid:
            raise IntereventStateError(f'interval_id {rid!r} equals the last accepted receipt: replay refused')
        return rid

    def _check_times(self, b, start, end):
        _forcing.check_fixed_offset_datetime(start, 'start')
        _forcing.check_fixed_offset_datetime(end, 'end')
        if start.utcoffset() != self._offset or end.utcoffset() != self._offset:
            raise IntereventError(f'offsets must equal the session offset {self._offset}')
        if not end > start:
            raise IntereventError('end must be after start')
        if start != b.cursor:
            if end <= b.cursor:
                why = 'replay (this period was already accounted)'
            elif start < b.cursor:
                why = 'overlap with the accounted period'
            else:
                why = 'gap before the interval (no catch-up over missing time)'
            raise IntereventStateError(f'interval start {start.isoformat()} != cursor {b.cursor.isoformat()}: {why}')

    def _advance_boundary(self, nb, end):
        if end >= nb:
            k = (end - nb) // self._cadence + 1
            nb = nb + k * self._cadence
        return nb

    def _publish(self, work, receipt, mem):
        work.sequence = int(work.sequence) + 1
        receipt['sequence'] = int(work.sequence)
        receipt['memory'] = {'max_bytes': self.max_bytes, 'stage_totals': dict(mem),
                             'peak_estimate_bytes': max(mem.values()) if mem else 0, 'scope': MEMORY_SCOPE}
        work.last_receipt = _clean(receipt)
        self._b = work          # the ONE accepted-boundary replacement
        return dict(work.last_receipt)

    # ------------------------------------------------------------------ open interevent interval
    def step_interevent(self, record, *, added_water_m=None, potential_evaporation_m_s=None, interval_id=None):
        '''One accepted interevent interval (module docstring). Open gate: needs the explicit native forcing
        arrays of the column namespace, record.pet_mm_per_day and an unsplit-free record. Closed gate: a PAUSE
        (no native step; forcing must be None). Returns a compact receipt dict.'''
        b = self._b
        if not isinstance(record, _forcing.TimestepRecord):
            raise IntereventError('record must be a leaf.forcing.TimestepRecord')
        self._check_times(b, record.start, record.end)
        if not record.active:
            if added_water_m is not None or potential_evaporation_m_s is not None:
                raise IntereventError('a closed-gate record runs NO native hydrology: do not pass forcing arrays')
            return self._pause('closed_gate', record.start, record.end, None, interval_id, record=record)
        if added_water_m is None or potential_evaporation_m_s is None:
            raise IntereventError('an open record needs added_water_m [L, N] and potential_evaporation_m_s [N] '
                                  '(native forcing prepared by the caller / SYRUP)')
        if record.pet_mm_per_day is None and self._has_plants:
            raise IntereventError('the record needs the shared pet_mm_per_day for plant demand')
        column, L, N = self._column, self._L, self._N
        _sw._check_native_array(column, added_water_m, 'added_water_m', (L, N), np.float64)
        _sw._check_native_array(column, potential_evaporation_m_s, 'potential_evaporation_m_s', (N,), np.float64)
        if self.backend == 'cuda':
            self._operator.check_device()
        reason = self._unsplit_reason(record, b)
        if reason:
            raise IntereventBoundaryError(reason)
        rid = self._prepare_ids(b, 'interevent', record.start, interval_id)
        return self._open_interval(b, record, added_water_m, potential_evaporation_m_s, rid)

    def _unsplit_reason(self, record, b):
        mids = _forcing.local_midnights_between(record.start, record.end)
        if mids:
            return (f'record {record.start.isoformat()}..{record.end.isoformat()} crosses local midnight '
                    f'{mids[0].isoformat()}; split it (split_record) before submission')
        if b.next_boundary < record.end:
            return (f'record crosses the growth boundary {b.next_boundary.isoformat()}; split it (split_record) '
                    'before submission')
        return None

    def _open_interval(self, b, record, added, evap, rid):
        ss, _ = _sw._native()
        column, xp = self._column, self._xp
        L, N, Pl = self._L, self._N, self._Pl
        dt_s, dt_days_exact = float(record.dt_seconds_exact), record.dt_days_exact
        end = record.end
        ny = _next_new_year(record.start)
        year_end = ny is not None and end == ny
        fire = (end == b.next_boundary or year_end) and self._has_plants
        forcing_bytes = 8 * L * N + 8 * N
        season = bool(b.season_due and self._has_plants)
        mem = {}
        carry0 = (40 * Pl + (0 if b.root_map is None else int(b.root_map.resident_bytes))) if season else 0
        # ---- global pre-admission of every planned stage BEFORE any work
        plan = self._stages(b, fire=fire, carry=carry0, forcing_bytes=forcing_bytes, native_extra=0)
        for name, v in plan.items():
            self._check_stage(name, v['total'], mem)
        if season:
            sp = self._grow_stages(b, carry=0, forcing_bytes=0, native_extra=0)
            for name, v in sp.items():
                self._check_stage('season_' + name, v['total'], mem)
        work = b.clone()
        rep = {'kind': 'interevent', 'interval_id': rid, 'start': record.start.isoformat(), 'end': end.isoformat(),
               'dt_seconds_exact': str(record.dt_seconds_exact), 'dt_days_exact': str(dt_days_exact), 'gate': 'open',
               'temperature_c': record.temperature_c, 'pet_mm_per_day': record.pet_mm_per_day,
               'accounted_days_for_ages': float(dt_days_exact)}
        cur, carry = b, 0
        # ---- deferred seasonal turnover (zero water, zero integrals, no event duration)
        if season:
            old_alive = 32 * Pl + 8 * Pl + (0 if b.root_map is None else int(b.root_map.resident_bytes))
            self._growth_update(work, cur=b, carry=0, year_end=True, forcing_bytes=forcing_bytes, native_extra=0,
                                mem=mem, rep=rep, label='deferred_season', cursor=record.start)
            rep['deferred_season_applied'] = True
            cur, carry = work, old_alive
        work.season_due = False
        # ---- 1. native non-T hydrology
        t_native0 = float(work.native_state.t_s)
        step = ss.step(column, work.native_state, dt_s, added_water_m=added, potential_evaporation_m_s=evap)
        nt_state, nt_budget = step.state, dataclasses.asdict(step.budget)
        del step
        rep['native_non_transpiration'] = nt_budget
        if not self._has_plants:
            work.native_state = nt_state
            rep['transpiration'] = None
            rep['growth'] = {'performed': False, 'reason': 'no live plants'}
        else:
            plan = self._stages(cur, fire=fire, carry=carry, forcing_bytes=forcing_bytes, native_extra=0)
            # ---- 2. demand on the existing period geometry
            month, yd = record.start.month, _year_days(record.start)
            st = plan['demand']
            self._check_stage('demand', st['total'], mem)
            dem = _growth.demand(self._table, B_kg=work.B, canopy_area_m2=work.area, species_index=self._sidx,
                                 live=self._live, month=month, temperature_c=record.temperature_c,
                                 pet_mm_per_day=record.pet_mm_per_day, dt_days=float(dt_days_exact),
                                 max_bytes=self.max_bytes, extra_reserve_bytes=st['reserve'])
            D = dem['requested_m3']
            rep['demand_packet'] = dict(dem['packet'])
            del dem
            a_inc, t_inc = _growth.integral_increments(self._table, month=month, temperature_c=record.temperature_c,
                                                       dt_days=float(dt_days_exact), year_days=yd,
                                                       max_bytes=self.max_bytes,
                                                       extra_reserve_bytes=_sum_items(self._persistent(cur)) + carry
                                                       + forcing_bytes + 8 * L * N + 8 * Pl)
            # ---- 3. the REAL native withdrawal exactly once
            st = plan['bridge']
            self._check_stage('bridge', st['total'], mem)
            wd = _sw.withdraw_plants(work.root_map, D, self._operator, nt_state, max_bytes=self.max_bytes,
                                     extra_reserve_bytes=st['reserve'])
            if float(wd.native_state.t_s) != float(nt_state.t_s):
                raise IntereventError('the bridge changed the native time')
            tb = {k: float(getattr(wd.native.budget, k)) for k in _sw._BUDGET_FIELDS}
            cons = {k: v for k, v in wd.conservation.items() if k not in ('checks', 'allocator_closure', 'tolerance_terms_m3',
                                                                          'physical_debit', 'definition')}
            cons['checks_all_true'] = bool(all(wd.conservation['checks'].values()))
            if not wd.conservation['closed']:
                raise IntereventError('the bridge conservation did not close')
            # ---- 4. accumulate D / U / integrals on the backend (fresh arrays)
            self._check_stage('accumulate', plan['accumulate']['total'], mem)
            D_new, U_new = work.D + D, work.U + wd.delivered_m3
            new_native = wd.native_state
            del wd, D
            work.D, work.U, work.native_state = D_new, U_new, new_native
            work.A, work.T = work.A + a_inc, work.T + t_inc
            work.pending_days = work.pending_days + dt_days_exact
            del nt_state
            # ---- water closure across the two native stages (reported and enforced)
            bound = _sw._bound_fn()
            scale = max(abs(nt_budget['storage_before_m3']), abs(nt_budget['storage_after_m3']),
                        abs(tb['storage_before_m3']), abs(tb['storage_after_m3']), abs(nt_budget['added_m3']), 1e-300)
            cont = tb['storage_before_m3'] - nt_budget['storage_after_m3']
            tol_c = float(bound(8 * L * N + 32, scale)) + nt_budget['bound_m3'] + tb['bound_m3']
            if not abs(cont) <= tol_c:
                raise IntereventError(f'native stage continuity violated: non-T storage after {nt_budget["storage_after_m3"]} '
                                      f'!= T storage before {tb["storage_before_m3"]} (tolerance {tol_c})')
            total_res = ((tb['storage_after_m3'] - nt_budget['storage_before_m3'])
                         - (nt_budget['added_m3'] - nt_budget['surface_return_m3'] - nt_budget['drainage_m3']
                            - nt_budget['evaporation_m3'] - tb['actual_m3']))
            rep['transpiration'] = {'native_budget': tb, 'bridge': cons}
            rep['water_closure'] = {'storage_before_m3': nt_budget['storage_before_m3'],
                                    'storage_after_m3': tb['storage_after_m3'], 'added_m3': nt_budget['added_m3'],
                                    'surface_return_m3': nt_budget['surface_return_m3'],
                                    'drainage_m3': nt_budget['drainage_m3'], 'evaporation_m3': nt_budget['evaporation_m3'],
                                    'transpiration_actual_m3': tb['actual_m3'], 'stage_continuity_m3': cont,
                                    'residual_m3': total_res, 'stage_tolerance_m3': tol_c,
                                    'residual_within_tolerance': bool(abs(total_res) <= 2.0 * tol_c)}
            # ---- 5. growth boundary
            if fire:
                self._growth_update(work, cur=cur, carry=carry, year_end=year_end, forcing_bytes=forcing_bytes,
                                    native_extra=8 * L * N, mem=mem, rep=rep, label='growth', cursor=end)
            else:
                rep['growth'] = {'performed': False}
        work.next_boundary = self._advance_boundary(b.next_boundary, end) if end >= b.next_boundary else b.next_boundary
        work.cursor = end
        work.n_growth = work.n_growth
        rep['native_time_s'] = float(work.native_state.t_s)
        rep['native_time_advanced_s'] = float(work.native_state.t_s) - t_native0
        rep['next_growth_boundary'] = work.next_boundary.isoformat()
        rep['season_due'] = False
        return self._publish(work, rep, mem)

    # ------------------------------------------------------------------ growth update (+ rebuild)
    def _diagnostics(self, res, work, year_end):
        xp, S = self._xp, self._S
        sidx64 = self._sidx.astype(xp.int64)

        def ssum(a):
            return [float(v) for v in _hostarr(xp.bincount(sidx64, weights=a, minlength=S))]

        names = ('required_support_kg_year', 'requested_support_kg_year', 'actual_support_kg_year',
                 'maintenance_deficit_kg_year', 'normal_gain_kg', 'extra_gain_kg', 'seasonal_loss_kg', 'B_new_kg')
        per = {n: ssum(res[n]) for n in names}
        per['requested_m3'] = ssum(work.D)
        per['delivered_m3'] = ssum(work.U)
        cnt = _hostarr(xp.bincount(sidx64, minlength=S))
        out = {}
        for i, sid in enumerate(self._table.species_ids):
            out[sid] = {k: v[i] for k, v in per.items()}
            out[sid]['n_live'] = int(cnt[i]) if i < len(cnt) else 0
        return out

    def _growth_update(self, work, *, cur, carry, year_end, forcing_bytes, native_extra, mem, rep, label, cursor):
        xp, Pl, L, N = self._xp, self._Pl, self._L, self._N
        gs = self._grow_stages(cur, carry=carry, forcing_bytes=forcing_bytes, native_extra=native_extra)
        self._check_stage(label + '_grow', gs['grow']['total'], mem)
        self._check_stage(label + '_rebuild_preflight', gs['rebuild_preflight']['total'], mem)
        res = _growth.grow(self._table, B_kg=work.B, E_kg=work.E, species_index=self._sidx, live=self._live,
                           requested_m3=work.D, delivered_m3=work.U, activity_integral=work.A,
                           temperature_integral=work.T, year_end=bool(year_end), max_bytes=self.max_bytes,
                           extra_reserve_bytes=gs['grow']['reserve'])
        pk, led = dict(res['packet']), dict(res['ledger'])
        if not led.get('population_closed', False):
            raise IntereventError('the aboveground population ledger did not close')
        B2, E2 = res['B_new_kg'].copy(), res['E_new_kg'].copy()    # copy out of the shared [12, P] buffer
        per_species = self._diagnostics(res, work, year_end)
        del res
        period_days = float(work.pending_days)
        unchanged = (pk['sum_seasonal_loss_kg'] == 0.0 and pk['sum_normal_gain_kg'] == 0.0
                     and pk['sum_extra_gain_kg'] == 0.0 and period_days == 0.0)
        info = {'performed': not unchanged, 'label': label, 'year_end': bool(year_end), 'period_days': period_days,
                'packet': pk, 'ledger': led, 'per_species': per_species}
        if unchanged:
            rep[label if label != 'growth' else 'growth'] = info
            return
        Bh, Eh = _hostarr(B2), _hostarr(E2)
        live_other = (_sum_items(self._persistent(cur)) + carry + forcing_bytes + native_extra + 16 * Pl)
        rm, area = self._build_geometry(Bh, Eh, cursor, live_other=live_other, mem=mem, stage=label + '_roots')
        # the next interval must also fit with the NEW map (otherwise refuse now, old boundary intact)
        tmp = _Boundary()
        tmp.root_map = rm
        nxt = self._stages(tmp, fire=False, carry=0, forcing_bytes=8 * L * N + 8 * N, native_extra=0)
        for name in ('demand', 'bridge', 'accumulate'):
            self._check_stage('next_' + name, nxt[name]['total'], mem)
        work.B, work.E, work.root_map, work.area = B2, E2, rm, area
        work.D, work.U = xp.zeros(Pl, dtype=np.float64), xp.zeros(Pl, dtype=np.float64)
        work.A, work.T = np.zeros(self._S), np.zeros(self._S)
        work.pending_days = Fraction(0)
        work.n_growth = int(work.n_growth) + 1
        work.n_rebuilds = int(work.n_rebuilds) + 1
        info['rebuild'] = {'n_edges': int(rm.n_edges), 'n_plants': int(rm.n_plants), 'builder': self.root_builder,
                           'n_inactive_rows': int(rm.n_inactive_plants),
                           'builder_budget_bytes': mem.get(label + '_roots_builder_budget')}
        work.last_growth = _clean({k: v for k, v in info.items()})
        rep[label if label != 'growth' else 'growth'] = info

    def flush_growth(self, interval_id=None):
        '''Finish a partial growth period AT THE CURRENT CURSOR (no clock advance, no new water, grid unchanged) so
        native consumers can export the current canopy. A no-op receipt when nothing is pending.'''
        b = self._b
        rid = self._prepare_ids(b, 'flush', b.cursor, interval_id)
        rep = {'kind': 'flush', 'interval_id': rid, 'start': b.cursor.isoformat(), 'end': b.cursor.isoformat(),
               'accounted_days_for_ages': 0.0}
        mem = {}
        if not self._has_plants or b.pending_days == 0:
            rep['growth'] = {'performed': False, 'reason': 'nothing pending'}
            rep['published'] = False        # a no-op flush accounts nothing and changes nothing
            return _clean(rep)
        work = b.clone()
        self._growth_update(work, cur=b, carry=0, year_end=False, forcing_bytes=0, native_extra=0, mem=mem, rep=rep,
                            label='growth', cursor=b.cursor)
        return self._publish(work, rep, mem)

    # ------------------------------------------------------------------ transport / pause
    def advance_transport(self, kind, start, end, *, native_state=None, interval_id=None, label=None):
        '''Advance the calendar over a transport event (wind / rain / recession). Requires NO pending growth; no T,
        growth, E or catch-up. native_state is required for rain / recession (the external native owner's result)
        and must be None for wind (water unchanged).'''
        if not isinstance(kind, str) or kind.lower() not in TRANSPORT_KINDS:
            raise IntereventError(f'transport kind must be one of {TRANSPORT_KINDS} (interevent records use '
                                  'step_interevent), got {kind!r}')
        kind = kind.lower()
        b = self._b
        self._check_times(b, start, end)
        if kind in EXTERNAL_WATER_KINDS and native_state is None:
            raise IntereventError(f'{kind!r} needs the external native owner\'s resulting SoilColumnState '
                                  '(native_state=...): LEAF runs no storm physics and does not replace physical water')
        if kind == 'wind' and native_state is not None:
            raise IntereventError('wind leaves native water unchanged: native_state must be None')
        return self._pause(kind, start, end, native_state, interval_id, label=label)

    def _pause(self, kind, start, end, native_state, interval_id, record=None, label=None):
        b = self._b
        if b.pending_days != 0:
            raise IntereventStateError('pending growth must be flushed (flush_growth) before a transport / paused interval')
        rid = self._prepare_ids(b, kind, start, interval_id)
        ss, _ = _sw._native()
        column, L, N = self._column, self._L, self._N
        dt_s = float(_forcing.exact_seconds(end - start))
        mem = {}
        work = b.clone()
        one = 8 * L * N
        P0 = self._persistent(b)
        t_expected = float(b.native_state.t_s) + dt_s
        if native_state is None:
            if not math.isfinite(t_expected):
                raise IntereventError('native time would not be finite')
            work.native_state = ss.SoilColumnState(b.native_state.water_m, t_expected)   # water unchanged (shared, never written)
            water_changed = False
        else:
            t_in = _sw._check_state(column, native_state, L, N)
            tol = TIME_TOL_ABS_S + TIME_TOL_REL * abs(t_expected)
            if abs(float(t_in) - t_expected) > tol:
                raise IntereventError(f'external native state time {t_in} != expected end time {t_expected} (tolerance {tol})')
            self._check_stage('transport_native_validity', _sum_items(P0) + 3 * one, mem)
            if self.backend == 'cuda':
                self._operator.check_device()
            _sw.availability_snapshot(self._operator, native_state, max_bytes=self.max_bytes,
                                      extra_reserve_bytes=_sum_items(P0, ('native_resident',)) + one)
            self._check_stage('transport_native_copy', _sum_items(P0) + 2 * one, mem)
            work.native_state = ss.column_state(column, native_state.water_m, native_state.t_s)
            water_changed = True
        ny = _next_new_year(start)
        season = ny is not None and ny <= end
        if season:
            work.season_due = True
        skipped = 0
        if end >= work.next_boundary:
            nb = self._advance_boundary(work.next_boundary, end)
            skipped = int((nb - work.next_boundary) // self._cadence)
            work.next_boundary = nb
        work.cursor = end
        rep = {'kind': kind, 'interval_id': rid, 'start': start.isoformat(), 'end': end.isoformat(),
               'dt_seconds_exact': str(_forcing.exact_seconds(end - start)),
               'dt_days_exact': str(_forcing.exact_seconds(end - start) / SECONDS_PER_DAY),
               'gate': 'closed' if record is not None else 'transport', 'label': label,
               'accounted_days_for_ages': float(_forcing.exact_seconds(end - start) / SECONDS_PER_DAY),
               'growth_boundaries_skipped': skipped, 'season_due': bool(work.season_due),
               'season_deferred_this_interval': bool(season), 'native_water_replaced_by_external_owner': water_changed,
               'native_time_s': float(work.native_state.t_s), 'next_growth_boundary': work.next_boundary.isoformat(),
               'no_transpiration_growth_or_evaporation': True}
        return self._publish(work, rep, mem)

    # ------------------------------------------------------------------ restart (implemented in leaf.checkpoint)
    def save(self, path, *, overwrite=False):
        from leaf import checkpoint
        return checkpoint.save_session(self, path, overwrite=overwrite)

    @classmethod
    def load(cls, path, catalog, **kwargs):
        from leaf import checkpoint
        return checkpoint.load_session(path, catalog, **kwargs)


__all__ = ['VERSION', 'BACKENDS', 'TRANSPORT_KINDS', 'SCOPE', 'MEMORY_SCOPE', 'IntereventError', 'IntereventStateError',
           'IntereventBoundaryError', 'IntereventAdmissionError', 'IntereventSession', 'native_fingerprint']
