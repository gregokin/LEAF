'''leaf.plant_session: public WATER-FREE daily biological session for an external coordinator (MASY). UNQUALIFIED, UNCALIBRATED.

LEAF owns plants, roots, potential demand, allocation, biological accumulators and growth. SYRUP owns the sole physical water inventory and
EVERY water update; MASY supplies all forcing / temporal controls, sequences the calls and accepts the combined state. This module holds NO
water, calls NO hydrology / evaporation / withdrawal and hides no sequence. Reused unchanged: `leaf.growth` (demand, demand_per_plant, grow,
integral_increments), `leaf.uptake` (propose / accept), `leaf.roots` / `leaf.canopy` (through `biocalendar.rebuild_geometry`), `leaf.plant_pet`,
`leaf.biocalendar`, and the EXISTING Phase 2 / 3 public evaluators `leaf.disturbance.evaluate_event`, `leaf.demography.decide`,
`leaf.recruitment.evaluate` with their counters, annual accumulators, keyed RNG counters, event binding and deferred-season order exactly as
`leaf.interevent.IntereventSession` applies them. No ecological law, parameter or response was added or changed.

Immutable parent + staged candidates. A `PlantSession` is an ACCEPTED, immutable parent. Every biology-changing operation is PURE: it returns a
`Staged` candidate (the complete next session inside) or raises, leaving the parent untouched. `parent.publish(staged)` returns the new accepted
session exactly once; stale parents, other lineages, discarded / published candidates and a second publication are refused before adoption.

Identity. `PlantSession.token` binds session id, a per-start random `lineage`, the sequence, calendar, biological counters, the resource cursor, the
last stage, the content hash of the accepted plant population (`pop_sha256`) and of the accepted root / canopy geometry (`geometry_sha256`).
Every plan carries a random nonce and is consumed at publication; proposals and accepted feedback are tied to plan, nonce, parent token and the
SYRUP read / debit identities. No population-sized readback happens per day (hashes are taken at start, restore and the cold rebuilds).

Public operations (names are the stable contract; see docs/masy-daily-plant-interface.md)
  PlantSession.start(base, config, anchor=..., day_binding=...)      accepted parent at a nominal local-midnight `anchor`
  prepare_daily(day, terrain_pet=...)                                 1. demand D [m3] for ONE open fragment -> DailyPlan (pure)
  propose_withdrawal(plan, read)                                      2. root-weighted proposal against a FRESH `AvailabilityRead` (m3)
  stage_day(proposal, accepted) / stage_dry_day(plan)                 3+4. actual feedback, credit, scheduled growth (no parent change)
  stage_flush(), stage_transport(tday), stage_deferred_season()       calendar boundaries
  stage_surface_event(event, surface), stage_annual_decision(year),
  stage_reproduction(year, surface, ...), stage_decline_reproduction(year)   Phase 2 / 3 boundaries (existing evaluators)
  publish(staged) / Staged.discard()                                  5. publish once / discard
  export_roots() / export_canopy() / snapshot_state()                 6. accepted geometry and plant state
  boundaries() / counters() / due_boundaries() / phase2_state() / phase3_state()   8. boundaries and counters
Biological-only save / restore: `leaf.plant_checkpoint`.

Nominal time. Date, month, year length and age derive ONLY from the MASY-supplied nominal fixed-offset instants and exact calendar elapsed time;
NEVER from native storm / wind / recession seconds. A nominal day is covered by `DayContext` OPEN fragments (`start`, optional `duration`; demand and
integrals use the ACTUAL enabled duration) and `TransportDay` fragments (any duration, calendar only: age advances, no E / T / growth / credit).
Several open fragments may share ONE SYRUP daily operation: each needs a fresh availability read after the preceding T request (same daily
ordinal, same hydrology count) - never a second hydrology step. The growth cadence is whole days 1..7 (weekly Euler envelope).

Boundaries. The accepted plant `Domain` carries the per-axis `periodic` flags (`leaf.state.Domain`, docs/periodic-boundaries.md). Every
geometry build of this session - start, growth / season rebuilds, Phase 2 / 3 rebinds, restore - reads them from that Domain through the
shared builders (`biocalendar.rebuild_geometry`), so roots and canopy-area PET weights wrap across periodic seams consistently (the demand
canopy area stays the nominal allometric pi r^2), and the population identity (`pop_sha256`) binds the Domain including its flags. A periodic daughter coordinate is wrapped by
`leaf.recruitment` (the only recruitment change). No water, hydrology, schedule or forcing is added: SYRUP stays the sole water authority.

Flags. transpiration_enabled=True / growth_enabled=False: real debit, per-plant actual feedback, NO new D / U / activity / temperature / time credit
(earned credit kept; flush before the disabled period). transpiration_enabled=False with growth_enabled=True is REFUSED (the qualified C1 rule).
'''
from __future__ import annotations

import copy
import dataclasses
import datetime as _dt
import hashlib
import json
import math
import types
import uuid
from fractions import Fraction

import numpy as np

from leaf import biocalendar as _bio
from leaf import demography as _demo
from leaf import disturbance as _dist
from leaf import events as _events
from leaf import forcing as _forcing
from leaf import growth as _growth
from leaf import lifecycle_bridge as _lb
from leaf import periodic as _periodic
from leaf import plant_pet as _pp
from leaf import recruitment as _rec
from leaf import roots as _roots
from leaf import state as _state
from leaf import traits as _traits
from leaf import uptake as _uptake

VERSION = '0.2.0'
BACKENDS = ('numpy', 'numba', 'cuda')
TRANSPORT_KINDS = ('wind', 'rain', 'recession', 'closed_gate')
TRANSPORT_SOURCE = {'wind': 'maple_wind', 'rain': 'syrup_water', 'recession': 'syrup_water'}
EVENT_TIME_TOL_DAYS = 1e-9
FP64_EPS = float(np.finfo(np.float64).eps)
SECONDS_PER_DAY = 86400
PER_LIVE_BYTES = 45            # B, E, D, U, canopy area (8 each) + species index 4 + live mask 1
ROW_BYTES_PER_PLANT = 62
MATERIALIZE_BYTES_PER_PLANT = _bio.MATERIALIZE_BYTES_PER_PLANT
GEOMETRY_BYTES_PER_PLANT = _roots.GEOMETRY_BYTES_PER_PLANT
STAGE_KINDS = ('day', 'dry_day', 'flush', 'transport', 'season', 'event', 'annual', 'reproduction', 'reproduction_declined', 'connect')
PHASE_SCOPE = ('UNQUALIFIED / UNCALIBRATED: the existing Phase 2 (disturbance, demography) and Phase 3 (reproduction) evaluators of leaf.* '
               'applied through public staged calls; aboveground dry biomass only; no new law or parameter')


class PlantSessionError(ValueError):
    '''Any session validation error (raised before a result exists; the accepted parent is intact).'''


class PlantSessionStateError(PlantSessionError):
    '''Stale / wrong parent, duplicate publication or debit, wrong resource lineage, discarded candidate.'''


class PlantSessionBoundaryError(PlantSessionError):
    '''A calendar / ecological boundary must be handled first (pending growth, due annual decision, wrong date).'''


class PlantSessionUnsupportedError(PlantSessionError):
    '''A combination the existing qualified rules refuse (never silently altered).'''


class PlantSessionAdmissionError(PlantSessionError):
    '''The global byte ledger would exceed `max_bytes`; raised BEFORE the stage allocates.'''


def _strict_bool(v, name):
    if not isinstance(v, (bool, np.bool_)):
        raise PlantSessionError(f'{name} must be a bool, got {v!r}')
    return bool(v)


def _nbytes(est):
    if isinstance(est, dict):
        return int(est['total_bytes']) if 'total_bytes' in est else int(
            sum(int(v) for v in est.values() if isinstance(v, (int, np.integer)) and not isinstance(v, bool)))
    return int(est)


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False, default=str).encode()).hexdigest()


def _hostarr(a):
    return np.array(a.get(), copy=True) if hasattr(a, 'get') else np.array(a, copy=True)


def _next_new_year(t):
    return _bio.next_new_year(t)


# ------------------------------------------------------------------ typed value objects
@dataclasses.dataclass(frozen=True)
class DayContext:
    '''ONE OPEN interevent fragment of a nominal day supplied by MASY: `start` (fixed-offset instant), optional `duration` (default: until the
    next local midnight, i.e. a whole day when `start` is a midnight), the fragment's mean temperature and the independent transpiration /
    growth flags. Demand, integrals and credit use exactly this ENABLED duration; excluded time (wind / rain / recession) is expressed with
    `TransportDay` fragments, never by shrinking the nominal calendar.'''
    start: _dt.datetime
    temperature_c: float
    transpiration_enabled: bool
    growth_enabled: bool
    process_kind: str = 'interevent'
    duration: object = None

    def __post_init__(self):
        try:
            _forcing.check_fixed_offset_datetime(self.start, 'start')
            _forcing.check_process_kind(self.process_kind)
        except _forcing.ForcingError as exc:
            raise PlantSessionError(str(exc)) from exc
        t = self.temperature_c
        if isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, float, np.integer, np.floating)) \
                or not math.isfinite(float(t)) or float(t) < -273.15:
            raise PlantSessionError(f'temperature_c must be a finite real >= -273.15, got {t!r}')
        object.__setattr__(self, 'temperature_c', float(t))
        _strict_bool(self.transpiration_enabled, 'transpiration_enabled')
        _strict_bool(self.growth_enabled, 'growth_enabled')
        if self.growth_enabled and not self.transpiration_enabled:
            raise PlantSessionUnsupportedError(
                'transpiration_enabled=False with growth_enabled=True is refused by the qualified rule (MASY leaf_water_interval refuses it too)')
        mid = _dt.datetime.combine(self.start.date() + _dt.timedelta(days=1), _dt.time(0), tzinfo=self.start.tzinfo)
        if self.duration is None:
            object.__setattr__(self, 'duration', mid - self.start)
        d = self.duration
        if not isinstance(d, _dt.timedelta) or not _dt.timedelta(0) < d:
            raise PlantSessionError('duration must be a positive datetime.timedelta')
        if self.start + d > mid:
            raise PlantSessionBoundaryError('an open fragment must not cross a local midnight (split at the nominal day boundary)')

    @property
    def end(self):
        return self.start + self.duration

    @property
    def nominal_date(self):
        return self.start.date()

    def record(self, pet_mm_per_day=None):
        return _forcing.TimestepRecord(self.start, self.end, self.temperature_c, pet_mm_per_day=pet_mm_per_day,
                                       process_kind='interevent', gate='open')


@dataclasses.dataclass(frozen=True)
class TransportDay:
    '''ONE transport fragment (wind / rain / recession / closed gate) of ANY duration: calendar only, no E / T / growth / catch-up.'''
    start: _dt.datetime
    kind: str
    duration: object = None

    def __post_init__(self):
        try:
            _forcing.check_fixed_offset_datetime(self.start, 'start')
        except _forcing.ForcingError as exc:
            raise PlantSessionError(str(exc)) from exc
        if self.kind not in TRANSPORT_KINDS:
            raise PlantSessionError(f'kind must be one of {TRANSPORT_KINDS}, got {self.kind!r}')
        if self.duration is None:
            object.__setattr__(self, 'duration', _dt.timedelta(days=1))
        if not isinstance(self.duration, _dt.timedelta) or not _dt.timedelta(0) < self.duration:
            raise PlantSessionError('duration must be a positive datetime.timedelta')

    @property
    def end(self):
        return self.start + self.duration


@dataclasses.dataclass(frozen=True)
class DayBinding:
    '''Explicit MASY nominal-day <-> SYRUP daily-run mapping: occurrence ordinal k (0-based, the LAST consumed ordinal of the cursor) belongs to the
    nominal date `origin_date + k days`; `run_digest` is the SYRUP daily run. Native seconds are not part of the mapping.'''
    run_digest: str
    origin_date: str

    def __post_init__(self):
        _lb._hex64(self.run_digest, 'run_digest')
        try:
            _dt.date.fromisoformat(self.origin_date)
        except (TypeError, ValueError) as exc:
            raise PlantSessionError(f'origin_date must be an ISO date: {exc}') from exc

    def ordinal_of(self, date):
        return (date - _dt.date.fromisoformat(self.origin_date)).days

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class StageId:
    '''Serializable identity of one stage (resume without replay: compare with `PlantSession.last_stage_id`).'''
    session_id: str
    parent_token: str
    sequence: int
    kind: str
    nominal_start: str
    nominal_end: str
    plan_digest: str
    debit_identity: object = None
    resource_before: object = None
    resource_after: object = None

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d):
        if not isinstance(d, dict) or set(d) != {f.name for f in dataclasses.fields(cls)}:
            raise PlantSessionError('stage id record keys mismatch (closed key set)')
        if d['kind'] not in STAGE_KINDS:
            raise PlantSessionError(f'unknown stage kind {d["kind"]!r}')
        return cls(**d)

    def digest(self):
        return _sha(self.to_dict())


@dataclasses.dataclass(frozen=True)
class SessionConfig:
    '''Immutable construction parameters (copies of mutable inputs are frozen; nothing is a hidden default). `canopy` (max_edges, max_bytes,
    optional valid_cell_mask / builder) is required for per-plant canopy PET. `max_bytes` is the ONE global ledger (host + device);
    `reserve_bytes` the caller's other live state and must not exceed it. `rng_seed` is REQUIRED when the catalog has an active mortality or
    reproduction law, `patch_size_m` when a seed reproduction law is active (as `IntereventSession`).'''
    catalog: object
    table: object
    backend: str
    max_edges: int
    max_bytes: int
    cadence_days: int
    reserve_bytes: int = 0
    builder: str = 'compiled'
    canopy: object = None
    layer_fraction_overrides: object = None
    rng_seed: object = None
    patch_size_m: object = None

    def __post_init__(self):
        if not isinstance(self.catalog, _traits.SpeciesCatalog):
            raise PlantSessionError('catalog must be a leaf.traits.SpeciesCatalog')
        try:
            _growth.check_catalog(self.table, self.catalog)
        except _growth.GrowthError as exc:
            raise PlantSessionError(str(exc)) from exc
        if self.backend not in BACKENDS or self.table.backend != self.backend:
            raise PlantSessionError(f'backend must be one of {BACKENDS} and equal the compiled table backend ({self.table.backend!r})')
        for n in ('max_edges', 'max_bytes', 'reserve_bytes'):
            v = getattr(self, n)
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or int(v) < 0:
                raise PlantSessionError(f'{n} must be a non-negative integer')
        if int(self.reserve_bytes) > int(self.max_bytes):
            raise PlantSessionAdmissionError(f'reserve_bytes {self.reserve_bytes} exceeds max_bytes {self.max_bytes}; refused before any allocation')
        c = self.cadence_days
        if isinstance(c, (bool, np.bool_)) or not isinstance(c, (int, np.integer)) or not 1 <= int(c) <= 7:
            raise PlantSessionError('cadence_days must be an integer in 1..7 (whole days; weekly-or-shorter Euler envelope)')
        if self.canopy is not None:
            if not isinstance(self.canopy, (dict, types.MappingProxyType)) or 'max_edges' not in self.canopy \
                    or 'max_bytes' not in self.canopy:
                raise PlantSessionError('canopy must be a dict with max_edges and max_bytes (or None for scalar PET only)')
            cc = dict(self.canopy)
            m = cc.get('valid_cell_mask')
            if m is not None:
                m = np.array(m, copy=True)
                m.flags.writeable = False
                cc['valid_cell_mask'] = m
            object.__setattr__(self, 'canopy', types.MappingProxyType(cc))
        if self.layer_fraction_overrides is not None:
            ov = self.layer_fraction_overrides
            if not isinstance(ov, (dict, types.MappingProxyType)) or not ov:
                raise PlantSessionError('layer_fraction_overrides must be a non-empty dict {plant_id: [L fractions]} or None')
            ov = dict(ov)
            try:
                _roots._validate_overrides(ov, np.array(sorted(ov), dtype=np.int64), len(next(iter(ov.values()))))
            except (_roots.RootsError, _traits.CatalogError, TypeError, ValueError) as exc:
                raise PlantSessionError(f'layer_fraction_overrides refused: {exc}') from exc
            object.__setattr__(self, 'layer_fraction_overrides', types.MappingProxyType(
                {int(k): tuple(float(x) for x in v) for k, v in ov.items()}))

    @property
    def cadence(self):
        return _dt.timedelta(days=int(self.cadence_days))

    def canopy_options(self):
        return None if self.canopy is None else dict(self.canopy)

    def identity(self):
        '''Declared BIOLOGY configuration that must not change silently across a restart: catalog hash, growth cadence, the ACTUAL keyed-RNG seed,
        the ecological patch size, the per-individual root layer overrides (all ids, also currently non-live plants) and whether / with which
        valid-cell mask the canopy map is built. Deliberately excluded (caller-adjustable on restore): byte and edge ceilings, the compiled / NumPy
        builder choice (identical algorithms, protected by the geometry hash) and the backend (restricted by the compiled table at construction).'''
        mask = None if self.canopy is None else self.canopy.get('valid_cell_mask')
        return {'catalog_hash': self.catalog.hash, 'cadence_days': int(self.cadence_days),
                'rng_seed': None if self.rng_seed is None else int(self.rng_seed),
                'patch_size_m': None if self.patch_size_m is None else float(self.patch_size_m).hex(),
                'root_overrides': None if not self.layer_fraction_overrides else
                [[int(k), [float(x).hex() for x in v]] for k, v in sorted(self.layer_fraction_overrides.items())],
                'canopy': None if self.canopy is None else {
                    'valid_cell_mask_sha256': None if mask is None else hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()}}

    def identity_sha256(self):
        h = getattr(self, '_idh', None)
        if h is None:
            h = _sha(self.identity())
            object.__setattr__(self, '_idh', h)
        return h


@dataclasses.dataclass(frozen=True, eq=False)
class _Laws:
    '''Compiled Phase 2 / 3 species tables (S-sized, shared by every successor) and the validated seed / patch.'''
    dist: object
    demo: object
    repro: object
    patch: object
    seed: object
    demo_on: bool
    stress_on: bool
    repro_on: bool
    bytes: int


@dataclasses.dataclass(frozen=True, eq=False)
class _Pop:
    '''Live-plant working arrays + geometry of one accepted instant (private; arrays are never mutated in place).'''
    B: object
    E: object
    D: object
    U: object
    A: object
    T: object
    sidx: object
    live: object
    live_rows: object
    live_ids: object
    geometry: object
    geom_sha: str
    generation: int
    stress: object = None
    ann_req: object = None
    ann_act: object = None
    ann_plan: object = None
    ann_gain: object = None


@dataclasses.dataclass(frozen=True, eq=False)
class DailyPlan:
    '''Pure demand plan of one open fragment: bound to the parent token, the fragment and the effective population; carries a nonce.'''
    parent_token: str
    sequence: int
    day: DayContext
    step: object
    season_applied: bool
    pop: object
    D: object
    packet: dict
    pet_kind: str
    month: int
    year_days: int
    plan_digest: str
    increments: object
    nonce: str
    _used: list = dataclasses.field(default_factory=list)

    @property
    def needs_withdrawal(self):
        return self.day.transpiration_enabled

    def report(self):
        return {'sequence': self.sequence, 'start': self.day.start.isoformat(), 'end': self.day.end.isoformat(), 'pet_kind': self.pet_kind,
                'season_turnover_first': self.season_applied, 'growth_fires': bool(self.step.fire_growth),
                'year_end': bool(self.step.year_end), 'demand_packet': dict(self.packet), 'plan_digest': self.plan_digest,
                'nonce': self.nonce}


@dataclasses.dataclass(frozen=True, eq=False)
class WithdrawalProposal:
    '''Root-weighted proposal against ONE fresh availability read. `proposed_m3` [R] feeds `lifecycle_bridge.request_depth_m`.'''
    parent_token: str
    plan: object
    read: object
    uptake_plan: object
    identity: str
    retained_feedback_bytes: int

    @property
    def proposed_m3(self):
        return self.uptake_plan.proposed

    def report(self):
        s = self.uptake_plan.summary
        return {'identity': self.identity, 'retained_feedback_bytes': self.retained_feedback_bytes,
                'proposed_total_m3': float(s['proposed_total_m3']), 'demand_total_m3': float(s['demand_total_m3']),
                'n_resources_limited': int(s['n_resources_limited'])}


@dataclasses.dataclass(frozen=True, eq=False)
class DailyFeedback:
    '''Per-plant ACTUAL result of an accepted withdrawal (arrays [P] on the backend; `live_ids` host int64).'''
    live_ids: object
    requested_m3: object
    delivered_m3: object
    unmet_m3: object
    satisfaction: object
    totals: dict
    closure: dict


@dataclasses.dataclass(frozen=True, eq=False)
class Staged:
    '''One staged, unpublished candidate: the complete next session, its serializable `stage_id` and a scalar report.'''
    stage_id: StageId
    parent_token: str
    session_id: str
    next: object
    feedback: object
    report: dict
    plan: object = None
    _status: list = dataclasses.field(default_factory=lambda: ['staged'])

    @property
    def status(self):
        return self._status[0]

    def discard(self):
        '''Drop the candidate: it can no longer be published; the accepted parent is untouched.'''
        if self._status[0] == 'published':
            raise PlantSessionStateError('this candidate was already published and cannot be discarded')
        self._status[0] = 'discarded'


def _xp(cfg):
    return _growth._cupy() if cfg.backend == 'cuda' else np


def _frozen(cfg, a):
    if cfg.backend != 'cuda':
        a.flags.writeable = False
    return a


def _dev(cfg, host):
    h = np.ascontiguousarray(host)
    return _growth._cupy().asarray(h) if cfg.backend == 'cuda' else h.copy()


def _geom_sha(geo):
    h = hashlib.sha256()
    for src in (geo.root_map.host_arrays(), {} if geo.canopy_map is None else geo.canopy_map.host_arrays(),
                {'canopy_area_m2': _hostarr(geo.canopy_area_m2)}):
        for k in sorted(src):
            a = np.ascontiguousarray(src[k])
            h.update(k.encode() + str(a.dtype).encode() + str(a.shape).encode())
            h.update(a.tobytes())
    return h.hexdigest()


def _pop_sha(base, catalog_hash):
    h = hashlib.sha256(catalog_hash.encode() + json.dumps(base.domain.to_dict(), sort_keys=True, default=str).encode())
    notlive = np.asarray(base.status) != _state.LIVE
    for k in _state.PlantState.ARRAY_NAMES:
        a = getattr(base, k)
        if k in ('agb_dry_kg', 'seasonal_agb_dry_kg', 'age_days'):     # dynamic for live rows (carried by the working arrays): frozen rows only
            a = a[notlive]
        a = np.ascontiguousarray(a)
        h.update(k.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


def _compile_laws(cfg, reserve):
    '''Phase 2 / 3 tables exactly as `IntereventSession._setup` (S-sized; an inactive catalog holds no array).'''
    dev = cfg.table.device if cfg.backend == 'cuda' else None
    try:
        dist = _dist.compile_species(cfg.catalog, cfg.backend, device=dev, max_bytes=cfg.max_bytes, extra_reserve_bytes=reserve)
        demo = _demo.compile_species(cfg.catalog, cfg.backend, device=dev, max_bytes=cfg.max_bytes,
                                     extra_reserve_bytes=reserve + dist.resident_bytes())
        repro = _rec.compile_species(cfg.catalog, max_bytes=cfg.max_bytes,
                                     extra_reserve_bytes=reserve + dist.resident_bytes() + demo.resident_bytes())
    except (_dist.DisturbanceBudgetError, _demo.DemographyBudgetError, _rec.ReproductionBudgetError) as exc:
        raise PlantSessionAdmissionError(f'phase 2 / 3 tables exceed max_bytes: {exc}') from exc
    except (_dist.DisturbanceError, _demo.DemographyError, _rec.ReproductionError) as exc:
        raise PlantSessionError(f'species catalog Phase 2 / 3 laws refused: {exc}') from exc
    demo_on, repro_on = bool(demo.any_active), bool(repro.any_active)
    if (demo_on or repro_on) and cfg.rng_seed is None:
        raise PlantSessionError('the catalog has an active mortality / reproduction law: an explicit integer rng_seed is REQUIRED')
    try:
        patch = None if cfg.patch_size_m is None else _rec.PatchConfig(cfg.patch_size_m)
        seed = None if cfg.rng_seed is None else _demo.check_seed(cfg.rng_seed)
    except (_rec.ReproductionError, _demo.DemographyError) as exc:
        raise PlantSessionError(str(exc)) from exc
    if repro.any_seed and patch is None:
        raise PlantSessionError('a seed reproduction law is active: an explicit patch_size_m is REQUIRED')
    return _Laws(dist, demo, repro, patch, seed, demo_on, bool(demo.any_stress), repro_on,
                 int(dist.resident_bytes() + demo.resident_bytes() + repro.resident_bytes()))


def _initial_bio(anchor):
    return {'n_annual': 0, 'n_disturbance': 0, 'n_population_rebinds': 0, 'annual_due': None, 'annual_cursor': anchor.isoformat(),
            'transport_pending': None, 'last_decision': None, 'n_reproduction': 0, 'repro_last_year': int(anchor.year) - 1,
            'repro_start_year': int(anchor.year), 'last_reproduction': None}


_COUNTERS = ('days_open', 'days_dry', 'days_transport', 'withdrawals_accepted', 'growth_updates', 'season_turnovers', 'flushes',
             'year_ends', 'geometry_rebuilds')


# ------------------------------------------------------------------ the accepted parent
@dataclasses.dataclass(frozen=True, eq=False)
class PlantSession:
    '''ACCEPTED immutable biological parent. Construct with `PlantSession.start`; never mutate. See the module docstring.'''
    config: SessionConfig
    laws: _Laws
    session_id: str
    lineage: str
    sequence: int
    base: object
    base_offset_days: Fraction
    pop: _Pop
    schedule: object
    counters_: dict
    bio: dict
    resource_cursor: object
    last_stage_id: object
    day_binding: object
    pop_sha: str
    token: str
    _succ: list = dataclasses.field(default_factory=list)

    # ---------------------------------------------------------------- construction
    @staticmethod
    def _map_scratch(n_plants, n_resources, backend):
        '''Minimum root-map resource scratch (the existing `uptake` map estimator; a conservative 16 B per resource if its shape differs).'''
        try:
            return _nbytes(_uptake.estimate_map_bytes(int(n_plants), 0, int(n_resources), backend))
        except Exception:                                    # pragma: no cover - estimator signature differences
            return 16 * int(n_resources) + 24 * int(n_plants)

    @classmethod
    def _lower_bound(cls, config, n_all, n_live_bound, ncells):
        '''Metadata-only lower bound of start / restore: reserve, base copy + materialize + geometry peak (all plants), working arrays and
        mask scratch (live bound), species + Phase 2/3-independent tables and the minimum root resource scratch (L * N resources).'''
        S = int(config.table.n_species)
        R = int(config.catalog.root_layers) * int(ncells)
        return (int(config.reserve_bytes) + (ROW_BYTES_PER_PLANT + MATERIALIZE_BYTES_PER_PLANT + GEOMETRY_BYTES_PER_PLANT
                                             + _roots.MASK_BYTES_PER_PLANT) * int(n_all)
                + (PER_LIVE_BYTES + 16) * int(n_live_bound) + 16 * S + _nbytes(_growth.table_memory(S, config.backend))
                + _nbytes(config.table.memory()) + cls._map_scratch(n_live_bound, R, config.backend))

    @staticmethod
    def _check_initial(binding, initial):
        if not isinstance(binding, DayBinding):
            raise PlantSessionError('an initial resource needs an explicit DayBinding (nominal origin + genuine daily run)')
        if not isinstance(initial, _lb.ResourceIdentity):
            raise PlantSessionError('initial_resource must be the ResourceIdentity of the SYRUP state whose T history is known consumed')
        if initial.daily_run != binding.run_digest or initial.daily_pending:
            raise PlantSessionError('the initial resource is not of the bound daily run / has a pending storm; refused')

    @classmethod
    def start(cls, base, config, *, anchor, session_id=None, day_binding=None, initial_resource=None):
        '''Accepted parent at the nominal local-midnight `anchor` (cursor = anchor, nothing pending, geometry from the shared builders). Every
        stage is admitted against the global ledger (incl. the caller reserve) BEFORE population-sized arrays exist. WATER connection needs
        BOTH `day_binding` (nominal origin + genuine daily run) and `initial_resource` (the ResourceIdentity of the SYRUP state whose T history
        - request sequence and T count - is KNOWN consumed): without them the session is transport-only and `propose_withdrawal` refuses. They
        may also be supplied later with `stage_connect_water`.'''
        if not isinstance(config, SessionConfig):
            raise PlantSessionError('config must be a SessionConfig')
        if not isinstance(base, _state.PlantState):
            raise PlantSessionError('base must be a leaf.state.PlantState')
        if base.catalog_hash != config.catalog.hash:
            raise PlantSessionError('the plant state catalog hash differs from the session catalog')
        try:
            _forcing.check_fixed_offset_datetime(anchor, 'anchor')
        except _forcing.ForcingError as exc:
            raise PlantSessionError(str(exc)) from exc
        if (anchor.hour, anchor.minute, anchor.second, anchor.microsecond) != (0, 0, 0, 0):
            raise PlantSessionBoundaryError('the anchor must be a local midnight')
        if day_binding is not None and not isinstance(day_binding, DayBinding):
            raise PlantSessionError('day_binding must be a DayBinding or None')
        sid = uuid.uuid4().hex if session_id is None else session_id
        if type(sid) is not str or not sid or len(sid) > 64 or not sid.isalnum():
            raise PlantSessionError('session_id must be an alphanumeric string of 1..64 characters')
        if initial_resource is not None:
            cls._check_initial(day_binding, initial_resource)
        # metadata-only admission BEFORE any population-sized operation (no status mask, rows, B / E, validation or copy exists yet): the
        # conservative bound counts EVERY plant as live; the live count is taken only after this admission
        lower = cls._lower_bound(config, int(base.n), int(base.n), base.domain.ncells)
        if lower > int(config.max_bytes):
            raise PlantSessionAdmissionError(f'start: lower bound {lower} bytes (incl. reserve {config.reserve_bytes}) exceeds '
                                             f'max_bytes={config.max_bytes}; refused before any allocation')
        Pa, S = int(base.n), int(config.table.n_species)
        base.require_resolved('starting a plant session')
        base.validate()
        n_live = int(np.count_nonzero(base.status == _state.LIVE))
        owned = base.copy()
        laws = _compile_laws(config, int(config.reserve_bytes) + ROW_BYTES_PER_PLANT * Pa)
        rows = np.flatnonzero(owned.status == _state.LIVE).astype(np.int64)
        resident_other = int(config.reserve_bytes) + ROW_BYTES_PER_PLANT * Pa + laws.bytes + (_demo.PERSISTENT_BYTES_PER_LIVE * n_live
                                                                                              if laws.stress_on else 0)
        pop = cls._build_pop(config, laws, owned, rows, np.array(owned.agb_dry_kg[rows]), np.array(owned.seasonal_agb_dry_kg[rows]),
                             reserve=resident_other + PER_LIVE_BYTES * n_live, generation=0)
        counters = {k: 0 for k in _COUNTERS}
        return cls._make(config, laws, sid, uuid.uuid4().hex, 0, owned, Fraction(0), pop, _bio.new_schedule(anchor, config.cadence),
                         counters, _initial_bio(anchor), initial_resource, None, day_binding, _pop_sha(owned, config.catalog.hash))

    @classmethod
    def _build_pop(cls, cfg, laws, base, rows, B_host, E_host, *, reserve, generation, D=None, U=None, A=None, T=None, stress=None,
                   ann=None, elapsed_days=0.0):
        ids = np.ascontiguousarray(base.ids[rows], dtype=np.int64)
        ov = None
        if cfg.layer_fraction_overrides:
            ov = {int(k): list(v) for k, v in cfg.layer_fraction_overrides.items() if int(k) in set(ids.tolist())} or None
        geo = _bio.rebuild_geometry(base, cfg.catalog, B_host, E_host, elapsed_days=float(elapsed_days), max_edges=cfg.max_edges,
                                    max_bytes=cfg.max_bytes, reserve_bytes=int(reserve), backend=cfg.backend, builder=cfg.builder,
                                    layer_fraction_overrides=ov, live_rows=rows, canopy=cfg.canopy_options(),
                                    provenance={'module': 'leaf.plant_session', 'version': VERSION})
        P, S = int(rows.shape[0]), int(cfg.table.n_species)
        sidx = _dev(cfg, np.ascontiguousarray(base.species_index[rows], dtype=np.int32))
        live = _dev(cfg, np.ones(P, dtype=np.bool_))
        xp = _xp(cfg)
        z = lambda n, dt=np.float64: _dev(cfg, np.zeros(n, dtype=dt))
        if laws.stress_on:
            st = z(P, np.int32) if stress is None else stress
            ar, ac, ap, ag = (z(P), z(P), z(P), z(P)) if ann is None else ann
        else:
            st = ar = ac = ap = ag = None
        rows_ro = np.array(rows, dtype=np.int64)
        rows_ro.flags.writeable = False
        ids.flags.writeable = False
        return _Pop(_frozen(cfg, _dev(cfg, B_host)), _frozen(cfg, _dev(cfg, E_host)), z(P) if D is None else D,
                    z(P) if U is None else U, z(S) if A is None else A, z(S) if T is None else T, _frozen(cfg, sidx),
                    _frozen(cfg, live), rows_ro, ids, geo, _geom_sha(geo), int(generation), st, ar, ac, ap, ag)

    @classmethod
    def _make(cls, config, laws, sid, lineage, sequence, base, offset, pop, schedule, counters, bio, cursor, last, binding, pop_sha):
        body = {'session_id': sid, 'lineage': lineage, 'sequence': int(sequence), 'schedule': schedule.to_dict(),
                'generation': pop.generation, 'geometry_sha256': pop.geom_sha, 'pop_sha256': pop_sha, 'counters': counters, 'bio': bio,
                'cursor': None if cursor is None else cursor.to_dict(), 'last': None if last is None else last.to_dict(),
                'catalog': config.catalog.hash, 'config': config.identity_sha256(), 'n_live': int(pop.live_rows.shape[0]),
                'binding': None if binding is None else binding.to_dict()}
        return cls(config, laws, sid, lineage, int(sequence), base, Fraction(offset), pop, schedule, dict(counters), dict(bio), cursor,
                   last, binding, pop_sha, _sha(body))

    @classmethod
    def restore_accepted(cls, config, state, *, session_id, lineage, sequence, schedule, base_offset_days, D, U, A, T, counters, bio,
                         resource_cursor, last_stage_id, day_binding, geometry_generation, stress, ann, expected_geometry_sha256,
                         expected_pop_sha256):
        '''Rebuild an accepted parent from saved biology (`plant_checkpoint.load`): `state` is the canonical PlantState AT the schedule cursor
        (it becomes the new base), geometry is rebuilt by the shared builders and MUST reproduce the saved geometry and population identities.'''
        Pa, S = int(state.n), int(config.table.n_species)
        lower = cls._lower_bound(config, Pa, Pa, state.domain.ncells)         # metadata only: every plant counted live, before any mask
        if lower > int(config.max_bytes):
            raise PlantSessionAdmissionError(f'restore: lower bound {lower} bytes exceeds max_bytes={config.max_bytes}; refused')
        rows = np.flatnonzero(state.status == _state.LIVE).astype(np.int64)
        P = int(rows.shape[0])
        if D.shape != (P,) or U.shape != (P,) or A.shape != (S,) or T.shape != (S,):
            raise PlantSessionError('saved pending arrays do not match the live population / species table')
        if not (np.all(np.isfinite(D)) and np.all(np.isfinite(U)) and np.all(D >= 0) and np.all(U >= 0)
                and np.all(U <= D * (1.0 + _growth.MAX_OVERDELIVERY_RTOL) + 0.0)):
            raise PlantSessionError('saved pending D / U are not finite, non-negative with U <= D (consistency check failed)')
        if not (np.all(np.isfinite(A)) and np.all(np.isfinite(T)) and np.all(A >= 0) and np.all(T >= 0) and np.all(T <= A)
                and (A.size == 0 or float(A.max()) <= _growth.MAX_ACTIVITY_INTEGRAL * (1.0 + _growth.ACTIVITY_INTEGRAL_SLACK))):
            raise PlantSessionError('saved pending activity / temperature integrals are inconsistent')
        laws = _compile_laws(config, int(config.reserve_bytes) + ROW_BYTES_PER_PLANT * Pa)
        if pop_sha_check := _pop_sha(state, config.catalog.hash):
            if expected_pop_sha256 is not None and pop_sha_check != expected_pop_sha256:
                raise PlantSessionStateError('the restored plant population does not reproduce the saved population identity; refused')
        if laws.stress_on != (stress is not None):
            raise PlantSessionError('saved stress / annual accumulators do not match the catalog laws (active mortality needs them)')
        reserve = int(config.reserve_bytes) + ROW_BYTES_PER_PLANT * Pa + laws.bytes + PER_LIVE_BYTES * P
        pop = cls._build_pop(config, laws, state, rows, np.array(state.agb_dry_kg[rows]), np.array(state.seasonal_agb_dry_kg[rows]),
                             reserve=reserve, generation=geometry_generation, D=_frozen(config, _dev(config, D)),
                             U=_frozen(config, _dev(config, U)), A=_frozen(config, _dev(config, A)), T=_frozen(config, _dev(config, T)),
                             stress=None if stress is None else _dev(config, stress),
                             ann=None if ann is None else tuple(_dev(config, a) for a in ann))
        if pop.geom_sha != expected_geometry_sha256:
            raise PlantSessionStateError('the rebuilt geometry does not reproduce the saved geometry identity (wrong catalog, domain, '
                                         'builder, overrides or corrupted state); refused')
        return cls._make(config, laws, session_id, lineage, sequence, state, Fraction(base_offset_days), pop, schedule, counters, bio,
                         resource_cursor, last_stage_id, day_binding, pop_sha_check)

    # ---------------------------------------------------------------- read-only views
    @property
    def n_live(self):
        return int(self.pop.live_rows.shape[0])

    @property
    def cursor(self):
        return self.schedule.cursor

    @property
    def geometry_sha256(self):
        return self.pop.geom_sha

    @property
    def pop_sha256(self):
        return self.pop_sha

    def counters(self):
        return dict(self.counters_)

    def elapsed_days(self):
        '''Exact calendar days since the base instant (a Fraction); drives live ages. Native seconds never enter.'''
        return _bio.elapsed_days(self.schedule.anchor, self.schedule.cursor) - self.base_offset_days

    def _time_days_at(self, instant):
        return float(self.base.time_days) + float(_bio.elapsed_days(self.schedule.anchor, instant) - self.base_offset_days)

    def boundaries(self):
        '''Calendar boundaries (`biocalendar.boundaries`) plus the flags that make a due boundary explicit.'''
        b = _bio.boundaries(self.schedule)
        b.update({'season_due': self.schedule.season_due, 'pending_growth_days': self.schedule.pending_days,
                  'needs_flush': _bio.needs_flush(self.schedule), **self.due_boundaries()})
        return b

    def due_boundaries(self):
        '''Explicit gates: the annual decision year, a transport interval awaiting its disturbance decision, the next reproduction
        opportunity year (None when not due / not active) and the deferred season.'''
        bio, cy = self.bio, int(self.schedule.cursor.year)
        nxt = int(bio['repro_last_year']) + 1
        return {'annual_decision_due_year': bio['annual_due'], 'transport_awaiting_disturbance': bio['transport_pending'],
                'reproduction_opportunity_year': nxt if self.laws.repro_on and nxt <= cy - 1 else None,
                'deferred_season_due': bool(self.schedule.season_due)}

    def needs_flush(self):
        return _bio.needs_flush(self.schedule)

    def flush_required(self, *, next_growth_enabled, next_is_transport):
        return _bio.flush_required(self.schedule, next_growth_enabled=next_growth_enabled, next_is_transport=next_is_transport)

    def phase2_state(self):
        b = self.bio
        return {'scope': PHASE_SCOPE, 'demography_active': self.laws.demo_on, 'stress_active': self.laws.stress_on,
                'disturbance_active': bool(self.laws.dist.any_active), 'rng_seed_set': self.laws.seed is not None,
                'n_annual_decisions': int(b['n_annual']), 'n_disturbance_events': int(b['n_disturbance']),
                'n_population_rebinds': int(b['n_population_rebinds']), 'annual_due_year': b['annual_due'],
                'annual_period_start': b['annual_cursor'], 'transport_pending': b['transport_pending'],
                'last_decision': b['last_decision']}

    def phase3_state(self):
        b = self.bio
        return {'scope': PHASE_SCOPE, 'reproduction_active': self.laws.repro_on, 'n_reproduction_opportunities': int(b['n_reproduction']),
                'last_opportunity_year': b['repro_last_year'], 'last_reproduction': b['last_reproduction'],
                'patch': None if self.laws.patch is None else self.laws.patch.to_dict(), 'n_total_records': int(self.base.n)}

    def stress_counters(self):
        return None if self.pop.stress is None else _hostarr(self.pop.stress)

    def annual_accumulators(self):
        p = self.pop
        if p.ann_req is None:
            return None
        return {'required_support_kg_year': _hostarr(p.ann_req), 'actual_support_kg_year': _hostarr(p.ann_act),
                'planned_growth_kg': _hostarr(p.ann_plan), 'actual_growth_kg': _hostarr(p.ann_gain)}

    def live_ids(self):
        return np.array(self.pop.live_ids, dtype=np.int64)

    def geometry_bytes(self):
        g = self.pop.geometry
        cm = 0 if g.canopy_map is None else int(g.canopy_map.resident_bytes)
        return int(g.root_map.resident_bytes) + cm + ROW_BYTES_PER_PLANT * int(g.state.n)     # + the retained GeometryRebuild.state

    def resident_bytes(self):
        '''Structural live bytes of this accepted parent: working arrays, live_rows / ids, stress + annual arrays, species tables (host + device),
        Phase 2 / 3 tables, the base population, the retained geometry (root map, canopy map, candidate PlantState). Borrowed caller inputs are
        excluded explicitly (they belong to `reserve_bytes`).'''
        P = self.n_live
        return (PER_LIVE_BYTES * P + 16 * P + (_demo.PERSISTENT_BYTES_PER_LIVE * P if self.laws.stress_on else 0)
                + 16 * int(self.config.table.n_species) + _nbytes(self.config.table.memory()) + self.laws.bytes
                + ROW_BYTES_PER_PLANT * int(self.base.n) + self.geometry_bytes())

    def candidate_bytes(self):
        '''Bytes of a coexisting CANDIDATE population (old and new coexist): arrays, geometry, base copy and the materialize / geometry peak.'''
        return (self.resident_bytes() - ROW_BYTES_PER_PLANT * int(self.base.n) + (MATERIALIZE_BYTES_PER_PLANT + GEOMETRY_BYTES_PER_PLANT
                                                                                    + ROW_BYTES_PER_PLANT) * int(self.base.n))

    def boundary(self):
        '''Boundary declaration of the session's plant domain (`leaf.periodic.boundary_record`): the per-axis `periodic` flags every
        geometry rebuild (growth, season, rebind, restore) reads from the base `Domain`. Metadata only; nothing is allocated.'''
        return _periodic.boundary_record(self.base.domain)

    def export_roots(self):
        '''Accepted root information: the immutable `RootMap` (plant-major CSR, fixed weights), live ids, generation and content hash.
        A periodic domain adds `boundary` (the wrapped footprints are already inside the map's unique plant-cell-layer edges).'''
        out = {'root_map': self.pop.geometry.root_map, 'live_ids': self.live_ids(), 'generation': self.pop.generation,
               'geometry_sha256': self.pop.geom_sha}
        if self.base.domain.any_periodic:
            out['boundary'] = self.boundary()
        return out

    def export_canopy(self):
        '''Accepted canopy information: per-live-plant canopy area [P] m2 (borrowed backend array), sparse `CanopyMap` (or None), live ids.
        A periodic domain adds `boundary`. `canopy_area_m2` is the nominal allometric pi r^2 demand area (unchanged for every boundary);
        the `canopy_map` areas / denominator are the physical (torus-union) PET support, one entry per plant-cell pair.'''
        g = self.pop.geometry
        out = {'canopy_area_m2': g.canopy_area_m2, 'canopy_map': g.canopy_map, 'live_ids': self.live_ids(),
               'generation': self.pop.generation, 'geometry_sha256': self.pop.geom_sha}
        if self.base.domain.any_periodic:
            out['boundary'] = self.boundary()
        return out

    def snapshot_state(self):
        '''Canonical PlantState at the cursor (explicit host copies of B / E; ages advanced by the exact calendar elapsed time).'''
        return _bio.materialize_state(self.base, self.pop.live_rows, _hostarr(self.pop.B), _hostarr(self.pop.E),
                                      elapsed_days=float(self.elapsed_days()))

    def pending(self):
        '''Host copies of the pending credit: D, U [P] (m3) and A, T [S] (years). Explicit transfer; for checkpoints and diagnostics.'''
        p = self.pop
        return {'D': _hostarr(p.D), 'U': _hostarr(p.U), 'A': _hostarr(p.A), 'T': _hostarr(p.T)}

    # ---------------------------------------------------------------- admission
    def _admit(self, stage, transient, extra=0, *, candidate=False):
        need = (int(self.config.reserve_bytes) + self.resident_bytes() + (self.candidate_bytes() if candidate else 0) + int(transient)
                + int(extra))
        if need > int(self.config.max_bytes):
            raise PlantSessionAdmissionError(f'{stage}: estimated {need} bytes exceed max_bytes={self.config.max_bytes}; refused before '
                                             'allocation (no bound is raised, no plant dropped)')
        return need

    def _live_other(self, extra=0):
        '''Reserve passed to a component that allocates beside everything live: caller reserve + accepted parent + candidate + extra.'''
        return int(self.config.reserve_bytes) + self.resident_bytes() + self.candidate_bytes() + int(extra)

    def _check_token(self, obj_token, what):
        if obj_token != self.token:
            raise PlantSessionStateError(f'{what} was prepared for another accepted parent (stale or foreign lineage); refused')
        if self._succ:
            raise PlantSessionStateError('this parent already has a published successor; stage from the successor (duplicate or stale use refused)')

    def _rtol(self, root_map):
        r = float(_uptake.overdelivery_bound_rel(root_map)) + 16.0 * FP64_EPS
        if not 0.0 <= r <= _growth.MAX_OVERDELIVERY_RTOL:
            raise PlantSessionError(f'allocator over-delivery bound {r!r} is outside the growth module range')
        return r

    def _elapsed_at(self, instant):
        return float(_bio.elapsed_days(self.schedule.anchor, instant) - self.base_offset_days)

    def _zero_acc(self, n):
        z = lambda dt=np.float64: _dev(self.config, np.zeros(n, dtype=dt))
        return z(np.int32), z(), z(), z(), z()

    # ---------------------------------------------------------------- growth / rebuild (pure)
    def _grow_pop(self, pop, *, year_end, instant):
        cfg = self.config
        P = int(pop.live_rows.shape[0])
        mem = _nbytes(_growth.memory_estimate('grow', P, cfg.table.n_species, cfg.backend))
        self._admit('growth', mem + 24 * P, 32 * P if pop.ann_req is not None else 0, candidate=True)
        out = _growth.grow(cfg.table, B_kg=pop.B, E_kg=pop.E, species_index=pop.sidx, live=pop.live, requested_m3=pop.D, delivered_m3=pop.U,
                           activity_integral=np.ascontiguousarray(_hostarr(pop.A)), temperature_integral=np.ascontiguousarray(_hostarr(pop.T)),
                           year_end=bool(year_end), max_bytes=int(cfg.max_bytes), extra_reserve_bytes=self._live_other(),
                           overdelivery_rtol=self._rtol(pop.geometry.root_map))
        if not out['ledger'].get('population_closed', False):
            raise PlantSessionError('the aboveground population ledger did not close')
        Bn, En = out['B_new_kg'].copy(), out['E_new_kg'].copy()
        ann = None
        if pop.ann_req is not None:
            ann = (pop.ann_req + out['required_support_kg_year'], pop.ann_act + out['actual_support_kg_year'],
                   pop.ann_plan + (out['normal_plan_kg'] + out['extra_plan_kg']), pop.ann_gain + (out['normal_gain_kg'] + out['extra_gain_kg']))
        info = {'ledger': dict(out['ledger']), 'packet': dict(out['packet']), 'year_end': bool(year_end)}
        del out
        Bh, Eh = _hostarr(Bn), _hostarr(En)
        new = self._build_pop(cfg, self.laws, self.base, pop.live_rows, Bh, Eh, reserve=self._live_other(16 * P), generation=pop.generation + 1,
                              stress=pop.stress, ann=ann, elapsed_days=self._elapsed_at(instant))
        return new, info

    # ---------------------------------------------------------------- 1. prepare
    def _gate_prepare(self):
        if self._succ:
            raise PlantSessionStateError('this parent already has a published successor; prepare from the successor')
        if self.bio['annual_due'] is not None and self.laws.demo_on and self.n_live:
            raise PlantSessionBoundaryError(f'the annual decision for year {self.bio["annual_due"]} is due: call stage_annual_decision('
                                            f'{self.bio["annual_due"]}) and publish before the next open fragment')

    def prepare_daily(self, day, *, terrain_pet):
        '''Pure demand plan of one open fragment. `terrain_pet` is the shared SYRUP terrain PET FIELD [N] mm/day float64 (canopy-area-weighted mean
        per plant, `leaf.plant_pet`) or a scalar mm/day (existing `growth.demand`). Demand uses the fragment's ACTUAL enabled duration
        (`dt_days <= 7`); with transpiration disabled D is exactly 0. A season turnover deferred by a year end is applied first (zero water).'''
        if not isinstance(day, DayContext):
            raise PlantSessionError('day must be a DayContext')
        cfg = self.config
        self._gate_prepare()
        scalar = isinstance(terrain_pet, (int, float, np.integer, np.floating)) and not isinstance(terrain_pet, (bool, np.bool_))
        if day.transpiration_enabled and not scalar and self.pop.geometry.canopy_map is None:
            raise PlantSessionError('a per-plant canopy PET needs SessionConfig.canopy (no site-mean fallback)')
        rec = day.record(pet_mm_per_day=float(terrain_pet) if scalar else None)
        try:
            step = _bio.open_step(self.schedule, rec, growth_enabled=day.growth_enabled, has_plants=self.n_live > 0)
        except _bio.BioCalendarBoundaryError as exc:
            raise PlantSessionBoundaryError(str(exc)) from exc
        except _bio.BioCalendarError as exc:
            raise PlantSessionBoundaryError(str(exc)) from exc
        P = self.n_live
        pop, season = self.pop, False
        if step.season_first:
            pop, _ = self._grow_pop(pop, year_end=True, instant=day.start)
            season = True
        dt = float(rec.dt_days_exact)
        month, yd = day.start.month, _bio.year_days(day.start)
        if not day.transpiration_enabled:
            D = _dev(cfg, np.zeros(P, dtype=np.float64))
            packet, kind = {'n_live': P, 'n_demanding': 0, 'sum_requested_m3': 0.0, 'max_requested_m3': 0.0}, 'disabled'
        else:
            op = 'demand' if scalar else 'demand_per_plant'
            est = _nbytes(_growth.memory_estimate(op, P, cfg.table.n_species, cfg.backend)) + 8 * P
            if not scalar:
                cm = pop.geometry.canopy_map
                est += _nbytes(_pp.estimate_pet_bytes(cm.n_plants, cm.n_edges, cm.backend))
            self._admit('prepare_daily', est, 8 * P + 16 * int(cfg.table.n_species), candidate=season)
            if scalar:
                r = _growth.demand(cfg.table, B_kg=pop.B, canopy_area_m2=pop.geometry.canopy_area_m2, species_index=pop.sidx, live=pop.live,
                                   month=month, temperature_c=day.temperature_c, pet_mm_per_day=float(terrain_pet), dt_days=dt,
                                   max_bytes=int(cfg.max_bytes), extra_reserve_bytes=self._live_other() if season else self._live_other() -
                                   self.candidate_bytes())
                kind = 'scalar'
            else:
                other = self._live_other() if season else self._live_other() - self.candidate_bytes()
                pp = _pp.canopy_pet(pop.geometry.canopy_map, terrain_pet, max_bytes=int(cfg.max_bytes), reserve_bytes=other)
                r = _growth.demand_per_plant(cfg.table, B_kg=pop.B, canopy_area_m2=pop.geometry.canopy_area_m2, species_index=pop.sidx,
                                             live=pop.live, month=month, temperature_c=day.temperature_c,
                                             pet_mm_per_day_per_plant=pp['pet_mm_per_day'], dt_days=dt, max_bytes=int(cfg.max_bytes),
                                             extra_reserve_bytes=other + 8 * P)
                kind = 'canopy_area_weighted'
            D, packet = r['requested_m3'], dict(r['packet'])
        a_inc, t_inc = _growth.integral_increments(cfg.table, month=month, temperature_c=day.temperature_c, dt_days=dt, year_days=yd,
                                                   max_bytes=int(cfg.max_bytes))
        _frozen(cfg, D)
        nonce = uuid.uuid4().hex
        digest = _sha({'parent': self.token, 'start': day.start.isoformat(), 'end': day.end.isoformat(), 'T': day.temperature_c,
                       'flags': [day.transpiration_enabled, day.growth_enabled], 'kind': kind, 'packet': packet, 'season': season,
                       'generation': pop.generation, 'geometry': pop.geom_sha, 'seq': self.sequence + 1, 'nonce': nonce})
        return DailyPlan(self.token, self.sequence + 1, day, step, season, pop, D, packet, kind, month, yd, digest,
                         (_dev(cfg, a_inc), _dev(cfg, t_inc)), nonce)

    def rebuild_plan(self, day, *, D_host, packet, pet_kind, plan_digest, nonce, scalar_pet=None):
        '''Reconstruct a saved `DailyPlan` (checkpoint resume; NO demand recomputed, no PET needed). The recomputed digest (parent token, fragment,
        flags, packet, generation, geometry, nonce) must equal the saved one.'''
        if not isinstance(day, DayContext):
            raise PlantSessionError('day must be a DayContext')
        cfg = self.config
        self._gate_prepare()
        P = self.n_live
        if D_host.shape != (P,) or D_host.dtype != np.float64 or not np.all(np.isfinite(D_host)) or (P and float(D_host.min()) < 0.0):
            raise PlantSessionError('saved demand does not match the live population (finite, non-negative float64 [P])')
        rec = day.record(pet_mm_per_day=None if scalar_pet is None else float(scalar_pet))
        try:
            step = _bio.open_step(self.schedule, rec, growth_enabled=day.growth_enabled, has_plants=P > 0)
        except _bio.BioCalendarError as exc:
            raise PlantSessionBoundaryError(str(exc)) from exc
        pop, season = self.pop, False
        if step.season_first:
            pop, _ = self._grow_pop(pop, year_end=True, instant=day.start)
            season = True
        digest = _sha({'parent': self.token, 'start': day.start.isoformat(), 'end': day.end.isoformat(), 'T': day.temperature_c,
                       'flags': [day.transpiration_enabled, day.growth_enabled], 'kind': pet_kind, 'packet': packet, 'season': season,
                       'generation': pop.generation, 'geometry': pop.geom_sha, 'seq': self.sequence + 1, 'nonce': nonce})
        if digest != plan_digest:
            raise PlantSessionStateError('the saved plan does not reproduce its digest (stale parent, corrupted or foreign plan); refused')
        dt = float(rec.dt_days_exact)
        month, yd = day.start.month, _bio.year_days(day.start)
        a_inc, t_inc = _growth.integral_increments(cfg.table, month=month, temperature_c=day.temperature_c, dt_days=dt, year_days=yd,
                                                   max_bytes=int(cfg.max_bytes))
        D = _frozen(cfg, _dev(cfg, np.ascontiguousarray(D_host)))
        return DailyPlan(self.token, self.sequence + 1, day, step, season, pop, D, dict(packet), pet_kind, month, yd, digest,
                         (_dev(cfg, a_inc), _dev(cfg, t_inc)), nonce)

    # ---------------------------------------------------------------- 2. propose
    def _check_resource_read(self, plan, new):
        cur, bind = self.resource_cursor, self.day_binding
        if bind is None or cur is None:
            raise PlantSessionStateError('water is not connected: the session needs an explicit DayBinding (nominal origin + SYRUP daily run) AND '
                                         'an initial resource cursor (start(..., day_binding=, initial_resource=) or stage_connect_water) '
                                         'before any proposal; a transport-only session cannot allocate')
        if new.role != 'interevent' or new.daily_pending or new.daily_run is None or new.daily_ordinal is None:
            raise PlantSessionStateError('availability must come from a COMPLETED daily lineage state (interevent, no pending storm, a consumed '
                                         'occurrence); legacy / lineage-free states are refused')
        if new.daily_run != bind.run_digest:
            raise PlantSessionStateError('the availability belongs to another SYRUP daily run; refused')
        want = bind.ordinal_of(plan.day.nominal_date)
        if new.daily_ordinal != want:
            raise PlantSessionStateError(f'the availability is of daily occurrence {new.daily_ordinal!r} but the nominal date '
                                         f'{plan.day.nominal_date.isoformat()} maps to occurrence {want}; refused (wrong nominal day)')
        if new.lop_sha256 != cur.lop_sha256:
            raise PlantSessionStateError('the availability belongs to another SYRUP operator / lineage; refused')
        if new.request_sequence != cur.request_sequence or new.n_requests != cur.n_requests:
            raise PlantSessionStateError(
                f'unhandled T history: the state has request sequence {new.request_sequence} / {new.n_requests} requests but LEAF has consumed '
                f'{cur.request_sequence} / {cur.n_requests}; a T debit without LEAF feedback is never silently adopted (hydrology / close / storm '
                'transitions with the SAME T counts are fine)')
        if new.revision < cur.revision or new.hydro_count < cur.hydro_count:
            raise PlantSessionStateError('stale availability: the resource lineage moved backwards relative to the last accepted debit')
        same_day = new.daily_ordinal is not None and new.daily_ordinal == cur.daily_ordinal
        if same_day:
            if new.hydro_count != cur.hydro_count:
                raise PlantSessionStateError('a second fragment of the same daily occurrence must not follow another hydrology step')
            if new.parent_digest != cur.parent_digest:
                raise PlantSessionStateError('the availability is not the state that followed the last accepted debit; refused')
        else:
            if new.daily_ordinal is not None and cur.daily_ordinal is not None and new.daily_ordinal < cur.daily_ordinal:
                raise PlantSessionStateError('the availability is of an EARLIER daily occurrence than the last accepted debit; refused')
            if new.hydro_count <= cur.hydro_count:
                raise PlantSessionStateError('stale availability: a fresh read after a new daily hydrology step is required')

    def propose_withdrawal(self, plan, read):
        '''Root-weighted proposal of `plan.D` against the FRESH LIVE `AvailabilityRead` (m3). Pure; refuses a consumed / foreign plan, a read of
        another nominal day or run, another operator, a stale or non-advancing lineage and any admission excess before allocation.'''
        if not isinstance(plan, DailyPlan) or not isinstance(read, _lb.AvailabilityRead):
            raise PlantSessionError('plan must be a DailyPlan and read an AvailabilityRead')
        self._check_token(plan.parent_token, 'the daily plan')
        if plan._used:
            raise PlantSessionStateError('this plan was already published (duplicate); refused')
        if not plan.day.transpiration_enabled:
            raise PlantSessionError('transpiration is disabled for this fragment: no withdrawal is proposed (use stage_dry_day)')
        self._check_resource_read(plan, read.identity)
        rm = plan.pop.geometry.root_map
        if (int(read.n_layers), int(read.n_cells)) != (int(rm.n_layers), int(rm.n_cells)) or read.backend != self.config.backend:
            raise PlantSessionError('the availability shape / backend differs from the root map')
        est = _nbytes(_uptake.estimate_plan_bytes(rm.n_plants, rm.n_edges, rm.n_resources, rm.backend))
        self._admit('propose_withdrawal', est, 8 * int(read.n_layers) * int(read.n_cells) + 8 * self.n_live, candidate=plan.season_applied)
        up = _uptake.propose(rm, plan.D, read.supply_m3)
        ident = _lb.debit_identity(f'leaf:{self.session_id[:16]}:{plan.sequence}:{plan.nonce[:16]}')
        return WithdrawalProposal(self.token, plan, read, up, ident, 4 * 8 * int(read.n_layers) * int(read.n_cells))

    # ---------------------------------------------------------------- 3. + 4. stage
    def stage_day(self, proposal, accepted):
        '''Consume the ACTUAL accepted withdrawal (`lifecycle_bridge.accepted_withdrawal`): per-plant delivered / unmet / satisfaction through
        `uptake.accept`, credit accumulation (growth enabled only), scheduled growth + geometry rebuild at the grid / new year and the calendar
        step. PURE: returns a `Staged` candidate; nothing accepted is touched and no water is read or written.'''
        if not isinstance(proposal, WithdrawalProposal) or not isinstance(accepted, _lb.AcceptedWithdrawal):
            raise PlantSessionError('proposal must be a WithdrawalProposal and accepted an AcceptedWithdrawal')
        self._check_token(proposal.parent_token, 'the withdrawal proposal')
        plan = proposal.plan
        if plan._used:
            raise PlantSessionStateError('this plan was already published (duplicate); refused')
        if accepted.identity != proposal.identity:
            raise PlantSessionStateError('the accepted withdrawal carries another debit identity (wrong / duplicate result); refused')
        if accepted.before != proposal.read.identity:
            raise PlantSessionStateError('the accepted withdrawal was not made from the availability read of this proposal; refused')
        cur = self.resource_cursor
        if cur is not None and (accepted.after.request_sequence <= cur.request_sequence or accepted.after.n_requests <= cur.n_requests):
            raise PlantSessionStateError('duplicate / stale debit: the resource lineage did not advance past the last accepted resource; refused')
        rm = plan.pop.geometry.root_map
        est = _nbytes(_uptake.estimate_feedback_bytes(rm.n_plants, rm.n_edges, rm.n_resources, rm.backend))
        self._admit('stage_day', est, 16 * int(accepted.actual_m3.size) + 8 * int(proposal.read.supply_m3.size)
                    + 4 * PER_LIVE_BYTES * self.n_live, candidate=True)
        try:
            fb = _uptake.accept(proposal.uptake_plan, accepted.actual_m3)
        except _uptake.UptakeError as exc:
            raise PlantSessionStateError(f'the accepted withdrawal is inconsistent with the proposal: {exc}') from exc
        return self._finish_day(plan, fb, accepted)

    def stage_dry_day(self, plan):
        '''A fragment without transpiration (and therefore without growth credit): calendar / age only; no water, no request.'''
        if not isinstance(plan, DailyPlan):
            raise PlantSessionError('plan must be a DailyPlan')
        self._check_token(plan.parent_token, 'the daily plan')
        if plan._used:
            raise PlantSessionStateError('this plan was already published (duplicate); refused')
        if plan.day.transpiration_enabled:
            raise PlantSessionError('transpiration is enabled for this fragment: propose and stage the withdrawal')
        return self._finish_day(plan, None, None)

    def _finish_day(self, plan, fb, accepted):
        cfg, step, day = self.config, plan.step, plan.day
        pop, P = plan.pop, self.n_live
        if accepted is not None and day.growth_enabled and P:
            rtol = self._rtol(pop.geometry.root_map)
            try:
                D, U, A, T = _bio.credit_increment(pop.D, pop.U, pop.A, pop.T, plan.D, fb.delivered, plan.increments[0], plan.increments[1],
                                                   growth_enabled=True, max_bytes=int(cfg.max_bytes),
                                                   reserve_bytes=self._live_other(), overdelivery_rtol=rtol, pending_days=step.period_days)
            except _bio.BioCalendarError as exc:
                raise PlantSessionError(f'credit accumulation refused: {exc}') from exc
            pop = dataclasses.replace(pop, D=_frozen(cfg, D), U=_frozen(cfg, U), A=_frozen(cfg, A), T=_frozen(cfg, T))
        grew = False
        if step.fire_growth and P:
            pop, info = self._grow_pop(pop, year_end=step.year_end, instant=day.end)
            grew = True
        c = dict(self.counters_)
        c['days_open' if accepted is not None else 'days_dry'] += 1
        c['withdrawals_accepted'] += 1 if accepted is not None else 0
        c['growth_updates'] += 1 if grew else 0
        c['season_turnovers'] += 1 if plan.season_applied else 0
        c['year_ends'] += 1 if step.year_end else 0
        c['geometry_rebuilds'] += (1 if plan.season_applied else 0) + (1 if grew else 0)
        bio = dict(self.bio)
        bio['transport_pending'] = None
        if step.year_end and self.laws.demo_on and P:
            bio['annual_due'] = int(day.start.year)
        feedback = None
        if fb is not None:
            tot = {'requested_m3': float(plan.packet.get('sum_requested_m3', 0.0)), 'delivered_m3': float(fb.summary['delivered_total_m3']),
                   'unmet_m3': float(fb.summary['unmet_total_m3']), 'accepted_m3': float(fb.summary['accepted_total_m3']),
                   'n_plants_unsatisfied': int(fb.summary['n_plants_unsatisfied']), 'n_plants_zero_demand': int(fb.summary['n_plants_zero_demand'])}
            feedback = DailyFeedback(self.live_ids(), plan.D, fb.delivered, fb.unmet, fb.satisfaction, tot, dict(fb.closure))
        return self._stage('day' if accepted is not None else 'dry_day', pop, step.schedule, c, bio, plan.plan_digest, day.start, day.end,
                           accepted, feedback, {'growth_fired': grew, 'season_turnover': plan.season_applied, 'year_end': bool(step.year_end),
                                                'credit_days': str(step.credit_days), 'duration_days': str(day.record().dt_days_exact),
                                                'annual_decision_due_year': bio['annual_due']}, plan=plan)

    def _stage(self, kind, pop, schedule, counters, bio, digest, start, end, accepted, feedback, report, *, plan=None, base=None,
               offset=None, pop_sha=None):
        sid = StageId(self.session_id, self.token, self.sequence + 1, kind, start.isoformat(), end.isoformat(), digest,
                      None if accepted is None else accepted.identity, None if accepted is None else accepted.before.to_dict(),
                      None if accepted is None else accepted.after.to_dict())
        cursor = self.resource_cursor if accepted is None else accepted.after
        nxt = self._make(self.config, self.laws, self.session_id, self.lineage, self.sequence + 1, self.base if base is None else base,
                         self.base_offset_days if offset is None else offset, pop, schedule, counters, bio, cursor, sid, self.day_binding,
                         self.pop_sha if pop_sha is None else pop_sha)
        return Staged(sid, self.token, self.session_id, nxt, feedback, dict(report, kind=kind, sequence=self.sequence + 1), plan)

    def _plain_digest(self, kind, start, end):
        return _sha({'parent': self.token, 'kind': kind, 'start': start.isoformat(), 'end': end.isoformat(), 'seq': self.sequence + 1})

    # ---------------------------------------------------------------- boundaries
    def stage_flush(self):
        '''Finish the pending ENABLED growth period AT THE CURSOR (year_end=False, geometry rebuilt, pending cleared). Refused when nothing is pending.'''
        self._check_token(self.token, 'flush')
        if not _bio.needs_flush(self.schedule):
            raise PlantSessionBoundaryError('no growth is pending: nothing to flush')
        sched = _bio.flush_step(self.schedule, growth_enabled=True)
        pop, info = self._grow_pop(self.pop, year_end=False, instant=self.schedule.cursor)
        c = dict(self.counters_)
        c['flushes'] += 1
        c['growth_updates'] += 1
        c['geometry_rebuilds'] += 1
        cur = self.schedule.cursor
        return self._stage('flush', pop, sched, c, dict(self.bio), self._plain_digest('flush', cur, cur), cur, cur, None, None,
                           {'growth': {'ledger': info['ledger']}})

    def stage_transport(self, tday):
        '''Advance the calendar over one transport fragment (any duration): no E / T / growth / catch-up, ages advance by the calendar. Pending
        growth must be flushed first. A year end inside it defers the season turnover and, with an active mortality law, sets the annual
        decision flag (at most ONE new year per fragment, an outstanding annual decision refuses, as `IntereventSession`).'''
        if not isinstance(tday, TransportDay):
            raise PlantSessionError('tday must be a TransportDay')
        self._check_token(self.token, 'transport')
        bio = dict(self.bio)
        if self.laws.demo_on and self.n_live:
            if bio['annual_due'] is not None:
                raise PlantSessionBoundaryError(f'the annual decision for year {bio["annual_due"]} is outstanding: stage_annual_decision first')
            ny0 = _next_new_year(tday.start)
            if ny0 is not None and ny0 <= tday.end:
                ny1 = _next_new_year(ny0)
                if ny1 is not None and ny1 <= tday.end:
                    raise PlantSessionBoundaryError('with live plants and an active mortality law a transport may cross at most ONE new year')
        try:
            ts = _bio.transport_step(self.schedule, tday.start, tday.end)
        except _bio.BioCalendarError as exc:
            raise PlantSessionBoundaryError(str(exc)) from exc
        bio['transport_pending'] = None
        if tday.kind in TRANSPORT_SOURCE:
            bio['transport_pending'] = {'sequence': self.sequence + 1, 'kind': tday.kind, 'source': TRANSPORT_SOURCE[tday.kind],
                                        'start': tday.start.isoformat(), 'end': tday.end.isoformat(),
                                        'start_days': self._time_days_at(tday.start), 'end_days': self._time_days_at(tday.end)}
        if ts.season_deferred and self.laws.demo_on and self.n_live:
            bio['annual_due'] = int(tday.end.year - 1)
        c = dict(self.counters_)
        c['days_transport'] += 1
        c['year_ends'] += 1 if ts.season_deferred else 0
        return self._stage('transport', self.pop, ts.schedule, c, bio, self._plain_digest('transport', tday.start, tday.end), tday.start,
                           tday.end, None, None, {'transport_kind': tday.kind, 'skipped_growth_boundaries': ts.skipped_boundaries,
                                                  'season_deferred': ts.season_deferred})

    def stage_deferred_season(self):
        '''Finalise the PRIOR season deferred by a year end inside a transport (`growth.grow(year_end=True)`, zero water, zero integrals, no
        duration), rebuild geometry and clear the flag (call it BEFORE the annual decision / reproduction, as `flush_deferred_season`).'''
        self._check_token(self.token, 'deferred season')
        if not self.schedule.season_due:
            raise PlantSessionBoundaryError('no deferred season is due')
        if self.schedule.pending_days != 0:
            raise PlantSessionBoundaryError('pending growth must be flushed before the deferred season')
        sched = dataclasses.replace(self.schedule, season_due=False)
        pop, c = self.pop, dict(self.counters_)
        if self.n_live:
            pop, _ = self._grow_pop(pop, year_end=True, instant=self.schedule.cursor)
            c['season_turnovers'] += 1
            c['geometry_rebuilds'] += 1
        cur = self.schedule.cursor
        return self._stage('season', pop, sched, c, dict(self.bio), self._plain_digest('season', cur, cur), cur, cur, None, None,
                           {'deferred_season_applied': bool(self.n_live)})

    # ---------------------------------------------------------------- Phase 2 / 3 population rebind (transactional, pure)
    def _rebind(self, pop, B_h, E_h, code, *, stress, ann, ann_reset, births=None):
        '''New canonical base at the cursor (ages frozen for non-live), live rows, geometry through the shared builders, new arrays. Mirrors
        `IntereventSession._rebind`: survivors keep row order, births are appended with fresh zero accumulators, root overrides of survivors
        are kept. Requires no pending growth. Returns (base, offset, pop, pop_sha).'''
        if self.schedule.pending_days != 0:
            raise PlantSessionBoundaryError('pending growth must be flushed before a population change')
        cfg, P = self.config, self.n_live
        keep = np.asarray(code) == _state.LIVE
        n_new = 0 if births is None else int(births['ids'].shape[0])
        Pa_new, Pl_new = int(self.base.n) + n_new, int(np.count_nonzero(keep)) + n_new
        self._admit('rebind', (MATERIALIZE_BYTES_PER_PLANT + GEOMETRY_BYTES_PER_PLANT + ROW_BYTES_PER_PLANT) * Pa_new
                    + 80 * Pl_new + 17 * P, (16 + (_demo.PERSISTENT_BYTES_PER_LIVE if self.laws.stress_on else 0)) * Pl_new, candidate=True)
        a = self.base._arrays(copy=True)
        rows = self.pop.live_rows
        ed = float(self.elapsed_days())
        if P:
            a['agb_dry_kg'][rows] = B_h
            a['seasonal_agb_dry_kg'][rows] = E_h
            a['age_days'][rows] = a['age_days'][rows] + ed
            a['status'][rows] = code
        if n_new:
            nb = n_new
            add = {'ids': births['ids'], 'species_index': births['species_index'], 'x_m': births['x_m'], 'y_m': births['y_m'],
                   'establishment_ground_m': births['establishment_ground_m'], 'establishment_unresolved': np.zeros(nb, dtype=np.bool_),
                   'agb_dry_kg': births['agb_dry_kg'], 'seasonal_agb_dry_kg': births['seasonal_agb_dry_kg'],
                   'age_days': np.zeros(nb, dtype=np.float64), 'status': np.full(nb, _state.LIVE, dtype=np.int8)}
            for k in a:
                a[k] = np.concatenate([a[k], np.asarray(add[k], dtype=a[k].dtype)])
        prov = dict(self.base.provenance)
        prov['plant_session'] = {'version': VERSION, 'cursor': self.schedule.cursor.isoformat(), 'sequence': self.sequence + 1}
        new_base = _state.PlantState(self.base.domain, self.base.species_names, self.base.catalog_hash, ids=a['ids'],
                                     species_index=a['species_index'], x_m=a['x_m'], y_m=a['y_m'],
                                     establishment_ground_m=a['establishment_ground_m'],
                                     establishment_unresolved=a['establishment_unresolved'], agb_dry_kg=a['agb_dry_kg'],
                                     seasonal_agb_dry_kg=a['seasonal_agb_dry_kg'], age_days=a['age_days'], status=a['status'],
                                     time_days=float(self.base.time_days) + ed, provenance=prov)
        new_rows = np.flatnonzero(new_base.status == _state.LIVE).astype(np.int64)
        expect = rows[keep]
        if n_new:
            expect = np.concatenate([expect, np.arange(int(self.base.n), Pa_new, dtype=np.int64)])
        if not np.array_equal(new_rows, expect):
            raise PlantSessionError('internal consistency: the rebind live-row mapping differs from the status codes')
        Bh2 = np.ascontiguousarray(np.asarray(B_h)[keep], dtype=np.float64)
        Eh2 = np.ascontiguousarray(np.asarray(E_h)[keep], dtype=np.float64)
        if n_new:
            Bh2 = np.ascontiguousarray(np.concatenate([Bh2, births['agb_dry_kg']]), dtype=np.float64)
            Eh2 = np.ascontiguousarray(np.concatenate([Eh2, births['seasonal_agb_dry_kg']]), dtype=np.float64)
        st = ann_n = None
        if self.laws.stress_on and Pl_new:
            kb = _dev(cfg, keep)

            def grow_rows(arr):
                out = arr[kb]
                return out if not n_new else _xp(cfg).concatenate([out, _xp(cfg).zeros(n_new, dtype=out.dtype)])
            st = grow_rows(stress)
            ann_n = None if ann_reset else tuple(grow_rows(x) for x in ann)
        if Pl_new == 0:
            zero = self._zero_acc(0)
            st, ann_n = (zero[0], tuple(zero[1:])) if self.laws.stress_on else (None, None)
        reserve = self._live_other(80 * Pl_new + ROW_BYTES_PER_PLANT * Pa_new)
        new_pop = self._build_pop(cfg, self.laws, new_base, new_rows, Bh2, Eh2, reserve=reserve, generation=self.pop.generation + 1,
                                  stress=st, ann=ann_n)
        offset = _bio.elapsed_days(self.schedule.anchor, self.schedule.cursor)
        return new_base, Fraction(offset), new_pop, _pop_sha(new_base, cfg.catalog.hash)

    def _rebind_stage(self, kind, label, pop, B_h, E_h, code, *, stress, ann, ann_reset, bio, report, births=None, changed=True):
        cur = self.schedule.cursor
        c = dict(self.counters_)
        digest = self._plain_digest(kind, cur, cur)
        if changed:
            base, offset, npop, psha = self._rebind(pop, B_h, E_h, code, stress=stress, ann=ann, ann_reset=ann_reset, births=births)
            bio['n_population_rebinds'] = int(bio['n_population_rebinds']) + 1
            c['geometry_rebuilds'] += 1
            return self._stage(kind, npop, self.schedule, c, bio, digest, cur, cur, None, None, dict(report, rebind=True), base=base,
                               offset=offset, pop_sha=psha)
        npop = pop
        if stress is not None and ann_reset:
            z = self._zero_acc(self.n_live)
            npop = dataclasses.replace(pop, stress=stress, ann_req=z[1], ann_act=z[2], ann_plan=z[3], ann_gain=z[4])
        return self._stage(kind, npop, self.schedule, c, bio, digest, cur, cur, None, None, dict(report, rebind=False))

    def _gate_boundary(self, what):
        self._check_token(self.token, what)
        if self.schedule.pending_days != 0:
            raise PlantSessionBoundaryError(f'pending growth must be flushed (stage_flush) before {what}')

    # ---------------------------------------------------------------- Phase 2: surface-event response
    def _lookup_rows(self, ids):
        q = _state._coerce_int(ids, 'plant_id', None, np.int64)
        if q.size == 0:
            return np.zeros(0, dtype=np.int64)
        if np.unique(q).shape[0] != q.shape[0]:
            raise _state.StateError('duplicate plant IDs in request')
        base_ids = self.base.ids
        if base_ids.shape[0] == 0:
            raise _state.StateError('unknown plant IDs (population is empty)')
        order = np.argsort(base_ids, kind='stable')
        pos = np.minimum(np.searchsorted(base_ids[order], q), base_ids.shape[0] - 1)
        found = base_ids[order][pos] == q
        if not found.all():
            raise _state.StateError(f'unknown plant IDs {q[~found][:10].tolist()}')
        return order[pos]

    def stage_surface_event(self, event, current_surface):
        '''Species mechanical response (`leaf.disturbance.evaluate_event`) of the LIVE plants named by `event`, applied right AFTER the accepted
        transport fragment it describes (bound by source, start and end in model days; one decision per fragment; replay / stale / wrong-source
        events refused). `current_surface` (ny, nx) is the ACTUAL surface after the event. No random state; water and time unchanged.'''
        if not isinstance(event, _events.SurfaceEventDiagnostics):
            raise PlantSessionError('event must be a leaf.events.SurfaceEventDiagnostics')
        self._gate_boundary('a disturbance decision')
        tp = self.bio['transport_pending']
        if tp is None:
            raise PlantSessionStateError('no accepted transport interval awaits a disturbance decision (stale, replayed or out-of-order event)')
        if event.source != tp['source']:
            raise PlantSessionError(f'event source {event.source!r} does not match the last transport interval ({tp["kind"]!r} -> {tp["source"]!r})')
        tol = max(EVENT_TIME_TOL_DAYS, 64.0 * FP64_EPS * max(abs(float(tp['end_days'])), 1.0))
        if abs(event.time_end_days - float(tp['end_days'])) > tol or abs(event.time_start_days - float(tp['start_days'])) > tol:
            raise PlantSessionStateError('the event does not match the last transport interval (start / end in model days); refused')
        base, n, P = self.base, event.n, self.n_live
        self._admit('event_evaluate', _nbytes(_dist.estimate_event_bytes(n, int(self.config.table.n_species), self.config.backend))
                    + 24 * int(base.n) + 8 * int(base.domain.ncells) + 96 * n, candidate=True)
        try:
            surface = base.domain.check_field(current_surface, 'current surface')
            rows = self._lookup_rows(event.plant_id)
        except _state.StateError as exc:
            raise PlantSessionError(f'event / surface refused: {exc}') from exc
        status = np.array(base.status[rows], dtype=np.int8)
        B_ev = np.array(base.agb_dry_kg[rows], dtype=np.float64)
        E_ev = np.array(base.seasonal_agb_dry_kg[rows], dtype=np.float64)
        pos = np.zeros(n, dtype=np.int64)
        is_live = np.zeros(n, dtype=bool)
        Bh, Eh = _hostarr(self.pop.B), _hostarr(self.pop.E)
        if P and n:
            p0 = np.searchsorted(self.pop.live_rows, rows)
            pos = np.minimum(p0, P - 1)
            is_live = (p0 < P) & (self.pop.live_rows[pos] == rows)
            if is_live.any():
                B_ev[is_live], E_ev[is_live] = Bh[pos[is_live]], Eh[pos[is_live]]
        if not np.array_equal(status == _state.LIVE, is_live):
            raise PlantSessionError('internal consistency: base status and the live-row mapping disagree')
        try:
            res = _dist.evaluate_event(self.laws.dist, self.config.catalog, event, species_index=base.species_index[rows], status=status,
                                       B=B_ev, E=E_ev, x_m=base.x_m[rows], y_m=base.y_m[rows],
                                       establishment_ground_m=base.establishment_ground_m[rows], domain=base.domain,
                                       current_surface=surface, max_bytes=int(self.config.max_bytes),
                                       extra_reserve_bytes=self._live_other(96 * n + 8 * int(base.domain.ncells)))
        except _dist.DisturbanceBudgetError as exc:
            raise PlantSessionAdmissionError(f'event_evaluate: {exc}') from exc
        except (_dist.DisturbanceError, _state.StateError) as exc:
            raise PlantSessionError(f'surface event refused: {exc}') from exc
        changed = bool(np.any(res['B_new'] != B_ev) or np.any(res['E_new'] != E_ev) or np.any(res['status_new'] != status))
        bio = dict(self.bio)
        bio['transport_pending'] = None
        bio['n_disturbance'] = int(bio['n_disturbance']) + 1
        bio['last_decision'] = {'kind': 'disturbance', 'event_id': event.event_id, 'source': event.source,
                                'start_days': event.time_start_days, 'end_days': event.time_end_days, 'sequence': self.sequence + 1,
                                'n_plants': int(n)}
        B_all, E_all, code = Bh.copy(), Eh.copy(), np.zeros(P, dtype=np.int8)
        if changed:
            idx = pos[is_live]
            B_all[idx], E_all[idx] = res['B_new'][is_live], res['E_new'][is_live]
            code[idx] = res['status_new'][is_live]
        ann = None if self.pop.ann_req is None else (self.pop.ann_req, self.pop.ann_act, self.pop.ann_plan, self.pop.ann_gain)
        return self._rebind_stage('event', 'event', self.pop, B_all, E_all, code, stress=self.pop.stress, ann=ann, ann_reset=False, bio=bio,
                                  report={'event_id': event.event_id, 'diagnostics': res['diagnostics'], 'rng_consumed': False},
                                  changed=changed)

    # ---------------------------------------------------------------- Phase 2: annual decision
    def stage_annual_decision(self, year):
        '''The annual age / background / stress decision (`leaf.demography.decide`) for the completed `year` on the existing annual accumulators.
        Refused unless that decision is outstanding and growth is flushed. Death -> dead_standing (B / E kept), dieback scales B and E, the
        accumulators reset, the keyed decision counter advances; no water, no time.'''
        if isinstance(year, (bool, np.bool_)) or not isinstance(year, (int, np.integer)):
            raise PlantSessionError(f'year must be an integer, got {year!r}')
        year = int(year)
        self._gate_boundary('the annual decision')
        bio = dict(self.bio)
        if not self.laws.demo_on:
            raise PlantSessionError('no species has an active mortality law: there is no annual decision')
        if bio['annual_due'] is None:
            raise PlantSessionStateError('no annual decision is due (already accepted or no new year crossed)')
        if year != bio['annual_due']:
            raise PlantSessionStateError(f'the outstanding annual decision is for year {bio["annual_due"]}, not {year}')
        P = self.n_live
        if not P:
            raise PlantSessionStateError('no live plants: nothing to decide')
        cur = self.schedule.cursor
        period_days = float(_forcing.exact_seconds(cur - _dt.datetime.fromisoformat(bio['annual_cursor'])) / SECONDS_PER_DAY)
        if not period_days > 0.0:
            raise PlantSessionStateError('the decision period has zero length')
        est = _demo.memory_estimate(P, int(self.config.table.n_species), self.config.backend)
        self._admit('annual_decide', _nbytes(est) + 36 * P + 8 * P + 60 * P, candidate=True)
        pop, base = self.pop, self.base
        age_end = np.ascontiguousarray(base.age_days[pop.live_rows] + float(self.elapsed_days()), dtype=np.float64)
        if self.laws.stress_on:
            st, rq, ac, pl, gn = pop.stress, pop.ann_req, pop.ann_act, pop.ann_plan, pop.ann_gain
        else:
            st, rq, ac, pl, gn = self._zero_acc(P)
        counter = int(bio['n_annual'])
        try:
            dec = _demo.decide(self.laws.demo, species_index=pop.sidx, plant_ids=pop.live_ids, age_end_days=age_end, stress=st,
                               required_support=rq, actual_support=ac, planned_growth=pl, actual_growth=gn, period_days=period_days,
                               seed=self.laws.seed, counter=counter, max_bytes=int(self.config.max_bytes),
                               extra_reserve_bytes=self._live_other(36 * P + 8 * P + 60 * P))
        except _demo.DemographyBudgetError as exc:
            raise PlantSessionAdmissionError(f'annual_decide: {exc}') from exc
        except _demo.DemographyError as exc:
            raise PlantSessionError(f'annual decision refused: {exc}') from exc
        cause, loss = _hostarr(dec['cause']), _hostarr(dec['loss_fraction'])
        stress_new, summ = dec['stress_new'], dec['summary']
        B_h, E_h = _hostarr(pop.B), _hostarr(pop.E)
        dead = cause != _demo.CAUSE_NONE
        dieback = (loss > 0.0) & (~dead)
        B_new = np.where(dieback, B_h * (1.0 - loss), B_h)
        E_new = np.where(dieback, np.minimum(E_h * (1.0 - loss), B_new), E_h)
        if bool(np.any(B_new <= 0.0)) or bool(np.any(E_new > B_new)):
            raise PlantSessionError('annual dieback would leave a standing plant with non-positive dry AGB (underflow)')
        removed = B_h - B_new
        if bool(np.any(np.abs(B_h - (B_new + removed)) > _dist.LEDGER_EPS_FACTOR * FP64_EPS * B_h + 1e-300)):
            raise PlantSessionError('annual dry AGB ledger does not close within 16 eps; refused')
        code = np.where(dead, _state.DEAD_STANDING, _state.LIVE).astype(np.int8)
        bio['n_annual'] = counter + 1
        bio['annual_due'] = None
        bio['annual_cursor'] = cur.isoformat()
        bio['last_decision'] = {'kind': 'annual_decision', 'year': year, 'decision_counter': counter, 'cursor': cur.isoformat(),
                                'sequence': self.sequence + 1}
        rep = {'year': year, 'decision_counter': counter, 'decision_counter_after': counter + 1, 'rng_consumed': True,
               'n_died_age_background': int(np.count_nonzero(cause == _demo.CAUSE_AGE)),
               'n_died_stress': int(np.count_nonzero(cause == _demo.CAUSE_STRESS)), 'n_dieback': int(np.count_nonzero(dieback)),
               'n_live_before': int(P), 'removed_dry_agb_kg': math.fsum(removed.tolist()),
               'flag_counts': np.asarray(summ['flag_counts']).tolist(), 'caveats': list(_demo.CAVEATS)}
        changed = bool(np.any(dead) or np.any(dieback))
        return self._rebind_stage('annual', 'annual', pop, B_new, E_new, code, stress=stress_new if self.laws.stress_on else None, ann=None,
                                  ann_reset=True, bio=bio, report=rep, changed=changed)

    # ---------------------------------------------------------------- Phase 3: reproduction
    def _repro_year_gate(self, year, *, decline):
        if isinstance(year, (bool, np.bool_)) or not isinstance(year, (int, np.integer)):
            raise PlantSessionError(f'year must be an integer, got {year!r}')
        year = int(year)
        if not self.laws.repro_on:
            raise PlantSessionError('no species has an active reproduction law: there is no reproduction opportunity')
        last, cy = int(self.bio['repro_last_year']), int(self.schedule.cursor.year)
        if year <= last:
            raise PlantSessionStateError(f'replay / stale opportunity refused: year {year} was already handled (last {last})')
        if year != last + 1:
            raise PlantSessionStateError(f'the next opportunity is year {last + 1}, not {year}: opportunities are never skipped silently')
        if year > cy - 1:
            raise PlantSessionStateError(f'the opportunity for year {year} is not due: the cursor has not crossed 1 January {year + 1}')
        if not decline and year < cy - 1:
            raise PlantSessionStateError(f'the opportunity for year {year} can no longer be taken (cursor in {cy}): decline it explicitly')
        return year

    def stage_reproduction(self, year, current_surface, *, max_new_plants, max_total_records, max_attempts):
        '''The annual reproduction boundary (`leaf.recruitment.evaluate`), after pending growth, the post-event decision, the deferred season and
        the due annual decision. Keyed by (rng_seed, accepted-opportunity counter); newborns request water only from the NEXT open fragment;
        existing plants keep id / location / age / status / overrides; donors lose exactly the transferred B / E.'''
        try:
            mx_new, mx_tot, mx_att = _rec.check_limits(max_new_plants, max_total_records, max_attempts)
        except _rec.ReproductionError as exc:
            raise PlantSessionError(str(exc)) from exc
        self._gate_boundary('the reproduction opportunity')
        year = self._repro_year_gate(year, decline=False)
        bio = dict(self.bio)
        if bio['transport_pending'] is not None:
            raise PlantSessionBoundaryError('a transport interval still awaits its disturbance decision: stage_surface_event first')
        if bio['annual_due'] is not None:
            raise PlantSessionBoundaryError(f'the annual decision for year {bio["annual_due"]} is outstanding: stage_annual_decision first')
        if self.schedule.season_due:
            raise PlantSessionBoundaryError('a deferred seasonal turnover is pending: stage_deferred_season first')
        P, Pa, base = self.n_live, int(self.base.n), self.base
        try:
            est1, _ = _rec.preflight_bytes(self.laws.repro, self.laws.patch, base.domain, P)
        except _rec.ReproductionError as exc:
            raise PlantSessionError(str(exc)) from exc
        gather = (80 + (16 if self.config.backend == 'cuda' else 0)) * P
        self._admit('reproduction_preflight', _nbytes(est1) + gather, candidate=True)
        rows = self.pop.live_rows
        if P:
            B_h, E_h = _hostarr(self.pop.B), _hostarr(self.pop.E)
            l_ids = np.ascontiguousarray(base.ids[rows], dtype=np.int64)
            l_sp = np.ascontiguousarray(base.species_index[rows], dtype=np.int32)
            l_age = np.ascontiguousarray(base.age_days[rows] + float(self.elapsed_days()), dtype=np.float64)
            l_x, l_y = np.ascontiguousarray(base.x_m[rows], dtype=np.float64), np.ascontiguousarray(base.y_m[rows], dtype=np.float64)
        else:
            B_h = E_h = l_age = l_x = l_y = np.zeros(0, dtype=np.float64)
            l_ids, l_sp = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32)
        counter = int(bio['n_reproduction'])
        try:
            res = _rec.evaluate(self.laws.repro, self.laws.patch, base.domain, current_surface, live_ids=l_ids, live_species=l_sp,
                                live_age_days=l_age, live_B=B_h, live_E=E_h, live_x=l_x, live_y=l_y,
                                max_existing_id=int(base.ids.max()) if Pa else None, n_total_records=Pa, seed=self.laws.seed, counter=counter,
                                max_new_plants=mx_new, max_total_records=mx_tot, max_attempts=mx_att, max_bytes=int(self.config.max_bytes),
                                extra_reserve_bytes=self._live_other(gather))
        except _rec.ReproductionBudgetError as exc:
            raise PlantSessionAdmissionError(f'reproduction_evaluate: {exc}') from exc
        except _rec.ReproductionError as exc:
            raise PlantSessionError(f'reproduction refused: {exc}') from exc
        diag, births = res['diagnostics'], res['births']
        n_new = int(diag['n_births'])
        cur = self.schedule.cursor
        bio['n_reproduction'] = counter + 1
        bio['repro_last_year'] = year
        bio['last_reproduction'] = {'year': year, 'counter': counter, 'cursor': cur.isoformat(), 'sequence': self.sequence + 1,
                                    'n_seed_recruits': int(diag['n_seed_recruits']), 'n_daughters_placed': int(diag['n_daughters_placed']),
                                    'n_daughters_rejected': int(diag['n_daughters_rejected']), 'n_births': n_new}
        if n_new and P == 0 and self.laws.demo_on:
            bio['annual_cursor'] = cur.isoformat()
        B_all, E_all = B_h.copy(), E_h.copy()
        if n_new:
            pr = res['parent_rows']
            if pr.shape[0]:
                B_all[pr], E_all[pr] = res['parent_B_new'], res['parent_E_new']
        ann = None if self.pop.ann_req is None else (self.pop.ann_req, self.pop.ann_act, self.pop.ann_plan, self.pop.ann_gain)
        return self._rebind_stage('reproduction', 'reproduction', self.pop, B_all, E_all, np.zeros(P, dtype=np.int8), stress=self.pop.stress,
                                  ann=ann, ann_reset=False, bio=bio, births=births if n_new else None, changed=bool(n_new),
                                  report={'year': year, 'reproduction_counter': counter, 'reproduction_counter_after': counter + 1,
                                          'rng_consumed': True, 'reproduction': diag, 'new_plants_request_water_from': 'the next open fragment'})

    def stage_decline_reproduction(self, year):
        '''Explicitly record that the opportunity `year` is NOT taken (no random state, no plant change; the only way to pass a missed year).'''
        self._gate_boundary('declining a reproduction opportunity')
        year = self._repro_year_gate(year, decline=True)
        bio = dict(self.bio)
        bio['repro_last_year'] = year
        cur = self.schedule.cursor
        bio['last_reproduction'] = {'year': year, 'counter': None, 'cursor': cur.isoformat(), 'sequence': self.sequence + 1, 'declined': True,
                                    'n_births': 0}
        return self._stage('reproduction_declined', self.pop, self.schedule, dict(self.counters_), bio,
                           self._plain_digest('reproduction_declined', cur, cur), cur, cur, None, None,
                           {'year': year, 'rng_consumed': False})

    def stage_connect_water(self, day_binding, initial_resource):
        '''Explicitly connect a transport-only session to SYRUP (staged like every change): `day_binding` = nominal origin + genuine daily run,
        `initial_resource` = the ResourceIdentity of the state whose T history is KNOWN consumed. Allowed once, before any resource cursor
        exists; no hydrology, no water, no time.'''
        self._check_token(self.token, 'water connection')
        if self.day_binding is not None or self.resource_cursor is not None:
            raise PlantSessionStateError('water is already connected (binding / cursor exist); it is never silently replaced')
        self._check_initial(day_binding, initial_resource)
        cur = self.schedule.cursor
        sid = StageId(self.session_id, self.token, self.sequence + 1, 'connect', cur.isoformat(), cur.isoformat(),
                      self._plain_digest('connect', cur, cur), None, None, initial_resource.to_dict())
        nxt = self._make(self.config, self.laws, self.session_id, self.lineage, self.sequence + 1, self.base, self.base_offset_days, self.pop,
                         self.schedule, dict(self.counters_), dict(self.bio), initial_resource, sid, day_binding, self.pop_sha)
        return Staged(sid, self.token, self.session_id, nxt, None, {'kind': 'connect', 'sequence': self.sequence + 1})

    def memory_ledger(self, *, n_edges_hint=None):
        '''PUBLIC byte ledger (structural estimates, NOT measurements) so a caller can compare it with measured peaks: the accepted parent items,
        the coexisting candidate, the metadata-only start lower bound and the per-stage component estimates for this population.'''
        P, S, cfg = self.n_live, int(self.config.table.n_species), self.config
        g = self.pop.geometry
        rm = g.root_map
        items = {'working_arrays_45_per_live': PER_LIVE_BYTES * P, 'live_rows_and_ids_16_per_live': 16 * P,
                 'stress_and_annual_arrays': _demo.PERSISTENT_BYTES_PER_LIVE * P if self.laws.stress_on else 0,
                 'species_tables_and_integrals': 16 * S + _nbytes(cfg.table.memory()), 'phase2_3_tables': self.laws.bytes,
                 'base_population': ROW_BYTES_PER_PLANT * int(self.base.n), 'root_map': int(rm.resident_bytes),
                 'canopy_map': 0 if g.canopy_map is None else int(g.canopy_map.resident_bytes),
                 'retained_geometry_state': ROW_BYTES_PER_PLANT * int(g.state.n)}
        stages = {'demand_per_plant': _nbytes(_growth.memory_estimate('demand_per_plant', P, S, cfg.backend)),
                  'grow': _nbytes(_growth.memory_estimate('grow', P, S, cfg.backend)),
                  'uptake_plan': _nbytes(_uptake.estimate_plan_bytes(rm.n_plants, rm.n_edges, rm.n_resources, rm.backend)),
                  'uptake_feedback': _nbytes(_uptake.estimate_feedback_bytes(rm.n_plants, rm.n_edges, rm.n_resources, rm.backend)),
                  'credit_increment': _nbytes(_bio.estimate_credit_bytes(P, S, backend='cupy' if cfg.backend == 'cuda' else 'numpy')),
                  'rebuild_materialize_and_geometry': (MATERIALIZE_BYTES_PER_PLANT + GEOMETRY_BYTES_PER_PLANT) * int(self.base.n)}
        if g.canopy_map is not None:
            stages['canopy_pet'] = _nbytes(_pp.estimate_pet_bytes(g.canopy_map.n_plants, g.canopy_map.n_edges, g.canopy_map.backend))
        return {'reserve_bytes': int(cfg.reserve_bytes), 'max_bytes': int(cfg.max_bytes), 'accepted_items': items,
                'accepted_total': self.resident_bytes(), 'candidate_coexistence': self.candidate_bytes(), 'stage_estimates': stages,
                'start_lower_bound': self._lower_bound(cfg, int(self.base.n), int(self.base.n), self.base.domain.ncells),
                'definition': 'structural conservative estimates of live bytes; interpreter, JIT, CUDA context and allocator pools excluded'}

    # ---------------------------------------------------------------- 5. publish
    def publish(self, staged):
        '''Adopt a staged candidate: returns the NEW accepted session. All checks precede any mutation; a stale parent, another lineage, a
        discarded / already published candidate and a second publication from this parent are refused. A day plan is consumed here.'''
        if not isinstance(staged, Staged):
            raise PlantSessionError('staged must be a Staged candidate')
        if staged.session_id != self.session_id or staged.parent_token != self.token:
            raise PlantSessionStateError('the candidate was staged from another parent / lineage (stale); refused')
        if staged.status != 'staged':
            raise PlantSessionStateError(f'the candidate is {staged.status}; it cannot be published (duplicate / discarded)')
        if self._succ:
            raise PlantSessionStateError('this parent already published a successor; duplicate publication refused')
        if staged.plan is not None and staged.plan._used:
            raise PlantSessionStateError('the plan of this candidate was already published (duplicate); refused')
        self._succ.append(staged.stage_id.digest())
        staged._status[0] = 'published'
        if staged.plan is not None:
            staged.plan._used.append(1)
        return staged.next

    def published_stage(self):
        return self._succ[0] if self._succ else None


__all__ = ['VERSION', 'BACKENDS', 'TRANSPORT_KINDS', 'PlantSessionError', 'PlantSessionStateError', 'PlantSessionBoundaryError',
           'PlantSessionUnsupportedError', 'PlantSessionAdmissionError', 'DayContext', 'TransportDay', 'DayBinding', 'StageId', 'SessionConfig',
           'DailyPlan', 'WithdrawalProposal', 'DailyFeedback', 'Staged', 'PlantSession']
