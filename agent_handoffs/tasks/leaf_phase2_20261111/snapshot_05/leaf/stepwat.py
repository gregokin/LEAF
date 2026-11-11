'''STEP-style individual aboveground growth: an OPT-IN weekly/year-end adaptation of the STEPWAT2 individual
growth rule for the existing native-water coupling (benchmarks.daily_water_growth.Simulation in
vegetation-only mode, driven by leaf.coupling.CoupledSimulation).

Source method (public STEPWAT2 revision 003cb9b4adbf13d64a39d6d62fd95280adac27fc; ST_resgroups.c rgroup_Grow /
_extra_growth, ST_species.c Species_GetBiomass, sxw_resource.c): relative size R = biomass / mature biomass,
normal annual increment dR = intrinsic_rate * gmod * R * (1 - R) with gmod = 0.95 * temperature_modifier /
max(PR, 1), PR = required / available biomass-support resources; surplus resources yield seasonal extra biomass
that is removed at the end of the season.

This module is a DOCUMENTED WEEKLY ADAPTATION of that rule, not a port and not an annual-trajectory
reproduction (see docs/stepwat-growth.md). It keeps the existing spatial proportional root competition and the
native residual-limited transpiration as the only water authority; STEP's patch-level largest-first resource
allocator, its added-transpiration resource correction and any nonzero dry fallback are deliberately absent.

Observed biomass target: TOTAL aboveground dry biomass per individual B (kg). A seasonal-extra component E (kg,
E <= B) is tracked internally so that persistent size growth (B - E) and season-end biomass loss stay distinct.
No root, leaf/wood or carbon pools; no new physical water store.

Per-plant growth runs in a compiled Numba kernel (CPU) or one fused CuPy RawKernel (GPU); plant vectors are
never copied to the host to grow them. Trait tables are S-sized (one row per root profile name). Only the small
daily temperature forcing ([S] floats) is transferred. Importing this module does not import the native package.
'''
from __future__ import annotations

import math

import numpy as np
from numba import njit

from benchmarks import spatial_mapping as sm

VERSION = '0.1.0'
MODEL_ID = 'stepwat_weekly_aboveground_v0.1'
MODEL_LABEL = ('STEP-style individual aboveground growth: weekly/year-end explicit-Euler adaptation of STEPWAT2 '
               'rgroup_Grow/_extra_growth (documented adaptation, not an unchanged STEPWAT routine or a port)')
DAYS_PER_YEAR = 365  # no-leap calendar; day 0 of the accepted-day clock is 1 January
WEEK = 7
GMOD_SCALE = 0.95  # STEP: gmod = 0.95 * temperature_modifier / max(PR, 1)
PI = math.pi
MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
MONTH_STARTS = tuple(sum(MONTH_DAYS[:m]) for m in range(12))  # 0, 31, 59, ..., 334
SOURCE_REVISION = '003cb9b4adbf13d64a39d6d62fd95280adac27fc'
_GH = 'https://github.com/DrylandEcology/STEPWAT2/blob/' + SOURCE_REVISION + '/'
SOURCE_URLS = {'ST_resgroups.c (rgroup_Grow, _extra_growth)': _GH + 'ST_resgroups.c',
               'ST_species.c (Species_GetBiomass)': _GH + 'ST_species.c',
               'sxw_resource.c (resource / transpiration mapping)': _GH + 'sxw_resource.c'}

# One row per root profile name (the authoritative species/PFT identifier). Every value explicit and finite.
TRAIT_DEFINITIONS = {
    'mature_biomass_kg': {'unit': 'kg dry aboveground mass per individual', 'range': '> 0',
                          'meaning': 'species mature total aboveground mass; R = (B - E) / mature_biomass_kg'},
    'intrinsic_rate_per_year': {'unit': '1/year', 'range': '[0, 1]',
                                'meaning': 'STEP intrinsic growth rate; bounded for the explicit weekly Euler step'},
    'biomass_support_kg_per_m3': {'unit': 'kg aboveground mass supported per m3 of actual transpiration per year',
                                  'range': '> 0',
                                  'meaning': 'ANNUAL biomass-support coefficient (STEP PR denominator); NOT a '
                                             'water-use efficiency and NOT a new-biomass production yield'},
    'live_fraction': {'unit': 'fraction of total aboveground mass that is live', 'range': '[0, 1]',
                      'meaning': 'scales both the required resource (B * live_fraction) and the canopy demand'},
    'canopy_biomass_density_kg_m2': {'unit': 'kg per m2 of canopy projection', 'range': '> 0',
                                     'meaning': 'allometric scale: B = pi * canopy_radius^2 * density, inverted '
                                                'after growth for the new canopy radius'},
    'potential_transpiration_m_per_day': {'unit': 'm/day of canopy-area water depth', 'range': '>= 0',
                                          'meaning': 'canopy transpiration DEMAND depth; explicitly distinct from '
                                                     'the native potential SOIL evaporation or reference ET'},
    'extra_growth_fraction': {'unit': 'kg seasonal mass per kg of surplus biomass-support', 'range': '[0, 1]',
                              'meaning': 'converts surplus (available - required) resource to seasonal mass E'},
    'monthly_activity': {'unit': '12 fractions (January..December), 365-day no-leap calendar', 'range': '[0, 1] each',
                         'meaning': 'explicit growth/uptake activity per month; selected at each period start; '
                                    '0 = dormancy (no demand, no temperature accumulation, no gain)'},
}
REQUIRED_TRAITS = tuple(TRAIT_DEFINITIONS)
_SCALAR_TRAITS = ('mature_biomass_kg', 'intrinsic_rate_per_year', 'biomass_support_kg_per_m3', 'live_fraction',
                  'canopy_biomass_density_kg_m2', 'potential_transpiration_m_per_day', 'extra_growth_fraction')
_N_DEVICE_S_ARRAYS = 7 + 12 + 4  # 7 scalar trait vectors, activity [S,12], temp integral, act_now, tf buffer, scratch
_N_HOST_S_ARRAYS = 7 + 12        # host trait mirror on the GPU (traits + activity)
_N_P_ARRAYS = 2                   # E (seasonal mass) and its candidate _Ec

RESOURCE_SUPPORT_NOTE = (
    'available = biomass_support_kg_per_m3 * actual period uptake (m3) is the aboveground mass that the water '
    'actually transpired in the period could SUPPORT for one year (STEP resource/PR convention); '
    'required = B * live_fraction * activity * dt_days / 365 is the same quantity pro-rated to the period. '
    'water_factor = min(1, available / required) (0 when required == 0) is 1 / max(PR, 1). The surplus '
    'max(available - required, 0) times extra_growth_fraction and the mean temperature factor becomes seasonal mass. '
    'Neither coefficient is a water-use efficiency or a production yield.')
CAVEATS = (
    'weekly (and partial year-end) explicit Euler adaptation of an annual rule; not an unchanged STEPWAT2 routine '
    'and not an annual-trajectory reproduction',
    'spatial proportional root competition (existing CSR graph, residual-limited native transpiration) replaces '
    'STEP patch-level largest-first resource allocation',
    'no added-transpiration resource correction, no nonzero dry fallback: zero actual uptake gives zero gain',
    'temperature_factor is a caller-supplied modifier in [0, 1]; no temperature-response law is asserted here',
    'no frost, recruitment, mortality, injury, plastic roots or detailed surface remapping; root vertical '
    'profiles and profile ids are fixed',
    'whole-plant carbon closure is not asserted; only the aboveground gain/loss ledger closes',
    'traits are caller-supplied fixtures or calibrations; this module ships no scientifically calibrated defaults',
)


# ------------------------------------------------------------ calendar helpers (pure)
def month_of_day(day):
    '''Month index 0..11 of accepted-day index `day` in the 365-day no-leap calendar (day 0 = 1 January).'''
    doy = int(day) % DAYS_PER_YEAR
    m = 0
    while m < 11 and MONTH_STARTS[m + 1] <= doy:
        m += 1
    return m


def is_growth_boundary(day):
    '''Growth updates happen when day % 7 == 0 OR day % 365 == 0 (day > 0).'''
    day = int(day)
    return day > 0 and (day % WEEK == 0 or day % DAYS_PER_YEAR == 0)


def expected_growth_updates(day):
    '''Weekly boundaries plus year-ends that are not themselves weekly boundaries (partial periods).'''
    day = int(day)
    n = day // WEEK
    for k in range(1, day // DAYS_PER_YEAR + 1):
        if (k * DAYS_PER_YEAR) % WEEK != 0:
            n += 1
    return n


# ------------------------------------------------------------ trait validation (pure; no arrays)
def _real(v, name):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name} must be a real number (not bool/str/None), got {v!r}')
    v = float(v)
    if not math.isfinite(v):
        raise ValueError(f'{name} must be finite, got {v}')
    return v


def _positive(v, name):
    v = _real(v, name)
    if v <= 0.0:
        raise ValueError(f'{name} must be > 0, got {v}')
    return v


def _nonnegative(v, name):
    v = _real(v, name)
    if v < 0.0:
        raise ValueError(f'{name} must be >= 0, got {v}')
    return v


def _unit(v, name):
    v = _real(v, name)
    if v < 0.0 or v > 1.0:
        raise ValueError(f'{name} must lie in [0, 1], got {v}')
    return v


_CHECKS = {'mature_biomass_kg': _positive, 'intrinsic_rate_per_year': _unit, 'biomass_support_kg_per_m3': _positive,
           'live_fraction': _unit, 'canopy_biomass_density_kg_m2': _positive,
           'potential_transpiration_m_per_day': _nonnegative, 'extra_growth_fraction': _unit}


class ValidatedTraits:
    '''Ordered per-profile trait rows (python floats) validated against the authoritative profile names.
    Holds NO arrays; host_arrays() materializes the S-sized tables only after the admission preflight.'''

    def __init__(self, names, rows):
        self.names = list(names)
        self.rows = rows  # list of dicts in profile order
        self.S = len(rows)

    def host_arrays(self):
        out = {k: np.ascontiguousarray([r[k] for r in self.rows], dtype=np.float64) for k in _SCALAR_TRAITS}
        out['monthly_activity'] = np.ascontiguousarray([r['monthly_activity'] for r in self.rows], dtype=np.float64)
        return out

    def as_report(self):
        return {name: {k: (list(r[k]) if k == 'monthly_activity' else r[k]) for k in REQUIRED_TRAITS}
                for name, r in zip(self.names, self.rows)}


def validate_traits(traits, profile_names):
    '''PURE validation of {profile name -> trait row}. Exactly one row per profile name of the root profile table
    (the authoritative species/PFT identifiers), exactly the REQUIRED_TRAITS keys per row, explicit finite values
    in the documented ranges; monthly_activity holds exactly 12 fractions. Nothing is inferred or defaulted.'''
    if not isinstance(traits, dict) or not traits:
        raise ValueError('stepwat_traits must be a non-empty mapping of root profile name -> trait row')
    names = list(profile_names)
    missing = [n for n in names if n not in traits]
    unknown = [k for k in traits if k not in names]
    if missing or unknown:
        raise ValueError(f'stepwat_traits must hold exactly one row per root profile name {names}; '
                         f'missing {missing}, unknown {unknown}')
    rows = []
    for name in names:
        row = traits[name]
        if not isinstance(row, dict):
            raise ValueError(f'stepwat_traits[{name!r}] must be a dict of trait values')
        extra = sorted(set(row) - set(REQUIRED_TRAITS))
        lacking = [k for k in REQUIRED_TRAITS if k not in row]
        if extra or lacking:
            raise ValueError(f'stepwat_traits[{name!r}] must define exactly {list(REQUIRED_TRAITS)}; '
                             f'missing {lacking}, unknown {extra}')
        out = {k: _CHECKS[k](row[k], f'stepwat_traits[{name!r}][{k!r}]') for k in _SCALAR_TRAITS}
        act = row['monthly_activity']
        if isinstance(act, (str, bytes, dict)) or not hasattr(act, '__len__') or len(act) != 12:
            raise ValueError(f'stepwat_traits[{name!r}][monthly_activity] must be a sequence of exactly 12 fractions')
        out['monthly_activity'] = [_unit(a, f'stepwat_traits[{name!r}][monthly_activity][{i}]') for i, a in enumerate(act)]
        rows.append(out)
    return ValidatedTraits(names, rows)


def validate_temperature_factor(tf, S):
    '''PURE validation of the HOST-ONLY temperature factor for one physical day: a real scalar or an [S]
    per-profile host sequence / 1-D NumPy array of finite values in [0, 1] (bool refused). Returns a contiguous
    float64 [S] host array. Device-backed values (anything exposing .get()/__cuda_array_interface__/.device
    that is not a NumPy array) are refused WITHOUT calling .get(): no transfer is ever attempted. NumPy shape and
    sequence length are inspected before any conversion and every sequence element is validated as a scalar
    before materialization, so malformed/nested/oversized input never triggers a bulk conversion. At most S host
    floats are materialized. Must be called BEFORE any native state or counter changes.'''
    S = int(S)
    if tf is None:
        raise ValueError('STEP growth mode requires an explicit temperature_factor (scalar or per-profile [S] values '
                         'in [0, 1]) for every physical day; none was supplied')
    if isinstance(tf, (bool, np.bool_)):
        raise ValueError('temperature_factor must not be a bool')
    if isinstance(tf, (int, float, np.integer, np.floating)):
        return np.full(S, _unit(tf, 'temperature_factor'), dtype=np.float64)
    if isinstance(tf, (str, bytes, dict)):
        raise ValueError('temperature_factor must be a number or an [S] host sequence/array')
    if isinstance(tf, np.ndarray):
        if tf.ndim != 1 or int(tf.shape[0]) != S:  # shape first: no conversion of wrongly shaped arrays
            raise ValueError(f'temperature_factor must be a scalar or have shape ({S},) (one value per root '
                             f'profile), got {tuple(tf.shape)}')
        if tf.dtype.kind not in 'fiu':
            raise ValueError(f'temperature_factor must be real-valued (not bool/object), got dtype {tf.dtype}')
        a = np.ascontiguousarray(tf, dtype=np.float64)  # S host floats
        if not bool(np.isfinite(a).all()) or float(a.min()) < 0.0 or float(a.max()) > 1.0:
            raise ValueError('temperature_factor values must be finite and lie in [0, 1]')
        return a
    if hasattr(tf, '__cuda_array_interface__') or hasattr(tf, 'get') or hasattr(tf, 'device'):
        raise ValueError('temperature_factor is a HOST-only forcing: device-backed values are refused (no .get() '
                         'transfer is attempted); pass a Python number, a host sequence or a 1-D NumPy array')
    if not hasattr(tf, '__len__'):
        raise ValueError('temperature_factor must be a number or an [S] host sequence/array')
    if len(tf) != S:  # length before touching any element
        raise ValueError(f'temperature_factor sequence must have exactly {S} entries (one per root profile), '
                         f'got {len(tf)}')
    vals = [_unit(v, f'temperature_factor[{i}]') for i, v in enumerate(tf)]  # scalar check per element, no bulk cast
    return np.array(vals, dtype=np.float64)


# ------------------------------------------------------------ memory accounting (pure)
def state_bytes_estimate(P, S):
    '''Bytes of ALL added resident/candidate/trait/scratch arrays on the compute side: E[P], _Ec[P] (FP64) plus
    S-sized FP64 trait vectors (7), monthly activity [S,12], temperature integral, current activity, temperature
    forcing buffer and one S-sized scratch.'''
    return 8 * (_N_P_ARRAYS * int(P) + _N_DEVICE_S_ARRAYS * int(S))


def host_mirror_bytes(S, gpu):
    '''Host trait mirror kept on the GPU backend (traits + activity); zero on the CPU (same arrays).'''
    return 8 * _N_HOST_S_ARRAYS * int(S) if gpu else 0


# ------------------------------------------------------------ compiled kernels
@njit(cache=True)
def _grow_cpu(B, E, U, pid, mature, rate, support, live, density, ptr, extra, activity, tint,
              m_cur, m_next, mult, dt_days, year_end, Bc, Ec, gc, lc, cc, rc, Dc):
    '''Per-plant STEP-style period update (candidates only; resident state untouched). See docs for symbols.'''
    for p in range(B.shape[0]):
        s = pid[p]
        b = B[p]
        e = E[p]
        persistent = b - e
        m = mature[s]
        rel = persistent / m
        act = activity[s, m_cur]
        frac = act * dt_days / 365.0
        required = b * live[s] * frac
        available = support[s] * U[p]
        water = 0.0
        if required > 0.0:
            water = available / required
            if water > 1.0:
                water = 1.0
        ng = rate[s] * 0.95 * tint[s] * water * persistent * (1.0 - rel)
        room = m - b
        if room < 0.0:
            room = 0.0
        if ng > room:
            ng = room
        if ng < 0.0:
            ng = 0.0
        surplus = available - required
        if surplus < 0.0:
            surplus = 0.0
        mean_tf = 0.0
        if frac > 0.0:
            mean_tf = tint[s] / frac
            if mean_tf > 1.0:
                mean_tf = 1.0
        xg = extra[s] * surplus * mean_tf
        room2 = m - b - ng
        if room2 < 0.0:
            room2 = 0.0
        if xg > room2:
            xg = room2
        if year_end:
            loss = e + xg
            e_new = 0.0
        else:
            loss = 0.0
            e_new = e + xg
        b_new = persistent + ng + e_new
        Bc[p] = b_new
        Ec[p] = e_new
        gc[p] = ng + xg
        lc[p] = loss
        c = math.sqrt(b_new / (PI * density[s]))
        cc[p] = c
        rc[p] = mult * c
        Dc[p] = PI * c * c * ptr[s] * live[s] * activity[s, m_next]


_CUDA_SRC = r'''
extern "C" __global__ void stepwat_grow(const double* B, const double* E, const double* U, const int* pid,
                                        const double* mature, const double* rate, const double* support,
                                        const double* live, const double* density, const double* ptr,
                                        const double* extra, const double* activity, const double* tint,
                                        const int m_cur, const int m_next, const double mult, const double dt_days,
                                        const int year_end, double* Bc, double* Ec, double* gc, double* lc,
                                        double* cc, double* rc, double* Dc, const long long P) {
    const double PI = 3.141592653589793;
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    int s = pid[p];
    double b = B[p];
    double e = E[p];
    double persistent = b - e;
    double m = mature[s];
    double rel = persistent / m;
    double act = activity[(long long)s * 12 + m_cur];
    double frac = act * dt_days / 365.0;
    double required = b * live[s] * frac;
    double available = support[s] * U[p];
    double water = 0.0;
    if (required > 0.0) { water = available / required; if (water > 1.0) water = 1.0; }
    double ng = rate[s] * 0.95 * tint[s] * water * persistent * (1.0 - rel);
    double room = m - b; if (room < 0.0) room = 0.0;
    if (ng > room) ng = room;
    if (ng < 0.0) ng = 0.0;
    double surplus = available - required; if (surplus < 0.0) surplus = 0.0;
    double mean_tf = 0.0;
    if (frac > 0.0) { mean_tf = tint[s] / frac; if (mean_tf > 1.0) mean_tf = 1.0; }
    double xg = extra[s] * surplus * mean_tf;
    double room2 = m - b - ng; if (room2 < 0.0) room2 = 0.0;
    if (xg > room2) xg = room2;
    double loss, e_new;
    if (year_end) { loss = e + xg; e_new = 0.0; } else { loss = 0.0; e_new = e + xg; }
    double b_new = persistent + ng + e_new;
    Bc[p] = b_new;
    Ec[p] = e_new;
    gc[p] = ng + xg;
    lc[p] = loss;
    double c = sqrt(b_new / (PI * density[s]));
    cc[p] = c;
    rc[p] = mult * c;
    Dc[p] = PI * c * c * ptr[s] * live[s] * activity[(long long)s * 12 + m_next];
}
'''
_GPU_CACHE = {}


def _gpu_kernel():
    if 'grow' not in _GPU_CACHE:
        _GPU_CACHE['grow'] = sm.cp.RawKernel(_CUDA_SRC, 'stepwat_grow', options=('-std=c++11',))
    return _GPU_CACHE['grow']


def warm_cpu_kernel():
    '''Compile/load the cached Numba kernel on a one-plant fixture (startup; values are discarded).'''
    one = np.ones(1)
    z = np.zeros(1)
    _grow_cpu(one, z, one, np.zeros(1, np.int32), np.full(1, 2.0), one, one, one, one, one, one,
              np.ones((1, 12)), z, 0, 0, 1.0, 7.0, 0, z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), z.copy())


# ------------------------------------------------------------ resident growth state (one per Simulation)
class StepwatState:
    '''Resident STEP growth state owned by a vegetation Simulation: S-sized trait tables (compute side; host
    mirror on the GPU), per-plant seasonal mass E and its candidate, the per-profile temperature integral, the
    held activity of the current period and the period clock. Allocated only AFTER the admission preflight.'''

    def __init__(self, traits, xp, gpu, P, mult):
        if not isinstance(traits, ValidatedTraits):
            raise TypeError('traits must be a ValidatedTraits (validate_traits())')
        self.traits, self.xp, self.gpu = traits, xp, bool(gpu)
        self.P, self.S, self.mult = int(P), traits.S, float(mult)
        self.host = host = traits.host_arrays()  # on the CPU these ARE the compute arrays (no mirror)
        dev = {k: xp.ascontiguousarray(xp.asarray(v, dtype=xp.float64)) for k, v in host.items()}
        self.mature, self.rate = dev['mature_biomass_kg'], dev['intrinsic_rate_per_year']
        self.support, self.live = dev['biomass_support_kg_per_m3'], dev['live_fraction']
        self.density, self.ptr = dev['canopy_biomass_density_kg_m2'], dev['potential_transpiration_m_per_day']
        self.extra, self.activity = dev['extra_growth_fraction'], dev['monthly_activity']
        self.E = xp.zeros(self.P, dtype=xp.float64)
        self._Ec = xp.zeros(self.P, dtype=xp.float64)
        self.temp_integral = xp.zeros(self.S, dtype=xp.float64)
        self.act_now = xp.zeros(self.S, dtype=xp.float64)
        self._tf = xp.zeros(self.S, dtype=xp.float64)
        self._sc = xp.zeros(self.S, dtype=xp.float64)
        self.period_start_day = 0
        self.growth_updates = self.year_end_updates = 0
        self.normal_gain_total = self.extra_gain_total = self.seasonal_loss_total = 0.0
        self.month_cur = -1
        self._set_period_month(0)

    # ---- period activity
    def _set_period_month(self, day):
        m = month_of_day(day)
        self.month_cur = m
        self.act_now[...] = self.activity[:, m]  # S-sized; held constant until the next successful commit

    # ---- memory
    def device_arrays(self):
        return (('stepwat.E', self.E), ('stepwat._Ec', self._Ec), ('stepwat.mature_biomass_kg', self.mature),
                ('stepwat.intrinsic_rate_per_year', self.rate), ('stepwat.biomass_support_kg_per_m3', self.support),
                ('stepwat.live_fraction', self.live), ('stepwat.canopy_biomass_density_kg_m2', self.density),
                ('stepwat.potential_transpiration_m_per_day', self.ptr), ('stepwat.extra_growth_fraction', self.extra),
                ('stepwat.monthly_activity', self.activity), ('stepwat.temperature_integral', self.temp_integral),
                ('stepwat.activity_now', self.act_now), ('stepwat._tf', self._tf), ('stepwat._sc', self._sc))

    def live_bytes(self):
        return int(sum(int(a.nbytes) for _, a in self.device_arrays()))

    def host_mirror_bytes(self):
        return int(sum(int(a.nbytes) for a in self.host.values())) if self.gpu else 0

    def array_records(self):
        where = 'device' if self.gpu else 'host'
        recs = [{'name': n, 'dtype': str(a.dtype), 'shape': [int(s) for s in a.shape], 'nbytes': int(a.nbytes),
                 'where': where} for n, a in self.device_arrays()]
        if self.gpu:
            recs += [{'name': 'stepwat.host.' + k, 'dtype': str(a.dtype), 'shape': [int(s) for s in a.shape],
                      'nbytes': int(a.nbytes), 'where': 'host'} for k, a in self.host.items()]
        return recs

    # ---- initial state (device ops; P-sized temporaries are covered by the scratch allowance)
    def initial_biomass(self, canopy, pid):
        '''B = pi * canopy_radius^2 * canopy_biomass_density_kg_m2[profile]; E = 0.'''
        return PI * canopy * canopy * self.density[pid]

    def check_initial_biomass(self, B, pid):
        xp = self.xp
        if not bool(xp.isfinite(B).all()) or float(B.min()) <= 0.0:
            raise ValueError('initial STEP aboveground biomass must be finite and positive')
        over = int((B > self.mature[pid]).sum())
        if over > 0:
            raise ValueError(f'{over} individual(s) start above their species mature_biomass_kg (B = pi*r^2*density '
                             'exceeds mature mass); refused before any water day')

    def initial_demand(self, canopy, pid):
        '''Dp = canopy_area * potential_transpiration_m_per_day * live_fraction * activity[month of day 0].'''
        return PI * canopy * canopy * self.ptr[pid] * self.live[pid] * self.act_now[pid]

    # ---- daily temperature forcing
    def validate_temperature_factor(self, tf):
        return validate_temperature_factor(tf, self.S)

    def accumulate_temperature(self, tf_host):
        '''temperature_integral[s] += temperature_factor[s] * activity_now[s] / 365 for one ACCEPTED day.'''
        xp = self.xp
        if self.gpu:
            self._tf.set(tf_host)  # small [S] H2D into the resident buffer
        else:
            self._tf[...] = tf_host
        xp.multiply(self._tf, self.act_now, out=self._sc)
        self._sc *= 1.0 / DAYS_PER_YEAR
        self.temp_integral += self._sc

    # ---- period growth candidate (no resident mutation)
    def growth_candidate(self, sim, dt_days, year_end):
        if int(dt_days) < 1:
            raise RuntimeError(f'growth update at day {sim.day} without accepted days since the last commit')
        m_next = month_of_day(sim.day)
        args = (sim.B, self.E, sim.uptake_week, sim.root_profile_ids, self.mature, self.rate, self.support, self.live,
                self.density, self.ptr, self.extra, self.activity, self.temp_integral)
        outs = (sim._Bc, self._Ec, sim._gc, sim._lc, sim._cc, sim._rc, sim._Dc)
        if self.gpu:
            _gpu_kernel()(((self.P + 255) // 256,), (256,), args + (
                np.int32(self.month_cur), np.int32(m_next), np.float64(self.mult), np.float64(dt_days),
                np.int32(1 if year_end else 0)) + outs + (np.int64(self.P),))
        else:
            _grow_cpu(*args, int(self.month_cur), int(m_next), float(self.mult), float(dt_days),
                      1 if year_end else 0, *outs)

    def check_candidates(self, sim):
        '''Candidate invariants (scalar readbacks): B, canopy, root finite positive; demand finite non-negative
        (zero during dormancy / zero transpiration trait); seasonal mass finite with 0 <= E <= B; gains/losses
        finite non-negative.'''
        xp = self.xp
        for name, a in (('_Bc', sim._Bc), ('_cc', sim._cc), ('_rc', sim._rc)):
            if not bool(xp.isfinite(a).all()) or float(a.min()) <= 0.0:
                raise RuntimeError(f'STEP growth candidate {name} not finite/positive at day {sim.day}; state preserved')
        for name, a in (('_Dc', sim._Dc), ('_gc', sim._gc), ('_lc', sim._lc), ('_Ec', self._Ec)):
            if not bool(xp.isfinite(a).all()) or float(a.min()) < 0.0:
                raise RuntimeError(f'STEP growth candidate {name} not finite/non-negative at day {sim.day}; state preserved')
        if float((self._Ec - sim._Bc).max()) > 0.0:
            raise RuntimeError(f'STEP seasonal candidate exceeds total biomass at day {sim.day}; state preserved')

    def commit(self, sim):
        '''Called inside the existing transaction AFTER every admission/invariant passed, together with the
        biomass/geometry/demand swap: swap seasonal mass, accumulate scalar totals, reset the temperature
        integral, select the new period's activity month and advance the period clock.'''
        f = float
        e_old, e_new, loss, gain = f(self.E.sum()), f(self._Ec.sum()), f(sim._lc.sum()), f(sim._gc.sum())
        extra = (e_new + loss) - e_old  # year-end: loss - E_old; otherwise E_new - E_old
        self.extra_gain_total += extra
        self.normal_gain_total += gain - extra
        self.seasonal_loss_total += loss
        self.E, self._Ec = self._Ec, self.E
        self.temp_integral.fill(0.0)
        self.period_start_day = sim.day
        self.growth_updates += 1
        if sim.day % DAYS_PER_YEAR == 0:
            self.year_end_updates += 1
        self._set_period_month(sim.day)

    # ---- reporting (streamed scalars; S-sized readback of the temperature integral only)
    def period_clock(self, sim):
        return {'start_day': self.period_start_day, 'pending_days': sim.day - self.period_start_day,
                'month_index': self.month_cur, 'growth_updates': self.growth_updates,
                'weekly_boundaries': sim.weeks_done, 'year_end_updates': self.year_end_updates,
                'expected_growth_updates': expected_growth_updates(sim.day), 'days_per_year': DAYS_PER_YEAR,
                'calendar': '365-day no-leap; day 0 of the accepted-day clock is 1 January',
                'update_rule': 'growth at day % 7 == 0 OR day % 365 == 0; dt_days = accepted days since the last '
                               'SUCCESSFUL commit (retry re-evaluates the same period once)'}

    def report(self, sim):
        f = float
        B_total, E_total = f(sim.B.sum()), f(self.E.sum())
        return {'model': MODEL_ID, 'label': MODEL_LABEL, 'version': VERSION,
                'source': {'revision': SOURCE_REVISION, 'urls': dict(SOURCE_URLS)},
                'biomass_unit': 'kg dry aboveground mass; the B/E ARRAYS are kg PER INDIVIDUAL, every scalar under '
                                'domain_totals_kg is a DOMAIN SUM over all individuals; total = persistent + seasonal',
                'domain_totals_kg': {'unit': 'kg, domain sum over all individuals (not per individual)',
                                     'initial': sim.B0, 'total': B_total, 'persistent': B_total - E_total,
                                     'seasonal': E_total, 'normal_gain': self.normal_gain_total,
                                     'extra_gain': self.extra_gain_total, 'seasonal_loss': self.seasonal_loss_total,
                                     'gain_cum': f(sim.gain_cum.sum()), 'loss_cum': f(sim.loss_cum.sum())},
                'per_individual_kg': {'unit': 'kg per individual (statistics of the resident B / E arrays)',
                                      'population': self.P, 'B_min': f(sim.B.min()), 'B_max': f(sim.B.max()),
                                      'B_mean': B_total / self.P, 'E_max': f(self.E.max()), 'E_mean': E_total / self.P},
                'traits': {'definitions': {k: dict(v) for k, v in TRAIT_DEFINITIONS.items()},
                           'per_profile': self.traits.as_report(), 'profile_names': list(self.traits.names),
                           'label': 'caller-supplied per root-profile rows; no calibrated defaults ship with the model'},
                'temperature': {'integral_per_profile': [f(v) for v in np.asarray(
                    self.temp_integral.get() if self.gpu else self.temp_integral)],
                    'label': 'sum over accepted days of temperature_factor*activity/365 since the last commit; '
                             'caller-supplied modifier, not an asserted temperature law'},
                'period': self.period_clock(sim),
                'resource_support': RESOURCE_SUPPORT_NOTE, 'caveats': list(CAVEATS)}


__all__ = ['VERSION', 'MODEL_ID', 'MODEL_LABEL', 'DAYS_PER_YEAR', 'WEEK', 'GMOD_SCALE', 'TRAIT_DEFINITIONS',
           'REQUIRED_TRAITS', 'SOURCE_REVISION', 'SOURCE_URLS', 'month_of_day', 'is_growth_boundary',
           'expected_growth_updates', 'validate_traits', 'validate_temperature_factor', 'ValidatedTraits',
           'StepwatState', 'state_bytes_estimate', 'host_mirror_bytes', 'warm_cpu_kernel']
