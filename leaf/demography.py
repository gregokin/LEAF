'''leaf.demography: opt-in age / background hazard and persistent low-growth (maintenance-stress) laws
(law `age_background_stress_v1`; UNQUALIFIED, UNCALIBRATED, explicit LEAF aboveground-only adaptation).

This is NOT an ECOTONE or STEPWAT2 reproduction. The precedents recorded in reference/method_notes.md (a first-year
exemption, a slow-growth counter that a healthy year DECREMENTS rather than resets, an age-dependent hazard) are used as
structure only; no source constant (0.368, 9.5 C, a 1.2 exponent, hard-coded thresholds ...) is adopted as a default.
Every coefficient is a field of the species' catalog `mortality` section; `"unimplemented"` / null is explicitly
inactive, anything else unknown or malformed REFUSES. There is no field default and no life-expectancy cutoff: the
catalog `life_expectancy` (a mean) is metadata and is never read here, so reaching it kills nothing.

Catalog
  "mortality": {"law": "age_background_stress_v1", "status": str, "notes": str, "units": {...},
    "traits": {"age_hazard": null | {"weibull_shape": > 0, "weibull_scale_years": > 0},
               "background_hazard_per_year": >= 0,
               "stress": null | {"metric": "support_ratio" | "growth_ratio", "low_threshold": (0, 1],
                                 "exempt_until_age_days": >= 0, "limit_years": int >= 1,
                                 "counter_cap_years": int >= limit_years, "mortality_probability": [0, 1],
                                 "biomass_loss_fraction": [0, 1)}}}

Decision period (one annual decision, the explicit scheduler boundary of the session). Years are 365-day years (the
catalog TIME_UNIT_DAYS convention). With the plant ages a0 = max(age_end - period_days, 0) and a1 = age_end (days, live
plants age only while live):
  cumulative age hazard   H(a) = (a / 365 / scale_years) ** shape                         (Weibull; 0 when off)
  total hazard            dH = max(H(a1) - H(a0), 0) + background_hazard_per_year * min(period_days, a1) / 365
                          (Phase 3 newborn correction: a plant born inside the period is charged only for the time it was
                          alive; identical to the Phase 2 formula whenever a1 >= period_days)
  death probability       p = -expm1(-dH)       p = 0 for dH = 0 and p = 1 exactly when dH is large enough
  keyed uniform           u = keyed_uniform(seed, decision_counter, STREAM_AGE, plant_id);  death iff u < p
so probability 0 never kills and probability 1 always kills. A plant that dies is `dead_standing`: it keeps its B / E.

Persistent low-growth stress (per plant counter, int32, persisted). The session accumulates over the year, per plant,
the growth outputs it already computes (no new maintenance-water fraction, no second application of satisfaction S, no
extra withdrawal): required support R = sum of `required_support_kg_year`, actual support A = sum of
`actual_support_kg_year` (kg*year; actual native feedback), planned growth P = sum(normal_plan + extra_plan) and actual
growth G = sum(normal_gain + extra_gain) (kg). Support ratio A/R and growth ratio G/P are DIAGNOSTICS OF SUPPORT /
GROWTH, not biomass-production yields. A plant is
  exempt (counter unchanged)         age_end <= exempt_until_age_days  (first-year rule),
  not evaluated (counter unchanged)  the denominator is 0: no required support (dormant / zero activity / zero live
                                     fraction) or no planned growth (mature at the size cap, dormant, zero potential),
                                     so DORMANCY OR MATURITY ALONE IS NEVER STRESS,
  low                                ratio < low_threshold  -> counter = min(counter + 1, counter_cap_years),
  healthy                            otherwise              -> counter = max(counter - 1, 0)   (recovery, no reset).
A LOW year with counter >= limit_years is a stressed year: the plant dies with `mortality_probability` (keyed draw,
STREAM_STRESS, same decision counter) and otherwise loses `biomass_loss_fraction` of B (E scaled equally). A healthy,
exempt or unevaluated year never triggers the stress response.

Keyed randomness. `keyed_uniform` is a COUNTER-BASED integer hash (SplitMix64 finalizer over (seed, stream, decision
counter, plant id)): stateless, no global RNG, independent of row order and of species order, identical on NumPy, Numba
and CuPy (uint64 arithmetic is exact), and reproducible after a restart because only (seed, counter) are state. The
counter advances only when a decision is accepted, so rejected calls consume nothing.

Backends: 'numpy' vectorised reference, 'numba' compiled serial kernel, 'cuda' the same vectorised code on CuPy arrays
bound to one explicit device (no per-plant Python loop; fixed-size diagnostic packets are read back). Hazard
probabilities use pow / expm1 whose last bit may differ between libraries: parity is exact for the integer hash and
flags and within a few ULP for probabilities. Memory is admitted before allocation (`memory_estimate`).
'''
from __future__ import annotations

import math

import numpy as np

from leaf import traits as _traits

VERSION = '1.0.0'
LAW = 'age_background_stress_v1'
BACKENDS = ('numpy', 'numba', 'cuda')
METRICS = ('support_ratio', 'growth_ratio')
STREAM_AGE, STREAM_STRESS = 1, 2
CAUSE_NONE, CAUSE_AGE, CAUSE_STRESS = 0, 1, 2
CAUSE_NAMES = {CAUSE_NONE: 'none', CAUSE_AGE: 'age_background', CAUSE_STRESS: 'stress'}
FLAG_NOT_EVALUATED, FLAG_EXEMPT, FLAG_HEALTHY, FLAG_LOW, FLAG_DEAD_AGE = 0, 1, 2, 3, 4
FLAG_NAMES = ('not_evaluated', 'exempt_first_year', 'healthy', 'low', 'died_age_background')
DAYS_PER_YEAR = 365.0
SEED_MAX = (1 << 64) - 1
PERSISTENT_BYTES_PER_LIVE = 4 + 4 * 8            # int32 stress counter + four float64 annual accumulators
BYTES_PER_PLANT_HOST = 448                       # ids / ages (16) + outputs (22) + temporaries (uint64 hash, masks, ~20 f64)
BYTES_PER_PLANT_CUDA = 896                       # host + device copies
_SECTION_KEYS = ('law', 'traits', 'status', 'notes', 'units')
_TRAIT_KEYS = ('age_hazard', 'background_hazard_per_year', 'stress')
_AGE_KEYS = ('weibull_shape', 'weibull_scale_years')
_STRESS_KEYS = ('metric', 'low_threshold', 'exempt_until_age_days', 'limit_years', 'counter_cap_years',
                'mortality_probability', 'biomass_loss_fraction')
_F_ROWS = ('shape', 'scale', 'bg', 'low', 'exempt', 'p_stress', 'loss')
_I_ROWS = ('age_on', 'stress_on', 'metric', 'limit', 'cap')
_FI = {n: i for i, n in enumerate(_F_ROWS)}
_II = {n: i for i, n in enumerate(_I_ROWS)}
(F_SHAPE, F_SCALE, F_BG, F_LOW, F_EXEMPT, F_PSTRESS, F_LOSS) = range(7)
(I_AGE_ON, I_STRESS_ON, I_METRIC, I_LIMIT, I_CAP) = range(5)
TABLE_BYTES_PER_SPECIES = 8 * len(_F_ROWS) + 8 * len(_I_ROWS)
# SplitMix64 constants (public-domain finalizer); the stream constant separates the independent streams
_M = (1 << 64) - 1
_GOLD, _C1, _C2, _CSTREAM = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB, 0xD6E8FEB86659FD93
_U_GOLD, _U_C1, _U_C2 = np.uint64(_GOLD), np.uint64(_C1), np.uint64(_C2)
_U_S30, _U_S27, _U_S31, _U_S11 = np.uint64(30), np.uint64(27), np.uint64(31), np.uint64(11)
_INV53 = 1.0 / 9007199254740992.0
CAVEATS = (
    'UNCALIBRATED explicit LEAF aboveground-only adaptation; not an ECOTONE or STEPWAT2 reproduction; no source '
    'constant is a default',
    'life_expectancy is metadata: no cutoff at the mean; the Weibull scale is its own species coefficient',
    'support / growth ratios are diagnostics of actual vs required (planned) support, not production yields',
    'per-plant counters and annual accumulators exist only for live plants (state bounded by the live population)',
)


class DemographyError(ValueError):
    '''Any law, argument or data error (nothing is returned / mutated).'''


class DemographyBudgetError(DemographyError):
    '''The explicit byte bound would be exceeded; raised before any allocation.'''


class DemographyBackendError(DemographyError):
    '''Explicit backend / device failure (no fallback, no silent device switch).'''


# ------------------------------------------------------------------ seed / keyed randomness
def check_seed(seed):
    '''Strict non-boolean integer in [0, 2**64).'''
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= int(seed) <= SEED_MAX:
        raise DemographyError(f'rng seed must be a non-boolean integer in [0, 2**64), got {seed!r}')
    return int(seed)


def check_counter(counter):
    if isinstance(counter, (bool, np.bool_)) or not isinstance(counter, (int, np.integer)) or int(counter) < 0:
        raise DemographyError(f'decision counter must be a non-negative integer, got {counter!r}')
    return int(counter)


def _mix_int(x):
    z = (x + _GOLD) & _M
    z = ((z ^ (z >> 30)) * _C1) & _M
    z = ((z ^ (z >> 27)) * _C2) & _M
    return z ^ (z >> 31)


def stream_key(seed, counter, stream):
    '''64-bit key of (seed, decision counter, stream) as a Python int (exact integer arithmetic).'''
    seed, counter = check_seed(seed), check_counter(counter)
    if stream not in (STREAM_AGE, STREAM_STRESS):
        raise DemographyError('unknown RNG stream')
    k0 = _mix_int((seed ^ ((_CSTREAM * (stream + 1)) & _M)) & _M)
    return _mix_int((k0 + ((counter * _GOLD) & _M) + 1) & _M)


def keyed_uniform_py(seed, counter, stream, plant_id):
    '''Independent pure-Python reference of the uniform draw u in [0, 1) of ONE plant (tests / documentation).'''
    k = stream_key(seed, counter, stream)
    h = _mix_int((k + ((int(plant_id) & _M) * _GOLD)) & _M)
    return float(h >> 11) * _INV53


def _uniform_xp(xp, key, ids_u64):
    with np.errstate(over='ignore'):
        z = ids_u64 * _U_GOLD + np.uint64(key)
        z = z + _U_GOLD
        z = (z ^ (z >> _U_S30)) * _U_C1
        z = (z ^ (z >> _U_S27)) * _U_C2
        z = z ^ (z >> _U_S31)
    return (z >> _U_S11).astype(xp.float64) * _INV53


def keyed_uniform(seed, counter, stream, plant_ids):
    '''Vectorised uniform draws for host int64 plant IDs (NumPy). Same values on every backend and any row order.'''
    ids = np.asarray(plant_ids)
    if ids.dtype.kind not in 'iu' or ids.ndim != 1:
        raise DemographyError('plant_ids must be a 1-D integer array')
    return _uniform_xp(np, stream_key(seed, counter, stream), ids.astype(np.uint64))


# ------------------------------------------------------------------ catalog law parsing
def _real(v, name):
    try:
        return _traits.real(v, name)
    except _traits.CatalogError as exc:
        raise DemographyError(str(exc)) from exc


def _exact(d, keys, where):
    if not isinstance(d, dict):
        raise DemographyError(f'{where} must be null or an object')
    missing = [k for k in keys if k not in d]
    unknown = sorted(set(d) - set(keys))
    if missing or unknown:
        raise DemographyError(f'{where} must hold exactly {list(keys)}; missing {missing}, unknown {unknown}')


def _int(v, name, lo):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or int(v) < lo:
        raise DemographyError(f'{name} must be an integer >= {lo}, got {v!r}')
    return int(v)


def _parse_species(sec, sid):
    where = f'species {sid!r} mortality'
    fl = {'shape': 1.0, 'scale': 1.0, 'bg': 0.0, 'low': 1.0, 'exempt': 0.0, 'p_stress': 0.0, 'loss': 0.0}
    it = {'age_on': 0, 'stress_on': 0, 'metric': 0, 'limit': 1, 'cap': 1}
    if sec is None or (isinstance(sec, dict) and sec.get('law') == _traits.UNIMPLEMENTED):
        return False, fl, it
    if not isinstance(sec, dict):
        raise DemographyError(f'{where} must be null or an object')
    law = sec.get('law')
    if law != LAW:
        raise DemographyError(f'{where}.law {law!r} is unknown to leaf.demography (supported: {LAW!r}, or '
                              f'"{_traits.UNIMPLEMENTED}" for an explicitly inactive section)')
    unknown = sorted(set(sec) - set(_SECTION_KEYS))
    if unknown:
        raise DemographyError(f'{where}: unknown keys {unknown} (allowed: {list(_SECTION_KEYS)})')
    tr = sec.get('traits')
    _exact(tr, _TRAIT_KEYS, f'{where}.traits')
    bg = _real(tr['background_hazard_per_year'], f'{where}.traits.background_hazard_per_year')
    if bg < 0.0:
        raise DemographyError(f'{where}.traits.background_hazard_per_year must be >= 0, got {bg}')
    fl['bg'] = bg
    ah = tr['age_hazard']
    if ah is not None:
        w = f'{where}.traits.age_hazard'
        _exact(ah, _AGE_KEYS, w)
        fl['shape'] = _traits.positive(ah['weibull_shape'], f'{w}.weibull_shape')
        fl['scale'] = _traits.positive(ah['weibull_scale_years'], f'{w}.weibull_scale_years')
        it['age_on'] = 1
    st = tr['stress']
    if st is not None:
        w = f'{where}.traits.stress'
        _exact(st, _STRESS_KEYS, w)
        m = st['metric']
        if not isinstance(m, str) or m not in METRICS:
            raise DemographyError(f'{w}.metric must be one of {list(METRICS)}, got {m!r}')
        it['metric'] = METRICS.index(m)
        low = _real(st['low_threshold'], f'{w}.low_threshold')
        if not 0.0 < low <= 1.0:
            raise DemographyError(f'{w}.low_threshold must lie in (0, 1], got {low}')
        fl['low'] = low
        ex = _real(st['exempt_until_age_days'], f'{w}.exempt_until_age_days')
        if ex < 0.0:
            raise DemographyError(f'{w}.exempt_until_age_days must be >= 0, got {ex}')
        fl['exempt'] = ex
        it['limit'] = _int(st['limit_years'], f'{w}.limit_years', 1)
        it['cap'] = _int(st['counter_cap_years'], f'{w}.counter_cap_years', it['limit'])
        p = _real(st['mortality_probability'], f'{w}.mortality_probability')
        loss = _real(st['biomass_loss_fraction'], f'{w}.biomass_loss_fraction')
        if not 0.0 <= p <= 1.0:
            raise DemographyError(f'{w}.mortality_probability must lie in [0, 1], got {p}')
        if not 0.0 <= loss < 1.0:
            raise DemographyError(f'{w}.biomass_loss_fraction must lie in [0, 1) (a surviving plant keeps positive '
                                  f'biomass; use mortality_probability for death), got {loss}')
        fl['p_stress'], fl['loss'] = p, loss
        it['stress_on'] = 1
    return True, fl, it


def _cupy():
    try:
        import cupy
    except Exception as exc:   # pragma: no cover - environment dependent
        raise DemographyBackendError(f'backend "cuda" needs the CuPy package: {exc}') from exc
    return cupy


def _numba():
    try:
        import numba
    except Exception as exc:   # pragma: no cover - environment dependent
        raise DemographyBackendError(f'backend "numba" needs the numba package: {exc}') from exc
    return numba


class CompiledDemography:
    '''Immutable S-sized species table (sorted catalog ids). `any_active` False => no array is held (zero cost).'''

    def __init__(self, species_ids, catalog_hash, backend, device, active, tf, ti, tf_dev, ti_dev):
        self.species_ids = tuple(species_ids)
        self.catalog_hash = catalog_hash
        self.backend = backend
        self.device = device
        self.active = tuple(active)
        self.any_active = bool(any(active))
        self.any_stress = bool(self.any_active and np.any(ti[I_STRESS_ON] == 1)) if ti is not None else False
        self._tf, self._ti, self._tf_dev, self._ti_dev = tf, ti, tf_dev, ti_dev

    @property
    def n_species(self):
        return len(self.species_ids)

    def resident_bytes(self):
        if not self.any_active:
            return 0
        return TABLE_BYTES_PER_SPECIES * self.n_species * (2 if self.backend == 'cuda' else 1)

    def counter_cap(self):
        '''int64 [S] counter caps (host copy; 0 for inactive species) for restart validation.'''
        if not self.any_active:
            return np.zeros(self.n_species, dtype=np.int64)
        return np.where(self._ti[I_STRESS_ON] == 1, self._ti[I_CAP], 0).astype(np.int64)

    def describe(self):
        return {'law': LAW, 'version': VERSION, 'backend': self.backend, 'device': self.device,
                'n_species': self.n_species, 'active_species': [s for s, a in zip(self.species_ids, self.active) if a],
                'any_stress': self.any_stress, 'catalog_hash': self.catalog_hash,
                'resident_bytes': self.resident_bytes()}

    def coefficients(self, species_id):
        if species_id not in self.species_ids:
            raise DemographyError(f'unknown species {species_id!r}')
        if not self.any_active:
            raise DemographyError('table holds no active species')
        i = self.species_ids.index(species_id)
        out = {n: float(self._tf[_FI[n], i]) for n in _F_ROWS}
        out.update({n: int(self._ti[_II[n], i]) for n in _I_ROWS})
        out['active'] = bool(self.active[i])
        return out


def table_compile_bytes(n_species, backend):
    '''Resident + transient bytes of ONE compile (host table built before the active / inactive decision is known,
    plus the device mirror on cuda), counted in full even when an inactive catalog drops the arrays.'''
    return TABLE_BYTES_PER_SPECIES * int(n_species) * (2 if backend == 'cuda' else 1)


def compile_species(catalog, backend, *, device=None, max_bytes=None, extra_reserve_bytes=0):
    '''Parse EVERY species' `mortality` section (refusing unknown / malformed active laws naming the species) and build
    the immutable table; an all-inactive catalog holds no array. cuda: an explicit current device id is required.
    `max_bytes` (optional strict integer) admits `table_compile_bytes` + `extra_reserve_bytes` BEFORE any host / device
    allocation or upload (DemographyBudgetError); None = the caller imposes no budget here.'''
    if not isinstance(catalog, _traits.SpeciesCatalog):
        raise DemographyError('catalog must be a leaf.traits.SpeciesCatalog')
    if backend not in BACKENDS:
        raise DemographyError(f'backend must be explicitly one of {BACKENDS}, got {backend!r}')
    ids = catalog.species_ids
    S = len(ids)
    if max_bytes is not None:
        for nm, v, lo in (('max_bytes', max_bytes, 1), ('extra_reserve_bytes', extra_reserve_bytes, 0)):
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or int(v) < lo:
                raise DemographyError(f'{nm} must be a non-boolean integer >= {lo}, got {v!r}')
        need = table_compile_bytes(S, backend) + int(extra_reserve_bytes)
        if need > int(max_bytes):
            raise DemographyBudgetError(f'compile_species needs {need} bytes > max_bytes={int(max_bytes)}; refused '
                                        'before any allocation')
    errors, active = [], []
    tf = np.zeros((len(_F_ROWS), S), dtype=np.float64)
    ti = np.zeros((len(_I_ROWS), S), dtype=np.int64)
    for k, sid in enumerate(ids):
        try:
            a, fl, it = _parse_species(catalog.section(sid, 'mortality'), sid)
        except (DemographyError, _traits.CatalogError) as exc:
            errors.append(str(exc))
            active.append(False)
            continue
        active.append(a)
        for n in _F_ROWS:
            tf[_FI[n], k] = fl[n]
        for n in _I_ROWS:
            ti[_II[n], k] = it[n]
    if errors:
        more = f' (+{len(errors) - 6} more)' if len(errors) > 6 else ''
        raise DemographyError('species catalog is not usable by leaf.demography: ' + ' | '.join(errors[:6]) + more)
    if not any(active):
        return CompiledDemography(ids, catalog.hash, backend, None, tuple(active), None, None, None, None)
    tf.setflags(write=False)
    ti.setflags(write=False)
    tf_dev = ti_dev = None
    dev_id = None
    if backend == 'cuda':
        cp = _cupy()
        if isinstance(device, (bool, np.bool_)) or not isinstance(device, (int, np.integer)) or int(device) < 0:
            raise DemographyBackendError('backend "cuda" needs an explicit non-negative integer device id')
        dev_id = int(device)
        cur = int(cp.cuda.runtime.getDevice())
        if cur != dev_id:
            raise DemographyBackendError(f'device {dev_id} requested but the current CUDA device is {cur}; this '
                                         'module never switches devices or falls back')
        tf_dev, ti_dev = cp.asarray(tf), cp.asarray(ti)
    elif backend == 'numba':
        _numba()
    return CompiledDemography(ids, catalog.hash, backend, dev_id, tuple(active), tf, ti, tf_dev, ti_dev)


def memory_estimate(n_live, n_species, backend):
    '''Itemised conservative bound of ONE `decide` call (host + device for cuda) and the persistent per-live bytes.'''
    if isinstance(n_live, (bool, np.bool_)) or not isinstance(n_live, (int, np.integer)) or int(n_live) < 0:
        raise DemographyError(f'n_live must be a non-negative integer, got {n_live!r}')
    P = int(n_live)
    per = BYTES_PER_PLANT_CUDA if backend == 'cuda' else BYTES_PER_PLANT_HOST
    return {'per_live_plant_bytes': per, 'decision_bytes': per * P, 'packets_bytes': 8 * (8 + 4 + 2) * int(n_species),
            'persistent_state_bytes': PERSISTENT_BYTES_PER_LIVE * P,
            'total_bytes': per * P + 8 * 14 * int(n_species),
            'scope': 'live-sized outputs, hash / mask temporaries, host + device copies on cuda; S-sized packets; '
                     'persistent = int32 counter + four float64 annual accumulators per live plant'}


# ------------------------------------------------------------------ kernels
def _decide_xp(xp, tf, ti, s, ids_u64, age, stress, req, act, plan, gain, period_days, key_age, key_stress):
    def f(row):
        return tf[row][s]

    def i(row):
        return ti[row][s]
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        y1 = age / DAYS_PER_YEAR
        y0 = xp.maximum(age - period_days, 0.0) / DAYS_PER_YEAR
        h_age = xp.power(y1 / f(F_SCALE), f(F_SHAPE)) - xp.power(y0 / f(F_SCALE), f(F_SHAPE))
        h_age = xp.where(i(I_AGE_ON) == 1, xp.maximum(h_age, 0.0), 0.0)
        # newborn correction (Phase 3): a plant only accrues background hazard for the time it was ALIVE in the period,
        # min(period_days, age_end); unchanged for every plant that lived the whole period (age_end >= period_days)
        expo = xp.minimum(age, period_days)
        dh = h_age + f(F_BG) * (expo / DAYS_PER_YEAR)
        p_age = -xp.expm1(-dh)
    dead_age = _uniform_xp(xp, key_age, ids_u64) < p_age
    st_on = i(I_STRESS_ON) == 1
    use_gain = i(I_METRIC) == 1
    num = xp.where(use_gain, gain, act)
    den = xp.where(use_gain, plan, req)
    exempt = age <= f(F_EXEMPT)
    has_den = den > 0.0
    evaluated = st_on & (~dead_age) & (~exempt) & has_den
    ratio = num / xp.where(has_den, den, 1.0)
    low = evaluated & (ratio < f(F_LOW))
    healthy = evaluated & (~low)
    c = stress.astype(xp.int64)
    c_new = xp.where(low, xp.minimum(c + 1, i(I_CAP)), xp.where(healthy, xp.maximum(c - 1, 0), c)).astype(xp.int32)
    stressed = low & (c_new >= i(I_LIMIT))
    dead_stress = stressed & (_uniform_xp(xp, key_stress, ids_u64) < f(F_PSTRESS))
    loss = xp.where(stressed & (~dead_stress), f(F_LOSS), 0.0)
    cause = xp.where(dead_age, CAUSE_AGE, xp.where(dead_stress, CAUSE_STRESS, CAUSE_NONE)).astype(xp.int8)
    flag = xp.where(dead_age, FLAG_DEAD_AGE, xp.where(~st_on, FLAG_NOT_EVALUATED, xp.where(
        exempt, FLAG_EXEMPT, xp.where(has_den, xp.where(low, FLAG_LOW, FLAG_HEALTHY), FLAG_NOT_EVALUATED)))).astype(xp.int8)
    return {'cause': cause, 'loss_fraction': loss, 'stress_new': c_new, 'flag': flag, 'p_age': p_age}


_KERNEL = {}


def _numba_kernel():
    if 'numba' in _KERNEL:
        return _KERNEL['numba']
    numba = _numba()

    @numba.njit(cache=False)
    def u01(key, pid):
        z = np.uint64(pid) * _U_GOLD + key
        z = z + _U_GOLD
        z = (z ^ (z >> _U_S30)) * _U_C1
        z = (z ^ (z >> _U_S27)) * _U_C2
        z = z ^ (z >> _U_S31)
        return np.float64(z >> _U_S11) * _INV53

    @numba.njit(cache=False)
    def kern(tf, ti, s, ids, age, stress, req, act, plan, gain, period_days, key_age, key_stress, o_cause, o_loss,
             o_stress, o_flag, o_p):
        n = age.shape[0]
        for p in range(n):
            k = s[p]
            a = age[p]
            y1 = a / DAYS_PER_YEAR
            a0 = a - period_days
            if a0 < 0.0:
                a0 = 0.0
            y0 = a0 / DAYS_PER_YEAR
            h_age = 0.0
            if ti[I_AGE_ON, k] == 1:
                h_age = (y1 / tf[F_SCALE, k]) ** tf[F_SHAPE, k] - (y0 / tf[F_SCALE, k]) ** tf[F_SHAPE, k]
                if h_age < 0.0:
                    h_age = 0.0
            expo = period_days if period_days < a else a      # newborn correction: time actually alive in the period
            dh = h_age + tf[F_BG, k] * (expo / DAYS_PER_YEAR)
            pa = -math.expm1(-dh)
            dead_age = u01(key_age, ids[p]) < pa
            st_on = ti[I_STRESS_ON, k] == 1
            num = act[p]
            den = req[p]
            if ti[I_METRIC, k] == 1:
                num = gain[p]
                den = plan[p]
            exempt = a <= tf[F_EXEMPT, k]
            has_den = den > 0.0
            evaluated = st_on and (not dead_age) and (not exempt) and has_den
            low = False
            healthy = False
            if evaluated:
                if num / den < tf[F_LOW, k]:
                    low = True
                else:
                    healthy = True
            c = np.int64(stress[p])
            if low:
                c = min(c + 1, ti[I_CAP, k])
            elif healthy:
                c = max(c - 1, 0)
            stressed = low and (c >= ti[I_LIMIT, k])
            dead_stress = stressed and (u01(key_stress, ids[p]) < tf[F_PSTRESS, k])
            loss = 0.0
            if stressed and (not dead_stress):
                loss = tf[F_LOSS, k]
            cause = CAUSE_NONE
            if dead_age:
                cause = CAUSE_AGE
            elif dead_stress:
                cause = CAUSE_STRESS
            flag = FLAG_NOT_EVALUATED
            if dead_age:
                flag = FLAG_DEAD_AGE
            elif not st_on:
                flag = FLAG_NOT_EVALUATED
            elif exempt:
                flag = FLAG_EXEMPT
            elif has_den:
                flag = FLAG_LOW if low else FLAG_HEALTHY
            o_cause[p] = cause
            o_loss[p] = loss
            o_stress[p] = c
            o_flag[p] = flag
            o_p[p] = pa

    _KERNEL['numba'] = kern
    return kern


def _hostify(a):
    return np.asarray(a.get() if hasattr(a, 'get') else a)


def _summaries(xp, table, s, out):
    '''Fixed-size diagnostic packets (S-sized): flag / cause counts, expected age deaths, dieback counts.'''
    S = table.n_species
    k = s.astype(xp.int64)
    flags = xp.bincount(k * 8 + out['flag'].astype(xp.int64), minlength=S * 8).reshape(S, 8)
    causes = xp.bincount(k * 4 + out['cause'].astype(xp.int64), minlength=S * 4).reshape(S, 4)
    p_sum = xp.bincount(k, weights=out['p_age'], minlength=S)
    dieback = xp.bincount(k, weights=(out['loss_fraction'] > 0.0).astype(xp.float64), minlength=S)
    return {'flag_counts': _hostify(flags), 'cause_counts': _hostify(causes), 'expected_age_deaths': _hostify(p_sum),
            'n_dieback': _hostify(dieback)}


GROWTH_RELATION_RTOL = 1e-9      # actual growth <= planned growth * (1 + rtol): sums of satisfaction-capped gains, FP64 rounding


def _validate_content(xp, table, s, ids, stress, req, act, plan, gain):
    '''Scientific content checks BEFORE any kernel, gather or bincount; scalar backend reductions only (no P readback,
    no Python P loop), nothing is clamped or repaired. Actual support MAY exceed required support (real surplus).'''
    S = table.n_species
    if int(s.min()) < 0 or int(s.max()) >= S:
        raise DemographyError(f'species_index must lie in [0, {S})')
    ids_h = np.asarray(ids)
    # canonical PlantState IDs are UNIQUE SIGNED int64 (zero, negatives and both endpoints are valid); the exact int64
    # dtype is enforced by `decide`. The keyed hash reads them modulo 2**64 unchanged (no abs, truncation or renumbering).
    if ids_h.dtype != np.int64 or int(np.unique(ids_h).shape[0]) != int(ids_h.shape[0]):
        raise DemographyError('plant_ids must be exact int64 values and unique (any sign, zero included)')
    for nm, a in (('required_support', req), ('actual_support', act), ('planned_growth', plan),
                  ('actual_growth', gain)):
        if not bool(xp.all(xp.isfinite(a) & (a >= 0.0))):
            raise DemographyError(f'{nm} must be finite and >= 0 for every plant')
    if bool(xp.any(gain > plan * (1.0 + GROWTH_RELATION_RTOL))):
        raise DemographyError('actual_growth exceeds planned_growth beyond the FP64 relation tolerance')
    if int(stress.min()) < 0:
        raise DemographyError('stress counters must be >= 0')
    cap = xp.asarray(table.counter_cap())
    if bool(xp.any(stress > cap[s.astype(xp.int64)])):
        raise DemographyError('a stress counter exceeds its species counter_cap_years (0 for species without a stress law)')


def decide(table, *, species_index, plant_ids, age_end_days, stress, required_support, actual_support, planned_growth,
           actual_growth, period_days, seed, counter, max_bytes, extra_reserve_bytes=0):
    '''One annual decision for P live plants (module docstring). `species_index` (int32), `stress` (int32) and the four
    annual accumulators (float64) are 1-D arrays in the backend namespace (NumPy for numpy / numba, CuPy on the table's
    device for cuda); `plant_ids` (int64) and `age_end_days` (float64) are HOST arrays (uploaded explicitly for cuda).
    Pure: nothing is mutated and no random state exists beyond (seed, counter). Returns backend arrays cause (int8),
    loss_fraction (float64), stress_new (int32), flag (int8), p_age (float64) and the host `summary` packets.'''
    if not isinstance(table, CompiledDemography):
        raise DemographyError('table must come from leaf.demography.compile_species')
    for nm, v, lo in (('max_bytes', max_bytes, 1), ('extra_reserve_bytes', extra_reserve_bytes, 0)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or int(v) < lo:
            raise DemographyError(f'{nm} must be a non-boolean integer >= {lo}, got {v!r}')
    be = table.backend
    seed, counter = check_seed(seed), check_counter(counter)
    P = int(np.shape(plant_ids)[0]) if np.ndim(plant_ids) == 1 else -1
    if P < 0:
        raise DemographyError('plant_ids must be a 1-D array')
    est = memory_estimate(P, table.n_species, be)
    if est['total_bytes'] + int(extra_reserve_bytes) > int(max_bytes):
        raise DemographyBudgetError(f'decide needs ~{est["total_bytes"] + int(extra_reserve_bytes)} bytes (P={P}) > '
                                    f'max_bytes={int(max_bytes)}; refused before any allocation')
    if not isinstance(period_days, (int, float, np.integer, np.floating)) or isinstance(period_days, (bool, np.bool_)) \
            or not math.isfinite(float(period_days)) or float(period_days) <= 0.0:
        raise DemographyError(f'period_days must be a finite number > 0, got {period_days!r}')
    period = float(period_days)
    ids = np.ascontiguousarray(plant_ids)
    age = np.ascontiguousarray(age_end_days)
    if ids.dtype != np.int64 or age.dtype != np.float64 or age.shape != ids.shape or not np.all(np.isfinite(age)) \
            or (P and float(age.min()) < 0.0):
        raise DemographyError('plant_ids must be int64 and age_end_days finite non-negative float64 of equal length')
    if not table.any_active or P == 0:
        raise DemographyError('decide needs an active table and at least one live plant')
    key_age, key_stress = stream_key(seed, counter, STREAM_AGE), stream_key(seed, counter, STREAM_STRESS)
    if be == 'cuda':
        cp = _cupy()
        cur = int(cp.cuda.runtime.getDevice())
        if cur != table.device:
            raise DemographyBackendError(f'table is bound to CUDA device {table.device} but the current device is {cur}')
        xp = cp
        for nm, a, dt in (('species_index', species_index, 'int32'), ('stress', stress, 'int32'),
                          ('required_support', required_support, 'float64'), ('actual_support', actual_support, 'float64'),
                          ('planned_growth', planned_growth, 'float64'), ('actual_growth', actual_growth, 'float64')):
            # an actual CuPy ndarray is required (NumPy >= 2 arrays also expose `.device`): check the type first
            if not isinstance(a, cp.ndarray) or str(a.dtype) != dt or a.shape != (P,) \
                    or int(a.device.id) != table.device:
                raise DemographyError(f'{nm} must be a {dt} CuPy array of length {P} on device {table.device}')
        s = species_index
        _validate_content(cp, table, s, ids, stress, required_support, actual_support, planned_growth, actual_growth)
        out = _decide_xp(cp, table._tf_dev, table._ti_dev, s.astype(cp.int64), cp.asarray(ids).astype(cp.uint64),
                         cp.asarray(age), stress, required_support, actual_support, planned_growth, actual_growth,
                         period, key_age, key_stress)
        summ = _summaries(cp, table, s, out)
    else:
        xp = np
        for nm, a, dt in (('species_index', species_index, np.int32), ('stress', stress, np.int32),
                          ('required_support', required_support, np.float64), ('actual_support', actual_support, np.float64),
                          ('planned_growth', planned_growth, np.float64), ('actual_growth', actual_growth, np.float64)):
            if not isinstance(a, np.ndarray) or a.dtype != dt or a.shape != (P,):
                raise DemographyError(f'{nm} must be a {np.dtype(dt)} NumPy array of length {P}')
        s = species_index
        _validate_content(np, table, s, ids, stress, required_support, actual_support, planned_growth, actual_growth)
        if be == 'numpy':
            out = _decide_xp(np, table._tf, table._ti, s.astype(np.int64), ids.astype(np.uint64), age, stress,
                             required_support, actual_support, planned_growth, actual_growth, period, key_age, key_stress)
        else:
            out = {'cause': np.empty(P, np.int8), 'loss_fraction': np.empty(P), 'stress_new': np.empty(P, np.int32),
                   'flag': np.empty(P, np.int8), 'p_age': np.empty(P)}
            _numba_kernel()(table._tf, table._ti, s.astype(np.int64), ids, age, stress, required_support, actual_support,
                            planned_growth, actual_growth, period, np.uint64(key_age), np.uint64(key_stress),
                            out['cause'], out['loss_fraction'], out['stress_new'], out['flag'], out['p_age'])
        summ = _summaries(np, table, s, out)
    pmin, pmax = float(out['p_age'].min()), float(out['p_age'].max())       # scalar reductions only
    if not (math.isfinite(pmin) and math.isfinite(pmax) and pmin >= 0.0 and pmax <= 1.0):
        raise DemographyError('hazard probability is not finite or outside [0, 1] (overflowing cumulative hazard); '
                              'refused, nothing consumed')
    out['summary'] = summ
    out['key_age'], out['key_stress'] = key_age, key_stress
    return out


__all__ = ['VERSION', 'LAW', 'BACKENDS', 'METRICS', 'STREAM_AGE', 'STREAM_STRESS', 'CAUSE_NONE', 'CAUSE_AGE',
           'CAUSE_STRESS', 'CAUSE_NAMES', 'FLAG_NAMES', 'FLAG_NOT_EVALUATED', 'FLAG_EXEMPT', 'FLAG_HEALTHY', 'FLAG_LOW',
           'FLAG_DEAD_AGE', 'PERSISTENT_BYTES_PER_LIVE', 'CAVEATS', 'DemographyError', 'DemographyBudgetError',
           'DemographyBackendError', 'CompiledDemography', 'check_seed', 'check_counter', 'stream_key',
           'keyed_uniform_py', 'keyed_uniform', 'compile_species', 'memory_estimate', 'decide']
