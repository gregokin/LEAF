'''leaf.recruitment: opt-in species-local simple seed establishment and nearby vegetative daughter plants
(law `simple_seed_clonal_v1`; UNQUALIFIED, UNCALIBRATED, explicit LEAF aboveground-only adaptation).

This is NOT a STEPWAT2 / ECOTONE reproduction. The STEPWAT2 precedent (reference/method_notes.md: a simple perennial
establishment path tests a probability and then draws an integer 1..max; clonal recovery adds seedling-size units to the
SAME parent) is used as STRUCTURE only. Two explicit LEAF adaptations are made and documented:
  * seeds: the trial is per ECOLOGICAL PATCH (an explicit, fixed physical size, independent of the hydraulic cell
    resolution) and species per annual opportunity, with an EXTERNAL-PROPAGULE assumption (no seed bank, no dispersal
    kernel, no light / nutrient / water filter);
  * vegetative reproduction creates separately located DAUGHTER plants (own ID, own age 0), not extra biomass on the
    parent. Their initial AGB is TRANSFERRED from the parent (parent B and E are debited exactly); no belowground pool
    exists and no biomass is created. The seedling AGB of a seed recruit is an EXTERNAL establishment / seed-reserve
    CREDIT in the ledger (not derived from current transpiration; no rain / ET / water is added or withdrawn).
No source constant is adopted: EVERY number is a field of the species' catalog `reproduction` section. null /
"unimplemented" is explicitly inactive; any other law, unknown key or malformed value REFUSES.

Catalog
  "reproduction": {"law": "simple_seed_clonal_v1", "status": str, "notes": str, "units": {...},
    "traits": {"seed": null | {"establishment_probability": [0, 1],      # per patch, species, annual opportunity
                               "max_recruits_per_patch": int >= 1,       # uniform integer 1..max on success
                               "initial_agb_dry_kg": > 0,                # per seedling, dry aboveground
                               "initial_seasonal_fraction": [0, 1]},     # E = fraction * B (explicit initialisation)
               "clonal": null | {"min_age_days": >= 0, "min_agb_dry_kg": > 0, "probability": [0, 1],
                                 "max_daughters_per_parent": int >= 1, "initial_agb_dry_kg": > 0,
                                 "max_donation_fraction": (0, 1),        # parent donates at most this fraction of B
                                 "dispersal_radius_m": > 0,              # uniform in a disk (sqrt radial draw)
                                 "max_attempts_per_daughter": int >= 1}}}
`seed` and `clonal` are independent: seed-only, clonal-only, both, or neither (inactive species).

Seeds. Ecological patches tile the domain from its SOUTH-WEST origin with a fixed `patch_size_m` square (the east / north
boundary patches are clipped to the closed domain; their ACTUAL clipped area defines the uniform position draw; the
trial probability is per patch irrespective of its area, an explicit assumption). For each species with a seed law, in
sorted species order, and each patch in row-major order: success iff u < p; then n = 1 + floor(u * max) recruits are placed
uniformly in the clipped patch rectangle. Seedlings take the current GROUND (Domain containing-cell sample of the
supplied current surface), age 0, B = initial_agb_dry_kg, E = fraction * B. Canopy overlap with existing plants is allowed
(not tested, not claimed otherwise).

Daughters. A LIVE parent is eligible when age >= min_age_days, B >= min_agb_dry_kg and max_donation_fraction * B >=
initial_agb_dry_kg. On a probability success it requests n = 1 + floor(u * max) daughters, capped so that the donation
never exceeds max_donation_fraction * B (a parent therefore stays LIVE with B' >= (1 - fraction) B > 0). Each requested
daughter draws up to `max_attempts_per_daughter` positions uniformly in a disk around the parent (r = R sqrt(u)); the first
position inside the CLOSED, NON-PERIODIC domain is accepted (nothing is clipped, wrapped or forced). Only ACCEPTED daughters
are paid for: parent B -= k * initial_agb_dry_kg and parent E -= k * initial_agb_dry_kg * (E / B); the daughter gets
exactly that B and that E (<= B). Failed placements cause no donation. Daughters take a new ID, the parent's species, age 0
and the current ground at their position. Parent / daughter canopies may overlap (documented, not tested).

IDs. New IDs are max(max existing ID, 0) + 1 ... in a deterministic order (seeds: species, patch, recruit; daughters:
parent ID, daughter index). Existing IDs may be zero, negative or the signed int64 endpoints and are never changed or
reused. Exhausting int64 refuses (before the birth arrays exist) only when births actually need IDs.

Keyed randomness. Every draw is a COUNTER-BASED integer hash (the SplitMix64 finalizer of leaf.demography, here with
two keys a and b) of (seed, reproduction counter, stream, a, b) with streams 11..24, DOMAIN-SEPARATED from the demography
streams 1 and 2 (whose results are unchanged). It is stateless: independent of row order, species order and restarts;
only (seed, counter) are state, and the counter advances only when an opportunity is accepted by the session.

Bounded work. All limits are explicit strict integers (max_new_plants, max_total_records, max_attempts, max_bytes);
nothing is truncated or raised. The REQUESTED recruit / daughter count is compared with max_new_plants (conservative:
placement failures do not relax it). Work arrays are sized by the ACTUAL number of patch trials, eligible parents, requested
daughters and births (prefix sums over counts; no per-plant Python loop, no dense matrix, no maximum-possible array).
The implementation is a vectorised NumPy HOST reference on every backend (explicit host population placement between
storms; on cuda the session transfers B / E to the host and counts it). Memory is admitted before each allocating stage.
'''
from __future__ import annotations

import math

import numpy as np

from leaf import demography as _demo
from leaf import state as _state
from leaf import traits as _traits

VERSION = '1.0.0'
LAW = 'simple_seed_clonal_v1'
STREAM_SEED_TRIAL, STREAM_SEED_COUNT, STREAM_SEED_X, STREAM_SEED_Y = 11, 12, 13, 14
STREAM_CLONAL_TRIAL, STREAM_CLONAL_COUNT, STREAM_CLONAL_R, STREAM_CLONAL_THETA = 21, 22, 23, 24
STREAMS = (STREAM_SEED_TRIAL, STREAM_SEED_COUNT, STREAM_SEED_X, STREAM_SEED_Y, STREAM_CLONAL_TRIAL,
           STREAM_CLONAL_COUNT, STREAM_CLONAL_R, STREAM_CLONAL_THETA)
INT64_MAX = (1 << 63) - 1
MAX_COUNT = 1 << 30                  # bound of integer counts / attempts (the draw index is packed into 32 bits)
FP64_EPS = float(np.finfo(np.float64).eps)
LEDGER_EPS_FACTOR = 32.0
# conservative bytes of the transient NumPy arrays (host)
BYTES_PER_TRIAL = 128
BYTES_PER_LIVE = 256
BYTES_PER_CANDIDATE = 192
BYTES_PER_BIRTH = 160
BYTES_PER_SPECIES_TABLE = 8 * 9 + 8 * 5
_SECTION_KEYS = ('law', 'traits', 'status', 'notes', 'units')
_SEED_KEYS = ('establishment_probability', 'max_recruits_per_patch', 'initial_agb_dry_kg', 'initial_seasonal_fraction')
_CLONAL_KEYS = ('min_age_days', 'min_agb_dry_kg', 'probability', 'max_daughters_per_parent', 'initial_agb_dry_kg',
                'max_donation_fraction', 'dispersal_radius_m', 'max_attempts_per_daughter')
_M = (1 << 64) - 1
_U_GOLD, _U_C1, _U_C2 = np.uint64(_demo._GOLD), np.uint64(_demo._C1), np.uint64(_demo._C2)
_U_S30, _U_S27, _U_S31, _U_S11, _U_S32 = np.uint64(30), np.uint64(27), np.uint64(31), np.uint64(11), np.uint64(32)
_U_ONE = np.uint64(1)
_INV53 = 1.0 / 9007199254740992.0
CAVEATS = (
    'UNCALIBRATED explicit LEAF aboveground-only adaptation; not a STEPWAT2 / ECOTONE reproduction; no source constant '
    'is a default',
    'seed recruits are an EXTERNAL-PROPAGULE establishment credit of dry AGB (no seed bank, no dispersal kernel, no '
    'light / nutrient / water filter, no water withdrawn); the probability is per patch regardless of clipped area',
    'daughters are separately located plants funded by an exact transfer of parent AGB and E (no belowground pool, no '
    'costless clonal biomass); canopy overlap with parents and neighbours is allowed and not tested',
    'finite non-periodic domain: daughter positions outside the domain are rejected after bounded attempts, never wrapped',
)


class ReproductionError(ValueError):
    '''Any law, argument or data error (nothing is returned / mutated).'''


class ReproductionBudgetError(ReproductionError):
    '''An explicit byte bound would be exceeded; raised before the stage allocates.'''


class ReproductionLimitError(ReproductionError):
    '''max_new_plants / max_total_records / max_attempts / int64 ID space would be exceeded (never truncated).'''


# ------------------------------------------------------------------ strict scalars
def _int(v, name, lo=0, hi=None):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or int(v) < lo \
            or (hi is not None and int(v) > hi):
        raise ReproductionError(f'{name} must be a non-boolean integer in [{lo}, {hi if hi is not None else "inf"}], '
                                f'got {v!r}')
    return int(v)


def _wrap(fn, *args):
    try:
        return fn(*args)
    except _traits.CatalogError as exc:
        raise ReproductionError(str(exc)) from exc


def check_limits(max_new_plants, max_total_records, max_attempts):
    '''Strict finite integer limits (no float truncation, no bool, no negative).'''
    return (_int(max_new_plants, 'max_new_plants', 0), _int(max_total_records, 'max_total_records', 0),
            _int(max_attempts, 'max_attempts', 0))


# ------------------------------------------------------------------ keyed randomness
def stream_key(seed, counter, stream):
    '''64-bit key of (seed, reproduction counter, stream) as a Python int; streams are separate from demography.'''
    try:
        seed, counter = _demo.check_seed(seed), _demo.check_counter(counter)
    except _demo.DemographyError as exc:
        raise ReproductionError(str(exc)) from exc
    if stream not in STREAMS:
        raise ReproductionError('unknown reproduction RNG stream')
    k0 = _demo._mix_int((seed ^ ((_demo._CSTREAM * (stream + 1)) & _M)) & _M)
    return _demo._mix_int((k0 + ((counter * _demo._GOLD) & _M) + 1) & _M)


def keyed_uniform_py(seed, counter, stream, a, b=0):
    '''Pure-Python reference of one uniform u in [0, 1) at integer keys (a, b) (tests / documentation).'''
    k = stream_key(seed, counter, stream)
    h = _demo._mix_int((k + ((int(a) & _M) * _demo._GOLD)) & _M)
    h = _demo._mix_int((h + ((int(b) & _M) * _demo._GOLD) + 1) & _M)
    return float(h >> 11) * _INV53


def _u(key, a_u64, b_u64):
    '''Vectorised uniforms for uint64 key arrays a, b (identical to keyed_uniform_py element-wise).'''
    with np.errstate(over='ignore'):
        z = a_u64 * _U_GOLD + np.uint64(key)
        z = z + _U_GOLD
        z = (z ^ (z >> _U_S30)) * _U_C1
        z = (z ^ (z >> _U_S27)) * _U_C2
        z = z ^ (z >> _U_S31)
        z = z + b_u64 * _U_GOLD + _U_ONE
        z = z + _U_GOLD
        z = (z ^ (z >> _U_S30)) * _U_C1
        z = (z ^ (z >> _U_S27)) * _U_C2
        z = z ^ (z >> _U_S31)
    return (z >> _U_S11).astype(np.float64) * _INV53


def _count_from_u(u, mx):
    '''Uniform integer 1..mx from u in [0, 1).'''
    return np.minimum(1 + np.floor(u * mx).astype(np.int64), mx)


# ------------------------------------------------------------------ catalog law parsing
def _exact(d, keys, where):
    if not isinstance(d, dict):
        raise ReproductionError(f'{where} must be null or an object')
    missing = [k for k in keys if k not in d]
    unknown = sorted(set(d) - set(keys))
    if missing or unknown:
        raise ReproductionError(f'{where} must hold exactly {list(keys)}; missing {missing}, unknown {unknown}')


def _parse_species(sec, sid):
    where = f'species {sid!r} reproduction'
    f = dict.fromkeys(('seed_p', 'seed_agb', 'seed_efrac', 'cl_min_age', 'cl_min_b', 'cl_p', 'cl_agb', 'cl_maxdon',
                       'cl_radius'), 0.0)
    i = dict.fromkeys(('seed_on', 'seed_max', 'cl_on', 'cl_maxd', 'cl_attempts'), 0)
    if sec is None or (isinstance(sec, dict) and sec.get('law') == _traits.UNIMPLEMENTED):
        return f, i
    if not isinstance(sec, dict):
        raise ReproductionError(f'{where} must be null or an object')
    law = sec.get('law')
    if law != LAW:
        raise ReproductionError(f'{where}.law {law!r} is unknown to leaf.recruitment (supported: {LAW!r}, or '
                                f'"{_traits.UNIMPLEMENTED}" for an explicitly inactive section)')
    unknown = sorted(set(sec) - set(_SECTION_KEYS))
    if unknown:
        raise ReproductionError(f'{where}: unknown keys {unknown} (allowed: {list(_SECTION_KEYS)})')
    tr = sec.get('traits')
    _exact(tr, ('seed', 'clonal'), f'{where}.traits')
    sd = tr['seed']
    if sd is not None:
        w = f'{where}.traits.seed'
        _exact(sd, _SEED_KEYS, w)
        f['seed_p'] = _wrap(_traits.unit_interval, sd['establishment_probability'], f'{w}.establishment_probability')
        i['seed_max'] = _wrap(_traits.strict_int, sd['max_recruits_per_patch'], f'{w}.max_recruits_per_patch', 1)
        if i['seed_max'] > MAX_COUNT:
            raise ReproductionError(f'{w}.max_recruits_per_patch must be <= {MAX_COUNT}')
        f['seed_agb'] = _wrap(_traits.positive, sd['initial_agb_dry_kg'], f'{w}.initial_agb_dry_kg')
        f['seed_efrac'] = _wrap(_traits.unit_interval, sd['initial_seasonal_fraction'], f'{w}.initial_seasonal_fraction')
        i['seed_on'] = 1
    cl = tr['clonal']
    if cl is not None:
        w = f'{where}.traits.clonal'
        _exact(cl, _CLONAL_KEYS, w)
        f['cl_min_age'] = _wrap(_traits.nonnegative, cl['min_age_days'], f'{w}.min_age_days')
        f['cl_min_b'] = _wrap(_traits.positive, cl['min_agb_dry_kg'], f'{w}.min_agb_dry_kg')
        f['cl_p'] = _wrap(_traits.unit_interval, cl['probability'], f'{w}.probability')
        i['cl_maxd'] = _wrap(_traits.strict_int, cl['max_daughters_per_parent'], f'{w}.max_daughters_per_parent', 1)
        i['cl_attempts'] = _wrap(_traits.strict_int, cl['max_attempts_per_daughter'], f'{w}.max_attempts_per_daughter', 1)
        if i['cl_maxd'] > MAX_COUNT or i['cl_attempts'] > MAX_COUNT:
            raise ReproductionError(f'{w}: counts must be <= {MAX_COUNT}')
        f['cl_agb'] = _wrap(_traits.positive, cl['initial_agb_dry_kg'], f'{w}.initial_agb_dry_kg')
        md = _wrap(_traits.real, cl['max_donation_fraction'], f'{w}.max_donation_fraction')
        if not 0.0 < md < 1.0:
            raise ReproductionError(f'{w}.max_donation_fraction must lie in (0, 1) (the parent stays live with positive '
                                    f'dry AGB), got {md}')
        f['cl_maxdon'] = md
        f['cl_radius'] = _wrap(_traits.positive, cl['dispersal_radius_m'], f'{w}.dispersal_radius_m')
        i['cl_on'] = 1
    return f, i


class CompiledReproduction:
    '''Immutable S-sized species table (sorted catalog ids). Inactive catalogs hold NO array (zero resident bytes).'''

    def __init__(self, species_ids, catalog_hash, f, i):
        self.species_ids = tuple(species_ids)
        self.catalog_hash = catalog_hash
        self._f, self._i = f, i
        if f is None:
            self.any_active = self.any_seed = self.any_clonal = False
        else:
            self.any_seed = bool(np.any(i['seed_on'] == 1))
            self.any_clonal = bool(np.any(i['cl_on'] == 1))
            self.any_active = self.any_seed or self.any_clonal

    @property
    def n_species(self):
        return len(self.species_ids)

    def resident_bytes(self):
        return BYTES_PER_SPECIES_TABLE * self.n_species if self.any_active else 0

    def active_species(self):
        if not self.any_active:
            return []
        return [s for k, s in enumerate(self.species_ids) if self._i['seed_on'][k] or self._i['cl_on'][k]]

    def describe(self):
        return {'law': LAW, 'version': VERSION, 'n_species': self.n_species, 'any_seed': self.any_seed,
                'any_clonal': self.any_clonal, 'active_species': self.active_species(),
                'seed_species': [] if not self.any_seed else [s for k, s in enumerate(self.species_ids)
                                                              if self._i['seed_on'][k]],
                'clonal_species': [] if not self.any_clonal else [s for k, s in enumerate(self.species_ids)
                                                                  if self._i['cl_on'][k]],
                'catalog_hash': self.catalog_hash, 'resident_bytes': self.resident_bytes()}

    def coefficients(self, species_id):
        if species_id not in self.species_ids or not self.any_active:
            raise ReproductionError(f'no active reproduction coefficients for {species_id!r}')
        k = self.species_ids.index(species_id)
        if not (self._i['seed_on'][k] or self._i['cl_on'][k]):
            raise ReproductionError(f'species {species_id!r} has no active seed or clonal pathway')
        out = {n: float(v[k]) for n, v in self._f.items()}
        out.update({n: int(v[k]) for n, v in self._i.items()})
        return out


def compile_species(catalog, *, max_bytes=None, extra_reserve_bytes=0):
    '''Parse EVERY species' `reproduction` section (unknown / malformed active laws refuse naming the species; null /
    "unimplemented" is inactive). `max_bytes` admits the S-sized table BEFORE it is built.'''
    if not isinstance(catalog, _traits.SpeciesCatalog):
        raise ReproductionError('catalog must be a leaf.traits.SpeciesCatalog')
    ids = catalog.species_ids
    S = len(ids)
    if max_bytes is not None:
        mb, rb = _int(max_bytes, 'max_bytes', 1), _int(extra_reserve_bytes, 'extra_reserve_bytes', 0)
        need = BYTES_PER_SPECIES_TABLE * S + rb
        if need > mb:
            raise ReproductionBudgetError(f'compile_species needs {need} bytes > max_bytes={mb}; refused before allocation')
    errors, rows_f, rows_i = [], [], []
    for sid in ids:
        try:
            f, i = _parse_species(catalog.section(sid, 'reproduction'), sid)
        except (ReproductionError, _traits.CatalogError) as exc:
            errors.append(str(exc))
            continue
        rows_f.append(f)
        rows_i.append(i)
    if errors:
        more = f' (+{len(errors) - 6} more)' if len(errors) > 6 else ''
        raise ReproductionError('species catalog is not usable by leaf.recruitment: ' + ' | '.join(errors[:6]) + more)
    if not any(r['seed_on'] or r['cl_on'] for r in rows_i):
        return CompiledReproduction(ids, catalog.hash, None, None)
    f = {n: np.array([r[n] for r in rows_f], dtype=np.float64) for n in rows_f[0]}
    i = {n: np.array([r[n] for r in rows_i], dtype=np.int64) for n in rows_i[0]}
    for a in list(f.values()) + list(i.values()):
        a.setflags(write=False)
    return CompiledReproduction(ids, catalog.hash, f, i)


# ------------------------------------------------------------------ explicit ecological patch configuration
class PatchConfig:
    '''Fixed physical ecological patch: a square of `patch_size_m` metres, tiling the domain from its south-west origin
    (boundary patches are clipped). Independent of the hydraulic cell resolution (nothing here reads dx / dy except
    through the domain extent).'''

    __slots__ = ('patch_size_m',)

    def __init__(self, patch_size_m):
        if isinstance(patch_size_m, (bool, np.bool_)) or not _traits.is_real(patch_size_m) \
                or not math.isfinite(float(patch_size_m)) or float(patch_size_m) <= 0.0:
            raise ReproductionError(f'patch_size_m must be a finite real number > 0 (metres), got {patch_size_m!r}')
        object.__setattr__(self, 'patch_size_m', float(patch_size_m))

    def __setattr__(self, name, value):
        raise AttributeError('PatchConfig is immutable')

    def shape(self, domain):
        '''(n_patches_x, n_patches_y) as Python ints (exact, no allocation).'''
        try:
            nxp = max(1, int(math.ceil(domain.width_m / self.patch_size_m)))
            nyp = max(1, int(math.ceil(domain.height_m / self.patch_size_m)))
        except (OverflowError, ValueError) as exc:
            raise ReproductionError(f'patch_size_m={self.patch_size_m} gives an unrepresentable patch count for this '
                                    'domain; refused before any array exists') from exc
        return nxp, nyp

    def n_patches(self, domain):
        nxp, nyp = self.shape(domain)
        n = nxp * nyp                        # exact Python integer
        if n >= (1 << 62):
            raise ReproductionError(f'{n} patches exceed the supported index range; increase patch_size_m')
        return n

    def to_dict(self):
        return {'patch_size_m': self.patch_size_m, 'geometry': 'square, SW-origin tiling, boundary patches clipped',
                'trial': 'one establishment trial per patch, species and annual opportunity'}

    def __eq__(self, other):
        return isinstance(other, PatchConfig) and other.patch_size_m == self.patch_size_m

    def __hash__(self):
        return hash(self.patch_size_m)


def patch_geometry(patch, domain, pid):
    '''x0, x1, y0, y1 (float64) of patch ids `pid` (int64, row-major: pid = iy * nxp + ix), clipped to the closed domain.'''
    nxp, _ = patch.shape(domain)
    ix, iy = pid % nxp, pid // nxp
    ps = patch.patch_size_m
    x0 = np.minimum(domain.origin_x_m + ix * ps, domain.x_max_m)
    x1 = np.minimum(domain.origin_x_m + (ix + 1) * ps, domain.x_max_m)
    y0 = np.minimum(domain.origin_y_m + iy * ps, domain.y_max_m)
    y1 = np.minimum(domain.origin_y_m + (iy + 1) * ps, domain.y_max_m)
    return x0, x1, y0, y1


# ------------------------------------------------------------------ memory
def memory_estimate(n_trials, n_live, n_candidates, n_births, n_cells):
    '''Conservative host bytes of one `evaluate` call (itemised): patch-trial arrays, live-sized arrays, daughter
    candidate arrays, birth / recruit arrays and the validated surface copy. All integer arguments are exact counts.'''
    items = {'patch_trials': BYTES_PER_TRIAL * int(n_trials), 'live_arrays': BYTES_PER_LIVE * int(n_live),
             'daughter_candidates': BYTES_PER_CANDIDATE * int(n_candidates), 'births': BYTES_PER_BIRTH * int(n_births),
             'surface_copy': 8 * int(n_cells)}
    items['total_bytes'] = int(sum(items.values()))
    return items


def preflight_bytes(table, patch, domain, n_live):
    '''Metadata-only bytes of the TRIAL stage (before any array exists): patch trials x seed species + live arrays.'''
    n_trials = 0
    if table.any_seed:
        if patch is None:
            raise ReproductionError('a seed law is active: an explicit PatchConfig (patch_size_m) is required')
        n_trials = patch.n_patches(domain) * int(np.count_nonzero(table._i['seed_on']))
    return memory_estimate(n_trials, n_live, 0, 0, domain.ncells), n_trials


# ------------------------------------------------------------------ evaluation
def _check_arrays(table, domain, ids, species, age, B, E, x, y):
    n = None
    for nm, a, kind in (('live_ids', ids, 'i'), ('live_species', species, 'iu'), ('live_age_days', age, 'f'),
                        ('live_B', B, 'f'), ('live_E', E, 'f'), ('live_x', x, 'f'), ('live_y', y, 'f')):
        if not isinstance(a, np.ndarray) or a.ndim != 1 or a.dtype.kind not in kind or (kind == 'f' and a.dtype != np.float64):
            raise ReproductionError(f'{nm} must be a 1-D NumPy array (int64 ids, integer species, float64 others)')
        if n is None:
            n = int(a.shape[0])
        elif a.shape[0] != n:
            raise ReproductionError('live arrays must have equal length')
    if ids.dtype != np.int64:
        raise ReproductionError('live_ids must be int64 (canonical IDs of any sign)')
    if n:
        for nm, a in (('live_age_days', age), ('live_B', B), ('live_E', E), ('live_x', x), ('live_y', y)):
            if not bool(np.all(np.isfinite(a))):
                raise ReproductionError(f'{nm} must be finite')
        if int(species.min()) < 0 or int(species.max()) >= table.n_species:
            raise ReproductionError(f'live_species must lie in [0, {table.n_species})')
        if float(B.min()) <= 0.0 or float(E.min()) < 0.0 or bool(np.any(E > B)) or float(age.min()) < 0.0:
            raise ReproductionError('live plants need B > 0, 0 <= E <= B and age >= 0')
        if not bool(np.all(domain.contains(x, y))):
            raise ReproductionError('live positions must lie inside the closed domain')
    return n


def evaluate(table, patch, domain, current_surface, *, live_ids, live_species, live_age_days, live_B, live_E, live_x,
             live_y, max_existing_id, n_total_records, seed, counter, max_new_plants, max_total_records, max_attempts,
             max_bytes, extra_reserve_bytes=0):
    '''One annual reproduction opportunity (module docstring). PURE: nothing is mutated and no random state exists
    beyond (seed, counter). `live_*` are the LIVE plants (any row order; results are by ID). `max_existing_id` is the
    maximum ID of ALL records (None for an empty population) and `n_total_records` their count (dead / removed
    included). Returns {'births': {...SoA...}, 'parent_rows': int64 indices into the live arrays of changed parents,
    'parent_B_new', 'parent_E_new', 'diagnostics': {...}}.'''
    if not isinstance(table, CompiledReproduction):
        raise ReproductionError('table must come from leaf.recruitment.compile_species')
    if not table.any_active:
        raise ReproductionError('the catalog has no active reproduction law')
    if not isinstance(domain, _state.Domain):
        raise ReproductionError('domain must be a leaf.state.Domain')
    mb = _int(max_bytes, 'max_bytes', 1)
    rb = _int(extra_reserve_bytes, 'extra_reserve_bytes', 0)
    max_new, max_total, max_att = check_limits(max_new_plants, max_total_records, max_attempts)
    n_total = _int(n_total_records, 'n_total_records', 0)
    if max_existing_id is not None:
        max_existing_id = _int(max_existing_id, 'max_existing_id', -(1 << 63), INT64_MAX)
    elif n_total:
        raise ReproductionError('max_existing_id is required for a non-empty population')
    try:
        seed, counter = _demo.check_seed(seed), _demo.check_counter(counter)
    except _demo.DemographyError as exc:
        raise ReproductionError(str(exc)) from exc
    # typed metadata only (an arbitrary list-like must never be materialised before admission)
    if not isinstance(live_ids, np.ndarray) or live_ids.ndim != 1:
        raise ReproductionError('live_ids must be a 1-D NumPy array (typed canonical int64 IDs)')
    n_live = int(live_ids.shape[0])
    if patch is not None and not isinstance(patch, PatchConfig):
        raise ReproductionError(f'patch must be a leaf.recruitment.PatchConfig (or None without a seed law), got {type(patch).__name__}')
    if n_total < n_live:
        raise ReproductionError(f'n_total_records={n_total} is smaller than the {n_live} live records provided')
    if n_total == 0 and max_existing_id is not None:
        raise ReproductionError('max_existing_id must be None for an empty population (n_total_records == 0)')
    est1, n_trials = preflight_bytes(table, patch, domain, n_live)
    if est1['total_bytes'] + rb > mb:
        raise ReproductionBudgetError(f'reproduction trials need ~{est1["total_bytes"] + rb} bytes (patch trials '
                                      f'{n_trials}, live {n_live}) > max_bytes={mb}; refused before any array exists')
    n_live = _check_arrays(table, domain, live_ids, live_species, live_age_days, live_B, live_E, live_x, live_y)
    if n_live:
        if int(np.unique(live_ids).shape[0]) != n_live:
            raise ReproductionError('live_ids must be unique canonical int64 IDs (any sign, zero included)')
        if max_existing_id is None or int(live_ids.max()) > int(max_existing_id):
            raise ReproductionError('max_existing_id is inconsistent with the live IDs (it must be >= every existing ID); '
                                    'refused to avoid ID collisions')
    try:
        surface = domain.check_field(current_surface, 'current surface')
    except _state.StateError as exc:
        raise ReproductionError(f'current surface refused: {exc}') from exc
    S, f, i = table.n_species, table._f, table._i
    u64 = np.uint64
    diag = {'seed': {}, 'clonal': {}}
    # ------------------------------------------------ seed trials (per patch and species; sorted species, row-major patch)
    n_seed = 0
    seed_trial = None
    if table.any_seed:
        seed_sp = np.flatnonzero(i['seed_on'] == 1).astype(np.int64)
        nP = patch.n_patches(domain)
        sp_t = np.repeat(seed_sp, nP)
        pid_t = np.tile(np.arange(nP, dtype=np.int64), seed_sp.shape[0])
        x0, x1, y0, y1 = patch_geometry(patch, domain, pid_t)
        area = (x1 - x0) * (y1 - y0)
        usable = area > 0.0
        u = _u(stream_key(seed, counter, STREAM_SEED_TRIAL), pid_t.astype(u64), sp_t.astype(u64))
        ok = usable & (u < f['seed_p'][sp_t])
        sel = np.flatnonzero(ok)
        mx = i['seed_max'][sp_t[sel]]
        uc = _u(stream_key(seed, counter, STREAM_SEED_COUNT), pid_t[sel].astype(u64), sp_t[sel].astype(u64))
        n_rec = _count_from_u(uc, mx)
        n_seed = int(n_rec.sum())
        seed_trial = (sel, n_rec, sp_t, pid_t, x0, x1, y0, y1)
        d_tr = np.bincount(sp_t[usable], minlength=S)
        d_ok = np.bincount(sp_t[sel], minlength=S)
        d_rec = np.bincount(sp_t[sel], weights=n_rec, minlength=S).astype(np.int64)
        diag['seed'] = {'trials': d_tr.tolist(), 'successes': d_ok.tolist(), 'failures': (d_tr - d_ok).tolist(),
                        'recruits': d_rec.tolist()}
    # ------------------------------------------------ clonal trials (live parents; keyed by ID)
    n_req = 0
    order = np.zeros(0, dtype=np.int64)
    counts = np.zeros(0, dtype=np.int64)
    sp_l = np.asarray(live_species, dtype=np.int64)
    if table.any_clonal and n_live:
        on = i['cl_on'][sp_l] == 1
        agb_l = f['cl_agb'][sp_l]
        elig = on & (live_age_days >= f['cl_min_age'][sp_l]) & (live_B >= f['cl_min_b'][sp_l]) \
            & (f['cl_maxdon'][sp_l] * live_B >= agb_l)
        ids_u = live_ids.astype(u64)
        zero = np.zeros(n_live, dtype=np.uint64)
        ut = _u(stream_key(seed, counter, STREAM_CLONAL_TRIAL), ids_u, zero)
        succ = elig & (ut < f['cl_p'][sp_l])
        uc = _u(stream_key(seed, counter, STREAM_CLONAL_COUNT), ids_u, zero)
        mxd = i['cl_maxd'][sp_l]
        draw = _count_from_u(uc, np.maximum(mxd, 1))
        cap = np.floor(np.minimum(f['cl_maxdon'][sp_l] * live_B / np.where(agb_l > 0, agb_l, 1.0), mxd.astype(np.float64)))
        nreq = np.where(succ, np.minimum(draw, cap.astype(np.int64)), 0)
        sel = np.flatnonzero(nreq > 0)
        order = sel[np.argsort(live_ids[sel], kind='stable')].astype(np.int64)     # stable ID order, row-order free
        counts = nreq[order]
        n_req = int(counts.sum())
        diag['clonal'] = {'eligible_parents': np.bincount(sp_l[elig], minlength=S).tolist(),
                          'trial_successes': np.bincount(sp_l[succ], minlength=S).tolist(),
                          'trial_failures': (np.bincount(sp_l[elig], minlength=S)
                                             - np.bincount(sp_l[succ], minlength=S)).tolist(),
                          'daughters_requested': np.bincount(sp_l[order], weights=counts, minlength=S).astype(np.int64).tolist()}
    # ------------------------------------------------ explicit work limits (nothing truncated)
    if n_seed + n_req > max_new:
        raise ReproductionLimitError(f'requested births {n_seed + n_req} (seed {n_seed}, daughters {n_req}) exceed '
                                     f'max_new_plants={max_new}; nothing is truncated')
    if n_total + n_seed + n_req > max_total:
        raise ReproductionLimitError(f'total records would reach {n_total + n_seed + n_req} > max_total_records={max_total}; '
                                     'dead records are never deleted to fit')
    # safe work count: float accumulation (no int64 wrap); anything near the int64 range is refused as a limit violation
    att_f = float(np.sum(counts.astype(np.float64) * i['cl_attempts'][sp_l[order]].astype(np.float64))) if n_req else 0.0
    if att_f >= 2.0 ** 62:
        raise ReproductionLimitError('placement attempt count is out of the supported integer range; refused')
    attempts_clonal_max = int(att_f)
    if n_seed + attempts_clonal_max > max_att:
        raise ReproductionLimitError(f'placement attempts up to {n_seed + attempts_clonal_max} exceed max_attempts={max_att}')
    est2 = memory_estimate(n_trials, n_live, n_req, n_seed + n_req, domain.ncells)
    if est2['total_bytes'] + rb > mb:
        raise ReproductionBudgetError(f'reproduction needs ~{est2["total_bytes"] + rb} bytes (births {n_seed + n_req}) > '
                                      f'max_bytes={mb}; refused before the candidate arrays exist')
    # ------------------------------------------------ daughter placement: bounded rejection, never clipped / wrapped
    placed = np.zeros(n_req, dtype=bool)
    cx = np.zeros(n_req)
    cy = np.zeros(n_req)
    n_used = 0
    par = np.repeat(np.arange(order.shape[0], dtype=np.int64), counts)
    starts = np.cumsum(counts) - counts
    didx = np.arange(n_req, dtype=np.int64) - np.repeat(starts, counts)
    prow = order[par]
    if n_req:
        a_ids = live_ids[prow].astype(u64)
        radius = f['cl_radius'][sp_l[prow]]
        att = i['cl_attempts'][sp_l[prow]]
        px, py = live_x[prow], live_y[prow]
        key_r, key_t = stream_key(seed, counter, STREAM_CLONAL_R), stream_key(seed, counter, STREAM_CLONAL_THETA)
        for r in range(int(att.max())):                 # bounded rounds (configured attempts), vectorised over candidates
            idx = np.flatnonzero((~placed) & (att > r))
            if idx.size == 0:
                break
            bb = (didx[idx].astype(u64) << _U_S32) | u64(r)
            rr = radius[idx] * np.sqrt(_u(key_r, a_ids[idx], bb))
            th = (2.0 * math.pi) * _u(key_t, a_ids[idx], bb)
            x, y = px[idx] + rr * np.cos(th), py[idx] + rr * np.sin(th)
            inside = domain.contains(x, y)
            hit = idx[inside]
            cx[hit], cy[hit] = x[inside], y[inside]
            placed[hit] = True
            n_used += int(idx.size)
    acc = np.flatnonzero(placed)
    n_cl = int(acc.shape[0])
    n_new = n_seed + n_cl
    if n_new:
        if max_existing_id is None:
            next_id = 1
        else:
            next_id = max(int(max_existing_id), 0) + 1
        if next_id + n_new - 1 > INT64_MAX:
            raise ReproductionLimitError(f'int64 ID space exhausted: {n_new} new IDs needed above {next_id - 1}; refused '
                                         'before any birth array exists')
    # ------------------------------------------------ parent donation (accepted daughters only)
    k_par = np.bincount(par[acc], minlength=order.shape[0]) if n_req else np.zeros(0, dtype=np.int64)
    B_new_par = np.zeros(0)
    E_new_par = np.zeros(0)
    donor_total = 0.0
    parent_rows = np.zeros(0, dtype=np.int64)
    if n_cl:
        donors = np.flatnonzero(k_par > 0)
        parent_rows = order[donors]
        kd = k_par[donors].astype(np.float64)
        agb = f['cl_agb'][sp_l[parent_rows]]
        Bp, Ep = live_B[parent_rows], live_E[parent_rows]
        dB = kd * agb
        B_new_par = Bp - dB
        e_each = np.minimum(agb * (Ep / Bp), agb)               # transferred E per daughter, <= its B
        E_new_par = np.minimum(np.maximum(Ep - kd * e_each, 0.0), B_new_par)
        if not (bool(np.all(B_new_par > 0.0)) and bool(np.all(np.isfinite(B_new_par)))):
            raise ReproductionError('a donating parent would not keep positive dry AGB; refused (nothing donated)')
        donor_total = float(dB.sum())
    # ------------------------------------------------ births (seeds first, then daughters in parent-ID order)
    births = None
    if n_new:
        ids_new = np.arange(next_id, next_id + n_new, dtype=np.int64)
        spx = np.empty(n_new, dtype=np.int32)
        bx, by = np.empty(n_new), np.empty(n_new)
        bB, bE = np.empty(n_new), np.empty(n_new)
        pid_par = np.full(n_new, -1, dtype=np.int64)
        is_cl = np.zeros(n_new, dtype=bool)
        if n_seed:
            sel, n_rec, sp_t, pid_t, x0, x1, y0, y1 = seed_trial
            tr = np.repeat(sel, n_rec)
            st_ = np.cumsum(n_rec) - n_rec
            j = np.arange(n_seed, dtype=np.int64) - np.repeat(st_, n_rec)
            a_ = pid_t[tr].astype(u64)
            b_ = (sp_t[tr].astype(u64) << _U_S32) | j.astype(u64)
            ux = _u(stream_key(seed, counter, STREAM_SEED_X), a_, b_)
            uy = _u(stream_key(seed, counter, STREAM_SEED_Y), a_, b_)
            bx[:n_seed] = np.minimum(x0[tr] + ux * (x1[tr] - x0[tr]), x1[tr])
            by[:n_seed] = np.minimum(y0[tr] + uy * (y1[tr] - y0[tr]), y1[tr])
            s_sp = sp_t[tr]
            spx[:n_seed] = s_sp
            bB[:n_seed] = f['seed_agb'][s_sp]
            bE[:n_seed] = np.minimum(f['seed_efrac'][s_sp] * bB[:n_seed], bB[:n_seed])
        if n_cl:
            c_sp = sp_l[prow[acc]]
            spx[n_seed:] = c_sp
            bx[n_seed:], by[n_seed:] = cx[acc], cy[acc]
            bB[n_seed:] = f['cl_agb'][c_sp]
            pr = prow[acc]
            bE[n_seed:] = np.minimum(bB[n_seed:] * (live_E[pr] / live_B[pr]), bB[n_seed:])
            pid_par[n_seed:] = live_ids[pr]
            is_cl[n_seed:] = True
        row, col = domain.cell_indices(bx, by)                  # containing cell (closed upper edge), all inside the domain
        ground = surface[row, col]
        births = {'ids': ids_new, 'species_index': spx, 'x_m': bx, 'y_m': by, 'establishment_ground_m': ground,
                  'agb_dry_kg': bB, 'seasonal_agb_dry_kg': bE, 'parent_id': pid_par, 'is_clonal': is_cl}
    # ------------------------------------------------ ledger and diagnostics
    n_log = math.log2(max(n_live, 1) + 2.0)
    old_total = float(np.sum(live_B, dtype=np.float64)) if n_live else 0.0
    seed_credit = float(np.sum(births['agb_dry_kg'][:n_seed])) if n_seed else 0.0
    dau_b = float(np.sum(births['agb_dry_kg'][n_seed:])) if n_cl else 0.0
    d_old = float(np.sum(live_B[parent_rows])) if n_cl else 0.0
    d_new = float(np.sum(B_new_par)) if n_cl else 0.0
    B_after = np.array(live_B, dtype=np.float64, copy=True)
    if n_cl:
        B_after[parent_rows] = B_new_par
    # retained total = (live plants after donation) + (all births); the donor transfer cancels internally
    new_total = (float(np.sum(B_after, dtype=np.float64)) if n_live else 0.0) + seed_credit + dau_b
    del B_after
    residual = (old_total + seed_credit) - new_total
    bound = (LEDGER_EPS_FACTOR + 4.0 * n_log) * FP64_EPS * max(old_total, new_total, 1e-300)
    transfer_residual = d_old - d_new - dau_b
    t_bound = (LEDGER_EPS_FACTOR + 4.0 * n_log) * FP64_EPS * max(d_old, 1e-300)
    if not (math.isfinite(residual) and abs(transfer_residual) <= t_bound):
        raise ReproductionError(f'dry AGB transfer does not close (residual {transfer_residual!r} kg, bound {t_bound!r})')
    e_old = float(np.sum(live_E[parent_rows])) if n_cl else 0.0
    e_new = float(np.sum(E_new_par)) if n_cl else 0.0
    e_dau = float(np.sum(births['seasonal_agb_dry_kg'][n_seed:])) if n_cl else 0.0
    if n_cl:
        diag['clonal'].update({
            'daughters_placed': np.bincount(sp_l[prow[acc]], minlength=S).tolist(),
            'daughters_rejected': (np.bincount(sp_l[prow], minlength=S) - np.bincount(sp_l[prow[acc]], minlength=S)).tolist(),
            'donor_agb_kg': np.bincount(sp_l[parent_rows], weights=dB, minlength=S).tolist()})
    elif diag['clonal']:
        diag['clonal'].update({'daughters_placed': [0] * S, 'daughters_rejected': diag['clonal']['daughters_requested'][:],
                               'donor_agb_kg': [0.0] * S})
    if n_seed:
        diag['seed']['seed_agb_kg'] = np.bincount(births['species_index'][:n_seed], weights=births['agb_dry_kg'][:n_seed],
                                                  minlength=S).tolist()
    elif diag['seed']:
        diag['seed']['seed_agb_kg'] = [0.0] * S
    diag.update({
        'species': list(table.species_ids), 'n_seed_recruits': n_seed, 'n_daughters_requested': n_req,
        'n_daughters_placed': n_cl, 'n_daughters_rejected': n_req - n_cl, 'n_births': n_new,
        'n_placement_attempts': int(n_used) + n_seed, 'n_patch_trials': int(n_trials), 'n_live_before': n_live,
        'n_donor_parents': int(parent_rows.shape[0]),
        'id_range': None if not n_new else [int(births['ids'][0]), int(births['ids'][-1])],
        'ledger': {'old_total_dry_agb_kg': old_total, 'external_seed_agb_kg': seed_credit,
                   'donor_agb_kg': donor_total, 'daughter_agb_kg': dau_b,
                   'new_total_dry_agb_kg': new_total, 'closure_residual_kg': residual, 'closure_bound_kg': bound,
                   'transfer_residual_kg': transfer_residual, 'transfer_bound_kg': t_bound,
                   'e_transfer_residual_kg': e_old - e_new - e_dau,
                   'definition': 'old total dry AGB + external seed AGB = new retained total dry AGB; the clonal '
                                 'transfer (parent loss = daughter gain) cancels internally',
                   'closed': bool(abs(residual) <= bound)},
        'caveats': list(CAVEATS), 'memory_estimate': est2})
    if not diag['ledger']['closed']:
        raise ReproductionError(f'dry AGB ledger does not close (residual {residual!r} kg, bound {bound!r} kg)')
    return {'births': births, 'parent_rows': parent_rows, 'parent_B_new': B_new_par, 'parent_E_new': E_new_par,
            'diagnostics': diag}


__all__ = ['VERSION', 'LAW', 'STREAMS', 'CAVEATS', 'ReproductionError', 'ReproductionBudgetError',
           'ReproductionLimitError', 'CompiledReproduction', 'PatchConfig', 'compile_species', 'check_limits',
           'stream_key', 'keyed_uniform_py', 'patch_geometry', 'memory_estimate', 'preflight_bytes', 'evaluate']
