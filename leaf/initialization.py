'''leaf.initialization: cover-driven multispecies population generator (bounded, reproducible).

Per species the configuration gives a target fractional cover f in [0, 1), a diameter mean m and standard
deviation s (lognormal with exactly that population mean/SD; s = 0 -> constant), and an initial age rule.
Biomass follows from each sampled diameter through the species inverse allometry (no age-biomass relation is
invented: ages are drawn independently of size).

Initial ages. DEFAULT (when `initial_age_days` is omitted) and explicit selector {"distribution":
"uniform_to_life_expectancy"}: each individual's age is drawn independently from a continuous UNIFORM distribution on
[0, L_sp) days, where L_sp = SpeciesCatalog.life_expectancy_days(species) is the nominal life expectancy of THAT species'
catalog record (life_expectancy.mean x 365 for unit "years", x 1 for "days"). The bound is read from the catalog, never
duplicated in the config or inferred from allometry or a mortality scale. It is an INITIAL SAMPLING range only: LEAF never
kills a plant because its age reaches the life expectancy (the nominal value is not a hard death age; mortality is the
separate probabilistic law of leaf.demography). A species without a usable life expectancy (null) makes the default /
selector a validation error (no guessed bound, no silent zero). Explicit rules remain for reproducible fixtures and
controlled experiments: a number of days, {"distribution": "constant", "value_days"} and {"distribution": "uniform",
"min_days", "max_days"} (these work with life_expectancy null). One vectorised `rng.uniform(lo, hi, k)` call draws the ages
of a species batch. Handwritten plant files, loaded snapshots/restarts and age-0 recruits are not affected.

Counts. With E[d^2] = m^2 + s^2 the expected crown area is E[A_c] = pi/4 * E[d^2]. Overlap allowed:
Poisson-Boolean start N0 = -A ln(1 - f) / E[A_c]; overlap disallowed: N0 = A f / E[A_c]. Counts are integers
and are then fitted to the MEASURED per-species union cover (leaf.canopy.measure_cover, midpoint raster of
resolution dx / cover_subsamples): N <- N * ln(1-f)/ln(1-c) (overlap) or N * f / c (no overlap), adding
newly sampled individuals or removing the most recently sampled ones, until |c - f| <= cover_tolerance
(a finite positive tolerance is mandatory) or max_iterations is exhausted (GenerationError; nothing is
returned). Edge clipping and same-species overlaps are therefore accounted for by measurement, not by
formula. Exact continuous cover or exact finite-sample diameter moments are not promised; all targets,
measurements, counts, sample statistics and the complete normalised configuration are reported.

Placement.
  overlap allowed     independent uniform centres in the domain, conditioned on the fitted count and the
                      cover acceptance (so NOT an unconditioned Poisson process); species in sorted-id
                      order, individuals in sample order.
  overlap disallowed  LARGE-CANOPY-FIRST exclusion placement: all marked sizes of the candidate population
                      are sampled first, sorted by descending diameter (stable: species order, then sample
                      order on ties) and placed sequentially with uniform random centre proposals rejected
                      on any canopy intersection with an already placed plant (roots may overlap). After a
                      count correction the COMPLETE candidate population is re-placed in size order (no
                      plant has an ID or an establishment yet, so nothing established ever moves). This is
                      a reproducible initialisation algorithm, not a Poisson process; random interleaving
                      is NOT used because small plants would occupy the gaps large plants need.
                      Neighbour queries use a sparse hashed bin index (bin >= largest diameter), never all
                      pairs. `max_attempts` bounds the total number of centre proposals over the whole
                      generation. Running out reports inability within these bounds (not infeasibility).

Periodic boundaries (`Domain.periodic`, per axis; default nonperiodic = everything above unchanged). Centres are sampled across
the FULL domain (no margin, no inset): a canopy near a periodic edge simply wraps across the seam and remains ONE plant with ONE
centre. Exclusion rejects overlap by the SHORTEST WRAPPED centre distance on each periodic axis (ordinary distance on the other
axes), with the same strict `< (ri + rj)^2` rule (tangency allowed). The spatial index uses exactly L / nb wide bins on a periodic
axis (nb = floor(L / bin), no short last bin) and deduplicated modular neighbours, so it is correct for any ratio of L to the bin
width and never compares a plant with its own image. A canopy whose diameter exceeds a periodic span would overlap itself: with
overlap disallowed the generation REFUSES (GenerationError, before any placement; nothing is rescaled, dropped or buffered). With
overlap allowed the independent uniform centres are kept and the cover is the correct torus union (`canopy.measure_cover`).
The random draws are identical to the nonperiodic case (same uniform x / y proposals); only the overlap test and the cover
measurement differ.

Bounds max_plants, max_attempts, max_cover_cells, max_cover_bytes, max_working_bytes and max_iterations are
explicit and checked before allocation/sampling; corrected counts are admitted before any new sample is
drawn; per-species sample storage is compacted on truncation (no dead trial slots are retained). RNG:
numpy.random.default_rng(seed) (PCG64); the draw order is recorded in the report together with the NumPy
version, so a run is reproducible from the report's `generation_config` + catalog hash.
'''
from __future__ import annotations

import copy
import math

import numpy as np

from leaf import canopy as _canopy
from leaf import periodic as _periodic
from leaf import state as _state
from leaf import traits as _traits

GENERATION_SCHEMA = 'leaf.generation_config'
GENERATION_SCHEMA_VERSION = 1
REPORT_SCHEMA = 'leaf.generation_report'
REPORT_SCHEMA_VERSION = 2
OVERLAP_MODES = ('allowed', 'disallowed')
DEFAULT_LIMITS = {'max_plants': 1_000_000, 'max_attempts': 20_000_000,
                  'max_cover_cells': _canopy.DEFAULT_MAX_COVER_CELLS, 'max_cover_bytes': _canopy.DEFAULT_MAX_COVER_BYTES,
                  'max_working_bytes': _canopy.DEFAULT_MAX_WORKING_BYTES}
DEFAULTS = {'overlap': 'allowed', 'cover_tolerance': 0.005, 'cover_subsamples': 4, 'max_iterations': 12,
            'time_days': 0.0, 'cover_definition': 'per_species_union'}
TOP_KEYS = ('schema', 'schema_version', 'domain', 'seed', 'overlap', 'cover_definition', 'cover_tolerance',
            'cover_subsamples', 'max_iterations', 'limits', 'species', 'time_days', 'notes')
SPECIES_KEYS = ('target_cover', 'diameter_mean_m', 'diameter_std_m', 'initial_age_days', 'notes')
REQUIRED_SPECIES_KEYS = SPECIES_KEYS[:3]       # initial_age_days is optional: omitted -> AGE_UNIFORM_TO_LIFE (the default)
AGE_UNIFORM_TO_LIFE = 'uniform_to_life_expectancy'
DEFAULT_AGE_SPEC = {'distribution': AGE_UNIFORM_TO_LIFE}
ORDERING = {'allowed': 'species in sorted catalog-id order, individuals in sample order (independent uniform centres)',
            'disallowed': 'large_canopy_first: all candidates sorted by descending diameter (stable tie-break: species '
                          'order, then sample order) and placed sequentially with global canopy exclusion'}
RNG_RECORD = {'constructor': 'numpy.random.default_rng(seed)', 'bit_generator': 'PCG64',
              'draw_order': {'allowed': 'per species batch: diameters (lognormal or none), x (uniform), y (uniform), '
                                        'ages (constant: none; uniform / uniform_to_life_expectancy: one vector '
                                        'rng.uniform(min, max, k) draw)',
                             'disallowed': 'per species batch: diameters, ages (same age draw as above); then per placement '
                                           'pass, per candidate in descending size order, per proposal: x (uniform), y (uniform)'}}
ORDERING_PERIODIC = {'allowed': ORDERING['allowed'] + '; periodic axes: centres over the full domain, cover = torus union',
                     'disallowed': ORDERING['disallowed'] + '; periodic axes: exclusion by shortest wrapped centre distance '
                                   '(minimum image per periodic axis, strict < (ri + rj)^2, tangency allowed)'}
PLACEMENT_PERIODIC = ('large-canopy-first sequential uniform proposals over the FULL domain (no inset) with global cross-species '
                      'canopy exclusion by shortest wrapped centre distance on periodic axes (roots may overlap); complete '
                      're-placement after each count correction; sparse hashed bins (exact width L / nb on periodic axes, '
                      'deduplicated modular neighbours), not all pairs; not a Poisson process')


def _ordering(cfg):
    return (ORDERING_PERIODIC if cfg['domain'].any_periodic else ORDERING)[cfg['overlap']]


class GenerationError(RuntimeError):
    '''Unattainable-within-bounds or unbounded request; raised with diagnostics, nothing is returned or written.'''


# ------------------------------------------------------------------ pure helpers
def lognormal_parameters(mean, std):
    '''(mu, sigma) of ln d such that d has the requested population mean and SD; None when std == 0.'''
    m, s = _traits.positive(mean, 'diameter_mean_m'), _traits.nonnegative(std, 'diameter_std_m')
    if s == 0.0:
        return None
    try:
        s2 = math.log(1.0 + (s / m) ** 2)
        mu, sigma = math.log(m) - 0.5 * s2, math.sqrt(s2)
    except (OverflowError, ValueError, ZeroDivisionError) as exc:   # Python float arithmetic raises on overflow
        raise _traits.CatalogError(f'lognormal parameters overflow for the requested diameter mean/std: {exc}') from exc
    if not (math.isfinite(mu) and math.isfinite(sigma)):
        raise _traits.CatalogError('lognormal parameters are not finite for the requested diameter mean/std')
    return mu, sigma


def expected_crown_area_m2(mean, std):
    '''pi/4 * E[d^2] with E[d^2] = mean^2 + std^2 (finite, positive); overflow raises CatalogError.'''
    try:
        ea = 0.25 * math.pi * (float(mean) ** 2 + float(std) ** 2)
    except OverflowError as exc:
        raise _traits.CatalogError(f'expected crown area overflows for the requested diameters: {exc}') from exc
    if not math.isfinite(ea) or ea <= 0.0:
        raise _traits.CatalogError('expected crown area is not finite/positive for the requested diameters')
    return ea


def initial_count(area_m2, target_cover, expected_area_m2, overlap):
    '''Integer starting count (>= 1 for a positive target); non-finite estimates are refused.'''
    if target_cover <= 0.0:
        return 0
    if overlap:
        n = -area_m2 * math.log(1.0 - target_cover) / expected_area_m2
    else:
        n = area_m2 * target_cover / expected_area_m2
    if not math.isfinite(n):
        raise GenerationError('initial count estimate is not finite (domain area / crown area / cover combination)')
    return max(1, int(round(n)))


def next_count(count, measured, target, overlap):
    '''Multiplicative count correction on the measured cover; always moves by at least one.'''
    if measured <= 0.0:
        raise GenerationError('measured cover is zero although plants were placed: the cover raster is too coarse '
                              'for these diameters; increase cover_subsamples')
    if measured >= 1.0:
        new = count // 2
    else:
        ratio = (math.log(1.0 - target) / math.log(1.0 - measured)) if overlap else (target / measured)
        if not math.isfinite(ratio):
            raise GenerationError('count correction ratio is not finite')
        new = int(round(count * ratio))
    if new == count:
        new = count + (1 if measured < target else -1)
    return max(new, 0)


def _age_spec(v, name):
    if _traits.is_real(v):
        return {'distribution': 'constant', 'value_days': _traits.nonnegative(v, name)}
    if not isinstance(v, dict) or 'distribution' not in v:
        raise _traits.CatalogError(f'{name} must be a number of days or an object with "distribution"')
    dist = v['distribution']
    if dist == AGE_UNIFORM_TO_LIFE:
        extra = sorted(set(v) - {'distribution'})
        if extra:
            raise _traits.CatalogError(f'{name}: {AGE_UNIFORM_TO_LIFE!r} takes no parameters (the bound is the species '
                                       f'catalog life_expectancy), got extra keys {extra}')
        return {'distribution': AGE_UNIFORM_TO_LIFE}
    if dist == 'constant':
        extra = sorted(set(v) - {'distribution', 'value_days'})
        if extra or 'value_days' not in v:
            raise _traits.CatalogError(f'{name}: constant distribution needs exactly value_days')
        return {'distribution': 'constant', 'value_days': _traits.nonnegative(v['value_days'], f'{name}.value_days')}
    if dist == 'uniform':
        extra = sorted(set(v) - {'distribution', 'min_days', 'max_days'})
        if extra or 'min_days' not in v or 'max_days' not in v:
            raise _traits.CatalogError(f'{name}: uniform distribution needs exactly min_days and max_days')
        lo = _traits.nonnegative(v['min_days'], f'{name}.min_days')
        hi = _traits.nonnegative(v['max_days'], f'{name}.max_days')
        if hi < lo:
            raise _traits.CatalogError(f'{name}: max_days < min_days')
        return {'distribution': 'uniform', 'min_days': lo, 'max_days': hi}
    raise _traits.CatalogError(f'{name}: unsupported age distribution {dist!r} '
                               f'({AGE_UNIFORM_TO_LIFE} | constant | uniform)')


def _resolve_age(age, catalog, sid, name, from_default):
    '''(sampling spec, resolved record) of one normalised age rule. The species-linked rule reads its upper bound from
    the catalog (SpeciesCatalog.life_expectancy_days: mean x 365 for "years") and refuses a species without one; explicit
    rules are passed through unchanged (they work with life_expectancy null).'''
    if age['distribution'] != AGE_UNIFORM_TO_LIFE:
        return dict(age), dict(age, rule='explicit', rule_source='explicit initial_age_days in the config')
    life = catalog.life_expectancy_days(sid)
    if life is None or not math.isfinite(life) or life <= 0.0:
        raise _traits.CatalogError(
            f'{name}: the age rule {AGE_UNIFORM_TO_LIFE!r}'
            + (' (the DEFAULT when initial_age_days is omitted)' if from_default else '')
            + f' needs a usable life_expectancy in the catalog record of species {sid!r}, but it is '
            f'{"null" if life is None else repr(life)}; no upper bound is guessed and zero is never used. Declare '
            'life_expectancy for the species, or give an explicit initial_age_days (a number of days, a constant or '
            'a bounded uniform distribution)')
    le = catalog.section(sid, 'life_expectancy')
    record = {'rule': AGE_UNIFORM_TO_LIFE,
              'rule_source': ('DEFAULT: initial_age_days omitted in the config' if from_default
                              else 'explicit selector in the config'),
              'distribution': 'uniform', 'min_days': 0.0, 'max_days': life, 'life_expectancy_days': life,
              'life_expectancy': {'mean': le['mean'], 'unit': le['unit']}, 'days_per_unit': _traits.TIME_UNIT_DAYS[le['unit']],
              'catalog_hash': catalog.hash, 'catalog_origin': f'species[{sid!r}].life_expectancy (mean x days per unit)',
              'note': 'initial sampling range only: the nominal life expectancy is not a hard death age (LEAF never kills a '
                      'plant for reaching it; mortality is the separate probabilistic law)'}
    return {'distribution': 'uniform', 'min_days': 0.0, 'max_days': life}, record


def validate_generation_config(config, catalog):
    '''PURE normalisation of a generation config dict against the catalog. Raises CatalogError.
    Species are processed in sorted id order (independent of JSON key order).'''
    E = _traits.CatalogError
    if not isinstance(catalog, _traits.SpeciesCatalog):
        raise E('catalog must be a leaf.traits.SpeciesCatalog')
    if not isinstance(config, dict):
        raise E('generation config must be a JSON object / dict')
    unknown = sorted(set(config) - set(TOP_KEYS))
    if unknown:
        raise E(f'generation config: unknown keys {unknown}')
    if config.get('schema', GENERATION_SCHEMA) != GENERATION_SCHEMA:
        raise E(f'generation config schema must be {GENERATION_SCHEMA!r}')
    ver = config.get('schema_version', GENERATION_SCHEMA_VERSION)
    if isinstance(ver, bool) or ver != GENERATION_SCHEMA_VERSION:
        raise E(f'unsupported generation config schema_version {ver!r}')
    if 'domain' not in config:
        raise E('generation config requires "domain"')
    try:
        domain = config['domain'] if isinstance(config['domain'], _state.Domain) else _state.Domain.from_dict(config['domain'])
    except _state.StateError as exc:
        raise E(f'generation config domain: {exc}') from exc
    seed = _traits.strict_int(config.get('seed'), 'seed', 0)
    overlap = config.get('overlap', DEFAULTS['overlap'])
    if overlap not in OVERLAP_MODES:
        raise E(f'overlap must be one of {OVERLAP_MODES}')
    if config.get('cover_definition', DEFAULTS['cover_definition']) != 'per_species_union':
        raise E("cover_definition must be 'per_species_union' (the only Phase 0 definition)")
    tol = config.get('cover_tolerance', DEFAULTS['cover_tolerance'])
    if tol is None or not _traits.is_real(tol):
        raise E('cover_tolerance must be a finite positive number < 1; target-cover acceptance cannot be disabled')
    tol = _traits.positive(tol, 'cover_tolerance')
    if tol >= 1.0:
        raise E('cover_tolerance must be < 1')
    sub = _traits.strict_int(config.get('cover_subsamples', DEFAULTS['cover_subsamples']), 'cover_subsamples', 1)
    it = _traits.strict_int(config.get('max_iterations', DEFAULTS['max_iterations']), 'max_iterations', 0)
    limits = dict(DEFAULT_LIMITS)
    user = config.get('limits', {})
    if not isinstance(user, dict) or sorted(set(user) - set(DEFAULT_LIMITS)):
        raise E(f'limits must be an object with keys from {sorted(DEFAULT_LIMITS)}')
    for k, v in user.items():
        limits[k] = _traits.strict_int(v, f'limits.{k}', 1)
    time_days = _traits.real(config.get('time_days', DEFAULTS['time_days']), 'time_days')
    sp = config.get('species')
    if not isinstance(sp, dict):
        raise E('generation config requires a "species" object (may be empty)')
    species = []
    for sid in sorted(sp):
        cfg = sp[sid]
        if not catalog.has(sid):
            raise E(f'generation config species {sid!r} is not in the catalog {list(catalog.species_ids)}')
        if not isinstance(cfg, dict):
            raise E(f'species[{sid!r}] must be an object')
        unknown = sorted(set(cfg) - set(SPECIES_KEYS))
        missing = [k for k in REQUIRED_SPECIES_KEYS if k not in cfg]
        if unknown or missing:
            raise E(f'species[{sid!r}]: unknown keys {unknown}, missing {missing}')
        age_name = f'species[{sid!r}].initial_age_days'
        from_default = 'initial_age_days' not in cfg
        age = _age_spec(dict(DEFAULT_AGE_SPEC) if from_default else cfg['initial_age_days'], age_name)
        age_sampling, age_resolved = _resolve_age(age, catalog, sid, age_name, from_default)
        f = _traits.nonnegative(cfg['target_cover'], f'species[{sid!r}].target_cover')
        if f >= 1.0:
            raise E(f'species[{sid!r}].target_cover must be < 1')
        m = _traits.positive(cfg['diameter_mean_m'], f'species[{sid!r}].diameter_mean_m')
        s = _traits.nonnegative(cfg['diameter_std_m'], f'species[{sid!r}].diameter_std_m')
        ea = expected_crown_area_m2(m, s)
        species.append({'id': sid, 'index': catalog.index(sid), 'target_cover': f, 'diameter_mean_m': m,
                        'diameter_std_m': s, 'initial_age_days': age, 'age_sampling': age_sampling,
                        'age_resolved': age_resolved, 'lognormal': lognormal_parameters(m, s), 'expected_crown_area_m2': ea,
                        'initial_count_estimate': initial_count(domain.area_m2, f, ea, overlap == 'allowed')})
    return {'schema': GENERATION_SCHEMA, 'schema_version': GENERATION_SCHEMA_VERSION, 'domain': domain, 'seed': seed,
            'overlap': overlap, 'cover_definition': 'per_species_union', 'cover_tolerance': tol, 'cover_subsamples': sub,
            'max_iterations': it, 'limits': limits, 'species': species, 'time_days': time_days,
            'notes': config.get('notes')}


def config_record(cfg):
    '''JSON-able complete SOURCE configuration: exactly what validate_generation_config accepts, so
    generate_population(catalog, report['generation_config'], surface=...) replays the run array-for-array
    (same catalog/surface). The species-linked age rule is recorded as its SOURCE RULE (the selector
    {"distribution": "uniform_to_life_expectancy"}, written explicitly even when the original config omitted
    initial_age_days) so the rule itself is never lost; its resolved bounds and catalog origin are in
    derived_record() / the report. Derived quantities live in derived_record().'''
    return {'schema': cfg['schema'], 'schema_version': cfg['schema_version'], 'domain': cfg['domain'].to_dict(),
            'seed': cfg['seed'], 'overlap': cfg['overlap'], 'cover_definition': cfg['cover_definition'],
            'cover_tolerance': cfg['cover_tolerance'], 'cover_subsamples': cfg['cover_subsamples'],
            'max_iterations': cfg['max_iterations'], 'limits': dict(cfg['limits']), 'time_days': cfg['time_days'],
            'species': {sp['id']: {'target_cover': sp['target_cover'], 'diameter_mean_m': sp['diameter_mean_m'],
                                   'diameter_std_m': sp['diameter_std_m'],
                                   'initial_age_days': dict(sp['initial_age_days'])} for sp in cfg['species']}}


def derived_record(cfg):
    '''JSON-able derived quantities of a normalised configuration (not accepted as config input).'''
    return {'ordering_convention': _ordering(cfg),
            'species': {sp['id']: {'diameter_distribution': ({'type': 'constant'} if sp['lognormal'] is None else
                                                             {'type': 'lognormal', 'mu': sp['lognormal'][0], 'sigma': sp['lognormal'][1],
                                                              'rule': 'sigma^2 = ln(1 + std^2/mean^2), mu = ln(mean) - sigma^2/2'}),
                                   'expected_crown_area_m2': sp['expected_crown_area_m2'],
                                   'initial_count_estimate': sp['initial_count_estimate'],
                                   'initial_age': copy.deepcopy(sp['age_resolved'])} for sp in cfg['species']}}


# ------------------------------------------------------------------ sampling helpers
def _sample_diameters(rng, spec, n):
    if n == 0:
        return np.zeros(0)
    if spec['lognormal'] is None:
        return np.full(n, spec['diameter_mean_m'], dtype=np.float64)
    mu, sigma = spec['lognormal']
    d = rng.lognormal(mu, sigma, n)
    if not np.all(np.isfinite(d)) or float(d.min()) <= 0.0:
        raise GenerationError('sampled diameters are not finite/positive')
    return d


def _sample_ages(rng, spec, n):
    '''Ages of n new individuals: constant, or ONE vectorised continuous uniform draw on [min, max) days (the resolved
    bounds: explicit, or 0 .. the species life expectancy for the species-linked rule).'''
    a = spec['age_sampling']
    if a['distribution'] == 'constant':
        return np.full(n, a['value_days'], dtype=np.float64)
    return rng.uniform(a['min_days'], a['max_days'], n) if n else np.zeros(0)


def _stats(d):
    if d.size == 0:
        return {'n': 0, 'mean': None, 'std': None, 'min': None, 'max': None}
    return {'n': int(d.size), 'mean': float(d.mean()), 'std': float(d.std()), 'min': float(d.min()), 'max': float(d.max())}


class _Samples:
    '''Per-species candidate samples (compact arrays, no dead slots).'''

    def __init__(self, overlap):
        self.overlap = overlap
        self.d = np.zeros(0)
        self.age = np.zeros(0)
        self.x = np.zeros(0)
        self.y = np.zeros(0)

    @property
    def n(self):
        return int(self.d.shape[0])

    def set_count(self, rng, domain, spec, n_new):
        '''Grow by sampling (draw order documented in RNG_RECORD) or truncate/compact to n_new.'''
        cur = self.n
        if n_new == cur:
            return
        if n_new < cur:
            self.d, self.age = self.d[:n_new].copy(), self.age[:n_new].copy()
            if self.overlap:
                self.x, self.y = self.x[:n_new].copy(), self.y[:n_new].copy()
            return
        k = n_new - cur
        d = _sample_diameters(rng, spec, k)
        if self.overlap:
            x = rng.uniform(domain.x_min_m, domain.x_max_m, k)
            y = rng.uniform(domain.y_min_m, domain.y_max_m, k)
        age = _sample_ages(rng, spec, k)
        self.d = np.concatenate([self.d, d])
        self.age = np.concatenate([self.age, age])
        if self.overlap:
            self.x = np.concatenate([self.x, x])
            self.y = np.concatenate([self.y, y])


def _place_large_first(rng, domain, d_all, max_attempts, attempts_used):
    '''Sequential exclusion placement of ALL candidates in descending size order (see module docstring).
    Returns (X, Y, order, attempts_used); raises GenerationError when max_attempts is exhausted.'''
    n = int(d_all.shape[0])
    X = np.empty(n, dtype=np.float64)
    Y = np.empty(n, dtype=np.float64)
    order = np.argsort(-d_all, kind='stable')
    if n == 0:
        return X, Y, order, attempts_used
    try:
        _periodic.check_no_self_overlap(domain, d_all, 'canopy')    # periodic axes only; before any proposal is drawn
    except _periodic.PeriodicGeometryError as exc:
        raise GenerationError(f'overlap disallowed: {exc}') from exc
    R = 0.5 * d_all
    size = max(float(d_all.max()), domain.dx_m, domain.dy_m)  # >= 2*r_max: conflicts only within adjacent bins
    x0, y0, x1, y1 = domain.x_min_m, domain.y_min_m, domain.x_max_m, domain.y_max_m
    lay = _periodic.bin_layout(domain, size)     # nonperiodic axes: the legacy bin edge; periodic: exact L / nb bins
    per_x, per_y = domain.periodic
    Lx, Ly = float(domain.width_m), float(domain.height_m)
    bins = {}
    uniform = rng.uniform
    placed = 0
    for i in order:
        i = int(i)
        r = float(R[i])
        while True:
            if attempts_used >= max_attempts:
                raise GenerationError(f'max_attempts {max_attempts} exhausted during large-canopy-first exclusion placement '
                                      f'({placed} of {n} candidates placed in this pass, next radius {r:.4g} m); the '
                                      'request is not attainable within the declared bounds of this algorithm (this does '
                                      'not prove that no packing exists); aborted without output')
            attempts_used += 1
            px = float(uniform(x0, x1))
            py = float(uniform(y0, y1))
            bx = _periodic.bin_key(px, x0, lay[0])
            by = _periodic.bin_key(py, y0, lay[1])
            cand = []
            for kx in _periodic.bin_neighbours(bx, lay[0]):
                for ky in _periodic.bin_neighbours(by, lay[1]):
                    lst = bins.get((kx, ky))
                    if lst:
                        cand.extend(lst)
            if cand:
                c = np.asarray(cand, dtype=np.int64)
                ddx = X[c] - px
                ddy = Y[c] - py
                if per_x:
                    ddx = _periodic.min_image(ddx, Lx)
                if per_y:
                    ddy = _periodic.min_image(ddy, Ly)
                rr = R[c] + r
                if bool(np.any(ddx * ddx + ddy * ddy < rr * rr)):
                    continue
            X[i], Y[i] = px, py
            bins.setdefault((bx, by), []).append(i)
            placed += 1
            break
    return X, Y, order, attempts_used


# ------------------------------------------------------------------ generator
def generate_population(catalog, config, *, surface=None):
    '''Generate a PlantState (ids 1..P in final placement order) and a report. See the module docstring.'''
    cfg = validate_generation_config(config, catalog)
    domain, limits, overlap = cfg['domain'], cfg['limits'], cfg['overlap'] == 'allowed'
    tol, sub, max_it = cfg['cover_tolerance'], cfg['cover_subsamples'], cfg['max_iterations']
    cover_kw = {'subsamples': sub, 'max_cover_cells': limits['max_cover_cells'], 'max_cover_bytes': limits['max_cover_bytes'],
                'max_working_bytes': limits['max_working_bytes']}
    try:
        admission = _canopy.cover_admission(domain, sub, total=True, max_cover_cells=limits['max_cover_cells'],
                                            max_cover_bytes=limits['max_cover_bytes'], max_working_bytes=limits['max_working_bytes'])
    except _canopy.ResourceBoundError as exc:
        raise GenerationError(f'cover measurement not admitted: {exc}; nothing allocated') from exc
    if surface is not None:
        surface = domain.check_field(surface, 'initial surface')
    rng = np.random.default_rng(cfg['seed'])
    A = domain.area_m2
    species = cfg['species']
    S = len(species)
    counts = [sp['initial_count_estimate'] for sp in species]
    if sum(counts) > limits['max_plants']:
        raise GenerationError(f'initial count estimate {sum(counts)} exceeds max_plants {limits["max_plants"]}; nothing sampled')
    samples = [_Samples(overlap) for _ in species]
    histories = [[] for _ in range(S)]
    attempts = 0
    passes = 0
    iterations_used = 0

    def admit_counts(new_counts):
        tot = sum(new_counts)
        if tot > limits['max_plants']:
            raise GenerationError(f'corrected population {tot} would exceed max_plants {limits["max_plants"]}; aborted '
                                  'before sampling, without output')

    if overlap:
        for si, spec in enumerate(species):
            smp = samples[si]
            smp.set_count(rng, domain, spec, counts[si])
            for it in range(max_it + 1):
                if smp.n:
                    c = _canopy.measure_cover(domain, smp.x, smp.y, 0.5 * smp.d, None, 1, total=False, **cover_kw)['groups'][0]['union_fraction']
                else:
                    c = 0.0
                histories[si].append({'count': smp.n, 'measured_cover': c})
                if abs(c - spec['target_cover']) <= tol:
                    break
                if it == max_it:
                    raise GenerationError(f'species {spec["id"]!r}: cover {c:.5f} not within {tol} of target '
                                          f'{spec["target_cover"]} after {max_it} count iterations; history {histories[si]}')
                iterations_used += 1
                new = next_count(smp.n, c, spec['target_cover'], True)
                probe = list(s.n for s in samples)
                probe[si] = new
                admit_counts(probe)
                smp.set_count(rng, domain, spec, new)
            counts[si] = smp.n
        order_x = np.concatenate([s.x for s in samples]) if S else np.zeros(0)
        order_y = np.concatenate([s.y for s in samples]) if S else np.zeros(0)
        d_all = np.concatenate([s.d for s in samples]) if S else np.zeros(0)
        age_all = np.concatenate([s.age for s in samples]) if S else np.zeros(0)
        sp_all = np.concatenate([np.full(s.n, si, dtype=np.int64) for si, s in enumerate(samples)]) if S else np.zeros(0, np.int64)
        X, Y = order_x, order_y
    else:
        for si, spec in enumerate(species):
            samples[si].set_count(rng, domain, spec, counts[si])
        for it in range(max_it + 1):
            d_all = np.concatenate([s.d for s in samples]) if S else np.zeros(0)
            age_all = np.concatenate([s.age for s in samples]) if S else np.zeros(0)
            sp_all = np.concatenate([np.full(s.n, si, dtype=np.int64) for si, s in enumerate(samples)]) if S else np.zeros(0, np.int64)
            Xp, Yp, order, attempts = _place_large_first(rng, domain, d_all, limits['max_attempts'], attempts)
            passes += 1
            if d_all.size:
                m = _canopy.measure_cover(domain, Xp, Yp, 0.5 * d_all, sp_all, max(S, 1), total=False, **cover_kw)
                cov = [m['groups'][si]['union_fraction'] for si in range(S)]
            else:
                cov = [0.0] * S
            pending = []
            for si, spec in enumerate(species):
                histories[si].append({'count': samples[si].n, 'measured_cover': cov[si], 'placement_pass': passes})
                if abs(cov[si] - spec['target_cover']) > tol:
                    pending.append(si)
            if not pending:
                break
            if it == max_it:
                bad = {species[si]['id']: histories[si] for si in pending}
                raise GenerationError(f'cover not within {tol} of target after {max_it} count iterations for {sorted(bad)}; '
                                      f'history {bad}; attempts used {attempts}')
            iterations_used += 1
            new_counts = list(s.n for s in samples)
            for si in pending:
                new_counts[si] = next_count(samples[si].n, cov[si], species[si]['target_cover'], False)
            admit_counts(new_counts)
            for si in range(S):
                samples[si].set_count(rng, domain, species[si], new_counts[si])
        # final arrays in placement order (descending size)
        X, Y = Xp[order], Yp[order]
        d_all, age_all, sp_all = d_all[order], age_all[order], sp_all[order]
        counts = [s.n for s in samples]
    # ---- final population arrays
    P = int(d_all.shape[0])
    r_all = 0.5 * d_all
    cat_idx = np.array([sp['index'] for sp in species], dtype=np.int64) if S else np.zeros(0, np.int64)
    species_index = cat_idx[sp_all] if P else np.zeros(0, np.int64)
    B = catalog.biomass_from_diameter(species_index, d_all) if P else np.zeros(0)
    if P and (not np.all(np.isfinite(B)) or float(B.min()) <= 0.0):
        raise GenerationError('inverse allometry produced non-finite/non-positive biomass')
    final = _canopy.measure_cover(domain, X, Y, r_all, sp_all, max(S, 1), total=True, **cover_kw)
    per_species = []
    for si, spec in enumerate(species):
        m = sp_all == si
        d = d_all[m]
        g = final['groups'][si]
        per_species.append({'species': spec['id'], 'target_cover': spec['target_cover'], 'measured_union_cover': g['union_fraction'],
                            'nominal_crown_area_fraction': g['nominal_fraction'], 'count': int(m.sum()),
                            'initial_count_estimate': spec['initial_count_estimate'],
                            'count_formula': ('N0 = -A ln(1-f) / E[A_c]' if overlap else 'N0 = A f / E[A_c]') + ', E[A_c] = pi/4 (mean^2 + std^2)',
                            'expected_crown_area_m2': spec['expected_crown_area_m2'],
                            'diameter_requested': {'mean_m': spec['diameter_mean_m'], 'std_m': spec['diameter_std_m'],
                                                   'distribution': 'constant' if spec['lognormal'] is None else 'lognormal'},
                            'diameter_sample_m': _stats(d), 'age_sample_days': _stats(age_all[m]),
                            'age_requested': copy.deepcopy(spec['age_resolved']),
                            'agb_dry_kg_total': float(B[m].sum()) if m.any() else 0.0,
                            'accepted': abs(g['union_fraction'] - spec['target_cover']) <= tol,
                            'iterations': len(histories[si]) - 1, 'history': histories[si],
                            'relative_error_scale': g['relative_error_scale']})
    if any(not p['accepted'] for p in per_species):
        raise GenerationError('final cover measurement outside tolerance (internal inconsistency); no output')
    report = {'schema': REPORT_SCHEMA, 'schema_version': REPORT_SCHEMA_VERSION, 'leaf_version': _traits.LEAF_VERSION,
              'catalog_hash': catalog.hash, 'catalog_species_table': list(catalog.species_ids),
              'generation_config': config_record(cfg),
              'generation_config_note': 'accepted SOURCE configuration: generate_population(catalog, generation_config, '
                                        'surface=...) with the same catalog/surface replays this run exactly (a species-linked age '
                                        'rule is recorded as its source rule and is re-resolved from that same catalog; the '
                                        'resolved bounds and their catalog origin are in derived.species.<id>.initial_age)',
              'initial_age_note': 'ages are drawn independently of size from each species\' resolved initial-age rule; the '
                                  'species-linked rule samples uniformly on [0, nominal life expectancy) of that species\' '
                                  'catalog record: an initial sampling range, not a hard death age and not an age-biomass '
                                  'relation',
              'derived': derived_record(cfg),
              'rng': {'seed': cfg['seed'], 'constructor': RNG_RECORD['constructor'], 'bit_generator': RNG_RECORD['bit_generator'],
                      'numpy_version': np.__version__, 'draw_order': RNG_RECORD['draw_order'][cfg['overlap']]},
              'seed': cfg['seed'], 'overlap': cfg['overlap'], 'cover_definition': 'per_species_union', 'cover_tolerance': tol,
              'cover_measurement': {k: final[k] for k in ('method', 'subsamples', 'raster_shape', 'resolution_m', 'raster_cells',
                                                         'raster_bytes', 'working_bytes', 'rows_per_chunk', 'index_bytes',
                                                         'error_basis', 'definition', 'boundary_note')},
              'cover_admission': admission,
              'total_union_cover_all_species': final['total_union_fraction'],
              'sum_of_species_union_covers': final['sum_of_group_union_fractions'],
              'domain': domain.to_dict(), 'domain_area_m2': A, 'n_plants': P, 'limits': dict(limits), 'attempts': attempts,
              'placement_passes': passes, 'count_iterations_total': iterations_used,
              'ordering_convention': _ordering(cfg),
              'placement': ('independent uniform centres conditioned on fitted count and cover acceptance (not an '
                            'unconditioned Poisson process)' if overlap else
                            (PLACEMENT_PERIODIC if domain.any_periodic else
                             'large-canopy-first sequential uniform proposals with global cross-species canopy exclusion '
                             '(roots may overlap); complete re-placement after each count correction; sparse hashed bins, '
                             'not all pairs; not a Poisson process')),
              'id_assignment': 'ids 1..P in final placement order',
              'establishment': 'resolved from the supplied initial surface' if surface is not None else 'SURFACE (unresolved)',
              'species': per_species, 'fixture_note': 'diameters/ages are configured fixtures, not calibrations'}
    if domain.any_periodic:       # only a periodic run adds this block: nonperiodic reports keep their exact key set
        report['boundary'] = dict(_periodic.boundary_record(domain),
                                  centres='uniform over the FULL domain (no margin / inset)',
                                  overlap_test=('shortest wrapped centre distance per periodic axis, strict < (ri + rj)^2'
                                                if not overlap else 'none (overlap allowed); cover is the torus union'),
                                  self_overlap=('refused (diameter > periodic span)' if not overlap else
                                                'allowed; union cover counts a self-overlapping canopy once'))
    prov = {'generator': REPORT_SCHEMA, 'generator_version': REPORT_SCHEMA_VERSION, 'catalog_hash': catalog.hash,
            'generation_config': report['generation_config'], 'derived': report['derived'], 'rng': report['rng'],
            'species': {p['species']: {'target_cover': p['target_cover'], 'measured_union_cover': p['measured_union_cover'],
                                       'count': p['count']} for p in per_species}}
    est = np.full(P, np.nan)
    state = _state.PlantState(domain, catalog.species_ids, catalog.hash, ids=np.arange(1, P + 1, dtype=np.int64),
                              species_index=species_index.astype(np.int32), x_m=X, y_m=Y, establishment_ground_m=est,
                              establishment_unresolved=np.ones(P, dtype=bool), agb_dry_kg=B,
                              seasonal_agb_dry_kg=np.zeros(P), age_days=age_all, status=np.zeros(P, dtype=np.int8),
                              time_days=cfg['time_days'], provenance=prov)
    if surface is not None:
        state.resolve_establishment(surface)
    return state, report


__all__ = ['GENERATION_SCHEMA', 'GENERATION_SCHEMA_VERSION', 'REPORT_SCHEMA', 'REPORT_SCHEMA_VERSION', 'OVERLAP_MODES',
           'DEFAULT_LIMITS', 'DEFAULTS', 'AGE_UNIFORM_TO_LIFE', 'DEFAULT_AGE_SPEC', 'ORDERING', 'RNG_RECORD', 'GenerationError', 'lognormal_parameters',
           'expected_crown_area_m2', 'initial_count', 'next_count', 'validate_generation_config', 'config_record',
           'derived_record', 'generate_population']
