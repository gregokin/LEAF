'''leaf.initialization: cover-driven multispecies population generator (bounded, reproducible).

Per species the configuration gives a target fractional cover f in [0, 1), a diameter mean m and standard
deviation s (lognormal with exactly that population mean/SD; s = 0 -> constant), and an initial age.
Biomass follows from each sampled diameter through the species inverse allometry.

Counts. With E[d^2] = m^2 + s^2 the expected crown area is E[A_c] = pi/4 * E[d^2]. Overlap allowed:
Poisson-Boolean start N0 = -A ln(1 - f) / E[A_c]; overlap disallowed: N0 = A f / E[A_c]. Counts are integers
and are then fitted to the MEASURED per-species union cover (leaf.canopy.measure_cover, midpoint raster of
resolution dx / cover_subsamples): N <- N * ln(1-f)/ln(1-c) (overlap) or N * f / c (no overlap), adding newly
sampled individuals or removing the most recently added ones, until |c - f| <= cover_tolerance or
max_iterations is exhausted (GenerationError; nothing is returned). Edge clipping and same-species overlaps
are therefore accounted for by measurement, not by formula. Exact continuous cover or exact finite-sample
diameter moments are not promised; all targets, measurements, counts and sample statistics are reported.

Placement. Overlap allowed: independent uniform centres in the domain, conditioned on the fitted count and
the cover acceptance (so NOT an unconditioned Poisson process). Overlap disallowed: sequential uniform
proposals with GLOBAL cross-species canopy exclusion (roots may overlap), species randomly interleaved
(ordering convention 'random_interleave' from the same RNG), bounded by max_attempts, using a sparse
hashed bin index (never all pairs).

Bounds max_plants, max_attempts, max_cover_cells, max_cover_bytes and max_iterations are explicit and
checked before allocation/sampling. RNG: numpy.random.default_rng(seed) (PCG64); the draw order is
documented in the report so a run is reproducible from seed + config + catalog hash.
'''
from __future__ import annotations

import math

import numpy as np

from leaf import canopy as _canopy
from leaf import state as _state
from leaf import traits as _traits

GENERATION_SCHEMA = 'leaf.generation_config'
GENERATION_SCHEMA_VERSION = 1
REPORT_SCHEMA = 'leaf.generation_report'
OVERLAP_MODES = ('allowed', 'disallowed')
DEFAULT_LIMITS = {'max_plants': 1_000_000, 'max_attempts': 20_000_000,
                  'max_cover_cells': _canopy.DEFAULT_MAX_COVER_CELLS, 'max_cover_bytes': _canopy.DEFAULT_MAX_COVER_BYTES}
DEFAULTS = {'overlap': 'allowed', 'cover_tolerance': 0.005, 'cover_subsamples': 4, 'max_iterations': 12,
            'time_days': 0.0, 'cover_definition': 'per_species_union'}
TOP_KEYS = ('schema', 'schema_version', 'domain', 'seed', 'overlap', 'cover_definition', 'cover_tolerance',
            'cover_subsamples', 'max_iterations', 'limits', 'species', 'time_days', 'notes')
SPECIES_KEYS = ('target_cover', 'diameter_mean_m', 'diameter_std_m', 'initial_age_days', 'notes')
ORDERING_CONVENTION = 'random_interleave'
RNG_LABEL = 'numpy.random.default_rng(seed) (PCG64); per batch: diameters, then x, then y, then ages'


class GenerationError(RuntimeError):
    '''Unattainable or unbounded request; raised with diagnostics, nothing is returned or written.'''


# ------------------------------------------------------------------ pure helpers
def lognormal_parameters(mean, std):
    '''(mu, sigma) of ln d such that d has the requested population mean and SD; None when std == 0.'''
    m, s = _traits.positive(mean, 'diameter_mean_m'), _traits.nonnegative(std, 'diameter_std_m')
    if s == 0.0:
        return None
    s2 = math.log(1.0 + (s / m) ** 2)
    return math.log(m) - 0.5 * s2, math.sqrt(s2)


def expected_crown_area_m2(mean, std):
    '''pi/4 * E[d^2] with E[d^2] = mean^2 + std^2.'''
    return 0.25 * math.pi * (float(mean) ** 2 + float(std) ** 2)


def initial_count(area_m2, target_cover, expected_area_m2, overlap):
    if target_cover <= 0.0:
        return 0
    if overlap:
        n = -area_m2 * math.log(1.0 - target_cover) / expected_area_m2
    else:
        n = area_m2 * target_cover / expected_area_m2
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
    raise _traits.CatalogError(f'{name}: unsupported age distribution {dist!r} (constant | uniform)')


def validate_generation_config(config, catalog):
    '''PURE normalisation of a generation config dict against the catalog. Raises CatalogError.'''
    E = _traits.CatalogError
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
    if tol is not None:
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
    for sid, cfg in sp.items():
        if not catalog.has(sid):
            raise E(f'generation config species {sid!r} is not in the catalog {list(catalog.species_ids)}')
        if not isinstance(cfg, dict):
            raise E(f'species[{sid!r}] must be an object')
        unknown = sorted(set(cfg) - set(SPECIES_KEYS))
        missing = [k for k in SPECIES_KEYS[:4] if k not in cfg]
        if unknown or missing:
            raise E(f'species[{sid!r}]: unknown keys {unknown}, missing {missing}')
        f = _traits.nonnegative(cfg['target_cover'], f'species[{sid!r}].target_cover')
        if f >= 1.0:
            raise E(f'species[{sid!r}].target_cover must be < 1')
        m = _traits.positive(cfg['diameter_mean_m'], f'species[{sid!r}].diameter_mean_m')
        s = _traits.nonnegative(cfg['diameter_std_m'], f'species[{sid!r}].diameter_std_m')
        species.append({'id': sid, 'index': catalog.index(sid), 'target_cover': f, 'diameter_mean_m': m,
                        'diameter_std_m': s, 'initial_age_days': _age_spec(cfg['initial_age_days'], f'species[{sid!r}].initial_age_days'),
                        'lognormal': lognormal_parameters(m, s), 'expected_crown_area_m2': expected_crown_area_m2(m, s)})
    return {'schema': GENERATION_SCHEMA, 'schema_version': GENERATION_SCHEMA_VERSION, 'domain': domain, 'seed': seed,
            'overlap': overlap, 'cover_definition': 'per_species_union', 'cover_tolerance': tol, 'cover_subsamples': sub,
            'max_iterations': it, 'limits': limits, 'species': species, 'time_days': time_days,
            'notes': config.get('notes')}


# ------------------------------------------------------------------ sampling helpers
def _sample_diameters(rng, spec, n):
    if n == 0:
        return np.zeros(0)
    if spec['lognormal'] is None:
        return np.full(n, spec['diameter_mean_m'], dtype=np.float64)
    mu, sigma = spec['lognormal']
    return rng.lognormal(mu, sigma, n)


def _sample_ages(rng, spec, n):
    a = spec['initial_age_days']
    if a['distribution'] == 'constant':
        return np.full(n, a['value_days'], dtype=np.float64)
    return rng.uniform(a['min_days'], a['max_days'], n) if n else np.zeros(0)


def _sample_positions(rng, domain, n):
    x = rng.uniform(domain.x_min_m, domain.x_max_m, n) if n else np.zeros(0)
    y = rng.uniform(domain.y_min_m, domain.y_max_m, n) if n else np.zeros(0)
    return x, y


def _stats(d):
    if d.size == 0:
        return {'n': 0, 'mean': None, 'std': None, 'min': None, 'max': None}
    return {'n': int(d.size), 'mean': float(d.mean()), 'std': float(d.std()), 'min': float(d.min()), 'max': float(d.max())}


class _Bins:
    '''Sparse hashed bins over plant centres for canopy exclusion (O(P) memory, local queries).'''

    def __init__(self, size):
        self.size = float(size)
        self.cells = {}
        self.rmax = 0.0

    def key(self, x, y):
        return (int(math.floor(x / self.size)), int(math.floor(y / self.size)))

    def add(self, i, x, y, r):
        self.cells.setdefault(self.key(x, y), []).append(i)
        if r > self.rmax:
            self.rmax = r

    def conflicts(self, x, y, r, X, Y, R, alive):
        k = int(math.ceil((r + self.rmax) / self.size))
        bx, by = self.key(x, y)
        for i in range(bx - k, bx + k + 1):
            for j in range(by - k, by + k + 1):
                lst = self.cells.get((i, j))
                if not lst:
                    continue
                for q in lst:
                    if not alive[q]:
                        continue
                    dx, dy = X[q] - x, Y[q] - y
                    rr = R[q] + r
                    if dx * dx + dy * dy < rr * rr:
                        return True
        return False


# ------------------------------------------------------------------ generator
def generate_population(catalog, config, *, surface=None):
    '''Generate a PlantState (ids 1..P in placement order) and a report. See the module docstring.'''
    if not isinstance(catalog, _traits.SpeciesCatalog):
        raise _traits.CatalogError('catalog must be a leaf.traits.SpeciesCatalog')
    cfg = validate_generation_config(config, catalog)
    domain, limits, overlap = cfg['domain'], cfg['limits'], cfg['overlap'] == 'allowed'
    tol, sub, max_it = cfg['cover_tolerance'], cfg['cover_subsamples'], cfg['max_iterations']
    cells, nbytes = _canopy.raster_bound(domain, sub, True)
    if cells > limits['max_cover_cells'] or nbytes > limits['max_cover_bytes']:
        raise GenerationError(f'cover raster needs {cells} cells / {nbytes} bytes > limits '
                              f"{limits['max_cover_cells']} / {limits['max_cover_bytes']}; nothing allocated")
    if surface is not None:
        surface = domain.check_field(surface, 'initial surface')
    rng = np.random.default_rng(cfg['seed'])
    A = domain.area_m2
    species = cfg['species']
    S = len(species)
    n0 = [initial_count(A, sp['target_cover'], sp['expected_crown_area_m2'], overlap) for sp in species]
    if sum(n0) > limits['max_plants']:
        raise GenerationError(f'initial count estimate {sum(n0)} exceeds max_plants {limits["max_plants"]}; nothing sampled')
    # growing python lists (bounded by max_plants); converted to arrays for measurement
    sp_idx, X, Y, R, AGE, alive = [], [], [], [], [], []
    attempts = 0
    bins = None

    def measure(species_i=None):
        a = np.flatnonzero(np.asarray(alive, dtype=bool))
        if species_i is not None and a.size:
            a = a[np.asarray(sp_idx, dtype=np.int64)[a] == species_i]
        x = np.asarray(X, dtype=np.float64)[a]
        y = np.asarray(Y, dtype=np.float64)[a]
        r = np.asarray(R, dtype=np.float64)[a]
        g = np.asarray(sp_idx, dtype=np.int64)[a] if species_i is None else np.zeros(a.size, dtype=np.int64)
        return _canopy.measure_cover(domain, x, y, r, g, S if species_i is None else 1, subsamples=sub,
                                     max_cover_cells=limits['max_cover_cells'], max_cover_bytes=limits['max_cover_bytes'],
                                     total=species_i is None)

    def add_batch(si, n):
        '''Sample n individuals of species si (diameters, x, y, ages) and place them. Overlap mode: direct;
        exclusion mode: sequential proposals with rejection (x, y resampled per attempt).'''
        nonlocal attempts, bins
        if n <= 0:
            return
        if sum(alive) + n > limits['max_plants']:
            raise GenerationError(f'population would exceed max_plants {limits["max_plants"]} while fitting '
                                  f'{species[si]["id"]!r} cover; aborted without output')
        spec = species[si]
        d = _sample_diameters(rng, spec, n)
        if overlap:
            x, y = _sample_positions(rng, domain, n)
            ages = _sample_ages(rng, spec, n)
            for k in range(n):
                sp_idx.append(si)
                X.append(float(x[k]))
                Y.append(float(y[k]))
                R.append(0.5 * float(d[k]))
                AGE.append(float(ages[k]))
                alive.append(True)
            return
        ages = _sample_ages(rng, spec, n)
        if bins is None:
            bins = _Bins(max(float(d.max()), domain.dx_m, domain.dy_m))
        for k in range(n):
            r = 0.5 * float(d[k])
            placed = False
            while not placed:
                if attempts >= limits['max_attempts']:
                    raise GenerationError(f'max_attempts {limits["max_attempts"]} exhausted placing {species[si]["id"]!r} '
                                          f'without canopy overlap (placed {sum(alive)} so far); request unattainable '
                                          'at this density/tolerance; aborted without output')
                attempts += 1
                px = float(rng.uniform(domain.x_min_m, domain.x_max_m))
                py = float(rng.uniform(domain.y_min_m, domain.y_max_m))
                if not bins.conflicts(px, py, r, X, Y, R, alive):
                    i = len(X)
                    sp_idx.append(si)
                    X.append(px)
                    Y.append(py)
                    R.append(r)
                    AGE.append(float(ages[k]))
                    alive.append(True)
                    bins.add(i, px, py, r)
                    placed = True

    def remove_last(si, n):
        '''Deactivate the n most recently added live individuals of species si.'''
        k = 0
        for i in range(len(X) - 1, -1, -1):
            if k >= n:
                break
            if alive[i] and sp_idx[i] == si:
                alive[i] = False
                k += 1

    histories = [[] for _ in range(S)]
    iterations_used = 0
    if overlap:
        for si, spec in enumerate(species):
            add_batch(si, n0[si])
            N = n0[si]
            for it in range(max_it + 1):
                c = measure(si)['groups'][0]['union_fraction'] if N > 0 else 0.0
                histories[si].append({'count': N, 'measured_cover': c})
                if tol is None or abs(c - spec['target_cover']) <= tol:
                    break
                if it == max_it:
                    raise GenerationError(f'species {spec["id"]!r}: cover {c:.5f} not within {tol} of target '
                                          f'{spec["target_cover"]} after {max_it} count iterations; history {histories[si]}')
                iterations_used += 1
                new = next_count(N, c, spec['target_cover'], True)
                if new > N:
                    add_batch(si, new - N)
                else:
                    remove_last(si, N - new)
                N = new
    else:
        # global random interleave of the initial items of all species (hidden species priority avoided)
        items = np.repeat(np.arange(S, dtype=np.int64), np.asarray(n0, dtype=np.int64)) if S else np.zeros(0, np.int64)
        order = rng.permutation(items.shape[0]) if items.size else np.zeros(0, np.int64)
        # the permuted species sequence is placed as run-length chunks (add_batch samples one chunk at a time)
        for si, n in _interleaved_chunks(items[order]):
            add_batch(int(si), int(n))
        counts = list(n0)
        for it in range(max_it + 1):
            m = measure() if sum(alive) else None
            cov = [(m['groups'][si]['union_fraction'] if m else 0.0) if counts[si] > 0 else 0.0 for si in range(S)]
            pending = []
            for si, spec in enumerate(species):
                histories[si].append({'count': counts[si], 'measured_cover': cov[si]})
                if tol is not None and abs(cov[si] - spec['target_cover']) > tol:
                    pending.append(si)
            if not pending:
                break
            if it == max_it:
                bad = {species[si]['id']: histories[si] for si in pending}
                raise GenerationError(f'cover not within {tol} of target after {max_it} iterations for {sorted(bad)}; '
                                      f'history {bad}')
            iterations_used += 1
            adds = []
            for si in pending:
                new = next_count(counts[si], cov[si], species[si]['target_cover'], False)
                if new > counts[si]:
                    adds.append((si, new - counts[si]))
                else:
                    remove_last(si, counts[si] - new)
                counts[si] = new
            if adds:
                items = np.concatenate([np.full(n, si, dtype=np.int64) for si, n in adds])
                for si, n in _interleaved_chunks(items[rng.permutation(items.shape[0])]):
                    add_batch(int(si), int(n))
    # ---- final arrays (placement order, alive only), final measurement over all species
    keep = np.flatnonzero(np.asarray(alive, dtype=bool))
    P = int(keep.size)
    sp_arr = np.asarray(sp_idx, dtype=np.int64)[keep] if P else np.zeros(0, np.int64)
    x = np.asarray(X, dtype=np.float64)[keep] if P else np.zeros(0)
    y = np.asarray(Y, dtype=np.float64)[keep] if P else np.zeros(0)
    r = np.asarray(R, dtype=np.float64)[keep] if P else np.zeros(0)
    age = np.asarray(AGE, dtype=np.float64)[keep] if P else np.zeros(0)
    cat_idx = np.array([species[si]['index'] for si in range(S)], dtype=np.int64) if S else np.zeros(0, np.int64)
    species_index = cat_idx[sp_arr] if P else np.zeros(0, np.int64)
    B = catalog.biomass_from_diameter(species_index, 2.0 * r) if P else np.zeros(0)
    final = _canopy.measure_cover(domain, x, y, r, sp_arr, max(S, 1), subsamples=sub, max_cover_cells=limits['max_cover_cells'],
                                  max_cover_bytes=limits['max_cover_bytes'], total=True)
    per_species = []
    for si, spec in enumerate(species):
        m = sp_arr == si
        d = 2.0 * r[m]
        g = final['groups'][si]
        per_species.append({'species': spec['id'], 'target_cover': spec['target_cover'], 'measured_union_cover': g['union_fraction'],
                            'nominal_crown_area_fraction': g['nominal_fraction'], 'count': int(m.sum()), 'initial_count_estimate': n0[si],
                            'count_formula': ('N0 = -A ln(1-f) / E[A_c]' if overlap else 'N0 = A f / E[A_c]') + ', E[A_c] = pi/4 (mean^2 + std^2)',
                            'expected_crown_area_m2': spec['expected_crown_area_m2'],
                            'diameter_requested': {'mean_m': spec['diameter_mean_m'], 'std_m': spec['diameter_std_m'],
                                                   'distribution': 'constant' if spec['lognormal'] is None else 'lognormal'},
                            'diameter_sample_m': _stats(d), 'age_sample_days': _stats(age[m]),
                            'agb_dry_kg_total': float(B[m].sum()) if m.any() else 0.0,
                            'accepted': tol is None or abs(g['union_fraction'] - spec['target_cover']) <= tol,
                            'iterations': len(histories[si]) - 1, 'history': histories[si],
                            'relative_error_scale': g['relative_error_scale']})
    report = {'schema': REPORT_SCHEMA, 'schema_version': 1, 'leaf_version': _traits.LEAF_VERSION, 'catalog_hash': catalog.hash,
              'seed': cfg['seed'], 'rng': RNG_LABEL, 'overlap': cfg['overlap'], 'cover_definition': 'per_species_union',
              'cover_tolerance': tol, 'cover_measurement': {k: final[k] for k in ('method', 'subsamples', 'raster_shape', 'resolution_m',
                                                                                 'raster_cells', 'raster_bytes', 'error_basis', 'definition')},
              'total_union_cover_all_species': final['total_union_fraction'],
              'sum_of_species_union_covers': final['sum_of_group_union_fractions'],
              'domain': domain.to_dict(), 'domain_area_m2': A, 'n_plants': P, 'limits': dict(limits), 'attempts': attempts,
              'count_iterations_total': iterations_used, 'ordering_convention': ORDERING_CONVENTION if not overlap else 'species order, sample order',
              'placement': ('independent uniform centres conditioned on fitted count and cover acceptance (not an '
                            'unconditioned Poisson process)' if overlap else
                            'sequential uniform proposals with global cross-species canopy exclusion (roots may overlap); '
                            'sparse hashed bins, not all pairs'),
              'establishment': 'resolved from the supplied initial surface' if surface is not None else 'SURFACE (unresolved)',
              'species': per_species, 'fixture_note': 'diameters/ages are configured fixtures, not calibrations'}
    prov = {'generator': REPORT_SCHEMA, 'seed': cfg['seed'], 'overlap': cfg['overlap'], 'catalog_hash': catalog.hash,
            'cover_tolerance': tol, 'cover_subsamples': sub, 'species': {p['species']: {'target_cover': p['target_cover'],
                                                                                        'measured_union_cover': p['measured_union_cover'],
                                                                                        'count': p['count']} for p in per_species}}
    est = np.full(P, np.nan)
    state = _state.PlantState(domain, catalog.species_ids, catalog.hash, ids=np.arange(1, P + 1, dtype=np.int64),
                              species_index=species_index.astype(np.int32), x_m=x, y_m=y, establishment_ground_m=est,
                              establishment_unresolved=np.ones(P, dtype=bool), agb_dry_kg=B,
                              seasonal_agb_dry_kg=np.zeros(P), age_days=age, status=np.zeros(P, dtype=np.int8),
                              time_days=cfg['time_days'], provenance=prov)
    if surface is not None:
        state.resolve_establishment(surface)
    return state, report


def _interleaved_chunks(order):
    '''Run-length chunks (species, n) of a randomly permuted species sequence (preserves the random interleave
    while letting add_batch sample a few individuals at a time).'''
    out = []
    if order.size == 0:
        return out
    cur, n = int(order[0]), 0
    for v in order:
        v = int(v)
        if v == cur:
            n += 1
        else:
            out.append((cur, n))
            cur, n = v, 1
    out.append((cur, n))
    return out


__all__ = ['GENERATION_SCHEMA', 'GENERATION_SCHEMA_VERSION', 'REPORT_SCHEMA', 'OVERLAP_MODES', 'DEFAULT_LIMITS', 'DEFAULTS',
           'ORDERING_CONVENTION', 'RNG_LABEL', 'GenerationError', 'lognormal_parameters', 'expected_crown_area_m2',
           'initial_count', 'next_count', 'validate_generation_config', 'generate_population']
