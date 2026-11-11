'''leaf.disturbance: species-specific mechanical response to surface events (law `mechanical_response_v1`; UNCALIBRATED).

Scope. A pure, opt-in module that turns one immutable `leaf.events.SurfaceEventDiagnostics` plus the ACTUAL current
surface into per-plant aboveground biomass injury, standing death or dislodgement of LIVE plants. It owns no plant
state and no clock: `evaluate_event` works on explicit per-plant arrays (event order) and returns candidate arrays;
`respond_state` applies that to a copy of a canonical `PlantState`; the transactional session boundary
(`leaf.interevent.IntereventSession.apply_surface_event`) binds time / source / IDs and publishes. No soil or surface
remapping, no wind or water physics, no fire; MAPLE / SYRUP are never imported. Aboveground dry biomass only: no
belowground or carbon pool exists or is created.

Catalog (every species coefficient lives in that species' `disturbance` section; no table in this module)
  "disturbance": {"law": "mechanical_response_v1", "traits": {"burial": null | {...}, "exposure": null | {...},
                  "abrasion": null | {...}}, "status": str, "notes": str, "units": {...}}
  `law` "unimplemented" (or a null section) is explicitly INACTIVE: Phase 0/1 catalogs are unchanged. Any other law,
  unknown keys, a missing trait, a non-finite / out-of-range value or a malformed mechanism REFUSES (never silently
  disables). A mechanism set to `null` is explicitly off; there are no defaults.
  burial   {increment_basis "net"|"gross", scale "m"|"height_fraction", injury_threshold, injury_saturation,
            max_injury_fraction [0,1), curve "linear"|"smoothstep", lethal_depth_m null|>0,
            lethal_height_fraction null|>0}
  exposure {increment_basis "net"|"gross", injury_threshold_m, injury_saturation_m, max_injury_fraction [0,1),
            curve, lethal_depth_m null|>0, dislodgement_depth_m null|>0}
  abrasion {dose_unit str, applies_to_sources [subset of events.SOURCES], injury_threshold, injury_saturation (both in
            `dose_unit`), max_injury_fraction [0,1), curve, lethal_dose null|>0}

Quantities (metres, event time order; all > = 0 unless stated). From the stored establishment ground and the current
surface sampled at the plant (containing cell): delta_after = current - establishment (signed NET change since
establishment, + deposition / burial, - erosion / exposure); burial_after = max(delta_after, 0); exposure_after =
max(-delta_after, 0). The state BEFORE the event is delta_before = delta_after - net_surface_delta_m (net event change).
Increments caused by THIS event only: net_burial_increase = max(burial_after - burial_before, 0), net_exposure_increase
analogously; gross values are the event's deposition_depth_m / erosion_depth_m (a deposition that merely refills an
earlier erosion hole has gross > net). Each mechanism declares which it uses (`increment_basis`). Accumulated
historical burial is never reapplied as new injury: injury responds to the event increment only (a plant is
re-evaluated only when an event names it); ABSOLUTE post-event burial / exposure / dose may cross explicit lethal or
dislodgement thresholds.
  Abrasion injury needs an ABRASION DOSE (event.abrasion_dose in event.abrasion_unit, exactly equal to the species'
  `dose_unit`, only for events whose source is in `applies_to_sources`). Transport FLUX is never converted to a dose
  and never read; a species with an applicable abrasion mechanism and an event without a (same-unit) dose REFUSES.

Response composition (simultaneous, bounded). Per mechanism an injury fraction f_k = max_injury_fraction * g(x_k) with
g the clipped linear or smoothstep ramp between `injury_threshold` and `injury_saturation` (g = 0 below the threshold).
Surviving fractions combine multiplicatively: retained = (1-f_burial)(1-f_exposure)(1-f_abrasion) in (0, 1], so the
total loss B*(1-retained) never exceeds the existing dry AGB. Seasonal biomass E scales with the same factor
(E_new = min(E*retained, B_new)), so 0 <= E <= B and the seasonal share is preserved. Precedence: dislodgement
(exposure_after >= dislodgement_depth_m) -> status removed, the whole remaining B / E is EXPORTED and zeroed, anchors
and age frozen; lethal (burial / exposure / dose threshold) -> dead_standing, B / E unchanged (no injury loss on top);
otherwise injured by the composed fraction, status live. Per-cause removed dry AGB: dislodgement = B; injury loss is
attributed to the causes in proportion to f_k (sum equals the total within a few ULP; the per-plant ledger
B == B_new + removed is verified inside a 16 eps bound and refuses otherwise). Only LIVE plants respond; events that
name dead_standing / removed plants (or plants of species with an inactive law) are counted and left unchanged. X / Y,
establishment anchor, stable IDs and ages are never touched here.

Backends. 'numpy' (vectorised reference), 'numba' (compiled serial kernel), 'cuda' (the same vectorised code on CuPy
arrays bound to ONE explicit device; no per-plant Python loop; event-sized arrays are uploaded / downloaded explicitly
and counted). Bitwise identity across backends holds for the integer / comparison logic and for +,-,*,/ ; transcendental-
free by design (no exp / pow here), so parity is exact up to compiler contraction (numba fastmath is off; CuPy
elementwise ops are not fused). Memory is admitted with `estimate_event_bytes` BEFORE any allocation.

Everything is UNCALIBRATED: the catalog fixture numbers are readable examples, not field fits.
'''
from __future__ import annotations

import math

import numpy as np

from leaf import events as _events
from leaf import state as _state
from leaf import traits as _traits

VERSION = '1.0.0'
LAW = 'mechanical_response_v1'
BACKENDS = ('numpy', 'numba', 'cuda')
SOURCES = _events.SOURCES
BASES = ('net', 'gross')
SCALES = ('m', 'height_fraction')
CURVES = ('linear', 'smoothstep')
CAUSES = ('burial', 'exposure', 'abrasion', 'dislodgement')
OUT_NONE, OUT_INJURED, OUT_DEAD, OUT_REMOVED = 0, 1, 2, 3
OUTCOME_NAMES = {OUT_NONE: 'no_response', OUT_INJURED: 'injured', OUT_DEAD: 'dead_standing', OUT_REMOVED: 'removed'}
LETHAL_BURIAL, LETHAL_EXPOSURE, LETHAL_ABRASION = 1, 2, 4
FP64_EPS = float(np.finfo(np.float64).eps)
LEDGER_EPS_FACTOR = 16.0
CLOCK_TOL_DAYS = 1e-9
BYTES_PER_EVENT_PLANT_HOST = 640          # conservative: inputs 9x8, outputs 11x8, NumPy temporaries ~30x8, masks
BYTES_PER_EVENT_PLANT_CUDA = 1280         # host copies + device copies of the same arrays
_SECTION_KEYS = ('law', 'traits', 'status', 'notes', 'units')
_TRAIT_KEYS = ('burial', 'exposure', 'abrasion')
_BURIAL_KEYS = ('increment_basis', 'scale', 'injury_threshold', 'injury_saturation', 'max_injury_fraction', 'curve',
                'lethal_depth_m', 'lethal_height_fraction')
_EXPOSURE_KEYS = ('increment_basis', 'injury_threshold_m', 'injury_saturation_m', 'max_injury_fraction', 'curve',
                  'lethal_depth_m', 'dislodgement_depth_m')
_ABRASION_KEYS = ('dose_unit', 'applies_to_sources', 'injury_threshold', 'injury_saturation', 'max_injury_fraction',
                  'curve', 'lethal_dose')
# float / int table rows (kernel-facing); disabled mechanisms are NEUTRAL (x0 = 0, x1 = 1, f = 0, lethal = inf)
_F_ROWS = ('b_x0', 'b_x1', 'b_f', 'b_lm', 'b_lf', 'e_x0', 'e_x1', 'e_f', 'e_lm', 'e_dm', 'a_x0', 'a_x1', 'a_f',
           'a_lethal')
_I_ROWS = ('b_basis', 'b_scale', 'b_curve', 'e_basis', 'e_curve', 'a_curve')
_FI = {n: i for i, n in enumerate(_F_ROWS)}
_II = {n: i for i, n in enumerate(_I_ROWS)}
(F_BX0, F_BX1, F_BF, F_BLM, F_BLF, F_EX0, F_EX1, F_EF, F_ELM, F_EDM, F_AX0, F_AX1, F_AF, F_ALETHAL) = range(14)
(I_BBASIS, I_BSCALE, I_BCURVE, I_EBASIS, I_ECURVE, I_ACURVE) = range(6)
TABLE_BYTES_PER_SPECIES = 8 * len(_F_ROWS) + 8 * len(_I_ROWS)
CAVEATS = (
    'UNCALIBRATED threshold / response curves: catalog fixtures only; no field fit, no wind or water physics',
    'aboveground dry biomass only; no belowground or carbon pool; dead_standing keeps its remaining B / E',
    'only live plants respond; the composition of simultaneous responses is multiplicative-survival (documented)',
    'flux is never converted to abrasion dose; abrasion needs the exact declared unit',
)


class DisturbanceError(ValueError):
    '''Any law, event or argument error (nothing is returned / mutated).'''


class DisturbanceBudgetError(DisturbanceError):
    '''The explicit byte bound would be exceeded; raised before any allocation.'''


class DisturbanceBackendError(DisturbanceError):
    '''Explicit backend / device failure (no fallback, no silent device switch).'''


# ------------------------------------------------------------------ catalog law parsing (pure, S-sized)
def _real(v, name):
    try:
        return _traits.real(v, name)
    except _traits.CatalogError as exc:
        raise DisturbanceError(str(exc)) from exc


def _exact(d, keys, where):
    if not isinstance(d, dict):
        raise DisturbanceError(f'{where} must be null or an object')
    missing = [k for k in keys if k not in d]
    unknown = sorted(set(d) - set(keys))
    if missing or unknown:
        raise DisturbanceError(f'{where} must hold exactly {list(keys)}; missing {missing}, unknown {unknown}')


def _choice(v, allowed, where):
    if not isinstance(v, str) or v not in allowed:
        raise DisturbanceError(f'{where} must be one of {list(allowed)}, got {v!r}')
    return allowed.index(v)


def _ramp_params(d, k0, k1, where, scale_note):
    x0 = _real(d[k0], f'{where}.{k0}')
    x1 = _real(d[k1], f'{where}.{k1}')
    f = _real(d['max_injury_fraction'], f'{where}.max_injury_fraction')
    if x0 < 0.0 or not x0 < x1:
        raise DisturbanceError(f'{where} needs 0 <= {k0} < {k1} ({scale_note}), got {x0}, {x1}')
    if not 0.0 <= f < 1.0:
        raise DisturbanceError(f'{where}.max_injury_fraction must lie in [0, 1) (a standing plant keeps positive '
                               f'biomass; use the lethal / dislodgement thresholds for death), got {f}')
    return x0, x1, f


def _opt_pos(v, where):
    if v is None:
        return math.inf
    x = _real(v, where)
    if x <= 0.0:
        raise DisturbanceError(f'{where} must be null (off) or > 0, got {x}')
    return x


def _parse_species(sec, sid):
    '''-> (active, floats dict, ints dict, info dict). Raises DisturbanceError naming the species.'''
    where = f'species {sid!r} disturbance'
    fl = {'b_x0': 0.0, 'b_x1': 1.0, 'b_f': 0.0, 'b_lm': math.inf, 'b_lf': math.inf, 'e_x0': 0.0, 'e_x1': 1.0,
          'e_f': 0.0, 'e_lm': math.inf, 'e_dm': math.inf, 'a_x0': 0.0, 'a_x1': 1.0, 'a_f': 0.0, 'a_lethal': math.inf}
    it = {n: 0 for n in _I_ROWS}
    info = {'burial': False, 'exposure': False, 'abrasion': False, 'dose_unit': None, 'source_mask': 0}
    if sec is None or (isinstance(sec, dict) and sec.get('law') == _traits.UNIMPLEMENTED):
        return False, fl, it, info
    if not isinstance(sec, dict):
        raise DisturbanceError(f'{where} must be null or an object')
    law = sec.get('law')
    if law != LAW:
        raise DisturbanceError(f'{where}.law {law!r} is unknown to leaf.disturbance (supported: {LAW!r}, or '
                               f'"{_traits.UNIMPLEMENTED}" for an explicitly inactive section)')
    unknown = sorted(set(sec) - set(_SECTION_KEYS))
    if unknown:
        raise DisturbanceError(f'{where}: unknown keys {unknown} (allowed: {list(_SECTION_KEYS)})')
    tr = sec.get('traits')
    _exact(tr, _TRAIT_KEYS, f'{where}.traits')
    bu, ex, ab = tr['burial'], tr['exposure'], tr['abrasion']
    if bu is not None:
        w = f'{where}.traits.burial'
        _exact(bu, _BURIAL_KEYS, w)
        it['b_basis'] = _choice(bu['increment_basis'], BASES, f'{w}.increment_basis')
        it['b_scale'] = _choice(bu['scale'], SCALES, f'{w}.scale')
        it['b_curve'] = _choice(bu['curve'], CURVES, f'{w}.curve')
        fl['b_x0'], fl['b_x1'], fl['b_f'] = _ramp_params(bu, 'injury_threshold', 'injury_saturation', w,
                                                         f'in {SCALES[it["b_scale"]]}')
        fl['b_lm'] = _opt_pos(bu['lethal_depth_m'], f'{w}.lethal_depth_m')
        fl['b_lf'] = _opt_pos(bu['lethal_height_fraction'], f'{w}.lethal_height_fraction')
        info['burial'] = True
    if ex is not None:
        w = f'{where}.traits.exposure'
        _exact(ex, _EXPOSURE_KEYS, w)
        it['e_basis'] = _choice(ex['increment_basis'], BASES, f'{w}.increment_basis')
        it['e_curve'] = _choice(ex['curve'], CURVES, f'{w}.curve')
        fl['e_x0'], fl['e_x1'], fl['e_f'] = _ramp_params(ex, 'injury_threshold_m', 'injury_saturation_m', w, 'in m')
        fl['e_lm'] = _opt_pos(ex['lethal_depth_m'], f'{w}.lethal_depth_m')
        fl['e_dm'] = _opt_pos(ex['dislodgement_depth_m'], f'{w}.dislodgement_depth_m')
        info['exposure'] = True
    if ab is not None:
        w = f'{where}.traits.abrasion'
        _exact(ab, _ABRASION_KEYS, w)
        unit = ab['dose_unit']
        if not isinstance(unit, str) or not unit:
            raise DisturbanceError(f'{w}.dose_unit must be a non-empty string (the exact declared dose unit)')
        srcs = ab['applies_to_sources']
        if not isinstance(srcs, list) or not srcs or any(not isinstance(s, str) or s not in SOURCES for s in srcs) \
                or len(set(srcs)) != len(srcs):
            raise DisturbanceError(f'{w}.applies_to_sources must be a non-empty list of distinct names from {list(SOURCES)}')
        it['a_curve'] = _choice(ab['curve'], CURVES, f'{w}.curve')
        fl['a_x0'], fl['a_x1'], fl['a_f'] = _ramp_params(ab, 'injury_threshold', 'injury_saturation', w,
                                                         f'in {unit!r}')
        fl['a_lethal'] = _opt_pos(ab['lethal_dose'], f'{w}.lethal_dose')
        info['abrasion'] = True
        info['dose_unit'] = unit
        info['source_mask'] = int(sum(1 << SOURCES.index(s) for s in srcs))
    return True, fl, it, info


def _cupy():
    try:
        import cupy
    except Exception as exc:   # pragma: no cover - environment dependent
        raise DisturbanceBackendError(f'backend "cuda" needs the CuPy package: {exc}') from exc
    return cupy


def _numba():
    try:
        import numba
    except Exception as exc:   # pragma: no cover - environment dependent
        raise DisturbanceBackendError(f'backend "numba" needs the numba package: {exc}') from exc
    return numba


class CompiledDisturbance:
    '''Immutable S-sized species table (sorted catalog ids). `any_active` False => no array is held (zero cost).'''

    def __init__(self, species_ids, catalog_hash, backend, device, active, info, tf, ti, tf_dev, ti_dev):
        self.species_ids = tuple(species_ids)
        self.catalog_hash = catalog_hash
        self.backend = backend
        self.device = device
        self.active = active                  # tuple[bool] per species
        self.info = info                      # tuple[dict] per species (mechanism flags, dose unit, source mask)
        self.any_active = bool(any(active))
        self._tf, self._ti, self._tf_dev, self._ti_dev = tf, ti, tf_dev, ti_dev

    @property
    def n_species(self):
        return len(self.species_ids)

    def resident_bytes(self):
        '''Owned bytes of the table(s): host (always when active) + device mirror (cuda).'''
        if not self.any_active:
            return 0
        n = TABLE_BYTES_PER_SPECIES * self.n_species
        return n * (2 if self.backend == 'cuda' else 1)

    def describe(self):
        return {'law': LAW, 'version': VERSION, 'backend': self.backend, 'device': self.device,
                'n_species': self.n_species, 'active_species': [s for s, a in zip(self.species_ids, self.active) if a],
                'catalog_hash': self.catalog_hash, 'resident_bytes': self.resident_bytes()}

    def coefficients(self, species_id):
        '''Copy of the kernel coefficients of one species (host; for tests / documentation).'''
        if species_id not in self.species_ids:
            raise DisturbanceError(f'unknown species {species_id!r}')
        if not self.any_active:
            raise DisturbanceError('table holds no active species')
        i = self.species_ids.index(species_id)
        out = {n: float(self._tf[_FI[n], i]) for n in _F_ROWS}
        out.update({n: int(self._ti[_II[n], i]) for n in _I_ROWS})
        out.update(self.info[i])
        out['active'] = bool(self.active[i])
        return out


def compile_species(catalog, backend, *, device=None):
    '''Parse EVERY species' `disturbance` section (refusing unknown / malformed active laws, naming the species) and
    build the immutable table. Inactive catalogs hold no array. cuda: an explicit device id is REQUIRED and must be the
    current device (no switching, no fallback).'''
    if not isinstance(catalog, _traits.SpeciesCatalog):
        raise DisturbanceError('catalog must be a leaf.traits.SpeciesCatalog')
    if backend not in BACKENDS:
        raise DisturbanceError(f'backend must be explicitly one of {BACKENDS}, got {backend!r}')
    ids = catalog.species_ids
    S = len(ids)
    errors, active, infos = [], [], []
    tf = np.zeros((len(_F_ROWS), S), dtype=np.float64)
    ti = np.zeros((len(_I_ROWS), S), dtype=np.int64)
    for k, sid in enumerate(ids):
        try:
            a, fl, it, info = _parse_species(catalog.section(sid, 'disturbance'), sid)
        except (DisturbanceError, _traits.CatalogError) as exc:
            errors.append(str(exc))
            active.append(False)
            infos.append({})
            continue
        active.append(a)
        infos.append(info)
        for n in _F_ROWS:
            tf[_FI[n], k] = fl[n]
        for n in _I_ROWS:
            ti[_II[n], k] = it[n]
    if errors:
        more = f' (+{len(errors) - 6} more)' if len(errors) > 6 else ''
        raise DisturbanceError('species catalog is not usable by leaf.disturbance: ' + ' | '.join(errors[:6]) + more)
    if not any(active):
        return CompiledDisturbance(ids, catalog.hash, backend, None, tuple(active), tuple(infos), None, None, None, None)
    tf.setflags(write=False)
    ti.setflags(write=False)
    tf_dev = ti_dev = None
    dev_id = None
    if backend == 'cuda':
        cp = _cupy()
        if isinstance(device, (bool, np.bool_)) or not isinstance(device, (int, np.integer)) or int(device) < 0:
            raise DisturbanceBackendError('backend "cuda" needs an explicit non-negative integer device id')
        dev_id = int(device)
        cur = int(cp.cuda.runtime.getDevice())
        if cur != dev_id:
            raise DisturbanceBackendError(f'device {dev_id} requested but the current CUDA device is {cur}; this '
                                          'module never switches devices or falls back')
        tf_dev, ti_dev = cp.asarray(tf), cp.asarray(ti)
    elif backend == 'numba':
        _numba()      # fail early and explicitly when the backend is unavailable
    return CompiledDisturbance(ids, catalog.hash, backend, dev_id, tuple(active), tuple(infos), tf, ti, tf_dev, ti_dev)


def estimate_event_bytes(n_event, n_species, backend):
    '''Itemised conservative bound of ONE evaluate_event call (bytes; host + device summed for cuda); excludes the
    resident table (`CompiledDisturbance.resident_bytes`), the interpreter, JIT / CUDA context and allocator pools.'''
    if isinstance(n_event, (bool, np.bool_)) or not isinstance(n_event, (int, np.integer)) or int(n_event) < 0:
        raise DisturbanceError(f'n_event must be a non-negative integer, got {n_event!r}')
    n = int(n_event)
    per = BYTES_PER_EVENT_PLANT_CUDA if backend == 'cuda' else BYTES_PER_EVENT_PLANT_HOST
    return {'per_event_plant_bytes': per, 'arrays_and_temporaries': per * n, 'validation_scratch': 64 * n,
            'total_bytes': (per + 64) * n,
            'scope': 'event-sized input / output arrays, NumPy / CuPy temporaries and boolean masks; no P-sized array'}


# ------------------------------------------------------------------ kernels
def _ramp_xp(xp, x, x0, x1, fmax, curve):
    g = xp.clip((x - x0) / (x1 - x0), 0.0, 1.0)
    g = xp.where(curve == 1, g * g * (3.0 - 2.0 * g), g)
    return fmax * g


def _respond_xp(xp, tf, ti, s, B, E, live, height, dep, ero, net, delta, dose):
    '''Vectorised reference (also the CUDA path with xp = cupy). All inputs are 1-D arrays in the namespace.'''
    def f(row):
        return tf[row][s]

    def i(row):
        return ti[row][s]
    bu_a = xp.maximum(delta, 0.0)
    ex_a = xp.maximum(-delta, 0.0)
    d_b = delta - net
    bu_b = xp.maximum(d_b, 0.0)
    ex_b = xp.maximum(-d_b, 0.0)
    bu_inc = xp.where(i(I_BBASIS) == 1, dep, xp.maximum(bu_a - bu_b, 0.0))
    ex_inc = xp.where(i(I_EBASIS) == 1, ero, xp.maximum(ex_a - ex_b, 0.0))
    xb = xp.where(i(I_BSCALE) == 1, bu_inc / height, bu_inc)
    fb = _ramp_xp(xp, xb, f(F_BX0), f(F_BX1), f(F_BF), i(I_BCURVE))
    fe = _ramp_xp(xp, ex_inc, f(F_EX0), f(F_EX1), f(F_EF), i(I_ECURVE))
    fa = _ramp_xp(xp, dose, f(F_AX0), f(F_AX1), f(F_AF), i(I_ACURVE))
    lb = (bu_a >= f(F_BLM)) | ((bu_a / height) >= f(F_BLF))
    le = ex_a >= f(F_ELM)
    la = dose >= f(F_ALETHAL)
    disl = live & (ex_a >= f(F_EDM))
    dead = live & (~disl) & (lb | le | la)
    retained = (1.0 - fb) * (1.0 - fe) * (1.0 - fa)
    inj = live & (~disl) & (~dead) & (retained < 1.0)
    r = xp.where(inj, retained, 1.0)
    bn_inj = B * r
    b_new = xp.where(disl, 0.0, xp.where(inj, bn_inj, B))
    e_new = xp.where(disl, 0.0, xp.where(inj, xp.minimum(E * r, bn_inj), E))
    loss = xp.where(inj, B - bn_inj, 0.0)
    fs = fb + fe + fa
    safe = xp.where(fs > 0.0, fs, 1.0)
    lost_b = loss * (fb / safe)
    lost_e = loss * (fe / safe)
    lost_a = loss * (fa / safe)
    lost_d = xp.where(disl, B, 0.0)
    code = xp.where(disl, OUT_REMOVED, xp.where(dead, OUT_DEAD, xp.where(inj, OUT_INJURED, OUT_NONE))).astype(xp.int8)
    bits = (lb.astype(xp.int8) * LETHAL_BURIAL + le.astype(xp.int8) * LETHAL_EXPOSURE
            + la.astype(xp.int8) * LETHAL_ABRASION).astype(xp.int8)
    bits = xp.where(live & ~disl, bits, 0).astype(xp.int8)
    z = xp.where(inj, 1.0, 0.0)
    return {'B_new': b_new, 'E_new': e_new, 'outcome': code, 'lethal_bits': bits, 'f_burial': fb * z,
            'f_exposure': fe * z, 'f_abrasion': fa * z, 'lost_burial': lost_b, 'lost_exposure': lost_e,
            'lost_abrasion': lost_a, 'lost_dislodgement': lost_d, 'exported_E': xp.where(disl, E, 0.0),
            'burial_after': bu_a, 'exposure_after': ex_a, 'burial_increment': bu_inc, 'exposure_increment': ex_inc}


_KERNEL = {}


def _numba_kernel():
    if 'numba' in _KERNEL:
        return _KERNEL['numba']
    numba = _numba()

    @numba.njit(cache=False)
    def kern(tf, ti, s, B, E, live, height, dep, ero, net, delta, dose, o_B, o_E, o_code, o_bits, o_fb, o_fe, o_fa,
             o_lb, o_le, o_la, o_ld, o_xe, o_bua, o_exa, o_bui, o_exi):
        n = B.shape[0]
        for p in range(n):
            k = s[p]
            h = height[p]
            dl = delta[p]
            bu_a = dl if dl > 0.0 else 0.0
            ex_a = -dl if -dl > 0.0 else 0.0
            d_b = dl - net[p]
            bu_b = d_b if d_b > 0.0 else 0.0
            ex_b = -d_b if -d_b > 0.0 else 0.0
            inc = bu_a - bu_b
            if inc < 0.0:
                inc = 0.0
            if ti[I_BBASIS, k] == 1:
                inc = dep[p]
            bu_inc = inc
            inc = ex_a - ex_b
            if inc < 0.0:
                inc = 0.0
            if ti[I_EBASIS, k] == 1:
                inc = ero[p]
            ex_inc = inc
            xb = bu_inc
            if ti[I_BSCALE, k] == 1:
                xb = bu_inc / h
            # burial ramp
            g = (xb - tf[F_BX0, k]) / (tf[F_BX1, k] - tf[F_BX0, k])
            g = min(max(g, 0.0), 1.0)
            if ti[I_BCURVE, k] == 1:
                g = g * g * (3.0 - 2.0 * g)
            fb = tf[F_BF, k] * g
            g = (ex_inc - tf[F_EX0, k]) / (tf[F_EX1, k] - tf[F_EX0, k])
            g = min(max(g, 0.0), 1.0)
            if ti[I_ECURVE, k] == 1:
                g = g * g * (3.0 - 2.0 * g)
            fe = tf[F_EF, k] * g
            g = (dose[p] - tf[F_AX0, k]) / (tf[F_AX1, k] - tf[F_AX0, k])
            g = min(max(g, 0.0), 1.0)
            if ti[I_ACURVE, k] == 1:
                g = g * g * (3.0 - 2.0 * g)
            fa = tf[F_AF, k] * g
            lb = (bu_a >= tf[F_BLM, k]) or ((bu_a / h) >= tf[F_BLF, k])
            le = ex_a >= tf[F_ELM, k]
            la = dose[p] >= tf[F_ALETHAL, k]
            lv = live[p]
            disl = lv and (ex_a >= tf[F_EDM, k])
            dead = lv and (not disl) and (lb or le or la)
            retained = (1.0 - fb) * (1.0 - fe) * (1.0 - fa)
            inj = lv and (not disl) and (not dead) and (retained < 1.0)
            b = B[p]
            e = E[p]
            r = 1.0
            if inj:
                r = retained
            bn_inj = b * r
            bn = b
            en = e
            if disl:
                bn = 0.0
                en = 0.0
            elif inj:
                bn = bn_inj
                en = min(e * r, bn_inj)
            loss = 0.0
            if inj:
                loss = b - bn_inj
            fs = fb + fe + fa
            safe = fs if fs > 0.0 else 1.0
            o_B[p] = bn
            o_E[p] = en
            code = OUT_NONE
            if disl:
                code = OUT_REMOVED
            elif dead:
                code = OUT_DEAD
            elif inj:
                code = OUT_INJURED
            o_code[p] = code
            bits = 0
            if lv and (not disl):
                if lb:
                    bits += LETHAL_BURIAL
                if le:
                    bits += LETHAL_EXPOSURE
                if la:
                    bits += LETHAL_ABRASION
            o_bits[p] = bits
            z = 1.0 if inj else 0.0
            o_fb[p] = fb * z
            o_fe[p] = fe * z
            o_fa[p] = fa * z
            o_lb[p] = loss * (fb / safe)
            o_le[p] = loss * (fe / safe)
            o_la[p] = loss * (fa / safe)
            o_ld[p] = b if disl else 0.0
            o_xe[p] = e if disl else 0.0
            o_bua[p] = bu_a
            o_exa[p] = ex_a
            o_bui[p] = bu_inc
            o_exi[p] = ex_inc

    _KERNEL['numba'] = kern
    return kern


_OUT_NAMES = ('B_new', 'E_new', 'outcome', 'lethal_bits', 'f_burial', 'f_exposure', 'f_abrasion', 'lost_burial',
              'lost_exposure', 'lost_abrasion', 'lost_dislodgement', 'exported_E', 'burial_after', 'exposure_after',
              'burial_increment', 'exposure_increment')


def _run_kernel(table, s, B, E, live, height, dep, ero, net, delta, dose):
    '''Backend dispatch on HOST event arrays; returns HOST output arrays (cuda: explicit upload / download).'''
    be = table.backend
    if be == 'numpy':
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            out = _respond_xp(np, table._tf, table._ti, s, B, E, live, height, dep, ero, net, delta, dose)
        return {k: np.ascontiguousarray(v) for k, v in out.items()}
    if be == 'numba':
        n = B.shape[0]
        out = {'B_new': np.empty(n), 'E_new': np.empty(n), 'outcome': np.empty(n, np.int8),
               'lethal_bits': np.empty(n, np.int8), 'f_burial': np.empty(n), 'f_exposure': np.empty(n),
               'f_abrasion': np.empty(n), 'lost_burial': np.empty(n), 'lost_exposure': np.empty(n),
               'lost_abrasion': np.empty(n), 'lost_dislodgement': np.empty(n), 'exported_E': np.empty(n),
               'burial_after': np.empty(n), 'exposure_after': np.empty(n), 'burial_increment': np.empty(n),
               'exposure_increment': np.empty(n)}
        _numba_kernel()(table._tf, table._ti, s, B, E, live, height, dep, ero, net, delta, dose,
                        *(out[k] for k in _OUT_NAMES))
        return out
    cp = _cupy()
    cur = int(cp.cuda.runtime.getDevice())
    if cur != table.device:
        raise DisturbanceBackendError(f'table is bound to CUDA device {table.device} but the current device is {cur}; '
                                      'no device switching and no fallback')
    args = [cp.asarray(a) for a in (s, B, E, live, height, dep, ero, net, delta, dose)]
    out = _respond_xp(cp, table._tf_dev, table._ti_dev, *args)
    return {k: np.ascontiguousarray(cp.asnumpy(v)) for k, v in out.items()}


# ------------------------------------------------------------------ event evaluation (host arrays, event order)
def _f64(a, name, n):
    try:
        arr = _state._coerce_float(a, name, n)
    except _state.StateError as exc:
        raise DisturbanceError(str(exc)) from exc
    if not np.all(np.isfinite(arr)):
        raise DisturbanceError(f'{name} must be finite')
    return arr


def evaluate_event(table, catalog, event, *, species_index, status, B, E, x_m, y_m, establishment_ground_m, domain,
                   current_surface, max_bytes, extra_reserve_bytes=0):
    '''Per-plant response of the n plants named by `event` (arrays are in EVENT order, host arrays; B / E of live
    plants are the CURRENT values). Pure: returns candidate arrays and diagnostics, mutates nothing.

    Returns a dict: B_new, E_new (kg), status_new (int8 state codes; non-live plants unchanged), outcome (int8,
    OUT_*; -1 for plants that did not respond because they are not live / their law is inactive), lethal_bits,
    removed_kg {cause -> [n]}, exported_E_kg, delta_after_m, burial_after_m, exposure_after_m, increments, and
    `diagnostics` (JSON-safe per-species and ledger sums). Raises DisturbanceError (units, missing dose, ledger) or
    DisturbanceBudgetError (before allocation).'''
    if not isinstance(table, CompiledDisturbance):
        raise DisturbanceError('table must come from leaf.disturbance.compile_species')
    if not isinstance(event, _events.SurfaceEventDiagnostics):
        raise DisturbanceError('event must be a leaf.events.SurfaceEventDiagnostics')
    if table.catalog_hash != catalog.hash:
        raise DisturbanceError('table was compiled for another catalog')
    for nm, v, lo in (('max_bytes', max_bytes, 1), ('extra_reserve_bytes', extra_reserve_bytes, 0)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or int(v) < lo:
            raise DisturbanceError(f'{nm} must be a non-boolean integer >= {lo}, got {v!r}')
    n = event.n
    est = estimate_event_bytes(n, table.n_species, table.backend)
    need = est['total_bytes'] + int(extra_reserve_bytes)
    if need > int(max_bytes):
        raise DisturbanceBudgetError(f'evaluate_event needs ~{need} bytes (n={n}) > max_bytes={int(max_bytes)}; refused '
                                     'before any allocation')
    event.revalidate()
    S = table.n_species
    sidx = _state._coerce_int(species_index, 'species_index', n, np.int64, 0, S)
    st = _state._coerce_int(status, 'status', n, np.int8, 0, 3)
    Bv, Ev = _f64(B, 'B', n), _f64(E, 'E', n)
    xv, yv = _f64(x_m, 'x_m', n), _f64(y_m, 'y_m', n)
    est_g = _f64(establishment_ground_m, 'establishment_ground_m', n)
    if n and (float(Bv.min()) < 0.0 or float(Ev.min()) < 0.0 or bool(np.any(Ev > Bv))):
        raise DisturbanceError('B and E must satisfy 0 <= E <= B')
    live = st == _state.LIVE
    act_sp = np.array(table.active, dtype=bool)
    responds = live & (act_sp[sidx] if S else live)
    if responds.any() and bool(np.any(Bv[responds] <= 0.0)):
        raise DisturbanceError('live plants must have positive dry AGB')
    # ---- abrasion dose: exact unit and applicable source, per species; flux is never read
    src_bit = 1 << SOURCES.index(event.source)
    dose = np.zeros(n, dtype=np.float64)
    dose_species = []
    for k in (np.unique(sidx[responds]) if n else []):
        info = table.info[int(k)]
        if not info['abrasion'] or not (info['source_mask'] & src_bit):
            continue
        sid = table.species_ids[int(k)]
        if event.abrasion_dose is None:
            raise DisturbanceError(f'species {sid!r} has an abrasion mechanism for source {event.source!r} but the '
                                   'event carries no abrasion_dose (transport flux is NOT a dose and is never '
                                   'converted); supply a dose in the declared unit or change the species law')
        if event.abrasion_unit != info['dose_unit']:
            raise DisturbanceError(f'event abrasion_unit {event.abrasion_unit!r} != species {sid!r} dose_unit '
                                   f'{info["dose_unit"]!r}: exact unit matching is required (no conversion)')
        m = responds & (sidx == int(k))
        dose[m] = event.abrasion_dose[m]
        dose_species.append(sid)
    # ---- actual current surface at the plant (containing cell) vs the FIXED establishment ground
    try:
        cur = domain.sample(current_surface, xv, yv, 'current surface') if n else np.zeros(0)
    except _state.StateError as exc:
        raise DisturbanceError(str(exc)) from exc
    delta = cur - est_g
    if not np.all(np.isfinite(delta)):
        raise DisturbanceError('surface delta is not finite')
    height = np.ones(n, dtype=np.float64)
    if responds.any():
        h = catalog.height_from_biomass(sidx[responds], Bv[responds])          # the ONE shared allometry
        if not np.all(h > 0.0):
            raise DisturbanceError('intact height must be positive for live plants')
        height[responds] = h
    dep, ero, net = event.deposition_depth_m, event.erosion_depth_m, event.net_surface_delta_m
    out = {}
    if table.any_active and n:
        out = _run_kernel(table, sidx, Bv, Ev, responds, height, np.array(dep), np.array(ero), np.array(net), delta, dose)
    else:
        zero = np.zeros(n)
        out = {'B_new': Bv.copy(), 'E_new': Ev.copy(), 'outcome': np.zeros(n, np.int8),
               'lethal_bits': np.zeros(n, np.int8), 'f_burial': zero.copy(), 'f_exposure': zero.copy(),
               'f_abrasion': zero.copy(), 'lost_burial': zero.copy(), 'lost_exposure': zero.copy(),
               'lost_abrasion': zero.copy(), 'lost_dislodgement': zero.copy(), 'exported_E': zero.copy(),
               'burial_after': np.maximum(delta, 0.0), 'exposure_after': np.maximum(-delta, 0.0),
               'burial_increment': zero.copy(), 'exposure_increment': zero.copy()}
    for k in _OUT_NAMES:
        a = out[k]
        if a.dtype.kind == 'f' and not np.all(np.isfinite(a)):
            raise DisturbanceError(f'kernel output {k} is not finite')
    outcome = out['outcome'].astype(np.int8)
    B_new, E_new = out['B_new'], out['E_new']
    status_new = st.copy()
    status_new[responds & (outcome == OUT_DEAD)] = _state.DEAD_STANDING
    status_new[responds & (outcome == OUT_REMOVED)] = _state.REMOVED
    stand = responds & (outcome != OUT_REMOVED)
    if bool(np.any(B_new[stand] <= 0.0)) or bool(np.any(E_new > B_new)) or bool(np.any(E_new < 0.0)):
        raise DisturbanceError('response would leave a standing plant with non-positive dry AGB or E > B (underflow)')
    removed = {'burial': out['lost_burial'], 'exposure': out['lost_exposure'], 'abrasion': out['lost_abrasion'],
               'dislodgement': out['lost_dislodgement']}
    removed_total = Bv - B_new                       # exact per-plant removal (dislodgement: all of B)
    attributed = out['lost_burial'] + out['lost_exposure'] + out['lost_abrasion'] + out['lost_dislodgement']
    tol_pp = LEDGER_EPS_FACTOR * FP64_EPS * Bv
    resid_pp = np.abs(Bv - (B_new + attributed))
    if bool(np.any(resid_pp > tol_pp + 1e-300)):
        raise DisturbanceError('per-plant dry AGB ledger B == B_new + removed does not close within 16 eps; refused')
    outcome_rep = np.where(responds, outcome, -1).astype(np.int8)
    diag = _diagnostics(table, catalog, sidx, st, responds, outcome, Bv, Ev, B_new, E_new, removed, out, dose_species,
                        event)
    return {'B_new': B_new, 'E_new': E_new, 'status_new': status_new, 'outcome': outcome_rep,
            'lethal_bits': out['lethal_bits'], 'removed_kg': removed, 'removed_total_kg': removed_total,
            'exported_E_kg': out['exported_E'], 'delta_after_m': delta, 'burial_after_m': out['burial_after'],
            'exposure_after_m': out['exposure_after'], 'burial_increment_m': out['burial_increment'],
            'exposure_increment_m': out['exposure_increment'], 'f_burial': out['f_burial'],
            'f_exposure': out['f_exposure'], 'f_abrasion': out['f_abrasion'], 'dose_used': dose,
            'height_m': height, 'diagnostics': diag}


def _diagnostics(table, catalog, sidx, st, responds, outcome, B, E, B_new, E_new, removed, out, dose_species, event):
    S = table.n_species
    per = {}
    live = st == _state.LIVE
    for k, sid in enumerate(table.species_ids):
        m = sidx == k
        if not m.any():
            continue
        r = m & responds
        per[sid] = {'n_event': int(m.sum()), 'n_live': int((m & live).sum()), 'n_evaluated': int(r.sum()),
                    'n_skipped_not_live': int((m & ~live).sum()), 'n_skipped_inactive_law': int((m & live & ~responds).sum()),
                    'n_no_response': int((r & (outcome == OUT_NONE)).sum()),
                    'n_injured': int((r & (outcome == OUT_INJURED)).sum()),
                    'n_dead_standing': int((r & (outcome == OUT_DEAD)).sum()),
                    'n_removed': int((r & (outcome == OUT_REMOVED)).sum()),
                    'lethal_burial': int((r & ((out['lethal_bits'] & LETHAL_BURIAL) != 0)).sum()),
                    'lethal_exposure': int((r & ((out['lethal_bits'] & LETHAL_EXPOSURE) != 0)).sum()),
                    'lethal_abrasion': int((r & ((out['lethal_bits'] & LETHAL_ABRASION) != 0)).sum()),
                    'dry_agb_before_kg': math.fsum(B[m].tolist()), 'dry_agb_after_kg': math.fsum(B_new[m].tolist()),
                    'seasonal_before_kg': math.fsum(E[m].tolist()), 'seasonal_after_kg': math.fsum(E_new[m].tolist()),
                    'exported_seasonal_kg': math.fsum(out['exported_E'][m].tolist()),
                    'removed_dry_agb_kg': {c: math.fsum(removed[c][m].tolist()) for c in CAUSES}}
    tb, ta = math.fsum(B.tolist()), math.fsum(B_new.tolist())
    rem = {c: math.fsum(removed[c].tolist()) for c in CAUSES}
    rem_total = math.fsum((B - B_new).tolist())
    resid = tb - (ta + rem_total)
    attributed = math.fsum(rem.values())
    bound = LEDGER_EPS_FACTOR * FP64_EPS * tb          # the sum of the per-plant 16 eps bounds
    if not (math.isfinite(resid) and abs(resid) <= bound + 1e-300):
        raise DisturbanceError(f'population dry AGB ledger residual {resid!r} kg exceeds the FP64 bound {bound!r} kg')
    return {'event_id': event.event_id, 'source': event.source, 'n_event': int(sidx.shape[0]),
            'abrasion_species': dose_species, 'abrasion_unit_event': event.abrasion_unit,
            'flux_ignored': event.transport_flux is not None, 'per_species': per,
            'ledger': {'dry_agb_before_kg': tb, 'dry_agb_after_kg': ta, 'removed_dry_agb_kg': rem_total,
                       'removed_by_cause_kg': rem, 'attributed_sum_kg': attributed,
                       'residual_kg': resid, 'attribution_residual_kg': rem_total - attributed,
                       'bound_kg': bound,
                       'definition': 'old dry AGB = new retained dry AGB + removed dry AGB (dead_standing keeps its '
                                     'B / E; dislodgement exports and zeroes them); per-plant bound 16 eps B'},
            'seasonal_ledger_kg': {'before': math.fsum(E.tolist()), 'after': math.fsum(E_new.tolist()),
                                   'exported': math.fsum(out['exported_E'].tolist())},
            'n_no_response': int(((outcome == OUT_NONE) & responds).sum()),
            'n_injured': int(((outcome == OUT_INJURED) & responds).sum()),
            'n_dead_standing': int(((outcome == OUT_DEAD) & responds).sum()),
            'n_removed': int(((outcome == OUT_REMOVED) & responds).sum()),
            'caveats': list(CAVEATS)}


# ------------------------------------------------------------------ pure state-level response
def respond_state(state, catalog, event, current_surface, *, table=None, backend='numpy', max_bytes=1 << 30,
                  require_clock=True):
    '''Apply `event` to a COPY of `state` (the input is never mutated): returns (candidate PlantState, report).
    The event must end at the state's clock (`require_clock`), so a post-transport state and its event cannot be
    mismatched. IDs resolve once (unknown / duplicate IDs refuse). Anchors, X / Y, IDs and ages are unchanged.'''
    if not isinstance(state, _state.PlantState):
        raise DisturbanceError('state must be a leaf.state.PlantState')
    if state.catalog_hash != catalog.hash:
        raise DisturbanceError('state catalog hash differs from the catalog')
    state.require_resolved('applying a surface event')
    if table is None:
        table = compile_species(catalog, backend, device=None)
    try:
        rows = _events.resolve_event(state, event, require_elapsed=False)
    except _events.EventError as exc:
        raise DisturbanceError(str(exc)) from exc
    if require_clock and abs(event.time_end_days - state.time_days) > max(CLOCK_TOL_DAYS, 64 * FP64_EPS * abs(state.time_days)):
        raise DisturbanceError(f'event ends at {event.time_end_days} days but the population clock is at '
                               f'{state.time_days}; apply a response only at the end of the event interval')
    res = evaluate_event(table, catalog, event, species_index=state.species_index[rows], status=state.status[rows],
                         B=state.agb_dry_kg[rows], E=state.seasonal_agb_dry_kg[rows], x_m=state.x_m[rows],
                         y_m=state.y_m[rows], establishment_ground_m=state.establishment_ground_m[rows],
                         domain=state.domain, current_surface=current_surface, max_bytes=max_bytes)
    cand = state.copy()
    if event.n:
        try:
            cand.update_plants(event.plant_id, agb_dry_kg=res['B_new'], seasonal_agb_dry_kg=res['E_new'],
                               status=res['status_new'])
        except _state.StateError as exc:
            raise DisturbanceError(f'response rejected by the canonical state validation: {exc}') from exc
    return cand, res


__all__ = ['VERSION', 'LAW', 'BACKENDS', 'SOURCES', 'CAUSES', 'OUT_NONE', 'OUT_INJURED', 'OUT_DEAD', 'OUT_REMOVED',
           'OUTCOME_NAMES', 'LETHAL_BURIAL', 'LETHAL_EXPOSURE', 'LETHAL_ABRASION', 'CAVEATS', 'DisturbanceError',
           'DisturbanceBudgetError', 'DisturbanceBackendError', 'CompiledDisturbance', 'compile_species',
           'estimate_event_bytes', 'evaluate_event', 'respond_state']
