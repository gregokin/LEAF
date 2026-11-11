'''leaf.growth: PURE species demand and aboveground growth kernels (growth law `stepwat_requested_support_v1`).

Scope. One versioned, opt-in aboveground demand + growth module. It has NO dependence on the older benchmark
driver, the native SYRUP water package, a calendar/scheduler, recruitment, disturbance, mortality or root
plasticity. Imports: NumPy and `leaf.traits` only; `numba` / `cupy` are imported lazily, and only when that
backend is explicitly selected. There is no backend fallback, no device switching and no implicit host copy of a
population array. Roots/allocator and the native bridge belong to other modules.

Public surface (keyword-only after the table; names are stable across tests and docs/phase1-growth.md)
  compile_species(catalog, backend, max_bytes, *, device=None, extra_reserve_bytes=0) -> CompiledSpecies
  temperature_factor(temperature_c, t_min_c, t_opt_c, t_max_c)             cardinal (triangular) response
  species_response(table, *, month, temperature_c)                          [S] activity and temperature factor
  integral_increments(table, *, month, temperature_c, dt_days, year_days)   [S] Aint / Tint increments
  demand(table, *, B_kg, canopy_area_m2, species_index, live, month, temperature_c, pet_mm_per_day, dt_days,
         max_bytes, extra_reserve_bytes=0)
  grow(table, *, B_kg, E_kg, species_index, live, requested_m3, delivered_m3, activity_integral,
       temperature_integral, year_end, max_bytes, extra_reserve_bytes=0, overdelivery_rtol=OVERDELIVERY_RTOL)
  memory_estimate(operation, n_plants, n_species, backend, *, has_canopy_area=True, extra_reserve_bytes=0)
  table_memory(n_species, backend)

Species parameters. ALL scientific parameters live in each species record of the existing catalog (no separate
parameter map and no hard-coded species default):
  growth     {"law": "stepwat_requested_support_v1", "traits": {mature_biomass_kg, intrinsic_rate_per_year,
              biomass_support_kg_per_m3, live_fraction, extra_growth_fraction, monthly_activity[12],
              temperature_min_c, temperature_opt_c, temperature_max_c}}
  water_use  {"law": "canopy_pet_v1", "traits": {transpiration_coefficient}}  or
             {"law": "biomass_specific_pet_v1", "traits": {demand_m3_per_kg_day, reference_pet_mm_per_day}}
Compilation covers EVERY species of the catalog (sorted ids, exact hash recorded). A missing, null,
"unimplemented", unknown or old (`stepwat_weekly_aboveground_v0.1`) law, a missing/unknown/out-of-range trait
refuses the compilation naming the species. Phase 0 reading of unimplemented species is unaffected.

Backends. 'numpy' (chunked vectorised reference), 'numba' (compiled serial kernels) and 'cupy' (fused CUDA
kernels bound to ONE explicit device, compiled with --fmad=false so the arithmetic matches the host operation
order). All P-sized inputs must already be 1-D C-contiguous arrays of the exact dtype in the backend namespace
(np.ndarray for numpy/numba, cupy.ndarray on the table's device for cupy): B/E/canopy area/requested/delivered
float64, species_index int32 (index into the SORTED catalog ids), live bool. Nothing is converted, copied or
transferred implicitly; nothing is mutated. Only S-sized tables/forcing are transferred (explicitly, host ->
device) and only fixed-size diagnostic packets are read back (<= 256 x 17 float64 on the GPU).

Admission. Every call checks types/shapes/dtypes/device and the explicit `max_bytes` BEFORE any allocation,
transfer or kernel launch (`GrowthBudgetError`). The estimate is itemized (memory_estimate) and covers resident
inputs, compiled tables (host and device mirror), simultaneous outputs, forcing/validation scratch and reduction
packets, plus the caller's `extra_reserve_bytes`. It is a conservative structural bound; the Python
interpreter, Numba/NVRTC JIT caches, the CUDA context and the CuPy memory pool (rounding/caching) are
EXCLUDED. NumPy temporaries are bounded by a fixed 65536-plant chunk, not by P.

Content validation happens inside the kernels (finite, ranges, relations, overflow, ledger) and is reported
through the packet. On ANY failure the call raises `GrowthDataError` and returns nothing; inputs are never
mutated, so the caller's state is unchanged.

This module computes DOCUMENTED LEAF demand adaptations and a documented weekly-or-shorter Euler adaptation of
STEPWAT2 rgroup_Grow/_extra_growth (pinned revision 003cb9b4adbf13d64a39d6d62fd95280adac27fc; relative size,
biomass support, seasonal extra growth). They are not claims about STEPWAT daily physiology, whole-plant carbon
closure, root biomass or water-use-efficiency production. See docs/phase1-growth.md.
'''
from __future__ import annotations

import math

import numpy as np

from leaf import traits as _traits

VERSION = '1.0.0'
GROWTH_LAW = 'stepwat_requested_support_v1'
OLD_GROWTH_LAW = _traits.STEPWAT_GROWTH_LAW
DEMAND_CANOPY_PET = 'canopy_pet_v1'
DEMAND_BIOMASS_SPECIFIC_PET = 'biomass_specific_pet_v1'
DEMAND_LAWS = (DEMAND_CANOPY_PET, DEMAND_BIOMASS_SPECIFIC_PET)
LAW_CODE = {DEMAND_CANOPY_PET: 0, DEMAND_BIOMASS_SPECIFIC_PET: 1}
BACKENDS = ('numpy', 'numba', 'cupy')
SOURCE_REVISION = '003cb9b4adbf13d64a39d6d62fd95280adac27fc'

ABSOLUTE_ZERO_C = -273.15
MAX_PERIOD_DAYS = 7.0                      # weekly-or-shorter envelope of the Euler adaptation
MAX_ACTIVITY_INTEGRAL = 7.0 / 365.0        # Aint bound (leap-year fractions are smaller)
ACTIVITY_INTEGRAL_SLACK = 1e-12            # relative slack on the Aint bound (FP64 spelling of 7/365)
GMOD_SCALE = 0.95                          # STEP gmod = 0.95 * temperature modifier / max(PR, 1)
OVERDELIVERY_RTOL = 1e-9                   # allowed relative FP64 over-delivery U <= D*(1 + rtol); then S = 1
MAX_OVERDELIVERY_RTOL = 1e-6
LEDGER_EPS_FACTOR = 16.0                   # per-plant ledger bound = 16 * eps * (B + Bnew + gains + loss)
FP64_EPS = 2.220446049250313e-16

TABLE_HOST_BYTES_PER_SPECIES = 8 * 10 + 8 * 12 + 4   # 10 f64 vectors + activity[S,12] + int32 law code = 180
TABLE_DEVICE_BYTES_PER_SPECIES = 8 * 5 + 4           # 5 f64 grow vectors + int32 law code = 44
NUMPY_CHUNK = 1 << 16
NUMPY_DEMAND_CHUNK_BYTES_PER_PLANT = 128   # conservative bound on live chunk temporaries (see _demand_numpy)
NUMPY_GROW_CHUNK_BYTES_PER_PLANT = 512     # conservative bound on live chunk temporaries (see _grow_numpy)
CUDA_THREADS = 256
CUDA_MAX_BLOCKS = 256

DEMAND_PACKET = ('bad_species_index', 'bad_input', 'bad_output', 'n_live', 'n_demanding', 'sum_requested_m3',
                 'max_requested_m3')
DEMAND_MAX_SLOTS = (6,)
GROW_PACKET = ('bad_species_index', 'bad_input', 'bad_water', 'bad_output', 'ledger_violations',
               'clamped_overdelivery', 'n_live', 'n_growing', 'sum_B_kg', 'sum_B_new_kg', 'sum_normal_gain_kg',
               'sum_extra_gain_kg', 'sum_seasonal_loss_kg', 'sum_requested_m3', 'sum_delivered_m3',
               'sum_maintenance_deficit_kg', 'max_ledger_residual_kg')
GROW_MAX_SLOTS = (16,)
GROW_OUTPUTS = ('B_new_kg', 'E_new_kg', 'normal_plan_kg', 'extra_plan_kg', 'normal_gain_kg', 'extra_gain_kg',
                'seasonal_loss_kg', 'satisfaction', 'required_support_kg', 'requested_support_kg',
                'actual_support_kg', 'maintenance_deficit_kg')
_COUNT_PREFIXES = ('bad_', 'n_', 'ledger_violations', 'clamped_')

GROWTH_TRAITS = ('mature_biomass_kg', 'intrinsic_rate_per_year', 'biomass_support_kg_per_m3', 'live_fraction',
                 'extra_growth_fraction', 'monthly_activity', 'temperature_min_c', 'temperature_opt_c',
                 'temperature_max_c')
DEMAND_TRAITS = {DEMAND_CANOPY_PET: ('transpiration_coefficient',),
                 DEMAND_BIOMASS_SPECIFIC_PET: ('demand_m3_per_kg_day', 'reference_pet_mm_per_day')}
_SECTION_KEYS = ('law', 'traits', 'status', 'notes', 'units')

DEMAND_FORMULAS = {
    DEMAND_CANOPY_PET: 'D[m3] = canopy_area_m2 * PET_mm_per_day * 0.001 * dt_days * transpiration_coefficient * '
                       'live_fraction * activity * temperature_factor',
    DEMAND_BIOMASS_SPECIFIC_PET: 'D[m3] = B_kg_dry_AGB * demand_m3_per_kg_day * (PET_mm_per_day / '
                                 'reference_pet_mm_per_day) * dt_days * activity * temperature_factor '
                                 '(per TOTAL dry AGB: live_fraction is NOT applied again)'}
CAVEATS = (
    'DOCUMENTED LEAF demand adaptations, not a claim about STEPWAT daily physiology; coefficients are '
    'caller/catalog supplied (the shipped example catalog holds uncalibrated FIXTURES, not literature fits)',
    'weekly-or-shorter explicit Euler adaptation of the annual STEP rule; intrinsic_rate_per_year is bounded to '
    '[0, 1]. Earlier fitted rates outside that range (e.g. the study grass rate 295/year) are NOT admitted by '
    'raising the bound: they need a different solver and requalification',
    'k = biomass_support_kg_per_m3 is annual biomass SUPPORTED per m3 of water (kg*year/m3), not kg new biomass '
    'per m3 and not a water-use efficiency; required support is the matching annual-support equivalent prorated '
    'to the period; no fixed physical percentage of water is assigned to maintenance',
    'only the aboveground ledger Bnew == B + gains - seasonal_loss is claimed; no whole-plant carbon closure, '
    'root biomass pool or production yield; seasonal loss is STEP-style bookkeeping, not stress mortality',
    'canopy area is supplied by the geometry boundary (leaf.canopy); biomass is never assumed proportional to '
    'area; root-overlapping canopies are not capped by cell area and demand is not silently altered',
    'PET, month, temperature and the interval integrals are supplied by the scheduler; this module computes '
    'none of them, applies no leap-year rule and replaces no missing value by a default',
)


class GrowthError(ValueError):
    '''Any growth/demand specification or argument error.'''


class GrowthBudgetError(GrowthError):
    '''The explicit max_bytes bound would be exceeded; raised before any copy, transfer or allocation.'''


class GrowthDeviceError(GrowthError):
    '''The CUDA device binding of a table/array differs from the current device (no switching, no fallback).'''


class GrowthDataError(GrowthError):
    '''Invalid or non-finite population/water/forcing content (inputs are unchanged; nothing is returned).'''


# ------------------------------------------------------------------ scalar validators
def _num(v, name):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
        raise GrowthError(f'{name} must be a real number (not bool/str/None/array), got {v!r}')
    f = float(v)
    if not math.isfinite(f):
        raise GrowthError(f'{name} must be finite, got {f}')
    return f


def _strict_int(v, name, lo):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
        raise GrowthError(f'{name} must be an integer, got {v!r}')
    v = int(v)
    if v < lo:
        raise GrowthError(f'{name} must be >= {lo}, got {v}')
    return v


def _strict_bool(v, name):
    if not isinstance(v, (bool, np.bool_)):
        raise GrowthError(f'{name} must be a bool, got {v!r}')
    return bool(v)


def _month(month):
    m = _strict_int(month, 'month', 1)
    if m > 12:
        raise GrowthError(f'month must be an integer in 1..12 (actual supplied calendar month), got {m}')
    return m


def _temperature_c(t):
    t = _num(t, 'temperature_c')
    if t < ABSOLUTE_ZERO_C:
        raise GrowthError(f'temperature_c {t} is below absolute zero ({ABSOLUTE_ZERO_C} C)')
    return t


def _pet(v):
    v = _num(v, 'pet_mm_per_day')
    if v < 0.0:
        raise GrowthError(f'pet_mm_per_day must be >= 0 (shared scheduler PET is required), got {v}')
    return v


def _dt(v):
    v = _num(v, 'dt_days')
    if not 0.0 < v <= MAX_PERIOD_DAYS:
        raise GrowthError(f'dt_days must lie in (0, {MAX_PERIOD_DAYS}] (callers split longer periods), got {v}')
    return v


# ------------------------------------------------------------------ species parsing (pure, S-sized)
def _section_traits(sec, section, laws, sid):
    where = f'species {sid!r} {section}'
    if not isinstance(sec, dict):
        raise GrowthError(f'{where} is null/missing: leaf.growth needs an explicit section with law in {list(laws)}')
    law = sec.get('law')
    if law == _traits.UNIMPLEMENTED:
        raise GrowthError(f'{where}.law is "unimplemented": select one of {list(laws)} explicitly')
    if law not in laws:
        hint = (f' (the older {OLD_GROWTH_LAW!r} has different traits and is not accepted here)'
                if law == OLD_GROWTH_LAW else '')
        raise GrowthError(f'{where}.law {law!r} is unknown to leaf.growth (supported: {list(laws)}){hint}')
    unknown = sorted(set(sec) - set(_SECTION_KEYS))
    if unknown:
        raise GrowthError(f'{where}: unknown keys {unknown} (allowed: {list(_SECTION_KEYS)})')
    tr = sec.get('traits')
    if not isinstance(tr, dict):
        raise GrowthError(f'{where}.traits must be an object holding the explicit parameters')
    return law, tr


def _exact_keys(tr, required, where):
    missing = [k for k in required if k not in tr]
    unknown = sorted(set(tr) - set(required))
    if missing or unknown:
        raise GrowthError(f'{where}.traits must hold exactly {list(required)}; missing {missing}, unknown {unknown}')


def _parse_growth(sec, sid):
    _, tr = _section_traits(sec, 'growth', (GROWTH_LAW,), sid)
    where = f'species {sid!r} growth'
    _exact_keys(tr, GROWTH_TRAITS, where)
    out = {'mature': _traits.positive(tr['mature_biomass_kg'], f'{where}.mature_biomass_kg')}
    rate = _traits.real(tr['intrinsic_rate_per_year'], f'{where}.intrinsic_rate_per_year')
    if not 0.0 <= rate <= 1.0:
        raise GrowthError(f'{where}.intrinsic_rate_per_year must lie in [0, 1] (the qualified weekly-Euler range), '
                          f'got {rate}; fitted rates above 1 (e.g. 295/year) are NOT admitted by raising the bound: '
                          'another solver and requalification are separate work')
    out['rate'] = rate
    out['support'] = _traits.positive(tr['biomass_support_kg_per_m3'], f'{where}.biomass_support_kg_per_m3')
    out['live_fraction'] = _traits.unit_interval(tr['live_fraction'], f'{where}.live_fraction')
    out['extra_fraction'] = _traits.unit_interval(tr['extra_growth_fraction'], f'{where}.extra_growth_fraction')
    act = tr['monthly_activity']
    if not isinstance(act, (list, tuple)) or len(act) != 12:
        raise GrowthError(f'{where}.monthly_activity must be a list of exactly 12 fractions (January..December)')
    out['activity'] = [_traits.unit_interval(a, f'{where}.monthly_activity[{i}]') for i, a in enumerate(act)]
    tmin = _traits.real(tr['temperature_min_c'], f'{where}.temperature_min_c')
    topt = _traits.real(tr['temperature_opt_c'], f'{where}.temperature_opt_c')
    tmax = _traits.real(tr['temperature_max_c'], f'{where}.temperature_max_c')
    if tmin < ABSOLUTE_ZERO_C:
        raise GrowthError(f'{where}.temperature_min_c must be >= absolute zero ({ABSOLUTE_ZERO_C})')
    if not tmin < topt < tmax:
        raise GrowthError(f'{where} needs temperature_min_c < temperature_opt_c < temperature_max_c, '
                          f'got {tmin}, {topt}, {tmax}')
    out['tmin'], out['topt'], out['tmax'] = tmin, topt, tmax
    return out


def _parse_water(sec, sid):
    law, tr = _section_traits(sec, 'water_use', DEMAND_LAWS, sid)
    where = f'species {sid!r} water_use'
    _exact_keys(tr, DEMAND_TRAITS[law], where)
    if law == DEMAND_CANOPY_PET:
        return {'law': law, 'a': _traits.nonnegative(tr['transpiration_coefficient'],
                                                      f'{where}.transpiration_coefficient'), 'b': 0.0}
    return {'law': law, 'a': _traits.nonnegative(tr['demand_m3_per_kg_day'], f'{where}.demand_m3_per_kg_day'),
            'b': _traits.positive(tr['reference_pet_mm_per_day'], f'{where}.reference_pet_mm_per_day')}


def _parse_species(catalog):
    ids = catalog.species_ids
    S = len(ids)
    host = {k: np.zeros(S, dtype=np.float64) for k in ('mature', 'rate', 'support', 'live_fraction', 'extra_fraction',
                                                       'tmin', 'topt', 'tmax', 'demand_a', 'demand_b')}
    host['activity'] = np.zeros((S, 12), dtype=np.float64)
    host['law_code'] = np.zeros(S, dtype=np.int32)
    laws, errors = [], []
    for i, sid in enumerate(ids):
        try:
            g = _parse_growth(catalog.section(sid, 'growth'), sid)
            w = _parse_water(catalog.section(sid, 'water_use'), sid)
        except (GrowthError, _traits.CatalogError) as exc:
            errors.append(str(exc))
            laws.append(None)
            continue
        for k in ('mature', 'rate', 'support', 'live_fraction', 'extra_fraction', 'tmin', 'topt', 'tmax'):
            host[k][i] = g[k]
        host['activity'][i, :] = g['activity']
        host['demand_a'][i], host['demand_b'][i] = w['a'], w['b']
        host['law_code'][i] = LAW_CODE[w['law']]
        laws.append(w['law'])
    if errors:
        more = f' (+{len(errors) - 6} more)' if len(errors) > 6 else ''
        raise GrowthError('species catalog is not usable by leaf.growth: ' + ' | '.join(errors[:6]) + more)
    for a in host.values():
        a.setflags(write=False)
    return host, tuple(laws)


# ------------------------------------------------------------------ backends / devices (lazy, explicit)
def _cupy():
    try:
        import cupy
    except Exception as exc:   # pragma: no cover - environment dependent
        raise GrowthError(f'backend "cupy" needs the CuPy package: {exc}') from exc
    return cupy


def _current_device_id():
    return int(_cupy().cuda.runtime.getDevice())


def _array_device_id(arr):
    return int(arr.device.id)


def _require_current_device(table):
    '''Exact device binding: the CURRENT CUDA device must be the table's device. Never switches devices.'''
    if table.backend != 'cupy':
        return
    cur = _current_device_id()
    if cur != table.device:
        raise GrowthDeviceError(f'table is bound to CUDA device {table.device} but the current device is {cur}; '
                                'this module never switches devices or falls back (select the device in the caller)')


_KERNELS = {}


def _numba_kernels():
    '''Compiled serial kernels; JIT happens at first use of each signature (not cached on disk).'''
    if 'numba' in _KERNELS:
        return _KERNELS['numba']
    try:
        import numba
    except Exception as exc:   # pragma: no cover - environment dependent
        raise GrowthError(f'backend "numba" needs the numba package: {exc}') from exc
    njit = numba.njit

    @njit(cache=False)
    def demand_cpu(B, area, sidx, live, krate, law, out, packet):
        P = B.shape[0]
        S = krate.shape[0]
        for p in range(P):
            lv = live[p]
            s = sidx[p]
            bad_index = 0
            if s < 0 or s >= S:
                bad_index = 1
                s = 0
            bad_input = 0
            dval = 0.0
            if lv:
                b = B[p]
                a = area[p]
                if (not math.isfinite(b)) or b <= 0.0:
                    bad_input = 1
                if law[s] == 0:
                    if (not math.isfinite(a)) or a < 0.0:
                        bad_input = 1
                    dval = a * krate[s]
                else:
                    dval = b * krate[s]
            bad_output = 0
            if (not math.isfinite(dval)) or dval < 0.0:
                bad_output = 1
            out[p] = dval
            packet[0] += bad_index
            packet[1] += bad_input
            packet[2] += bad_output
            if lv:
                packet[3] += 1.0
            if dval > 0.0:
                packet[4] += 1.0
            packet[5] += dval
            if dval > packet[6]:
                packet[6] = dval

    @njit(cache=False)
    def grow_cpu(B, E, sidx, live, D, U, mature, rate, support, lfrac, xfrac, A, T, year_end, lim, out, packet):
        P = B.shape[0]
        S = mature.shape[0]
        for p in range(P):
            b = B[p]
            e = E[p]
            d = D[p]
            u = U[p]
            lv = live[p]
            s = sidx[p]
            bad_index = 0
            if s < 0 or s >= S:
                bad_index = 1
                s = 0
            bad_input = 0
            bad_water = 0
            bad_output = 0
            ledger_bad = 0
            clamped = 0
            if (not math.isfinite(b)) or (not math.isfinite(e)) or b < 0.0 or e < 0.0:
                bad_input = 1
            if lv and (b <= 0.0 or e > b):
                bad_input = 1
            if (not math.isfinite(d)) or (not math.isfinite(u)) or d < 0.0 or u < 0.0:
                bad_water = 1
            if (not lv) and (d != 0.0 or u != 0.0):
                bad_water = 1
            if lv and d == 0.0 and u != 0.0:
                bad_water = 1
            if lv and d > 0.0 and u > d * lim:
                bad_water = 1
            nplan = 0.0
            xplan = 0.0
            ng = 0.0
            eg = 0.0
            loss = 0.0
            sat = 0.0
            required = 0.0
            reqsup = 0.0
            actsup = 0.0
            deficit = 0.0
            bnew = b
            enew = e
            if lv:
                m = mature[s]
                persistent = b - e
                a = A[s]
                t = T[s]
                required = b * lfrac[s] * a
                reqsup = support[s] * d
                actsup = support[s] * u
                fplan = 0.0
                if required > 0.0:
                    fplan = reqsup / required
                    if fplan > 1.0:
                        fplan = 1.0
                nplan = 0.95 * rate[s] * t * fplan * persistent * (1.0 - persistent / m)
                if nplan < 0.0:
                    nplan = 0.0
                room = m - b
                if room < 0.0:
                    room = 0.0
                if nplan > room:
                    nplan = room
                meant = 0.0
                if a > 0.0:
                    meant = t / a
                surplus = reqsup - required
                if surplus < 0.0:
                    surplus = 0.0
                xplan = xfrac[s] * surplus * meant
                room2 = m - b - nplan
                if room2 < 0.0:
                    room2 = 0.0
                if xplan > room2:
                    xplan = room2
                if d > 0.0:
                    sat = u / d
                    if sat > 1.0:
                        sat = 1.0
                    if u > d:
                        clamped = 1
                ng = nplan * sat
                eg = xplan * sat
                deficit = required - actsup
                if deficit < 0.0:
                    deficit = 0.0
                if year_end != 0:
                    loss = e + eg
                    enew = 0.0
                    bnew = persistent + ng
                else:
                    enew = e + eg
                    bnew = persistent + ng + enew
            if not (math.isfinite(bnew) and math.isfinite(enew) and math.isfinite(nplan) and math.isfinite(xplan)
                    and math.isfinite(ng) and math.isfinite(eg) and math.isfinite(loss) and math.isfinite(sat)
                    and math.isfinite(required) and math.isfinite(reqsup) and math.isfinite(actsup)
                    and math.isfinite(deficit)):
                bad_output = 1
            if enew < 0.0 or bnew < 0.0 or enew > bnew:
                bad_output = 1
            ref = ((b + ng) + eg) - loss
            resid = abs(bnew - ref)
            tol = 16.0 * 2.220446049250313e-16 * (b + bnew + ng + eg + loss)
            if resid > tol:
                ledger_bad = 1
            out[0, p] = bnew
            out[1, p] = enew
            out[2, p] = nplan
            out[3, p] = xplan
            out[4, p] = ng
            out[5, p] = eg
            out[6, p] = loss
            out[7, p] = sat
            out[8, p] = required
            out[9, p] = reqsup
            out[10, p] = actsup
            out[11, p] = deficit
            packet[0] += bad_index
            packet[1] += bad_input
            packet[2] += bad_water
            packet[3] += bad_output
            packet[4] += ledger_bad
            packet[5] += clamped
            if lv:
                packet[6] += 1.0
            if ng + eg > 0.0:
                packet[7] += 1.0
            packet[8] += b
            packet[9] += bnew
            packet[10] += ng
            packet[11] += eg
            packet[12] += loss
            packet[13] += d
            packet[14] += u
            packet[15] += deficit
            if resid > packet[16]:
                packet[16] = resid

    _KERNELS['numba'] = (demand_cpu, grow_cpu)
    return _KERNELS['numba']


_CUDA_SRC = r'''
extern "C" {
#define NT 256
#define KD 7
#define KG 17
__device__ __forceinline__ bool fin(double x) { return (x - x) == 0.0; }

__global__ void leaf_growth_demand(const double* B, const double* area, const int* sidx,
                                   const unsigned char* live, const double* krate, const int* law,
                                   double* out, double* part, const int S, const long long P) {
    __shared__ double sh[NT];
    double acc[KD];
    for (int j = 0; j < KD; ++j) acc[j] = 0.0;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x; p < P; p += stride) {
        const bool lv = live[p] != 0;
        int s = sidx[p];
        int bad_index = 0, bad_input = 0, bad_output = 0;
        if (s < 0 || s >= S) { bad_index = 1; s = 0; }
        double dval = 0.0;
        if (lv) {
            const double b = B[p];
            const double a = area[p];
            if (!fin(b) || b <= 0.0) bad_input = 1;
            if (law[s] == 0) {
                if (!fin(a) || a < 0.0) bad_input = 1;
                dval = a * krate[s];
            } else {
                dval = b * krate[s];
            }
        }
        if (!fin(dval) || dval < 0.0) bad_output = 1;
        out[p] = dval;
        acc[0] += bad_index; acc[1] += bad_input; acc[2] += bad_output;
        if (lv) acc[3] += 1.0;
        if (dval > 0.0) acc[4] += 1.0;
        acc[5] += dval;
        if (dval > acc[6]) acc[6] = dval;
    }
    for (int j = 0; j < KD; ++j) {
        sh[threadIdx.x] = acc[j];
        __syncthreads();
        for (int off = NT / 2; off > 0; off >>= 1) {
            if (threadIdx.x < off) {
                if (j == 6) sh[threadIdx.x] = fmax(sh[threadIdx.x], sh[threadIdx.x + off]);
                else sh[threadIdx.x] += sh[threadIdx.x + off];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) part[(long long)blockIdx.x * KD + j] = sh[0];
        __syncthreads();
    }
}

__global__ void leaf_growth_grow(const double* B, const double* E, const int* sidx, const unsigned char* live,
                                 const double* D, const double* U, const double* mature, const double* rate,
                                 const double* support, const double* lfrac, const double* xfrac,
                                 const double* A, const double* T, const int year_end, const double lim,
                                 const int S, double* out, double* part, const long long P) {
    __shared__ double sh[NT];
    double acc[KG];
    for (int j = 0; j < KG; ++j) acc[j] = 0.0;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x; p < P; p += stride) {
        const double b = B[p], e = E[p], d = D[p], u = U[p];
        const bool lv = live[p] != 0;
        int s = sidx[p];
        int bad_index = 0, bad_input = 0, bad_water = 0, bad_output = 0, ledger_bad = 0, clamped = 0;
        if (s < 0 || s >= S) { bad_index = 1; s = 0; }
        if (!fin(b) || !fin(e) || b < 0.0 || e < 0.0) bad_input = 1;
        if (lv && (b <= 0.0 || e > b)) bad_input = 1;
        if (!fin(d) || !fin(u) || d < 0.0 || u < 0.0) bad_water = 1;
        if (!lv && (d != 0.0 || u != 0.0)) bad_water = 1;
        if (lv && d == 0.0 && u != 0.0) bad_water = 1;
        if (lv && d > 0.0 && u > d * lim) bad_water = 1;
        double nplan = 0.0, xplan = 0.0, ng = 0.0, eg = 0.0, loss = 0.0, sat = 0.0;
        double required = 0.0, reqsup = 0.0, actsup = 0.0, deficit = 0.0, bnew = b, enew = e;
        if (lv) {
            const double m = mature[s];
            const double persistent = b - e;
            const double a = A[s];
            const double t = T[s];
            required = b * lfrac[s] * a;
            reqsup = support[s] * d;
            actsup = support[s] * u;
            double fplan = 0.0;
            if (required > 0.0) { fplan = reqsup / required; if (fplan > 1.0) fplan = 1.0; }
            nplan = 0.95 * rate[s] * t * fplan * persistent * (1.0 - persistent / m);
            if (nplan < 0.0) nplan = 0.0;
            double room = m - b; if (room < 0.0) room = 0.0;
            if (nplan > room) nplan = room;
            double meant = 0.0;
            if (a > 0.0) meant = t / a;
            double surplus = reqsup - required; if (surplus < 0.0) surplus = 0.0;
            xplan = xfrac[s] * surplus * meant;
            double room2 = m - b - nplan; if (room2 < 0.0) room2 = 0.0;
            if (xplan > room2) xplan = room2;
            if (d > 0.0) {
                sat = u / d;
                if (sat > 1.0) sat = 1.0;
                if (u > d) clamped = 1;
            }
            ng = nplan * sat;
            eg = xplan * sat;
            deficit = required - actsup; if (deficit < 0.0) deficit = 0.0;
            if (year_end != 0) { loss = e + eg; enew = 0.0; bnew = persistent + ng; }
            else { enew = e + eg; bnew = persistent + ng + enew; }
        }
        if (!(fin(bnew) && fin(enew) && fin(nplan) && fin(xplan) && fin(ng) && fin(eg) && fin(loss) && fin(sat)
              && fin(required) && fin(reqsup) && fin(actsup) && fin(deficit))) bad_output = 1;
        if (enew < 0.0 || bnew < 0.0 || enew > bnew) bad_output = 1;
        const double ref = ((b + ng) + eg) - loss;
        const double resid = fabs(bnew - ref);
        const double tol = 16.0 * 2.220446049250313e-16 * (b + bnew + ng + eg + loss);
        if (resid > tol) ledger_bad = 1;
        out[p] = bnew;
        out[P + p] = enew;
        out[2 * P + p] = nplan;
        out[3 * P + p] = xplan;
        out[4 * P + p] = ng;
        out[5 * P + p] = eg;
        out[6 * P + p] = loss;
        out[7 * P + p] = sat;
        out[8 * P + p] = required;
        out[9 * P + p] = reqsup;
        out[10 * P + p] = actsup;
        out[11 * P + p] = deficit;
        acc[0] += bad_index; acc[1] += bad_input; acc[2] += bad_water; acc[3] += bad_output;
        acc[4] += ledger_bad; acc[5] += clamped;
        if (lv) acc[6] += 1.0;
        if (ng + eg > 0.0) acc[7] += 1.0;
        acc[8] += b; acc[9] += bnew; acc[10] += ng; acc[11] += eg; acc[12] += loss;
        acc[13] += d; acc[14] += u; acc[15] += deficit;
        if (resid > acc[16]) acc[16] = resid;
    }
    for (int j = 0; j < KG; ++j) {
        sh[threadIdx.x] = acc[j];
        __syncthreads();
        for (int off = NT / 2; off > 0; off >>= 1) {
            if (threadIdx.x < off) {
                if (j == 16) sh[threadIdx.x] = fmax(sh[threadIdx.x], sh[threadIdx.x + off]);
                else sh[threadIdx.x] += sh[threadIdx.x + off];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) part[(long long)blockIdx.x * KG + j] = sh[0];
        __syncthreads();
    }
}
}
'''


def _cuda_kernels(device):
    key = ('cupy', int(device))
    if key not in _KERNELS:
        cp = _cupy()
        opts = ('-std=c++11', '--fmad=false')
        _KERNELS[key] = (cp.RawKernel(_CUDA_SRC, 'leaf_growth_demand', options=opts),
                         cp.RawKernel(_CUDA_SRC, 'leaf_growth_grow', options=opts))
    return _KERNELS[key]


def _n_blocks(P):
    return max(1, min(-(-int(P) // CUDA_THREADS), CUDA_MAX_BLOCKS))


# ------------------------------------------------------------------ memory model (pure arithmetic)
def table_memory(n_species, backend):
    '''Itemized bytes of the compiled species tables. Host tables always exist (forcing is computed there); on
    the cupy backend a device mirror of the five grow vectors and the int32 law code is added.'''
    S = _strict_int(n_species, 'n_species', 1)
    if backend not in BACKENDS:
        raise GrowthError(f'backend must be one of {BACKENDS}, got {backend!r}')
    items = [{'name': 'table.host (10 f64 S-vectors + activity[S,12] + int32 law)',
              'bytes': TABLE_HOST_BYTES_PER_SPECIES * S, 'where': 'host'}]
    if backend == 'cupy':
        items.append({'name': 'table.device (5 f64 S-vectors + int32 law)',
                      'bytes': TABLE_DEVICE_BYTES_PER_SPECIES * S, 'where': 'device'})
    return {'items': items, 'total_bytes': int(sum(i['bytes'] for i in items))}


def memory_estimate(operation, n_plants, n_species, backend, *, has_canopy_area=True, extra_reserve_bytes=0):
    '''Itemized conservative byte estimate of one call (resident inputs + tables + outputs + scratch + packets +
    reserve). `where` is 'host' or 'device': P-sized arrays live on the compute side (device for cupy), S-sized
    forcing and packets are host (plus a device copy / partial-sum buffer on cupy). Excluded: interpreter,
    Numba/NVRTC JIT, CUDA context, CuPy memory-pool rounding. Pure arithmetic; allocates nothing.'''
    if operation not in ('demand', 'grow'):
        raise GrowthError("operation must be 'demand' or 'grow'")
    P = _strict_int(n_plants, 'n_plants', 0)
    S = _strict_int(n_species, 'n_species', 1)
    res = _strict_int(extra_reserve_bytes, 'extra_reserve_bytes', 0)
    gpu = backend == 'cupy'
    pw = 'device' if gpu else 'host'
    items = [dict(i, kind='table') for i in table_memory(S, backend)['items']]

    def add(name, nbytes, where, kind):
        items.append({'name': name, 'bytes': int(nbytes), 'where': where, 'kind': kind})

    kk = len(DEMAND_PACKET) if operation == 'demand' else len(GROW_PACKET)
    if operation == 'demand':
        add('input B_kg (f64)', 8 * P, pw, 'resident_input')
        if has_canopy_area:
            add('input canopy_area_m2 (f64)', 8 * P, pw, 'resident_input')
        add('input species_index (i32)', 4 * P, pw, 'resident_input')
        add('input live (bool)', P, pw, 'resident_input')
        add('output requested_m3 (f64)', 8 * P, pw, 'output')
        add('forcing S-vectors: activity, temperature_factor, krate (f64, host)', 24 * S, 'host', 'scratch')
        add('forcing temporaries (response arithmetic, host)', 48 * S, 'host', 'scratch')
        if gpu:
            add('krate device copy (f64)', 8 * S, 'device', 'scratch')
        chunk = NUMPY_DEMAND_CHUNK_BYTES_PER_PLANT
    else:
        for n in ('B_kg', 'E_kg', 'requested_m3', 'delivered_m3'):
            add(f'input {n} (f64)', 8 * P, pw, 'resident_input')
        add('input species_index (i32)', 4 * P, pw, 'resident_input')
        add('input live (bool)', P, pw, 'resident_input')
        add('input activity_integral, temperature_integral (f64 [S], host)', 16 * S, 'host', 'resident_input')
        add(f'outputs {len(GROW_OUTPUTS)} x f64[P] (one [12,P] buffer)', 8 * len(GROW_OUTPUTS) * P, pw, 'output')
        if gpu:
            add('integrals device copies (f64 [S] x2)', 16 * S, 'device', 'scratch')
        chunk = NUMPY_GROW_CHUNK_BYTES_PER_PLANT
    if backend == 'numpy':
        add(f'numpy chunk temporaries (<= {chunk} B x min(P, {NUMPY_CHUNK}))', chunk * min(P, NUMPY_CHUNK), 'host',
            'scratch')
    if gpu:
        nb = _n_blocks(P)
        add(f'reduction partials [{nb},{kk}] f64', 8 * nb * kk, 'device', 'packet')
        add(f'packet readback [{nb},{kk}] f64', 8 * nb * kk, 'host', 'packet')
    else:
        add(f'packet [{kk}] f64', 8 * kk, 'host', 'packet')
    add('caller extra_reserve_bytes', res, 'caller', 'reserve')
    host = sum(i['bytes'] for i in items if i['where'] in ('host', 'caller'))
    dev = sum(i['bytes'] for i in items if i['where'] == 'device')
    return {'operation': operation, 'backend': backend, 'n_plants': P, 'n_species': S, 'items': items,
            'host_bytes': int(host), 'device_bytes': int(dev), 'total_bytes': int(host + dev),
            'allocating_bytes': int(sum(i['bytes'] for i in items if i['kind'] in ('output', 'scratch', 'packet'))),
            'exclusions': 'Python interpreter, Numba/NVRTC JIT caches, CUDA context and CuPy memory-pool rounding '
                          'or caching are not counted; the bound is structural (O(P + S), no grid/graph/history)'}


def _admit(mem, max_bytes):
    if mem['total_bytes'] > max_bytes:
        top = sorted(mem['items'], key=lambda i: -i['bytes'])[:4]
        raise GrowthBudgetError(
            f"{mem['operation']} needs {mem['total_bytes']} bytes (host {mem['host_bytes']}, device "
            f"{mem['device_bytes']}) > max_bytes={max_bytes}; largest items: "
            + '; '.join(f"{i['name']}={i['bytes']}" for i in top) + '. Refused before any copy or allocation')


# ------------------------------------------------------------------ compiled species table
class CompiledSpecies:
    '''Immutable S-sized demand/growth tables compiled from a SpeciesCatalog (sorted ids, exact hash).

    Attributes: backend, device (cupy only), catalog_hash, catalog_id, species_ids, n_species, demand_laws,
    host (read-only float64/int32 arrays), dev (device mirrors on cupy; the host arrays otherwise).'''

    def __init__(self, backend, device, catalog_hash, catalog_id, species_ids, host, dev, demand_laws):
        self.backend, self.device = backend, device
        self.catalog_hash, self.catalog_id = catalog_hash, catalog_id
        self.species_ids = tuple(species_ids)
        self.host, self.dev = host, dev
        self.demand_laws = tuple(demand_laws)
        self.growth_law = GROWTH_LAW

    @property
    def n_species(self):
        return len(self.species_ids)

    def memory(self):
        return table_memory(self.n_species, self.backend)

    def report(self):
        h = self.host
        per = {}
        for i, sid in enumerate(self.species_ids):
            per[sid] = {'demand_law': self.demand_laws[i], 'mature_biomass_kg': float(h['mature'][i]),
                        'intrinsic_rate_per_year': float(h['rate'][i]),
                        'biomass_support_kg_per_m3': float(h['support'][i]),
                        'live_fraction': float(h['live_fraction'][i]),
                        'extra_growth_fraction': float(h['extra_fraction'][i]),
                        'monthly_activity': [float(a) for a in h['activity'][i]],
                        'temperature_min_c': float(h['tmin'][i]), 'temperature_opt_c': float(h['topt'][i]),
                        'temperature_max_c': float(h['tmax'][i]),
                        'demand_primary': float(h['demand_a'][i]), 'demand_reference_pet_mm_per_day':
                            (float(h['demand_b'][i]) if self.demand_laws[i] == DEMAND_BIOMASS_SPECIFIC_PET else None)}
        return {'module': 'leaf.growth', 'version': VERSION, 'growth_law': GROWTH_LAW, 'backend': self.backend,
                'device': self.device, 'catalog_hash': self.catalog_hash, 'catalog_id': self.catalog_id,
                'species_ids': list(self.species_ids), 'per_species': per, 'memory': self.memory(),
                'demand_formulas': dict(DEMAND_FORMULAS), 'caveats': list(CAVEATS)}


def _check_table(table):
    if not isinstance(table, CompiledSpecies):
        raise GrowthError('table must be a leaf.growth.CompiledSpecies (compile_species)')


def check_catalog(table, catalog):
    '''Refuse a catalog whose exact hash or sorted species table differs from the compiled table.'''
    _check_table(table)
    if not isinstance(catalog, _traits.SpeciesCatalog):
        raise GrowthError('catalog must be a leaf.traits.SpeciesCatalog')
    if catalog.hash != table.catalog_hash or tuple(catalog.species_ids) != table.species_ids:
        raise GrowthError(f'catalog hash {catalog.hash} / species table differs from the compiled table '
                          f'({table.catalog_hash})')
    return True


def compile_species(catalog, backend, max_bytes, *, device=None, extra_reserve_bytes=0):
    '''Compile and validate the growth/demand tables of EVERY species (strict opt-in; see module docstring).

    `backend` is explicit ('numpy' | 'numba' | 'cupy'); there is no default and no fallback. 'cupy' requires an
    integer `device` that must equal the CURRENT CUDA device (never switched); the tables are bound to it.
    `max_bytes` bounds the table bytes (+ extra_reserve_bytes) and is checked before any parsing/allocation.'''
    if not isinstance(catalog, _traits.SpeciesCatalog):
        raise GrowthError('catalog must be a leaf.traits.SpeciesCatalog')
    if backend not in BACKENDS:
        raise GrowthError(f'backend must be explicitly one of {BACKENDS}, got {backend!r}')
    mb = _strict_int(max_bytes, 'max_bytes', 1)
    res = _strict_int(extra_reserve_bytes, 'extra_reserve_bytes', 0)
    if backend == 'cupy':
        if device is None:
            raise GrowthError('backend "cupy" requires an explicit integer device (exact binding, no fallback)')
        device = _strict_int(device, 'device', 0)
    elif device is not None:
        raise GrowthError(f'device is only meaningful for backend "cupy" (got backend {backend!r})')
    tm = table_memory(catalog.n_species, backend)
    if tm['total_bytes'] + res > mb:
        raise GrowthBudgetError(f"species tables need {tm['total_bytes']} + reserve {res} bytes > max_bytes={mb}; "
                                'refused before parsing or allocation')
    if backend == 'numba':
        _numba_kernels()
    elif backend == 'cupy':
        cur = _current_device_id()
        if cur != device:
            raise GrowthDeviceError(f'requested CUDA device {device} but the current device is {cur}; '
                                    'this module never switches devices')
    host, laws = _parse_species(catalog)
    if backend == 'cupy':
        cp = _cupy()
        dev = {k: cp.asarray(host[k]) for k in ('mature', 'rate', 'support', 'live_fraction', 'extra_fraction',
                                                'law_code')}
    else:
        dev = host
    doc = catalog.metadata()
    return CompiledSpecies(backend, device, catalog.hash, doc.get('catalog_id'), catalog.species_ids, host, dev, laws)


# ------------------------------------------------------------------ temperature / activity helpers
def temperature_factor(temperature_c, t_min_c, t_opt_c, t_max_c):
    '''Cardinal temperature factor: 0 at/below Tmin and at/above Tmax, 1 at Topt, linear on each side.
    Vectorised over species arrays; requires finite values and Tmin < Topt < Tmax. Returns FP64.'''
    t = np.asarray(temperature_c, dtype=np.float64)
    lo, op, hi = (np.asarray(x, dtype=np.float64) for x in (t_min_c, t_opt_c, t_max_c))
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(lo)) and np.all(np.isfinite(op)) and np.all(np.isfinite(hi))):
        raise GrowthError('temperature and cardinal temperatures must be finite')
    if np.any(lo >= op) or np.any(op >= hi):
        raise GrowthError('cardinal temperatures must satisfy Tmin < Topt < Tmax')
    rise = (t - lo) / (op - lo)
    fall = (hi - t) / (hi - op)
    f = np.where(t <= lo, 0.0, np.where(t >= hi, 0.0, np.where(t <= op, rise, fall)))
    f = np.minimum(np.maximum(f, 0.0), 1.0)
    return float(f) if f.ndim == 0 else f


def species_response(table, *, month, temperature_c):
    '''Host [S] activity (selected by the ACTUAL month 1..12) and temperature factor at `temperature_c`.
    Missing/impossible month or temperature is refused, never defaulted.'''
    _check_table(table)
    m = _month(month)
    t = _temperature_c(temperature_c)
    h = table.host
    act = np.array(h['activity'][:, m - 1], dtype=np.float64)
    tf = np.asarray(temperature_factor(t, h['tmin'], h['topt'], h['tmax']), dtype=np.float64)
    return {'month': m, 'temperature_c': t, 'activity': act, 'temperature_factor': tf, 'effective': act * tf}


def integral_increments(table, *, month, temperature_c, dt_days, year_days):
    '''Increments of the interval integrals over ONE sub-step with constant month/temperature:
    Aint[s] += activity_s * dt_days / year_days ; Tint[s] += temperature_factor_s * activity_s * dt_days / year_days.
    year_days is the explicit length (365 or 366) of the actual calendar year; wind/rain/recession/closed
    durations are simply not passed. Date splitting is the scheduler's responsibility.'''
    r = species_response(table, month=month, temperature_c=temperature_c)
    dt = _dt(dt_days)
    yd = _strict_int(year_days, 'year_days', 365)
    if yd not in (365, 366):
        raise GrowthError(f'year_days must be 365 or 366 (explicit Gregorian year length), got {yd}')
    a = r['activity'] * dt / yd
    return a, r['temperature_factor'] * r['activity'] * dt / yd


# ------------------------------------------------------------------ argument / array checks (metadata only)
def _check_vector(table, name, arr, dtype, n=None):
    if table.backend == 'cupy':
        cp = _cupy()
        if not isinstance(arr, cp.ndarray):
            raise GrowthError(f'{name} must be a cupy.ndarray on CUDA device {table.device} (no implicit host '
                              f'transfer is made), got {type(arr).__name__}')
        did = _array_device_id(arr)
        if did != table.device:
            raise GrowthDeviceError(f'{name} lives on CUDA device {did} but the table is bound to {table.device}')
    elif type(arr) is not np.ndarray:
        raise GrowthError(f'{name} must be a plain numpy.ndarray for backend {table.backend!r} (no implicit '
                          f'conversion), got {type(arr).__name__}')
    if arr.ndim != 1:
        raise GrowthError(f'{name} must be 1-D, got shape {tuple(arr.shape)}')
    if arr.dtype != np.dtype(dtype):
        raise GrowthError(f'{name} must have dtype {np.dtype(dtype)} exactly (no conversion), got {arr.dtype}')
    if not arr.flags.c_contiguous:
        raise GrowthError(f'{name} must be C-contiguous (no implicit copy)')
    if n is not None and int(arr.shape[0]) != n:
        raise GrowthError(f'{name} has {int(arr.shape[0])} entries, expected {n}')
    return int(arr.shape[0])


def _check_population(table, spec):
    P = None
    for name, arr, dtype in spec:
        P = _check_vector(table, name, arr, dtype, P)
    return P


def _host_S_vector(table, name, arr):
    if type(arr) is not np.ndarray or arr.dtype != np.float64 or arr.shape != (table.n_species,):
        raise GrowthError(f'{name} must be a host float64 numpy.ndarray of shape ({table.n_species},)')
    if not arr.flags.c_contiguous:
        raise GrowthError(f'{name} must be C-contiguous')
    return arr


def _packet_dict(names, raw):
    out = {}
    for n, v in zip(names, raw):
        out[n] = int(v) if n.startswith(_COUNT_PREFIXES) and math.isfinite(float(v)) else float(v)
    return out


def _reduce_partials(part, max_slots):
    r = np.array(part.sum(axis=0), dtype=np.float64)
    for j in max_slots:
        r[j] = part[:, j].max()
    return r


def _unfinite(packet):
    return [k for k, v in packet.items() if not math.isfinite(float(v))]


# ------------------------------------------------------------------ demand
def _forcing(table, month, temperature_c, pet, dt):
    r = species_response(table, month=month, temperature_c=temperature_c)
    h = table.host
    act, tf = r['activity'], r['temperature_factor']
    code = h['law_code']
    with np.errstate(all='ignore'):
        canopy = h['demand_a'] * h['live_fraction'] * (pet * 0.001) * dt * act * tf
        ref = np.where(code == 1, h['demand_b'], 1.0)
        bio = h['demand_a'] * (pet / ref) * dt * act * tf
        krate = np.where(code == 0, canopy, bio)
    if not (np.all(np.isfinite(krate)) and float(krate.min()) >= 0.0):
        raise GrowthDataError('species demand rate is not finite/non-negative (overflow in PET*dt*coefficient)')
    return {'activity': act, 'temperature_factor': tf, 'krate': np.ascontiguousarray(krate)}


def _demand_numpy(table, B, area, sidx, live, krate, out):
    S = table.n_species
    code = table.host['law_code']
    acc = np.zeros(len(DEMAND_PACKET))
    P = B.shape[0]
    for lo in range(0, P, NUMPY_CHUNK):
        hi = min(P, lo + NUMPY_CHUNK)
        b, a, si, lv = B[lo:hi], area[lo:hi], sidx[lo:hi], live[lo:hi]
        with np.errstate(all='ignore'):
            bad_idx = (si < 0) | (si >= S)
            s = np.where(bad_idx, 0, si)
            is_canopy = code[s] == 0
            bad_in = (lv & (~np.isfinite(b) | (b <= 0.0))) | (lv & is_canopy & (~np.isfinite(a) | (a < 0.0)))
            d = np.where(lv, np.where(is_canopy, a, b) * krate[s], 0.0)
            bad_out = ~np.isfinite(d) | (d < 0.0)
        out[lo:hi] = d
        acc[0] += np.count_nonzero(bad_idx)
        acc[1] += np.count_nonzero(bad_in)
        acc[2] += np.count_nonzero(bad_out)
        acc[3] += np.count_nonzero(lv)
        acc[4] += np.count_nonzero(d > 0.0)
        acc[5] += float(d.sum())
        m = float(d.max())
        if m > acc[6]:
            acc[6] = m
    return acc


def demand(table, *, B_kg, canopy_area_m2, species_index, live, month, temperature_c, pet_mm_per_day, dt_days,
           max_bytes, extra_reserve_bytes=0):
    '''Requested plant water D[p] (m3) for one sub-step of `dt_days` (<= 7) with constant month/temperature/PET.

    Law per species (catalog water_use): canopy_pet_v1 uses the SUPPLIED canopy_area_m2; biomass_specific_pet_v1
    uses B_kg (total dry AGB) and ignores the area (it may then be None only if no species uses the canopy law).
    Only live plants demand; dead/removed plants get exactly 0. Canopies are never capped by cell area.
    Returns {'requested_m3' (new array on the backend), 'packet', 'forcing' (S-sized host), 'memory', ...}.'''
    _check_table(table)
    _require_current_device(table)
    m = _month(month)
    t = _temperature_c(temperature_c)
    pet = _pet(pet_mm_per_day)
    dt = _dt(dt_days)
    mb = _strict_int(max_bytes, 'max_bytes', 1)
    res = _strict_int(extra_reserve_bytes, 'extra_reserve_bytes', 0)
    spec = [('B_kg', B_kg, np.float64), ('species_index', species_index, np.int32), ('live', live, np.bool_)]
    if canopy_area_m2 is not None:
        spec.append(('canopy_area_m2', canopy_area_m2, np.float64))
    elif bool(np.any(table.host['law_code'] == 0)):
        raise GrowthError('canopy_area_m2 is required: at least one species uses the canopy_pet_v1 demand law')
    P = _check_population(table, spec)
    mem = memory_estimate('demand', P, table.n_species, table.backend, has_canopy_area=canopy_area_m2 is not None,
                          extra_reserve_bytes=res)
    _admit(mem, mb)
    f = _forcing(table, m, t, pet, dt)
    area = B_kg if canopy_area_m2 is None else canopy_area_m2
    if table.backend == 'cupy':
        cp = _cupy()
        out = cp.empty(P, dtype=cp.float64)
        if P:
            nb = _n_blocks(P)
            part = cp.empty((nb, len(DEMAND_PACKET)), dtype=cp.float64)
            kd = _cuda_kernels(table.device)[0]
            kd((nb,), (CUDA_THREADS,), (B_kg, area, species_index, live, cp.asarray(f['krate']),
                                        table.dev['law_code'], out, part, np.int32(table.n_species), np.int64(P)))
            raw = _reduce_partials(part.get(), DEMAND_MAX_SLOTS)
        else:
            raw = np.zeros(len(DEMAND_PACKET))
    else:
        out = np.empty(P, dtype=np.float64)
        if table.backend == 'numba':
            raw = np.zeros(len(DEMAND_PACKET))
            if P:
                _numba_kernels()[0](B_kg, area, species_index, live, f['krate'], table.host['law_code'], out, raw)
        else:
            raw = _demand_numpy(table, B_kg, area, species_index, live, f['krate'], out)
    pk = _packet_dict(DEMAND_PACKET, raw)
    problems = []
    if pk['bad_species_index']:
        problems.append(f"{pk['bad_species_index']} species_index value(s) outside [0, {table.n_species})")
    if pk['bad_input']:
        problems.append(f"{pk['bad_input']} live plant(s) with non-finite/non-positive B or invalid canopy area")
    if pk['bad_output']:
        problems.append(f"{pk['bad_output']} requested demand value(s) non-finite or negative (overflow?)")
    if _unfinite(pk):
        problems.append(f'aggregate packet not finite: {_unfinite(pk)}')
    if problems:
        raise GrowthDataError('invalid demand input: ' + '; '.join(problems) + '; inputs unchanged, nothing returned')
    return {'requested_m3': out, 'packet': pk,
            'forcing': {'month': m, 'temperature_c': t, 'pet_mm_per_day': pet, 'dt_days': dt, **f},
            'laws': dict(zip(table.species_ids, table.demand_laws)), 'formulas': dict(DEMAND_FORMULAS),
            'memory': mem}


# ------------------------------------------------------------------ growth
def _check_integrals(table, activity_integral, temperature_integral):
    A = _host_S_vector(table, 'activity_integral', activity_integral)
    T = _host_S_vector(table, 'temperature_integral', temperature_integral)
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(T))):
        raise GrowthDataError('activity/temperature integrals must be finite')
    if float(A.min()) < 0.0 or float(T.min()) < 0.0:
        raise GrowthDataError('activity/temperature integrals must be >= 0')
    if bool(np.any(T > A)):
        raise GrowthDataError('temperature_integral must not exceed activity_integral (temperature factor <= 1)')
    if float(A.max()) > MAX_ACTIVITY_INTEGRAL * (1.0 + ACTIVITY_INTEGRAL_SLACK):
        raise GrowthDataError(f'activity_integral exceeds 7/365 = {MAX_ACTIVITY_INTEGRAL}: the kernel is a '
                              'weekly-or-shorter Euler adaptation; callers split longer periods')
    return A, T


def _grow_numpy(table, B, E, sidx, live, D, U, A, T, year_end, lim, out):
    h = table.host
    S = table.n_species
    mature, rate, support, lfrac, xfrac = h['mature'], h['rate'], h['support'], h['live_fraction'], h['extra_fraction']
    acc = np.zeros(len(GROW_PACKET))
    P = B.shape[0]
    for lo in range(0, P, NUMPY_CHUNK):
        hi = min(P, lo + NUMPY_CHUNK)
        b, e, d, u, lv, si = B[lo:hi], E[lo:hi], D[lo:hi], U[lo:hi], live[lo:hi], sidx[lo:hi]
        with np.errstate(all='ignore'):
            bad_idx = (si < 0) | (si >= S)
            s = np.where(bad_idx, 0, si)
            bad_in = (~np.isfinite(b)) | (~np.isfinite(e)) | (b < 0.0) | (e < 0.0) | (lv & ((b <= 0.0) | (e > b)))
            bad_w = ((~np.isfinite(d)) | (~np.isfinite(u)) | (d < 0.0) | (u < 0.0) | ((~lv) & ((d != 0.0) | (u != 0.0)))
                     | (lv & (d == 0.0) & (u != 0.0)) | (lv & (d > 0.0) & (u > d * lim)))
            m, a, t, k = mature[s], A[s], T[s], support[s]
            persistent = b - e
            required = b * lfrac[s] * a
            reqsup = k * d
            actsup = k * u
            fplan = np.where(required > 0.0, np.minimum(reqsup / required, 1.0), 0.0)
            nplan = GMOD_SCALE * rate[s] * t * fplan * persistent * (1.0 - persistent / m)
            nplan = np.maximum(nplan, 0.0)
            nplan = np.minimum(nplan, np.maximum(m - b, 0.0))
            meant = np.where(a > 0.0, t / a, 0.0)
            xplan = xfrac[s] * np.maximum(reqsup - required, 0.0) * meant
            xplan = np.minimum(xplan, np.maximum(m - b - nplan, 0.0))
            sat = np.where(d > 0.0, np.minimum(u / d, 1.0), 0.0)
            clamp = lv & (d > 0.0) & (u > d)
            ng = nplan * sat
            eg = xplan * sat
            deficit = np.maximum(required - actsup, 0.0)
            if year_end:
                loss = e + eg
                enew = np.zeros_like(e)
                bnew = persistent + ng
            else:
                loss = np.zeros_like(e)
                enew = e + eg
                bnew = persistent + ng + enew
            z = 0.0
            nplan, xplan, ng, eg, loss = (np.where(lv, v, z) for v in (nplan, xplan, ng, eg, loss))
            sat, required, reqsup, actsup, deficit = (np.where(lv, v, z) for v in (sat, required, reqsup, actsup,
                                                                                    deficit))
            bnew = np.where(lv, bnew, b)
            enew = np.where(lv, enew, e)
            finite = (np.isfinite(bnew) & np.isfinite(enew) & np.isfinite(nplan) & np.isfinite(xplan)
                      & np.isfinite(ng) & np.isfinite(eg) & np.isfinite(loss) & np.isfinite(sat)
                      & np.isfinite(required) & np.isfinite(reqsup) & np.isfinite(actsup) & np.isfinite(deficit))
            bad_out = (~finite) | (enew < 0.0) | (bnew < 0.0) | (enew > bnew)
            ref = ((b + ng) + eg) - loss
            resid = np.abs(bnew - ref)
            tol = LEDGER_EPS_FACTOR * FP64_EPS * (b + bnew + ng + eg + loss)
            led_bad = resid > tol
        for j, v in enumerate((bnew, enew, nplan, xplan, ng, eg, loss, sat, required, reqsup, actsup, deficit)):
            out[j, lo:hi] = v
        acc[0] += np.count_nonzero(bad_idx)
        acc[1] += np.count_nonzero(bad_in)
        acc[2] += np.count_nonzero(bad_w)
        acc[3] += np.count_nonzero(bad_out)
        acc[4] += np.count_nonzero(led_bad)
        acc[5] += np.count_nonzero(clamp)
        acc[6] += np.count_nonzero(lv)
        acc[7] += np.count_nonzero((ng + eg) > 0.0)
        for j, v in ((8, b), (9, bnew), (10, ng), (11, eg), (12, loss), (13, d), (14, u), (15, deficit)):
            acc[j] += float(v.sum())
        r = float(resid.max())
        if r > acc[16]:
            acc[16] = r
    return acc


def grow(table, *, B_kg, E_kg, species_index, live, requested_m3, delivered_m3, activity_integral,
         temperature_integral, year_end, max_bytes, extra_reserve_bytes=0, overdelivery_rtol=OVERDELIVERY_RTOL):
    '''One requested-water growth update over an actual interevent period (B/E and geometry held fixed).

    D = requested_m3 and U = delivered_m3 are the per-plant SUMS over the period (m3, U the actual accepted
    water, never clipped). activity_integral[s] = SUM activity*dt/year_days and temperature_integral[s] = SUM
    temperature_factor*activity*dt/year_days (host float64 [S]; 0 <= Tint <= Aint <= 7/365).
    For a live plant with k = biomass_support_kg_per_m3:
      required = B*live_fraction*Aint ; requested_support = k*D ; actual_support = k*U
      Fplan = min(1, requested_support/required) (0 if required == 0)
      normal_plan = max(0, 0.95*r*Tint*Fplan*(B-E)*(1-(B-E)/mature)), capped to max(mature-B, 0)
      extra_plan  = extra_growth_fraction*max(requested_support-required, 0)*(Tint/Aint), capped to
                    max(mature-B-normal_plan, 0)
      S = min(1, U/D) (D > 0, else 0) ; normal_gain = normal_plan*S ; extra_gain = extra_plan*S
    Actual satisfaction S is applied ONCE, after the plan is capped. Ordinary update: Enew = E + extra_gain,
    Bnew = (B-E) + normal_gain + Enew. year_end=True: seasonal_loss = E + extra_gain, Enew = 0,
    Bnew = (B-E) + normal_gain. Non-live plants are copied unchanged (D and U must be exactly 0).
    U may exceed D only by the documented FP64 tolerance (overdelivery_rtol, relative, <= 1e-6): then S = 1,
    the plant is counted in packet['clamped_overdelivery'] and actual_support still reports k*U.
    Returns per-plant arrays (views of one [12, P] buffer), the aggregate packet, the ledger and the memory
    estimate; raises GrowthDataError (nothing returned, inputs untouched) on any invalid/non-finite content.'''
    _check_table(table)
    _require_current_device(table)
    ye = _strict_bool(year_end, 'year_end')
    mb = _strict_int(max_bytes, 'max_bytes', 1)
    res = _strict_int(extra_reserve_bytes, 'extra_reserve_bytes', 0)
    rtol = _num(overdelivery_rtol, 'overdelivery_rtol')
    if not 0.0 <= rtol <= MAX_OVERDELIVERY_RTOL:
        raise GrowthError(f'overdelivery_rtol must lie in [0, {MAX_OVERDELIVERY_RTOL}], got {rtol}')
    P = _check_population(table, [('B_kg', B_kg, np.float64), ('E_kg', E_kg, np.float64),
                                  ('species_index', species_index, np.int32), ('live', live, np.bool_),
                                  ('requested_m3', requested_m3, np.float64),
                                  ('delivered_m3', delivered_m3, np.float64)])
    _host_S_vector(table, 'activity_integral', activity_integral)
    _host_S_vector(table, 'temperature_integral', temperature_integral)
    mem = memory_estimate('grow', P, table.n_species, table.backend, extra_reserve_bytes=res)
    _admit(mem, mb)
    A, T = _check_integrals(table, activity_integral, temperature_integral)
    lim = 1.0 + rtol
    K = len(GROW_PACKET)
    if table.backend == 'cupy':
        cp = _cupy()
        out = cp.empty((len(GROW_OUTPUTS), P), dtype=cp.float64)
        if P:
            nb = _n_blocks(P)
            part = cp.empty((nb, K), dtype=cp.float64)
            kg = _cuda_kernels(table.device)[1]
            d = table.dev
            kg((nb,), (CUDA_THREADS,), (B_kg, E_kg, species_index, live, requested_m3, delivered_m3, d['mature'],
                                        d['rate'], d['support'], d['live_fraction'], d['extra_fraction'],
                                        cp.asarray(A), cp.asarray(T), np.int32(1 if ye else 0), np.float64(lim),
                                        np.int32(table.n_species), out, part, np.int64(P)))
            raw = _reduce_partials(part.get(), GROW_MAX_SLOTS)
        else:
            raw = np.zeros(K)
    else:
        out = np.empty((len(GROW_OUTPUTS), P), dtype=np.float64)
        if table.backend == 'numba':
            raw = np.zeros(K)
            if P:
                h = table.host
                _numba_kernels()[1](B_kg, E_kg, species_index, live, requested_m3, delivered_m3, h['mature'],
                                    h['rate'], h['support'], h['live_fraction'], h['extra_fraction'], A, T,
                                    1 if ye else 0, lim, out, raw)
        else:
            raw = _grow_numpy(table, B_kg, E_kg, species_index, live, requested_m3, delivered_m3, A, T, ye, lim, out)
    pk = _packet_dict(GROW_PACKET, raw)
    problems = []
    if pk['bad_species_index']:
        problems.append(f"{pk['bad_species_index']} species_index value(s) outside [0, {table.n_species})")
    if pk['bad_input']:
        problems.append(f"{pk['bad_input']} plant(s) with non-finite/negative B or E, live B <= 0 or E > B")
    if pk['bad_water']:
        problems.append(f"{pk['bad_water']} plant(s) with invalid water: non-finite/negative D or U, water for a "
                        f'non-live plant, U > 0 with D == 0, or U > D*(1+{rtol})')
    if pk['bad_output']:
        problems.append(f"{pk['bad_output']} plant(s) with non-finite or inconsistent outputs (overflow?)")
    if pk['ledger_violations']:
        problems.append(f"{pk['ledger_violations']} plant(s) violate the aboveground ledger bound")
    if _unfinite(pk):
        problems.append(f'aggregate packet not finite: {_unfinite(pk)}')
    if problems:
        raise GrowthDataError('invalid growth input/result: ' + '; '.join(problems)
                              + '; inputs unchanged, nothing returned')
    gains = pk['sum_normal_gain_kg'] + pk['sum_extra_gain_kg']
    scale = pk['sum_B_kg'] + pk['sum_B_new_kg'] + gains + pk['sum_seasonal_loss_kg']
    resid = pk['sum_B_new_kg'] - (pk['sum_B_kg'] + gains - pk['sum_seasonal_loss_kg'])
    tol = (P + 16) * FP64_EPS * scale
    ledger = {'sum_B_kg': pk['sum_B_kg'], 'sum_B_new_kg': pk['sum_B_new_kg'],
              'sum_normal_gain_kg': pk['sum_normal_gain_kg'], 'sum_extra_gain_kg': pk['sum_extra_gain_kg'],
              'sum_seasonal_loss_kg': pk['sum_seasonal_loss_kg'], 'population_residual_kg': resid,
              'population_tolerance_kg': tol, 'population_closed': abs(resid) <= tol,
              'max_plant_residual_kg': pk['max_ledger_residual_kg'],
              'plant_bound': f'{LEDGER_EPS_FACTOR} * eps * (B + Bnew + normal_gain + extra_gain + seasonal_loss)',
              'statement': 'Bnew == B + normal_gain + extra_gain - seasonal_loss (aboveground only; no carbon closure)'}
    result = {name: out[i] for i, name in enumerate(GROW_OUTPUTS)}
    result.update({'packet': pk, 'ledger': ledger, 'memory': mem, 'year_end': ye, 'overdelivery_rtol': rtol,
                   'growth_law': GROWTH_LAW})
    return result


__all__ = ['VERSION', 'GROWTH_LAW', 'DEMAND_CANOPY_PET', 'DEMAND_BIOMASS_SPECIFIC_PET', 'DEMAND_LAWS', 'BACKENDS',
           'OVERDELIVERY_RTOL', 'MAX_ACTIVITY_INTEGRAL', 'DEMAND_PACKET', 'GROW_PACKET', 'GROW_OUTPUTS',
           'GROWTH_TRAITS', 'DEMAND_TRAITS', 'DEMAND_FORMULAS', 'CAVEATS', 'GrowthError', 'GrowthBudgetError',
           'GrowthDeviceError', 'GrowthDataError', 'CompiledSpecies', 'compile_species', 'check_catalog',
           'temperature_factor', 'species_response', 'integral_increments', 'demand', 'grow', 'memory_estimate',
           'table_memory']
