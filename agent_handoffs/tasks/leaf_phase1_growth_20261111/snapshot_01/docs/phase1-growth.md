# Phase 1: pure species demand and growth kernels (`leaf/growth.py`)

Law identifier `stepwat_requested_support_v1` (distinct from the older `stepwat_weekly_aboveground_v0.1`).
This is ONE subtask of Phase 1: a pure, versioned aboveground demand + growth module. It is independent of the
older benchmark driver and of native SYRUP water, imports only NumPy and `leaf.traits` (Numba / CuPy are
imported lazily, only when that backend is explicitly selected) and does **not** implement the calendar driver,
publication/restart, recruitment, disturbance, stochastic death, root plasticity, the root allocator or the
native bridge. The candidate was written without executing anything: see `report.md` for what is NOT RUN.

Source precedent (unchanged from `docs/stepwat-growth.md`): STEPWAT2 revision
`003cb9b4adbf13d64a39d6d62fd95280adac27fc`, `rgroup_Grow` / `_extra_growth`, relative size and biomass
support. LEAF's spatial root competition replaces STEP's patch allocator. The equations below are a documented
weekly-or-shorter Euler adaptation, **not** a port, not an annual-trajectory reproduction and not a claim about
STEPWAT daily physiology.

## 1. Species records (strict opt-in)

All scientific parameters live in each species record of the existing catalog
(`examples/phase1/species.json` is the complete two-species FIXTURE; every number is uncalibrated).
There is no separate parameter map and no hard-coded species default.

```
growth      {"law": "stepwat_requested_support_v1",
             "traits": {mature_biomass_kg, intrinsic_rate_per_year, biomass_support_kg_per_m3, live_fraction,
                        extra_growth_fraction, monthly_activity[12], temperature_min_c, temperature_opt_c,
                        temperature_max_c}}
water_use   {"law": "canopy_pet_v1",           "traits": {transpiration_coefficient}}
         or {"law": "biomass_specific_pet_v1", "traits": {demand_m3_per_kg_day, reference_pet_mm_per_day}}
```

| trait | unit | range |
|---|---|---|
| `mature_biomass_kg` | kg dry AGB per individual | > 0 |
| `intrinsic_rate_per_year` | 1/year (weekly-or-shorter Euler adaptation) | [0, 1] |
| `biomass_support_kg_per_m3` | kg aboveground biomass **supported** per m3 of water per year (kg·year/m3) | > 0 |
| `live_fraction` | live share of total dry AGB | [0, 1] |
| `extra_growth_fraction` | kg seasonal mass per kg·year of surplus support | [0, 1] |
| `monthly_activity` | 12 fractions, January..December | [0, 1] each |
| `temperature_min_c` < `temperature_opt_c` < `temperature_max_c` | °C | finite, min ≥ −273.15 |
| `transpiration_coefficient` | dimensionless | ≥ 0 |
| `demand_m3_per_kg_day` | m3 per kg total dry AGB per day at the reference PET | ≥ 0 |
| `reference_pet_mm_per_day` | mm/day | > 0 |

`compile_species` compiles **every** species of the catalog (sorted ids, exact `catalog.hash` kept). A missing,
null, `"unimplemented"`, unknown or old growth/water law, a missing/unknown trait or an out-of-range value
refuses the compilation and names the species. Phase 0 reading of unimplemented species is untouched.

**Rate limitation (kept explicit).** `intrinsic_rate_per_year` must lie in [0, 1], the range qualified for the
bounded weekly Euler adaptation. Earlier fitted grass rates (for example 295/year) are outside it and are **not**
admitted by raising the bound: another solver and requalification would be separate work.

`k = biomass_support_kg_per_m3` is annual biomass *supported* by water. It is not kg new biomass per m3 and not
a water-use efficiency. `required` below is the matching annual-support-equivalent prorated to the period; no
fixed physical percentage of water is assigned to maintenance.

## 2. Temperature and activity

```
temperature_factor(T) = 0 for T <= Tmin or T >= Tmax ; 1 at Topt ;
                        (T - Tmin)/(Topt - Tmin) on [Tmin, Topt] ; (Tmax - T)/(Tmax - Topt) on [Topt, Tmax]
activity = monthly_activity[month - 1]          month = actual supplied integer 1..12
```

A missing, non-finite, bool, out-of-range month or a temperature below absolute zero is refused; nothing is
defaulted. Date splitting and exact Gregorian year fractions belong to the later scheduler.
`integral_increments(table, month, temperature_c, dt_days, year_days)` returns the per-species increments of

```
Aint[s] += activity_s * dt_days / year_days
Tint[s] += temperature_factor_s * activity_s * dt_days / year_days          (year_days: 365 or 366, explicit)
```

which the caller accumulates over the actual growth period. They exclude wind, rain, recession and closed
durations (those durations are simply never added). `grow` validates finite, `0 <= Tint <= Aint` and
`Aint <= 7/365`; longer periods must be split by the caller (weekly-or-shorter envelope).

## 3. Water demand (a declared species law)

`demand(...)` returns the requested volume `D[p]` in **m3** for one sub-step with constant month,
temperature and **required** shared scheduler PET (`pet_mm_per_day`, finite >= 0; this module never computes
PET). `0 < dt_days <= 7`. Only live plants demand; dead/removed plants get exactly 0.

```
canopy_pet_v1:            D = canopy_area_m2 * PET * 0.001 * dt_days * transpiration_coefficient
                              * live_fraction * activity * temperature_factor
biomass_specific_pet_v1:  D = B_kg * demand_m3_per_kg_day * (PET / reference_pet_mm_per_day) * dt_days
                              * activity * temperature_factor        (per TOTAL dry AGB: live_fraction NOT applied again)
```

Both are documented LEAF demand adaptations, **not** claims about STEPWAT daily physiology. `canopy_area_m2` is
supplied by the geometry boundary (derived through `leaf.canopy` / the catalog allometry); biomass is never assumed
proportional to area, and canopies overlapping roots are not capped by cell area or otherwise altered. Zero
PET, zero activity (dormancy) or zero temperature factor give `D = 0`. `canopy_area_m2` may be `None` only when no
species uses the canopy law.

## 4. Growth: requested-water plan, actual satisfaction applied once

State: total aboveground dry mass `B` (kg) and seasonal extra `E` (kg), `0 <= E <= B`; persistent mass `B - E`.
No whole-plant carbon closure, root pool or production yield is claimed. All water is volume (m3). The caller
holds `B`, `E` and geometry fixed over the period and supplies the period sums `D[p]` (requested) and `U[p]`
(actual accepted water), plus `Aint[s]`, `Tint[s]`.

For a live plant (species traits `r, mature, k, live_fraction, extra_growth_fraction`):

```
required          = B * live_fraction * Aint                     (kg·year)
requested_support = k * D ;  actual_support = k * U              (kg·year)
Fplan       = min(1, requested_support / required)  if required > 0 else 0
normal_plan = max(0, 0.95 * r * Tint * Fplan * (B-E) * (1 - (B-E)/mature))
normal_plan = min(normal_plan, max(mature - B, 0))               (never shrinks B because B > mature)
meanT       = Tint / Aint if Aint > 0 else 0
extra_plan  = extra_growth_fraction * max(requested_support - required, 0) * meanT
extra_plan  = min(extra_plan, max(mature - B - normal_plan, 0))
S           = min(1, U / D) if D > 0 else 0
normal_gain = normal_plan * S ;  extra_gain = extra_plan * S
```

Actual satisfaction `S` is applied **once**, after the planned components are capped. The plan uses the
*requested* support, so 75 % delivery yields exactly 75 % of the growth planned for full delivery (not 56.25 %).

```
ordinary update : Enew = E + extra_gain ;  Bnew = (B-E) + normal_gain + Enew ;  seasonal_loss = 0
year_end = True : seasonal_loss = E + extra_gain ; Enew = 0 ; Bnew = (B-E) + normal_gain
```

The year-end loss is STEP-style seasonal bookkeeping, not water-stress mortality. Non-live plants are copied
unchanged (`Bnew = B`, `Enew = E`, all other outputs 0) and must have `D == U == 0`.
`maintenance_deficit = max(required - actual_support, 0)` is reported per plant.

Ledger: per plant `Bnew == B + normal_gain + extra_gain - seasonal_loss` within
`16 * eps * (B + Bnew + normal_gain + extra_gain + seasonal_loss)` (checked inside the kernels, violation refuses
the call); the population residual of the aggregate packet is returned as `ledger`.

**Actual water.** `U` is never clipped. `U > D` is accepted only within the FP64 roundoff tolerance
`overdelivery_rtol = 1e-9` (relative, `U <= D*(1+rtol)`, user bound <= 1e-6, compatible with the allocator's root-row
tolerance 1e-9); then `S = 1`, the plant is counted in `packet['clamped_overdelivery']` and `actual_support` still
reports `k*U`. `D == 0` requires `U == 0`. No hidden balancing flux exists.

## 5. Public surface

```python
table = growth.compile_species(catalog, backend, max_bytes, *, device=None, extra_reserve_bytes=0)
growth.temperature_factor(temperature_c, t_min_c, t_opt_c, t_max_c)
growth.species_response(table, *, month, temperature_c)                       # [S] activity, temperature factor
growth.integral_increments(table, *, month, temperature_c, dt_days, year_days)  # [S] Aint, Tint increments
d = growth.demand(table, *, B_kg, canopy_area_m2, species_index, live, month, temperature_c,
                  pet_mm_per_day, dt_days, max_bytes, extra_reserve_bytes=0)     # d['requested_m3']
r = growth.grow(table, *, B_kg, E_kg, species_index, live, requested_m3, delivered_m3, activity_integral,
                temperature_integral, year_end, max_bytes, extra_reserve_bytes=0,
                overdelivery_rtol=growth.OVERDELIVERY_RTOL)
growth.memory_estimate(operation, n_plants, n_species, backend, *, has_canopy_area=True, extra_reserve_bytes=0)
growth.table_memory(n_species, backend)
growth.check_catalog(table, catalog)
```

`grow` returns arrays `B_new_kg, E_new_kg, normal_plan_kg, extra_plan_kg, normal_gain_kg, extra_gain_kg,
seasonal_loss_kg, satisfaction, required_support_kg, requested_support_kg, actual_support_kg,
maintenance_deficit_kg` (rows of **one** `[12, P]` buffer, so keep or copy what the driver needs), the fixed-size
aggregate `packet`, the `ledger`, the `memory` estimate and the law id. Inputs are never mutated; the caller swaps
`B`/`E` itself. Errors: `GrowthError` (arguments/specification), `GrowthBudgetError`, `GrowthDeviceError`,
`GrowthDataError` (invalid or non-finite content; all raised without returning outputs).

Input arrays are typed on the chosen namespace and are neither converted nor copied: 1-D C-contiguous,
`B/E/canopy_area/requested/delivered` float64, `species_index` int32 (index into the SORTED catalog ids), `live`
bool; `np.ndarray` (exact type) for `numpy`/`numba`, `cupy.ndarray` on the table's device for `cupy`.
`activity_integral` / `temperature_integral` are host float64 `[S]` arrays (explicit S-sized transfer on the GPU).

## 6. Backends and device binding

`numpy` is the chunked vectorised reference (65536 plants per chunk). `numba` is a compiled serial kernel
(first-use JIT, no on-disk cache). `cupy` launches fused CUDA kernels built with `--fmad=false` (identical
operation order to the host code) bound to ONE explicit device: `compile_species(..., 'cupy', ..., device=d)` requires
`d` to be the current device, every call re-checks the current device and each array's device, and nothing is
ever switched or moved. There is no fallback between backends. Each kernel validates the inputs while computing
and reduces a fixed-size packet (per-block partial sums, <= 256 blocks, grid-stride loop); only that packet
(<= 256 x 17 float64) is read back. Population arrays never leave the backend.

## 7. Memory and admission

Every public path checks types/shapes/dtypes/device and the explicit `max_bytes` **before** any copy, transfer,
allocation or kernel launch (`GrowthBudgetError`). `memory_estimate` is itemized (resident inputs, S-sized tables
and device mirror, simultaneous outputs, forcing/validation scratch, reduction partials and readback packet,
`extra_reserve_bytes`) and split into `host_bytes` and `device_bytes`; the bound compares their sum. Work is
O(P + S); no graph/grid/history arrays exist.

| operation | P-sized resident inputs | P-sized outputs |
|---|---|---|
| `demand` | 21 B/plant (B 8, area 8, index 4, live 1; 13 without area) | 8 B/plant |
| `grow` | 33 B/plant (B, E, D, U 8 each, index 4, live 1) | 96 B/plant (12 × f64) |

Tables: 180 B/species on the host (10 f64 vectors + `[S,12]` activity + int32 law) plus 44 B/species on the device
for `cupy`. NumPy temporaries are bounded by a fixed chunk (<= 128 / 512 B x min(P, 65536)), not by P. The
estimates are conservative structural bounds. **Excluded:** the Python interpreter, Numba/NVRTC JIT and caches, the
CUDA context, CuPy memory-pool rounding/caching. Actual CPU/GPU consumption is to be qualified by Codex.

## 8. Limits and caveats

- Documented adaptations only: demand laws, the weekly Euler rule, the cardinal temperature response and the
  seasonal-extra bookkeeping are LEAF/STEP-style adaptations, not validated physiology.
- The shipped catalog is a fixture, not a literature fit; no scientifically calibrated default exists.
- `live_fraction = 0` with `biomass_specific_pet_v1` can give demand and surplus-support extra growth (required = 0);
  choose traits accordingly.
- Aboveground ledger only; no carbon closure, root biomass or WUE production yield.
- Plant loops are O(P) for any population; per-call outputs are fresh arrays (no in-place update, no input
  mutation); the scheduler/driver owns the persistent B/E inventory.
- Not covered here: roots/allocator, native water bridge, calendar and year-fraction logic, recruitment,
  disturbance, mortality, publication/restart.
