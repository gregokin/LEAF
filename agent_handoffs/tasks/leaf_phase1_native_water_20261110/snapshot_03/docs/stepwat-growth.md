# STEP-style individual aboveground growth (opt-in weekly adaptation)

Module: `leaf/stepwat.py`, hooked into `benchmarks/daily_water_growth.Simulation` (vegetation-only mode) and
`leaf/coupling.CoupledSimulation`. Task `stepwat_growth_20261031`.

Without the opt-in keyword (`stepwat_traits=None`, the default) every legacy default, ledger output, counter
dictionary and numerical result is unchanged.

## 1. Source and scope

Source method: public STEPWAT2 revision `003cb9b4adbf13d64a39d6d62fd95280adac27fc`

- `ST_resgroups.c` (`rgroup_Grow`, `_extra_growth`):
  https://github.com/DrylandEcology/STEPWAT2/blob/003cb9b4adbf13d64a39d6d62fd95280adac27fc/ST_resgroups.c
- `ST_species.c` (`Species_GetBiomass`):
  https://github.com/DrylandEcology/STEPWAT2/blob/003cb9b4adbf13d64a39d6d62fd95280adac27fc/ST_species.c
- `sxw_resource.c` (resource / transpiration mapping):
  https://github.com/DrylandEcology/STEPWAT2/blob/003cb9b4adbf13d64a39d6d62fd95280adac27fc/sxw_resource.c

STEP rule (annual): relative size `R = biomass / mature biomass`; normal increment
`dR = intrinsic_rate * gmod * R * (1 - R)` with `gmod = 0.95 * temperature_modifier / max(PR, 1)`,
`PR = required / available` biomass-support resources; surplus resources give seasonal extra biomass that is
removed at the end of the season.

This implementation is a **documented weekly (and partial year-end) explicit-Euler adaptation** of that rule.
It is not a port, not an unchanged STEPWAT routine and does not reproduce annual STEP trajectories.

Retained from the existing coupling (authoritative, untouched): root layer fractions (vertical profiles),
spatial individuals and the CSR root graph, native residual-limited transpiration as the only water authority,
inactive-cell masks and the area conversion. Deliberately **not** introduced: STEP's patch-level largest-first
resource allocator (spatial proportional root competition stays), STEP's added-transpiration resource
correction, any nonzero dry fallback, a new physical water store, root/leaf/wood/carbon pools.

## 2. State and units

- `B[p]` – total aboveground dry biomass per individual, kg. **The sole observed biomass target.**
- `E[p]` – seasonal-extra component, kg, `0 <= E <= B`, kept internally so persistent size growth (`B - E`) and
  season-end loss remain distinct.
- `uptake_week[p]` – existing per-plant actual uptake accumulator, m3 (now: since the last successful commit).
- `temperature_integral[s]` – per root profile, `sum(temperature_factor[s] * activity[s] / 365)` over accepted
  days since the last successful commit (S-sized; no P-sized daily array, no plant/cell history).
- `activity_now[s]` – activity of the current period's start month, held constant with the demand cache `Dp`.
- `gain_cum`, `loss_cum` – existing ledgers; in this mode they hold normal + extra gain and seasonal loss.

Traits (one row per root profile **name**; profile names and `inputs['root_profile_id']` are the authoritative
species/PFT identifiers; every value explicit and finite; no defaults):

[Study-based candidates](study-stepwat-parameters.md) now provide a conditional
Setaria normal-growth fit and mesquite seedling priors. They remain inactive:
the fitted/required normal rates exceed the current `[0, 1]` validation range,
and adult geometry/phenology are unidentified. The study diagnostic includes a
final partial observation update; it does not establish a full native calendar
replay. Runtime traits and defaults below remain unchanged.

| trait | unit | range | meaning |
|---|---|---|---|
| `mature_biomass_kg` | kg | > 0 | species mature total aboveground mass |
| `intrinsic_rate_per_year` | 1/year | [0, 1] | STEP intrinsic rate (bounded for the Euler step) |
| `biomass_support_kg_per_m3` | kg supported per m3 transpired per year | > 0 | **annual biomass-support coefficient** (STEP PR denominator); not WUE, not a production yield |
| `live_fraction` | – | [0, 1] | live share of total aboveground mass |
| `canopy_biomass_density_kg_m2` | kg/m2 canopy | > 0 | allometric scale |
| `potential_transpiration_m_per_day` | m/day over canopy area | >= 0 | canopy transpiration **demand** depth; distinct from native potential soil evaporation / reference ET |
| `extra_growth_fraction` | kg seasonal mass per kg surplus support | [0, 1] | surplus → seasonal mass |
| `monthly_activity` | 12 fractions | [0, 1] | January..December, 365-day no-leap calendar |

Daily host forcing `temperature_factor`: a real scalar, a host sequence of exactly `S` reals, or a 1-D NumPy
array of shape `(S,)`, values in [0, 1]. It is **host-only**: device-backed values (anything exposing
`.get()`, `__cuda_array_interface__` or `.device` that is not a NumPy array) are refused without attempting a
transfer; NumPy shape / sequence length are checked before any conversion and every sequence element is validated
as a scalar before materialization, so at most `S` host floats are ever materialized. It is a **caller-supplied
modifier** (no temperature-response law is asserted), required in STEP mode and validated **before** the native
hydro step (coupled) / before any accumulator or counter changes (vegetation-only); a refused forcing leaves soil
state, counters, uptake and the temperature integral untouched.

### Units of reported values

- Resident arrays `B[p]`, `E[p]`, `gain_cum[p]`, `loss_cum[p]`: **kg per individual**.
- `plant_summary()['biomass']` / coupled `summary()['biomass']` (`initial`, `growth`, `turnover`, `total`,
  `persistent`, `seasonal`, `residual`): **kg, domain sums over all individuals** (`scope: 'domain_sum'`,
  `population`, `per_individual_mean`; only `min` is per individual).
- `summary()['stepwat']['domain_totals_kg']`: domain sums (initial, total, persistent, seasonal, normal_gain,
  extra_gain, seasonal_loss, gain_cum, loss_cum). `summary()['stepwat']['per_individual_kg']`: per-individual
  statistics (population, B_min/B_max/B_mean, E_max/E_mean). Domain totals must never be read as per-individual
  masses.

## 3. Equations (per individual `p`, profile `s = root_profile_id[p]`)

Initialisation (refused before any water day if `B > mature_biomass_kg`):

```
B = pi * canopy_radius^2 * canopy_biomass_density_kg_m2[s]      E = 0
Dp = pi * canopy_radius^2 * potential_transpiration_m_per_day[s] * live_fraction[s] * activity[s, month(day 0)]
```

Daily (accepted physical day only; failed days accumulate nothing):

```
uptake_week[p]          += actual uptake gathered from the native scale (existing path)
temperature_integral[s] += temperature_factor[s] * activity_now[s] / 365
```

Growth update when `day % 7 == 0 OR day % 365 == 0`, with `dt_days` = accepted days since the last
**successful** commit (so the period ending at day 365 has `dt_days = 1`; 365 days are not 52 complete weeks):

```
persistent  = B - E
R           = persistent / mature
required    = B * live_fraction * activity_now * dt_days / 365
available   = biomass_support_kg_per_m3 * uptake_week            (m3 -> kg of annual support)
water       = min(1, available / required)   if required > 0 else 0     (= 1 / max(PR, 1))
normal_gain = intrinsic_rate * 0.95 * temperature_integral * water * persistent * (1 - R)
normal_gain = min(normal_gain, max(mature - B, 0))                 (actual, bounded gain is recorded)
mean_tf     = temperature_integral / (activity_now * dt_days / 365)   (0 when activity_now == 0)
extra_gain  = min(extra_growth_fraction * max(available - required, 0) * mean_tf,
                  max(mature - B - normal_gain, 0))
year-end (day % 365 == 0):  loss = E + extra_gain,  E_new = 0
otherwise:                  loss = 0,               E_new = E + extra_gain
B_new       = persistent + normal_gain + E_new
gain_cum   += normal_gain + extra_gain ;  loss_cum += loss
canopy_new  = sqrt(B_new / (pi * density[s]));  root_new = root_multiplier * canopy_new
Dp_new      = pi * canopy_new^2 * potential_transpiration * live_fraction * activity[s, month(day)]
```

Consequences: no gain without actual water, none during dormancy (`activity 0`), none at `temperature_factor
0`; `B = B0 + sum(gains) - sum(losses)` closes (`biomass_budget` check); whole-plant carbon closure is **not**
asserted. No turnover from the legacy fixture applies in this mode.

Adaptation of annual quantities: the STEP annual rate and support coefficient are pro-rated by the active period
length through `temperature_integral` (which carries `dt_days/365` and the activity) and `required` (which
carries `activity*dt_days/365`); `available` is the support the period's actual uptake would provide for one
year, so `available/required` is dimensionless exactly as STEP's `PR`.

## 4. Order of operations

Daily (coupled `step_day`): phase check → latch check → device check → **temperature_factor validation** →
request from the held `Dp` cache → native hydro + residual-limited transpiration → compiled gather of the actual
per-plant uptake and closure check → publish soil state/counters → `commit_daily_uptake(cand,
temperature_factor)`: accumulate uptake and temperature → advance day → growth transaction if at a boundary.

Growth transaction (existing candidate/build/admit/swap, `Simulation._weekly_update`): compiled kernel writes
candidates (`_Bc, _Ec, _gc, _lc, _cc, _rc, _Dc`) → candidate invariants (B/canopy/root finite positive, demand
finite ≥ 0, `0 <= E <= B`, gains/losses ≥ 0) → rebuild-bound ledger (includes STEP bytes) → caller admission →
graph build → actual old+new ledger and admission → graph invariants → **commit together**: biomass, seasonal
mass, geometry, demand, graph, `gain_cum/loss_cum`, scalar totals, reset of `uptake_week`,
`temperature_integral`, period clock, new period month/activity. Any refusal latches with the day already
accepted and all accumulators retained; `retry_weekly_update()` re-evaluates the same period once (no second
water day).

Counters: `weeks_done` = weekly boundaries committed; `growth_updates` = weekly + partial year-end commits;
`maps_built = growth_updates + 1`; `pending_days = day - period_start_day`; `year_end_updates`; `month_index`.

## 5. Fixture example (labelled; NOT a calibration)

```python
FIXTURE_TRAITS = {  # arbitrary bounded test values
    'grass': {'mature_biomass_kg': 10.0, 'intrinsic_rate_per_year': 0.8, 'biomass_support_kg_per_m3': 20.0,
              'live_fraction': 0.5, 'canopy_biomass_density_kg_m2': 1.0,
              'potential_transpiration_m_per_day': 0.004, 'extra_growth_fraction': 0.25,
              'monthly_activity': [1.0] * 12},
}
# coupled (native column/state prepared as in tests/test_coupling.py):
sim = CoupledSimulation(case, column, state, inputs=inputs_with_root_profile_id,
                        stepwat_traits=FIXTURE_TRAITS)
sim.step_day(added_water_m, potential_evaporation_m_s, temperature_factor=0.7)        # scalar
sim.step_days([(added, pet, [0.7]), ...])                                              # per-profile [S]
# vegetation-only (external water authority supplies the per-plant actual uptake):
veg = Simulation(case, 'cpu', inputs=inputs, vegetation_only=True, stepwat_traits=FIXTURE_TRAITS)
veg.commit_daily_uptake(candidate_m3, temperature_factor=0.7)
report = sim.summary()['stepwat']   # model id, units, traits, domain_totals_kg / per_individual_kg, clock, caveats
```

Running the tests (existing interpreter, no new environment; an explicit writable Numba cache directory because
verification source trees can be read-only):

From `/home/okin/leaf`:

```sh
PYTHONPATH="$PWD:/home/okin/SYRUP/src" NUMBA_CACHE_DIR=/tmp/leaf-stepwat-numba \
  CUDA_VISIBLE_DEVICES='' \
  /home/okin/MAPLE/.venv-cupy/bin/python -m unittest discover -s tests -v
```

For CUDA checks, select the allocated device explicitly with `CUDA_VISIBLE_DEVICES`
and use writable `CUPY_CACHE_DIR` and `CUDA_CACHE_PATH` directories under `/tmp`.
The recorded checks use GPU0 UUID `GPU-5c8bd3d4-2885-82c6-a0f0-ad2b7f87b871`.
CUDA tests are skip-labelled through the existing `dwg.gpu_available()` gate.
The [source lock](../dependencies/stepwat-growth.lock.json) pins this installation;
[verification evidence](../agent_handoffs/tasks/stepwat_growth_20261031/verification.md)
records exact commands. Earlier 30-year timing/capacity runs used synthetic fixture
growth and do not measure this new mode's performance.

STEP mode requires `vegetation_only=True` (synthetic standalone water is refused), supplied `inputs` with an
explicit `root_profile_id` (no species inferred from size) and a trait row for every root profile name.

## 6. Memory and backends

Added arrays, all counted in `leaf.stepwat.state_bytes_estimate(P, S) = 8*(2P + 23S)` bytes: `E[P]`, `_Ec[P]`,
seven S-sized trait vectors, `monthly_activity[S,12]`, `temperature_integral[S]`, `activity_now[S]`, the
temperature forcing buffer `[S]` and one `[S]` scratch. On the GPU a host trait mirror of `8*19*S` bytes is also
counted (`host_mirror_bytes(S, gpu)`). These appear in every vegetation ledger (`stepwat_state_bytes`,
`stepwat_host_mirror_bytes`, added to `total_bytes`), in the coupled init scalar bound, in `live_bytes()` and
therefore in every daily native combined admission, in `array_records()` and in `memory_report()`
(`stepwat_bytes`). Allocation happens only after the existing admission preflight. No cap is raised. Legacy
ledgers/reports are unchanged when the mode is absent.

CPU: `@njit(cache=True)` per-plant kernel. GPU: one fused CuPy `RawKernel` with identical arithmetic (plants
never leave the device; only the `[S]` temperature forcing is transferred daily into a resident buffer). Lazy
imports: `leaf.stepwat` needs numpy/numba (and CuPy only when the GPU backend is used); the native package is
never imported by this module.

## 7. Deferred features and limits

- No frost, recruitment, mortality, injury, plastic roots or detailed surface remapping; root vertical profiles
  and ids are fixed; no STEP largest-first allocation; no added-resource water; no nonzero dry fallback.
- Calendar: 365-day no-leap; day 0 of the accepted-day clock is 1 January (no start-day offset is accepted).
- `temperature_factor` is a modifier supplied by the host; no temperature law is asserted.
- The legacy fixture `wue` vector is still allocated (unused) so legacy memory formulas hold; it is not reported
  as a STEP parameter.
- No scientific calibration is shipped or claimed; the fixture values above exist for tests only. Weekly
  behaviour of annual coefficients has not been validated against STEPWAT2 outputs.
