# `jornada_grass05_shrub10_256` — Jornada-style grass / shrub LEAF trial case

**Status: input-authoring candidate, authored 2026-11-11 (task `trial_case_20261111`). Nothing was executed**: no LEAF, MAPLE, SYRUP, MASY,
generator, validator, test or GPU run, and no commit. All values are uncalibrated trial inputs, not field validation.

A 64 m × 64 m (256 × 256 cells of 0.25 m) landscape with a user-specified black grama proxy (`BOER`, cover 0.05, diameter 0.10 ± 0.02 m) and honey
mesquite proxy (`PROGLA`, cover 0.10, diameter 1.0 ± 0.10 m) population, NOAA El Paso 1991–2020 average-climate forcing repeated for 2027,
and a design for 36 alternating wind/rain event pairs whose monthly rain sums equal the normals (223.012 mm/year).

## Read this first — what exists and what does not

* **No `leaf run case.toml` exists.** No generic loader reads `case.toml`, `landscape.toml`, `soil.toml`, `scheduler/settings.toml` or
  `scheduler/annual_plan.csv`. Those are **planning / API-mapping manifests** documenting the arguments a future script or driver would pass.
* **No end-to-end coupled year driver** is supplied or qualified. MASY native `gap_policy` supports only `refuse`/`idle`, and **idle gaps run no
  evaporation, transpiration or growth**; the annual plan therefore is a reviewed **design example** whose interevent biology a driver must fill at run
  time from actual accepted transport end times. The rain `recession_guard_s` is a **maximum**, not the realized duration.
* **No surface and no plants file.** `inputs/surface.npy` and `inputs/plants.csv` do not exist; no fake placeholders are provided. They require separate,
  explicitly authorised preparation steps.
* All relative paths in the manifests resolve from **this case directory**.

## File map

| File | Native / planning | Purpose |
|---|---|---|
| `README.md` | doc | this overview |
| `PARAMETERS.md` | doc | exhaustive parameter table with units, status (USER/AUTHOR/DERIVED/LIT…) and provenance |
| `SOURCES.md` | doc | primary links, shortcodes, what each source does and does not supply |
| `generation.json` | **native** `leaf.generation_config` v1 | domain, USER covers/diameters, seed, overlap=disallowed, limits |
| `species.json` | **native** `leaf.species_catalog` v1 | one complete record each for `BOER`, `PROGLA` (allometry, roots, porosity, growth, water use, reproduction, life expectancy, mortality, disturbance) |
| `climate/weather_2027.json` | **native** `masy.weather.v1` | 365 daily rows (2027-01-01…12-31) of `tmin_c`, `tmax_c` |
| `climate/monthly_normals.csv` | planning/derivation table | NOAA monthly normals, °F→°C and in→mm conversions, derived daily means |
| `case.toml` | planning manifest | calendar, runtime/session arguments, reproduction, snapshots (30-day cadence), checkpoints, budgets, limitations |
| `landscape.toml` | planning manifest | grid, finite boundaries, terrain recipe (not generated), vegetation references |
| `soil.toml` | planning manifest | SYRUP API mapping (layers, θ values, evaporation rule) |
| `scheduler/README.md` | doc | the alternating wind/rain year, recession guard, interevent rules |
| `scheduler/settings.toml` | planning manifest | design settings for wind, rain, pairing, interevent |
| `scheduler/annual_plan.csv` | design table | 72 rows = 36 wind/rain pairs with offsets and exact rain depths |
| `scheduler/examples/{wind,rain,interevent}.json` | **native** MASY event files | `masy.wind_event.v1`, `masy.rain_event.v1`, `masy.interevent.v1` |
| `inputs/README.md` | doc | expected-but-ungenerated inputs |
| `outputs/README.md` | doc | planned output layout |

## Key modelling statements

* **Vegetation** — `overlap=disallowed`: conditional uniform random centres under a large-canopy-first hard-core rejection process (not an unconditioned Poisson process),
  **lognormal** diameters with the requested arithmetic mean/SD (existing generator; not a truncated Gaussian), per-species union cover, finite edge-clipped
  domain (no toroidal biology, no invented edge mode). Later growth/recruitment may overlap. Expected counts ≈ 25 073 BOER and ≈ 516 PROGLA are *estimates*;
  exact sample moments and counts are not claimed until generation. Initial ages (730 d, 1825 d) are author assumptions; initial seasonal AGB is zero.
  Aboveground biomass is dry mass.
* **Water** — SYRUP owns persistent soil water; LEAF demand is biomass-specific; growth fraction U/D is applied once; no second rainfall/infiltration authority and no
  double debit; `added_water_m` is zero in interevent steps. The soil-evaporation partition (0.7 × shared PET) is **provisional** and must be qualified by SYRUP/the caller
  before full coupling; the ledger closes water, not energy.
* **Disabled / deferred** — surface-moisture feedback to wind is disabled; evolving soil-surface-to-soil remapping is deferred; abrasion is disabled until an explicit
  calibrated dose exists; the temperature gate does **not** implement freezing mortality or fire. Full MAPLE/SYRUP transport calibration and profile sets belong upstream
  and are not copied here (`trial_sandy_loam` must be bound by the water owner).
* **Reporting cadence** — `SnapshotWriter(cadence_days=30)`: an exact 30-day cadence (author choice, user selectable), approximately monthly, **not** calendar months.
  Initial, final and event snapshots are written separately; a due time crossed inside transport is recorded as missed, never interpolated.
* **Climate** — an average regional proxy (monthly normals repeated daily), not an observed year; PET is computed later by SYRUP `ecohyd_hargreaves` with a latitude bound in
  the MAPLE `case.yaml` (31.8° N rounded El Paso-area proxy).

## Not run / not validated

See `../../report.md` for the honest check scope. Native validators, importers, generators and tests were **not** executed by the author of these files.
