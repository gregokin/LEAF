# `jornada_grass05_shrub10_256` — Jornada-style grass / shrub LEAF trial case

**Status (historical input-authoring record): input-authoring candidate, authored 2026-11-11 (task `trial_case_20261111`). At that time nothing was executed**: no LEAF, MAPLE, SYRUP, MASY,
generator, validator, test or GPU run, and no commit. This statement is superseded only for the separate initial-vegetation preview (see the last section); all other parts remain unrun. All values are uncalibrated trial inputs, not field validation.

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
  explicitly authorised preparation steps. (A plants file is a plain CSV with a companion `<stem>.metadata.json`; see the preview section below.)
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
  exact sample moments and counts are not claimed until generation. **Initial ages (current `generation.json`):** each plant's age is drawn independently and uniformly from 0 to its species'
  nominal life expectancy read from `species.json` — BOER **0–15 years** (0–5475 d), PROGLA **0–80 years** (0–29 200 d) (provisional author life expectancies; selector `uniform_to_life_expectancy`). This is the *initial sampling range only*, not a death age: the
  probabilistic mortality law is unchanged and a plant is not removed for reaching these ages. (Earlier versions used constant ages 730 d / 1825 d; the existing preview below was made with those and is not a realization of the uniform-age rule.) Initial seasonal AGB is zero.
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

Static-only verification note: the author reviewed these files manually (reading and hand arithmetic). Codex separately parsed the plain JSON/TOML/CSV files with standard-library tools and did static date, unit-conversion and event arithmetic. **No native LEAF/MAPLE/SYRUP/MASY module was invoked**: no native validator, importer, generator, model or test has been run, so this is input-only acceptance, not runtime qualification.

Other notes: the root-weighting law is uniform per cell positively overlapped by the root disk (not overlap-area weighting, not a measured radial density); on CUDA the weekly root rebuild is host/compiled with explicit CSR uploads, not fully GPU-resident. The MASY native `gap_policy` values are `refuse` and `idle` only.

## Literature-provenance revision (task `literature_parameters_20261111`, rev1)

The active `species.json` now has `catalog_id` `jornada_grass05_shrub10_litparam_rev2_20261111` (baseline `jornada_grass05_shrub10_trial_20261111`). The literature pass adopted published equations or model inputs for a **limited set of fields**, each with a direct source link in PARAMETERS.md (none is a Jornada field calibration):

* PROGLA diameter law `D = 1.1087951253470727·B^0.3351206434316354` (was 1.6, 0.4): Northup et al. 2005 Table 7, *P. glandulosa* total oven-dry aboveground mass vs crown area, La Copita, southern Texas, algebraically converted (log-median curve, no retransformation factor).
* PROGLA height law `H = 1.1015229296155495·B^0.2546916890080429` (was 0.8, 0.4, H = D/2): *P. velutina* proxy, McClaran et al. 2013 Table 2, Santa Rita, Arizona, composed with the Northup law. At the user's mean D = 1 m this gives B = 0.735 kg and H = 1.018 m (before: 0.309 kg, 0.5 m).
* PROGLA `live_fraction` 0.906 (was 0.4) = 1 − 9.4% standing dead wood (Northup 2005 Table 1; includes live wood, not a transpiring fraction) and k = 0.23867228661749212 (was 0.10537407797681772).
* BOER `intrinsic_rate_per_year` 0.474 (was 0.8), clonal daughter mass 0.0015096 kg (was 0.001) from the pinned STEPWAT2 `species.in` row `bogr` (*Bouteloua gracilis*, a related grass; a model input, not black grama and not a field estimate), and clonal minimum parent mass 0.015096 kg (was 0.01; author-derived, 10 × daughter mass). With this threshold most initial BOER plants (0.0076 kg at D = 0.10 m) cannot yet produce daughters.
* `soil.toml` `theta_field_capacity` 0.23 (was 0.20): FAO-56 Chapter 7 Table 19 / Example 35, sandy loam texture prior.

All other numbers are unchanged. Mature masses and mean lifespans stay author assumptions with explicit gaps (BOER 15 yr stands against 2–4 yr field clump persistence in Wright 1972 and Smith et al. 2004; this is a validation target). Nothing was run. The exact pre-edit catalog is kept as `preview/species_at_initialization.json`; the saved preview was made with that frozen catalog at the old allometry and constant ages, its historical files are untouched, the active catalog has a different hash, and any run using the new catalog needs a new initialization (not run). See `preview/README.md`.

## Initial-vegetation preview (initialization only)

Image: [`preview/vegetation_initial.png`](preview/vegetation_initial.png) · details and commands: [`preview/README.md`](preview/README.md).
Status: renderer authored; the image and records exist only once the reviewed script has been run. It shows the initial placement
and initial surface only (no growth or transport). It is a separate preview population: the normal inputs `inputs/plants.csv` and
`inputs/surface.npy` remain ungenerated and this preview is not adopted as production input.

**The existing preview is the EARLIER initialization.** It was generated with the earlier constant initial ages (730 d BOER, 1825 d PROGLA), before the species-linked uniform-age rule (above, current `generation.json`) was adopted. Its plant rows, image, surface, original `preview_provenance.json` and initialization report are preserved unchanged; it is **not** a realization of the new rule, and no new initialization was run. The new generation config applies to the next explicitly requested initialization.

Plant-file convention (project rule: no embedded metadata): the preview plant table `preview/initial_plants.csv` is a **plain**
nine-column CSV (header + one nine-field row per plant; no `#` line, JSON or comments). Its metadata — time (0 d), domain, species
catalog hash and table, row count, units, `initial` declaration, provenance and the CSV SHA-256 — is the companion
`preview/initial_plants.metadata.json`. The pair is never separated: LEAF's `read_plant_csv` loads and verifies the companion and
refuses a CSV that does not match it. The same convention applies to the future production `inputs/plants.csv`
(`inputs/plants.metadata.json`, written by `python -m leaf generate` / `init`; a handwritten list needs no companion). The
preview population itself is generated time-zero data: converting the file layout moved only the former first line into the
companion; plant rows and the image were not regenerated or altered. The existing picture was made before this change (it was
not produced by the modified helper): `preview/preview_provenance.json` is its preserved original record and carries the
pre-split CSV hash; the current pair hashes and the row/image preservation checks are in the separate addendum
`preview/plant_files_migration.json` (see `preview/README.md`).
