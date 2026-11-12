# Report — trial_case_20261111 (revision 3: documentation corrections on top of revision 2)

## Revision 3 changes (documentation/plain-text only)

No scientific parameter number, weather row (365), event date/depth/duration (72 rows) or native format key set was changed. No execution of any kind; Read/Edit/Write only; uncommitted.
1. Stale `../` references in `species.json` (case-root file) now name case-root files (`soil.toml`, `generation.json`, `case.toml`, `PARAMETERS.md`, `SOURCES.md`); the same for the weather provenance text (`landscape.toml`). The root README's `../../report.md` link was removed and replaced by a self-contained static-only verification note.
2. `annual_plan.csv` `forcing_ref` values are now case-root paths `scheduler/examples/wind.json` / `scheduler/examples/rain.json`, documented as PATTERN references (each row builds a fresh event; July example not replayed). Display-name vs manifest-path convention clarified in `scheduler/README.md`.
3. CSV column `latest_block_end_offset_s` renamed `forcing_plus_rain_guard_end_offset_s` (numbers unchanged; = nominal forcing end + rain guard, zero guard for wind). Documented as static arithmetic, not a terminal time; wind settlement may reduce/eliminate the nominal 12 h gap; start collisions refuse. `scheduler/README.md` and `settings.toml` updated.
4. `rain_depth_mm` documented as the authoritative budget; `rain_intensity_mm_h` is a 9-decimal display value (error ≤ ~2e-9 mm), rebuilt by an adapter from depth/duration.
5. `soil.toml` header shows keyword-style real signatures (`cell_area_m2=`, `theta_by_layer=`); `theta_initial` documented as a local name; texture text moved to a new `[metadata]` table so `[soil_layers]` holds only API arguments; native evaporation multiplier (0 at/below residual, (θ/stress)² below stress, 1 above) stated, formula unchanged.
6. `landscape.toml`: reference level shown as 0.5·12·0.1 = 0.6 m, an arbitrary computational datum.
7. `PARAMETERS.md`: β and lognormal μ/σ are dimensionless; `SOURCES.md` now states Carsel & Parrish's table does contain Ksat but it is not used as the case's diffusivity (field capacity and diffusivity are author assumptions).
8. Horizontal root law (uniform weight per cell positively overlapped by the root disk, finite-domain clipped, not area-weighted, not a measured density; multiplies the vertical fractions) and CUDA scope (daily demand/withdrawal on-device; weekly root rebuild host/compiled with explicit CSR uploads) added to `species.json` roots notes, `PARAMETERS.md`, `case.toml`.
9. Concrete optional output examples added (`outputs/event_plants_{index:04d}.csv`; checkpoint `outputs/checkpoints/boundary_{index:04d}`, policy `after_each_accepted_transport_and_final`, `overwrite=false`) as AUTHOR planning/caller-policy choices; 30-day main cadence unchanged; nothing created.
10. Peters (2002) ECOTONE citation (<https://doi.org/10.1016/S0304-3800(01)00460-4>) added to `SOURCES.md`, `PARAMETERS.md` and the `species.json` mortality notes/references, as structure only.
Disposition kept: MASY `gap_policy` is `refuse`/`idle` (not `reject`); evaporation formula unchanged.

Revision-3 check scope: manual review of the edits only. Codex's separate standard-library parsing and static arithmetic are Codex's, not mine. NOT RUN by me: any parser, native validator/importer, generator, model or test; the edited JSON/TOML/CSV were not re-parsed after editing.

---

# Revision 2 report (retained)

Baseline Git `1e468e1410992430e0aa5045ca44637b382ac0e9` plus the immutable `baseline.json` / `source-manifest.json`. Candidate left **uncommitted**.
Authoring only, with the ordinary Read / Edit / Write tools. **No model, generator, validator, importer, test, GPU job, package install, Git command or network access was run or used.**
Reads were confined to `SOURCE_INDEX.txt`, `context/**`, this prompt and my own outputs. No file outside the allowed write list was touched (root `AGENTS` scaffold, context, launcher, task controls and source manifest are unmodified).

## Files written (all under `cases/jornada_grass05_shrub10_256/` unless noted)

Native-format inputs
* `generation.json` — `leaf.generation_config` v1
* `species.json` — `leaf.species_catalog` v1 (one complete record each for `BOER`, `PROGLA`; `root_layers` 3)
* `climate/weather_2027.json` — `masy.weather.v1`, 365 rows
* `scheduler/examples/wind.json`, `rain.json`, `interevent.json` — `masy.wind_event.v1`, `masy.rain_event.v1`, `masy.interevent.v1`

Planning manifests / tables (explicitly NOT native schemas, no loader or CLI)
* `case.toml`, `landscape.toml`, `soil.toml`, `scheduler/settings.toml`
* `scheduler/annual_plan.csv` (72 rows, design example), `climate/monthly_normals.csv` (12 rows)

Documentation
* `README.md`, `PARAMETERS.md`, `SOURCES.md`, `scheduler/README.md`, `inputs/README.md`, `outputs/README.md`
* `report.md` (candidate root, this file)

Intentionally **not** created: `inputs/surface.npy`, `inputs/plants.csv`, any generated population/terrain, any executable source.

## Assumptions, decisions and limitations (as authored)

* No `leaf run case.toml` driver exists; TOML/CSV files are labelled planning/API manifests. `annual_plan.csv` is a reviewed **design example**, not a native `masy.scenario.v1`. MASY `idle` gaps do no E/T/growth, so interevent biology must be filled at run time from actual accepted transport end times; no `gap_policy = interevent` is invented. The rain `recession_guard_s` (21600) is a **maximum**, not the realized duration; initial and final interevent periods, no overlap (refuse, never truncate/shift), no catch-up during wind/rain/recession, no second rainfall/infiltration authority and no double debit are stated in `scheduler/` and `soil.toml`.
* **Revision-2 correction applied:** the existing generator samples **lognormal** diameters with the requested arithmetic mean and SD (`leaf/initialization.py`, confirmed in the supplied source); it is not a truncated Gaussian. Documented that way, with derived μ/σ. The E[D²]=mean²+SD² count estimate (≈ 25 073 BOER, ≈ 516 PROGLA) is kept and labelled as an estimate only.
* **Snapshot cadence:** `SnapshotWriter(cadence_days=30)` is an exact 30-day AUTHOR choice, approximately monthly, not calendar months; initial/final/event snapshots are separate; due times crossed inside transport are recorded as missed (never interpolated). Event snapshots need a caller-side second writer or `write_plant_csv` because `SnapshotWriter` has only initial/cadence/final kinds.
* Surface soil feedback to wind disabled; surface-to-soil remapping deferred; abrasion `null` (disabled); temperature gate is not a freezing-mortality/fire model; soil-evaporation partition 0.7 × PET is provisional and must be qualified by SYRUP/the caller; full MAPLE/SYRUP transport calibration/profiles are not copied (`trial_sandy_loam` must be bound upstream).
* All parameters are tagged USER / AUTHOR / DERIVED / NOAA / LIT-DERIVED / PROXY / EXAMPLE / REF in `PARAMETERS.md`; no author-selected number is attributed to a paper that only motivates structure.
* Judgement calls to review: (1) rain-event depth rounding — first N−1 events use total/N rounded to the nearest 0.001 mm, half rounding up (only August's 10.6045 is a tie, giving 10.605 ×3 and a 10.603 remainder), last event takes the remainder so each month sums exactly; (2) converted temperatures are written to 9 decimals; (3) `event_id`, `weather_id` and `water_profile_id` strings use lowercase/underscore patterns modelled on the supplied examples, but `masy/event_files.py` was not in the bounded context so its exact ID pattern could not be checked; (4) `humidity.supplied = none` is used per the facts document and the supplied `weather.py` accepts exactly `{"supplied"}` for that mode.

## Checks — MANUAL REVIEW ONLY (reading/hand arithmetic; none was executed by a tool that could fail)

Manually reviewed against the supplied sources:
* `species.json` keys vs `leaf/traits.py`, `growth.py`, `demography.py`, `disturbance.py`, `recruitment.py` strict key sets and ranges (growth traits exactly the 9 keys; water-use exactly 2 keys; mortality/disturbance/reproduction trait sets exact; section keys only from `law,traits,status,notes,units`; diameter/height objects only `a,b`).
* Root `layer_fractions` have length 3 and sum to 1 within about 1e-15 by hand addition; derived values re-computed by hand: initial biomass/height, `biomass_support_kg_year_per_m3` = live_fraction/(365·q), Weibull scales = mean/Γ(1.5), β^120 tails, lognormal μ/σ, expected counts.
* `generation.json` keys vs `leaf/initialization.py` (`TOP_KEYS`, `SPECIES_KEYS`, limits keys, domain keys); cover bitmap arithmetic 2048² = 4 194 304 cells, ×2 = 8 388 608 bytes.
* `weather_2027.json` keys/units/declarations vs `masy/weather.py`; row count 365 confirmed from the file's line numbers (rows on lines 23–387) and spot-checked month boundaries (Jan 31→Feb 1, Mar 31→Apr 1, Jun 30→Jul 1, Oct 31→Nov 1); each row holds only `date`, `tmin_c`, `tmax_c`.
* NOAA conversions (°F→°C, in→mm) re-derived by hand; monthly rain sums equal 223.012 mm (Jul–Sep 121.158 mm); per-month event depths re-summed to the monthly totals.
* `annual_plan.csv`: 73 lines (header + 72 rows) confirmed from line numbers; strict wind→rain alternation over 36 pairs; offsets hand-computed (rain start = day-of-year·86400 + 32400; wind start = rain start − 86400; wind nominal end 21:00); wind end → rain start nominal gap 43 200 s; static forcing-plus-guard end (nominal + 21 600 s; column now `forcing_plus_rain_guard_end_offset_s`) precedes every following wind start (static arithmetic, not measured transport).
* The three event examples vs the supplied MASY example files (exact key sets).

## Checks NOT RUN (not authorised; no claim of success)

* JSON / TOML / CSV **parser** runs of any file (all parse-ability claims are by reading only).
* `leaf.traits.SpeciesCatalog` loading / `validate_catalog_document`; `validate_generation_config`; `generate_population`; `leaf.growth.compile_species`; demography / disturbance / recruitment compile steps.
* `masy.weather.WeatherSeries.from_input`, `masy.event_files.parse_event_file`, `masy.scenario` loading; SYRUP `weather_pet` PET evaluation; `ecohyd_hargreaves`.
* SYRUP `soil_layers` / `prepare_column` / `uniform_state` / `prepare_transpiration`; `SnapshotWriter`; any `IntereventSession` use.
* Any unit, property, regression or GPU test; any generator, surface generation, plant generation, MAPLE/SYRUP/MASY/LEAF execution.
* Any independent numerical recomputation by code (all arithmetic above is hand arithmetic and should be re-checked by Codex).
* Any Git operation (status, diff, commit, push).
