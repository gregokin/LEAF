# Parameters, units, status and provenance

Case `jornada_grass05_shrub10_256` (authored 2026-11-11). Paths are relative to the case directory.
**Nothing here is a field calibration or a validation.** Values are *trial hypotheses*; the status column says exactly what kind.

## Status legend

| Tag | Meaning |
|---|---|
| **USER** | Chosen by the user in the request (cover, diameters, 3–5× root range). Not a measured field estimate. |
| **AUTHOR** | Author's explicit trial/scenario/numerical choice with rationale. Legitimate provenance for a scenario value; **not** a literature measurement. |
| **DERIVED** | Arithmetic from tagged inputs; formula given. |
| **NOAA** | Published 1991–2020 El Paso normal (unit-converted only). |
| **LIT-DERIVED** | Published fitted quantity used through an author integration/derivation (never "the paper's number"). |
| **PROXY** | Published value from a different species/condition used as a stated proxy. |
| **EXAMPLE** | Copied from an existing LEAF example fixture (uncalibrated). |
| **REF** | Copied from the MAPLE reference *computational* case (not terrain data). |
| **API** | Value dictated by the native code/API schema (not science). |

Structural papers (STEPWAT2, ECOTONE/Peters 2002, FEIS summaries) motivate model *structure/qualitative plausibility* only; they are **never** the
source of an author-selected number. Short codes (e.g. `VitoSala2026`) are defined in [SOURCES.md](SOURCES.md).
"Aboveground biomass" always means **dry mass**, excluding roots and tissue water.

---

## 1. Domain, generation and expected counts (`generation.json`, `landscape.toml`)

| Path | Value | Units | Status | Provenance / note |
|---|---|---|---|---|
| `generation.json › domain.nx, ny` | 256, 256 | cells | REF | MAPLE reference `leaf_shrubs11_bimodal135_256_12h` |
| `domain.dx_m, dy_m` | 0.25, 0.25 | m | REF | same reference; gives 64 m × 64 m = 4096 m² (DERIVED) |
| `domain.origin_x_m, origin_y_m` | 0.0, 0.0 | m | REF | south-west corner, landscape frame |
| `seed` | 20261111 | – | AUTHOR | LEAF RNG `numpy.random.default_rng` seed |
| `overlap` | `disallowed` | – | AUTHOR | large-canopy-first hard-core rejection, global cross-species exclusion; not a Poisson process |
| `cover_definition` | `per_species_union` | – | API | only value supported |
| `cover_tolerance` | 0.001 | absolute fraction (0.1 percentage point) | AUTHOR | numerical acceptance, not science |
| `cover_subsamples` | 8 | per cell side | AUTHOR | cover raster 2048 × 2048 (DERIVED 256·8) |
| `max_iterations` | 12 | count iterations | AUTHOR | numerical |
| `limits.max_plants` | 200000 | plants | AUTHOR | resource cap |
| `limits.max_attempts` | 2000000 | centre proposals | AUTHOR | resource cap; exhaustion is reported, not promised away |
| `limits.max_cover_cells` | 4194304 | cells | AUTHOR/DERIVED | = 2048² |
| `limits.max_cover_bytes` | 8388608 | bytes | AUTHOR/DERIVED | two 2048² bitmaps (2 × 4 194 304); generator's own byte admission decides |
| `limits.max_working_bytes` | 1048576 | bytes | AUTHOR | resource cap |
| `time_days` | 0.0 | days | API | model time of the initial population |
| `species.BOER.target_cover` | 0.05 | fraction (per-species union) | USER | not a measured estimate |
| `species.BOER.diameter_mean_m / diameter_std_m` | 0.10 / 0.02 | m (arithmetic population mean / SD) | USER | **lognormal** sampling (existing generator) |
| `species.PROGLA.target_cover` | 0.10 | fraction | USER | not a measured estimate |
| `species.PROGLA.diameter_mean_m / diameter_std_m` | 1.0 / 0.10 | m | USER | **lognormal** sampling |
| `species.BOER.initial_age_days` | 730 | days | AUTHOR | trial assumption |
| `species.PROGLA.initial_age_days` | 1825 | days | AUTHOR | trial assumption |
| (generated) initial seasonal AGB `E` | 0 | kg | API | generator sets zero |
| Lognormal μ, σ BOER | −2.322195, 0.198042 | dimensionless (for ln(D/1 m)) | DERIVED | σ²=ln(1+SD²/mean²), μ=ln(mean)−σ²/2 |
| Lognormal μ, σ PROGLA | −0.004975, 0.099751 | dimensionless (for ln(D/1 m)) | DERIVED | same rule |
| Expected crown area | BOER 8.168×10⁻³ m², PROGLA 0.7933 m² | m² | DERIVED | π/4·(mean²+SD²) (E[D²]=mean²+SD²) |
| Expected initial count | **≈ 25 073 BOER, ≈ 516 PROGLA** | plants | DERIVED | N = A f / E[A_c], A = 4096 m²; **estimate only, not generated output**; exact sample mean/SD/count not promised until generation |

## 2. Species traits (`species.json`, one complete record per species)

Catalog-level: `schema` `leaf.species_catalog`, `schema_version` 1, `units` `{biomass: kg, length: m, time: days}` (API),
`root_layers` 3 (AUTHOR, = the three soil layers), `catalog_id` (AUTHOR label).

### 2a. Allometry (kg dry AGB → m)

| Path (per species) | BOER | PROGLA | Units | Status | Provenance / note |
|---|---|---|---|---|---|
| `allometry.diameter.a` | 0.9 | 1.6 | m kg^−b | EXAMPLE+AUTHOR | `D = a_d·B^b_d`; diameter law follows LEAF example fixtures |
| `allometry.diameter.b` | 0.45 | 0.4 | – | EXAMPLE+AUTHOR | same |
| `allometry.height.a` | 0.9 | 0.8 | m kg^−b | AUTHOR | `H = a_h·B^b_h`; PROGLA gives H = D/2 as in the MAPLE reference shrub geometry |
| `allometry.height.b` | 0.3 | 0.4 | – | AUTHOR | same |
| Initial biomass implied by mean D | 0.0075763685 | 0.3088161778 | kg dry AGB | DERIVED | B = (D/a_d)^(1/b_d) |
| Initial height implied | ≈ 0.208 | 0.5 | m | DERIVED | from H law |

These are **author trial power-law geometry anchors**, not published standing-AGB equations. USDA FEIS gives qualitative/height plausibility only; ANPP regressions must not be substituted for standing biomass.

### 2b. Roots and porosity

| Path | BOER | PROGLA | Units | Status | Provenance / note |
|---|---|---|---|---|---|
| `roots.lateral_radius_multiplier` | 3.0 | 5.0 | × canopy radius | USER range, AUTHOR endpoints | user's 3–5× range; grass = lower, shrub = upper endpoint |
| `roots.layer_fractions[0]` (0–0.2 m) | 0.39639133268704063 | 0.33283263377678235 | fraction of uptake | LIT-DERIVED | fraction = (β^lo − β^hi)/(1 − β^120), depth in **cm** |
| `roots.layer_fractions[1]` (0.2–0.5 m) | 0.3360854884442371 | 0.32719499000870794 | fraction | LIT-DERIVED | same |
| `roots.layer_fractions[2]` (0.5–1.2 m) | 0.26752317886872223 | 0.33997237621450965 | fraction | LIT-DERIVED | same; sums to 1 within 1e−9 |
| β (combined two-year fit) | 0.977 | 0.983 | dimensionless; calibrated for depth expressed in 1 cm increments | LIT | Vito & Sala 2026 tracer-uptake fits, `F(z)=1−β^z` |
| Omitted tail below 1.2 m | 0.0612840765 | 0.1277680127 | fraction of fitted curve | DERIVED | β^120; conditional normalisation over 1.2 m |
| `porosity.value` | 0.2 | 0.2 | fraction | EXAMPLE | uncalibrated constant, unused by LEAF ecology but **consumed by native MAPLE wind** |

Tracer absorption is an **uptake proxy**, not measured root density; fixed profile (no plasticity). The fitted-curve integration does **not** impose
the paper's binned "majority of shrub uptake below 50 cm" statement. Both species reach every layer. Lateral density uses the shared finite root builder. **Horizontal law:** uniform weight per cell positively overlapped by the root disk (including barely touched cells), clipped to the finite domain; it is **not** overlap-area weighting and **not** a measured radial root-density law, and it multiplies the per-species vertical fractions above. On CUDA, daily demand/withdrawals run on-device, but the weekly geometry/root rebuild is host/compiled with explicit CSR uploads (not fully GPU-resident).

### 2c. Growth (`growth.law = stepwat_requested_support_v1`)

| Trait | BOER | PROGLA | Units | Status | Provenance / note |
|---|---|---|---|---|---|
| `mature_biomass_kg` | 0.1 | 10.0 | kg dry AGB | AUTHOR | structural STEP precedent only |
| `intrinsic_rate_per_year` | 0.8 | 0.2 | 1/year | AUTHOR | inside native admitted 0..1 range; **not** the 295/yr Setaria diagnostic |
| `live_fraction` | 0.6 | 0.4 | fraction | AUTHOR | |
| `extra_growth_rate_per_year` | 0.1 | 0.05 | 1/year | AUTHOR | |
| `biomass_support_kg_year_per_m3` | 0.07024938531787846 | 0.10537407797681772 | kg·year/m³ | DERIVED | k = live_fraction/(365·q) at reference conditions; **support-balance prior, not a production WUE/yield** and not an original joint fit |
| `monthly_activity` Jan…Dec | [0,0,0,0.1,0.3,0.6,1,1,1,0.4,0.05,0] | [0,0,0,0.4,0.8,1,1,1,0.8,0.4,0.05,0] | fraction | AUTHOR | inspired by FEIS seasonal descriptions; not measured |
| `temperature_min_c / opt / max` | 5 / 30 / 45 | 5 / 30 / 45 | °C | AUTHOR | assumed cardinal temperatures, not measured fits |

Temperature multiplier gates demand/growth; there is **no freezing-mortality or fire** module. Growth fraction U/D is applied **once**, total across roots; no relocation of unmet demand.

### 2d. Water use (`water_use.law = biomass_specific_pet_v1`)

| Trait | BOER | PROGLA | Units | Status | Provenance / note |
|---|---|---|---|---|---|
| `demand_m3_per_kg_day` | 0.0234 | 0.0104 | m³ per kg **standing dry AGB** per day | PROXY | BOER: local descriptive reanalysis of moist-soil savanna grasses (Belovitch 2023 data; 23.4 L/kg/day, median of 30 individual means, harvest mass, not a printed universal coefficient). PROGLA: conditional **velvet mesquite seedling/pot** proxy (Peterson & Neofotis 2004; 10.3–10.5 L/kg/day, 10.4 chosen by the author), **not adult honey mesquite** |
| `reference_pet_mm_per_day` | 5.0 | 5.0 | mm/day | AUTHOR | normalisation only; not a measured site/experiment mean PET |

Demand `D = B · demand · (PET/reference_PET) · dt · activity · temperature_factor` — scaling by PET ratio, activity and temperature is LEAF's adaptation, not a source-calibrated equation.
Water is owned and debited by SYRUP exactly once.

### 2e. Mortality (`mortality.law = age_background_stress_v1`) — every number AUTHOR, uncalibrated

| Trait | BOER | PROGLA | Units | Status | Note |
|---|---|---|---|---|---|
| `life_expectancy.mean` / `unit` | 15 / years | 80 / years | years | AUTHOR | **metadata only, not a cutoff** |
| `age_hazard.weibull_shape` | 2.0 | 2.0 | – | AUTHOR | |
| `age_hazard.weibull_scale_years` | 16.92568750643269 | 90.27033336764101 | years | DERIVED | mean/Γ(1.5), Γ(1.5)=0.886226925 |
| `background_hazard_per_year` | 0.0 | 0.0 | 1/year | AUTHOR | isolates the explicit age law |
| `stress.metric` | `support_ratio` | `support_ratio` | – | AUTHOR | |
| `stress.low_threshold` | 0.25 | 0.25 | ratio | AUTHOR | |
| `stress.exempt_until_age_days` | 365 | 730 | days | AUTHOR | |
| `stress.limit_years` | 2 | 3 | years | AUTHOR | |
| `stress.counter_cap_years` | 10 | 10 | years | AUTHOR | ≥ limit_years |
| `stress.mortality_probability` | 0.25 | 0.1 | probability | AUTHOR | |
| `stress.biomass_loss_fraction` | 0.2 | 0.1 | fraction | AUTHOR | |

STEPWAT2 and ECOTONE (`Peters2002_ECOTONE`) are structural precedents only. Full model lifespan can be shortened by stress/disturbance; the metadata mean is never an age cutoff.

### 2f. Disturbance (`mechanical_response_v1`) — every threshold an AUTHOR testable hypothesis

| Trait | BOER | PROGLA | Units | Status |
|---|---|---|---|---|
| `burial.increment_basis / scale / curve` | net / height_fraction / linear | net / height_fraction / linear | – | API+AUTHOR |
| `burial.injury_threshold` | 0.1 | 0.1 | fraction of height | AUTHOR |
| `burial.injury_saturation` | 0.8 | 1.0 | fraction of height | AUTHOR |
| `burial.max_injury_fraction` | 0.5 | 0.5 | fraction | AUTHOR |
| `burial.lethal_depth_m` | null | null | m (off) | AUTHOR |
| `burial.lethal_height_fraction` | 1.0 | 1.5 | fraction of height | AUTHOR |
| `exposure.increment_basis / curve` | net / linear | net / linear | – | API+AUTHOR |
| `exposure.injury_threshold_m` | 0.02 | 0.05 | m | AUTHOR |
| `exposure.injury_saturation_m` | 0.15 | 0.5 | m | AUTHOR |
| `exposure.max_injury_fraction` | 0.5 | 0.4 | fraction | AUTHOR |
| `exposure.lethal_depth_m` | null | null | m (off) | AUTHOR |
| `exposure.dislodgement_depth_m` | 0.2 | 0.6 | m | AUTHOR |
| `abrasion` | **null (disabled)** | **null (disabled)** | – | AUTHOR |

Abrasion stays disabled until an explicit calibrated dose unit and response exist; sediment flux is never equated with dose. The establishment ground anchor is kept forever (no initial plant-height field). Surface→soil-layer remapping is deferred.

### 2g. Reproduction (`reproduction.law = simple_seed_clonal_v1`) — AUTHOR trial numbers

| Trait | BOER | PROGLA | Units |
|---|---|---|---|
| `seed` | null | see below | – |
| `seed.establishment_probability` | – | 0.05 | per patch, species, annual opportunity |
| `seed.max_recruits_per_patch` | – | 1 | count |
| `seed.initial_agb_dry_kg` | – | 0.002 | kg dry AGB (external credit) |
| `seed.initial_seasonal_fraction` | – | 0.0 | E/B |
| `clonal` | see below | null | – (resprouting ≠ independent ramets) |
| `clonal.min_age_days` | 365 | – | days |
| `clonal.min_agb_dry_kg` | 0.01 | – | kg |
| `clonal.probability` | 0.1 | – | per eligible parent per year |
| `clonal.max_daughters_per_parent` | 1 | – | count |
| `clonal.initial_agb_dry_kg` | 0.001 | – | kg, transferred from parent |
| `clonal.max_donation_fraction` | 0.1 | – | fraction of parent B |
| `clonal.dispersal_radius_m` | 0.25 | – | m |
| `clonal.max_attempts_per_daughter` | 8 | – | count |

Qualitative grass-vegetative / shrub-seed precedent: USDA FEIS (`Simonin2000_FEIS_BOER`, `Steinberg2001_FEIS_PROGLA`); the exact numbers are AUTHOR choices.
Native reproduction needs an explicit session `reproduction_patch_size_m` = 5 m (AUTHOR; `case.toml › reproduction`), not one trial per hydraulic cell.

## 3. Climate (`climate/`)

| Item | Value | Units | Status | Provenance |
|---|---|---|---|---|
| Monthly Tmin, Tmean, Tmax, precipitation | see `monthly_normals.csv` | °F, in (source) | NOAA | El Paso 1991–2020 normals |
| Conversion | C = (F − 32)·5/9; mm = 25.4·in | – | DERIVED | exact; temperatures rounded to 9 decimals |
| Annual precipitation | 223.012 | mm (8.78 in) | NOAA/DERIVED | sum of 12 monthly totals |
| Jul–Sep precipitation | 121.158 (≈ 54.3 %) | mm | DERIVED | |
| `weather_2027.json › rows` | 365 rows, 2027-01-01 … 2027-12-31 | `tmin_c`, `tmax_c` degC | DERIVED/AUTHOR | each month's converted Tmin/Tmax repeated every day; no interpolation, no observed year, no extremes |
| Daily mean temperature | (tmin+tmax)/2 | degC | API | MASY `arithmetic_mean_of_tmin_tmax`; differs slightly from the published Tmean |
| Relative humidity, shortwave radiation | **not supplied** | – | – | none invented (`humidity.supplied = none`, `radiation.supplied = false`) |
| `pet.method` | `ecohyd_hargreaves` | – | API | FAO-56 Eq. 52 precedent; runtime uses the actual SYRUP implementation; no PET computed here |
| `utc_offset` | −07:00 | – | AUTHOR | fixed MST, no DST |
| Latitude | 31.8 | degrees N | AUTHOR | rounded El Paso-area proxy; to be bound in MAPLE `case.yaml` `location.latitude_deg`; not a weather field, not derived from NOAA |
| Average regional climate | – | – | – | no claim of an exact Jornada 271.2 mm record |

## 4. Scheduler design (`scheduler/`) — all AUTHOR synthetic choices except monthly totals

| Item | Value | Units | Status |
|---|---|---|---|
| Wind/rain pairs, transport rows | 36 pairs, 72 rows | count | AUTHOR |
| Monthly rain counts | [2,2,3,3,3,3,4,4,4,3,3,2] | events | AUTHOR |
| Dates (day of month) | 2/month: 4,18; 3/month: 4,14,24; 4/month: 3,10,17,24 | day | AUTHOR |
| Rain start | 09:00 MST | local time | AUTHOR |
| Rain duration | 21600 (Jan,Feb,Mar,Nov,Dec); 14400 (Apr,May,Oct); 10800 (Jun–Sep) | s | AUTHOR |
| Monthly rain totals | 9.906, 10.160, 6.096, 4.318, 10.922, 18.542, 40.132, 42.418, 38.608, 14.986, 10.922, 16.002 | mm | NOAA (×25.4) |
| Per-event depth | monthly total / N at 0.001 mm; first N−1 rounded to nearest, last = remainder | mm | DERIVED (rule is AUTHOR) |
| Rain intensity | depth / (duration/3600) | mm/h | DERIVED |
| `recession_guard_s` | 21600 | s | AUTHOR; native **maximum**, not realized duration |
| Wind start / duration | 09:00 previous day / 43200 | s | AUTHOR |
| `shear_velocity_m_s` | 0.6 | m/s | AUTHOR (reference experimental forcing; not a wind climate) |
| `direction_deg` | 22.5 and 157.5 (to) | deg | AUTHOR (135° separation, vector mean east 90°) |
| Wind block durations | 2 × 21600 | s | AUTHOR (MAPLE reference toggles with 600 s blocks) |
| `water_profile_id` | `trial_sandy_loam` | – | AUTHOR (upstream owner must bind a qualified profile) |
| Interevent day | 86400 | s | API example |
| Background rain | none | – | AUTHOR |

## 5. Soil / SYRUP mapping (`soil.toml`)

| Item | Value | Units | Status | Provenance |
|---|---|---|---|---|
| `depth_edges_m` | [0, 0.2, 0.5, 1.2] | m | AUTHOR | proposed layers; same 3 as `root_layers` |
| `theta_residual` | 0.065 | m³/m³ | LIT | Carsel & Parrish 1988 sandy-loam prior (not site-specific) |
| `theta_saturation` | 0.41 | m³/m³ | LIT | same |
| `theta_field_capacity` | 0.20 | m³/m³ | AUTHOR | illustrative; **not** from the C&P table |
| `theta_evaporation_stress` | 0.15 | m³/m³ | AUTHOR | illustrative |
| `diffusivity_m2_s` | 2.0e−7 | m²/s | AUTHOR | illustrative; **not Ksat** |
| `evaporation_factor` | 1.0 | – | AUTHOR | existing-API illustrative |
| `gravity_drainage` | true | – | AUTHOR | existing-API illustrative |
| `theta_initial` | [0.10, 0.12, 0.14] | m³/m³ per layer, shallow→deep | AUTHOR | |
| Active cells, cell area | 65536, 0.0625 | cells, m² | DERIVED | |
| Potential soil evaporation | 0.7 × shared PET (mm/day → m/s) | – | AUTHOR | **provisional partition**, not E+T ≤ PET, not an energy partition; SYRUP/caller must qualify before full coupling |
| `added_water_m` between storms | 0 | m | AUTHOR | no extra rain/irrigation/run-on |

## 6. Runtime / output / budget (`case.toml`)

| Item | Value | Units | Status |
|---|---|---|---|
| Calendar start / end | 2027-01-01T00:00−07:00 / 2028-01-01T00:00−07:00 | – | AUTHOR |
| Model epoch | 0 | s | API |
| Backend | `cuda` (explicit device, no fallback) | – | AUTHOR |
| Root builder | `compiled` | – | AUTHOR |
| Daily accounting max | 86400 | s | API |
| Growth cadence | 7 | days | API max, AUTHOR choice |
| `max_edges` | 20000000 | edges | AUTHOR cap (not measured) |
| `max_bytes` | 2147483648 | bytes | AUTHOR cap |
| `reserve_bytes` | 67108864 | bytes | AUTHOR cap |
| `rng_seed` | 20261111 | – | AUTHOR |
| `reproduction_patch_size_m` | 5.0 | m | AUTHOR |
| `max_new_plants / max_total_records / max_attempts` | 20000 / 200000 / 200000 | counts | AUTHOR caps |
| Snapshot `cadence_days` | **30.0** | days | AUTHOR, user selectable; **exact 30 days, approximately monthly, not calendar months** |
| Initial / final / event snapshots | separate explicit writes | – | AUTHOR |
| Missed due times in transport | recorded in `skipped_due_times_days`, never interpolated | – | API |
| Snapshot path template | `outputs/plants_{index:04d}.csv` | – | AUTHOR |
| Checkpoint dir / format | `outputs/checkpoints/`, restart format 3 | – | AUTHOR / API |
| Event snapshot path template | `outputs/event_plants_{index:04d}.csv` (separate event writer) | – | AUTHOR planning choice / caller policy |
| Checkpoint `path_template` / policy / overwrite | `outputs/checkpoints/boundary_{index:04d}` / `after_each_accepted_transport_and_final` / `false` | – | AUTHOR planning choice / caller policy; not a new loader; nothing created |
| CSV `rain_intensity_mm_h` | 9-decimal display value; `rain_depth_mm` is authoritative (adapter rebuilds depth/(duration/3600); display error ≤ ~2e−9 mm) | mm/h | DERIVED |
| CSV `forcing_plus_rain_guard_end_offset_s` | nominal forcing end + rain guard (0 for wind); static arithmetic, not a terminal time | s | DERIVED |

## 7. Reference grid / terrain recipe (`landscape.toml`, not generated)

| Item | Value | Units | Status |
|---|---|---|---|
| Surface seed | 1173 | – | REF |
| Reference level | 0.6 | m | REF |
| Relief (peak-to-peak, exact) | 0.05 | m | REF |
| Correlation lengths x, y | 1.0, 1.0 | m | REF |
| Sediment voxels | 12 × 0.1 | count × m | REF |
| Generator | upstream `generate_spectral_gaussian_field` (not duplicated) | – | REF |

These are computational reference settings, **not observed El Paso terrain**. No `surface.npy` exists.
