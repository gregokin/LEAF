# Parameters, units, status and provenance

Case `jornada_grass05_shrub10_256` (authored 2026-11-11; literature revision `literature_parameters_20261111` rev2, 2026-11-12). Paths are relative to the case directory.
**Nothing here is a field calibration or a validation.** Values are *trial hypotheses* unless the row says a published equation or model input is used; every
row's last column gives the original source (full URL), the ORIGINAL species/site/treatment, the value and units, and whether it is measured, derived, a proxy,
a model input or an assumption. Where no compatible estimate exists the row says so (no paper is implied).
Short codes (e.g. `VitoSala2026`) are defined in [SOURCES.md](SOURCES.md); derivations and checked-versus-inferred detail: `research_notes.md` in the task directory.
"Aboveground biomass" always means **standing dry mass** in kg, excluding roots and tissue water (not fresh mass, not ANPP).

## Status legend

| Tag | Meaning |
|---|---|
| **USER** | Chosen by the user in the request (cover, diameters, 3–5× root range). Not a measured field estimate. |
| **AUTHOR** | Author's explicit trial/scenario/numerical choice with rationale. Legitimate provenance for a scenario value; **not** a literature measurement. |
| **DERIVED** | Arithmetic from tagged inputs; formula given. |
| **LIT** | Published value used as stated (unit conversion only). |
| **LIT-DERIVED** | Published fitted quantity transformed algebraically to a native coefficient (never "the paper's number"). |
| **PROXY** | Published value from a different species/condition used as a stated proxy. |
| **MODEL-PROXY** | Input of another species in a pinned simulation model (not a field estimate). |
| **NOAA** | Published 1991–2020 El Paso normal (unit-converted only). |
| **EXAMPLE** | Copied from an existing LEAF example fixture (uncalibrated). |
| **REF** | Copied from the MAPLE reference *computational* case (not terrain data). |
| **API** | Value dictated by the native code/API schema (not science). |
| **ASSUMED-SHAPE** | Explicit prior curve (activity/cardinal temperatures) whose calendar/qualitative limits come from a linked field source; never a measured coefficient. |
| **PROXY-CONSISTENT** | Value coincides with a measured quantity of a related population/trait (stated) but the native trait is a different quantity. |
| **TEXTURE-PRIOR** | Texture-class value from an authoritative guideline or soil-class database; not site-specific. |

## Numeric changes in rev2 versus the frozen baseline catalog (the only changed numbers)

| Field | Baseline | Rev2 | Basis |
|---|---:|---:|---|
| PROGLA `allometry.diameter.a` / `.b` | 1.6 / 0.4 | 1.1087951253470727 / 0.3351206434316354 (≈ 1.109 / 0.335) | Northup et al. 2005 Table 7, §2a |
| PROGLA `allometry.height.a` / `.b` | 0.8 / 0.4 | 1.1015229296155495 / 0.2546916890080429 (≈ 1.102 / 0.255) | McClaran et al. 2013 Table 2 (*P. velutina* proxy) composed with the Northup law, §2a |
| PROGLA `growth.live_fraction` | 0.4 | 0.906 | Northup et al. 2005 Table 1, §2c |
| PROGLA `growth.biomass_support_kg_year_per_m3` | 0.10537407797681772 | 0.23867228661749212 (≈ 0.239) | DERIVED k = 0.906/(365 × 0.0104), §2c |
| BOER `growth.intrinsic_rate_per_year` | 0.8 | 0.474 | pinned STEPWAT2 `species.in` row `bogr` (*Bouteloua gracilis*), §2c |
| BOER `reproduction.clonal.initial_agb_dry_kg` | 0.001 | 0.0015096 | `bogr` seedling size 1.5096 g, §2g |
| BOER `reproduction.clonal.min_agb_dry_kg` | 0.01 | 0.015096 | AUTHOR-DERIVED 10 × daughter mass, §2g |
| `soil.toml › theta_field_capacity` | 0.20 | 0.23 | FAO-56 Table 19 / Example 35, §5 |

Every other numeric value is unchanged. The active catalog is `jornada_grass05_shrub10_litparam_rev2_20261111` (baseline `jornada_grass05_shrub10_trial_20261111`, frozen as
[`preview/species_at_initialization.json`](preview/species_at_initialization.json)). The saved preview population used the frozen catalog and the old geometry/ages; see [preview/README.md](preview/README.md).

---

## 1. Domain, generation and expected counts (`generation.json`, `landscape.toml`) — no literature parameters

All rows here are user requests, MAPLE-reference computational settings, API values or numerical author choices. No published ecological estimate is involved; none is cited.

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
| `species.BOER.initial_age_days` | `{"distribution": "uniform_to_life_expectancy"}` → uniform **0 … 15 years** (0 … 5475 d) | days (bound read from the catalog) | USER policy, bound = AUTHOR `life_expectancy` | each plant's age drawn independently; bound read from `species.json › BOER › life_expectancy`, not duplicated. Initial sampling range only: **not a death age** (§2e). Field clump persistence is much shorter (§2e validation gap) |
| `species.PROGLA.initial_age_days` | `{"distribution": "uniform_to_life_expectancy"}` → uniform **0 … 80 years** (0 … 29 200 d) | days | USER policy, bound = AUTHOR `life_expectancy` | same rule, `species.json › PROGLA › life_expectancy` |
| Existing preview population | constant 730 d (BOER) / 1825 d (PROGLA) | days | historical | the preview in `preview/` was generated **before** the uniform-age rule with the FROZEN earlier catalog (old PROGLA allometry); not a realization of this catalog (a new initialization is a separate request, not run) |
| (generated) initial seasonal AGB `E` | 0 | kg | API | generator sets zero |
| Lognormal μ, σ BOER | −2.322195, 0.198042 | dimensionless (for ln(D/1 m)) | DERIVED | σ²=ln(1+SD²/mean²), μ=ln(mean)−σ²/2 |
| Lognormal μ, σ PROGLA | −0.004975, 0.099751 | dimensionless (for ln(D/1 m)) | DERIVED | same rule |
| Expected crown area | BOER 8.168×10⁻³ m², PROGLA 0.7933 m² | m² | DERIVED | π/4·(mean²+SD²) (E[D²]=mean²+SD²); depends on the diameter targets only, so unchanged by the allometry revision |
| Expected initial count | **≈ 25 073 BOER, ≈ 516 PROGLA** | plants | DERIVED | N = A f / E[A_c], A = 4096 m²; **estimate only, not generated output** |

## 2. Species traits (`species.json`, one complete record per species)

Catalog-level: `schema` `leaf.species_catalog`, `schema_version` 1, `units` `{biomass: kg, length: m, time: days}` (API), `root_layers` 3 (AUTHOR, = the three soil layers),
`catalog_id` `jornada_grass05_shrub10_litparam_rev2_20261111` (AUTHOR label).

### 2a. Allometry (kg dry AGB → m)

| Path | BOER | PROGLA | Units | Status | Source, original species / context, value |
|---|---|---|---|---|---|
| `allometry.diameter.a` | 0.9 | **1.1087951253470727** (≈ 1.109) | m kg^−b | BOER: EXAMPLE+AUTHOR; PROGLA: LIT-DERIVED | **BOER**: LEAF example fixture `examples/phase3/species.json` grass diameter law; no compatible black grama equation for total standing dry mass vs crown diameter was found (Nafus et al. 2009 is current-year mass vs basal diameter, a different quantity, not adopted). **PROGLA**: Northup, Zitzer, Archer, McMurtry & Boutton (2005), *J. Arid Environ.* 62:23–43, <https://doi.org/10.1016/j.jaridenv.2004.09.019> (PDF <https://cales.arizona.edu/research/archer/publications/refereed_journal_articles/Northup-et-al_2005_JAridEnv.pdf>), **Table 7 p. 34**, ORIGINAL species *Prosopis glandulosa* (target species), La Copita Research Area near Alice, southern Texas, mean rain ≈ 680 mm/yr, sampled May–Aug 1994/1995: ln(TB_g) = 6.960 + 1.492 ln(CA_m²), n = 37 (df 36), R² = 0.98. TB = TOTAL oven-dry (70 °C) aboveground mass (leaves + live wood + standing dead wood, no roots); CA = projected elliptical **crown** area from longest/perpendicular axes. Log-space median used (source retransformation factor 1.006 intentionally **not** applied): B_kg = e^6.960/1000 · CA^1.492 = 1.0536335572423194 · CA^1.492; with D = √(4CA/π): D = a·B^b, a = 1.1087951253470727, b = 1/2.984 = 0.3351206434316354. Algebraic inversion = geometry approximation, not a separate inverse fit. A wetter Texas site, **not a Jornada calibration**; Table 1 gives means only, so no valid prediction range is claimed |
| `allometry.diameter.b` | 0.45 | **0.3351206434316354** (≈ 0.335) | – | BOER: EXAMPLE+AUTHOR; PROGLA: LIT-DERIVED | **BOER**: same fixture. **PROGLA**: b = 1/2.984 from the Northup 2005 Table 7 exponent 1.492 (above) |
| `allometry.height.a` | 0.9 | **1.1015229296155495** (≈ 1.102) | m kg^−b | BOER: AUTHOR; PROGLA: PROXY-derived | **BOER**: author trial anchor; no compatible estimate found. **PROGLA**: McClaran, McMurtry & Archer (2013), *J. Arid Environ.* 88:39–42, <https://doi.org/10.1016/j.jaridenv.2012.08.015> (PDF <https://cales.arizona.edu/research/archer/publications/refereed_journal_articles/McClaran-et-al_2012_JAridEnv.pdf>), **Table 2 p. 41**, ORIGINAL species *Prosopis velutina* (velvet mesquite, a related species), Santa Rita Experimental Range near Tucson, Arizona, June 2004: HT_m = e^0.11 · CA^0.38 (CF 1.00), R² = 0.91, n = 31, sample CA 0.1–77 m², H 0.5–6.9 m. Composed with the Northup law CA = (B/1.0536335572423194)^(1/1.492): a_h = e^0.11 · 1.0536335572423194^(−0.38/1.492) = 1.1015229296155495. Mixed-study deterministic proxy; no target-species calibration |
| `allometry.height.b` | 0.3 | **0.2546916890080429** (≈ 0.255) | – | BOER: AUTHOR; PROGLA: PROXY-derived | **BOER**: author trial anchor. **PROGLA**: b = 0.38/1.492 (McClaran Table 2 exponent 0.38 ÷ Northup exponent 1.492) |
| Initial biomass implied by mean D | 0.0075763685 (≈ 0.00758) | **0.7347907779912205** (≈ 0.735) | kg dry AGB | DERIVED | B = (D/a_d)^(1/b_d) at the USER mean diameters 0.10 m / 1.0 m (B at the mean D, not the mean B). PROGLA previously 0.3088161778 |
| Initial height implied | ≈ 0.208 | **1.0183722297676923** (≈ 1.018) | m | DERIVED | H = a_h·B^b_h; PROGLA is no longer H = D/2 (previously 0.5 m). At D = 2 m: B = 5.813493672465879 kg (≈ 5.81), H = 1.7246038235919356 m (≈ 1.72) |

The mesquite laws are published or related-species fits applied to the user diameters, **not** a Jornada field calibration; no claim is made that future seedlings or plants outside the sampled ranges are qualified. Grass geometry remains a gap.

### 2b. Roots and porosity

| Path | BOER | PROGLA | Units | Status | Source, original species / context, value |
|---|---|---|---|---|---|
| `roots.lateral_radius_multiplier` | 3.0 | 5.0 | × canopy radius | USER range, AUTHOR endpoints | the user's 3–5× range; grass = lower, shrub = upper endpoint (author choice); no literature estimate used |
| `roots.layer_fractions[0]` (0–0.2 m) | 0.39639133268704063 (≈ 0.396) | 0.33283263377678235 (≈ 0.333) | fraction of uptake | LIT-DERIVED | Vito & Sala (2026), *Ecology* 107:e70384, <https://doi.org/10.1002/ecy.70384>; ORIGINAL species BOER = *Bouteloua eriopoda* (black grama), PROGLA = *Prosopis glandulosa* (honey mesquite); combined-two-years isotope-tracer uptake fit F(z)=1−β^z, z in cm (site/treatment details not re-verified in rev2). fraction = (β^lo − β^hi)/(1 − β^120), edges 0/20/50/120 cm |
| `roots.layer_fractions[1]` (0.2–0.5 m) | 0.3360854884442371 (≈ 0.336) | 0.32719499000870794 (≈ 0.327) | fraction | LIT-DERIVED | same paper/derivation, edges 20–50 cm |
| `roots.layer_fractions[2]` (0.5–1.2 m) | 0.26752317886872223 (≈ 0.268) | 0.33997237621450965 (≈ 0.340) | fraction | LIT-DERIVED | same paper/derivation, edges 50–120 cm; fractions sum to 1 within 1e−9 |
| β (combined two-year fit) | 0.977 | 0.983 | dimensionless; for depth in 1 cm increments | LIT | Vito & Sala 2026, <https://doi.org/10.1002/ecy.70384>, black grama (0.977) and honey mesquite (0.983) |
| Omitted tail below 1.2 m | 0.0612840765 | 0.1277680127 | fraction of fitted curve | DERIVED | β^120; conditional normalisation over 1.2 m (same paper) |
| `porosity.value` | 0.2 | 0.2 | fraction | EXAMPLE | uncalibrated LEAF example fixture `examples/phase3/species.json`; no literature value; consumed by native MAPLE wind |

Tracer absorption is an **uptake proxy**, not measured root density; fixed profile (no plasticity). The fitted-curve integration does **not** impose the paper's binned "majority of shrub uptake below 50 cm" statement. Both species reach every layer. **Horizontal law:** uniform weight per cell positively overlapped by the root disk (including barely touched cells), clipped to the finite domain; not overlap-area weighting and not a measured radial root-density law; it multiplies the per-species vertical fractions. On CUDA, daily demand/withdrawals run on-device, but the weekly geometry/root rebuild is host/compiled with explicit CSR uploads (not fully GPU-resident).

### 2c. Growth (`growth.law = stepwat_requested_support_v1`)

| Trait | BOER | PROGLA | Units | Status | Source, original species / context, value |
|---|---|---|---|---|---|
| `mature_biomass_kg` | 0.1 | 10.0 | kg dry AGB (saturation cap of standing dry mass) | AUTHOR (both) | **Gap, no compatible estimate.** BOER: the pinned STEP `bogr` maxbio 15.096 g (<https://github.com/DrylandEcology/STEPWAT2/blob/003cb9b4adbf13d64a39d6d62fd95280adac27fc/testing.sagebrush.master/Stepwat_Inputs/Input/species.in>, *B. gracilis* model input) is not a demonstrated black grama maximum and would lie below many admissible initial plants — not adopted. PROGLA: inverting the Northup 2005 law gives B = 10 kg at CA ≈ 4.5 m², D ≈ 2.4 m, H ≈ 2.0 m, which is arithmetic only and not evidence for a saturation cap (the earlier ">10 kg" Ludwig 1975 search-snippet is **not** used) |
| `intrinsic_rate_per_year` | **0.474** | 0.2 | 1/year | BOER: MODEL-PROXY; PROGLA: AUTHOR | **BOER**: pinned STEPWAT2 revision 003cb9b4, `species.in` row `bogr` = *Bouteloua gracilis* (blue grama, related C4 grass; sagebrush/shortgrass-steppe model input, **not** black grama and **not** a field estimate), irate = 0.474 (file header credits Bradford & Lauenroth 2006 Table A1.4, underlying table not checked): <https://github.com/DrylandEcology/STEPWAT2/blob/003cb9b4adbf13d64a39d6d62fd95280adac27fc/testing.sagebrush.master/Stepwat_Inputs/Input/species.in> (raw <https://raw.githubusercontent.com/DrylandEcology/STEPWAT2/003cb9b4adbf13d64a39d6d62fd95280adac27fc/testing.sagebrush.master/Stepwat_Inputs/Input/species.in>, SHA-256 c20a67b0a8095f4c9c920c3f73a711437ecc43039a934a1f4a351bc864bce2f3). Inside the native admitted 0..1 range (not raised or clamped). Previously 0.8 [AUTHOR]. **PROGLA**: no compatible estimate; not the 295/yr Setaria diagnostic |
| `live_fraction` | 0.6 | **0.906** | fraction of total dry AGB | BOER: AUTHOR; PROGLA: LIT | **BOER**: no compatible estimate; unchanged. **PROGLA**: Northup et al. 2005 (<https://doi.org/10.1016/j.jaridenv.2004.09.019>) **Table 1 p. 28**, *P. glandulosa*, La Copita southern Texas, n = 37: leaves 10.1%, live stems 80.4%, standing dead wood 9.4% of total oven-dry aboveground mass (rounds to 99.9%); live_fraction = 1 − 0.094. Includes live **wood**: it is **not** leaf area, metabolically active tissue or a measured transpiring fraction; a constant is a size/site approximation. q (§2d) is per TOTAL B, so live_fraction is not multiplied into q again |
| `extra_growth_rate_per_year` | 0.1 | 0.05 | 1/year | AUTHOR | no compatible estimate; the STEP `ratep`/`vegprop` probabilities in `species.in` have different semantics and were not mapped |
| `biomass_support_kg_year_per_m3` | 0.07024938531787846 (≈ 0.0702) | **0.23867228661749212** (≈ 0.239) | kg·year/m³ | DERIVED | k = live_fraction/(365·q) at reference PET: BOER 0.6/(365 × 0.0234) (unchanged); PROGLA 0.906/(365 × 0.0104) = 0.23867228661749212 (was 0.10537407797681772 at live_fraction 0.4). **Support-balance prior, not a production WUE/yield** and not a joint fit; q from §2d (Belovitch 2023 grass proxy <https://doi.org/10.1111/nph.18909>; Peterson & Neofotis 2004 *P. velutina* seedling proxy <https://doi.org/10.1007/s00442-004-1688-y>) |
| `monthly_activity` Jan…Dec | [0,0,0,0.1,0.3,0.6,1,1,1,0.4,0.05,0] | [0,0,0,0.4,0.8,1,1,1,0.8,0.4,0.05,0] | fraction 0..1 | ASSUMED-SHAPE (both) | **BOER**: FEIS black grama review (Simonin 2000, <https://research.fs.usda.gov/feis/species-reviews/boueri>, Phenology, Chihuahuan Desert): growth begins late April–early May, flowering late Aug–early Sep, fruiting late Sep–early Oct; Jornada 1925–1935 growth periods 0–176 days, rainfall 1.57–11.95 in (40–304 mm). **PROGLA**: FEIS honey mesquite review (Steinberg 2001, <https://research.fs.usda.gov/feis/species-reviews/progla>, western Texas phenology table): dormant Nov–Mar, leaf-out 16 Apr, full leaves 10 May, leaf drop Nov–Dec. Existing curves are consistent with these qualitative limits (Nov 0.05 mesquite slightly above "dormant", unchanged); levels are not measured; Texas not Jornada dates for mesquite |
| `temperature_min_c` | 5 | 5 | °C | ASSUMED prior | no compatible estimate for either species (FEIS gives mesquite bud break near soil 18 °C and frost-triggered leaf drop only) |
| `temperature_opt_c` | 30 | 30 | °C | BOER: ASSUMED prior; PROGLA: PROXY-CONSISTENT | **PROGLA**: Hanson (1982), *J. Range Manage.* 35(4):455–458, <https://rangelandsgateway.org/dlio/22027>, ORIGINAL *P. glandulosa* from east and west Texas, controlled environment: maximum net photosynthesis at 20 °C (east) and 30 °C (west); a photosynthesis optimum, not a growth cardinal. **BOER**: no measurement (C4 warm-season qualitative only; FEIS) |
| `temperature_max_c` | 45 | 45 | °C | ASSUMED prior | no compatible estimate; Hanson 1982 reports reduced assimilation at 35 °C but no cardinal maximum |

Temperature multiplier gates demand/growth; there is **no freezing-mortality or fire** module. Growth fraction U/D is applied **once**, total across roots; no relocation of unmet demand.

### 2d. Water use (`water_use.law = biomass_specific_pet_v1`) — unchanged proxies

| Trait | BOER | PROGLA | Units | Status | Source, original species / context, value |
|---|---|---|---|---|---|
| `demand_m3_per_kg_day` | 0.0234 | 0.0104 | m³ per kg **standing dry AGB** per day | PROXY (both) | **BOER**: Belovitch et al. (2023), *New Phytol.*, <https://doi.org/10.1111/nph.18909>; released data <https://github.com/mbelovitch/NPH-MS-2023-42507/tree/862abe68c7a9f1eb6397c36516bca35c9d445298>; ORIGINAL plants: savanna grasses (not black grama), sealed growth-chamber pots in dry-down, moist bin (soil water potential above −0.5 MPa); descriptive reanalysis by this repository (`docs/biomass-specific-water-use.md`): 23.4 L/kg/day, median of 30 individual means, harvest mass not daily mass; not a printed coefficient. **PROGLA**: Peterson & Neofotis (2004), *Oecologia*, <https://doi.org/10.1007/s00442-004-1688-y>; ORIGINAL species *Prosopis velutina* (velvet mesquite) seedlings, two per pot treated as one virtual plant, ambient 38 Pa CO₂, pot figure readings (repository `docs/study-stepwat-parameters.md`): 10.3–10.5 L/kg dry AGB/day, 10.4 chosen by the author; **not adult honey mesquite** |
| `reference_pet_mm_per_day` | 5.0 | 5.0 | mm/day | AUTHOR | normalisation only; no compatible estimate; not a measured site/experiment mean PET |

Demand `D = B · demand · (PET/reference_PET) · dt · activity · temperature_factor`, per **total** dry AGB (live_fraction is not applied again) — scaling by PET ratio, activity and temperature is LEAF's adaptation, not a source-calibrated equation. Water is owned and debited by SYRUP exactly once.

### 2e. Mortality (`mortality.law = age_background_stress_v1`)

| Trait | BOER | PROGLA | Units | Status | Source / context / gap |
|---|---|---|---|---|---|
| `life_expectancy.mean` / `unit` | 15 / years | 80 / years | years | AUTHOR (both; unchanged, **not validated**) | **BOER validation gap**: Wright (1972) *J. Range Manage.* 25:476–478, <https://journals.uair.arizona.edu/index.php/jrm/article/viewFile/6125/5735>, Table 1: mean black grama clump persistence 2.2 yr (SD 2.4, n = 6528), maximum 27 yr (Latin-name OCR of the table ambiguous; page image not inspected); Smith et al. (2004) Gramas chapter p. 993, <https://www.ars.usda.gov/arsuserfiles/30300000/Publications/AgronMono45-975-1002MRH.pdf>, distinguishes Wright & Van Dyne 1976 mean ≈ 2 yr including all seedlings, ≈ 4 yr conditional on first-year survival, maximum 28 yr. These field values include drought/seedling/other mortality (not an age-only hazard), so 15 yr is kept as an AUTHOR assumption and the 2–4 yr values are a validation target; FEIS (Simonin 2000, <https://research.fs.usda.gov/feis/species-reviews/boueri>) "long-lived" refers to genet persistence by stolons. **PROGLA**: FEIS (Steinberg 2001, <https://research.fs.usda.gov/feis/species-reviews/progla>): MAXIMUM stand ages 40–110 yr (Amarillo TX) and 172–217 yr (Rio Grande Plains; 89–93% of plants < 100 yr) — maxima/age structure do not determine a mean. Nominal mean, not a cutoff; it also bounds the initial-age draw (§1) |
| `age_hazard.weibull_shape` | 2.0 | 2.0 | – | AUTHOR | no compatible estimate |
| `age_hazard.weibull_scale_years` | 16.92568750643269 (≈ 16.93) | 90.27033336764101 (≈ 90.27) | years | DERIVED (unchanged) | mean/Γ(1.5), Γ(1.5)=0.886226925 |
| `background_hazard_per_year` | 0.0 | 0.0 | 1/year | AUTHOR | isolates the explicit age law |
| `stress.metric` | `support_ratio` | `support_ratio` | – | AUTHOR | |
| `stress.low_threshold` | 0.25 | 0.25 | ratio | AUTHOR | no compatible estimate |
| `stress.exempt_until_age_days` | 365 | 730 | days | AUTHOR | no compatible estimate |
| `stress.limit_years` | 2 | 3 | years | AUTHOR; BOER agrees with MODEL-PROXY | BOER: equals `bogr` slow2yr in the pinned `species.in` (<https://github.com/DrylandEcology/STEPWAT2/blob/003cb9b4adbf13d64a39d6d62fd95280adac27fc/testing.sagebrush.master/Stepwat_Inputs/Input/species.in>, *B. gracilis*) — agreement only, not evidence for the other support_ratio coefficients. PROGLA: no compatible estimate |
| `stress.counter_cap_years` | 10 | 10 | years | AUTHOR | ≥ limit_years |
| `stress.mortality_probability` | 0.25 | 0.1 | probability | AUTHOR | no compatible estimate |
| `stress.biomass_loss_fraction` | 0.2 | 0.1 | fraction | AUTHOR | no compatible estimate |

STEPWAT2 and ECOTONE (Peters 2002 *Ecol. Model.* 152:5–32, <https://doi.org/10.1016/S0304-3800(01)00460-4>) are structural precedents only. Full model lifespan can be shortened by stress/disturbance; the nominal mean is never an age cutoff (a plant older than 15 or 80 years is not removed by it). STEP max age 35 yr (`bogr`) is a model input and is not used as a mean lifespan. FEIS fire/freeze mortality data are unused: the model has no fire or freeze module.

### 2f. Disturbance (`mechanical_response_v1`) — every threshold an AUTHOR testable hypothesis

No compatible, quantified burial/exposure/dislodgement response for black grama or honey mesquite was verified (FEIS gives only mesquite *seed* burial depth limits, <https://research.fs.usda.gov/feis/species-reviews/progla>, which are not plant burial thresholds). Every row below is therefore an AUTHOR hypothesis with no source. Burial thresholds are fractions of height, so the revised PROGLA height law changes their absolute depths; they were not re-tuned.

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

### 2g. Reproduction (`reproduction.law = simple_seed_clonal_v1`)

| Trait | BOER | PROGLA | Units | Status | Source, original species / context, value |
|---|---|---|---|---|---|
| `seed` | null | see below | – | AUTHOR | BOER seed pathway off (Peters 2002 *Am. J. Bot.* 89:1616, <https://doi.org/10.3732/ajb.89.10.1616>, abstract only: seed production/viability comparison of *B. eriopoda* and *B. gracilis*; no establishment probability) |
| `seed.establishment_probability` | – | 0.05 | per patch, species, annual opportunity | AUTHOR | no compatible estimate. FEIS honey mesquite (<https://research.fs.usda.gov/feis/species-reviews/progla>): seed by about age 3 yr; ~131 or 97 pods/tree; ~12 seeds/pod (~5 destroyed by bruchids); one-year seedling survival 27% (clipped) vs 9.4% (unclipped) plots — none is a per-patch annual probability |
| `seed.max_recruits_per_patch` | – | 1 | count | AUTHOR | no source |
| `seed.initial_agb_dry_kg` | – | 0.002 | kg dry AGB (external credit) | AUTHOR | no compatible estimate; with the revised allometry D ≈ 0.14 m, outside both source equations' adult ranges |
| `seed.initial_seasonal_fraction` | – | 0.0 | E/B | AUTHOR | |
| `clonal` | see below | null | – | AUTHOR | PROGLA resprouting ≠ independent ramets |
| `clonal.min_age_days` | 365 | – | days | AUTHOR | no source |
| `clonal.min_agb_dry_kg` | **0.015096** | – | kg dry AGB | AUTHOR-DERIVED | = 10 × `clonal.initial_agb_dry_kg`, so that `max_donation_fraction` 0.1 affords one daughter (the native eligibility test also requires donation × B ≥ daughter mass). Not a measured minimum reproductive mass; numerically equals `bogr` maxbio 15.096 g only because that input is 10 × minbio (not claimed as a maturity size). Previously 0.01 |
| `clonal.probability` | 0.1 | – | per eligible parent per year | AUTHOR | unchanged. FEIS black grama (<https://research.fs.usda.gov/feis/species-reviews/boueri>) cites Valentine 1970 (NMSU Bull. 553, New Mexico grazing study): 2.1–2.8 stolons/plant, 0.19–0.31 rooted sets/plant (time basis not verified), Nelson 1934 (stolon needs a favorable year, rooting another) and Neilson 1986 (seedlings in 7 of 53 years); these are counts/frequencies, not a per-parent annual daughter probability. STEP `ratep`/`vegprop` have different semantics |
| `clonal.max_daughters_per_parent` | 1 | – | count | AUTHOR | |
| `clonal.initial_agb_dry_kg` | **0.0015096** | – | kg dry AGB, transferred from parent | MODEL-PROXY size | pinned STEPWAT2 `species.in` row `bogr` (*B. gracilis*, <https://github.com/DrylandEcology/STEPWAT2/blob/003cb9b4adbf13d64a39d6d62fd95280adac27fc/testing.sagebrush.master/Stepwat_Inputs/Input/species.in>): seedling minbio 1.5096 g = 0.0015096 kg. One STEP seedling-sized unit given to one separately tracked daughter; a size proxy only, **not** STEP's biomass-added clonal algorithm and not a black grama measurement. Previously 0.001 |
| `clonal.max_donation_fraction` | 0.1 | – | fraction of parent B | AUTHOR | |
| `clonal.dispersal_radius_m` | 0.25 | – | m | AUTHOR | no source |
| `clonal.max_attempts_per_daughter` | 8 | – | count | AUTHOR | numerical |

**Consequence:** at the user's mean diameter 0.10 m a BOER plant holds 0.00758 kg, below `min_agb_dry_kg` 0.015096 kg, so most initial grass plants are not eligible to produce daughters until they grow (D = 0.9·B^0.45 ≈ 0.137 m at 0.015096 kg). Native reproduction needs an explicit session `reproduction_patch_size_m` = 5 m (AUTHOR; `case.toml › reproduction`), not one trial per hydraulic cell.

## 2h. Candidate evidence NOT active (for later decision)

| Item | Value | Source | Why not active |
|---|---|---|---|
| Alternative *P. velutina* total-biomass law | B_kg = e^−0.59 · CA^1.60 (CF 1.06 omitted); 0.3766258 kg at D = 1 m, about half the active 0.7348 kg. Native form (hand-derived here, approximate): D = a·B^b with b = 1/3.2 = 0.3125, a ≈ 1.357 | McClaran et al. 2013 **Table 3** (Santa Rita, Arizona; <https://doi.org/10.1016/j.jaridenv.2012.08.015>; paper text and table use different exponents, the table was used) | a related species at a different site; shows a factor-of-about-2 location/species uncertainty, not a second active catalog |
| BOER lifespan validation | clump persistence mean 2.2 yr (SD 2.4, n = 6528), max 27 yr; Wright & Van Dyne 1976 mean ≈ 2 yr (all seedlings) / ≈ 4 yr (after first-year survival), max 28 yr | Wright 1972 <https://journals.uair.arizona.edu/index.php/jrm/article/viewFile/6125/5735> Table 1; Smith et al. 2004 <https://www.ars.usda.gov/arsuserfiles/30300000/Publications/AgronMono45-975-1002MRH.pdf> p. 993 | includes drought/seedling mortality (not an age-only hazard); taxon of the Wright 1972 table OCR ambiguous; unresolved discrepancy with the 15-yr AUTHOR assumption |
| Grass relative growth | RGR 0.072 /day (CV 0.17), BOER seedlings, nursery 18–37 °C, whole-plant dry mass including roots; ×365 = 26.28 /yr | Fernandez & Reynolds (2000) *Oecologia* 123:90–98 (citation supplied by the reviewer, not verified by me; no URL) | whole-plant (incl. roots) seedling rate, not a native aboveground STEP-style annual rate; cannot be clamped to ≤ 1 |
| Grass geometry | – | Nafus et al. 2009 | current-year mass vs basal diameter, not total standing mass vs crown |
| STEP `bogr` maxbio 15.096 g, max age 35 yr, vegindv 3, clonal y | – | pinned `species.in` (link above) | *B. gracilis* model inputs; not adopted for mature mass or lifespan |

## 2i. Remaining gaps after rev2 (hypotheses retained)

BOER and PROGLA: `mature_biomass_kg` (both), live fraction (BOER), growth rates (PROGLA; BOER uses a related-species model input), `extra_growth_rate`, activity levels and min/max cardinal temperatures, per-parent clonal probability and per-patch seed establishment probability, mortality/stress parameters, **every burial/exposure/dislodgement threshold**, adult water-demand coefficients (proxies only), grass allometry and root horizontal law, mean lifespans of both species.

## 3. Climate (`climate/`)

| Item | Value | Units | Status | Source / provenance |
|---|---|---|---|---|
| Monthly Tmin, Tmean, Tmax, precipitation | see `monthly_normals.csv` | °F, in (source) | NOAA | NOAA/NWS El Paso area climate, 1991–2020 normals, El Paso, Texas (a regional proxy, not Jornada): <https://www.weather.gov/epz/epz_area_climate> |
| Conversion | C = (F − 32)·5/9; mm = 25.4·in | – | DERIVED | exact; temperatures rounded to 9 decimals; same NOAA table |
| Annual precipitation | 223.012 | mm (8.78 in) | NOAA/DERIVED | sum of the 12 NOAA monthly totals (<https://www.weather.gov/epz/epz_area_climate>) |
| Jul–Sep precipitation | 121.158 (≈ 54.3 %) | mm | DERIVED | NOAA monthly totals, same link |
| `weather_2027.json › rows` | 365 rows, 2027-01-01 … 2027-12-31 | `tmin_c`, `tmax_c` degC | DERIVED/AUTHOR | each month's converted NOAA Tmin/Tmax repeated every day; no interpolation, no observed year, no extremes |
| Daily mean temperature | (tmin+tmax)/2 | degC | API | MASY `arithmetic_mean_of_tmin_tmax`; differs slightly from the published Tmean |
| Relative humidity, shortwave radiation | **not supplied** | – | – | none invented (`humidity.supplied = none`, `radiation.supplied = false`) |
| `pet.method` | `ecohyd_hargreaves` | – | API | FAO-56 Chapter 3 Eq. 52 precedent <https://www.fao.org/4/x0490e/x0490e07.htm> (Allen et al. 1998); runtime uses the actual SYRUP implementation; no PET computed here |
| `utc_offset` | −07:00 | – | AUTHOR | fixed MST, no DST |
| Latitude | 31.8 | degrees N | AUTHOR | rounded El Paso-area proxy; not derived from NOAA; to be bound in MAPLE `case.yaml` |
| Average regional climate | – | – | – | no claim of an exact Jornada 271.2 mm record |

## 4. Scheduler design (`scheduler/`) — all AUTHOR synthetic choices except monthly totals

| Item | Value | Units | Status |
|---|---|---|---|
| Wind/rain pairs, transport rows | 36 pairs, 72 rows | count | AUTHOR |
| Monthly rain counts | [2,2,3,3,3,3,4,4,4,3,3,2] | events | AUTHOR |
| Dates (day of month) | 2/month: 4,18; 3/month: 4,14,24; 4/month: 3,10,17,24 | day | AUTHOR |
| Rain start | 09:00 MST | local time | AUTHOR |
| Rain duration | 21600 (Jan,Feb,Mar,Nov,Dec); 14400 (Apr,May,Oct); 10800 (Jun–Sep) | s | AUTHOR |
| Monthly rain totals | 9.906, 10.160, 6.096, 4.318, 10.922, 18.542, 40.132, 42.418, 38.608, 14.986, 10.922, 16.002 | mm | NOAA ×25.4, <https://www.weather.gov/epz/epz_area_climate> (El Paso 1991–2020 monthly precipitation in inches) |
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

| Item | Value | Units | Status | Source, context, value |
|---|---|---|---|---|
| `depth_edges_m` | [0, 0.2, 0.5, 1.2] | m | AUTHOR | proposed layers; same 3 as `root_layers`; no literature value |
| `theta_residual` | 0.065 | m³/m³ | LIT / TEXTURE-PRIOR | Carsel & Parrish (1988), *Water Resour. Res.* 24(5):755–769, <https://doi.org/10.1029/WR024i005p00755>; US soil-texture class **sandy loam** (not site-specific); an independent simple prior, **not** part of one fitted van Genuchten retention curve; not a plant wilting point |
| `theta_saturation` | 0.41 | m³/m³ | LIT / TEXTURE-PRIOR | same paper, sandy loam; not site-specific |
| `theta_field_capacity` | **0.23** | m³/m³ | TEXTURE-PRIOR (changed from AUTHOR 0.20) | Allen et al. (1998) FAO-56 Chapter 7, **Table 19**, sandy loam: field capacity 0.18–0.28 and wilting point 0.06–0.16 m³/m³; **Example 35** uses θ_FC = 0.23 (WP 0.10): <https://www.fao.org/4/x0490e/x0490e0c.htm>. Illustrative texture-scale prior (worked example / near mid-range), not a Jornada measurement and not a Carsel–Parrish value |
| `theta_evaporation_stress` | 0.15 | m³/m³ | AUTHOR | no compatible estimate; FAO-56 evaporation thresholds / readily evaporable water belong to a different evaporation law and were not adopted as the native `(θ/stress)²` ramp threshold |
| `diffusivity_m2_s` | 2.0e−7 | m²/s | AUTHOR | illustrative; **not Ksat** (Carsel & Parrish Ksat is not used); no compatible estimate |
| `evaporation_factor` | 1.0 | – | AUTHOR | existing-API illustrative |
| `gravity_drainage` | true | – | AUTHOR | existing-API illustrative |
| `theta_initial` | [0.10, 0.12, 0.14] | m³/m³ per layer, shallow→deep | AUTHOR | antecedent moisture between residual and field capacity; no site data |
| Active cells, cell area | 65536, 0.0625 | cells, m² | DERIVED | |
| Potential soil evaporation | 0.7 × shared PET (mm/day → m/s) | – | AUTHOR | **provisional partition**, not E+T ≤ PET, not an energy partition; SYRUP/caller must qualify before full coupling |
| `added_water_m` between storms | 0 | m | AUTHOR | no extra rain/irrigation/run-on |

## 6. Runtime / output / budget (`case.toml`) — no literature parameters

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
| Checkpoint dir / format | `outputs/checkpoints/`, restart format 4 (plain `plants.csv` + `plants.metadata.json`, arrays and manifest) | – | AUTHOR / API |
| Event snapshot path template | `outputs/event_plants_{index:04d}.csv` (separate event writer) | – | AUTHOR planning choice / caller policy |
| Checkpoint `path_template` / policy / overwrite | `outputs/checkpoints/boundary_{index:04d}` / `after_each_accepted_transport_and_final` / `false` | – | AUTHOR planning choice / caller policy; not a new loader; nothing created |
| CSV `rain_intensity_mm_h` | 9-decimal display value; `rain_depth_mm` is authoritative (adapter rebuilds depth/(duration/3600); display error ≤ ~2e−9 mm) | mm/h | DERIVED |
| CSV `forcing_plus_rain_guard_end_offset_s` | nominal forcing end + rain guard (0 for wind); static arithmetic, not a terminal time | s | DERIVED |

## 7. Reference grid / terrain recipe (`landscape.toml`, not generated) — computational reference, no literature parameters

| Item | Value | Units | Status |
|---|---|---|---|
| Surface seed | 1173 | – | REF |
| Reference level | 0.6 | m | REF |
| Relief (peak-to-peak, exact) | 0.05 | m | REF |
| Correlation lengths x, y | 1.0, 1.0 | m | REF |
| Sediment voxels | 12 × 0.1 | count × m | REF |
| Generator | upstream `generate_spectral_gaussian_field` (not duplicated) | – | REF |

These are computational reference settings, **not observed El Paso terrain**. No `surface.npy` exists.
