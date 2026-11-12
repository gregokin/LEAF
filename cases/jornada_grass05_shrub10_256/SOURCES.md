# Sources and shortcodes

Short paraphrases only; no article text is copied. "Origin" separates what a source *supplies* from what the **author chose or derived**.
Per-parameter use is in [PARAMETERS.md](PARAMETERS.md). A numeric value chosen by the author with a stated rationale is legitimate provenance for a scenario value,
but it is **not** a literature measurement, and a structural paper never establishes an exact author coefficient.

## Primary / external sources

| Code | Reference and link | What it supplies here | What it does NOT supply |
|---|---|---|---|
| **NOAA_EPZ** | NOAA/NWS El Paso area climate, 1991–2020 normals — <https://www.weather.gov/epz/epz_area_climate> | Monthly minimum, mean, maximum temperature (°F) and precipitation (in) in `climate/monthly_normals.csv`; the 12 monthly rain totals | An event schedule, daily variability, extremes, humidity, radiation, or any Jornada station record; no claim of an exact Jornada 271.2 mm |
| **VitoSala2026** | Vito & Sala (2026), *Ecology* 107:e70384 — <https://doi.org/10.1002/ecy.70384> | Tracer-uptake fit `F(z)=1−β^z` (z in cm), β = 0.977 black grama and 0.983 honey mesquite (combined two years) | Root biomass density; layer fractions (our finite-depth integration over 0/0.2/0.5/1.2 m with conditional normalisation is the **author's derivation**); the paper's binned depth statements are not reproduced by it |
| **Simonin2000_FEIS_BOER** | Simonin (2000), USDA FEIS black grama species review — <https://research.fs.usda.gov/feis/species-reviews/boueri> | Qualitative: long-lived warm-season C4 bunchgrass, roots concentrated in the upper ~25 cm and reaching ~1.2 m, summer growth, 20–60 cm stems, vegetative spread | Any exact LEAF coefficient (allometry, activity, temperatures, rates) |
| **Steinberg2001_FEIS_PROGLA** | Steinberg (2001), USDA FEIS honey mesquite species review — <https://research.fs.usda.gov/feis/species-reviews/progla> | Qualitative: southwestern deciduous seasonality, seed establishment, deep roots | Any exact LEAF coefficient |
| **Palmquist2018_STEPWAT2** | Palmquist et al. (2018) STEPWAT2, *Ecosphere* — <https://doi.org/10.1002/ecs2.2394> | Structural precedent for growth/resource/demography laws | Any adopted number. LEAF is a hybrid aboveground-only adaptation, not a complete STEPWAT reproduction |
| **Peters2002_ECOTONE** | Peters (2002), Plant species dominance at a grassland-shrubland ecotone: an individual-based gap dynamics model of herbaceous and woody species, *Ecological Modelling* 152:5–32 — <https://doi.org/10.1016/S0304-3800(01)00460-4> (cited from the user's project outline) | Structural precedent for age/stress demography and seasonal structure (ECOTONE) | Any adopted coefficient; our exact mortality/recruitment numbers are AUTHOR choices |
| **Belovitch2023** | Belovitch et al. (2023), *New Phytologist* — <https://doi.org/10.1111/nph.18909>; released data <https://github.com/mbelovitch/NPH-MS-2023-42507/tree/862abe68c7a9f1eb6397c36516bca35c9d445298> | Data for the repository's descriptive reanalysis: moist-bin savanna-grass median 23.4 L/kg dry AGB/day (30 individual means) — `docs/biomass-specific-water-use.md` in the LEAF repo | A printed universal coefficient; daily biomass (harvest mass is used); adult black grama calibration |
| **PetersonNeofotis2004** | Peterson & Neofotis (2004), *Oecologia* — <https://doi.org/10.1007/s00442-004-1688-y> | Conditional *velvet* mesquite seedling/pot value 10.3–10.5 L/kg dry AGB/day (LEAF repo `docs/study-stepwat-parameters.md`; two seedlings in one pot = one virtual plant; AGB allocation assumptions); **10.4 is the author's central selection** | Adult honey mesquite water use or calibration |
| **CarselParrish1988** | Carsel & Parrish (1988), *Water Resources Research* 24(5):755 — <https://doi.org/10.1029/WR024i005p00755>; reproduction <https://doi.org/10.1002/2014GL059827> | Sandy-loam texture priors θr = 0.065 and θs = 0.41 | Field capacity or diffusivity: the paper's table does contain Ksat, but it is **not used** as this case's diffusivity; field capacity and diffusivity are author/API illustrative assumptions. Not site-specific soil |
| **Allen1998_FAO56** | Allen et al. (1998), FAO-56 chapter 3, Eq. 52 (Hargreaves precedent) — <https://www.fao.org/4/x0490e/x0490e07.htm> | Precedent for `ecohyd_hargreaves` | The exact native implementation (SYRUP `weather_pet`), which is **not** a re-parameterisation of the FAO text; no PET computed here |

## Internal / non-literature origins

| Code | Meaning |
|---|---|
| **LEAF_SRC** | Existing LEAF source/examples (schemas, generator behaviour, example fixture allometry/porosity) — uncalibrated fixtures; `examples/phase0/generation.json`, `examples/phase3/species.json` |
| **MAPLE_REF** | MAPLE reference computational case `leaf_shrubs11_bimodal135_256_12h` (`case.yaml`, `source/generation.json`) and `random_field.py::generate_spectral_gaussian_field`; grid, seed 1173, 0.6 m level, 0.05 m relief, 1.0 m correlation, 12 × 0.1 m voxels. Computational settings, **not** El Paso terrain; its sediment calibration is test-only and not imported |
| **SYRUP_API** | SYRUP `subsurface.py` (branch feature/subsurface-hydrology) public API names and defaults used in `soil.toml` |
| **MASY_SRC** | MASY native schemas (`masy.weather.v1`, `masy.wind_event.v1`, `masy.rain_event.v1`, `masy.interevent.v1`, `masy.scenario.v1`) |
| **USER** | User-specified request: vegetation covers, diameter means/SDs, 3–5× root radius range, year of alternating wind/water events, average regional climate |
| **CASE_AUTHOR_20261111** | Author of this case (2026-11-11): all choices tagged AUTHOR in PARAMETERS.md — seeds, caps, the species life expectancies (15 / 80 years, which also bound the user-requested uniform initial-age sampling 0 … life expectancy), growth/mortality/disturbance/reproduction numbers, activity arrays, cardinal temperatures, reference PET, soil illustrative values, evaporation partition 0.7, scheduler dates/counts/durations/wind forcing, 30-day snapshot cadence, resource budgets. Rationale is stated beside each value; none is a measurement |

## Distinction summary

* **Literature values used (with proxy qualifiers):** NOAA normals; Vito & Sala β; Belovitch grass demand (descriptive reanalysis); Peterson & Neofotis conditional seedling demand; Carsel & Parrish θr/θs.
* **Derivations by the author:** layer fractions, support k, Weibull scales, lognormal parameters, initial biomass/height, expected counts, unit conversions, monthly rain disaggregation.
* **User targets:** covers, diameters, root-radius range.
* **Author hypotheses (uncalibrated):** everything else, as listed in PARAMETERS.md. No field validation is claimed.
