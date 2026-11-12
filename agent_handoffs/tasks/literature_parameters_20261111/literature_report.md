# literature_report.md — literature_parameters_20261111 revision 3 (documentation corrections on top of rev2)

## Revision 3 — what changed (documentation only; all numeric values and the `catalog_id` are exactly as in rev2)
1. Vito & Sala 2026 described as a **lithium-tracer** uptake study (not isotope tracing) in PARAMETERS.md and SOURCES.md; β and layer fractions untouched.
2. Direct links added in-row (no "same/above" dependence) to: the summary numeric-change table, PROGLA diameter.b / height.b and the implied B/H rows, root layers 1/2 and omitted tail, climate conversion / Jul–Sep / daily disaggregation / per-event rain depth (NOAA), soil `theta_saturation` (Carsel & Parrish), BOER clonal rows (pinned STEPWAT2 `species.in`).
3. Both `life_expectancy.notes` (species.json) and the PARAMETERS row no longer say "metadata only": "nominal age-component mean, also used by the initial-age draw; not a deterministic death age". FEIS "long-lived" is now stated as also describing multi-decade stolon persistence that does not establish a mean ramet/clump lifespan. 15/80 and Weibull scales unchanged.
4. `seed.initial_agb_dry_kg` note: D ≈ 0.14 m is below the McClaran 2013 height-fit canopy-area range (min CA 0.1 m²); Northup small-plant validity unqualified (range not reported). Same softening in the PROGLA species note and reproduction note.
5. `temperature_max_c`: unverified "35 °C" claim removed; 45 °C is an explicit assumption. Hanson 1982 DOI https://doi.org/10.2307/3898604 added to the temperature_opt rows (30 °C west Texas verified by Codex).
6. Candidate table: Fernández & Reynolds 2000 DOI + primary Table 1 link (reviewer-checked, not fetched by me), Nafus et al. 2009 DOI https://doi.org/10.2111/08-003, and the pinned `species.in` full link; SOURCES.md gained candidate-only rows for both.
7. Case README heading is now "Literature parameter update"; PROGLA `live_fraction` is LIT-DERIVED (complement 1 − 0.094, 0.906 not printed) in species.json status/units/catalog notes, PARAMETERS legend/rows/summary table, SOURCES and README.

**Checks.** Codex already ran (per the rev3 request) 70 read-only checks on rev2, all PASS: schema/parse, source-equation sizes, support/lifespan arithmetic, donor affordability, JSON/TOML, frozen-catalog and preview hashes, companion read/refusal. **My (unexecuted) checks:** none run; I only re-read the edited strings. Because `species.json` text changed again (notes only), its catalog hash differs from the rev2 file; Codex should re-parse it as JSON, re-confirm the numeric fields equal rev2, and re-confirm `preview/species_at_initialization.json` and the historical preview files are untouched (I did not modify them).

---

# Revision 2 report (retained)

## Summary
Implemented items A–E of `codex_evidence_02.md`. Ten numeric fields changed (table below); everything else numeric is unchanged. Provenance documentation was rewritten so every scientific row in PARAMETERS.md carries a direct source link, original species/site/context, units and a measured/derived/proxy/model-input/assumed status, or an explicit "no compatible estimate". Nothing was committed or executed; nothing is a Jornada field calibration.

## Actual numeric changes versus baseline (`jornada_grass05_shrub10_trial_20261111`)
| File › field | Baseline | Rev2 | Basis |
|---|---:|---:|---|
| species.json › PROGLA › allometry.diameter.a | 1.6 | 1.1087951253470727 | Northup 2005 Table 7 (algebraic, log-median, CF not applied) |
| species.json › PROGLA › allometry.diameter.b | 0.4 | 0.3351206434316354 | 1/2.984 |
| species.json › PROGLA › allometry.height.a | 0.8 | 1.1015229296155495 | McClaran 2013 Table 2 (*P. velutina* proxy) ∘ Northup |
| species.json › PROGLA › allometry.height.b | 0.4 | 0.2546916890080429 | 0.38/1.492 |
| species.json › PROGLA › growth.live_fraction | 0.4 | 0.906 | 1 − 0.094, Northup 2005 Table 1 |
| species.json › PROGLA › growth.biomass_support_kg_year_per_m3 | 0.10537407797681772 | 0.23867228661749212 | 0.906/(365 × 0.0104) |
| species.json › BOER › growth.intrinsic_rate_per_year | 0.8 | 0.474 | pinned STEPWAT2 `species.in` row `bogr` (*B. gracilis*) |
| species.json › BOER › reproduction.clonal.initial_agb_dry_kg | 0.001 | 0.0015096 | `bogr` minbio 1.5096 g |
| species.json › BOER › reproduction.clonal.min_agb_dry_kg | 0.01 | 0.015096 | AUTHOR-DERIVED 10 × daughter mass |
| soil.toml › theta_field_capacity | 0.20 | 0.23 | FAO-56 Table 19 / Example 35 |
| species.json › catalog_id | jornada_grass05_shrub10_trial_20261111 | jornada_grass05_shrub10_litparam_rev2_20261111 | label |

Unchanged on purpose: mature masses (gap), life expectancies 15 / 80 yr and Weibull scales (gap/validation target), q demand proxies, grass live fraction, extra growth, activity arrays, cardinal temperatures, roots, mortality/disturbance, other reproduction fields, user grid/cover/diameters/age policy/timing, abrasion disabled, θr, θs, evaporation stress, diffusivity.

## Files changed (all within the allowed list)
* `cases/jornada_grass05_shrub10_256/species.json` — full rewrite keeping both complete records; numeric changes above; notes/status/units/references updated; only native keys used.
* `cases/jornada_grass05_shrub10_256/soil.toml` — `theta_field_capacity` 0.23 and comments/links.
* `cases/jornada_grass05_shrub10_256/PARAMETERS.md` — rewritten (changed-numbers table; per-row source links for allometry, roots, growth, water use, mortality, reproduction, climate, scheduler NOAA totals, soil; candidate-not-active §2h; gaps §2i).
* `cases/jornada_grass05_shrub10_256/SOURCES.md`, `README.md`, `preview/README.md` — consistent revision statements; reader example and reinterpretation warning in the preview README.
* `research_notes.md`, `literature_report.md`.
* NOT touched: `preview/species_at_initialization.json` (frozen, hash-verified by Codex), all historical preview data, `generation.json`, climate, scheduler, `case.toml`, `landscape.toml`.

## Checks actually run
None (no execution available). Only reading and hand arithmetic: derivations in `research_notes.md` were reproduced to ≈ 4 digits by hand; stored 16-digit coefficients come from `codex_evidence_02.md`. Codex should: parse `species.json` (native catalog validation, unique keys), confirm numeric diffs against the baseline, recompute a_d, b_d, a_h, b_h, k, check the `soil.toml` TOML syntax, and re-run any consistency tests.

## Limitations honestly stated in the files
* Northup 2005 diameter law: Texas site (≈ 680 mm/yr), target species, retransformation factor not applied, algebraic inverse, no prediction range. McClaran 2013 height law: *P. velutina*, Arizona, mixed-study proxy; H = D/2 no longer holds.
* `live_fraction` 0.906 includes live wood; not a transpiring fraction; q is per total B and not multiplied again.
* BOER rate/daughter size are model inputs of *B. gracilis*; the 0.015096 kg threshold is author-derived; as a consequence most initial BOER plants (0.0076 kg) are not eligible for daughters until they grow (D ≈ 0.137 m needed).
* BOER mean lifespan 15 yr conflicts with field clump persistence 2.2 yr (SD 2.4, n = 6528; Wright 1972 Table 1, taxon OCR ambiguous) and ≈ 2/≈ 4/28 yr (Smith et al. 2004 p. 993); retained as AUTHOR assumption and flagged as a validation gap. PROGLA 80 yr: FEIS gives only maxima.
* Ludwig 1975 ">10 kg" snippet no longer used. Prior rev1 references to unverified sources (Ansley 2010, Golubov 1999, Ludwig 1975) were removed from the catalog and PARAMETERS tables.
* Fernandez & Reynolds 2000 RGR (26.28/yr, incl. roots) and McClaran Table 3 total-biomass law are candidate-only; not active.

## Web access
No web calls were made in rev2 (evidence supplied by Codex). Rev1 failures: `getaddrinfo ENOTFOUND archive.jornada.nmsu.edu`; HTTP 403 (Wiley, PeerJ, repository.arizona.edu, data.zhenzhi.com); redirect to `https://www.uidaho.edu/404.htm`; "Claude Code is unable to fetch from web.archive.org"; binary PDFs without extractable text.

## Remaining unsupported gaps
Mature biomass (both); grass live fraction, geometry and extra growth; PROGLA rates; activity levels and min/max cardinal temperatures; clonal probability and seed establishment probability and seedling masses; all stress/age-hazard parameters; every burial/exposure/dislodgement threshold; adult water-demand coefficients; evaporation stress and diffusivity; mean lifespans; Jornada-specific validation of all of the above.

## Notes for Codex
* A new initialization is required before any run with this catalog: old preview masses read through the new PROGLA law would give D ≈ 0.75 m and H ≈ 0.82 m for the 1.0 m plant (hand arithmetic). `read_plant_csv(..., active_catalog)` is expected to raise a hash-mismatch `StateError`; the reader example uses the frozen catalog.
* Possible floating tie at exactly B = `min_agb_dry_kg` in the clonal eligibility test (0.1 × 0.015096 vs 0.0015096); no effect for B above the threshold.
