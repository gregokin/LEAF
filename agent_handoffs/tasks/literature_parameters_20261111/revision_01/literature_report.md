# literature_report.md — literature_parameters_20261111 revision 1

## Summary
Literature pass over black grama (BOER) and honey mesquite (PROGLA). **No numeric trait changed**: no accessible source quantity maps defensibly onto a native trait definition. Provenance, links, status tags, references and `catalog_id` were updated. Nothing was committed or executed.

## Files changed (all within the allowed list)
* `cases/jornada_grass05_shrub10_256/preview/species_at_initialization.json` — NEW, intended exact copy of baseline `species.json` (written before any edit; byte identity unverified by me).
* `cases/jornada_grass05_shrub10_256/species.json` — `catalog_id` -> `jornada_grass05_shrub10_litprov_rev1_20261111`; catalog notes; BOER/PROGLA references lists; status/units/notes text for activity, cardinal temperatures, reproduction, lifespan, growth. All numeric trait values identical to baseline (re-derivation of k and Weibull scale not needed).
* `cases/jornada_grass05_shrub10_256/PARAMETERS.md` — legend tags, catalog/preview paragraph, growth/water-use/mortality/reproduction rows with links, gap inventory (§2h), disturbance note.
* `cases/jornada_grass05_shrub10_256/SOURCES.md`, `README.md`, `preview/README.md` — new source codes, revision and frozen-catalog statements.
* `research_notes.md`, `literature_report.md` — this report.
* Not modified: `soil.toml` (no soil work; Codex lane), all other preview historical files, `generation.json`, climate, scheduler.

## Checks actually run
None (no execution allowed). Checks were reading and manual arithmetic only. JSON validity of the edited `species.json` was NOT machine-checked; Codex should parse it and verify that `preview/species_at_initialization.json` has the same SHA-256 as the baseline `species.json`.

## Web access results
* Worked: USDA FEIS BOER and PROGLA pages; Crossref API record for Peters 2002 (abstract); rangelandsgateway abstract of Hanson 1982; Plymouth portal abstract of Golubov 1999; GitHub STEPWAT2 tree and `rgroup.in`; PMC PeerJ summary.
* Failed (literal errors): `getaddrinfo ENOTFOUND archive.jornada.nmsu.edu`; HTTP 403 Forbidden for Wiley (Peters 2002), PeerJ, repository.arizona.edu, data.zhenzhi.com; redirect to `https://www.uidaho.edu/404.htm` for the Ludwig 1975 PDF; "Claude Code is unable to fetch from web.archive.org"; journals.uair.arizona.edu redirects; PDFs returned as binary (Northup 2005, McClaran 2012, Hughes 2006, ARS desert shrub report, mountainscholar) with no extractable text. Details in `research_notes.md`.
* Fetched pages were summarised by a small model; quoted numbers should be re-verified from the original before adoption.

## What the literature did support (labels, not value changes)
* PROGLA `monthly_activity` consistent with western-Texas FEIS phenology (ASSUMED-SHAPE); BOER activity consistent with FEIS Chihuahuan Desert growth/flowering dates.
* PROGLA `temperature_opt_c` 30 = west-Texas honey mesquite photosynthesis optimum (Hanson 1982) — PROXY-CONSISTENT.
* `mature_biomass_kg` 10 for PROGLA: only an unverified secondary ">10 kg aboveground" mean (Ludwig 1975).
* Lifespans: FEIS gives maxima/qualitative genet persistence; means unchanged.

## Unsupported gaps (retained AUTHOR hypotheses)
BOER mature biomass; both species' live fraction, growth/extra-growth rates, activity levels, min/max cardinal temperatures; clonal probability and seed establishment probability and initial masses; all age-hazard and stress parameters; every burial/exposure/dislodgement threshold; adult water-demand coefficients (proxies); PROGLA/BOER ECOTONE (Peters 2002 Ecol. Model.) parameter tables not retrieved; Ludwig 1975 and Northup 2005 tables not read. STEPWAT2 pinned inputs contain only sagebrush-steppe groups (no proxy possible).

## Notes for Codex integration
* Allometry (D, H vs dry AGB) and soil were deliberately not touched; if revision 2 changes allometry/mature mass, `species.json` must be re-edited and the catalog_id consistently bumped; the preview freeze stays valid.
* If you adopt Ludwig/Northup-type live-fraction evidence, recompute `biomass_support_kg_year_per_m3 = live_fraction/(365*q)` (q 0.0234 / 0.0104 m3/kg/day).
* A new initialization is required before any run with the revised catalog; it was not run.
