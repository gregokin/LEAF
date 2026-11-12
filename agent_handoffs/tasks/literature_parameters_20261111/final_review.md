# Codex final review — ACCEPT

Task literature_parameters_20261111, baseline b854fee6af2c521b075973940655927a717eb389 (clean source). Claude authored bounded candidate changes; Codex independently inspected primary publications/model inputs, corrected, installed and verified. Session 06a62227-72c8-4350-87b8-a6676670d464 revisions1–3 all returned structured success/is_error=false, exit0 and reaped children. [Scope audit](scope_audit.json) found no forbidden file/tool access.

## Accepted changes

10 numeric entries: PROGLA diameter/height coefficients (4), live fraction and consistent support (2); BOER STEP-related intrinsic rate, clonal daughter dry mass and affordable parent minimum (3); sandy-loam field capacity (1). See the case PARAMETERS.md changed-value table for units, original species/site and direct sources. Other values have literature/proxy/context or explicit unresolved assumptions. No numerical clamping, raised bounds or new physics.

Northup2005 honey-mesquite TOTAL standing oven-dry AGB/crown-area median fit is algebraically inverted, without retransformation CF. Height is a mixed-study Prosopis velutina proxy from McClaran2013. A constant live share includes live wood; water demand remains per total dry AGB. Support k stays kg*year/m3, not production WUE. BOER rate/daughter size are related blue-grama MODEL inputs at pinned STEP revision003cb9b4, not field black-grama estimates. Field capacity .23 is a FAO texture/example prior, not measured Jornada soil.

## Independent verification

Final candidate and installed trees both passed all 70 checks: native catalog/generation schema; all four ecological trait consumers; source-equation geometry at 4 diameters; support, mean/Weibull and donor arithmetic; strict case JSON/TOML parsing; saved CSV+companion read with the exact frozen old catalog; rejection with the active revised catalog; baseline hashes of all historical preview data/image/terrain/provenance and unchanged code/config files. [Installed output](installed_validation.stdout.json), [exit/command](installed_validation.exit.json), [source manifest](final_source_manifest.json), [environment](verification_environment.json). git diff --check passed. No new environment/install, forward model run, population generation, rendering, transport or GPU benchmark occurred. No upstream files changed.

## Scientific limits

No field calibration/validation is claimed. Grass allometry/live fraction, mature masses, shrub rate, recruitment frequencies, age/stress and mechanical-damage laws remain assumptions. The2–4yr field grass-clump persistence versus nominal 15 yr age-component mean is explicitly unresolved; all-cause field persistence is not an isolated aging parameter. Seedling/adult and site-transfer uncertainties are retained. Current scenario files remain planning/API manifests where they were previously, with no full annual coupled driver introduced.

Preview frozen catalog is an exact baseline species.json copy, preserving the recorded catalog hash. Historical preview rows remain fixed-age with old allometry; a new initialization for active traits is a separately requested action.

Verdict: **accept**. Standing user instruction authorizes commit/push after completion. No active Claude writer or model remains; publication is the remaining operator action.
