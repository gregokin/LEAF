# LEAF Project Brief

LEAF stands for **Landscape Ecohydrology with Agent-based Flora**.

## Phase 0 implementation — 2026-11-10

The user authorized implementation, automatic Codex–Claude orchestration and
sharing, followed by a Codex commit and push. Claude wrote an isolated candidate;
Codex independently reviewed and installed revision 3. See the
[Phase 0 guide](../docs/phase0-vegetation.md) and
[MAPLE developer handoff](../docs/maple-leaf-integration.md).

Canonical plant CSV records stable IDs, species, subcell X/Y, establishment
**ground** elevation, aboveground **dry** biomass (kg), seasonal biomass, age
(days), and live/dead-standing/removed status. Species JSON separately holds
allometry, roots, constant porosity, growth and demography/disturbance sockets.
Initial `SURFACE` tokens resolve once; snapshots carry concrete anchors and
catalog/domain/time metadata. Recording cadence is user-selected.

Seeded generation uses per-species union cover, mean/SD positive-lognormal
diameters and uniform candidate locations. Optional global canopy exclusion
places large canopies first. Cover is sampled on a finite non-periodic domain;
tolerance, resolution, placement and memory bounds are explicit. Crowded or
infeasible requests fail rather than silently changing traits or resource limits.

The NumPy-only core supports MAPLE prescribed vegetation without moisture or
growth. Shared LEAF functions publish native-compatible canopy arrays and ID
sidecars, with explicit burial/exposure handling. Constant example porosity is
0.20, unused by LEAF ecology but used by MAPLE wind. Event diagnostics and a
transactional species-law callback are supplied; injury laws and MAPLE adoption
are deferred. This supersedes the interface/porosity status in the historical
2026-11-08 section below.

Legacy growth/water equations and upstream sources are preserved. An optional
ID-based bridge captures driver biomass only when output is requested and rejects
incompatible allometry or lost seasonal mass. CSV replay is vegetation state,
not a complete hydrology/calendar/RNG restart. Scientific trait calibration,
recruitment, mortality and detailed surface remapping remain later work.

## Canopy allometry and MAPLE coupling — 2026-11-08

User clarification: species-specific allometric relationships will connect each
plant's **aboveground dry biomass** to canopy height and width. Together with the
plant's stable ID and X/Y location, this geometry will supply the canopies used
for MAPLE shear-stress partitioning. LEAF owns the authoritative plant state;
MAPLE vegetation objects/rasters are derived physical views of that state.

Porosity is initially a constant placeholder, with its numerical value still to
be selected; biomass-dependent porosity is deferred. Mark it explicitly as
**unused in the current LEAF code**. Current LEAF growth uses a simplified
biomass-to-circular-canopy-radius relationship; height allometry, porosity and
publication to MAPLE are not yet implemented. Any later adapter that passes
porosity into MAPLE must document its actual use.

The coupling responsibilities are:

- **LEAF ↔ SYRUP:** daily access to native subsurface moisture and actual
  transpiration withdrawal, retaining SYRUP's single physical water budget.
- **LEAF → MAPLE:** publish allometry-derived canopy geometry before wind
  transport for shear-stress partitioning; refresh it after growth or injury.
- **MAPLE/shared surface → LEAF:** use wind exposure/abrasion diagnostics and
  shared-surface erosion/deposition to evaluate species-specific damage, burial,
  root exposure or dislodgement. Surface changes caused by SYRUP water events
  must feed the same plant-injury handling. LEAF applies the ecological response
  and updates plant state and exposed geometry.

Keep growth/transpiration disabled during transport events. Establishment ground
elevation remains the intended burial/exposure reference. Native SYRUP moisture
coupling exists; MAPLE canopy publication and event injury remain future work.
Detailed soil-water/layer remapping stays deferred. This clarification records
design requirements without changing runtime code or authorizing model runs.

## Paired growth and water-use observations — 2026-11-01

The user clarified that validation needs individual growth and simultaneous
ET/transpiration, suggesting Ravi, Breshears and Biosphere 2. The
[focused survey](../docs/paired-growth-water-data.md) verifies public Setaria
weight/harvest tables and 176 exact plant-ID joins. Its repeated mass is
image-derived; destructive dry mass is an endpoint. Biosphere 2 mesquite supplies
two-day evaporation-corrected water and initial/final component mass, but each
pot holds two seedlings. Ravi's grass study supplies physiology, moisture,
control-treatment mass and survival; whole-plant ET and archive contents remain
unverified. Additional public sapling/forb archives are identified with access
limits. No ingestion into the model or calibration was performed.

The user also requested the discussed Jornada uptake values as potential model
values. The [inactive candidate record](../parameters/candidates/jornada-uptake.json)
preserves source, coefficients, confidence intervals, depth units, dry/wet
fractions and the optional grouped size fit. These are uptake priors, with
physical layer integration and tail handling still to be selected. Runtime
profiles, growth traits and defaults remain unchanged. This task is research
and documentation; automatic implementation collaboration remains enabled.

## Validation-data survey — proposed next scientific checks

The user requested validation data. The [inventory](../docs/validation-data.md)
prioritizes Jornada seasonal species biomass, independent reference harvests,
profile and automated shallow moisture, the 2018–2019 grass/mesquite uptake-depth
experiment, and NEON JORN harvest/structure/water products. Metadata versions and
access limits are archived in [the survey task](tasks/validation_data_survey_20261031/README.md).
Observation tables have not been ingested. The proposal separates calibration
and withheld validation, identifies provisional biomass and whole-shrub mass
ambiguity, and treats total ET separately from transpiration. No calibration,
field-validation run or model change was assigned or performed.

## Current implementation — STEP-style aboveground growth

The user authorized a STEP-consistent version after selecting total aboveground
biomass. The opt-in [adapter](../docs/stepwat-growth.md) uses actual native SYRUP
transpiration as biomass-support resources, species-specific mature mass and
logistic growth rates, and weekly/year-end biomass and root-map updates. Total
aboveground biomass remains the observational target; a seasonal aboveground
subcomponent allows STEP-style year-end senescence without leaf/wood/root pools.

Species traits and monthly activity are explicit, and each day requires a host
temperature modifier. Root-layer profiles remain fixed and all water stays in
SYRUP. Existing spatial proportional root competition is retained; STEP's patch
largest-first allocation and added-resource adjustment are not copied. This is
a documented weekly adaptation requiring calibration, not a complete STEP port.
See [current source lock](../dependencies/stepwat-growth.lock.json) and
[task evidence](tasks/stepwat_growth_20261031/README.md). Detailed surface remapping
and demography remain deferred; no new long scenario runs were assigned.

## Current scope clarification — total aboveground biomass

The user selected total aboveground biomass per individual because more data are
available for that quantity. Use it as the modeled growth state and calibration
target; separate leaf, wood, root and reserve mass pools are not required. Daily
actual transpiration supports aboveground production accumulated for weekly
biomass updates. The subsequent STEP-style implementation above selects
biomass-support pressure and seasonal loss; it does not use a direct WUE yield.
Any later direct aboveground production efficiency must include the appropriate
implicit allocation costs; a whole-plant efficiency cannot be used unchanged.

Species/PFT allometry can derive canopy dimensions and root extent from aboveground
biomass, retaining existing species-specific layer fractions. Phenology can modify
leaf activity without separately tracking tissue masses. Match the biomass units
and live/standing-dead convention to observations during parameter selection.
This supersedes the survey's initial multi-pool recommendation. The legacy generic
fixture remains available; the new STEP-style option is described above. See the
[survey and alternatives](../docs/water-to-biomass-survey.md).

## Water-to-biomass survey — background and alternatives

The user requested comparison of model approaches for converting water use into
grass and shrub biomass. The [survey](../docs/water-to-biomass-survey.md) inspects
active local ECOTONE growth, pinned STEPWAT2 growth/resource code and primary
literature for Tietjen, GRASSMIND, Caylor, Williams–Albertson and AquaCrop.
Its initial recommendation combined actual-transpiration-constrained production,
climate/phenology and tissue-specific allocation/turnover; the later user decision
above narrows this to total aboveground biomass. ECOTONE is the closest
local individual precedent; the daily grass/wood carbon model and GRASSMIND supply
complementary accounting ideas. These are proposed adapters, not exact model
reproduction or a selected/calibrated law. No numerical code or model runs changed.
Those direct production/efficiency options remain alternatives; the later user
instruction selected the STEP-style resource-supported growth adaptation above.
Detailed surface remapping stays deferred.

## Current scope clarification — simplified soil profile

The user explicitly deferred detailed surface/soil/root remapping after erosion
or deposition. Keep the current simplified layered representation; soil texture
still informs soil-water properties. Detailed material tracking, layer-boundary
changes, redistribution of existing water after surface change and corresponding
root-access changes are future implementation, not current prerequisites.
This narrows the immediate scope while retaining the shared MASY surface as the
long-term authority. Daily native water accounting and weekly growth/root-map
updates continue as implemented. See the
[coupling guide](../docs/coupled-transpiration.md#simplified-soil-profile-and-deferred-surface-remapping).

## Authority and current scope

This brief records the user's outline and subsequent design clarifications.
The full context is preserved in the
[bootstrap outline](tasks/bootstrap_20261030/project_outline.md).
Handoff setup is complete. The subsequent assignment authorized source/literature
and design assessment, focusing on individual vegetation, speed, memory and GPU
compatibility. Its [assessment](../docs/individual-vegetation-feasibility.md)
recommends an architecture and bounded phases for discussion. It does not select
scientific laws or authorize implementation. Commits, pushes, model runs,
installations and upstream changes require separately agreed scope.

## Purpose and ownership

Develop vegetation and persistent soil-water processes for eventual coupling
with MAPLE–SYRUP. Vegetation should respond to water, climate, establishment,
competition, mortality, and eventually landscape change, then provide consistent
physical properties to both transport models.

MAPLE owns aeolian transport and sediment storage; SYRUP uses that storage for
water erosion and sediment transport. MASY coordinates sequential wind and
water events. The two transport processes do not run simultaneously. Reuse
scientific implementations where practical, preserving independently usable
upstream models. Do not copy their transport physics into LEAF. Exclude
ECOTONE/WEMO wind physics entirely.

## Candidates and existing evidence

The user's subsequent design requires spatial individuals and biomass, so cover
alone is no longer the complete vegetation target. Tietjen remains a hydrological
reference. Caylor's spatial canopy/root methods, Hi-sAFe's shared soil voxels and
GRASSMIND's individual physiology supplement the three original candidates.
The assessment separates their relevant methods from drop-in code availability.

These are provided findings to verify against the actual selected sources,
papers, and final independent review dispositions; they do not select a model.

| Candidate | Ecological detail | Questions and reuse constraints |
| --- | --- | --- |
| Tietjen | Two water layers and dynamic grass/shrub fractional cover. | Small state may suit native CPU/GPU kernels if cover is sufficient. MAHLERAN adaptations require checking against the papers, including questionable source behavior. |
| STEPWAT2 | Individual plants, species, functional groups; daily multilayer SOILWAT2 water with monthly resources and annual vegetation. | Reusable C demography, but inspected global state, gridded water reinitialization, generated-weather assumptions, and calendar integration need attention. “Added transpiration” is resource accounting, not new physical water. |
| Supplied ECOTONE ecology | Individual biomass, age, species, and layered roots; annual ecology and daily water. | Local source is available. Intel/DEC portability, fixed-capacity memory, shared soil-type arrays, and active growth behavior require verification. No qualified portable ecological executable exists in the prior assessment. |

Prior assessments live under
`/home/okin/SYRUP/agent_handoffs/tasks/tietjen_feasibility_20261028/` and
`/home/okin/SYRUP/agent_handoffs/tasks/vegetation_model_comparison_20261030/`.
Read the final assessment sections and root dispositions, including
`root_review_disposition.json`, `review_initial.md`, and `review_ecotone.md`.
The local ECOTONE source supersedes earlier source-availability objections.

Neither STEPWAT2 nor ECOTONE was compiled or run in that assessment. Soil-type
indexing does not prove one daily column computation per soil type: ECOTONE's
annual driver calls the soil driver for each plot. No runtime ranking or measured
bottleneck is established. Active `gro_new.f` is principally water-driven;
historical nitrogen/light comments need not describe its growth formula.

## Water, state, and coupling contracts

### Clarified individual design and target scale

- Individuals retain stable identity, X/Y and **ground elevation at establishment**,
  total aboveground biomass, age, species and PFT. Do not add establishment plant
  height; current height and root/canopy geometry follow selected allometry.
- Roots and canopies cross cells and preserve subcell plant positions. Do not
  instantiate a separate whole community per hydraulic cell.
- Shared MASY soil/ground conditions, including evolving texture, are authoritative.
  The multilayer moisture component will likely reside in SYRUP.
- Between transport events, run water use, soil evaporation and drainage regularly
  (likely daily). Accumulate actual transpiration for coarser biomass updates
  (weekly is provisional). Transport and biological water use/growth are mutually
  exclusive; preserve elapsed physical/calendar time and partial intervals.
- Initially moisture does not modify wind transport; reserve a future interface.
- Species-specific burial, erosion/root exposure or dislodgement and abrasion
  responses can differ within a PFT. Add temperature/frost, age and stochastic
  mortality. Fire is deferred. Ground-height difference alone may not encode
  cumulative injury or root damage.
- Compare simple seed and vegetative recruitment. Define grass individuals
  (genet, ramet or clump) before assigning population densities.
- Target grids are 128²–512², with spacing interpreted as 0.25–2 m. Plant counts
  remain unspecified; capacity examples are illustrative, not ecological choices.

### Shared water and physical views

Maintain one authoritative water budget and one authoritative vegetation state.
A continuous mode must retain subsurface moisture across storms and dry periods,
accounting explicitly for ET and drainage rather than silently applying SYRUP's
event-finalization reset. Do not double-count rainfall/infiltration in SYRUP and
SOILWAT. Assign ownership of intake, storage, ET, drainage, and runoff/runon.
Water/root layers are distinct from MAPLE's sediment active layer and voxels.

The assessment must define storm/daily/seasonal/annual update order, persistent
state and RNG/restart, roots and demand by water layer, patch-to-hydraulic-cell
mapping, biomass/cover-to-surface properties, and eventual erosion/deposition
effects on depth, roots, burial, and exposure. A hybrid of existing hydraulics
and changed ecology/water requires documented departures and benchmarks, not an
exact-reproduction claim.

## CPU, GPU, and memory

The user subsequently authorized the bounded geometry prototype in parallel with
Jornada population estimates. Initial water discretization is **two or three
layers**. Supplied crown widths: **10–20 cm grasses and approximately 2 m
mesquite**. The [follow-up](../docs/jornada-capacity.md) distinguishes geometric
individual equivalents from observed foliar cover and records the executed
synthetic CPU/GPU measurements. Growth, hydrology, demography and full native
integration remain outside that prototype.

Evaluate CPU and GPU architecture from the start. Prefer compiled CPU kernels
and GPU-suitable arrays over translated Python loops. Compiled CPU ecology
between GPU storms is a valid initial option to benchmark. Avoid repeated
transfers, hidden synchronization, full-domain copies, maximum-population
allocation, and full histories for every cell. Bound storage and scratch by
actual populations and active work; stream seasonal statistics where possible.

Preserve individual plants if selected: cohorts or aggregate cover change the
science. Justify ecological patch/resource resolution separately from hydraulic
resolution. Measure startup and steady-state time, CPU/GPU peak memory, transfer
cost, and scaling while preserving conservation and equivalent restart. Do not
raise limits or discard state to pass checks.

## Assessment deliverable

Verify the provisional findings, identify the ecological detail needed for the
scientific questions, and propose bounded phases. Distinguish unchanged reusable
code, adapters, and new physics/numerical work. Re-establish source identities
before reliance; use the [upstream notes](upstreams.md). Model choice depends on
scientific questions, not coding convenience. No selection is made by this brief.

## Root-radius working range — user correction

Use lateral root radius **3–5 times canopy radius**. This replaces the initial
2× illustrative baseline, without changing individual counts or choosing root
depth. Grass diameters 10–20 cm imply root radii 0.15–0.50 m; mesquite diameter 2 m
implies root radii 3–5 m. Updated estimates and 3× endpoint measurements are in the
[Jornada follow-up](../docs/jornada-capacity.md). The prior 5× benchmark remains
valid; no implementation code or physical laws changed.

## Authorized performance follow-up

The completed daily-water task adds a bounded synthetic prototype only:
10950daily steps,1564weekly biomass/root-map updates, full plant retention and
water/biomass audits. CPU/GPU measurements and allocator findings are in
[the30-year report](../docs/daily-water-30year.md). This does not select scientific
parameters or implement coupled storm/soil-water physics. User root range3–5×
canopy radius, two/three layers and grid/population context are unchanged.

## Species-specific root profiles — authorized extension

The user requested grasses weighted toward surface layers and shrubs toward
deeper layers, with both retaining roots in all two or three initial soil layers.
The isolated benchmark now supports configurable named profiles and explicit
per-plant integer profile IDs, including additional species independently of
canopy size. Initial illustrative defaults are grass/shrub 80/20 versus 25/75%
for two layers, and 70/20/10 versus 15/25/60% for three layers (top to bottom).
These are uncalibrated fixtures rather than selected field parameters.

Daily uptake uses available cell/layer water and matching request/gather weights.
Biomass and horizontal root maps update weekly; vertical profile membership and
fractions remain fixed. This does not introduce adaptive rooting or a complete
species parameter library. See the [profile follow-up](../docs/species-root-profiles.md)
for configuration, independent checks and measurements. Preserve historical
v0.1 source/results separately. No upstream solver edits or new environments.

## Authorized native transpiration coupling

The user requested that vegetation read and deplete SYRUP's new subsurface state
and clarified that it is on a separate branch. The qualified opt-in implementation
now uses the actual `feature/subsurface-hydrology` column with a native transpiration
operator. LEAF holds no duplicate soil store in coupled mode. Daily hydro then T
returns actual uptake per plant; existing weekly biomass/horizontal root refresh
and fixed named layer fractions remain. Evaporation and transpiration have separate
budgets, with native active-cell storage and one cell-area conversion.

CPU232/GPU261 tests, independent analytical/mask/area/biomass/retry oracles and
bounded365-day capacity measurements are complete. The [guide](../docs/coupled-transpiration.md)
and [source lock](../dependencies/syrup-subsurface.lock.json) state exact sources,
commands, memory scopes and limits. Only the new native module/test were added
upstream; no existing source, branch, dirty work or shared-surface physics changed.

Storm/MASY scheduling, evolving texture/depth/root remapping, restart serialization,
recruitment/mortality and ecological calibration remain separate scope. The
continuous physical state is native; do not reactivate the standalone synthetic
water/recharge path when coupling. Future storm handoff must explicitly retain
and supply the current native state rather than a legacy dry reset.

## Study-based initial parameters — 2026-11-01

The user authorized starting STEP-style parameter derivation from Feldman grass
and Peterson–Neofotis mesquite, representing each two-seedling pot as one model
plant. Extensive biomass/water quantities double; relative rates and mass/water
ratios do not. The [study assessment](../docs/study-stepwat-parameters.md) and
[inactive candidates](../parameters/candidates/study-stepwat.json) preserve
observations, conditional estimates and unidentified traits.

Grass fitting uses training genotypes only, with 34 withheld dry-harvest checks
(RMSE 1.52 g, R² 0.70). Mesquite supports a conditional ambient AGB/water balance
prior near 0.264 kg/m³, without an invented absolute biomass time series.
The seedling growth rates are incompatible with the unchanged adapter cap;
weekly diagnostic timing, annual phenology, persistent/seasonal allocation and
adult capacity need scientific review before activation. Pot ET is not measured
T. No field-ready trait vector or native scenario is established by these fits.
