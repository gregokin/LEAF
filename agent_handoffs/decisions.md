# Decisions and Open Questions

## Share canopy geometry across wind and water consumers — 2026-11-10

User requests neutral naming and shared implementations so vegetation changes
apply consistently to MAPLE and SYRUP. Use `leaf.canopy.canopy_view` as the common
aboveground representation derived from authoritative PlantState, the species
catalog and current surface. Compute allometry, positions, dimensions, porosity
and burial/exposure through one code path. Keep stable IDs throughout.

Retain `maple_vegetation_view` as a compatibility adapter that delegates to that
common path and applies MAPLE's current native error/exclude/flag policies.
The shared view retains valid buried/exposed standing plants; native-MAPLE
restrictions are not universal plant validity or mortality rules. No duplicate
SYRUP allometry/rasterizer or new water/drag law. The CLI retains its existing
MAPLE projection. Upstream consumers still require integration and qualification;
they must not independently recalculate or store competing canopy geometries.
See [shared API and consumer responsibilities](../docs/shared-canopy.md).

## Phase 0 state and prescribed vegetation — 2026-11-10

User-approved scope implemented with Claude as sole candidate writer and Codex
as independent reviewer/installer. Root `AGENTS.md`, legacy numerical sources,
historical locks and upstream checkouts are preserved.

- One human-readable plant CSV schema serves initialization, snapshots and final
  output; species traits live in a separate versioned JSON catalog. Stable IDs,
  age in days, aboveground **dry** kg and establishment **ground** elevation are
  explicit. Initial `SURFACE` resolves once; no initial plant-height field.
- Generated cover means per-species union ground cover, clipped to a finite
  domain. Seeded positive-lognormal diameters use specified mean/SD; overlap
  placement uses uniform candidate positions. Exclusion is a sequential hard-core
  process with larger canopies placed first; it is not an unconditioned Poisson
  process. Numerical tolerance, resource bounds and ordering are recorded.
- MAPLE can use LEAF's NumPy-only loader/generator/allometry without growth,
  hydrology, Numba or CUDA. Use its native rasterizer with ID sidecars. Default
  publication uses exposed height and refuses fully buried or elevated-base
  plants unless an explicit handling policy is supplied.
- Porosity is constant per species (example 0.20). LEAF ecology does not use it;
  native MAPLE wind does. Species allometry and life-history fixtures remain
  uncalibrated. Life expectancy does not impose an invented mortality threshold.
- Age advances for elapsed intervals, including transport. Recording cadence
  never invents intermediate states. Event diagnostics distinguish erosion,
  deposition, flux and abrasion; species injury/recruitment laws are sockets.
- Legacy driver biomass remains its numerical authority while attached; capture
  B/E by ID at requested outputs. Exact diameter-law compatibility and seasonal
  mass preservation are checked. CSV is not a full coupled restart.

See [interfaces and limits](../docs/phase0-vegetation.md),
[MAPLE handoff](../docs/maple-leaf-integration.md) and
[current source lock](../dependencies/phase0-vegetation.lock.json). The statements
about unimplemented geometry and open porosity in the historical Nov 8 section
below are superseded. No upstream integration or ecological calibration is claimed.

## Project name — 2026-11-09

The user selected **Landscape Ecohydrology with Agent-based Flora (LEAF)** as
the official project name. Use this expansion in project documentation.

## Canopy geometry and transport feedback — 2026-11-08

User direction: derive canopy height and width from each individual's
**aboveground dry biomass** through species-specific allometry. These canopies,
with individual locations and identities, will underpin MAPLE shear-stress
partitioning. Keep one authoritative LEAF plant population and publish derived
MAPLE vegetation geometry.

Hold porosity constant initially; its numerical value remains open, and it is
**unused in current LEAF code**. Add biomass-dependent porosity later. Current
LEAF has simplified biomass-to-canopy-radius geometry; height/porosity allometry
and the MAPLE publication adapter remain unimplemented.

LEAF accesses moisture and actual transpiration through SYRUP. It also needs
MAPLE wind-event diagnostics and shared-surface changes for species-specific
abrasion, burial and erosion/root-exposure responses. Include terrain changes
from water events in that same injury handling. LEAF applies plant responses
and refreshes exposed geometry; MAPLE retains wind/shear physics and SYRUP
retains hydrology. Growth/transpiration remain inactive during transport events.

This is a documented coupling requirement, not an implemented MAPLE interface
or injury law. Detailed soil-layer/water remapping remains deferred. See the
[current brief](project_brief.md#canopy-allometry-and-maple-coupling--2026-11-08).
No runtime, defaults, locks, upstream sources or accepted task archives changed.

## Uptake candidates and paired validation — 2026-11-01

User direction: retain the discussed Jornada uptake values as potential
parameters and prioritize experiments pairing individual growth with water use.
Store the literature coefficients/uncertainty and dry/wet depth fractions in
[an inactive candidate record](../parameters/candidates/jornada-uptake.json).
Do not alter defaults or infer absolute transpiration, root biomass or WUE from
the tracer. Layer depths, profile-tail handling and shrub biomass-to-volume
conversion remain future choices.

Research recommendation: start with the verified Setaria water/harvest joins,
qualifying image-derived increments and evaporation before fitting; seek the
Biosphere 2 mesquite raw pot observations for a woody test. Ravi–Breshears grass
physiology/mass/survival is useful but its published methods do not establish
whole-plant transpiration. Access failures do not establish absent data.
See [paired-data survey](../docs/paired-growth-water-data.md) and
[durable evidence](tasks/paired_growth_et_survey_20261101/README.md).
This does not authorize calibration, model runs or a runtime parameter change.

## Validation-data research recommendation — 2026-10-31

Use Jornada NPP grass/mesquite sites as the first candidate validation system,
with separate tests for seasonal biomass, persistent profile water and daily
dry-down/ET. Reserve sites/years and harvests before calibration. Seasonal biomass
and annual NPP share source observations; NPP cover is cumulative and cannot be
treated as true canopy union cover. Whole-shrub woody-mass conventions, biomass
allometry revisions/provisional years, data access/QC, and calendar mapping must
be resolved before a field validation run. The new root tracer data can challenge
fixed uptake profiles but do not directly measure daily transpiration.

This is Codex's recommendation from the requested survey, not a newly approved
simulation or scientific parameter set. See [inventory](../docs/validation-data.md)
and [evidence](tasks/validation_data_survey_20261031/README.md). No numerical source,
locks, upstream files, root AGENTS.md or prior accepted records were changed.

## STEP-style growth implementation — 2026-10-31

The user authorized STEP-consistent growth/water use and explicitly approved
the bounded candidate-source/specification transfer to Anthropic, reconfirming
automatic orchestration. Claude implemented and corrected; Codex independently
reviewed the equations, native coupling, memory and source scope.

Use opt-in `stepwat_traits`: actual daily native transpiration supplies annual
biomass-support resources pro-rated against demand, limiting species-specific
logistic size growth. Biomass/root maps update weekly and at year-end; the
internal seasonal aboveground component is removed at year-end with an explicit
loss ledger. Total aboveground biomass is the observational target. Keep fixed
species root-layer fractions, existing spatial root competition and native water
ownership. No added-transpiration adjustment, dry growth fallback or root/tissue
carbon pools. Explicit traits, monthly activity and daily host temperature factors
are required. Parameters are uncalibrated; calendar and spatial adaptations are
documented. Legacy fixture defaults persist. See
[guide](../docs/stepwat-growth.md), [source lock](../dependencies/stepwat-growth.lock.json)
and [evidence](tasks/stepwat_growth_20261031/README.md).

## User decision — total aboveground biomass, 2026-10-31

Track one total aboveground biomass value per individual, reflecting the greater
availability of observations. Separate tissue and belowground mass pools are not
required. Daily actual transpiration supports production, with weekly aboveground
updates and species/PFT allometry for canopy/root geometry. Retain species-specific
vertical root fractions and SYRUP's authoritative water state. An aboveground
production efficiency must consistently include implicit belowground allocation
and respiratory costs; a whole-plant coefficient is not interchangeable.

The exact efficiency, climate response, aggregate loss rules and observed live/dead
mass convention remain open. This supersedes mandatory tissue-pool development in
the original survey recommendation, not its descriptions of published models.
Documentation only: no numerical implementation, calibration or model run assigned.
See the [current design](../docs/water-to-biomass-survey.md).

## Water-to-biomass survey — original recommendation, subsequently narrowed

The user requested a survey of grass/shrub growth methods. See
[the comparison](../docs/water-to-biomass-survey.md) and
[source evidence](tasks/water_biomass_survey_20261031/README.md).
Recommendation: retain native actual uptake and develop a small production module
with explicit carbon/dry-matter units, daily climate/phenology and weekly tissue
allocation. Compare constant and climate-corrected transpiration efficiency;
separate leaf/root turnover from persistent shrub structure. Do not mix inverse
ECOTONE WUE, STEPWAT2 biomass-support coefficients and gross/net efficiencies.
ECOTONE, daily savanna carbon accounting and GRASSMIND offer complementary methods;
Tietjen cover growth is not an individual biomass ledger. This is a recommendation,
not selection or authorization of a growth law, parameter set or new simulation.
All qualified runtime code and accepted evidence remain unchanged; evolving-surface
remapping remains explicitly deferred.

## User clarification — defer detailed surface remapping

For now, retain a simplified soil-layer representation with texture-dependent
soil-water properties. Defer explicit erosion/deposition-driven changes to soil
material, layer geometry, stored-water placement and root access. Record these
as future implementation; they are not a prerequisite for the current vegetation
and native hydrology work. The shared MASY surface remains the long-term source
of ground/soil conditions. Existing daily water accounting and weekly root-map
updates from growth are unaffected. This clarification changes documentation and
scope only; no numerical code or accepted qualification evidence was changed.

## Native transpiration coupling — 2026-10-31

The user authorized implementation against SYRUP's separate subsurface branch.
Use actual `feature/subsurface-hydrology` at HEAD `deb60c29834a4aad0bf8cb4cfbb611fc28ebe090`
plus the exact uncommitted source identities in
[the source lock](../dependencies/syrup-subsurface.lock.json). MAPLE runtime HEAD
is `963b06ad1cf0f3e2c420a7d7fb50939e3bd43b7d`, independently qualified as used.

SYRUP owns one authoritative soil-water state and the physical transpiration
sink. LEAF owns plants, species-root fractions, demand and actual-uptake-driven
weekly growth. The opt-in native module performs hydro then residual-limited T;
E and T have distinct budgets. Daily depth requests and plant uptake volumes
match through exactly one cell-area conversion. No wetter-layer compensation
or new species wilting law was selected. Standalone benchmark defaults persist.

Claude authored the isolated candidate; independent Codex reviews/reproductions
and correction rounds resolved masked budgets, device binding, admission order,
complete reserves, weekly retry and NVRTC compilation. CPU232/GPU261 tests pass;
two GPU skips test deliberately unavailable hardware paths. A refused weekly
rebuild retains its accepted physical day/uptake and latches until explicit retry.
Memory estimates and measured allocator peaks are distinct; no cap was raised
and no individual was discarded. All real CUDA work used allocated GPU0.

Only a new native module/test were added to SYRUP; every original upstream file
and dirty change was preserved. No commits, pushes, package installs or new environments.
Daily interevent coupling is qualified; storm/pond/MASY handoffs, changing-ground
or texture remapping, restart serialization and production ecology are later scope.
See [guide](../docs/coupled-transpiration.md) and
[task evidence](tasks/coupled_transpiration_20261031/verification.md).

## Bootstrap user direction — 2026-10-30 (historical scope)

- Work in `/home/okin/leaf` and use Claude for bounded implementation with
  independent Codex review and verification.
- Maintain automatic orchestration for requested tasks until changed. The
  current task is documentation setup; no unbounded research or background
  service is authorized.
- Read and preserve the supplied project outline. Compare Tietjen, STEPWAT2, and
  supplied ECOTONE ecology before proposing implementation phases. No candidate
  or architecture has been selected.
- Leave scientific implementation, commits, pushes, model runs, new environments,
  package installation, and upstream changes outside this setup.
- Preserve the existing root `AGENTS.md`; add project-specific workflow guidance
  in this directory and connect it through `README.md` and `claude.md`.
- Use ordinary, explicitly scoped Claude permissions and one writer per
  implementation scope. Claude leaves changes uncommitted.

## Bootstrap execution record

The first Claude attempt failed with API/DNS error `EAI_AGAIN` before producing
assigned files. It was not a usage limit. An attempted network-enabled retry
containing the detailed project outline was rejected by automatic approval
review before launch. Do not resend that rejected payload without resolving its
authorization. The safer template request contains generic workflow instructions
only, with no project context or file tools. The outline and scientific brief
remain local. Exact prompts, launch records, and outcomes are archived under
`tasks/bootstrap_20261030/`; the current state records the actual result.

## Original assessment questions

- Which scientific questions require species, individual age/size, rooting,
  establishment, competition, disturbance, or only grass/shrub cover?
- Which ecological patch/resource resolution and climate/forcing calendars are
  scientifically justified relative to the hydraulic grid?
- Which component will own each water-budget term, and which persistent-water
  and event-finalization changes require new scientific/numerical decisions?
- Which unchanged code, adapters, portability work, and new model laws are needed
  for each option, with what CPU/GPU, memory, and restart qualification phases?

These questions motivated the subsequently authorized assessment. Do not invent
scientific laws merely to complete scaffolding or make a capacity test pass.

## Subsequent user direction — individual vegetation assessment

The user authorized examination of existing methods/code and the proposed hybrid
architecture, focusing on speed, memory and GPU compatibility. Target scales:
128²–512² cells, 0.25–2 m spacing; plant density remains open. Retain X/Y and
ground elevation at establishment, not plant height at establishment. Preserve
individuals, biomass, PFT allometry and species-specific burial/erosion/abrasion
responses. Shared MASY surface and evolving texture control soil conditions.
Daily interevent water use/drainage and coarser biomass updates are provisional;
soil moisture initially has no wind effect. See [current brief](project_brief.md).

## Assessment recommendation — pending scientific choices

The [assessment](../docs/individual-vegetation-feasibility.md) recommends spatial
plants with sparse root-to-layered-grid connections. ECOTONE offers a direct
water-to-biomass precedent, Caylor spatial structure, Hi-sAFe shared soil voxels,
and STEPWAT2 species demography. No complete existing driver meets the combined
spatial, calendar, persistent-water and GPU contracts unchanged.

Recommend measuring mapping/canopy projection with compiled CPU/GPU kernels
before a full port, then qualifying persistent water with fixed plants, individual
growth and finally demography/event injury. These are proposed phases, not an
approved implementation assignment or accepted scientific parameterization.
Root graph size is a capacity risk; no runtime ranking has been measured.

The latest public-only Claude review completed and was corrected after Codex
review. Codex independently integrated local source and user design. The earlier
rejected private-outline payload was not retransmitted. Durable evidence is in
`tasks/individual_vegetation_feasibility_20261030/`.

## Jornada capacity and bounded prototype — subsequent authorization

The user supplied 10–20 cm grass widths, approximately 2 m mesquite widths,
and two/three initial soil layers; requested Jornada cover-based hectare counts
and memory estimates; and authorized the recommended geometry prototype in
parallel. This supersedes the prior implementation gate only for that bounded
synthetic mapping/reduction/canopy task.

The [follow-up](../docs/jornada-capacity.md) records the literature and results.
All 20 unit tests and 11 CPU/GPU comparison cases passed. Sparse map capacity is
promising at these counts; root extent, soil/plant science and complete model
memory remain separate choices. The 8×8 canopy sampler can omit small crowns
in coarse cells, so production root access must remain geometric and future
canopy publication needs a representation decision. No ecological laws or
upstream integration were implemented. Model runs, installs and Git actions
remain outside this completed task.

## Root-radius correction — 2026-10-30

The user supplied roots of 3–5 times plant radius. Adopt 3–5× canopy radius as the
working lateral range; supersede the previous 2× illustration. Counts and two/three layers
stay unchanged; this is not a root-depth rule. The existing benchmark code was
reused at 3× in four new CPU/GPU cases, all passed. Existing 5× results are preserved.
Updated memory ranges are in the Jornada follow-up. No new coding assignment,
scientific model or upstream change was introduced.

## Daily water/growth performance benchmark — 2026-10-30

The user separately authorized simple daily water accounting and weekly growth
for30modelyears. Claude authored the isolated benchmark; Codex reviewed and ran
compiled CPU/GPU variants. Fourteen full runs and final36-test suites pass.
The [report](../docs/daily-water-30year.md) supports feasibility at the specified
grids/populations. Synthetic recharge, WUE, turnover and allometry remain
performance fixtures, not selected science. No upstream solver changes.

Default CuPy caching accumulated materially more memory than live arrays.
A separate identical-workload repeat applies the existing1.5billion-byte cap
to the pool, retains all state and matches budgets/counters/scalar outcomes.
Future coupling must budget persistent state and allocator/scratch reuse.

## Species-specific root profile implementation — 2026-10-31

The user authorized extending the synthetic daily-water benchmark with different
vertical root distributions by species/PFT. Use a small FP64 profile/layer table
and int32 per-plant IDs with the existing horizontal CSR graph. Apply identical
fractions to cached demand and daily accepted-uptake gathering. All configured
fractions are finite, positive and normalized; all layers retain roots.

Keep fixed profile traits distinct from weekly biomass, radius and graph updates.
Named profiles and explicit IDs support additional species without inferring
rooting from canopy size. Existing two-class WUE remains a separate fixture;
no full ecological species library is implied. Default surface/deep fractions
are illustrative and configurable, not literature-calibrated.

Claude authors code/tests/benchmark guide; Codex independently reviews, verifies
and measures. Current measurements retain the original plant populations,
10-million-edge bound and 1.5-billion-byte cap. Preserve old equal-layer source
and accepted evidence; report v0.2 measurements separately in the
[profile follow-up](../docs/species-root-profiles.md). No upstream edits, installs
or Git actions. Task: `species_root_profiles_20261031`.

## Study parameterization — 2026-11-01

User directs initial parameters from Feldman et al. grass and Peterson &
Neofotis mesquite; treat both seedlings in a mesquite pot as one plant. Keep
all extensive quantities on this aggregate basis and leave intensive quantities
unchanged. Claude authors bounded analysis; Codex independently reviews and
verifies. Task: [study_parameterization_20261101](tasks/study_parameterization_20261101/README.md).

Accept the reproducible pipeline and inactive candidate evidence. Grass joint
normal-only fit: support 0.01488 kg/m³ paired with rate 295.1/year and a conditional
17.59 g capacity scenario. Its separate balance prior is 0.08335 kg/m³; do not
substitute it into that fitted pair. Withheld actual harvests: n=34, RMSE 1.52 g,
R² 0.70. Mesquite ambient conditional AGB balance prior: approximately
0.264 kg/m³; final daily aggregate-pot T: 15.9–20.6 mL/day. Final two-day T is
not cumulative water consumption. Aboveground allocation is conditional.

Do not activate these candidates or silently raise the adapter's rate cap.
Both studies' fast seedling growth conflicts with its current normal annual-rate
range. Diagnostic timing and adult/annual phenology, mature size, seasonal mass,
canopy/demand and ET/T partition require resolution before runtime traits can
be selected. Runtime/locks and earlier accepted archives are preserved.

## Phase1 mechanics accepted —2026-11-11

See [final independent review](tasks/leaf_phase1_interevent_20261111/final_review.md). STEP-style requested-support planning plus actual/requested water once; species traits co-located, authoritative real SYRUP withdrawals, dated interevent session, weekly shared geometry and complete LEAF-native checkpoint. Fixtures are uncalibrated. Full transport/E-partition/rational-calendar adapters and evolving-surface remapping remain deferred. User authorizes publication and automatic progression to Phase2/3.

## Phase2 mechanics and Phase3 ramet choice —2026-11-11

Accept opt-in species-specific mechanical injury, age/background/stress mortality and complete version2 restart; [independent review](tasks/leaf_phase2_20261111/final_review.md). Fixtures are uncalibrated, standing-dead removal remains deferred, and native water is untouched by instantaneous biology. The user explicitly selected vegetative reproduction as nearby daughter plants with their own IDs and ages. Phase3 will implement that choice alongside simple seed establishment; publish Phase2 first.

## Phase3 reproduction mechanics accepted —2026-11-11

Accept simple_seed_clonal_v1, following the user's explicit nearby daughter-plant choice: separate IDs/age0/current establishment ground, parent-funded dry-AGB/E transfers, external seedling credit, fixed physical ecological patches, overlap allowed and finite bounded placement. Species numbers remain uncalibrated and co-located. Shared population/root/canopy transactions preserve actual native water and existing partial-year metrics; prior seasons finalize explicitly before births. Full restart becomes format3 with strict sequential opportunity/RNG history. [Independent acceptance,actual CPU/GPU tests and bounded memory/timing](tasks/leaf_phase3_20261111/final_review.md). Publish whole Phase3; no additional implementation phase is inferred.

## Plain plant tables — user decision, 2026-11-11

Plant CSV files must contain only the canonical column header and individual plant records. Never embed JSON, provenance, run configuration or other metadata in plant files. Keep contextual metadata in companion JSON or the run/checkpoint manifest. Apply this to initial input, snapshots and final output; preserve species parameters separately. Legacy embedded metadata may be read for compatibility, but must never be written. User explicitly authorized commit and push after this change is independently accepted. Task: plain_plant_csv_20261111.

## Initial generated ages — user decision, 2026-11-11

Normally generated individuals receive independent uniform ages from zero to their species nominal life expectancy, read from the species catalog. This is an initial-condition rule, not a deterministic death age or an age-biomass relationship. Supplied plant records, restart ages and age-zero recruits retain their explicit ages. Existing preview remains historical; the next requested initialization will use the new rule.
