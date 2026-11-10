# Phase 0 vegetation infrastructure — revision 1

You are Claude, sole implementation writer in `candidate/`. Codex is the
orchestrator, scientific/specification authority, reviewer and test executor.
The user explicitly approved this Phase 0, sharing its bounded source/spec with
Anthropic, and automatic implement/review/correct orchestration. Codex will
commit/push accepted work; YOU MUST leave it uncommitted. No new environments,
installs, network tools, shell, subprocesses, other agents or upstream writes.
Use Read/Edit/Write tools only. Do not claim to have executed checks.

## Governing workflow and source scope

The governing instructions are embedded here for safe-mode launch. The current
assignment is this prompt; baseline is ../baseline.json. One writer owns only
the exact candidate write paths below. Codex owns controls, records, tests/runs
and upstream inspection. Preserve every other file. Root AGENTS.md is immutable.
On a genuine tool/auth/network/rate-limit failure preserve work and report the
literal error. Automatic preference remains enabled. Finish with report.md
stating files, behavior, limitations and tests NOT RUN. Codex verifies separately.

May read: this prompt; candidate files you create; these copied current sources:
leaf/__init__.py, leaf/coupling.py, leaf/stepwat.py,
benchmarks/daily_water_growth.py, benchmarks/spatial_mapping.py. No other
repository brief, raw research data, credentials, prior archives or upstream
source may be read. All required user design context is in this prompt.

Allowed writes, relative to candidate/:

- leaf/__init__.py (replace eager exports with compatible lazy imports)
- leaf/traits.py
- leaf/state.py
- leaf/canopy.py
- leaf/initialization.py
- leaf/recording.py
- leaf/events.py
- leaf/runtime.py
- leaf/__main__.py
- tests/test_phase0.py
- docs/phase0-vegetation.md
- docs/maple-leaf-integration.md
- examples/phase0/species.json
- examples/phase0/plants.csv
- examples/phase0/generation.json
- pyproject.toml
- report.md

Do not change copied leaf/coupling.py, leaf/stepwat.py or benchmarks/. Do not
write outside candidate/, change control files, use git, run Python or install
anything. You have generous room in this one bounded task: implement complete,
testable infrastructure with clear modules rather than a speculative scaffold.

## Purpose and scope

Official name: Landscape Ecohydrology with Agent-based Flora (LEAF).
Phase 0 initializes, tracks, records and exchanges spatial individual plants.
It MUST work in MAPLE wind-with-prescribed-vegetation mode without soil water,
ET, growth, CUDA, Numba, MAPLE or SYRUP imports. Fully coupled growth is optional
later; no existing equations, bounds, fixture defaults or locks change now.
Use Python stdlib + existing NumPy for the independent Phase 0 interface. A
lazy optional compiled acceleration inside generation is acceptable only if
the pure NumPy/stdlib path remains functional and documented.

MAPLE owns wind/shear physics, SYRUP owns water and actual transpiration, LEAF
owns plants/species/geometry and ecological damage responses. Wind, water and
growth/T occupy exclusive intervals. Detailed evolving soil-layer/water/root
remapping and calibration remain deferred. No recruitment or damage physics
is invented here. Species traits and response sockets are provided for later.

## 1. Separate species JSON and one human-readable vegetation-state format

Use a versioned species JSON file, keyed by stable, case-sensitive species IDs
such as grass, shrub or BOER. Validate before mutation/output: duplicate JSON
keys, invalid numbers, unknown schema version, unknown species and inconsistent
shapes/fractions fail clearly. No NaN/Infinity, bool-as-number or silent clipping.
Define explicit units and model/schema identifiers. Catalog content hash is
canonical and stable and accompanies generated/saved states.

Species traits include:
- Functional type/name.
- Monotone positive power allometry for diameter and height versus total
  aboveground DRY biomass kg. Prefer clear a/b coefficients with documented
  dimensions: diameter_m = a_d * B_kg**b_d and height_m = a_h * B_kg**b_h.
  Supply an exact inverse diameter -> B for generation.
- Lateral root-radius multiplier and named layer fractions (initially 2/3
  layers; validate finite, nonnegative, unit sum; a consistent L per catalog).
- Constant porosity in [0,1]. The examples can explicitly choose 0.20 to match
  current MAPLE convention, labeled uncalibrated/constant. LEAF water/growth
  does not use it; do not claim MAPLE wind ignores it.
- Growth/water-use parameters, reproduction configuration, life expectancy
  with explicit time units, age-dependent mortality configuration and
  burial/erosion/abrasion sensitivities. Explicit `unimplemented`/null sections
  are appropriate for unselected laws; no magic default mortality or new
  scientific parameters. Example numbers are fixtures, never calibrations.
  Expose these sections cleanly for future consumers and preserve on round trip.

Use CSV for the plant-state rows. One CSV schema for initial input, evolving
snapshots and final output. A short leading `# ...` JSON metadata line is fine
for schema, clock, domain/units, catalog hash and generation provenance; a
handwritten initial CSV can omit metadata if the API supplies context.
Columns include stable unique int64 plant ID, species, x_m, y_m,
establishment_ground_m, agb_dry_kg, age_days, status and seasonal_agb_dry_kg.
Include any additional vegetation accumulators needed by your actual bridge;
do not pretend this file checkpoints soil water/RNG/full MASY internals.

Use compact NumPy structure-of-arrays state, int species indices + one species
name table, stable IDs independent of row ordering, status codes with a clear
human-readable CSV mapping. Retain removed/dead identities; never silently
compact/discard them. Aboveground dry mass is the single main biomass state;
seasonal mass E is a subcomponent (0 <= E <= B), not an independent tissue pool.

Explicitly support `SURFACE` in the establishment-ground CSV column for initial
placement. Never interpret arbitrary missing/NaN strings as SURFACE. Resolve
that token ONCE from the caller's initial surface, save the concrete elevation
thereafter, and preserve it after erosion/deposition. No establishment plant
height is needed. Unresolved state cannot be published to MAPLE or used as a
noninitial snapshot. Positive/live/dead-standing mass and finite nonnegative
ages are validated. Removed plants can have zero mass. Empty populations work.

Provide read/write/update/advance-time functions. Full finite FP64 values must
round-trip (no rounded mass/position loss). Ages advance with elapsed calendar
time, including transport intervals; ecological stepping is not required to
track age. Do not advance age twice when exporting repeated snapshots.
Dead plant age at death can remain fixed; document the convention. No mortality
is applied just because age reaches life expectancy (mean != hard maximum).
File reads and updates are all-or-nothing. Reject malformed/truncated/duplicate
ID rows, unsupported statuses, bad dimensions, nonfinite/out-of-domain positions,
negative masses/ages, E>B, catalog mismatch and invalid units/versions clearly.

## 2. Shared placement, canopy and surface API

Use explicit Domain(nx,ny,dx_m[,dy_m,origin_x_m,origin_y_m]) information. Array
axes are (ny,nx), x=east, y=north, cell centers origin+(col+.5)*dx, etc.; validate
shape, units and finite values. Nearest containing-cell surface sampling is an
acceptable Phase 0 convention; name it and test subcell/edge cases. No silent
transpose, integer-cell treatment or boundary wrapping. Declared finite-domain
edge semantics apply consistently to placement and cover; no hidden periodicity.

The independent API reads specified plants or generates them, resolves SURFACE,
and returns derived geometry arrays with IDs/species/X/Y, allometric height,
diameter/semi-axes, establishment elevation and constant porosity. It does not
create landscape/soil/growth state. Canopy and roots may span many cells.
Current surface and establishment reference remain distinguishable. Expose
geometric burial/exposure diagnostics when current surface is supplied; do not
invent root dislodgement or mortality. Be explicit whether height is intact or
exposed and whether elevated canopy bases can be represented by current MAPLE.

Retain one canonical population; geometry is derived, not independently editable
plant authority. O(P) arrays; no P*N search or permanently stored trajectories.

MAPLE source inspected by Codex (read only): its VegetationObjectSet fields are
center_x_m, center_y_m, semi_major_m, semi_minor_m, orientation_deg, height_m,
porosity, drag_coefficient. Coordinates are domain-local metres, +x east/+y north,
cell centers ((col+.5)*dx,(row+.5)*dy). Circles have equal semi axes. Drag=1.0 is
metadata there; default porosity=0.20 IS used by native wind. MAPLE rasterizer
already exists and owns native raster semantics. Do not copy it or wind physics.
LEAF can return a plain dict/SoA view compatible with these fields, plus stable
ID/provenance sidecars; never import MAPLE just to load/derive canopies.

## 3. Cover-driven multispecies generator

Inputs per species: target fractional cover of domain [0,1), diameter mean_m
and std_m, and initial age (constant or explicit distribution). Global seed,
overlap allowed/disallowed, domain and bounded resource/tolerance settings.
Use a positive lognormal diameter distribution matching the requested population
mean/SD (std=0 is constant); sample statistics vary and must be reported.
Infer each plant's AGB from its sampled diameter using species allometry.
Calculate count using E[d^2]=mean^2+std^2, not mean diameter squared alone.

Default cover definition is TRUE PER-SPECIES projected union ground cover,
counting same-species intersections once and clipping to domain. Cross-species
overlap contributes separately to each species' cover. Distinguish nominal sum
of crown areas and union cover in diagnostics. In overlap mode, the Poisson
Boolean estimate N ~ -A*log(1-f)/E[crown area] is a starting count, with finite
edge/sample corrections and declared tolerance. In nonoverlap mode nominal
count starts N ~ A*f/E[crown area]. Counts are integers; do not promise exact
continuous cover or exact finite-sample mean/SD. Report targets, measured cover,
measurement resolution/error basis, count and sample diameter stats.

Overlap allowed: independent uniform random centers, conditioned on chosen count
and any explicit acceptance on cover; document that distinction from an
unconditioned Poisson process. Overlap disallowed: sequential random proposals
with cross-species canopy exclusion (roots can overlap), not a pure Poisson
process. Use a bounded spatial index/binning scheme, never all-pairs P^2 at
production populations. Randomize species placement/interleave if needed to
avoid hidden priority; record any selected ordering convention.

Measure union cover with a bounded, explicitly documented deterministic spatial
sampling/raster method (and adjustable resolution), or an equally bounded method.
Never claim approximate sampled cover is exact analytic area. Ensure independent
fine-resolution tests on simple circles and representative generated cases.
Target-cover acceptance must be tied to stated measurement/tolerance, account
for same-species overlaps and domain clipping, and fail transparently if an
unattainable request exhausts attempts/resources. Do not silently change
diameters, drop individuals, return incorrect cover, relax no-overlap or raise
limits. Preserve RNG provenance. No maximum population/history per every cell.

Bounds: max_plants, max_attempts and max_cover_cells/bytes are explicit and
checked before large allocations. Do not hold a P x N or P x P array. This must
be practical at ~60k individuals/512^2 cells and explain scaling to denser grass
populations. Pure Python per-plant I/O/setup is acceptable where bounded; daily
ecological numeric loops are not being introduced.

## 4. Recording, runtime bridge and event sockets

Snapshot writer has user-selected cadence in elapsed model days; explicit initial
and final output are supported. The caller advances time, including event time.
Write only when due; no mandatory daily/weekly dumps, no P history in memory.
Specify behavior if a caller skips past multiple requested output times: do NOT
fabricate/interpolate missing historical states. State times/cadence are explicit.
Atomic output and default refusal to overwrite existing files. Validate first;
failed writes/validation must not advance writer state or partially replace data.

Provide a narrow bridge to capture vegetation snapshots from current Simulation
or CoupledSimulation (B, seasonal E if STEP, stable ids), preserving the template's
species, establishment anchors and initial ages. Snapshot copying happens only
on requested output, not daily. Handle ID reordering by ID, not array position;
refuse missing/unknown/duplicate IDs and mismatched profile assignments.
Document that initialization/restart of the existing growth driver is restricted
by its current area-proportional canopy law and separate pending-water/calendar
state. Do not silently feed an arbitrary power allometry to a legacy driver
that would recompute a different initial biomass. Do not change that driver's
science in Phase 0. Static MAPLE initialization is unrestricted by that legacy
growth law. Distinguish vegetation state replay from full coupled restart.

Event diagnostics are a versioned, stable-ID keyed, explicit-time/unit interface
for MAPLE wind/shared-surface and SYRUP water-event observations: erosion,
deposition/burial, abrasion exposure and optionally transport flux. Physical
surface lowering, raising and flux are distinct; no invented conversion from
flux to abrasion/injury. Define nonnegative depths/doses and any signed delta
convention clearly. Provide validated import/construction and a response hook
that later species laws can use to propose biomass/status changes. Default is
diagnostics/no ecological response, and must never invent threshold mortality.
Unknown IDs, duplicate ambiguous diagnostics, bad units/time/data fail before
mutation. Handler failure must not partially change vegetation. No surface
remapping or real MAPLE/SYRUP modifications.

## 5. CLI, examples, documentation, packaging

`python -m leaf` provides useful initialize-from-list, generate-from-cover and
inspect/export-canopy commands with --help. Surface can be .npy (allow_pickle=False)
or an explicit constant surface for simple examples. Configs contain domain,
species/generation settings, seed/limits and output cadence where appropriate.
Never run native hydrology/growth merely to load a population. Atomic output,
errors with nonzero status and no overwrite by default. Supply working examples
for two species, SURFACE and explicit ground anchors, age, fixed porosity and
fixture allometry/growth/reproduction/mortality/injury sections. Users can copy
and edit these; no literature calibration is claimed.

Minimal pyproject.toml makes leaf an installable package with a NumPy-only core;
growth dependencies optional, existing benchmarks namespace included if needed
for legacy lazy exports. Do not install. Keep `from leaf import CoupledSimulation,
CoupledPhaseError, stepwat` compatible via lazy module __getattr__ or equivalent.
`import leaf`, `leaf.state`, initialization and CLI must not import numba/cupy/
maple_syrup/benchmarks daily kernels. Existing full CPU/GPU suites must still pass.

docs/phase0-vegetation.md: schemas, public API/CLI, explicit conventions, coverage
approximation/limits, reproducibility, age/cadence/state replay and examples.
docs/maple-leaf-integration.md: concrete handoff for Codex-Claude MAPLE developers:
no-vegetation vs prescribed LEAF vegetation vs fully coupled; call shared LEAF
loading/placement/allometry; fields, coordinates, surface code resolution and
species file hash; use MAPLE's existing rasterizer; IDs retained beside rasters;
porosity is constant/unused by LEAF ecology but native MAPLE wind uses it;
event-feedback schema/hooks/units; unsupported buried/elevated-base cases must
be handled explicitly, not silently flattened. No MAPLE changes in this task.

## Acceptance tests (Codex runs, you write meaningful tests)

Use unittest, no new framework/dependency. Cover: positive/inverse allometry,
schema errors and exact CSV numerical round trips; SURFACE resolution once,
subcell/non-square/offset domain and boundaries; unknown species/IDs and invalid
data transactional behavior; reproducible generation, species cover/counts with
size variance, overlap allowed and global nonoverlap, bounded impossible cases,
empty/zero-cover cases; analytically checked circle cover and clipped edges,
independent finer coverage check; lifespan metadata without automatic death,
age advancing in wind-only mode; cadence initial/final/skip/overwrite/error
behavior; canopy view purity, stable IDs and elevation; event ID/time/unit checks
and failed handler rollback; bridge captures actual B/E by ID with profile/age
validation; import isolation in a subprocess (test executor may launch subprocess,
you may not), CLI round trips, examples and old public lazy exports.

Codex will execute with existing /home/okin/MAPLE/.venv-cupy/bin/python and
PYTHONPATH including candidate and existing SYRUP/src only for legacy tests.
No approved scientific growth-rate change, full storm scenario, ecological
calibration or new 30-year model run. Phase 0 bounded performance checks are
authorized. End with report.md and a concise implementation summary.
