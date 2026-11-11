# LEAF Phase 3 — simple seed establishment and nearby vegetative daughter plants

Task leaf_phase3_20261111; baseline5e887e66415f0522771ea789654b8345226d2496.
Phase1 f0d0b56 and whole Phase2 5e887e6 are independently accepted, committed AND pushed.
Exact session a5c6d72f-e529-49ec-beea-b376caaa8814. User explicitly authorizes automatic Codex-Claude implementation,
relevant source sharing with Anthropic and whole-phase commit/push by Codex after acceptance.
User explicitly selected: vegetative reproduction creates nearby daughter plants, each with
its OWN ID and age, rather than adding clonal-recovery biomass to the parent.

## Roles, bounded context and tools

Claude is sole implementation writer in this isolated candidate. Codex scopes/reviews,
executes all checks, installs accepted bytes and publishes. Ordinary Read/Edit/Write only;
NO Bash, execution, agents, settings, Git, installation, network or outside reads. Read
only this prompt and candidate/**, inventoried in SOURCE_INDEX.txt. No raw datasets,
credentials, terrain/weather inputs, other tasks or upstream source. Already-inspected
scientific precedents are in reference/method_notes.md. Never claim exact STEP/ECOTONE
reproduction or calibrated ecology. All author checks are NOT RUN; report honestly.

Allowed writes EXACTLY:
leaf/recruitment.py
leaf/interevent.py
leaf/checkpoint.py
leaf/demography.py
leaf/__init__.py
tests/test_recruitment.py
tests/test_interevent.py
tests/test_demography.py
examples/phase3/species.json
examples/phase3/run_recruitment.py
docs/phase3-reproduction.md
docs/maple-leaf-integration.md
README.md
report.md
All other source/tests/fixtures read-only; no native bridge, allocator, root, growth-law,
canonical CSV/PlantState or species-parser changes. These support the needed generic
sections/shared geometry already. If something really requires widening, explain it in
report and implement independent authorized work. Do not weaken source guards, change
resource caps, discard state, alter upstream projects or create environments/install.

## Scope and scientific choices

Implement explicit opt-in, species-local simple reproduction laws. Existing null and
law=unimplemented remain inactive and allocate nothing. Unknown/malformed active laws
refuse clearly. All species-specific numbers belong in that species reproduction section
beside its other traits; no hardcoded species, no separate scattered parameter tables.
The examples must label ALL new numbers as UNCALIBRATED FIXTURES, not field defaults.

1. Seeds: a simple STEP-inspired establishment-probability trial per ecological patch and
species per defined annual opportunity, then a bounded integer number of recruits (e.g.
uniform1..configured max on success). Require an explicit fixed physical ecological patch
size/geometry, independent of hydraulic cell resolution. No implicit full community per
fine hydraulic cell. Uniform within-patch positions using actual clipped boundary-patch
area. Explicit external-propagule assumption, no full seed bank/light/nutrient physics.
Seedlings have species-defined positive initial kg DRY ABOVEGROUND biomass; this is an
external establishment/seed-reserve AGB credit in the ledger, not newly created biomass
claimed to derive from current transpiration. No extra rain, ET or hidden water withdrawal.
Seed-only species and establishment onto a zero-live or zero-record domain must work.

2. Vegetative daughters: eligible LIVE parents, species-configurable minimum age/size,
probability, bounded daughter number, initial dry AGB and physical dispersal radius.
Draw nearby daughter positions uniformly in a disk (sqrt radial draw), explicitly finite
nonperiodic domain. Use bounded rejection attempts; never clip/wrap to force placement.
Each accepted daughter gets a NEW stable ID, same species, age0, current sampled GROUND
at establishment, E initialized explicitly. No plant-height-at-establishment field.
Support clonal-only grasses (seed pathway disabled), seed-only, both and inactive species.
The user selected a distinct ramet representation, an explicit adaptation from STEP's
same-parent clonal recovery. For initial transparent AGB accounting, fund daughter AGB
by subtracting exactly that AGB from its parent; bound per-parent donation fraction below1
and eligibility so parents stay LIVE with positive B. Debit E consistently (e.g. proportional
to parent E/B), initialize daughter E consistently with transferred E, and account exactly.
No hidden belowground pool, costless clonal biomass, floors, automatic parent revival or
unselected reproduction mortality. Failed placements cause no donation.

3. State & ledger: old total dry AGB + external seedling AGB = new retained total dry AGB
within justified FP64 tolerance; clonal transfer cancels internally. Report accepted and
rejected seed/clonal counts, donor/seed AGB and total closure in small fixed-size/per-species
diagnostics. Do not retain per-event history or a Python dictionary per plant. Existing
IDs, locations, ages, establishment anchors, statuses and dead/removed records persist;
only donor B/E can change. Parent IDs may be optional bounded diagnostic arrays, but no
unrequested schema rewrite or permanent lineage graph is required. Row order has no
meaning; keyed random eligibility/placement and ID assignment should yield the same
by-ID result under parent row permutation. Existing canonical IDs include zero, negative
values and both signed-int64 endpoints: NEVER impose positivity on existing plants.
Generate new IDs above max(existing IDs,0), no reuse, and fail before allocation on exhausted
int64 space only when births actually need IDs. No failure merely because an inactive or
zero-birth population contains max-int64. Stable ID order for assignment, fixed species
name/patch order for seeds. Preserve existing Phase2 RNG golden results; domain-separate
new seed/clonal streams from mortality, with seed/counters owned by session and restart.
Rejected calls consume no state/random opportunity. Invalid integer/bool/dtype/nonfinite,
device mismatch, malformed laws and budgets refuse before array work.

## Session population transaction / scheduling

Expose a documented explicit annual/coarse reproduction boundary API on IntereventSession,
with the actual CURRENT surface, explicit ecological patch configuration and global finite
work/population guards. Define the precise calendar opportunity and ordering in docs/tests:
complete prior growth, apply post-event damage, apply due annual mortality, then recruit;
new plants request water only on the next interevent step. No reproduction during active
wind/rain/recession, or with pending growth, stale/replayed opportunity, unresolved annual
mortality or pending transport diagnostics. Do not run catch-up growth or silently drop
missed opportunities, and don't charge multiple annual chances by calling repeatedly.
A first post-transport boundary may be later than Jan1: explain/refuse unsupported multi-
year gaps, preserve partial annual metrics and don't relabel old annual sums as a new year.
Bareland/inactive-mortality cases need a clear usable annual opportunity too.

Generalize the Phase2 _rebind through ONE shared staged population path, handling both
status/B changes and appended births. Reuse canonical PlantState, Domain.sample (containing
cell), common allometry/canopy and root builder. Do not duplicate MAPLE/SYRUP canopy physics,
reinitalize all vegetation through the Phase0 generator, reposition old plants or reset
native water/time/season/growth-grid. Preserve existing per-individual root overrides for
survivors; daughters default to species roots unless an explicit supported policy says
otherwise. Initialize stress/annual accumulators only for new plants. A staged failure in
root/geometry/admission must leave old base, B/E, native state, clock/grid, RNG/counters,
roots, root overrides, annual sums, replay guards and receipts EXACTLY unchanged.

Important newborn issue: existing Phase2 background mortality uses global period_days.
A new plant must NOT accrue background hazard for time before birth. Cap exposure by
actual time alive (at least min(period_days,age_days)); retain existing age-hazard behavior
and document/test this deliberate newborn correction. When plants reappear after several
empty years, don't inherit empty-period stress/hazards; existing parents' real partial-year
metrics must never be reset. Add no extra daily readbacks beyond existing Phase1 behavior.

Use the same canonical plant CSV throughout initial/current/final/snapshots. New restart
state (reproduction counter, last opportunity, next ID if persisted, patch configuration,
any required policy) MUST survive full checkpoints, strict types/source/native/root guards
and content hashes. Bump complete checkpoint to explicit version3 if required fields/layout
change; NO silent/defaulted migration of version1 or2, even allow_source_change=True.
Canonical plant CSV remains version1. Document this compatibility boundary clearly.

## CPU/GPU and bounded memory

CPU compiled kernels (Numba) or vectorized NumPy reference; CUDA CuPy/compiled eligibility
where suitable. Explicit host population placement/geometry between storms is a legitimate
initial hybrid, with all transfers/readbacks counted and documented. Do not pretend it is
fully GPU resident. No Python per-plant hot loops for eligibility, stochastic decisions or
placement, no dense plants-by-grid matrices, maximum possible plant populations, whole
history or full-domain state copies for each daughter. At most loops over small species/
patch groups around compiled/vectorized bounded candidate work. Dynamic actual candidate
counts and prefix sums/compact SoA, not objects/linked lists per plant.

Every allocating public path preflights metadata/budgets BEFORE population/patch/candidate
arrays, copies, validation scratch or device uploads. Explicit strict finite max-new-
plants, max-total-records, max-attempts and global max_bytes/max_edges; limits fail clearly,
never truncate state/counts or raise caps. Count old/new state, all working arrays, native
resident water/params, root maps, host/device copies and scratch coexistence in one ledger
including caller reserve. Birth-heavy and mixed live/dead populations must remain bounded
by actual P+edges+candidate work; no dead-record deletion to fit. No history-growing RNG
lists. Reproduction inactive adds zero population-sized arrays and preserves prior behavior.

Overlap: ecological rules not yet calibrated. Initial simple reproduction may explicitly
allow canopy overlap (document this, including parent/daughter overlap). Do NOT silently
claim nonoverlap or invent expensive all-pairs spacing. Initial Phase0 generator continues
to support both overlap/exclusion independently. Surface remapping, seed banks and field
recruitment calibration remain future work.

## Independent acceptance by Codex

- All existing Phase0/1/2 CPU tests preserved; relevant actual CUDA paths tested on one
  selected idle GPU with unchanged cap. No source-guard/test weakening or unrelated edits.
- Seed-only / clonal-only / both / inactive; species-dependent law validation, physical
  patch invariance across hydraulic resolution, zero live and zero total records.
- Own unique IDs, age0, actual ground sampling, unchanged parents' anchors/locations,
  dry-AGB/E transfer ledger, no phantom water, live positive donors and no donation on reject.
- Exact seeded by-ID behavior under row reorder, restart and retry; canonical signed IDs,
  int64 exhaustion guard, no opportunity replay or wrong ordering, deterministic dead retention.
- Shared live-row/canopy/root rebuild, preserved surviving overrides, new defaults; real
  native SYRUP water unchanged by reproduction, then delivered uptake includes new plants.
- Correct newborn annual hazard exposure, no inherited empty-period stress, partial-year
  accumulators preserved; inactive catalogs need no RNG seed or new persistent arrays.
- Complete mid-growth/checkpoint reproduction state exact, old-format refusal without
  migration, corruption/wrong types/root/source/native guards retained.
- Admission/fault-injection before publish leaves ALL accepted state unchanged; valid retry
  counts exactly once. Linear actual population/edge memory and explicit host/device cost.

Provide a bounded compiled CPU example using a complete per-species catalog with the
same shared CSV/allometry/root interfaces; optionally backend CLI for selected CUDA.
Document exact API/units/calendars/ledgers/limits and downstream MAPLE/SYRUP developer
instructions (do not edit either model). Save incrementally; leave uncommitted. Report
all checks NOT RUN, exact new APIs, reuse/adapters, scientific deviations and open limits.
