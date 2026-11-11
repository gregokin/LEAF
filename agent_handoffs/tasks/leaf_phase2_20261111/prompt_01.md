# LEAF Phase 2 — disturbance, biomass loss and persistent mortality state

Task leaf_phase2_20261111, baseline f0d0b56854a576b35ee38bce4a8eab6dd623baf6,
exact session 711c5973-64c3-45f8-af8a-379ba0e357a7. Phase1 is independently
accepted, committed AND pushed. User authorizes automatic Codex–Claude work,
relevant source sharing with Anthropic, and separate whole-phase commit/push by
Codex after acceptance. Implement this phase only; Phase3 follows separately.

## Roles, tools, source and writes

Claude is the SOLE implementation writer in this isolated candidate. Codex scopes,
reviews, executes checks, integrates, commits/pushes. Source-only ordinary
Read/Edit/Write tools; NO execution/Bash/agents/settings/Git/installs/network tools.
Read only THIS prompt and candidate/** (SOURCE_INDEX.txt lists supplied files).
No outside files, raw datasets, upstream source trees, terrain/weather, credentials
or other tasks. reference/method_notes.md supplies already-inspected scientific
precedents. No exact ECOTONE/STEP reproduction claim; this is an explicit LEAF
aboveground-only adaptation. Tests you write are NOT RUN; report honestly.

Allowed writes EXACTLY:
leaf/disturbance.py, leaf/demography.py, leaf/interevent.py, leaf/checkpoint.py,
tests/test_disturbance.py, tests/test_demography.py, tests/test_interevent.py,
examples/phase2/species.json, examples/phase2/run_disturbance.py,
docs/phase2-disturbance.md, README.md, report.md.
All other source/tests/fixtures read-only. Do not alter canonical state/traits
modules, root builder, allocator, growth laws, native bridge or source guards.
No upstream changes, resource-limit changes or new environments/packages.

## User's Phase2 objective

Species-specific growth/mortality responses to burial, erosion, abrasion, root
exposure and dislodgement after transport. MAPLE wind can do all; SYRUP water
erodes/deposits. Ground elevation at establishment is the fixed reference, NOT
plant height at establishment. Age, stochastic processes and repeated slow
growth / maintenance shortage can also kill or reduce biomass. Mean-life metadata
is NOT a hard maximum. No soil/surface-layer remapping, fire or wind physics.

User approved configurable species threshold/response curves with explicit
uncalibrated fixtures. Every species-specific coefficient belongs in that
species's catalog record (mortality/disturbance sections beside allometry, roots,
growth etc). No species tables scattered in modules, no assumed field defaults.
Recognize law='unimplemented' as explicitly inactive for backward compatibility;
unknown/malformed active laws must refuse, not silently disable.

## Required numerical / scientific behavior

1. Species-dependent mechanical law, with explicit units/thresholds and bounded
   responses. Use existing immutable stable-ID SurfaceEventDiagnostics and shared
   allometry. Take actual current surface information to evaluate burial/exposure
   relative to establishment ground. Distinguish signed NET ground change from
   gross erosion/deposition and abrasion dose. Flux is NOT abrasion dose: no
   invented conversion; active dose laws require exact declared unit matching.
   Positive event increments can cause biomass injury; absolute post-event burial
   or root exposure can cross explicit lethal/dislodgement thresholds. Do not
   reapply accumulated historical burial as new damage each event. Preserve X/Y,
   anchors, stable IDs and ages. Define simultaneous-response composition so total
   loss stays <= existing dry AGB, and scale E consistently with total biomass.
2. Explicit opt-in age/stochastic and persistent low-growth/maintenance-stress
   laws. Prefer simple documented laws: configurable age hazard (e.g. Weibull)
   plus background hazard; annual low actual/planned growth counter, healthy year
   decrement, optional repeated-stress biomass loss/mortality. First-year/zero-
   potential/dormant cases must be defined so dormancy/maturity alone cannot be
   mistaken for water stress. Keep actual support vs required support diagnostics
   distinct from biomass-production yield. No new maintenance-water fraction.
   Use the current growth outputs/actual native feedback; do not double-apply S
   or turn unmet demand into extra withdrawal. Scientific constants from references
   are precedents only, NOT silently adopted defaults.
3. State remains bounded by actual populations. Persistent per-plant stress and
   annual accumulators, RNG seed/key/counters and decision/replay guards belong in
   accepted ecological state and complete checkpoints. Biological decisions must
   be reproducible after restart and independent of plant row ordering; use explicit
   ID/keyed deterministic RNG or equally rigorous owned-state design. Rejected
   calls consume neither random state nor biomass/water. No global RNG mutation,
   whole-history/event-ID lists or max-population allocations. The canonical CSV
   remains the single plant-record format; auxiliary pending/annual decision state
   may live in the existing full-checkpoint arrays, with that limitation explicit.
4. Biomass ledger: old dry AGB = new retained dry AGB + removed dry AGB (within
   justified FP64 bound), with per-cause diagnostics. Dead-standing plants retain
   their remaining B/E; removal/dislodgement exports remaining B/E and zeros them.
   Dead/removed ages freeze, records remain, roots of non-live plants stop taking
   water. Do not count death alone as unexplained loss or create a belowground/
   carbon pool that the user did not request. A dead canopy can still matter to
   transport through the shared canopy view.

## Integration and transaction boundary

Keep one authoritative water state and B/E. Extend IntereventSession with explicit
documented biological boundary APIs, retaining all existing Phase1 behavior when
Phase2 is inactive. Post-transport disturbance runs only AFTER the event, never
growth/ET during it. Require pending growth flushed before population/status
changes. Bind event time/source/IDs and reject duplicate/stale decisions; no
silent replay, gap or double injury. Annual stress decisions can be explicit
scheduler calls at a defined boundary; document ordering and ensure per-plant
actual/planned growth and support metrics are available without daily population
readback or retaining trajectories. Deferred year ends inside transport must not
accrue growth or reset/discard a real pending year incorrectly.

Implement a transactional population rebind/rebuild method as needed: changed
status/biomass yields new live-row mapping, common geometry and roots BEFORE
publication; accepted old state/native time/water/clock/grid, pending diagnostics,
RNG and counters remain intact on ANY failure. Reinitialization must preserve
the established growth grid, deferred season and accumulated calendar age; no
catch-up, double-aging or resetting of water. Explicit host geometry / canonical
snapshots at these infrequent boundaries are acceptable and must be counted.
No alternative canopy representation or independent hydrology calculations.

Every allocating public path validates scalar budgets/types and admits BEFORE
population-sized allocation/copy/upload. Count old/new session, map, native and
demography coexistence, host/device copies, serialization/restore scratch. One
global max_bytes, caller reserve preserved, no raised limits or discarded state.
Prefer compiled Numba CPU/CUDA population kernels (NumPy reference allowed),
species tables and fixed/small diagnostic packets. No Python per-plant hot loops
or dense plants-by-cells/layers arrays. Explicit device binding, strict input
validation and documented memory estimates matching actual owned allocations.

## Acceptance to be run independently by Codex

- Existing full Phase0/1 suite remains passing, source guards unchanged.
- Species contrasts for burial/exposure/abrasion; exact units; net-vs-gross cases;
  zero injury, partial loss, dead-standing and dislodgement; anchors never moved;
  dose/flux distinguished and repeats/stale/wrong times/IDs refused.
- Dry-AGB/E/status/age ledgers; dead plants request no water on next native day;
  common canopy export reflects surviving/standing/removed plants.
- Active age/background/stress laws: probability0/1, dormant and first-year cases,
  repeated bad years, recovery; no hidden field constants or life-mean cutoff.
- Real native water unchanged by every biological operation; actual STEP growth
  metrics accumulated once; split-run checkpoint/RNG/counter equivalence.
- Row-permutation / species-order invariance of keyed randomness, CPU/Numba/CUDA
  parity (CUDA tests skip without device); clear explicit backend/device failures.
- Budget refusal before allocation/rebuild and fault injection leave the entire
  accepted state unchanged, valid retry accounts once. Large-population memory
  remains linear in plants+edges, no history/max-capacity expansion.

Provide small UNCALIBRATED fixtures in one complete per-species catalog and a
bounded example using existing APIs, with clear invocation/units and limitations.
Update README/docs/report. Save incrementally, leave uncommitted; report exact
API and any deviations/errors, and ALL author checks NOT RUN. Codex independently
reviews/tests/corrects before whole-phase publication.
