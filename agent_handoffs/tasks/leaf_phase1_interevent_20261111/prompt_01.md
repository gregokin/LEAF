# LEAF Phase 1 — dated native-water/growth session and complete restart

You are Claude, sole implementation writer in THIS isolated candidate. User
authorized automatic relevant sharing with Anthropic, correction/review loops,
and commit/push of each whole accepted phase before proceeding to the next.
Codex scopes, executes, independently reviews, integrates and publishes. This
task is the remaining Phase 1 integration, not permission to start Phases2/3.

READ only this prompt and candidate/**; SOURCE_INDEX.txt lists every existing
file, so do not guess/read missing paths or directories. Read/Edit/Write only;
no execution/Bash/agents/network/settings/installs/Git/outside/archive reads.
WRITES exactly leaf/interevent.py, leaf/checkpoint.py, tests/test_interevent.py,
docs/phase1-interevent.md, examples/phase1/run_interevent.py, README.md, report.md.
All other files READ-ONLY, including the allocator, roots, forcing, traits,
growth, state and native bridge. Native reference files are exact current SYRUP
source for API inspection only, not code to vendor/fork. No actual saved terrain,
weather or simulation datasets are supplied. Tests are NOT RUN by you.

## Supplied components and verified context

The candidate composes the complete package with current isolated revisions:
- uptake.RootMap/propose/accept: fixed general CSR roots, local proportional
  sharing, actual accepted feedback, no compensation; typed arrays, budgets and
  device affinity. NumPy/Numba/CUDA numerical tests independently pass.
- roots.build_root_map: common canopy allometry, square finite grid, uniform
  positive-overlap horizontal law, species layer fractions and ID-keyed
  individual overrides. Admission covers builder work. Caller must reserve
  other live state and OLD root map separately from the new builder's budget.
- forcing.TimestepRecord: fixed-offset actual Gregorian start/end, temperature_c,
  optional shared PET_mm/day, interevent gate; exact microsecond Fractions.
  MASY rational submicrosecond instants must be preserved OR explicitly refused
  before mutation. No fabricated January1/noleap clock or catch-up over events.
- growth: revised NEW law, canonical backends numpy/numba/cuda. Complete species
  records own growth.traits and water_use.traits. New names include
  biomass_support_kg_year_per_m3, extra_growth_rate_per_year, and support/deficit
  outputs *_kg_year. Read its ACTUAL signatures and memory estimators. Public
  response/integral helpers require max_bytes. Rate range0..1/year is the qualified
  weekly Euler envelope; historical study rate295 remains inactive, not rebounded.
- syrup_water.withdraw_plants: real native operator exactly once, no water
  subtraction in LEAF, actual returned physical debit verified, pure and time
  preserving. Its native_provenance(verify=True) binds sources at init, not daily.
  The bridge's old one test has a brittle error-string expectation; another
  isolated writer is correcting TEST/REPORT ONLY. Do not change that test here.

Codex independently ran183 CPU tests on the uptake/root/forcing tree (10GPU
skips),17 actual CUDA root tests and21 actual CUDA allocator tests. Growth's
revised tree passes53 CPU tests (2GPU skips), including finite-ledger refusal;
prior version49 tests passed actual CUDA. The bridge passes independent native
CPU cases and now rejects missing physical debit. Its CUDA live-byte probes fit
their declared bounds. These are Codex's evidence, not checks you executed.

## Required public session

Implement a small explicit session (e.g. IntereventSession), initialized with
PlantState + SpeciesCatalog, actual prepared native column/transpiration operator
and native soil state, actual fixed-offset start instant, backend, growth cadence
(positive <=7days), root builder/individual layer overrides, max_edges/max_bytes
and caller reserve. Require matching domains/layers/area/device/catalog/IDs;
resolve establishment before use. Do not silently switch native backend, alter
parameters, generate weather or adopt historical benchmark approximations.

One accepted boundary holds the native SYRUP water state, authoritative living
plant B/E, pending requested/delivered water and species activity/temperature
integrals, current root map, exact calendar cursor, next growth boundary and last
receipt. Stable IDs/species/XY/establishment-ground/age/status remain recoverable
for ALL plants, including dead/removed records. Working arrays/root edges may
use only live plants to avoid wasting memory on dead-root communities; document
the row mapping explicitly. No daily host population readback. Host PlantState
is a synchronized export/geometry snapshot, not a competing biomass authority.

Process one explicit interval with record.start==cursor; reject gaps, overlaps,
replay, wrong offsets/kind/forcing/shape/device before publication. It is fine to
require the caller to split at midnight, next weekly boundary and new year, and
refuse unsplit records clearly before mutation; expose the needed boundaries
and reuse forcing split helpers. Whole-day and partial-day records must work.
An interval receipt ID and exact cursor prevent repeated accounting; retain only
current/last metadata, no unbounded list of daily arrays or trajectory history.

### Actual native hydrology followed by plant T

Compose the REAL native ss.step with EXPLICIT caller/SYRUP-provided
added_water_m[L,N] and potential_evaporation_m_s[N] for this already-split interval.
The caller owns how E is partitioned/derived from shared weather. Do not compute
PET or invent an E share from canopies in LEAF. Current upstream requires E's
share after limiting combined potential soil loss, then separate LEAF T; that
scientific partition belongs to SYRUP. This narrow driver accepts native forcing
prepared for that contract; it does not claim full persistent storm/E adoption.
Synthetic vegetation qualification may explicitly supply E0. No hidden default
rainfall/infiltration calculation or second soil-water inventory.

Stage native non-T step from the accepted native state, compute D on existing
period geometry using actual month/T/shared PET and active duration, then call
the actual syrup_water bridge once on the staged native state. LEAF computes
requests/competition; native SYRUP accepts/debits. Gather ACTUAL delivery and
accumulate D/U on the same backend. Compute activity and temperature integrals
using actual365/366year lengths and active interevent time only.

If a growth boundary is due, apply the existing growth.grow to accumulated
requested/actual water: requested-water support makes the full planned gain,
actual/requested multiplies it ONCE. Keep actual support/maintenance diagnostics
needed for Phase2. Rebuild roots/canopy from the shared catalog implementation,
preserving species/individual overrides. Never redirect unmet roots to wet cells.
Copy retained B/E out of the grow result's shared12xP output buffer if needed so
views do not silently retain all diagnostics. Do not retain daily large results.

Publish native water, biological candidates, maps, integrals, clock/receipt with
ONE accepted-boundary replacement ONLY after ALL stages/closures/admissions pass.
Any native, gather, growth, root or checkpoint validation failure leaves the
previous accepted boundary intact. A rejected interval can be retried without
double debit, age advance or lost pending demand. Results may include compact
scalar diagnostics; large daily debug arrays are not retained by the session.

### Transport gates and ages

Provide an explicit flush_growth at the CURRENT cursor to finish a partial
interevent period before a transport event, so native consumers can export the
current canopy. An advance_transport(kind,start,end,...) must refuse nonempty
pending growth (caller flushes first), advance calendar/live ages and skip any
weekly boundaries crossed, with NO T/growth/E/catch-up for transport duration.
For wind, native water can be unchanged with only its native-time metadata
advanced consistently. A rain/recession result must be explicitly supplied from
the external native owner, with matching column/backend/end time; LEAF must not
pretend to run storm physics. Do not silently replace physical water.

Year-end seasonal turnover inside transport is deferred to the next OPEN
interevent boundary and recorded in restart; it must not accrue event duration
as growth. A closed biological gate likewise produces no plant T/growth; make
its native hydrology behavior explicit and preserve mutual exclusion. Ages
advance exactly once for every accepted elapsed interval, including events;
snapshots do not advance the clock. No mortality/disturbance laws here.

Expose a snapshot/export that returns canonical PlantState at the current cursor
and can use existing SnapshotWriter/shared canopy_view. Use the ONE allometry
implementation; arbitrary power-law diameters/heights must work, no old B∝area
restriction. Surface remapping remains deferred. Custom nonseparable external
root maps may be refused unless an explicit compatible rebuild policy is given;
never silently replace one with default species-separable roots.

## Complete restart and bounded memory

Add save/load restart (leaf.checkpoint may separate I/O). Canonical plants.csv
remains the same human-readable initial/snapshot/final format. A restart ALSO
stores native water/time, exact cursor/next boundary/deferred season, pendingD/U
and species integrals, exact current root CSR and overrides, IDs/row mapping,
species/source/config hashes, backend and last receipt. Restore a full accepted
boundary, not merely CSV. Use typed arrays (allow_pickle=False), content hashes
and strict schema/shape/finite/time/source validation. Reject corruption,
inconsistent parameters/roots/water/boundaries BEFORE adoption. No silent source,
catalog/backend migration. Publication of checkpoint files must be atomic; do
not expose a half-written checkpoint or overwrite existing artifacts by default.

Every allocating stage must pre-admit a truthful GLOBAL ledger with other live
state/caller reserve and old/new coexistence. Reuse native/allocator/growth
estimates, conservatively sum stages if needed and label double-counting. Account
for canonical host population, GPU B/E/accumulators, native original+nonT+T
candidates/results, candidate integrals, root graphs and geometry/transfer
scratch, retained views and restart/CSV serialization buffers. Component calls
get the actual remaining budget, not a fresh unrestricted cap. Native available
bytes checks and root-map own cap remain active. Count actual root edges after
its bounded count pass; do not reserve MAXIMUM POSSIBLE populations/trajectories,
drop individuals or raise limits. If a rebuild cannot fit, refuse transaction
without losing its old boundary. Document interpreter/JIT/context/pool exclusions.
Any weekly GPU↔host transfer for current shared geometry is explicit and counted;
do not claim the current host allometry/root builder is fully GPU resident.

## Verification artifacts to write, not execute

Write meaningful unittest cases against ACTUAL native APIs: daily and partial
intervals, leap day/new year, weighted water/growth law applied once, species and
individual root profiles, native water/AGB closure, arbitrary common allometry,
weekly/root changes, explicit flush, no event catch-up, transport ages and stale
inputs/replay refusal, closed gate, deferred seasonal turnover. Inject failures
at native/T/grow/root stages and verify the entire accepted boundary remains
unchanged then retry succeeds once. Test too-small GLOBAL admission before work.
Save/restore mid-week pending state and compare continuous versus split run;
test corrupt arrays, catalog/source mismatches and failure atomicity. CPU/CUDA
parity when available, no other-device allocations. No fake water authority.

Example script uses the complete examples/phase1/species.json with actual LEAF
population loading/generation and synthetic explicit native soil/weather forcing.
Mark all examples UNCALIBRATED and require an explicit backend/resource budget;
no packages/env setup, no full transport scenario. README/doc explain executable
use, species parameter ownership, units, coupled-restart scope and what remains
upstream/unqualified. No field calibration or full MASY/MAPLE storm/wind claim.

Save incremental code, then finish report.md: exact files, API, assumptions,
memory/transfer limits, all tests NOT RUN and honest tool errors. Leave
uncommitted. Codex will freeze/review/run independent CPU/CUDA/closure/restart/
memory checks and dispatch corrections. No user permission is needed for routine
defects; flag only material science not resolved by these explicit requirements.
