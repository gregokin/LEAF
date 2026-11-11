# LEAF Phase 1 — bounded native water bridge

You are Claude, sole writer in THIS isolated candidate. Codex scopes, reviews,
executes checks and publishes accepted phases. The user explicitly authorized
automatic collaboration/relevant source sharing with Anthropic for Phases1–3.
This is an independent Phase1 subtask, not phase completion. Another isolated
writer is correcting the allocator; you may not touch that scope or its controls.

## Scope

READ only this prompt and candidate/**. Ordinary Read/Edit/Write tools only.
No execution/Bash/agents/network/settings/installs/Git or upstream/archive access.
WRITES exactly: `leaf/syrup_water.py`, `tests/test_syrup_water.py`,
`docs/phase1-native-water.md`, `report.md`. Everything else read-only.

Candidate contains frozen Phase1-partA modules, all existing LEAF modules/tests,
examples/phase0/species.json and two compiled benchmarks. `native_reference/`
contains exact CURRENT SYRUP source copies for API inspection ONLY: subsurface.py
and subsurface_transpiration.py. Do not vendor or fork those numerical routines.
No actual terrain/forcing/simulation datasets are supplied or readable.
No growth, temperature law, weather/PET implementation, mortality, recruitment,
surface remapping or upstream source changes in this subtask.

## Required behavior

Implement a small PURE bridge, e.g.

`withdraw_plants(root_map, demand_m3, operator, soil_state, *, max_bytes, extra_reserve_bytes=0)`

1. Validate shape, namespace, dtype/device compatibility, explicit budget and
   binding of the prepared native operator to its native column/state BEFORE
   allocating or calling native withdrawal. Root resources use
   `layer*n_cells+cell`; native water is FP64 `[L,N]` depth in metres, actual
   column cell area is m2. CPU NumPy/Numba are compatible NumPy namespaces;
   CUDA requires the exact same CuPy device throughout. No implicit fallback,
   host/device population readback, device switch or dense P*L*N arrays.
2. Expose/construct a read-only availability snapshot from the native inventory
   and the prepared operator's PUBLIC residual/mask fields. At current native
   contract, `A=area*max(water_m-residual_xp[:,None],0)` on active cells, zero
   elsewhere. This is a projection of SYRUP-owned state/floor, NOT a separate
   water store or configurable LEAF wilting floor. Document the pinned API
   dependency and fail clearly for unsupported objects/contracts. Do not create
   another hydrology process. Validate finite/native-valid content before using
   it, without a second dummy native withdrawal. Use bounded NumPy reference /
   compiled Numba CPU / CuPy kernel projection as appropriate; lazy imports.
3. Call `leaf.uptake.propose` with that offered VOLUME and plant D. Its rule is
   Q=sum(D*w), proposal=min(A,Q), fixed normalized roots, no compensation or
   species priority. Do NOT replace it with the old benchmark allocator.
4. Convert proposed volume to native depth by dividing by native cell area
   exactly once. Call ACTUAL
   `maple_syrup.subsurface_transpiration.withdraw_transpiration(operator,state,request_depth,...)`
   exactly once for the invocation. That native API alone creates the new soil
   inventory; it is pure, preserves native time, and returns accepted depth,
   unmet, scale and a physical budget. Do not implement subtraction yourself.
5. Convert its actual accepted depth to volume once and call
   `leaf.uptake.accept(plan,accepted_volume)`. Plants gather from actual accepted
   water, never proposed water or the least-satisfied layer. Sum delivered must
   equal the native actual volume within the native/allocator FP64 policy.
   Preserve actual accepted feedback including honest area-conversion roundoff;
   no balancing flux, clipping away water or doubled native acceptance.
6. Return a clearly named result with the fresh native state/native result,
   plant plan/feedback, conservation diagnostics and a truthful memory report.
   Input native state and plant/root arrays are unchanged even if native,
   gather, validation or memory admission fails. This function does NOT publish
   either inventory or plant state and does not advance a scheduler clock. The
   later dated driver must atomically publish native+biological state and bind
   an exactly-once interval receipt; do not claim this pure function supplies a
   complete restart/idempotency/event scheduler by itself.

## Actual upstream API and provenance

SYRUP HEAD c76c603dde4e8f79ec48b18d954a386580d8bc89,
feature/subsurface-hydrology; actual module VERSION0.2.0. Native source hashes:
subsurface.py 5e77f14df8af4e473e8a22278700b519ccb2888c722883a9f7da508abdf14ae9;
subsurface_transpiration.py 3b9191872522755924271bbaba4ae2bac290e1cad4ecffebc2d2735c4a0a9fb9.
Read the supplied references selectively for exact fields and signatures. The
operator has column/xp/backend/n_layers/n_cells/residual_xp/saturation_xp and
check_device(). Native storage water_m is a depth in metres; active mask is
column.active_u8. Native `withdraw_transpiration` preserves t_s. Preparing an
operator is the caller's separate operation; require an existing prepared one
to avoid hidden compile/setup costs. No transport/model scenario is assigned.

Codex independently connected PartA snapshot to this actual API on NumPy and
Numba: the user's D100, weights .5/.5, availability25/200 gives U75; unequal-root
competition gives62.5/12.5; dry shallow/deep profiles give20/80; inactive weights
remain unmet; areas .0625,1,2.3 preserve water units. Inputs and native time stay
unchanged. These were actual native numerical fixtures, not a completed driver.

## Allocator under independent correction (do not fix its files here)

Candidate has partA snapshot1. The other writer is fixing: CSR offset overflow,
nonfinite closure, memory estimates/pre-admission, CUDA affinity, accepted excess
handling and canonical multi-file metadata. Public core API stays
`RootMap/propose/accept`, plan.proposed_grid(), feedback.delivered/unmet/
satisfaction/accepted/closure. A corrected allocator will be composed for final
tests. Do not depend on its currently incorrect per-operation scratch estimate
being sufficient without explicit ledger separation/reserves. Native shape and
volume math must not be weakened to pass the old defects. In particular current
accept clips ulp-level excess, which the other writer is removing: a correct
new test of EXACT native feedback may fail that old file until composition.

## Memory and performance

Preflight scalar dimensions and metadata before allocation, transfer, source
conversion or native calls. Bound all simultaneously live bridge work by the
declared max_bytes: root map, original native state, native operator/column
resident arrays and native withdrawal outputs/rows/scratch, supplied demand,
availability/proposed-depth/accepted-volume buffers, allocator plan/feedback and
their scratch, plus caller reserve and native column reserve without silent
omissions or double-counting. Reuse authoritative native estimate functions and
component ledgers where possible. Conservatively reserving unused stage space is
fine if labelled; claiming a lower peak than live arrays is not. Separate host /
device accounting and state precisely what external arrays, caches/context/JIT /
Python bookkeeping are excluded. Respect native available_bytes_fn before work
and its own admission on the withdrawal. Do not raise limits, discard plants,
drop diagnostics/state needed for correctness or allocate histories to pass.
Do not reread source hashes or transfer population vectors each day; provenance
binding belongs to initialization/run evidence. Batch GPU diagnostic reductions
into small packets. Keep existing NumPy-only imports independent of native,
Numba and CuPy packages until the new bridge is explicitly used.

## Independent acceptance expected

Write meaningful unittest cases against the REAL imported native API (skip only
if it is genuinely unavailable, never replace it with a fake water authority):
user75% case and leftover deep water; unequal root profiles/competition; inactive
cells/no renormalization; several non-unit cell areas; accepted shortfall
feedback (a narrow spy may force reduced native acceptance while still calling
the real operator once); finite/nonnegative and unit closure; unchanged input
and native time; rejecting wrong dimensions/area/device/backends before native
call; budget refusal before large allocation/native execution and without
changing the old state; native or gather failure after staging still preserves
inputs. Actual CUDA parity when available, with no tests allocating on another
device just to check mismatch guards. Core import isolation test is relevant.

Document the public API/units, ownership and publication boundary, exact pinned
native availability projection, memory scope and remaining dated-growth/restart
work. In report.md list changed files and tests NOT RUN: you cannot execute.
State any ordinary Read/Edit/Write errors exactly; do not call them auth/rate
failures. Leave code uncommitted. Codex will independently inspect scope/diffs,
compose corrected allocator, execute real native CPU/CUDA checks and correct.
