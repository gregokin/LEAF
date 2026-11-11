# Phase 2 revision 2 — verified guard, admission and fixture corrections

Resume EXACT session711c5973-64c3-45f8-af8a-379ba0e357a7; preserve first draft.
Same12-file write scope / user automatic source-sharing authority as prompt1.
Sole implementation writer, ordinary Read/Edit/Write only. Read this prompt and
candidate/**, no outside/archives/upstreams/settings/agents/execution/Git/installs.
No new physics or field defaults; all checks you author remain NOT RUN. Codex
executes, reviews and publishes whole phases. Save incrementally.

## Independent results on snapshot_01 (all writer code frozen)

First writer completed/reaped, structured success, scope violations/toolerrors0.
Full CPU442 tests:24 errors,23CUDA skips; ALL errors are nominal wind fixtures
activating grass abrasion but omitting required abrasion_dose. Earlier Phase1
tests pass. Independent REAL native numba AND actual CUDA fixtures PASS burial
death/dislodgement, no dead-root uptake, frozen dead ages, water unchanged by bio,
surviving and zero-live populations, stale/replay refusal and exact restart.
Production bytes in interim_review_01 and final snapshot_01 are identical.

Synthetic reviewer files (read only; do not follow imports outside candidate):
review_reference/probe_entrypoints_02.py, profile_after_damage_02.py,
verify_native_biology_02.py. They are supplied before this revision in a NEW
baseline_revision_02.json; these copies are Codex context, NOT your changes.

## Required corrections

1. **Nominal test dose fixtures.** Fix missing dose in the new nominal tests in
   test_disturbance.py and test_interevent.py: a wind event under an applicable
   abrasion law must explicitly supply zero dose in the correct unit if testing
   burial/erosion alone. Keep dedicated missing-dose / flux-is-not-dose tests.
   Do NOT weaken production unit/dose guards to make nominal fixtures pass.

2. **Validate scientific array content BEFORE numerical kernels.** decide only
   validates shapes/dtypes. Independent numpy AND numba probes accept NaN required
   support, negative actual support, negative stress counters and actual_growth2
   with planned_growth1. Negative species indices enter the kernel and fail later
   in bincount (or can read the wrong species). Validate finite nonnegative annual
   metrics/ages, valid species bounds, unique positive IDs, counters within their
   species caps, actual growth <= planned growth within justified FP64 tolerance,
   and input relations BEFORE any kernel/gather. Actual support may exceed required
   support (surplus is real); do not impose that false bound. Use backend reductions
   / compiled checks and small packets, no P readback or Python P loops. CUDA and
   numba must fail as clearly as numpy; no unsafe out-of-bounds kernel access.
   Do not silently clamp/repair invalid supplied data.

3. **ID lookup creates unbudgeted PERSISTENT arrays, and happens before admission.**
   apply_surface_event calls base.indices_of before event_evaluate preflight. This
   populates base._order/_sorted_ids (16 bytes per TOTAL plant), even for a zero-
   injury event or a later rejected call; _persistent counts neither. Actual GPU
   probe with100000 retained records,50000 live,4096cells,2layers, no-injury event
   then an ordinary native day: observed simultaneous array-data LOWER BOUND
   19725656B > declared19237416B (confirmed488240B). ID cache alone1600000B.
   This is not pool/context/Python metadata. With all100000 live the extra cache
   narrowly fits a different conservative stage margin; that is NOT a general fix.
   Use admitted local lookup scratch without mutating the accepted base, or admit
   and count actual persistent indexes in EVERY later ledger/restart/rebind path.
   Preflight before lookup/surface-validation scratch/population copies/uploads;
   failure must preserve accepted scientific state and avoid hidden cached state.
   Re-run valid event, no-injury, mixed dead/live and failure/retry paths.

4. **Pure entry-point admission / table compilation.** respond_state with two
   valid plants, an empty event, max_bytes1 currently RETURNS a copied population;
   the real PlantState.copy spy sees [2]. It also resolves IDs/compiles before
   checking any global population budget. Admit original+candidate population,
   validation/index/update scratch and owned table/device copies BEFORE work.
   compile_species in BOTH new modules currently allocates/uploads species tables
   before a caller can impose a budget; session checks initialization_phase2 only
   AFTER both compilers return. Add explicit admission/budget support and preflight
   known S-sized host/device/scratch costs before allocation/upload. Preserve the
   caller's max_bytes/reserve, no hidden limit increase. Audit the new P-sized
   readback accessors / helper allocation scopes too: either use session admission
   or clearly bind them to an already admitted caller stage. Document table and
   input resident costs; do not exclude owned buffers by renaming them scratch.

5. **Legal full dieback is inconsistent.** The stress parser accepts
   biomass_loss_fraction1 (docs[0,1]); apply_annual_decision then raises 'underflow'
   because B_new0 is invalid for LIVE. Define this explicitly: either full loss
   causes a documented death/removal with correct ledger/status, or require <1
   when configuring an injury law and direct complete death to mortality law.
   No hidden biomass floor, no discarded mass, and no accepting an unexecutable
   coefficient. Add endpoint tests and accurate docs.

6. **Never overwrite an outstanding annual decision.** Open growth is guarded,
   but _pause still accepts further transports and can overwrite annual_due with
   end.year-1 while retaining older annual metrics. Preserve the original year /
   observation period. Require outstanding decisions before further calendar
   advancement where necessary; refuse unsupported multi-year transport crossings
   with live active demography rather than relabel/discard a year or invent years
   of stress during transport. Document the bounded policy, preserve no E/T/growth
   during events, and add exact restart / rejection / valid-order regressions.

## Preserve successful behavior

Keep current dry-AGB and native-water ownership, multiplicative event injury,
fixed anchors, event-only increments versus absolute lethal depths, species
co-location, annual stress structure, keyed RNG, canonical CSV and common canopy.
Do not rewrite source guards/native bridge/allocator/roots/growth/state/traits.
No lowered resource caps, deletion of state or weakened checks. Biomechanics of
standing dead tissue remain an explicitly documented limitation of this selected
live-plant response law; do not invent new field laws in this correction.

Report actual final signatures/budgets/deviations; tests NOT RUN. Codex will run
full CPU, actual CUDA, the independent probes, native/restart fixtures, examples
and measured global memory before accepting or correcting again. Phase2 remains
uninstalled/unpublished; Phase3 does not start until Phase2 commit/push succeeds.
