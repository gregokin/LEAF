# Phase 0 — bounded correction revision 2

Continue the exact Phase 0 session. Same user authority, candidate-only writer,
Read/Edit/Write tools, no shell/agents/network tools/installs/Git/execution claims.
Preserve all copied legacy numerical modules and upstreams. Codex independently
ran checks; this is correction, NOT permission to widen the scientific scope.
All original requirements in prompt_01 remain authoritative with the explicit
placement-order clarification below. Leave changes uncommitted.

Allowed writes are the SAME exact 17 candidate paths in revision 1. May also
READ these task-local Codex review artifacts outside candidate: review_initial.md,
review_probes_01.py, review_geometry_recording_01.py, independent_reference.py,
generation_oracles.py, early_state_probes.stdout.log,
early_geometry_recording.stdout.log, diagnostic_generation.stderr.log,
candidate_phase0_01.stderr.log. No other outside source. Codex owns these files;
do not edit them. Tools have no execution ability; do not claim checks ran.

## Observed results

- All original 90 legacy CPU checks still pass with your lazy imports (83 pass,
  7 CUDA skips), and original numerical sources are preserved.
- Your first Phase 0 suite ran 33 tests in ~1.9s, with three failures. Two are
  incorrect fixture expectations; one is an output scheduling issue (below).
- After your final dry review, the archived revision_01 suite still runs 33
  tests, with two failures: the status-count arithmetic, and a collision test
  whose index/time/kind template does not actually collide. Surface subset and
  skipped-time expectations were already partly corrected; retain those fixes.
- Independent exact FP64 CSV round trip passes and allowed-overlap cover agrees
  with an independent random-point/SciPy geometry oracle.
- Independent state/geometry/recording probes found failures listed below.
- A modest mixed no-overlap case spent >150 CPU seconds placing only ~3000
  plants; exact verification child was terminated and reaped. A diagnostic
  15-second deadline confirms the stack in exclusion placement. No Claude writer
  was interrupted, no limits raised, and no partial population was accepted.

## A. Fix the three test outcomes for the RIGHT reasons

1. `test_surface_token_resolved_once_and_anchor_preserved`: three rows have
   explicit, different establishment anchors. On the heterogeneous surface these
   do NOT all have burial=0.3 after SURFACE rows resolve. Keep explicit anchors
   intact. Check delta=sampled_current_surface-stored_anchor for all rows, and
   the exact 0.3/0.2 changes on only the SURFACE subset. Do not change placement
   semantics to satisfy a wrong test expectation.
2. `test_updates_are_transactional_and_by_id`: fixture has six plants, initially
   four live/one dead_standing/one removed. Removing ID 1 leaves THREE live,
   one dead_standing and two removed. Correct this arithmetic and retain tests.
3. Cadence snapshot at t=35 after t=10 with cadence 10 has missed requested times
   20 AND 30. Save only the actual t=35 state, record both missing requested
   times, next_due=40. A snapshot actually at t=30 would miss only t=20.
4. In the final collision test, w2.write_initial(s) uses t=37, unlike the
   existing initial file at t=0, so no collision exists. Likewise a new writer's
   index=0 cadence path differs from the previous writer's index=1 path. Build
   an ACTUAL matching destination (same kind/index/time) before asserting
   FileExistsError, and still test that failure leaves scheduling untouched.

## B. Establishment, strict state and catalog ownership

- Metadata-bearing noninitial CSV with SURFACE must fail BEFORE sampling any
  newer surface. SURFACE is allowed in a handwritten initial list or explicitly
  marked initial input; later saved states carry concrete anchors. Reject a
  missing/false initial declaration on metadata-bearing unresolved files.
- Direct constructor NaN must NOT implicitly request SURFACE. Require the
  explicit unresolved mask; reject other nonfinite anchors. Prevent advancing
  unresolved initial placements into later time without first resolving them.
- Strict CSV parsing must reject an unterminated last quoted field even when
  the parser would otherwise recover it and the row count agrees.
- Preflight clock/age updates for finite results and commit together. A failed
  advance (overflow, invalid state or read-only input array) changes nothing.
- Ensure constructor inputs do not unexpectedly alias the canonical state and
  that stable ID lookup cannot become stale through writable/caller-owned ID
  arrays. Prefer owned arrays and read-only identity fields, or an equally clear
  robust ownership model. Keep by-ID operations correct after supported updates.
- Catalog.document must be a defensive copy/read-only view of private content;
  user edits must not change canonical content behind a stale hash/coefficient
  cache. Keep the name/index table deterministic across JSON key reordering,
  or validate/remap it in EVERY consumer (including legacy bridge).
- Reject nonfinite allometry OUTPUTS (finite input can overflow), bool/complex/
  string numerical inputs, invalid physical domain area and conflicting optional
  domain metadata such as a claimed periodic edge or different sampling rule.
  Validate a plant state and derived output before canopy publication.
- State X/Y values with a nonzero origin are landscape-frame coordinates, not
  already origin-relative. Correct the metadata labels consistently with code.

## C. Safe MAPLE canopy publication

Current default `maple_vegetation_view` exports a fully buried plant at full
intact height with n_unsupported=0 when current_surface is supplied. This loses
the establishment reference. Make safe publication use the CURRENT surface and
the anchor by default, require enough terrain information to validate placement,
and use exposed geometry or refuse unsupported cases. Partially/fully buried
plants must not silently reappear at full height atop deposited soil. Elevated
bases must still fail/exclude explicitly because native MAPLE cannot represent
them. Generic intact allometry remains available from derive_geometry.

If retaining explicit diagnostic/approximation modes, label them unmistakably
and do not call a zero-height/unsupported object native-MAPLE-valid. Align CLI,
examples, docs and tests with the safe default. No new ecological damage law.

## D. Recording and output transactions

- Failed write_initial currently changes start/next-due state. Stage ALL
  scheduling/index/provenance changes until successful write (also after initial
  validation failure or I/O error). No wrong bookkeeping on failures.
- Fix missed requested times as described in A, keep actual snapshot times.
- Atomic no-overwrite must not fall back to check-then-os.replace when link
  fails. Fail safely on filesystems without required atomic primitive rather
  than risking an existing destination. Explicit overwrite remains intentional.
- Reject generate --report identical/aliased to --out before generation or
  output; currently --overwrite can replace the plant CSV with JSON. Preflight
  both destinations and serialise/validate both outputs before committing files.
  Avoid leaving a new population output after an avoidable report error. Preserve
  existing files and do not hide a failed report behind a success exit.

## E. Correct bounded exclusion placement and cover working memory

Replace random interleave for no-overlap with explicitly documented,
reproducible LARGE-CANOPY-FIRST exclusion placement. Sampling marked sizes then
sorting by descending diameter is an acceptable initialisation algorithm; it
is NOT an unconditioned Poisson process. User was informed of this implementation
decision. Both modes still use uniform random center proposals and the exact
requested size distribution, cover and global overlap constraints.

IMPORTANT: count corrections that add shrubs after grasses also encounter the
same failure. Rebuild/re-place the complete candidate population in size order
after count changes, or an equivalently bounded strategy. This occurs BEFORE
any plants receive final IDs/are established, so it does not move live plants in
an ongoing run. Do not rescale diameters, relax cover/exclusion, keep a partial
population, or raise bounds to make tests pass. Report inability within the
algorithm's bounds without claiming mathematical infeasibility of all packings.

- Remove repeated sum(alive) and other quadratic bookkeeping from tiny batches.
  Maintain counts incrementally. Bound all allocated candidate slots, including
  dead trial positions, and preflight correction counts before allocating items
  arrays/repeats. Rebuild/compact transient trial storage where needed.
- Cover computation allocates an unadmitted footprint-sized FP64 temporary:
  512^2 test reports 524288 raster bytes but tracemalloc peaks at 2894552.
  Tile/row-process footprints and reserve working scratch in the byte admission
  estimate; report precisely scoped raster and working bytes. Same applies to
  generation's preflight. Do not merely raise max_cover_bytes.
- Normal target-cover generation must not mark untested counts accepted via
  cover_tolerance=None. Enforce a finite positive tolerance for this mode.
- Save the complete small generation specification (diameter and age rules,
  bounds, domain, seed, RNG algorithm/NumPy version and ordering) or an equally
  reproducible referenced configuration, not only seed/count/cover. Reject
  nonfinite derived count/diameter-area quantities before allocation.
- Add a mixed-size regression at 128^2, dx=.25m, grass cover .06, diameter mean
  .15m SD .05m; shrub cover .05, mean 2m SD .3m, seed 7751. It must construct
  global nonoverlap and correct sampled covers within the SAME limits, e.g.
  max_attempts=50000, rather than needing millions of failed shrub insertions.

## F. Event-response transaction integrity

ReadOnlyView currently shares mutable Domain with the original state. Event
arrays can alias caller input and are mutable after construction; a handler
receives the live event and can change event.plant_id after indices were
resolved. Fix ownership/immutability/revalidation: frozen/copied domain and
diagnostics, stable validated IDs used through the whole transaction, and no
handler path can modify canonical state or redirect proposals on failure.
Test a handler attempting to mutate domain/event data then raising; original
state/domain/IDs must be unchanged. Mutated negative depths must fail or be
impossible. Preserve diagnostics-only default and species-law hooks.

## G. Legacy bridge must not silently alter plant state

- Check state validity, species name/index compatibility and exact integer ID/
  profile types/ranges before conversion (including unsigned int64 overflow).
- Reject unknown requested profile names and finite/nonnegative rtol errors;
  validate finite positive density from STEP traits. NaN tolerance/density must
  not bypass refusal.
- Refuse nonzero seasonal E on fresh legacy inputs unless it is explicitly and
  correctly restored; the current fresh driver starts E=0. Do not discard state.
  Capturing a legacy non-STEP driver must not silently erase pre-existing E.
- Default consistency must test the SPECIES LAW, not just one initial B: an
  arbitrary exponent can happen to match B at one size but diverge after growth.
  Enforce the legacy square-root diameter law/coefficient for growth reuse, or
  clearly refuse the incompatible handoff. The default fixture density is
  actually 1.0; an arbitrary helper-only density cannot make the unmodified
  driver consistent. Keep explicit approximation options honest and separate.
- Clock/age are caller-owned: preserve them and document advance once per elapsed
  interval. Driver B/E are the numerical authority while attached; capture creates
  the record view only at requested output. No independent edits to both copies,
  no claim that Phase 0 event proposals already update the legacy growth driver,
  and no full soil/calendar/pending-uptake restart claim.

## H. Packaging, exports and integration instructions

- pyproject readme must point at a deliverable that exists in the installed repo,
  e.g. docs/phase0-vegetation.md (present in candidate and final tree), not the
  candidate-only report.md.
- Make __all__/lazy exports internally consistent for advertised core modules.
  Keep import isolation and legacy imports working; no core NumPy-only import
  may pull in numba/cupy/native physics.
- Document finite clipped cover/exclusion vs any future periodic MAPLE raster
  boundary semantics. The future MAPLE adapter must match/requalify boundaries,
  not claim exact cover/nonoverlap under a different periodic interpretation.
- Update docs/examples to the actual corrected APIs and limitations, including
  proper surface-aware publication. Keep fixture parameters labeled.
- Make at least the main generation example reflect the intended initial scale:
  128^2 cells at .25m, grass mean diameter .15m (e.g. SD .03m) and shrub mean 2m
  (e.g. SD .3m), modest declared fixture cover. Ensure the documented list-init
  and export commands actually work with their explicit anchors/current surface;
  unsupported cases should be separate examples or handled explicitly.

Add focused regression tests for corrections, retain the existing meaningful
checks, and update report.md to revision 2, clearly stating tests NOT RUN by you.
Codex will rerun your suite, independent probes, real legacy capture, packaging
and bounded capacity. Finish the complete correction; no new scientific laws.
