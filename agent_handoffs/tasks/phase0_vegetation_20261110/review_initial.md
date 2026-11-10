# Preliminary independent review — revision 1 still being written

Codex review of completed modules while Claude remains the only candidate
writer. These are correction findings, not final acceptance. Source/tests/docs
may still change before the first report. No implementation files edited by Codex.

## Reproduced state and geometry failures

`early_state_probes` (exit 1; raw streams/launch/exit archived):

- Exact FP64 state round trip passes.
- A metadata-bearing noninitial t=7 snapshot carrying SURFACE is accepted and
  resolves the anchor to the supplied newer surface (55m). Reject this before
  sampling; initial anchors must resolve once and remain persistent.
- Direct PlantState construction interprets an unmarked numeric NaN as SURFACE.
  Only an explicit unresolved mask/token should request surface placement.
- `advance_time` can overflow age and mutate the clock before failing invariants.
  Preflight candidate time/ages and commit atomically.
- The default non-strict CSV reader accepts an unterminated quoted last status.
- Finite mass plus large allometry exponent can return infinite diameter.

`early_geometry_recording` (exit 1):

- Failed initial snapshot changes start/next-due bookkeeping (None -> 7 days).
- Default MAPLE view with current ground 10m, establishment 3m and intact height
  0.2m exports that fully buried canopy at full height with n_unsupported=0.
  Safe publication must account for current surface and anchors by default.
- Mutating catalog.document changes canonical content but not cached allometry
  or catalog hash. Establish immutable/private catalog ownership.
- 512^2 cover measurement reports a 524288-byte raster budget but tracemalloc
  sees 2894552 bytes because of footprint-sized FP64 temporaries. Include scratch
  in admission or tile the calculation and declare the precise memory scope.

## Additional source-review items to check at first completed revision

- State ID-index cache and constructor arrays alias mutable caller arrays;
  stable identities/transactional methods must remain sound.
- Validate scalar clock, domain area, output allometry and state before publishing.
- Atomic output fallback from failed hard link to check-then-replace can overwrite
  a racing destination despite overwrite=False; fail safely on unsupported links.
- ReadOnlyView shares a mutable Domain with original state, and handlers receive
  the live mutable event object. A raising handler must not mutate state/domain
  or change the diagnostic IDs used to apply proposals.
- Event arrays claim to be copies but can alias caller arrays. Revalidate or
  freeze records before application; mutated negative depths must not be accepted.
- Legacy bridge must reject nonzero seasonal mass that a fresh driver resets,
  and reject an allometric law that only happens to agree at initial mass but
  diverges from the driver's area law after growth. No silent lost state.
- Legacy bridge currently checks catalog hash but not the species-code table;
  reordered catalog keys have the same canonical hash and can misassign species.
- Reject nonfinite/negative tolerance/density, noninteger driver profile IDs and
  unsigned ID overflow, and unknown requested profile names before conversion.

Any correction prompt will be issued only after the current writer finishes and
its child is reaped. Core scope remains Phase 0; existing growth science and
upstream files stay unchanged.

## Generation findings

The independent 128^2, 0.25m test requests 6% grass cover (mean diameter 0.15m,
SD 0.05m) and 5% shrub cover (2m, SD 0.3m). Allowed-overlap checks completed;
the no-overlap run consumed >150 CPU seconds. Codex terminated only the exact
recorded check child and reaped it (early_generation exit -15; record preserved).
The diagnostic rerun exited after its explicit 15-second faulthandler deadline,
with the stack in no-overlap placement (`_Bins.conflicts` from add_batch).

Cause: random interleaving allows many small grasses to occupy the potential
spaces needed by later large shrubs. This can produce millions of rejections at
modest cover. Use a documented, reproducible large-canopy-first placement rule
for the exclusion mode, retaining sampled sizes and global canopy exclusion.
Do not relax the cover/overlap requirements or raise attempts to hide the cost.

Also fix:
- Repeated `sum(alive)` in small placement batches introduces quadratic
  bookkeeping; use an incrementally maintained count.
- Dead trial slots remain in every Python list/bin after count adjustments;
  bound allocated storage, not only live count, and compact/rebuild as needed.
- Correction batches allocate repeated-species arrays before checking whether
  the new population fits max_plants. Admit counts before those allocations.
- Preserve the complete small generation config (diameter/age rules, RNG
  algorithm/version, bounds), not only seed/target/count, in reproducibility data.
- `cover_tolerance=None` currently marks every prior-only population accepted;
  the normal cover-target contract must not silently bypass acceptance.

## Packaging/CLI findings

- pyproject readme points to report.md (candidate report), which is not a
  repository-root deliverable. Use README.md and verify a wheel in an isolated
  temporary build tree without installing dependencies or making an environment.
- `__all__` includes Phase 0 modules but __getattr__ has no lazy target for them;
  star exports must not raise for advertised names when optional legacy deps exist.
- generate --report equal to --out can overwrite the CSV with JSON when
  --overwrite is selected. Refuse path collisions before any output/generation;
  preflight both destinations and avoid partial outputs on failed validation.
- Document finite clipped cover/exclusion vs any future periodic MAPLE boundary
  semantics; integration must not claim identical cover across incompatible edges.
