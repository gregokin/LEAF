# Phase 2 final independent review

Verdict: **accept** snapshot_05, session711c5973-64c3-45f8-af8a-379ba0e357a7. All five revisions completed and reaped; each immutable scope audit reports zero scope violations and zero intermediate tool errors. Claude wrote the implementation and tests without executing them; Codex independently reviewed and ran verification. The eleven original implementation outputs were installed byte-for-byte, recorded in installed_phase2.json and dependencies/phase2-disturbance.lock.json. Root AGENTS.md and all upstream source files remain unchanged.

## Accepted behavior

Opt-in, species-local mechanical_response_v1 handles live-plant burial, erosion/root exposure/dislodgement and abrasion with explicit diagnostic units and an aboveground dry-biomass ledger. Age/background/stress mortality uses stable signed-int64 plant IDs, counter-keyed RNG and persistent annual metrics from actual growth outputs. Session biology is transactional, rebuilds surviving roots/shared canopy geometry, preserves authoritative SYRUP water, and prevents stale or repeated decisions. Canonical plant CSV stays version1; the expanded complete checkpoint explicitly becomes version2. Old checkpoint versions are refused without a qualified migration.

## Independent evidence

- snapshot05_cpu: 457 tests, 25 CUDA skips, zero failures (432 executed passes).
- snapshot05_cuda: 457 tests on the selected idle physical GPU1, one multi-device skip, zero failures (456 executed passes). The existing 1.5-billion-byte pool cap was retained.
- snapshot05_native_biology and snapshot05_native_biology_cuda: actual native SYRUP water, mortality/dislodgement, dead-root exclusion, frozen dead ages, replay rejection, unchanged water during instantaneous biology and exact checkpoint continuation through year-end; both mixed live/dead and zero-live cases pass.
- snapshot05_signed_ids_cuda and snapshot03_signed_ids: independent hash golden values, row permutation, zero/negative IDs and both int64 endpoints pass.
- snapshot04_entrypoints: eleven independent malformed-input and one-byte-cap refusal probes pass. R5 changes the CuPy namespace guard, not numerical rules or these admission paths.
- snapshot05_legacy_checkpoint: real Phase1 format1 refused both normally and with source-change override; original files unchanged.
- snapshot05_example: compiled CPU example completes disturbance, annual biology, CSV and format2 restart.
- snapshot02_half_dead_memory: actual 100000 records / 50000 live / 4096 cells / two layers, observed array-data lower bound18125656B <= declared19237416B, zero persistent ID-cache bytes. Final interevent.py and disturbance.py are byte-identical to this measured revision; later demography changes only validation. This is not a complete allocator/context/Python-object peak measurement or a full-domain runtime benchmark.

Exact commands, process start identities, separate streams, reap/exit records and selected-device inventory are preserved under these artifact prefixes.

## Corrections and limits

Early CPU fixtures omitted a required explicit abrasion dose; fixtures now supply zero rather than weakening the law. Direct-input guards, pure-copy admission, an unaccounted population-ID cache, annual ordering and two CUDA guard/fixture issues were corrected. Codex's erroneous positive-ID instruction was explicitly withdrawn; signed canonical IDs are preserved. Reviewer harness errors (no-live annual calls, incorrect restart filename and uncaught secondary probe exception) are retained and corrected in later harness revisions, not attributed to product defects.

All new ecological response numbers are uncalibrated fixtures. Standing-dead mechanical removal/decay, field calibration, full MAPLE/MASY transport adapters, evolving-surface root remapping, fire and frost mortality are outside this phase. No transport scenario, package installation, upstream edits, raised resource cap or discarded state was used to pass checks.

## Next authorized phase

Publish this whole phase, then begin bounded Phase3. User explicitly selected nearby vegetative daughter plants with their own IDs and ages; same-parent clonal recovery is not the selected design. Seeds and vegetative establishment remain opt-in species-local laws with clearly labelled uncalibrated examples and explicit biomass accounting.
