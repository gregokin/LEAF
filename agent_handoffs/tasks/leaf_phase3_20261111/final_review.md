# Phase 3 final independent review

Verdict: **accept** snapshot_03, exact Claude session a5c6d72f-e529-49ec-beea-b376caaa8814. Revisions1–3 completed/reaped with zero scope violations and zero intermediate tool errors. Claude was the sole implementation writer, using ordinary source tools and running no checks; Codex independently reviewed, tested and profiled. Thirteen original implementation outputs were installed byte-for-byte, recorded in installed_phase3.json and dependencies/phase3-reproduction.lock.json. Root AGENTS.md and all upstream files are preserved. Phase1 f0d0b56 and Phase2 5e887e6 were separately committed/pushed before this phase.

## Accepted implementation

Opt-in simple_seed_clonal_v1 stores every species coefficient in the species catalog. Seeds use one probability trial and bounded integer count per species and fixed physical ecological patch, independent of hydraulic resolution. Their initial dry aboveground biomass is an explicit external establishment credit. Vegetative daughters are separately located plants with their own stable IDs, age0 and sampled current establishment ground; accepted daughters receive an exact aboveground biomass/seasonal-mass transfer from eligible parents. Parent biomass stays positive, rejected placements donate nothing, existing coordinates/anchors/IDs/dead records persist and canopy overlap is explicitly allowed.

The shared session rebind appends births and rebuilds the common canopy/root state. Surviving individual root overrides persist; new plants default to species roots. Native SYRUP water and time are untouched by instantaneous biology; new roots withdraw actual water only on the next interevent step. Random decisions are deterministic by stable signed IDs, seed, opportunity counter and separate streams. Opportunities are sequential; missed years require explicit decline, with no RNG consumption or catch-up growth.

An explicit flush_deferred_season boundary finalizes only prior-season tissue before reproduction, with no time/water change and full failure rollback. Newborn background hazard excludes time before birth. Complete restart format3 carries the reproduction state and validates its history against calendar/counters; canonical plant CSV remains format1. Earlier complete restart formats1/2 are refused without a qualified migration.

## Independent final verification

- snapshot03_cpu: 527 tests,45 CUDA skips,zero failures (482 executed passes).
- snapshot03_cuda: 527 tests,one multi-device skip,zero failures (526 executed passes), actual selected physical GPU1. No other GPU was used or other job touched.
- snapshot03_native and snapshot03_native_cuda: real native SYRUP water, empty/mixed populations, donor/seed dry-AGB ledgers, new actual next-day uptake, frozen dead records, exact split-run and pending-growth restart, rollback after injected root failure, opportunity replay refusal. Existing partial-year metrics and root overrides survive; newborn metrics start zero and species root fractions match.
- snapshot03_deferred_season: empty,live-zero-E and live-positive-E cases; refused out-of-order birth, failed finalization rollback, finalization exactly once, fixed water/time/growth grid, and no prior-season loss applied to newborn tissue.
- snapshot03_example: bounded compiled CPU example writes shared plant CSV and complete checkpoint, then restores successfully.
- snapshot02_guards, snapshot02_restart_metadata, snapshot02_legacy and snapshot02_newborn pass. The relevant recruitment.py/checkpoint.py/demography.py bytes are identical in final snapshot03. These cover malformed/duplicate/contradictory IDs and patches, int64 endpoint/exhaustion, grid-resolution-invariant patch draws, impossible checkpoint histories, actual historical checkpoint refusal and independent newborn probability golden values.

Every check has exact argv/cwd, PID/start identity, separate stdout/stderr, start/end timestamps and reaped exit records under the named prefixes. Structured author completion is not substituted for any verification.

## Executed memory and timing qualification

Selected GPU1 GTX1080Ti, CuPy14.2.0, unchanged1.5-billion-byte pool cap; session bound512MiB, one-million root-edge limit. Synthetic clustered population on64x64 cells,two layers; these are bounded qualification cases, not field or full MASY scenarios.

- snapshot03_birth_memory:100000 existing records/50000 live;50000 daughters+64 seedlings produce150064 total/100064 live,574654 root edges. Startup/warmup1.53s (R2 repeat), final annual recruitment0.488s, following actual native day0.00687s. Recruitment declared peak138461632B; measured host new traced peak and device allocations are separately recorded (tracemalloc also includes Python overhead). Following-day simultaneous array lower34565976B <= declared36395608B. Pool reserved approximately33.2MB. Host placement/readback/upload is explicit; this is not fully device-resident reproduction.
- snapshot03_reject_many_memory:100000 records/25000 live,62403 requested daughters,all rejected. Exact concurrent NumPy allocation-domain bytes + resident host arrays + current hooked CUDA arrays give a24018366B lower bound <= corrected42398464B declaration. Actual next-day lower14207944B <= declared14911616B. This case's recruitment wall time includes observation instrumentation and is not a performance claim.
- Both profiles have zero unknown frees and zero hooked live bytes after cleanup. Measurements exclude unobserved runtime/context/process overhead; no complete process peak or30-year runtime extrapolation is claimed.

## Disposition of earlier failures and limitations

R1 inactive-species coefficient query, standalone identity/patch guards, impossible restart metadata and real prebirth seasonal loss were corrected. R2's all-rejected candidate case retained pure admission but omitted evaluate work from the session peak receipt; observed24018366B exceeded reported20432608B. R3 records the full already-admitted evaluate stage on every accepted call, including zero births. No cap, traits, state or numerical physics was changed to hide that failure. All original failing artifacts and interim trees remain archived.

All example ecological coefficients remain UNCALIBRATED. Seed banks, calibrated recruitment/dispersal/spacing, belowground/carbon pools, frost/fire mortality, standing-dead removal, evolving-surface root remapping and full MAPLE/MASY transport adapters remain outside this phase. No upstream edits, package installation, new environment or transport scenario was performed. The existing Phase0 cover/diameter generator remains complete and tested; this phase adds runtime recruitment rather than rerunning initial placement.
