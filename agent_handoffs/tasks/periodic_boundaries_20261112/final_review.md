# Independent acceptance: periodic canopies and roots

**Accept revision4**, exact source in `review_04.manifest.json` and `qualification.json`. Claude is the sole implementation writer; Codex independently reviewed, executed, corrected and accepted. All four Claude launches were reaped; actual file access/write scope passed. Recovered tool errors and rejected earlier candidates are preserved, never counted as acceptance.

## Result

One immutable per-axis `Domain.periodic` drives full-domain generation, canopy exclusion/union cover, exact canopy PET support, unique layered root footprints and daughter wrapping. Species root fractions and the demand/allocation/growth laws are retained. Physical water still belongs solely to SYRUP. One ID/record per plant, exact upper aliases, mixed axes/origins, bounded sparse arrays, source-bound checkpoints and the plain CSV plus companion convention are preserved.

The shared exact circle–rectangle integrator also corrects a pre-existing zero-breakpoint tangency error (a circle tangent to all four cell sides previously returned the entire square). Random default reference outputs remain byte-identical; that degenerate numerical case intentionally changes to the analytical circle area.

## Executed on the completed frozen source

- CPU:730 tests,677 passed,53 skipped. CUDA:730 tests,722 passed,8 skipped. CPU-only environment skips CUDA; CUDA suite skips the unavailable-path test, two omitted case helpers and five native tests. The five actual native tests are separately configured and pass5/5 on EACH CPU/CUDA. No configured failure was skipped.
- Independent geometry:640 root cases,32 canopy PET cases and4 sampled covers on EACH NumPy/Numba/CUDA, including roots wider than the domain, unique cell-layer resources and opposite-seam water. 48 decimal upper-alias and32 mixed-boundary endpoint cases pass. 80 independent adaptive-quadrature canopy-cell union cases pass. Legacy seeded plant/root/canopy hashes match exactly.
- Native Phase1.5: two eight-day6x4x2-layer CPU/CUDA harnesses use the genuine source guards. Each has8 daily hydrology updates,8 accepted zero-time transpiration debits and691200 physical seconds; weekly growth, failure rollback and pending/closed restart without replay pass, with residual inside the native bound. Biological CPU/CUDA outputs agree under the recorded FP64 contract.
- Historical c76 standalone driver: periodic eight-day continuous vs day3 checkpoint/resume numerical state matches on CPU/CUDA; wrong boundary declarations are refused and the native source guard stays enabled. The first Codex harness indexed the restored calendar twice and was correctly refused; corrected harness and original failure remain recorded.
- Cover peak after mask-lifetime correction:6,246,595 traced bytes versus6,291,472 declared array bytes in the1M-midpoint synthetic fixture (plus32KiB allowed Python metadata tolerance). Before correction it used6,705,429. Caps unchanged. Synthetic25,079-daughter placement remains below its existing9.09MB conservative estimate and closes dry-biomass bookkeeping.
- The new small periodic CLI example generated576 plants in scratch output; no trial case was regenerated or advanced.

Commands, separate streams, reaped exits, actual source identities and numerical/memory scopes are in `qualification.json`. Cases, catalogs, existing tests, AGENTS.md and historical locks remain unchanged. Real native qualification selects published SYRUP a5b8b183/package82e9c011 with MAPLE9f466d2/packagefbc504bb; the dirty live MAPLE checkout is read-only handoff context, not test input. No full coupled transport or ecological validation is claimed.

## Handoff / remaining work

`developer_handoff.md` gives actual public calls and the current MAPLE adapter step: propagate GeometrySpec periodic flags through `leaf_domain`, remove the2m wrapper inset and inner-cover correction only when separately regenerating. MAPLE's canopy/shelter wrapping already works. Square-cell restriction and explicit JSON/Domain declaration remain. Old checkpoints require explicit source compatibility handling; no historical lock was rebound. Phase3 scientific work stays paused at `phase3_preservation.md`.

Codex may now install exact accepted source, run installed-tree checks and commit/push under the standing authorization. This acceptance does not authorize trial regeneration, calibration, upstream edits or new ecological laws.
