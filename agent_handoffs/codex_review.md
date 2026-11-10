# Current Codex Review

Verdict: **accept** — shared canopy interface revision 2, 2026-11-10.

Claude implemented the candidate; Codex independently verified and installed it.
See [durable review](tasks/shared_canopy_view_20261110/codex_review_02.md) and [evidence](tasks/shared_canopy_view_20261110/verification.md).
`canopy_view` is the primary MAPLE/SYRUP geometry API; the existing MAPLE function
is a thin projection, preserving legacy geometry, policy and CLI behavior.

137 CPU checks: 130 pass / 7 CUDA skips. Exact legacy fields/sidecars match five
frozen cases; shared identity/geometry, single computation, import isolation,
state purity and native MAPLE ABI pass. Historical numerical/GPU qualification
remains applicable to unchanged kernels; no new GPU or integrated run claim.

Current source lock: [canopy-view.lock.json](../dependencies/canopy-view.lock.json).
Commit/push authority continues for this bounded Phase 0 refinement.
