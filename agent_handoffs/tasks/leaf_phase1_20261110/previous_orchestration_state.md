# Orchestration State

Status: **idle — shared canopy interface accepted and installed**, 2026-11-10.
Latest task: [tasks/shared_canopy_view_20261110](tasks/shared_canopy_view_20261110/codex_review_02.md), revision 2.
Automatic Codex–Claude collaboration and authorized bounded sharing remain enabled
for user-requested tasks. No writer is active; no daemon/scheduler is created.

Claude session `a71f51bb-359e-4900-bf96-e903d709713c` completed two scoped revisions;
structured success, exit 0, both children reaped. Codex independently reviewed,
tested and installed the final source. [Evidence](tasks/shared_canopy_view_20261110/verification.md): 137 CPU
checks, 130 pass / 7 CUDA skips, plus exact five-case legacy equivalence, shared
arrays by ID, single computation, purity/import isolation and native MAPLE ABI.

Primary common API is `leaf.canopy.canopy_view`, with one geometry implementation
for MAPLE/SYRUP. Existing `maple_vegetation_view` is a thin compatibility adapter.
Both consumers must use matching state/catalog/current surface and refresh at
matching coupling times; no duplicate allometry or plant population. Upstream
adoption and physical interpretation still require separate integration work.

[Current lock](../dependencies/canopy-view.lock.json) supersedes LEAF identities in
historical Phase 0 locks. Numerical drivers, original geometry/cover algorithms,
traits, fixtures, AGENTS.md, previous archives/locks and upstreams are preserved.
See [shared guide](../docs/shared-canopy.md) and [MAPLE handoff](../docs/maple-leaf-integration.md).

Continuing Phase 0 commit/push authority applies to this bounded refinement.
Origin: https://github.com/gregokin/LEAF.git; main tracks origin/main. Codex handles
accepted Git publication. Prior Phase 0/publication commits are 250c0ef/316d9f6;
Git history records the follow-up commit. No new implementation is assigned.
