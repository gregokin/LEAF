# Current Verification

Shared canopy revision 2 passed independent verification and is installed by
frozen hashes. [Evidence](tasks/shared_canopy_view_20261110/verification.md), [installation](tasks/shared_canopy_view_20261110/installation.json),
[preservation](tasks/shared_canopy_view_20261110/scope_final.json).

- 137 CPU checks: 130 pass / 7 CUDA skips; all 47 Phase 0 checks pass.
- Five pre-refactor fixtures match legacy geometry/IDs exactly; shared arrays and
  the MAPLE projection agree by ID, including exclusions.
- Single shared/geometry evaluation, purity/import isolation and native MAPLE ABI pass.
- Numerical geometry/cover functions and existing tests are AST-identical;
  historical locks/archives, AGENTS.md, traits/fixtures/drivers and upstreams preserved.

No installs, new environments, physical model scenarios or new GPU benchmarks.
