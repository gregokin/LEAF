# Shared canopy — focused revision 2

Same bounded task, source scope and Read/Edit/Write-only rules as revision 1.
May read this prompt and candidate/. Seven allowed write paths remain exactly
the same. No execution, Git, installs, agents, upstream/context reads or new physics.
Codex ran all 137 checks (130 pass / 7 CUDA skips), independent five-case legacy
equivalence/purity/single-computation checks and native MAPLE ABI: all pass.

Correct ONLY descriptive strings/docstrings/docs and report.md:

1. In shared canopy_view docstring and `meta['height_anchor']`, and the shared
   guide/Phase 0 guide, exposed height must not claim its base is always on current
   ground. On erosion, establishment ground remains fixed ABOVE current ground;
   canopy height remains the intact height, and the exposed gap is separate.
   For a non-fully-buried exposed canopy, visible base elevation is
   max(establishment_ground_m, current_ground_m). On partial burial it is current
   ground; on erosion it is establishment ground. Fully buried plants have zero
   exposed height/no visible canopy and retain their establishment anchor. Make
   this clear in metadata and sample interpretation. Do NOT add an array or change
   ANY numerical calculation/field/sidecar/selection. Keep native MAPLE compatibility
   adapter legacy metadata/behavior unchanged; these are shared-view descriptions.
2. Qualify "can never see different vegetation" in new/edited guides: common code
   yields consistent geometry when consumers use the SAME plant state, catalog and
   surface; they must refresh at matching coupling times. Upstream adoption remains
   pending; no absolute promise or integrated run claim.
3. Change "removed plants never published" in affected guides to "excluded by
   default" (an explicit `statuses` selection can include a removed positive-size
   record). Scope "only allometry implementation" to the canopy export path,
   avoiding a claim about every traits/generator utility in the repository.

Keep all existing implementation and tests unchanged. Update report.md briefly
with revision 2 corrections and tests NOT RUN. Codex verifies and handles Git.
