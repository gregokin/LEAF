# Codex review — shared canopy revision 1

Verdict: **correct** (focused metadata/documentation clarification).

All 137 CPU checks pass (130 pass / 7 CUDA skips). Independent exact field/sidecar
comparisons against five frozen legacy cases pass, including one geometry/shared
call per adapter call, unchanged state and pure imports. Native MAPLE ABI passes.
Writer scope and structured success/reaping verified. Numerical work is correct.

The shared `height_anchor` and shared guide claim the exposed canopy base is always
on the current surface. For an eroded anchor, data correctly retain establishment
ground and `height_m == intact_height_m`, so its canopy base is elevated above
current ground. Metadata must describe that distinction: the retained geometry
must not appear translated down onto the eroded surface. Correct the shared
docstring/metadata and affected guides; preserve every numerical array/algorithm,
legacy adapter schema/behavior and test expectations. Also qualify the new guides'
absolute "can never see different vegetation" claim (same state/catalog/surface
required; upstream integration pending) and say removed plants are excluded by
default, since an explicit status selection can include them.

See prompt_02.md. No new physics, coordinates, fields, resource bounds or tests
are needed for these descriptive corrections.
