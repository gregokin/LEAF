# Codex review — shared canopy revision 2

Verdict: **accept**, 2026-11-10. Revision 2 is installed byte-identically to the
frozen candidate (`installation.json`). Claude was sole candidate writer; Codex
independently reviewed, verified and installed the six production/test/guide files.

`leaf.canopy.canopy_view` is now the primary shared aboveground interface for
MAPLE and SYRUP. It derives geometry through the existing `derive_geometry` and
assembles domain-local arrays once. The existing `maple_vegetation_view` delegates
to it and applies native-MAPLE representation policies. Supported fields/sidecars
and metadata contracts remain equivalent to the published interface; legacy CLI
behavior/schema are retained. Buried/elevated standing plants remain in the common
view, with stable IDs and geometric visibility distinct from ecological status.

Independent final CPU suite: **137 checks, 130 passed / 7 CUDA skips** (all 47
Phase 0 checks pass, including three focused shared-interface tests). Five frozen
legacy output cases match exactly by ID. Instrumentation confirms one shared-view
call and one geometry evaluation per adapter call; tests also pass deliberately
modified shared arrays through the adapter, proving it does not recalculate them.
Import isolation, state purity, native MAPLE container ABI and installed-tree
independent checks pass. `git diff --check` passes.

Revision 1 already passed numerical checks; revision 2 corrected descriptive
height-anchor metadata and guide wording. After erosion, the plant's base remains
at establishment elevation; the exposed gap is separate from intact canopy height.
This changed no numerical output. Common geometry consistency requires the same
plant state, species catalog and current surface at matching coupling times.

Scope checks preserve all 1,563 baseline tracked files outside the explicit change
set. AST comparisons prove original geometry/cover numerical functions, existing
test functions and package initialization behavior unchanged. Historical locks,
accepted archives, AGENTS.md, traits, fixtures, numerical drivers and upstream
sources are preserved. All 71 SYRUP runtime hashes and MAPLE's native container
hash match the previous qualification. New source lock is
`dependencies/canopy-view.lock.json`; prior locks remain historical.

No new hydrological law, rasterization, GPU kernel, ecological physics, model
scenario or upstream integration was introduced. Existing GPU qualification is
historical; no new GPU performance or integrated execution is claimed. Upstream
adoption must use this shared view and retain ID association; SYRUP must not copy
allometry or inherit MAPLE-only container restrictions.

The continuing Phase 0 commit/push authority applies to this bounded refinement.
Codex may commit and push the accepted result to the configured user-supplied
`https://github.com/gregokin/LEAF` remote, using a normal fast-forward push.
