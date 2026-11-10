# Shared aboveground vegetation interface — revision 1

You are Claude, sole implementation writer in candidate/. Codex orchestrates,
reviews, tests, installs, and handles Git. User approved automatic collaboration
and sharing for Phase 0 and now requests neutral naming and maximum shared code
between MAPLE and SYRUP, preventing inconsistent vegetation representations.
This is the bounded Phase 0 interface correction, not new model physics.

## Instructions and scope

Safe-mode assignment: this prompt embeds governing instructions. May read ONLY
this prompt and files within candidate/. Candidate contains current LEAF modules,
one Phase 0 test module, two numerical benchmark modules, example fixtures and two
interface guides. No outside brief, raw research data, credentials, settings,
other projects or historical evidence. Tools: ordinary Read/Edit/Write only.
No shell, Python execution, agents, network tools, Git, installs or model runs.
Report tests honestly as NOT RUN. Preserve all files outside allowed writes.
One writer: you own candidate scope only; Codex owns controls/verification.
On tool/auth/network/rate errors preserve work and report exact literal errors;
automatic collaboration remains enabled. Do not bypass permissions.

Allowed writes relative to candidate/:

- leaf/canopy.py
- leaf/__init__.py (module documentation only)
- tests/test_phase0.py (focused meaningful shared-interface/compatibility coverage)
- docs/phase0-vegetation.md
- docs/maple-leaf-integration.md
- docs/shared-canopy.md (new shared contract for MAPLE AND SYRUP)
- report.md

All other copied files, numerical equations, resource bounds and fixtures stay
unchanged. Do not edit leaf/__main__.py: its current export-canopy behavior remains
the MAPLE-compatible projection for CLI compatibility. No upstream writes.

## Required design

1. Add public `leaf.canopy.canopy_view(state, catalog, current_surface, *,
   statuses=('live', 'dead_standing'), height_mode='exposed', drag_coefficient=1.0)`.
   This is the shared, consumer-neutral aboveground representation. Use existing
   `derive_geometry` as the single allometry/surface-geometry implementation.
   Convert to domain-local positions and canopy fields in ONE place only.
   Return the familiar dict sections `fields`, `sidecar`, `meta`; O(P) arrays;
   no grid rasterization, physics, state mutation, caches, new native dependency.
   Shared fields remain center_x_m/center_y_m, equal semi_major_m/semi_minor_m,
   orientation_deg, height_m, porosity, drag_coefficient. Allometry conventions,
   exposed/intact height modes and constant porosity stay unchanged.
   IDs, species, status, intact/exposed heights, establishment/current ground,
   burial/exposure and root-radius sidecars stay consistent with existing values.
   Use neutral `CANOPY_FIELDS` (preserve `MAPLE_FIELDS` alias for compatibility).
   Neutral schema `leaf.canopy_view`, version 1, explicit coordinate/height/units
   and catalog/time/boundary metadata. Clearly distinguish geometric visibility
   from plant death or removal. Preserve per-plant native MAPLE compatibility
   flags as informational metadata, not a universal physical validity judgment.

2. A valid shared view MUST RETAIN fully buried and elevated-base plants in the
   selected standing population, with correct exposed height/anchor diagnostics;
   it must NOT reject or drop them because MAPLE's current object container
   cannot represent them. Invalid/unresolved state/current surface still fails.
   Default selected statuses and zero-diameter behavior remain as before. Data
   are derived from authoritative PlantState, never a second editable population.

3. Retain public `maple_vegetation_view` with its existing signature/defaults and
   legacy output schema `leaf.maple_vegetation_view`, version 2. Refactor it into
   a thin compatibility adapter OVER canopy_view: ONLY native-MAPLE error/exclude/
   flag policy, projection/filtering and compatibility metadata. No duplicate
   allometry, surface sampling, canopy geometry or coordinate conversion there.
   Existing outputs (fields, sidecars, counts, validity, diagnostic height labels)
   must remain equivalent; all original Phase 0 tests must pass unchanged in their
   expectations. When policies exclude plants, apply the SAME index selection
   to ALL plant arrays and update metadata consistently. Stable ID association
   is essential. Keep meaningful actionable error messages.

4. Both future consumers call canopy_view for the same geometry. The legacy MAPLE
   adapter remains available because its current restrictions are MAPLE-specific;
   SYRUP must not inherit those restrictions. Do not introduce a second SYRUP
   allometry, a rasterizer, a hydrological roughness law or an invented adapter.
   Explain that upstream adoption/physics qualification is still future work.
   If an upstream model needs a raster, it derives it from the shared arrays and
   retains IDs; PlantState remains authoritative and water ownership unchanged.

5. Documentation: identify canopy_view as primary shared API; show a short call
   and interpretation, exact units/anchor/height/porosity semantics and refresh
   after state/surface changes. MAPLE-specific guide describes the thin adapter;
   existing CLI remains MAPLE-compatible. No claim of integrated SYRUP/MAPLE
   execution or changed ecological physics. Update package module description.

6. Add only focused tests needed for this separation: shared retains buried and
   exposed plants; supported geometry is identical across shared and MAPLE views
   by ID; adapter uses shared computation and preserves its legacy policies.
   Do not rewrite existing unrelated tests or add a duplicate allometry formula.

## Acceptance (Codex executes)

- Full existing CPU suite passes plus focused shared-view tests.
- Independent exact field/sidecar checks by stable ID across views, including
  native-MAPLE exclusions; pure imports need only NumPy; native MAPLE ABI still valid.
- Allometry and surface-geometry computations have one implementation, shared
  view independent of consumer-specific representation limitations.
- Strict allowed-write diff; original AGENTS.md, old task archives, numerical
  drivers/traits, input fixtures, old dependency locks and upstreams untouched.
- Report exact files, design/compatibility/limits and tests NOT RUN in report.md.
