# Shared aboveground canopy contract — `leaf.canopy.canopy_view` (schema `leaf.canopy_view` v1)

For the MAPLE and SYRUP developers. **`canopy_view` is the primary shared API**: one consumer-neutral
aboveground representation of the LEAF plant population, computed by one implementation. Because both
models call the same code, consumers that use the **same plant state, catalog and current surface** obtain
consistent geometry — provided they refresh the view at matching coupling times. This is a property of the
shared code, not an absolute promise: upstream adoption is still pending and no integrated MAPLE/SYRUP run
is claimed. This document is the contract for Phase 0. It is a bounded interface correction: **no new model
physics, no roughness law, no rasterizer, no integrated MAPLE/SYRUP execution.** Upstream adoption and
physics qualification remain future work (§7).

---

## 1. Authority and single implementation

* `leaf.state.PlantState` (+ the species catalog) is the **only** authoritative population. A canopy view
  is *derived on request*; it is never an editable second population and nothing in it writes back.
* On the canopy export path, `leaf.canopy.derive_geometry` is the **only** allometry
  (`diameter = a_d·B^b_d`, `height = a_h·B^b_h`), root-radius and surface-sampling implementation: every
  published view derives from it. (Other utilities in the repository — catalog inverse allometry, the
  generator, the legacy bridge — are outside this statement.) `canopy_view` calls it and is the **only**
  place that converts landscape positions to domain-local centres and assembles the canopy fields.
* `maple_vegetation_view` (§5) is a thin projection of `canopy_view`; it contains no geometry code.
* Consumers do not re-implement allometry, surface sampling or coordinate conversion.

## 2. Call and interpretation

```python
from leaf import traits, state, canopy
cat = traits.SpeciesCatalog.from_file('species.json')
pop = state.read_plant_csv('plants.csv', cat, domain=state.Domain(nx, ny, dx_m, dy_m))
# plants.csv is a PLAIN nine-column CSV (no embedded metadata). A neighbouring plants.metadata.json companion (written
# by LEAF; docs/phase0-vegetation.md section 2.3) is loaded and verified automatically and supplies domain / time, so
# `domain=` is only needed for a handwritten list (and must agree with the companion when both exist).
pop.resolve_establishment(initial_surface)                 # SURFACE anchors resolved ONCE (ny, nx)

view = canopy.canopy_view(pop, cat, current_surface)       # current (ny, nx) surface REQUIRED
f, sc, m = view['fields'], view['sidecar'], view['meta']
# f['center_x_m'][i], f['center_y_m'][i]  domain-local metres of plant sc['plant_id'][i]
# f['semi_major_m'][i] == f['semi_minor_m'][i]  canopy radius (circle), f['orientation_deg'][i] == 0
# f['height_m'][i]     exposed height = max(intact - burial, 0)  (0 when sc['fully_buried'][i])
# visible canopy base  max(sc['establishment_ground_m'][i], sc['current_ground_m'][i]):
#                      current ground under partial burial; the FIXED establishment ground under erosion,
#                      where the intact height is unchanged and sc['exposure_depth_m'][i] is the gap below it
# f['porosity'][i]     constant per-species catalog value; f['drag_coefficient'][i] uniform pass-through
```

Signature (defaults are the Phase 0 defaults and are unchanged from the legacy view):

```python
canopy_view(state, catalog, current_surface, *, statuses=('live', 'dead_standing'),
            height_mode='exposed', drag_coefficient=1.0) -> {'fields', 'sidecar', 'meta'}
```

Every array is 1-D, length `meta['n_objects']`, same order in `fields` and `sidecar`; O(P) work and memory;
no grid is rasterized. The result is a fresh set of arrays (safe to keep, never aliased to the state).

## 3. Selection — what is in the view

* Plants whose `status` is in `statuses` (default `live` + `dead_standing`: both have a canopy) **and**
  whose canopy diameter is `> 0`. `removed` plants are retained in `PlantState` and **excluded by
  default**; an explicit `statuses` selection that names `removed` includes removed records with a
  positive diameter. `meta['excluded_by_status']`, `meta['excluded_zero_diameter']` count the others.
* **Fully buried plants (exposed height ≤ 0) and elevated-base plants (eroded anchor) are RETAINED** with
  correct diagnostics (`fully_buried`, `elevated_base`, depths, `exposed_height_m = 0` when buried). A valid
  shared view never drops a standing plant because some consumer's object container cannot hold it.
* **Visibility is geometric, not biological.** `fully_buried`/`elevated_base`/`exposed_height_m` describe
  the canopy relative to the *current* surface. Death and removal are expressed only by `status`
  (ecological state, set by LEAF laws/events). A buried plant is still `live` until an ecological law says
  otherwise; nothing in the view infers dislodgement, injury or mortality.

## 4. Exact semantics

| field | unit / meaning |
|---|---|
| `center_x_m`, `center_y_m` | m, **domain-local**: landscape `x_m − origin_x_m`, `y_m − origin_y_m`; +x east, +y north. Raster cell centres are `((col+0.5)·dx, (row+0.5)·dy)` for the same `nx, ny, dx, dy`; arrays are `(ny, nx)` |
| `semi_major_m` = `semi_minor_m` | m, canopy radius = ½ · allometric diameter (circles) |
| `orientation_deg` | always 0 |
| `height_m` | m above the **visible canopy base**. `height_mode='exposed'` (default): `max(intact − burial, 0)`; the visible base is at `max(establishment_ground_m, current_ground_m)` — the current ground under partial burial, the **fixed establishment ground** under erosion (the intact height is unchanged; `exposure_depth_m` is the separate gap between the base and the current ground). A fully buried plant has `height_m = 0`, no visible canopy, and retains its establishment anchor. `height_mode='intact_diagnostic'`: intact allometric height, base at the **establishment ground**; labelled a DIAGNOSTIC APPROXIMATION |
| `porosity` | fraction [0, 1], **constant per-species catalog value** (uncalibrated fixture; unused by LEAF ecology; native MAPLE wind uses it; **no hydrological meaning is defined**) |
| `drag_coefficient` | dimensionless uniform pass-through (default 1.0), not a LEAF trait |

Sidecars (same order): `plant_id` (stable int64), `species`, `status`, `establishment_ground_m`,
`current_ground_m` (containing-cell sample of the current surface), `burial_depth_m`, `exposure_depth_m`
(both ≥ 0; `current − establishment` split by sign), `intact_height_m`, `exposed_height_m`, `root_radius_m`
(canopy radius × species lateral multiplier), `elevated_base`, `fully_buried`, `native_maple_valid`.

`native_maple_valid` (per plant) and `meta['native_maple_compatibility']` are **informational** — they
record whether the *native MAPLE object container* can hold the plant (base on the current surface,
positive exposed height). They are not a physical validity judgment of the shared view and SYRUP must not
filter on them.

Metadata (`meta`): `schema`, `schema_version`, `n_objects`, `fields`, `sidecar_fields`, `coordinates`,
`coordinate_frame`, `units`, `sidecar_units`, `domain` (with LEAF axis/edge/sampling conventions),
`catalog_hash`, `time_days`, `height_mode`, `height_definition`, `height_anchor`, `canopy_shape`,
`surface_reference`, `statuses_included`, exclusion counts, `n_fully_buried`, `n_elevated_base`,
`visibility_note`, `native_maple_compatibility`, `authority`, `porosity_note`, `drag_coefficient_note`,
`boundary_note` (finite, closed, non-periodic domain), `rasterization`, `consumers`.

**Refresh.** The view is a snapshot of `(PlantState, current surface)` at call time. There is no cache:
after `update_plants`, `apply_event_diagnostics`, `advance_time` with a driver capture, a new population,
or **any change of the surface** (erosion/deposition by MAPLE or SYRUP), call `canopy_view` again. Pair
views across consumers by `plant_id`, never by array position. Views from different states/surfaces must
not be mixed.

**Failures.** Missing current surface → `ValueError`; unknown `height_mode`/status → `ValueError`/
`StateError`; wrong surface shape, non-finite surface, unresolved SURFACE anchors, catalog hash/table
mismatch, non-finite geometry → `StateError`. Nothing is partially returned.

## 5. Consumers

* **MAPLE (wind)** — call `canopy_view` for the geometry. Because MAPLE's *current* `VegetationObjectSet`
  cannot represent fully buried or elevated-base objects, LEAF keeps the thin adapter
  `maple_vegetation_view(…, unsupported_policy='error'|'exclude'|'flag')` (schema
  `leaf.maple_vegetation_view` v2) which takes the shared view and applies only that MAPLE-specific policy
  plus compatibility metadata (`native_maple_valid`, counts, `derived_from`). Fields and sidecars are
  **exactly** the shared values by `plant_id`; exclusion uses one index selection over every array. The
  CLI `export-canopy` keeps writing this MAPLE-compatible projection. See `docs/maple-leaf-integration.md`.
* **SYRUP (water)** — call `canopy_view` directly. SYRUP does **not** inherit the MAPLE container
  restrictions: buried and elevated plants stay in the population with their diagnostics. Phase 0 defines
  **no** SYRUP adapter, roughness/interception/shading law or porosity interpretation; how SYRUP uses
  canopy radius, height or porosity is SYRUP-side qualification work, outside this contract. Water
  ownership is unchanged (SYRUP owns water and actual transpiration; LEAF owns plants).
* **Rasters.** LEAF publishes no raster. If an upstream model needs one, it derives it from these arrays
  with its own rasterizer under its own documented semantics and **keeps `plant_id` beside the raster** so
  results can be reported back per plant (`leaf.events.SurfaceEventDiagnostics`). LEAF's own cover
  measurement (`measure_cover`) is a finite, clipped, non-periodic sampled diagnostic, not a consumer raster.

## 6. Invariants the tests check (`tests/test_phase0.py::TestSharedCanopyView`)

* Buried and elevated-base plants are retained with correct diagnostics, `status` unchanged, zero exposed
  height when buried; the state is not mutated; invalid/unresolved inputs still fail.
* Every field/sidecar of every `maple_vegetation_view` policy equals the `canopy_view` value for the same
  `plant_id`; exclusions remove whole rows consistently; legacy verdicts and counts are preserved.
* The adapter obtains its arrays from `canopy_view` (one `derive_geometry` evaluation per call) and does
  not recompute geometry.

## 7. Not in this revision (future work)

Upstream adoption by MAPLE and SYRUP (both still have to call `canopy_view`), physics qualification of
porosity/height/radius for hydrology, any MAPLE container extension that would let `unsupported_policy`
become unnecessary, rasterization conventions in either model, periodic-boundary requalification, and all
ecological responses to burial/exposure. The CSV columns and row semantics, allometry conventions, resource bounds and
fixtures are unchanged; only the location of the file metadata moved (a plain CSV plus a companion `<stem>.metadata.json`,
no embedded metadata line).
