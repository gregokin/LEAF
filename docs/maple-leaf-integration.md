# MAPLE ↔ LEAF integration handoff (Phase 0, revision 3 — shared canopy view)

For the Codex-Claude MAPLE developers. **No MAPLE change is part of this task**; this document states what
LEAF provides and how MAPLE should consume it. LEAF never imports MAPLE or SYRUP.

**Primary shared API: `leaf.canopy.canopy_view`** (`docs/shared-canopy.md`) — the one aboveground
representation for MAPLE and SYRUP. Common code yields consistent geometry when both consumers use the
**same plant state, catalog and current surface** and refresh at matching coupling times; upstream adoption
by either model is still pending and no integrated run is claimed.
This guide describes the **thin MAPLE adapter** `maple_vegetation_view` on top of it. The adapter exists
because MAPLE's *current* `VegetationObjectSet` cannot hold fully buried or elevated-base objects; those
restrictions are MAPLE-specific and are not part of the shared view.

## 1. Three vegetation modes

| mode | vegetation source | LEAF calls | notes |
|---|---|---|---|
| no vegetation | none | none | unchanged MAPLE |
| **prescribed LEAF vegetation** (wind-only, Phase 0) | LEAF population (CSV or generator), static or updated only by calendar/event bookkeeping | `read_plant_csv` / `generate_population`, `resolve_establishment`, `canopy_view(state, catalog, current_surface)` → `maple_vegetation_view(...)` for the native object set, `advance_time`, `SnapshotWriter`, `events.*` | no soil water, ET, growth, numba, cupy |
| fully coupled (later) | LEAF population exchanged with the growth/water drivers | adds `leaf.runtime.capture_snapshot` / `legacy_inputs_from_state` | wind, water and growth/T occupy exclusive intervals; not in scope now |

In all vegetation modes MAPLE should call the **shared LEAF loading/placement/allometry** code instead of
re-implementing it: species catalog (`leaf.traits.SpeciesCatalog`), plant list / generator
(`leaf.state`, `leaf.initialization`), geometry and views (`leaf.canopy`: `derive_geometry` is the single
allometry implementation on the canopy export path; `canopy_view` the single canopy representation).

## 2. Canonical handoff sequence (prescribed vegetation)

```python
from leaf import traits, state, canopy
cat = traits.SpeciesCatalog.from_file('species.json')
pop = state.read_plant_csv('plants.csv', cat, domain=state.Domain(nx, ny, dx_m, dy_m))   # or generated
pop.resolve_establishment(initial_surface)            # (ny, nx) elevation; resolves SURFACE anchors ONCE
shared = canopy.canopy_view(pop, cat, current_surface)  # the SHARED geometry (same call SYRUP makes)
view = canopy.maple_vegetation_view(pop, cat, current_surface)   # MAPLE projection of the same geometry;
assert view['meta']['native_maple_valid']                       # current surface REQUIRED (== initial at t0)
objects = VegetationObjectSet(**view['fields'])       # MAPLE side: its existing rasterizer rasterizes them
ids = view['sidecar']['plant_id']                     # keep beside the raster for feedback by ID
```

`view['fields']` / `view['sidecar']` are **exactly** `shared['fields']` / `shared['sidecar']` for the same
`plant_id` (the adapter adds no geometry); `view['meta']['derived_from']` names the shared schema. The
`export-canopy` CLI keeps writing this MAPLE-compatible projection.

* **Fields** (`view['fields']`, one float64 array each, same order): `center_x_m`, `center_y_m`,
  `semi_major_m`, `semi_minor_m` (equal: circles), `orientation_deg` (0), `height_m`, `porosity`,
  `drag_coefficient` (uniform pass-through metadata, default 1.0).
* **Coordinates**: LEAF stores landscape-frame positions; the view publishes domain-local metres
  `center = position − domain origin`, `+x` east, `+y` north; MAPLE cell centres are
  `((col + 0.5)·dx, (row + 0.5)·dy)` for the same `nx, ny, dx, dy`. LEAF arrays are `(ny, nx)`.
* **Surface code resolution**: the CSV token `SURFACE` (initial files only) is resolved by LEAF from MAPLE's
  *initial* surface with the containing-cell rule (`col = floor((x − x0)/dx)` clipped; upper edge in the
  last cell). After that the establishment elevation is fixed and saved; later surfaces never rewrite it.
* **Species file hash and table**: `view['meta']['catalog_hash']` (also in every CSV). LEAF refuses to
  pair a state with a different catalog hash or a different (sorted) species table.
* **Rasterization**: use MAPLE's existing rasterizer; LEAF does not copy it and does not define raster
  semantics. Keep `sidecar['plant_id']` (and `species`, `status`) beside the raster so wind results can be
  reported back per plant.
* **Porosity** is a constant per-species catalog value, unused by LEAF ecology; native MAPLE wind uses it.
  The example value 0.20 matches the current MAPLE default and is uncalibrated.
* Statuses `live` and `dead_standing` are published by default (both exert drag); `removed` plants are
  retained in the LEAF state and **excluded by default** (an explicit `statuses` selection can include a
  removed record with a positive diameter).

## 3. Heights, burial and elevated bases (the thin adapter's only job)

MAPLE's object set places every canopy base on the **current** surface. The shared `canopy_view` already
requires the current surface, compares it with each plant's establishment anchor and publishes exposed
heights; it **retains** buried and elevated-base plants with diagnostics. `maple_vegetation_view` takes that
view and applies only the native-MAPLE container policy:

* default `height_mode='exposed'`: `height_m = intact allometric height − burial depth` (partially buried
  plants shrink; nothing reappears at full height on deposited soil) — computed in `canopy_view`;
* **unsupported by the native container** (a MAPLE limitation, not a physical verdict): eroded anchors
  (canopy base above the surface, `exposure_depth_m > 0`) and fully buried plants (exposed height ≤ 0);
* `unsupported_policy='error'` (default) refuses the whole publication with the offending IDs;
  `'exclude'` drops them (`meta['n_unsupported_excluded']`) using **one index selection over every field
  and sidecar array**; `'flag'` keeps them with `sidecar['native_maple_valid'] = False` and
  `meta['native_maple_valid'] = False`;
* `height_mode='intact_diagnostic'` ignores burial and is labelled a diagnostic approximation; it is
  **never** `native_maple_valid`. MAPLE must check `meta['native_maple_valid']` before feeding native wind.

Burial/exposure diagnostics never imply dislodgement or mortality; `status` alone carries death/removal.
If MAPLE's container is later extended, the adapter policy can be relaxed without touching the shared view.

## 4. Boundaries: finite clipped cover vs a periodic raster

LEAF's cover (`measure_cover`) and exclusion placement use a **finite, closed, non-periodic** domain:
canopies are clipped at the edge for cover and the edge is a hard boundary for overlap. If a MAPLE raster
interprets the boundary periodically, both the cover of edge plants and cross-boundary overlaps differ.
The future adapter must either match LEAF's finite semantics or requalify cover/non-overlap under the
periodic interpretation; it must not claim exact cover or guaranteed non-overlap across incompatible edges.

## 5. Calendar and snapshots

LEAF ages live plants only through `pop.advance_time(days)`; MAPLE (or the orchestrator) calls it once per
elapsed interval, including wind/transport intervals, so wind-only runs still age plants. Snapshots are
written only when due (`SnapshotWriter`), plus explicit initial/final; every missed due time is listed,
never fabricated; all bookkeeping is committed only after a successful write. Snapshot CSVs carry the
catalog hash, species table, domain (with its conventions), time and provenance.

## 6. Event feedback (MAPLE wind / shared surface, SYRUP water)

Schema `leaf.surface_event_diagnostics` v1 (`leaf.events.SurfaceEventDiagnostics`, JSON via `to_dict`):

```json
{"schema": "leaf.surface_event_diagnostics", "schema_version": 1, "source": "maple_wind",
 "time_unit": "days", "time_start_days": 120.0, "time_end_days": 121.0, "depth_unit": "m",
 "plant_id": [17, 42], "erosion_depth_m": [0.01, 0.0], "deposition_depth_m": [0.0, 0.08],
 "abrasion_dose": [12.5, 0.4], "abrasion_unit": "kg m^-1",
 "transport_flux": [0.3, 0.01], "flux_unit": "kg m^-1 s^-1"}
```

* Keyed by stable `plant_id`; depths/doses ≥ 0; surface **lowering**, **raising** and **flux** are distinct;
  `net_surface_delta_m = deposition − erosion`. LEAF never converts flux into abrasion or injury.
* Records are immutable after validation; `apply_event_diagnostics(pop, event)` validates (unknown/duplicate
  IDs, units, times) **before** any mutation and, by default, records diagnostics only. Later species laws
  plug into the `handler` socket; proposals are applied all-or-nothing against the IDs resolved before the
  handler ran, and a failing or misbehaving handler changes nothing. No threshold mortality.
* MAPLE keeps authority over the surface; LEAF performs no surface remapping. Supply the *current* surface
  to `canopy_view` / `maple_vegetation_view` (or `derive_geometry`) for burial/exposure diagnostics and
  exposed heights, and call them again after every surface change (the views are uncached snapshots).

## 7. What the fully coupled mode will add (not now)

Vegetation state replay into the existing growth driver via `leaf.runtime.legacy_inputs_from_state`
(restricted to species whose diameter law is the driver's square-root area law; otherwise refused unless the
approximation is acknowledged) and `capture_snapshot` back by ID at requested output. While attached, the
driver's B/E are the numerical authority; Phase 0 event proposals do not reach it. Full coupled restart (soil
water, pending uptake, STEP clock, RNG) is not provided by the CSV. Wind, water and growth/T remain
exclusive intervals.
