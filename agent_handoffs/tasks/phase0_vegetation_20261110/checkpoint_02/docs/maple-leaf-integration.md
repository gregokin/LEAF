# MAPLE ↔ LEAF integration handoff (Phase 0)

For the Codex-Claude MAPLE developers. **No MAPLE change is part of this task**; this document states what
LEAF provides and how MAPLE should consume it. LEAF never imports MAPLE or SYRUP.

## 1. Three vegetation modes

| mode | vegetation source | LEAF calls | notes |
|---|---|---|---|
| no vegetation | none | none | unchanged MAPLE |
| **prescribed LEAF vegetation** (wind-only, Phase 0) | LEAF population (CSV or generator), static or updated only by calendar/event bookkeeping | `read_plant_csv` / `generate_population`, `resolve_establishment`, `maple_vegetation_view`, `advance_time`, `SnapshotWriter`, `events.*` | no soil water, ET, growth, numba, cupy |
| fully coupled (later) | LEAF population exchanged with the growth/water drivers | adds `leaf.runtime.capture_snapshot` / `legacy_inputs_from_state` | wind, water and growth/T occupy exclusive intervals; not in scope now |

In all vegetation modes MAPLE should call the **shared LEAF loading/placement/allometry** code instead of
re-implementing it: species catalog (`leaf.traits.SpeciesCatalog`), plant list / generator
(`leaf.state`, `leaf.initialization`), geometry and view (`leaf.canopy`).

## 2. Canonical handoff sequence (prescribed vegetation)

```python
from leaf import traits, state, canopy
cat = traits.SpeciesCatalog.from_file('species.json')
pop = state.read_plant_csv('plants.csv', cat, domain=state.Domain(nx, ny, dx_m, dy_m))   # or generated
pop.resolve_establishment(initial_surface)        # (ny, nx) elevation; resolves SURFACE anchors ONCE
view = canopy.maple_vegetation_view(pop, cat)     # intact allometric heights, domain-local metres
objects = VegetationObjectSet(**view['fields'])   # MAPLE side: its existing rasterizer rasterizes them
ids = view['sidecar']['plant_id']                 # keep beside the raster for feedback by ID
```

* **Fields** (`view['fields']`, one float64 array each, same order): `center_x_m`, `center_y_m`,
  `semi_major_m`, `semi_minor_m` (equal: circles), `orientation_deg` (0), `height_m`, `porosity`,
  `drag_coefficient` (uniform pass-through metadata, default 1.0).
* **Coordinates**: domain-local metres, `+x` east, `+y` north, `center = plant position − domain origin`;
  MAPLE cell centres are `((col + 0.5)·dx, (row + 0.5)·dy)` for the same `nx, ny, dx, dy`. LEAF arrays are
  `(ny, nx)`; the view's `meta['domain']` carries `nx, ny, dx_m, dy_m, origin`.
* **Surface code resolution**: the CSV token `SURFACE` is resolved by LEAF from MAPLE's *initial* surface
  with the containing-cell rule (`col = floor((x − x0)/dx)` clipped, upper edge in the last cell). After
  that the establishment elevation is fixed; it is never rewritten by erosion/deposition.
* **Species file hash**: `view['meta']['catalog_hash']` (also in every CSV). Log it with the run; LEAF
  refuses to pair a state with a different catalog.
* **Rasterization**: use MAPLE's existing rasterizer; LEAF does not copy it and does not define raster
  semantics. Keep `sidecar['plant_id']` (and `species`, `status`) beside the raster so wind results can be
  reported back per plant.
* **Porosity** is a constant per-species catalog value, unused by LEAF ecology; native MAPLE wind uses it.
  The example value 0.20 matches the current MAPLE default and is uncalibrated.
* Statuses `live` and `dead_standing` are published by default (both exert drag); `removed` plants are
  retained in the LEAF state but never published.

## 3. Heights, burial and elevated bases

`height_m` in the default view is the **intact** allometric height above the establishment ground. MAPLE's
object set places every canopy base at the current surface, so two situations are not representable:

1. **Eroded anchor** (current surface below establishment): the real canopy base is elevated above the
   surface. LEAF detects it when `current_surface` is supplied (`exposure_depth_m > 0`).
2. **Fully buried plant** in `height_mode='exposed'` (burial ≥ height).

`maple_vegetation_view(..., current_surface=cur, height_mode='intact'|'exposed', unsupported_policy=...)`:

* `'error'` (default) — refuse the whole publication with the offending IDs;
* `'exclude'` — drop those plants (count reported in `meta['n_unsupported_excluded']`);
* `'flag'` — publish them with the documented value (intact or exposed height, base at the surface) and set
  `sidecar['elevated_base']` / `sidecar['fully_buried']`.

MAPLE must choose a policy explicitly; nothing is silently flattened. `height_mode='exposed'` publishes
`height − burial_depth` (partially buried plants shrink); elevated bases still require a policy. Burial /
exposure diagnostics never imply dislodgement or mortality.

## 4. Calendar and snapshots

LEAF ages live plants only through `pop.advance_time(days)`; MAPLE (or the orchestrator) calls it once per
elapsed interval, including wind/transport intervals, so wind-only runs still age plants. Snapshots are
written only when due (`SnapshotWriter`), plus explicit initial/final; skipped due times are listed, never
fabricated. Snapshot CSVs carry the catalog hash, domain, time and provenance.

## 5. Event feedback (MAPLE wind / shared surface, SYRUP water)

Schema `leaf.surface_event_diagnostics` v1 (`leaf.events.SurfaceEventDiagnostics`, JSON via `to_dict`):

```json
{"schema": "leaf.surface_event_diagnostics", "schema_version": 1, "source": "maple_wind",
 "time_unit": "days", "time_start_days": 120.0, "time_end_days": 121.0, "depth_unit": "m",
 "plant_id": [17, 42], "erosion_depth_m": [0.01, 0.0], "deposition_depth_m": [0.0, 0.08],
 "abrasion_dose": [12.5, 0.4], "abrasion_unit": "kg m^-1",
 "transport_flux": [0.3, 0.01], "flux_unit": "kg m^-1 s^-1"}
```

* Keyed by stable `plant_id`; depths/doses are ≥ 0; surface **lowering** (`erosion_depth_m`), **raising**
  (`deposition_depth_m`) and **flux** are distinct; `net_surface_delta_m = deposition − erosion`.
  LEAF never converts flux into abrasion or injury.
* `apply_event_diagnostics(pop, event)` validates (unknown/duplicate IDs, units, times) **before** any
  mutation and, by default, records diagnostics only. Later species laws plug into the `handler` socket;
  proposals are applied all-or-nothing and a failing handler changes nothing. No threshold mortality.
* MAPLE keeps authority over the surface; LEAF performs no surface remapping. Supply the *current* surface
  to `derive_geometry` / `maple_vegetation_view` when burial/exposure diagnostics or exposed heights are
  needed.

## 6. What the fully coupled mode will add (not now)

Vegetation state replay into the existing growth driver via `leaf.runtime.legacy_inputs_from_state`
(restricted by that driver's area-proportional initial biomass law; refused unless acknowledged) and
`capture_snapshot` back by ID. Full coupled restart (soil water, pending uptake, STEP clock, RNG) is not
provided by the CSV. Wind, water and growth/T remain exclusive intervals.
