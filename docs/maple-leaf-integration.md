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

## 0. Phase 3 addendum — newborn plants (reproduction) for MAPLE / SYRUP developers

LEAF Phase 3 can ADD plants between storms (`IntereventSession.apply_reproduction`, `docs/phase3-reproduction.md`). No MAPLE or
SYRUP change is made or required, but consumers must not assume a fixed plant list:

* **IDs are stable and append-only.** Existing plants keep ID, X/Y, establishment ground, age and status (dead_standing and
  removed records persist, never compacted or renumbered). New plants get fresh int64 IDs above max(existing IDs, 0); IDs
  of any sign (zero, negative, the endpoints) may already exist. Key per-plant feedback by ID, never by row.
* **Re-export the canopy after the boundary.** Call `session.flush_growth()` (no pending growth is required anyway) and then
  `session.export_canopy_view(current_surface)` / `canopy.canopy_view(session.snapshot(), catalog, current_surface)` for the
  next MAPLE wind or SYRUP interval; newborn plants (age 0, small dry AGB from the species allometry) and debited donor
  parents appear through the SAME shared allometry. Canopies may overlap (parents / daughters / neighbours): the shared view
  already permits overlap; nothing is clipped or separated by LEAF.
* **Water.** Newborn plants request water only from the NEXT interevent interval; the reproduction call itself changes no
  native soil water, time or rain / ET. Seedling AGB is an external credit, not water-derived biomass.
* **Old season first.** If a year end passed inside a transport / closed gate, call `session.flush_deferred_season()` before
  `apply_reproduction` (otherwise refused): newborns never pay the previous season's turnover.
* **Event decisions first.** `apply_reproduction` is refused while a transport interval still awaits its
  `apply_surface_event` decision, with pending growth, or with an unresolved annual mortality decision: run those first.
* **Checkpoints** are layout version 4 (plain `plants.csv` + companion `plants.metadata.json`; version 3 embedded the metadata in
  the CSV and is refused, not migrated). The plant CSV columns and rows (initial / snapshots / final / checkpoint) are unchanged;
  see section 2 for where the file metadata lives.

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

**Plant files are plain CSV (project rule).** LEAF writes no metadata inside a plant CSV. The metadata of every saved
plant file (time, domain, species-catalog hash and table, row count, units, initial / `SURFACE` declaration, provenance and
the CSV's SHA-256) is the companion `<stem>.metadata.json` beside it (`plants.csv` → `plants.metadata.json`;
`docs/phase0-vegetation.md` §2.3). `read_plant_csv` loads and verifies the companion automatically and refuses a CSV that does
not match it; a handwritten plain list needs no companion (pass the `domain` as above, optionally `time_days`). MAPLE-side
code that reads or copies LEAF plant files must carry the pair together. Old files with an embedded `# {json}` first line are
still readable, but are never written.

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
  last cell; on a PERIODIC axis the exact upper endpoint `x == x_max` is the first cell). After that the establishment elevation is fixed and saved; later surfaces never rewrite it.
* **Species file hash and table**: `view['meta']['catalog_hash']` (also in every plant file's companion
  `<stem>.metadata.json`). LEAF refuses to pair a state with a different catalog hash or a different (sorted) species table.
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

## 4. Boundaries: finite clipped domain (default) or per-axis periodic domain

A `leaf.state.Domain` is **finite, closed and non-periodic by default**: `measure_cover` and exclusion placement clip
canopies at the edge and the edge is a hard boundary for overlap. Each axis can instead be declared **periodic**
(`Domain(nx, ny, dx_m, ..., periodic=(True, True))`; see `docs/periodic-boundaries.md`). For a periodic axis LEAF itself
wraps its canopy cover, exclusion placement, canopy-area PET weights and root footprints across the seam, and the shared
view carries the declaration in `meta['domain']['periodic']` / `meta['boundary']`. If a MAPLE raster interprets the
boundary periodically while LEAF's domain is nonperiodic, cover and cross-boundary overlap differ and nothing may be
claimed about them: the two sides must use the SAME declaration.

### 4.1 Periodic handoff (MAPLE developers — nothing here changes MAPLE or SYRUP)

**Required adapter step (upstream, not done here).** `src/maple/case_tools/importers/leaf_vegetation.py::leaf_domain` builds the LEAF
`Domain` without flags and `vegetation_from_leaf_population` compares `Domain` equality (periodicity is part of equality). Derive the strict
`(x, y)` boolean flags from the `GeometrySpec` boundary kinds and pass `periodic=(px, py)` through that helper, the imports and the population
comparisons; otherwise LEAF periodicity is dropped or rejected. No wind / rasterizer work is needed.

MAPLE already wraps LEAF-imported canopy footprints and shelter across periodic seams, and a shrub at `x = 63.8, r = 0.5`
on a 64 m domain remains ONE plant (one row, one `plant_id`) whose footprint appears on both sides. LEAF therefore exports
**one object per plant with its stored centre inside the domain** — no image rows, no coordinate shift. Public periodic centers are
canonical half-open `[0, L)`: a stored center exactly at the upper endpoint (the lower edge's alias) is exported as 0, because MAPLE's
object container refuses upper centers; the stored LEAF record is unchanged:

* generate or declare the population on the FULL periodic domain: `Domain(nx, ny, dx_m, periodic=(True, True))` (or the
  `"periodic": [true, true]` key of the generation / domain JSON). **Do not inset the placement** (the old wrapper placed 60 x 60
  inside 64 x 64 — a 2 m inset — because LEAF's exclusion was finite) and **do not apply an inner-area cover correction**: centres are
  sampled over the whole domain and the measured cover is the periodic union;
* one plant = one CSV row = one `plant_id`; `canopy_view` / `maple_vegetation_view` keep that (`meta['boundary']
  ['one_record_per_plant']`); MAPLE's rasterizer wraps the footprint;
* the same Domain flags must reach every LEAF consumer (MASY's common boundary config): LEAF rebuilds of roots and canopy
  weights (growth, season, reproduction, restart) read them from the state's `Domain`; a plant file / checkpoint whose domain
  flags differ is refused, and a source-hash difference needs the explicit override of `docs/masy-daily-plant-interface.md`;
* when MAPLE chooses to regenerate a case, change ONLY the domain declaration and drop the wrapper inset; historical
  populations are not rewritten here and the existing finite Jornada case is unchanged.

MAPLE's old finite-domain populations stay valid under a nonperiodic `Domain`; their reading is unchanged.

## 5. Calendar and snapshots

LEAF ages live plants only through `pop.advance_time(days)`; MAPLE (or the orchestrator) calls it once per
elapsed interval, including wind/transport intervals, so wind-only runs still age plants. Snapshots are
written only when due (`SnapshotWriter`), plus explicit initial/final; every missed due time is listed,
never fabricated; all bookkeeping is committed only after a successful write of BOTH files. Each snapshot is a
plain CSV plus its `<stem>.metadata.json` companion, which carries the catalog hash, species table, domain (with its
conventions), time, provenance and the CSV SHA-256 (the CSV itself has no metadata line).

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
