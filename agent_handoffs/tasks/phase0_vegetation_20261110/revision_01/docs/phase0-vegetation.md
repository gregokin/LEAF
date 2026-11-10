# LEAF Phase 0 — spatial individual-plant infrastructure

**Landscape Ecohydrology with Agent-based Flora (LEAF)**, Phase 0 (`leaf` 0.1.0).

Phase 0 initializes, tracks, records and exchanges spatially explicit individual plants. It works in
MAPLE *wind-with-prescribed-vegetation* mode without soil water, evapotranspiration, growth, CUDA, Numba,
MAPLE or SYRUP imports: `import leaf`, `leaf.state`, `leaf.initialization` and `python -m leaf` import only
the Python standard library and NumPy. No existing equation, bound, fixture default or lock of the legacy
coupled growth driver (`leaf.coupling`, `leaf.stepwat`, `benchmarks.*`) changes; those stay reachable
through lazy exports.

Ownership: MAPLE owns wind/shear physics, SYRUP owns water and actual transpiration, LEAF owns plants,
species, geometry and (later) ecological damage responses. Phase 0 invents **no** recruitment, mortality,
injury or damage physics; it ships species trait *sockets* and event diagnostics for later laws.

---

## 1. Species catalog (`leaf.traits`)

### 1.1 File schema (`leaf.species_catalog`, version 1)

```json
{
  "schema": "leaf.species_catalog", "schema_version": 1,
  "catalog_id": "optional string",
  "units": {"biomass": "kg", "length": "m", "time": "days"},
  "root_layers": 2,
  "species": {
    "grass": {
      "functional_type": "perennial_bunchgrass",
      "allometry": {"model": "power_law_v1",
                    "diameter": {"a": 0.9, "b": 0.45}, "height": {"a": 0.6, "b": 0.3}},
      "roots": {"lateral_radius_multiplier": 2.0, "layer_fractions": [0.8, 0.2]},
      "porosity": {"value": 0.2, "status": "uncalibrated constant ..."},
      "growth": {"law": "unimplemented", ...}, "water_use": {...}, "reproduction": {...},
      "life_expectancy": {"mean": 8, "unit": "years"},
      "mortality": {"law": "unimplemented", ...}, "disturbance": {"law": "unimplemented", ...}
    }
  }
}
```

* Species IDs are stable and **case-sensitive** (`[A-Za-z0-9_.-]{1,64}`, e.g. `grass`, `shrub`, `BOER`).
* `units` must be exactly `{"biomass": "kg", "length": "m", "time": "days"}` (kg **dry aboveground** mass per
  individual, metres, days). Other spellings fail.
* **Allometry** (`power_law_v1`): `diameter_m = a_d · B_kg^b_d`, `height_m = a_h · B_kg^b_h`, with
  `a > 0` (units `m·kg^-b`) and `b > 0`, hence monotone increasing and invertible. The generator uses the
  exact analytic inverse `B_kg = (d / a_d)^(1/b_d)`.
* **Roots**: `lateral_radius_multiplier > 0` (root circle radius = multiplier × canopy radius) and
  `layer_fractions` of length `root_layers` (a single L per catalog), finite, non-negative, summing to 1
  within `1e-9`. Never normalized silently. (The legacy driver additionally requires strictly positive
  fractions; `leaf.runtime` checks that when building legacy inputs.)
* **Porosity**: constant in `[0, 1]` with a mandatory `status` label. The examples choose `0.20` to match the
  current MAPLE default; it is **uncalibrated**. LEAF ecology never reads it; native MAPLE wind **does**.
* **Law sections** `growth`, `water_use`, `reproduction`, `mortality`, `disturbance`: `null` or an object
  with a non-empty string `"law"` (`"unimplemented"` for unselected laws). Contents are validated
  generically (finite numbers, no NaN/Infinity) and preserved verbatim on round trip
  (`SpeciesCatalog.section(id, name)`). `growth.law == "stepwat_weekly_aboveground_v0.1"` with a `traits`
  object can be extracted by `SpeciesCatalog.stepwat_traits()` for the opt-in legacy driver (validated there).
* **`life_expectancy`**: `null` or `{"mean": > 0, "unit": "days" | "years"}` → `life_expectancy_days(id)`.
  Metadata only: a *mean* is not a hard maximum and **no plant ever dies because its age reaches it**.
* Example numbers are fixtures, never calibrations.

### 1.2 Validation and hash

`SpeciesCatalog.from_file / from_text / (dict)` validates before anything is used: duplicate JSON keys,
`NaN`/`Infinity`, booleans in numeric fields, unknown schema version, wrong units, unknown keys, bad
allometry coefficients, inconsistent layer counts or fraction sums, porosity outside `[0, 1]`, bad time
units. Errors raise `CatalogError`.

`catalog.hash = "sha256:" + sha256(canonical JSON)` where the canonical form is the compact, key-sorted,
ASCII serialization of the parsed document. Whitespace and key order do not change the hash; numeric
spelling (`1` vs `1.0`) does. The hash is written into every plant CSV, every generation report and every
MAPLE view; reading a CSV with a different catalog fails.

---

## 2. Domain, plant state and CSV (`leaf.state`)

### 2.1 Domain

`Domain(nx, ny, dx_m, dy_m=None, origin_x_m=0.0, origin_y_m=0.0)`

* Raster axes are `(ny, nx)`: row = y (north), column = x (east). Cell centres are
  `origin + (col + 0.5)·dx`, `origin + (row + 0.5)·dy`. `dy_m` defaults to `dx_m` (non-square cells allowed).
* The domain is finite, **closed** `[x_min, x_max] × [y_min, y_max]` and **non-periodic**; positions outside
  it (including by 1e-6) are errors. No wrapping anywhere (placement, cover, sampling).
* Surface sampling convention **`containing_cell_closed_upper`**: a point samples the cell that contains
  it, `col = floor((x − origin_x)/dx)` clipped to `[0, nx−1]`, so the upper edge `x == x_max` belongs to the
  last column. No interpolation, no integer-cell rounding, no transpose. Points exactly on interior cell
  boundaries follow IEEE `floor` of the quotient (document and test with safe values).
* `check_field(a)` validates `(ny, nx)` numeric, finite fields; a transposed `(nx, ny)` field is refused
  with a hint.

### 2.2 Canonical population (`PlantState`)

Structure-of-arrays, O(P), one species name table (catalog order), stable `int64` IDs independent of row
order (`indices_of(ids)` uses a sorted index; unknown or duplicated query IDs raise).

| array | dtype | meaning |
|---|---|---|
| `ids` | int64 | unique plant ID |
| `species_index` | int32 | index into `species_names` |
| `x_m`, `y_m` | float64 | centre, metres east/north of the origin |
| `establishment_ground_m` | float64 | ground elevation at establishment (NaN only while unresolved) |
| `establishment_unresolved` | bool | SURFACE token not yet resolved |
| `agb_dry_kg` | float64 | **the** aboveground dry biomass state B |
| `seasonal_agb_dry_kg` | float64 | seasonal subcomponent E, `0 ≤ E ≤ B` (not an independent pool) |
| `age_days` | float64 | finite, ≥ 0 |
| `status` | int8 | 0 `live`, 1 `dead_standing`, 2 `removed` |

Scalars: `time_days` (model calendar), `catalog_hash`, `domain`, `provenance` (small dict).

Invariants (checked on construction, after every update, before every write): unique IDs, known species,
finite in-domain positions, finite anchors for resolved plants, `B > 0` for live/dead_standing, `B ≥ 0`
for removed, `0 ≤ E ≤ B`, `age ≥ 0`, known status. Removed/dead identities are retained forever (never
compacted). Empty populations are valid.

**Age convention.** `advance_time(days)` advances `time_days` and ages **live** plants by `days`. The caller
calls it once per elapsed interval of any kind (wind, water, transport, growth); ecological stepping is not
required. Dead/removed plants keep the age they had when their status changed (age at death stays fixed).
Exporting snapshots never touches ages, so repeated exports cannot age a plant twice. Reaching a species
life expectancy changes nothing.

**Status transitions.** `removed → anything` and `dead_standing → live` are refused. Removed plants may hold
zero mass; nothing else may.

**Updates are transactional.** `update_plants(ids, agb_dry_kg=…, seasonal_agb_dry_kg=…, status=…)` validates
a candidate copy and only then swaps arrays; unknown IDs, duplicates, `E > B`, non-positive live mass or
illegal transitions leave the state untouched.

### 2.3 SURFACE establishment anchors

An **initial** CSV may carry the literal token `SURFACE` in `establishment_ground_m`. Only that exact token
is an anchor request; `nan`, `NA`, `""`, `surface`, `inf` … are errors. `resolve_establishment(initial_surface)`
resolves every unresolved anchor **once** from the caller's initial `(ny, nx)` surface using the containing
cell and stores the concrete elevation; already resolved anchors are never touched, and later
erosion/deposition (a different *current* surface) never rewrites them. Unresolved states cannot be written
as non-initial snapshots, recorded by the cadence writer, or published to MAPLE. No establishment height
is needed.

### 2.4 Plant-state CSV (`leaf.plant_state_csv`, version 1)

One schema for initial input, evolving snapshots and final output:

```
# {"schema": "leaf.plant_state_csv", "schema_version": 1, "time_days": 0.0, "n_plants": 6, "catalog_hash": "sha256:…", "domain": {...}, "units": {...}, "status_codes": {...}, "establishment_resolved": false, "initial": true, "provenance": {...}}
plant_id,species,x_m,y_m,establishment_ground_m,agb_dry_kg,seasonal_agb_dry_kg,age_days,status
1,grass,2.5,3.5,SURFACE,0.12,0.0,730.0,live
```

* The metadata line is optional for handwritten initial lists (then the API/CLI supplies the domain and
  time). When present, the schema version, catalog hash, species table, units and row count are checked;
  a supplied domain/time must agree with it.
* Floats are written with `repr()` and parsed with a strict decimal regex, so full FP64 values round trip
  exactly (no rounding, no `1_000`, no `nan`/`inf`).
* Reads are all-or-nothing: wrong header, truncated/malformed rows, duplicate IDs, unknown species or
  status, non-finite or out-of-domain values, `E > B`, negative mass/age, catalog mismatch, bad versions
  all raise `StateError` before a state exists.
* Writes validate first, serialize to memory, then write atomically (temp file + fsync + link/replace).
  Existing files are refused unless `overwrite=True`.
* The file holds vegetation state only. It does **not** checkpoint soil water, RNG state, pending weekly
  uptake or other MASY/driver internals (see §7).

---

## 3. Geometry, MAPLE view and cover (`leaf.canopy`)

`derive_geometry(state, catalog, current_surface=None)` returns O(P) arrays for all plants: `plant_id`,
`species`, `status`, `x_m`, `y_m`, `diameter_m`, `radius_m` (circles, equal semi-axes), `height_m`
(**intact** allometric height above the establishment ground), `root_radius_m`, `establishment_ground_m`,
constant `porosity`. Geometry is derived on request from the canonical population; it is never an
editable plant authority. Canopies and roots may span many cells.

With `current_surface`: `surface_delta_m = current − establishment` (+ deposition, − erosion),
`burial_depth_m = max(delta, 0)`, `exposure_depth_m = max(−delta, 0)`, `exposed_height_m = max(height −
burial, 0)`, `fully_buried`. These are geometric diagnostics only — no dislodgement, injury or mortality.

`maple_vegetation_view(state, catalog, current_surface=None, statuses=('live','dead_standing'),
height_mode='intact'|'exposed', unsupported_policy='error'|'exclude'|'flag')` returns
`{'fields': {center_x_m, center_y_m, semi_major_m, semi_minor_m, orientation_deg, height_m, porosity,
drag_coefficient}, 'sidecar': {plant_id, species, status, establishment_ground_m, …}, 'meta': {...}}` with
**domain-local** coordinates (origin subtracted). Eroded anchors (canopy base above the current surface)
and, in `exposed` mode, fully buried plants cannot be represented by MAPLE's object set; the default policy
raises, `exclude` drops them (counted) and `flag` keeps them with sidecar flags. See
`docs/maple-leaf-integration.md`.

`measure_cover(domain, x, y, radius, group_index, n_groups, subsamples=4, max_cover_cells, max_cover_bytes)`
measures **per-group projected union ground cover clipped to the domain** with a deterministic boolean
midpoint raster of resolution `dx/subsamples × dy/subsamples`: a raster midpoint counts as covered when it
lies strictly inside any circle of the group. Same-group overlaps count once; cross-group overlaps count in
each group; a total union over all groups is also returned. Nominal cover (sum of unclipped crown areas /
domain area) is reported separately. This is a sampled approximation — never claimed to be the analytic
area — with per-circle relative error of order `resolution / radius` (reported as
`relative_error_scale`). Memory is one or two boolean rasters (`ny·s × nx·s` bytes each), checked against
the explicit bounds before allocation; no P×N or P×P arrays.

---

## 4. Generator (`leaf.initialization`)

Config (`leaf.generation_config`, version 1; see `examples/phase0/generation.json`):

| key | meaning |
|---|---|
| `domain` | `{nx, ny, dx_m[, dy_m, origin_x_m, origin_y_m]}` |
| `seed` | integer ≥ 0 → `numpy.random.default_rng(seed)` (PCG64) |
| `overlap` | `allowed` (default) or `disallowed` (global cross-species canopy exclusion; roots may overlap) |
| `cover_definition` | `per_species_union` (the only Phase 0 definition) |
| `cover_tolerance` | absolute acceptance tolerance on measured cover (default 0.005; `null` = accept the first estimate) |
| `cover_subsamples` | raster resolution `dx / s` (default 4) |
| `max_iterations` | count-fit iterations (default 12) |
| `limits` | `max_plants`, `max_attempts`, `max_cover_cells`, `max_cover_bytes` |
| `species.<id>` | `target_cover ∈ [0, 1)`, `diameter_mean_m > 0`, `diameter_std_m ≥ 0`, `initial_age_days` (number or `{distribution: constant|uniform, …}`) |
| `time_days` | model time of the generated state |

Algorithm:

1. Diameters follow a lognormal with exactly the requested population mean and SD
   (`σ² = ln(1 + s²/m²)`, `μ = ln m − σ²/2`); `std = 0` is constant. Sample statistics vary and are reported.
2. Biomass of each plant = species inverse allometry of its sampled diameter.
3. Expected crown area `E[A_c] = π/4 · E[d²]` with **`E[d²] = mean² + std²`**. Starting counts:
   overlap allowed `N₀ = −A ln(1 − f) / E[A_c]` (Poisson-Boolean), disallowed `N₀ = A f / E[A_c]`.
4. Counts are integers, fitted to the **measured** per-species union cover:
   `N ← N · ln(1−f)/ln(1−c)` (overlap) or `N · f/c` (exclusion), adding newly sampled individuals or
   removing the most recently added ones, until `|c − f| ≤ cover_tolerance` or `max_iterations` is
   exhausted (then `GenerationError` with the full history; nothing is returned). Edge clipping and
   same-species overlaps are thereby accounted for by measurement. Exact continuous cover is not promised.
5. Placement. Overlap allowed: independent uniform centres **conditioned on the fitted count and the cover
   acceptance** (not an unconditioned Poisson process). Overlap disallowed: sequential uniform proposals
   with global canopy exclusion (`|Δ| < r_i + r_j` rejects), species randomly interleaved
   (`ordering_convention: random_interleave`), using sparse hashed bins (never all pairs), bounded by
   `max_attempts`.
6. Bounds (`max_plants`, raster cells/bytes) are checked before sampling/allocation. Unattainable requests
   raise; diameters are never altered, individuals never silently dropped, exclusion never relaxed.
7. IDs are `1..P` in placement order; establishment anchors are `SURFACE` (unresolved) unless an initial
   surface is supplied; status `live`; `seasonal_agb_dry_kg = 0`.

The report lists targets, measured union cover, nominal crown fraction, counts, initial estimates, history,
sample diameter/age statistics, measurement method/resolution/error basis, tolerance, seed/RNG label,
attempts and bounds; a compact subset goes into `state.provenance`.

**Scaling.** Cost is O(P) Python-level work per measurement (bounding boxes of a few raster cells for
grass) plus one raster of `ny·s × nx·s` bytes. For ~60 k individuals on a 512² grid with `s = 4`
(2048² = 4 MB raster) a measurement is a few seconds of pure NumPy; denser grass populations scale linearly
in P, and raster memory scales with `s²·N`. Lower `cover_subsamples` (coarser, still unbiased on average
for random placement) or raise the explicit bounds when needed. No compiled acceleration is required or
shipped.

---

## 5. Recording (`leaf.recording.SnapshotWriter`)

```python
w = SnapshotWriter('out/veg_{index:04d}_{time_days:.1f}.csv', cadence_days=30.0)
w.write_initial(pop)              # explicit initial snapshot; fixes the cadence origin at pop.time_days
...; pop.advance_time(dt)         # the caller owns the clock (wind, water, growth, event intervals alike)
w.maybe_write(pop)                # writes ONE snapshot only when a due time has been reached
w.write_final(pop)                # explicit final snapshot
w.write_manifest('out/manifest.json')
```

* Due times are `start + k·cadence` (integer k). `maybe_write` writes only when due; nothing is written
  daily/weekly by default. Without a cadence only explicit initial/final snapshots exist.
* If the caller skips past several due times, **one** snapshot of the current state is written and the
  missed due times are listed (`skipped_due_times_days`) in the record and manifest. Missing historical
  states are never fabricated or interpolated.
* Validation and the atomic write happen before any bookkeeping changes; failures (unresolved anchors,
  existing file, invalid state, non-monotone time) leave the schedule and existing files untouched.
* No P-sized history is held in memory (records are small dicts).

---

## 6. Event diagnostics (`leaf.events`)

`SurfaceEventDiagnostics(plant_id, time_start_days, time_end_days, source, erosion_depth_m=…,
deposition_depth_m=…, abrasion_dose=…, abrasion_unit=…, transport_flux=…, flux_unit=…)` (schema
`leaf.surface_event_diagnostics` v1, JSON via `to_dict/from_dict`):

* `source ∈ {maple_wind, syrup_water, other}`; times in days with `end ≥ start`.
* Depths are non-negative metres: `erosion_depth_m` (surface lowering) and `deposition_depth_m` (raising /
  burial) are distinct; `net_surface_delta_m = deposition − erosion` (+ raised, − lowered).
* `abrasion_dose` and `transport_flux` are optional, non-negative, and require explicit unit strings.
  Flux is never converted into abrasion or injury.
* Unknown or duplicate IDs, bad units/times/values fail before any mutation.

`apply_event_diagnostics(state, event, handler=None, require_elapsed=True)` resolves IDs and returns
diagnostics. The default is **no ecological response**. A future species law may be passed as
`handler(view, event, indices) -> {'agb_dry_kg': …, 'seasonal_agb_dry_kg': …, 'status': …} | None`; it sees
a read-only copy, its proposals are validated on a candidate copy and applied all-or-nothing, and any
exception leaves the vegetation untouched. No threshold mortality is ever invented. By default the event
interval must already have elapsed on the population clock (advance time first).

---

## 7. Bridge to the existing growth drivers (`leaf.runtime`)

* `capture_snapshot(population, sim)` copies the current per-plant B (and STEP seasonal E when present) out
  of a `benchmarks.daily_water_growth.Simulation` or `leaf.coupling.CoupledSimulation` **by stable ID**,
  preserving the population's species, anchors and ages. Missing live plants, unknown/duplicate driver IDs,
  non-live population plants in the driver and mismatched profile assignments are refused before any
  change. Call it only when a snapshot is requested (writer cadence), not daily.
* `legacy_inputs_from_state(population, catalog, biomass_policy='refuse'|'accept_driver_recompute', …)`
  builds the driver's `inputs` (x, y, canopy_radius, root_radius, ids, root_profile_id) for the live plants.
  The legacy driver recomputes initial biomass from canopy area (`B = π r²·1 kg m⁻²`, or
  `π r²·canopy_biomass_density_kg_m2` in STEP mode), uses one `root_multiplier` and strictly positive layer
  fractions, a `[0, W]×[0, H]` square-cell grid and a separate pending-water/calendar state. A LEAF power
  allometry generally disagrees with the area law, so the bridge **refuses by default** and only proceeds
  when the mismatch is acknowledged explicitly. This is vegetation **state replay/handoff**, not a full
  coupled restart (soil water, pending uptake, STEP period clock and RNG are not restored). The driver's
  science is unchanged. Static MAPLE initialization (§3) is unrestricted by that legacy law.

---

## 8. CLI

```
python -m leaf validate-species --species species.json
python -m leaf init     --species S --plants plants.csv --nx 20 --ny 20 --dx-m 1 --surface-constant 100 --out veg0.csv
python -m leaf generate --species S --config generation.json --surface-constant 12 --out gen.csv --report gen.json
python -m leaf inspect  --species S --plants gen.csv [--surface current.npy] [--cover-subsamples 8]
python -m leaf export-canopy --species S --plants gen.csv --out view.json|view.npz [--height-mode exposed --surface current.npy --unsupported exclude]
```

Surfaces are `.npy` (`np.load(allow_pickle=False)`, shape `(ny, nx)`) or a constant. Every command writes
atomically, refuses existing outputs without `--overwrite`, exits 1 on errors (2 on usage errors) and
never runs hydrology or growth.

---

## 9. Reproducibility and examples

* Generation is reproducible from `seed + config + catalog hash` (RNG draw order is documented in the
  report: per batch diameters, x, y, ages; exclusion mode draws per proposal).
* `examples/phase0/species.json` (two fixture species, 2 root layers, porosity 0.20, allometry, growth /
  water-use / reproduction / life-expectancy / mortality / disturbance sections),
  `examples/phase0/plants.csv` (SURFACE and explicit anchors, ages, dead and removed rows) and
  `examples/phase0/generation.json` (cover-driven two-species generation). All numbers are fixtures.

## 10. Limitations (Phase 0)

No growth, water use, recruitment, mortality, injury, root dislodgement, soil-layer remapping or
calibration. Cover is sampled, not analytic. The CSV is vegetation state only. The legacy driver's
area-proportional initial biomass law restricts replay into it (§7).
