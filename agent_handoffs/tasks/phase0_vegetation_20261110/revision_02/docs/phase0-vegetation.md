# LEAF Phase 0 — spatial individual-plant infrastructure (revision 2)

**Landscape Ecohydrology with Agent-based Flora (LEAF)**, Phase 0 (`leaf` 0.1.0).

Phase 0 initializes, tracks, records and exchanges spatially explicit individual plants. It works in
MAPLE *wind-with-prescribed-vegetation* mode without soil water, evapotranspiration, growth, CUDA, Numba,
MAPLE or SYRUP imports: `import leaf`, `leaf.state`, `leaf.initialization` and `python -m leaf` import only
the Python standard library and NumPy. No existing equation, bound, fixture default or lock of the legacy
coupled growth driver (`leaf.coupling`, `leaf.stepwat`, `benchmarks.*`) changes; those stay reachable
through lazy exports (`from leaf import CoupledSimulation, CoupledPhaseError, stepwat`).

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
  individual, metres, days).
* **Allometry** (`power_law_v1`): `diameter_m = a_d · B_kg^b_d`, `height_m = a_h · B_kg^b_h`, `a > 0`
  (units `m·kg^-b`), `b > 0`: monotone and invertible; the generator uses the exact inverse
  `B_kg = (d / a_d)^(1/b_d)`. Inputs must be real numeric (bool/complex/str refused); **non-finite outputs**
  (a finite mass with a large exponent can overflow) are refused with `CatalogError`.
* **Roots**: `lateral_radius_multiplier > 0`; `layer_fractions` of length `root_layers`, finite,
  non-negative, summing to 1 within `1e-9`; never normalized silently.
* **Porosity**: constant in `[0, 1]` with a mandatory `status` label; the examples use 0.20 (matches the
  current MAPLE default; **uncalibrated**). LEAF ecology never reads it; native MAPLE wind **does**.
* **Law sections** `growth`, `water_use`, `reproduction`, `mortality`, `disturbance`: `null` or an object
  with a non-empty string `"law"` (`"unimplemented"` for unselected laws); validated generically and
  preserved verbatim (`SpeciesCatalog.section(id, name)` returns a copy).
  `growth.law == "stepwat_weekly_aboveground_v0.1"` + `traits` → `SpeciesCatalog.stepwat_traits()`.
* **`life_expectancy`**: `null` or `{"mean": > 0, "unit": "days" | "years"}` → `life_expectancy_days(id)`.
  Metadata only: **no plant ever dies because its age reaches it** (mean ≠ hard maximum).
* Example numbers are fixtures, never calibrations.

### 1.2 Validation, ownership, hash, species table

* `SpeciesCatalog.from_file / from_text / (dict)` validates before use (duplicate JSON keys,
  `NaN`/`Infinity`, booleans in numeric fields, unknown version, wrong units, unknown keys, bad
  coefficients, layer counts/sums, porosity range, time units).
* **Ownership**: the parsed document is private. `catalog.document` returns a *deep copy*; editing it never
  changes the canonical content, the cached coefficient tables (frozen arrays) or the hash.
* `catalog.hash = "sha256:" + sha256(canonical JSON)` (compact, key-sorted). Whitespace and key order do
  not change it; numeric spelling (`1` vs `1.0`) does.
* **Species name/index table** = `catalog.species_ids` = the **sorted** tuple of IDs, deterministic across
  JSON key reordering (which also leaves the hash unchanged). Every consumer (state CSV, canopy, bridge)
  checks the table (`check_species_table`) in addition to the hash.

---

## 2. Domain, plant state and CSV (`leaf.state`)

### 2.1 Domain (immutable)

`Domain(nx, ny, dx_m, dy_m=None, origin_x_m=0.0, origin_y_m=0.0)`

* Raster axes `(ny, nx)`: row = y (north), col = x (east). Cell centres `origin + (col + 0.5)·dx`,
  `origin + (row + 0.5)·dy`; non-square cells allowed; area must be finite and positive.
* **Coordinates are landscape-frame**: `x_m`, `y_m` live in the same frame as the origin (they are *not*
  origin-relative). The MAPLE view subtracts the origin when publishing domain-local centres.
* Finite, **closed** `[x_min, x_max] × [y_min, y_max]`, **non-periodic**; nothing wraps (placement, cover,
  sampling). `to_dict()` records `axes`, `cell_sampling`, `edges` and `coordinate_frame`; `from_dict()`
  refuses conflicting metadata (e.g. `edges: periodic`, another sampling rule).
* Sampling convention **`containing_cell_closed_upper`**: `col = floor((x − origin_x)/dx)` clipped to
  `[0, nx−1]`; the upper edge belongs to the last column; no interpolation/transpose.
* `Domain` objects are frozen (attribute assignment raises), so sharing them is safe.

### 2.2 Canonical population (`PlantState`)

Structure-of-arrays, O(P), one species table (catalog order), stable `int64` IDs independent of row order.

| array | dtype | meaning |
|---|---|---|
| `ids` | int64 | unique plant ID |
| `species_index` | int32 | index into `species_names` |
| `x_m`, `y_m` | float64 | centre, landscape-frame metres |
| `establishment_ground_m` | float64 | ground elevation at establishment (NaN only while unresolved) |
| `establishment_unresolved` | bool | explicit SURFACE-pending mask |
| `agb_dry_kg` | float64 | **the** aboveground dry biomass state B |
| `seasonal_agb_dry_kg` | float64 | seasonal subcomponent E, `0 ≤ E ≤ B` |
| `age_days` | float64 | finite, ≥ 0 |
| `status` | int8 | 0 `live`, 1 `dead_standing`, 2 `removed` |

**Ownership.** Constructor inputs are copied (never aliased) and every canonical array is **read-only**;
`state.ids = …` or `state.agb_dry_kg[0] = …` raise. All supported mutations (`resolve_establishment`,
`advance_time`, `update_plants`, `adopt`) validate a candidate and replace whole arrays, so the by-ID index
can never go stale. `time_days` must stay finite.

**Invariants** (construction, every update, every write): unique IDs, known species, finite in-domain
positions, finite anchors for resolved plants, `B > 0` for live/dead_standing, `B ≥ 0` removed,
`0 ≤ E ≤ B`, `age ≥ 0`, known status. Removed/dead identities are retained forever. Empty populations work.

**SURFACE anchors.** Direct construction requires the explicit `establishment_unresolved` mask; a bare NaN
or inf anchor is an error, not a surface request. `resolve_establishment(initial_surface)` resolves pending
anchors **once** (containing cell) and never touches resolved ones; later surfaces never rewrite them.
Unresolved states cannot be advanced in time, recorded as non-initial snapshots or published.

**Age / clock.** `advance_time(days)` preflights the new clock and ages (overflow → `StateError`, nothing
changes) and then commits both; live plants age, dead/removed keep their age at status change. Call it once
per elapsed interval (wind, water, transport, growth alike); exports never age plants; life expectancy is
metadata only.

**Transactions.** `update_plants(ids, agb_dry_kg=…, seasonal_agb_dry_kg=…, status=…)` validates a candidate
copy; unknown/duplicate IDs, `E > B`, non-positive live mass, `removed → *` or `dead_standing → live` leave
the state untouched.

### 2.3 Plant-state CSV (`leaf.plant_state_csv`, version 1)

```
# {"schema": "leaf.plant_state_csv", "schema_version": 1, "time_days": 0.0, "n_plants": 6, "catalog_hash": "sha256:…", "species_names": [...], "domain": {...}, "units": {...}, "coordinate_frame": "...", "establishment_resolved": false, "initial": true, "provenance": {...}}
plant_id,species,x_m,y_m,establishment_ground_m,agb_dry_kg,seasonal_agb_dry_kg,age_days,status
1,grass,2.5,3.5,SURFACE,0.12,0.0,730.0,live
```

* One schema for initial input, snapshots and final output. Handwritten initial lists may omit the
  metadata line (the API/CLI supplies domain and time). With a metadata line: schema version, catalog
  hash, species table, units, coordinate frame, domain (and a supplied domain), row count and the
  **initial declaration** are checked before anything is built.
* `SURFACE` is accepted only in a handwritten list or a metadata-bearing file declared `"initial": true`
  (and not `establishment_resolved: true`). A saved non-initial snapshot carrying the token is refused
  **before any surface is sampled**, so a later surface can never be mistaken for the establishment surface.
* Floats are written with `repr()` and parsed with a strict decimal regex (exact FP64 round trip; no
  `nan`/`inf`/`1_000`). **No quoting**: a double-quote character anywhere is a format error (an
  unterminated quoted field is never "recovered"). Reads are all-or-nothing (`StateError`).
* Writes: `preflight_destination` (directory exists, no clobber), validate, serialise, then
  `atomic_write_bytes`: temp file + fsync, published with `os.link` when `overwrite=False` (fails if the
  destination exists; **no check-then-replace fallback** — on filesystems without hard links the write is
  refused, use `overwrite=True` to replace intentionally via `os.replace`).
* The file holds vegetation state only (no soil water, RNG, pending uptake or driver internals).

---

## 3. Geometry, safe MAPLE publication and cover (`leaf.canopy`)

`derive_geometry(state, catalog, current_surface=None)` validates the state and returns O(P) arrays for all
plants: `diameter_m`, `radius_m` (circles), **intact** `height_m`, `root_radius_m`, anchors, constant
`porosity`, …; non-finite derived outputs are refused. With `current_surface`: `surface_delta_m = current −
establishment` (+ deposition, − erosion), `burial_depth_m`, `exposure_depth_m`, `exposed_height_m`,
`fully_buried` (geometric diagnostics only).

`maple_vegetation_view(state, catalog, current_surface, *, statuses=('live','dead_standing'),
height_mode='exposed', unsupported_policy='error')` — **the current surface is required**: every canopy is
validated against its establishment anchor.

* Safe default `height_mode='exposed'`: `height = intact − burial`, canopy base on the current surface.
* Unsupported for native MAPLE: **eroded anchors** (base above the surface) and **fully buried plants**
  (exposed height ≤ 0). `'error'` refuses the whole view (default), `'exclude'` drops them (counted),
  `'flag'` keeps them (zero/exposed height) with `sidecar['native_maple_valid'] = False` and
  `meta['native_maple_valid'] = False`.
* `height_mode='intact_diagnostic'` publishes intact heights ignoring burial; it is labelled a
  **DIAGNOSTIC APPROXIMATION** and is never `native_maple_valid`. Partially or fully buried plants never
  silently reappear at full height on deposited soil.
* Fields: `center_x_m, center_y_m` (domain-local), `semi_major_m = semi_minor_m`, `orientation_deg = 0`,
  `height_m`, `porosity`, `drag_coefficient`; sidecars: `plant_id, species, status, establishment_ground_m,
  current_ground_m, burial_depth_m, exposure_depth_m, intact_height_m, exposed_height_m, elevated_base,
  fully_buried, native_maple_valid`. See `docs/maple-leaf-integration.md`.

`measure_cover(domain, x, y, radius, group_index, n_groups, subsamples=4, max_cover_cells, max_cover_bytes,
max_working_bytes)` — per-group projected union cover clipped to the finite domain (same-group overlap
once; cross-group overlap in each group; total union too), midpoint raster of resolution `dx/subsamples`.
Sampled, never analytic (`relative_error_scale ≈ resolution / radius`). **Memory admission before any
allocation** (`cover_admission`): `raster_bytes` (boolean union raster(s), bound `max_cover_bytes`) and
`working_bytes` (1-D coordinate scratch + **row-tiled** footprint temporaries, bound `max_working_bytes`,
default 1 MiB); the result reports both, the rows per chunk and the P-sized `index_bytes`. Never P×N.

**Boundaries.** Cover and exclusion use the finite, clipped, non-periodic domain. A periodic MAPLE raster
interpretation would change both cover and overlap; an adapter must requalify them, not assume equality.

---

## 4. Generator (`leaf.initialization`)

Config (`leaf.generation_config` v1; `examples/phase0/generation.json`): `domain`, `seed`, `overlap`
(`allowed` | `disallowed`), `cover_definition` (`per_species_union`), `cover_tolerance` (**finite positive
< 1, mandatory** — acceptance cannot be disabled), `cover_subsamples`, `max_iterations`, `limits`
(`max_plants`, `max_attempts`, `max_cover_cells`, `max_cover_bytes`, `max_working_bytes`), `time_days`,
`species.<id>` = `target_cover ∈ [0,1)`, `diameter_mean_m`, `diameter_std_m`, `initial_age_days` (number or
`{distribution: constant|uniform, …}`). Species are processed in sorted-id order.

1. Diameters: lognormal with the requested population mean/SD (`σ² = ln(1 + s²/m²)`, `μ = ln m − σ²/2`;
   `std = 0` constant). Biomass = species inverse allometry. Non-finite derived quantities (crown area,
   counts, parameters) are refused before allocation.
2. Counts: `E[A_c] = π/4 · (mean² + std²)`; start `N₀ = −A ln(1−f)/E[A_c]` (overlap) or `A f / E[A_c]`
   (exclusion); then fitted to the **measured** per-species union cover (`N ← N·ln(1−f)/ln(1−c)` or
   `N·f/c`) until `|c − f| ≤ cover_tolerance` or `max_iterations` (→ `GenerationError`, nothing returned).
   Corrected counts are admitted against `max_plants` **before** any new sample is drawn; per-species
   sample storage is compacted on truncation (no dead slots).
3. Placement, overlap allowed: independent uniform centres, conditioned on the fitted count and the cover
   acceptance (not an unconditioned Poisson process).
4. Placement, overlap disallowed — **large-canopy-first**: all marked sizes are sampled, sorted by
   descending diameter (stable tie-break: species order, sample order) and placed sequentially with uniform
   centre proposals rejected on any canopy intersection (roots may overlap), using sparse hashed bins
   (bin ≥ largest diameter; never all pairs). After every count correction the **complete candidate
   population is re-placed** in size order (nothing has an ID or an anchor yet, so no established plant ever
   moves). `max_attempts` bounds the total centre proposals of the whole generation; exhausting it reports
   inability *within these bounds* (not infeasibility of all packings). Random interleaving is not used:
   small plants occupying the gaps large plants need made a modest mixed case cost >150 CPU s.
5. Diameters are never rescaled, individuals never silently dropped, exclusion never relaxed, bounds never
   raised. IDs are `1..P` in final placement order; anchors `SURFACE` unless an initial surface is given.

The report carries the **complete normalised configuration** (`generation_config`: domain, species
distributions with μ/σ, age rules, limits, tolerance, subsamples, ordering), the RNG record
(`numpy.random.default_rng`, PCG64, NumPy version, draw order), targets, measured covers, nominal crown
fractions, counts, histories, sample statistics, cover admission, attempts and placement passes; a compact
copy goes into `state.provenance`. A run is reproducible from `generation_config` + catalog hash.

**Scaling.** Python-level O(P) loops over small NumPy bounding boxes; the mixed 128² / 0.25 m regression
(≈3.1 k grasses + ≈16 shrubs, exclusion) is bounded by `max_attempts = 50 000`. Timing is not asserted.

---

## 5. Recording (`leaf.recording.SnapshotWriter`)

```python
w = SnapshotWriter('out/veg_{index:04d}_{time_days:.1f}.csv', cadence_days=30.0)
w.write_initial(pop); ...; pop.advance_time(dt); w.maybe_write(pop); ...; w.write_final(pop)
w.write_manifest('out/manifest.json')
```

* Due times are `start + k·cadence` (integer k). `maybe_write` writes only when due. Every due time ≤ the
  current time is consumed; those not equal to it were **missed** and are listed as
  `skipped_due_times_days` (cadence 10: snapshots at 10 then 35 → `[20, 30]`; at 10 then 30 → `[20]`).
  Nothing is fabricated or interpolated; the snapshot always carries the actual state time.
* **All** bookkeeping (origin, due index, records, skipped totals, last time) is staged and committed only
  after the file exists; a failed initial/cadence/final write (validation, unresolved anchors, existing
  file, missing directory, I/O) leaves the writer exactly as before.
* Destination collisions (same kind/index/time path) raise `FileExistsError` unless `overwrite=True`.

---

## 6. Event diagnostics (`leaf.events`)

`SurfaceEventDiagnostics(plant_id, time_start_days, time_end_days, source, erosion_depth_m=…,
deposition_depth_m=…, abrasion_dose=…, abrasion_unit=…, transport_flux=…, flux_unit=…)` (schema
`leaf.surface_event_diagnostics` v1; `to_dict/from_dict`): non-negative metres for erosion (lowering) and
deposition (raising); `net_surface_delta_m = deposition − erosion`; optional doses/fluxes with explicit
units; no flux→injury conversion. Records are **immutable** (owned read-only arrays, attributes cannot be
rebound), so negative depths cannot be introduced after validation.

`apply_event_diagnostics(state, event, handler=None, require_elapsed=True)`: IDs are resolved once and used
through the whole transaction; the default records diagnostics only. A handler receives a `ReadOnlyView`
(read-only copies + the immutable `Domain`) and the frozen event; its proposals are validated on a candidate
copy and applied all-or-nothing; a raising handler, an invalid proposal or any attempt to redirect the event
IDs leaves state, domain and IDs unchanged. No threshold mortality is ever invented.

---

## 7. Bridge to the existing growth drivers (`leaf.runtime`)

While a driver is attached, **its B/E are the numerical authority**; LEAF keeps identity, species,
anchors, ages and the clock, and creates the record view only at requested output via
`capture_snapshot(population, sim, seasonal_policy='refuse'|'keep'|'zero')` (by stable ID; exact integer
ID/profile types and ranges incl. unsigned overflow, missing/unknown/duplicate IDs, non-live plants and
profile mismatches are refused; a non-STEP driver has no E, so pre-existing population E is never erased
silently). Phase 0 event proposals do **not** reach a running driver.

`legacy_inputs_from_state(population, catalog, biomass_policy='refuse'|'accept_driver_recompute',
seasonal_policy='refuse'|'restore'|'acknowledge_reset', stepwat_traits=None, rtol=1e-9)` builds the driver
`inputs` for live plants. The unmodified driver recomputes `B = π r²·ρ` (fixture ρ = 1.0 kg m⁻², or the STEP
trait density) and keeps that square-root law during growth, so consistency is judged on the **species
law** (`b_d = 0.5`, `a_d = 2/√(πρ)`), not on one initial biomass; anything else is refused unless the
approximation is acknowledged explicitly (recorded). Nonzero E is refused unless restored
(`restore_seasonal_mass` after constructing in STEP mode) or its reset is acknowledged. Unknown profile
names, non-finite/negative `rtol`, non-finite densities, offset/non-square domains, mixed root multipliers
and zero layer fractions are refused. This is state replay/handoff, **not** a coupled restart (soil water,
pending uptake, STEP clock, RNG are not restored). The driver's science is unchanged.

---

## 8. CLI

```
python -m leaf validate-species --species examples/phase0/species.json
python -m leaf init --species S --plants examples/phase0/plants.csv --nx 20 --ny 20 --dx-m 1 --surface-constant 100 --out veg0.csv
python -m leaf generate --species S --config examples/phase0/generation.json --surface-constant 12 --out gen.csv --report gen.json
python -m leaf inspect --species S --plants veg0.csv --surface-constant 100.5 --cover-subsamples 8
python -m leaf export-canopy --species S --plants veg0.csv --surface-constant 100 --out view.json
python -m leaf export-canopy --species S --plants veg0.csv --surface-constant 101 --unsupported exclude --out buried.json
```

`init`/`generate` take the INITIAL surface (optional; without it the output is an initial file keeping the
`SURFACE` token). `inspect`/`export-canopy` take the CURRENT surface; `export-canopy` **requires** it. In the
example, `plants.csv` holds explicit anchors (99.9, 100.0, 99.8) at or below the surface of 100 m, so the
first export is native-MAPLE-valid; the second command shows the explicit handling of a deposition that
fully buries the grasses (refused by default). `generate --report` must differ from `--out`; both
destinations are preflighted and both outputs serialised before any file is committed. Outputs are atomic;
existing files are refused without `--overwrite`; errors exit 1 (usage 2).

## 9. Examples and reproducibility

`examples/phase0/species.json` (two fixture species), `plants.csv` (SURFACE and explicit anchors, ages,
dead and removed rows), `generation.json` (128² cells of 0.25 m, grass 6 % at 0.15 ± 0.03 m, shrub 5 % at
2.0 ± 0.3 m, overlap allowed, seed 20261110). All numbers are fixtures. Generation is reproducible from the
report's `generation_config` + catalog hash.

## 10. Limitations (Phase 0)

No growth, water use, recruitment, mortality, injury, dislodgement, surface remapping or calibration.
Cover is sampled, not analytic. The CSV is vegetation state only. The legacy driver's area law restricts
replay into it (§7). Exclusion placement is a bounded initialisation algorithm, not a packing proof.
