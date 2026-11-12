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
* **`life_expectancy`**: `null` or `{"mean": > 0, "unit": "days" | "years"}` → `life_expectancy_days(id)` (years = 365 d).
  A nominal mean: **no plant ever dies because its age reaches it** (mean ≠ hard maximum; mortality is the separate
  probabilistic law). It also supplies the upper bound of the generator's default initial-age sampling (§4): ages are
  drawn uniformly on `[0, life_expectancy)`; `null` makes that default unusable (explicit ages still work).
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

### 2.3 Plant CSV and companion metadata

**Convention (persistent project rule): do not embed metadata in a plant file.** A LEAF plant file is a
**plain nine-column CSV**: line 1 is the header, every following line is one plant with exactly nine
fields, and there is no `#` line, JSON or comment — in every file LEAF writes (`init` / `generate`,
snapshots, final output, checkpoint `plants.csv`, preview tables). There is no option, flag or keyword that
writes embedded metadata. The saved-file metadata lives in a **separate human-readable companion JSON**
named `<stem>.metadata.json` in the same directory (`initial_plants.csv` → `initial_plants.metadata.json`,
`veg_0001.csv` → `veg_0001.metadata.json`, `plants.csv` → `plants.metadata.json`; the stem is the file name
without its last extension; `leaf.state.metadata_path(csv_path)`).

`plants.csv` (plain, exactly as written — only the header and rows):

```
plant_id,species,x_m,y_m,establishment_ground_m,agb_dry_kg,seasonal_agb_dry_kg,age_days,status
1,grass,2.5,3.5,SURFACE,0.12,0.0,730.0,live
```

`plants.metadata.json` (schema `leaf.plant_state_metadata`, version 1; abbreviated):

```json
{
  "schema": "leaf.plant_state_metadata", "schema_version": 1, "model": "LEAF", "leaf_version": "0.1.0",
  "csv": {"file": "plants.csv", "sha256": "sha256:…", "n_bytes": 1234, "columns": ["plant_id", "species", …]},
  "time_days": 0.0, "n_plants": 6, "catalog_hash": "sha256:…", "species_names": ["grass", "shrub"],
  "domain": {"nx": 20, "ny": 20, "dx_m": 1.0, …}, "units": {…}, "coordinate_frame": "…",
  "status_codes": {…}, "establishment_resolved": false, "initial": true, "cell_sampling": "…",
  "provenance": {…}
}
```

* One CSV format for initial input, snapshots and final output. **Readers** (`read_plant_csv`, the CLI,
  the checkpoint loader) load the companion automatically and verify it **against the exact CSV bytes**:
  schema and integer version, `csv.n_bytes` (a non-negative integer equal to the exact CSV size) and `csv.sha256`
  (a stale, edited, reformatted or half-written pair is refused),
  `n_plants`, catalog hash and species table, domain and `time_days` (and any domain / time the caller also
  supplied must agree), units, coordinate frame, header columns, and the **initial declaration** — all before
  anything is built. A companion that lacks a required key, has a wrong type, duplicate keys or `NaN` is
  refused. `csv.file` is informational only (a pair may be moved or renamed together; the hash is the
  binding).
* **No companion = handwritten plain list.** Handwritten initial lists need no companion: the caller supplies
  the domain (`--domain` / `--nx --ny --dx-m` in the CLI), optionally `time_days` (default 0) and, for
  `SURFACE` anchors, the initial surface. `read_plant_csv(path, catalog, metadata=False)` deliberately
  ignores a companion (the companion's declared time and SURFACE safety then no longer protect the read).
  `parse_plant_csv(text, catalog, metadata=<dict | JSON text | bytes>)` verifies an in-memory pair, and
  `plant_csv_text(state, initial=)` / `plant_metadata(state, csv_text, initial=, provenance=)` /
  `plant_files_text(...)` produce them; the in-memory serializer **always returns plain CSV**.
* `SURFACE` is accepted only in a plain list without companion or in a file whose companion declares
  `"initial": true` (and `establishment_resolved: false`, which must agree with the CSV). A file whose companion
  does not declare it initial but whose CSV carries the token is refused **before any surface is sampled**, so
  a later surface can never be mistaken for the establishment surface. Non-initial snapshots cannot be
  written with the token.
* Floats are written with `repr()` and parsed with a strict decimal regex (exact FP64 round trip; no
  `nan`/`inf`/`1_000`). Rows are **standard strict CSV** (`csv.reader(strict=True)`): ordinary quoted
  fields such as `"0.12"` or `"live"` are accepted (spreadsheet writers quote freely); an unterminated or
  stray quote is malformed and refused. Reads are all-or-nothing (`StateError`).
* **Two-file writes** (`write_plant_files` / `write_plant_csv`, snapshots, `init`, `generate`, preview):
  validate and serialise **both** files first; preflight **both** destinations (directory exists; neither the
  CSV nor its companion exists unless `overwrite=True`; an overwrite never replaces a metadata file that is not a
  LEAF plant companion, so unrelated `<stem>.metadata.json` files are never clobbered); then publish the
  companion first and the CSV second, each with `atomic_write_bytes` (temp file + fsync, published with `os.link`
  when `overwrite=False` — **no check-then-replace fallback**, on filesystems without hard links the write is
  refused; `os.replace` only for `overwrite=True`). If the CSV cannot be published a newly created companion is
  removed (a replaced one is restored) and the error is re-raised. **Two renames are not one atomic
  operation:** if the process dies between them, an orphan companion without a CSV (every reader fails: missing
  CSV) or, on overwrite, a new companion beside the old CSV (every reader fails: SHA-256 mismatch) remains;
  such a pair is *detected and refused*, never silently used. Remove the stale companion (or rewrite with
  `overwrite=True`) to recover. A checkpoint directory is still published by one directory rename.
* **Editing a CSV by hand** invalidates its companion hash: either delete the companion (the file is then a
  plain list read with caller context) or regenerate the pair (`python -m leaf init --plants edited.csv …`
  after removing the old companion, which writes a verified pair).
* **Legacy files.** Files written before this convention carry the metadata as a first line
  `# {"schema": "leaf.plant_state_csv", "schema_version": 1, …}`. They are still **read** (same checks as
  before, domain and time from that line), but not together with a companion (refused as ambiguous), and no
  code path writes that form. A first line starting with `#` that is not that JSON line is refused — plain
  files have no comments. To convert one, split the first line into `<stem>.metadata.json` (new schema, with
  the CSV SHA-256) using `plant_metadata`; the rows are unchanged.
* The file holds vegetation state only (no soil water, RNG, pending uptake or driver internals).

---

## 3. Geometry, shared canopy view, MAPLE adapter and cover (`leaf.canopy`)

`derive_geometry(state, catalog, current_surface=None)` validates the state and returns O(P) arrays for all
plants: `diameter_m`, `radius_m` (circles), **intact** `height_m`, `root_radius_m`, anchors, constant
`porosity`, …; non-finite derived outputs are refused. With `current_surface`: `surface_delta_m = current −
establishment` (+ deposition, − erosion), `burial_depth_m`, `exposure_depth_m`, `exposed_height_m`,
`fully_buried` (geometric diagnostics only). On the canopy export path it is the **single** allometry /
surface-sampling implementation; every view below is built from it.

### 3.1 Shared canopy view — primary API (`canopy_view`, schema `leaf.canopy_view` v1)

`canopy_view(state, catalog, current_surface, *, statuses=('live','dead_standing'), height_mode='exposed',
drag_coefficient=1.0)` is the **one consumer-neutral aboveground representation for MAPLE and SYRUP**
(`docs/shared-canopy.md` is the contract). The current surface is **required**. It returns
`{'fields', 'sidecar', 'meta'}` with O(P) arrays for the selected standing population (status in
`statuses`, diameter > 0): `CANOPY_FIELDS = center_x_m, center_y_m` (domain-local metres), `semi_major_m =
semi_minor_m` (canopy radius), `orientation_deg = 0`, `height_m`, constant `porosity`, uniform
`drag_coefficient`; sidecars `plant_id, species, status, establishment_ground_m, current_ground_m,
burial_depth_m, exposure_depth_m, intact_height_m, exposed_height_m, root_radius_m, elevated_base,
fully_buried, native_maple_valid`. `MAPLE_FIELDS` is kept as an alias of `CANOPY_FIELDS`.

* `height_mode='exposed'` (default): `height_m = max(intact − burial, 0)`; the visible canopy base is at
  `max(establishment_ground_m, current_ground_m)` — the current ground under partial burial, the **fixed
  establishment ground** under erosion (intact height unchanged, `exposure_depth_m` is the separate gap
  below the base); a fully buried plant has height 0, no visible canopy and keeps its anchor.
  `'intact_diagnostic'`: intact height, base at the establishment ground, labelled a
  **DIAGNOSTIC APPROXIMATION**.
* `removed` plants are **excluded by default** (an explicit `statuses` selection can include removed
  records with a positive diameter). Consumers using the same state, catalog and surface at matching
  coupling times obtain consistent geometry; upstream adoption is pending.
* **Fully buried and elevated-base (eroded-anchor) plants are RETAINED** with correct diagnostics
  (`fully_buried`, `elevated_base`, depths, exposed height 0 when buried). Geometric visibility is distinct
  from death/removal, which only `status` expresses. Per-plant `native_maple_valid` flags and
  `meta['native_maple_compatibility']` are **informational** (MAPLE container limits), not a validity
  judgment of the view; there is no whole-view `native_maple_valid` verdict here.
* Pure and uncached: derived from the authoritative `PlantState`; recompute after any state or surface
  change; pair consumers by `plant_id`. Invalid/unresolved state or surface still fails (`StateError`).
* Metadata carries explicit units, coordinate frame, height definition/anchor, catalog hash, time, domain
  conventions, boundary note and exclusion/visibility counts.

### 3.2 Thin MAPLE adapter (`maple_vegetation_view`, schema `leaf.maple_vegetation_view` v2)

`maple_vegetation_view(state, catalog, current_surface, *, statuses=('live','dead_standing'),
height_mode='exposed', unsupported_policy='error', drag_coefficient=1.0)` — unchanged signature, defaults
and schema; now a **projection of `canopy_view`** that contains no allometry, surface sampling, canopy
geometry or coordinate conversion. It only applies the native-MAPLE policy for plants MAPLE's current
object container cannot hold (**eroded anchors** and **fully buried plants**): `'error'` refuses the whole
view with the offending IDs (default), `'exclude'` drops them (counted) with **one index selection over
every field and sidecar array**, `'flag'` keeps them with `sidecar['native_maple_valid'] = False` and
`meta['native_maple_valid'] = False`. `intact_diagnostic` is never `native_maple_valid`. Fields and
sidecars are **exactly** the shared values by `plant_id`; `meta['derived_from']` names the shared schema.
These restrictions are MAPLE-specific; SYRUP calls `canopy_view` directly and does not inherit them. See
`docs/maple-leaf-integration.md`.

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
`species.<id>` = `target_cover ∈ [0,1)`, `diameter_mean_m`, `diameter_std_m` and the OPTIONAL `initial_age_days`
(see "Initial ages" below). Species are processed in sorted-id order.

**Initial ages (default: uniform from 0 to the species' life expectancy).** When `initial_age_days` is omitted — and
for the explicit selector `{"distribution": "uniform_to_life_expectancy"}`, which the shipped example and the Jornada
trial config write out — every generated plant's age is drawn **independently** from a continuous **uniform**
distribution on `[0, L)` days, where `L = SpeciesCatalog.life_expectancy_days(species)` is the nominal
`life_expectancy` of **that species' catalog record** (`mean × 365` for `unit: years`, `mean × 1` for `days`).
The bound is read from the catalog: it is never duplicated in the config and never inferred from allometry or from a
mortality (Weibull) scale. One vectorised `rng.uniform(0, L, k)` call per species batch uses the existing seeded RNG;
diameters, placement, cover acceptance and memory limits are unchanged and **no age–biomass relation is invented**
(ages are independent of size). Fixture catalog: grass 8 years → 0..2920 d, shrub 40 years → 0..14 600 d; Jornada trial
case: BOER 15 years → 0..15 years (0..5475 d), PROGLA 80 years → 0..80 years (0..29 200 d). **This is only the
initial sampling range.** The life expectancy is a *nominal mean*, not a death age: LEAF never kills a plant because its
age reaches it (a plant may start, or become, older than `L` only by explicit input; mortality is the separate
probabilistic law of `leaf.demography`). A species whose `life_expectancy` is `null` cannot use the default or the
selector: `validate_generation_config` raises a `CatalogError` naming the species (no guessed bound, no silent zero).
Explicit controls remain for reproducible fixtures and controlled experiments and work with `life_expectancy: null`:
a number of days, `{"distribution": "constant", "value_days": …}` and `{"distribution": "uniform", "min_days": …,
"max_days": …}`. Handwritten plant files, loaded snapshots/restarts and age-0 recruits (Phase 3) are not touched by the
generator rule. Reporting: `report['generation_config']` keeps the **source rule** (the selector, written explicitly even
if the original omitted the key) so that replay with the same catalog re-resolves the same bounds; the **resolved**
bounds, the life-expectancy value/unit, the catalog hash, its origin (`species[<id>].life_expectancy`) and whether the
rule was the default are in `report['derived']['species'][<id>]['initial_age']`, `report['species'][i]['age_requested']`
and the state provenance (hence in the companion metadata, never in the plain CSV). Replay with a *different* catalog
gives a different range by design (its hash differs).

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

The report carries the **accepted source configuration** (`generation_config`: schema, domain, seed,
overlap, tolerance, subsamples, iterations, limits, time, species target/diameter/age rules) which is
*directly replayable*: `generate_population(catalog, report['generation_config'], surface=...)` with the
same catalog and surface regenerates the population array-for-array (also after a JSON round trip).
Derived quantities (`derived`: ordering convention, lognormal μ/σ, expected crown areas, initial count
estimates), the RNG record (`numpy.random.default_rng`, PCG64, NumPy version, draw order), targets, measured
covers, nominal crown fractions, counts, histories, sample statistics, cover admission, attempts and
placement passes are reported outside the config; a compact copy goes into `state.provenance`.

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
  after **both** files (the plain snapshot CSV and its `<stem>.metadata.json` companion, §2.3) exist; a failed
  initial/cadence/final write (validation, unresolved anchors, existing CSV or companion, missing directory,
  I/O) leaves the writer exactly as before. Each record lists `path` (CSV) and `metadata_path`; the
  companion's provenance carries the snapshot kind, index, cadence and skipped due times.
* Destination collisions (same kind/index/time path, or an existing companion) raise `FileExistsError` unless
  `overwrite=True`.

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
(`restore_seasonal_mass` right after constructing a **fresh** STEP driver: it enforces driver `day == 0`,
coupled accepted `days == 0`, the same ID/species/live association as capture and `E ≤ driver B` before
touching anything; a mid-run call raises and never resets a clock) or its reset is acknowledged. Device-backed
driver vectors (CuPy-like `.get()`) are checked on shape/ndim/dtype metadata against P **before** the single
host transfer, and validated again afterwards. Unknown profile
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
`SURFACE` token, with `"initial": true` in its companion). `inspect`/`export-canopy` take the CURRENT surface;
`export-canopy` **requires** it. In the example, `plants.csv` holds explicit anchors (99.9, 100.0, 99.8) at or
below the surface of 100 m, so the first export is native-MAPLE-valid; the second command shows the explicit
handling of a deposition that fully buries the grasses (refused by default).

`init` and `generate` write a **plain CSV plus its companion** (`veg0.csv` + `veg0.metadata.json`,
`gen.csv` + `gen.metadata.json`; the JSON output lists `written` and `metadata_written`). Every command that
reads a plant CSV loads and verifies the companion automatically (no `--domain` is needed when one exists; a
conflicting `--domain` / `--nx…` or an edited CSV is refused, exit 1); a handwritten plain list has no companion,
so give the domain. `generate --report` must differ from `--out` **and from `--out`'s companion**
(`<stem>.metadata.json`); the CSV destination, the companion destination and the report are preflighted and
all outputs serialised before any file is committed; the report is published first and removed again if the
plant pair cannot be published (a newly created companion is removed too). Outputs are atomic per file;
existing files (including an existing companion, which may belong to another file) are refused without
`--overwrite`; errors exit 1 (usage 2).

## 9. Examples and reproducibility

`examples/phase0/species.json` (two fixture species), `plants.csv` (SURFACE and explicit anchors, ages,
dead and removed rows), `generation.json` (128² cells of 0.25 m, grass 6 % at 0.15 ± 0.03 m, shrub 5 % at
2.0 ± 0.3 m, overlap allowed, seed 20261110, initial ages uniform 0..life expectancy of each species via the selector). All
numbers are fixtures. Generation is reproducible from the
report's `generation_config` + catalog hash.

## 10. Limitations (Phase 0)

No growth, water use, recruitment, mortality, injury, dislodgement, surface remapping or calibration.
Cover is sampled, not analytic. The CSV is vegetation state only. The legacy driver's area law restricts
replay into it (§7). Exclusion placement is a bounded initialisation algorithm, not a packing proof.
The shared canopy view (§3.1) is an interface contract: MAPLE and SYRUP have not yet adopted it, no
SYRUP adapter, roughness law or rasterizer is defined, and porosity/height have no qualified hydrological
interpretation; no integrated MAPLE/SYRUP execution is claimed.
