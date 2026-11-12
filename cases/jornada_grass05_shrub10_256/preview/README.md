# Initial-vegetation preview (`preview/`)

![Initial vegetation, plan view](vegetation_initial.png)

(The image is the preserved historical render of this case; see "The existing picture in this directory" below for how it
relates to the plain-CSV layout and to the helper as it is written now.)

**Frozen species catalog (added by task `literature_parameters_20261111`).** The saved preview population was initialised with the earlier catalog `jornada_grass05_shrub10_trial_20261111`, preserved here as [`species_at_initialization.json`](species_at_initialization.json) (a byte-for-byte copy of the case `species.json` before the literature edits; hash-verified by the reviewer against the baseline). The case's active `species.json` now has `catalog_id` `jornada_grass05_shrub10_litparam_rev2_20261111`, revised PROGLA allometry/live fraction/support and BOER rate/clonal numbers, so its catalog hash no longer matches the hash recorded in `initial_plants.metadata.json`. No historical preview file was modified and no new plants or plot were generated.

**Reading the saved preview (use the frozen catalog).** The public reader is `leaf.state.read_plant_csv(path, catalog, *, domain=None, surface=None, time_days=None, provenance=None, metadata=None)`; the companion `initial_plants.metadata.json` beside the CSV is loaded automatically (it supplies domain/time) and the pair is verified against the catalog hash:

```python
from leaf.traits import SpeciesCatalog
from leaf.state import read_plant_csv

case = "cases/jornada_grass05_shrub10_256"
frozen = SpeciesCatalog.from_file(f"{case}/preview/species_at_initialization.json")
state = read_plant_csv(f"{case}/preview/initial_plants.csv", frozen)     # OK: catalog hash matches the companion

active = SpeciesCatalog.from_file(f"{case}/species.json")
# read_plant_csv(f"{case}/preview/initial_plants.csv", active)           # raises StateError: catalog hash mismatch
```

**Why the active catalog must not reinterpret the old plants.** Each plant row stores dry aboveground mass `agb_dry_kg`; crown diameter and height are *derived* from it by the catalog allometry. The saved PROGLA masses were generated from the user's diameters with the old law D = 1.6·B^0.4 (D = 1 m ⇔ B = 0.3088 kg). Applying the revised law D = 1.1087951253470727·B^0.3351206434316354 to those same masses would give D ≈ 0.75 m and H ≈ 0.82 m for that plant (instead of 1.0 m and 0.5 m), i.e. it would shrink the shrub cover and silently change the geometry, whereas a plant of D = 1 m under the revised law holds 0.735 kg. The saved plants also carry the old constant ages (730 d / 1825 d), not the uniform-age draw. The readers therefore refuse a hash mismatch by design; the revised species parameters require a **new initialization** (not run), which regenerates masses, positions and ages from the user diameters with the active catalog.

**INITIALIZATION ONLY.** This is a scientific plan-view map of the *initial* LEAF population and the initial
random surface of the `jornada_grass05_shrub10_256` trial case. No growth, hydrology, wind/water events,
mortality, recruitment, evolution or any other ecological/transport simulation is run; simulation time is 0 d.
It is not an illustration: every circle is an actual plant from `leaf.initialization.generate_population`.

## Generate (future use; Codex runs the reviewed version)

Dependencies: only the existing `/home/okin/MAPLE/.venv-cupy` environment (NumPy, Matplotlib; `tomllib` from
Python ≥ 3.11 or `tomli`). Nothing is installed. Python cache writes are disabled by the caller and
`MPLCONFIGDIR` is pointed under `/tmp`:

```bash
cd <leaf repo>
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/leaf_preview_mpl \
/home/okin/MAPLE/.venv-cupy/bin/python cases/jornada_grass05_shrub10_256/preview/render_preview.py \
    --repo <abs leaf repo> --case <abs leaf repo>/cases/jornada_grass05_shrub10_256 \
    --output <abs leaf repo>/cases/jornada_grass05_shrub10_256/preview
```

`--repo`, `--case` and `--output` are required absolute paths. All six output files (including the plant-table
companion) are checked before any work and are created exclusively: an existing file stops the run, nothing is
overwritten.

## The existing picture in this directory (historical record) versus a new run of the helper

**Initial ages of the existing picture.** It is the EARLIER initialization: its population was generated with the constant
ages 730 d (BOER) and 1825 d (PROGLA) that `generation.json` then specified. The case's current `generation.json` has since
been changed to the species-linked rule (each age uniform on 0 … the species' nominal life expectancy: BOER 0–15 years,
PROGLA 0–80 years; an initial sampling range, not a death age), so this picture is **not** a realization of the new rule and
was not regenerated; a new initialization would need a separate request. Because the helper reads the case's current
`generation.json`, a newly executed run of it would produce uniformly distributed ages, and it would NOT reproduce the existing
population: the uniform age draws consume random numbers that the constant ages did not, so later diameters and positions
(and hence counts and the image) would differ as well.

The `vegetation_initial.png`, `initial_surface.npy`, `initial_plants.csv` and `initialization_report.json` that are already
in this directory were produced **before** the plain-CSV change, by the earlier operator's run of the preview code; they were
**not** produced by the helper as it is written now, and no new run, regeneration or re-rendering was done for the
plain-CSV change. Consequences, stated precisely:

* `preview_provenance.json` is the **preserved original render record**. It is immutable here and carries the SHA-256 of the
  *pre-split* `initial_plants.csv` (which then began with an embedded metadata line). It contains **no** companion hash and no
  `plant_files` block.
* The plain `initial_plants.csv` + `initial_plants.metadata.json` layout of this directory comes from a **pure metadata
  split** of that same file: the former first line was moved into the companion and every plant row (bytes, order, count) and
  the image were preserved unchanged. The current pair hashes, the original provenance hash and the row/image preservation
  checks are recorded in a **separate addendum, `plant_files_migration.json`**, written by the conversion step (not by the
  helper). Treat the addendum, not `preview_provenance.json`, as the record of the current plain-CSV files.

What follows ("Generate", "What it does", "Files a NEW run writes") describes a **future, newly executed** run of the
modified helper, which would write the companion and a provenance that includes its hash. It is not how the existing picture
was made.

## What it does

1. Reads `generation.json`, `species.json`, `landscape.toml` from `--case` (unchanged; nothing is modified).
2. Builds the initial surface by calling the **original** MAPLE
   `/home/okin/MAPLE/src/maple/case_tools/generators/random_field.py::generate_spectral_gaussian_field`
   (loaded with `importlib`, not copied) with the exact `landscape.toml` `surface.call_arguments`
   (256 × 256, 0.25 m, correlation length 1 m, relief 0.05 m, seed 1173) and adds `reference_level_m = 0.6`.
3. Calls `leaf.initialization.generate_population` (seed 20261111, BOER cover 0.05, PROGLA cover 0.10, lognormal
   diameters, non-overlap large-canopy-first placement) with that surface, then
   `leaf.canopy.derive_geometry(plants, catalog, current_surface=surface)`. Plants are never re-sampled,
   moved or resized by the script. Anchors resolve from this initial surface.
4. Draws the full 64 × 64 m domain and an 8 × 8 m close-up (x, y = 28–36 m) with Matplotlib `EllipseCollection`
   in data units (diameters are exact metres, no fixed-size markers), edge-clipped, grass bright green, shrubs
   dark teal. The legend shows accepted counts, measured union covers and sampled diameter mean ± SD from
   the generator report; terrain is subtle shading of the 5 cm relief only (no mounds implied).

## What it does not do

No dynamic model, no cover/count tuning, no change of generator bounds after a refusal, no roots or photosynthesis
imagery, no tests or benchmarks, no writes to `inputs/`, the upstream repos or any other output. The population
is a **separate preview population**, not adopted as the production `inputs/plants.csv` / `inputs/surface.npy`,
which remain ungenerated.

## Files a NEW run of the helper would write

(Not a description of the existing historical files above: see the preceding section.)

| File | Content |
|---|---|
| `vegetation_initial.png` | 15 × 9 in, 200 dpi (≈ 3000 × 1800 px) figure |
| `initial_surface.npy` | the (256, 256) float64 initial surface, m |
| `initial_plants.csv` | PLAIN canonical initial plant CSV (`leaf.state.write_plant_csv`): nine-column header + one nine-field row per plant; no embedded metadata, comment or JSON (project rule) |
| `initial_plants.metadata.json` | companion metadata of the CSV (schema `leaf.plant_state_metadata`): time 0 d, domain, catalog hash / species table, `n_plants`, units, `initial` / `establishment_resolved`, provenance, the CSV SHA-256 and the exact CSV size (`csv.n_bytes`) |
| `initialization_report.json` | the exact generator report |
| `preview_provenance.json` | source/config SHA-256 hashes, **SHA-256 of every output file including `initial_plants.csv` and `initial_plants.metadata.json`**, a `plant_files` block (plain-CSV statement, the CSV hash the companion declares, and whether the written pair read back exactly equal to the generated population), environment versions, image settings, `INITIALIZATION_ONLY`, measured consistency checks (anchors vs surface, time 0, geometry vs report diameters) |

After writing, the helper reads the pair back through `read_plant_csv` (companion verified) and exits 1 if it does not equal
the generated population. Edit neither file by hand: changing the CSV invalidates the companion hash and every reader refuses it.

Expected (not enforced) from the case notes: ~25k BOER and ~500 PROGLA; actual counts and covers are those in the
report. A new run needs separate authorisation and must target a directory where none of the six files exists.

## Scope note

The parent case README states that nothing was executed. That prior no-run statement is superseded **only** for this
initial picture (initial placement/surface); the rest of the case remains unrun and unvalidated.
