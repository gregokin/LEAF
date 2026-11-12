# Initial-vegetation preview (`preview/`)

![Initial vegetation, plan view](vegetation_initial.png)

(The image exists only after the renderer below has been run and accepted; until then the link is empty.)

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

`--repo`, `--case` and `--output` are required absolute paths. All five output files are checked before any
work and are created exclusively: an existing file stops the run, nothing is overwritten.

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

## Files written on rendering

| File | Content |
|---|---|
| `vegetation_initial.png` | 15 × 9 in, 200 dpi (≈ 3000 × 1800 px) figure |
| `initial_surface.npy` | the (256, 256) float64 initial surface, m |
| `initial_plants.csv` | canonical initial plant CSV (`leaf.state.write_plant_csv`) |
| `initialization_report.json` | the exact generator report |
| `preview_provenance.json` | source/config SHA-256 hashes, environment versions, image settings, `INITIALIZATION_ONLY`, measured consistency checks (anchors vs surface, time 0, geometry vs report diameters) |

Expected (not enforced) from the case notes: ~25k BOER and ~500 PROGLA; actual counts and covers are those in the
report. These files do not exist until a run is performed.

## Scope note

The parent case README states that nothing was executed. That prior no-run statement is superseded **only** for this
initial picture (initial placement/surface); the rest of the case remains unrun and unvalidated.
