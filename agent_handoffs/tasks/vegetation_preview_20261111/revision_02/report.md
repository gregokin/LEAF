# Revision2 (layout only) — appended note
Edited only `render_preview.py` (and this report), source-only, uncommitted, nothing executed or inspected:
- zoom axes `[0.625, 0.44, 0.33, 0.445]` (top edge unchanged at 0.885); legend/text axes `[0.585, 0.075, 0.395, 0.28]`; main plot, ZOOM, data and circle sizes untouched.
- Right-column text re-broken into short lines (circle note, close-up counts, two terrain lines); font sizes unchanged (8.6 pt text, 9.5 pt legend); sampled-diameter lines kept; text anchor moved to y=0.62 of the text axes.
- Legend: counts with thousands separators, covers and targets as percentages with 2 decimals; wording shortened to fit the column.
- Layout fit/overlap is unverified until Codex re-renders and inspects. Native initialization/provenance code unchanged.

# Report — vegetation preview source authoring (revision1)

## Written (uncommitted, only the four allowed paths)
- `cases/jornada_grass05_shrub10_256/preview/render_preview.py` — renderer (new)
- `cases/jornada_grass05_shrub10_256/preview/README.md` — commands, scope, files (new)
- `cases/jornada_grass05_shrub10_256/README.md` — top status clarified as historical; preview section appended
- `report.md` — this file

## Design summary
- Required absolute `--repo`, `--case`, `--output`; all five output destinations preflighted; exclusive (`x`) creation, no overwrite; output refused under case `inputs/` or `leaf/`.
- Surface: original MAPLE `generate_spectral_gaussian_field` loaded via `importlib.util.spec_from_file_location`, called with `landscape.toml` `surface.call_arguments` verbatim, plus `reference_level_m`.
- Population: `traits.load_catalog`, `initialization.generate_population(catalog, generation.json dict, surface=surface)`, `canopy.derive_geometry(plants, catalog, current_surface=surface)`; no independent sampling or resizing.
- Plot: Agg, 15×9 in at 200 dpi, `EllipseCollection` in data units (`units='xy'`, widths = heights = geometry diameter, compat fallback for `transOffset`), full view all plants, 8×8 m zoom at 28–36 m using circle–rectangle intersection, connector lines, legend with report counts/covers and diameter mean±SD (sampled and configured), caption text.
- Provenance records measured checks (time 0, anchors vs surface max diff, geometry vs report diameter mean, status counts, age values), hashes, versions.

## Verification honesty
- **Not run**: no Python execution of any kind (no import, syntax check, `--help`, parser check), no rendering, no tests, no network, no Git.
- Performed: manual reading of the supplied interface copies (`initialization.py`, `canopy.derive_geometry`, `state.write_plant_csv`/`Domain`, `traits.load_catalog`, MAPLE generator signature, case configs) to verify call signatures.
- Not verified: matplotlib version behaviours (`offset_transform` vs `transOffset` handled by try/except; `ConnectionPatch` between axes; text layout/legibility), Python ≥3.11 `tomllib` availability in `.venv-cupy` (falls back to `tomli`), that the generator will accept the cover/bounds (no bound changes were made or are made on refusal), actual counts/covers (none claimed).
- Known review points for Codex: legend/text layout in the right-hand column should be inspected in the PNG; `leaf.__init__` hash is recorded via `leaf.__file__`; preview README image link is empty until a run is accepted.
- No parameters, case inputs, root AGENTS or core code were modified.
