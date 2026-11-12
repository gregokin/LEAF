# inputs/ — expected generated inputs (NOT present)

This directory intentionally contains **no data files**. Nothing here has been generated, and no placeholder is provided that could be mistaken for real data.

| Expected file | Produced by | Status |
|---|---|---|
| `inputs/surface.npy` | a future preparation step that calls the **existing upstream** public generator `generate_spectral_gaussian_field(ny=256, nx=256, dx_m=0.25, dy_m=0.25, correlation_length_x_m=1.0, correlation_length_y_m=1.0, relief_m=0.05, seed=1173)` and then adds the 0.6 m reference level (see `../landscape.toml`) | **not generated**; no fake surface exists |
| `inputs/plants.csv` | a future, explicitly authorised run of the LEAF generator with `../generation.json` + `../species.json` and the actual surface (SURFACE establishment anchors resolved once) | **not generated**; no plant population exists |
| `inputs/plants.metadata.json` | written beside the plain plant CSV by the same generator call | **not generated**; holds time, domain, catalog identity, units, provenance and the CSV hash/size |

Notes

* The generator's lognormal diameters, hard-core large-canopy-first placement and count correction are described in `../generation.json`; the quoted counts (≈ 25 073 BOER, ≈ 516 PROGLA) are estimates only.
* Initial ages follow `uniform_to_life_expectancy`: uniform from zero to each species' nominal life expectancy in `../species.json`. The existing `../preview/` is the earlier initialization with fixed ages; it is separate from these future inputs.
* The surface recipe values are MAPLE reference *computational* settings, not observed El Paso terrain. Hydraulic/soil layers are not sediment voxels.
* Do not hand-write a `plants.csv`: the generator is the only intended source so that ids, positions and anchors stay consistent.
* All paths in the case manifests resolve from the case directory.
