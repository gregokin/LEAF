# Periodic LEAF geometry: MAPLE and MASY handoff

LEAF owns canopy/root geometry and its generator. SYRUP continues to own physical water. MAPLE already wraps canopy footprints and shelter; this task changes no MAPLE/SYRUP/MASY source. Independent acceptance and the qualified content identities are recorded in this task's final review/qualification, not inferred from an installed checkout's name.

## Declare one boundary choice

```python
from leaf.state import Domain

domain = Domain(nx, ny, dx_m, dy_m, origin_x_m, origin_y_m,
                periodic=(periodic_x, periodic_y))  # strict booleans, x then y
```

The default remains `(False, False)` and the old nonperiodic dictionary representation. Generation/domain JSON accepts `"periodic": [true, true]`; mixed axes work. MASY must derive these flags from its common scenario boundary declaration and check the other components use that declaration before starting. LEAF's water-availability object does not independently discover or configure native boundary conditions.

Use the same `PlantState.domain` for initialization, canopy views, roots and `PlantSession`. Its flags survive growth rebuilds, daughter wrapping and biological/standalone checkpoint metadata; changed flags cannot be silently substituted on restart. New source hashes need explicit compatibility qualification when loading older checkpoints.

## Generate over the entire domain

Use `leaf.initialization.generate_population(catalog, generation_config, surface=initial_surface)` with the FULL hydraulic dimensions/origin/spacing and desired periodic flags. Set `overlap: "disallowed"` for canopy exclusion across seams, including corners; or `"allowed"` for independent uniform centers. Measured cover is the physical periodic union.

For MAPLE's buffered historical case, remove the wrapper's 2 m `placement_inset` AND the inner-region cover correction when separately authorized to regenerate. Keep the requested domain cover; don't inflate it to compensate for a deleted inset. This task does not rewrite that historical population or the Jornada case.

## Connect the current MAPLE adapter

Read-only inspection of `/home/okin/MAPLE/src/maple/case_tools/importers/leaf_vegetation.py` found `leaf_domain(...)` still constructs a Domain without periodic flags, while `vegetation_from_leaf_population(...)` checks Domain equality. Derive the strict `(x, y)` flags from `GeometrySpec`'s `BoundaryKind.PERIODIC` choices and pass them through that helper, the population load and the comparison. See `maple_adapter_reference.json` for the actual inspected HEAD/file identity; its live dirty checkout was not the native test source.

Continue consuming `leaf.canopy.canopy_view(plants, catalog, current_surface)` (or the thin `maple_vegetation_view` policy adapter): ONE object per plant, stable ID sidecar. On periodic axes an exact stored upper endpoint exports as the equivalent lower-edge local center; the authoritative CSV/state coordinate stays unchanged. Consumers wrap the footprint, never create new plant records. Boundary metadata accompanies the view. No replacement wind/shelter physics is needed.

## Roots and daily water

`leaf.roots.build_root_map(plants, catalog, max_edges=..., max_bytes=..., backend=..., builder="compiled")` wraps footprints on periodic axes and clips only finite axes. It retains uniform weights over unique touched cells times the existing species/individual layer fractions. One physical plant-cell-layer pair appears once, even when roots exceed a domain span. Roots of different plants may overlap despite canopy exclusion.

Use the existing public `PlantSession` daily sequence in `docs/masy-daily-plant-interface.md`: native daily hydrology once; fresh availability; LEAF demand/proposal; native zero-time withdrawal; actual feedback; staged/published growth; combined checkpoint. The new boundary support adds no water operation. Potential demand retains the original nominal allometric canopy area; PET averages use the represented physical canopy union.

Production cold geometry builds use compiled CPU count/fill and bounded sparse edges; NumPy/Numba/CUDA uptake backends remain supported. Keep explicit byte/edge limits. Large roots can legitimately create many sparse edges, so caps still refuse oversized work; no dense plant-by-grid tensor or image population is introduced. Square cells remain required for root/PET builders.

## Canonical file-reader detail

For path-based input use `leaf.state.read_plant_csv(path, catalog, domain=domain, surface=initial_surface)`, which automatically verifies the companion when present. A text-only parse without the companion loses that file-level boundary/time/catalog binding. Pass the same native-derived Domain; reject a conflicting companion rather than silently relabeling an old finite population as periodic.
