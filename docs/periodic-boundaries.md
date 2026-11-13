# Periodic boundaries in LEAF (per-axis, opt-in) — revision `periodic_boundaries_20261112`

Status: interface and mechanics only. **The author executed nothing: every author check is NOT RUN**; independent execution
(CPU / Numba / CUDA / native N15) is recorded by Codex. Nothing here is an ecological validation, a calibration or a
performance claim. No equation of demand, allocation, growth, recruitment, mortality or water changed; SYRUP stays the sole
physical water authority.

## 1. Why, and what changed

A MAPLE case showed an artificial bare border: its wrapper placed plants in a 60 x 60 inset of a 64 x 64 domain because LEAF's
canopy exclusion was finite and non-periodic. MAPLE itself already wraps LEAF-imported canopy footprints and shelter across
periodic seams (a shrub at `x = 63.8`, `r = 0.5` is ONE plant whose footprint appears on both sides). LEAF now lets a domain
declare which axes are periodic and wraps ITS OWN geometry accordingly: generated placement, measured cover, canopy-area-weighted
PET and root footprints. The default is unchanged: finite, closed, non-periodic.

## 2. API and configuration

```python
from leaf import state
dom = state.Domain(64, 64, 0.25, origin_x_m=0.0, origin_y_m=0.0, periodic=(True, True))   # per axis: (x, y); default (False, False)
dom.periodic, dom.any_periodic, dom.edge_convention, dom.cell_sampling_convention
dom.wrap_positions(x, y)        # explicit wrap of NEWLY COMPUTED coordinates (periodic axes only; nonperiodic returned unchanged)
```

* `periodic` is one immutable keyword-only declaration. Only a list / tuple of exactly two real booleans (`bool` or `numpy.bool_`)
  is accepted; ints, strings, `None`, a scalar or a wrong length raise `StateError`. Mixed axes (`(True, False)`) and any origin work.
* `Domain.from_dict` accepts the optional key `"periodic": [bool, bool]` (generation config `domain`, plant-file companion,
  `--domain` JSON). `edges` / `cell_sampling` documentation keys, when present, must match the declaration; a legacy
  `"edges": "periodic"` **without** a valid declaration, or a finite `edges` beside `periodic: [true, ...]`, is refused.
* Serialisation: a fully nonperiodic Domain writes exactly the legacy dictionary (no `periodic` key, legacy texts; plain CSV +
  companion files and reports stay compatible). A periodic Domain adds `periodic: [px, py]` plus
  `edges` (`"periodic (torus) in x+y"`, `"periodic (torus) in x; finite closed non-periodic in y"`, ...) and
  `cell_sampling: containing_cell_periodic_axes_upper_edge_equals_lower_edge;nonperiodic_axes_closed_upper`.
* Equality, hash and repr distinguish periodicity (the nonperiodic repr is unchanged).
* **Upper edge (exact, canonical).** On a periodic axis the EXACT endpoint `x == Domain.x_max_m` (equality on the landscape
  coordinate; never a tolerance, never `x - origin` compared with the span, which rounds on decimal spacings / origins, e.g.
  `Domain(3, 2, .2, origin_x_m=.4, periodic=(True, False))` where `x_max = 1.0` but `1.0 - .4 = .6 < .6000000000000001`) is the SAME place
  as `x_min`: `cell_indices` returns index 0, `wrap_positions` returns the lower edge, the root / PET mappers use local coordinate 0.0 and the
  shared canopy export publishes center 0. Every other position keeps `floor((x - origin) / dx)` clipped to `[0, n-1]` — a real
  just-inside value stays in the last cell and keeps a public center strictly below `L`. Nonperiodic axes keep the closed-upper rule
  (last column) and plain `x - origin` centers.
* **Public centers are half-open.** On a periodic axis `canopy_view` / `maple_vegetation_view` centers lie in `[0, L)` (the MAPLE
  object container refuses an upper center); the authoritative `PlantState` / CSV coordinate is NOT changed and there is still one ID /
  one object per plant, no image record.
* **Validation.** Ordinary plant positions must still lie in the closed domain on every axis; nothing is silently normalised.
  A stored coordinate equal to the upper edge is legal (also when `x - origin` rounds beyond the span) and equivalent to the lower edge.
  Computed coordinates are wrapped explicitly with `Domain.wrap_positions` (a value already in `[min, max)` is returned unchanged; the
  exact upper endpoint — decided on the ORIGINAL input, since `v - origin` can round above the span before the modulo, and also on a result
  that rounded onto it — becomes exactly the lower edge; non-finite input is refused for every domain). In the canopy-PET builder a domain
  with any periodic axis validates BOTH axes on the original landscape coordinates against the Domain bounds (an outside point is refused
  before any canonicalisation or clipping), then canonicalises the periodic upper aliases and clips harmless rounding of the closed
  nonperiodic local coordinates to the span; the default nonperiodic branch is unchanged.
* Generation config (tiny runnable example `examples/periodic/generation.json`, the species file is the existing
  `examples/phase0/species.json` — no trait is duplicated):

```json
{"domain": {"nx": 64, "ny": 64, "dx_m": 0.25, "periodic": [true, true]}, "overlap": "disallowed", "seed": 20261112, "...": "..."}
```
```sh
python -m leaf generate --species examples/phase0/species.json --config examples/periodic/generation.json \
    --surface-constant 0 --out /tmp/periodic_plants.csv --report /tmp/periodic_report.json
```

(The `--nx/--ny/--dx-m` flags describe the default nonperiodic domain only; declare periodicity in the config or a `--domain` JSON.)

## 3. Semantics

### 3.1 One plant, one ID, one record
A plant has ONE centre in the closed domain, ONE ID and ONE CSV row. There are no image records, no coordinate shifts and no
duplicate IDs. All wrapping is arithmetic on distances / index ranges: the **minimum-image distance** `d - L * rint(d / L)` per
periodic axis (axes are independent, so the minimum over all 2-D images of the squared distance is the sum of the per-axis
minimum-image squares). The shared canopy view / MAPLE adapter export one object per plant with the stored centre and
`meta['boundary']` (`periodic`, `one_record_per_plant`); the consumer wraps the footprint itself.

### 3.2 Initial placement and measured cover
* Centres are sampled over the FULL domain: **no margin, no inset, no inner cover correction**. Draws are identical to the
  nonperiodic case (the same uniform x / y proposals, the same lognormal sizes, independent age draws, large-first order,
  count fitting, `max_attempts` / resource bounds, failure-atomic output).
* Default-compatibility: the default `measure_cover` / generation report keeps its legacy key set; periodic fields (`periodic`, `boundary`)
  appear only for a periodic domain.
* **Exclusion** (`overlap: "disallowed"`): reject when the shortest wrapped centre distance is `< ri + rj` (same strict rule: exact
  tangency is allowed) on periodic axes, ordinary distance elsewhere. The sparse hashed index uses `nb = floor(L / bin)` bins of
  EXACT width `L / nb` on a periodic axis (no short last bin) and deduplicated modular neighbours, so it is correct for any
  ratio of `L` to the bin width, never all pairs, and never treats an image of the same plant as a neighbour.
* A canopy whose diameter exceeds a periodic span overlaps ITSELF: with exclusion the generation **refuses** explicitly
  (`GenerationError`, before any proposal; nothing is rescaled, dropped or buffered). Overlap-allowed keeps independent uniform
  centres and the torus-union cover.
* **`canopy.measure_cover`**: a raster midpoint is covered when its minimum-image distance to a centre is strictly inside the
  circle, so seam fragments (corners included) enter the union once, however many images reach a cell, including the self-image of
  an oversized circle. Mixed axes clip only the nonperiodic axis. The nominal `sum(pi r^2)` is reported separately and **may exceed
  the physical torus union**; it is not a claim about cover. Reports / `boundary_note` / `definition` name the actual mode;
  a periodic report carries a `boundary` block, a nonperiodic report keeps its exact key set.

### 3.3 Canopy-area-weighted PET (`plant_pet`)
The SAME exact circle-rectangle integrator is used. A canopy that crosses a seam reads the terrain PET across it; each physical
plant-cell pair has ONE sparse entry whose area is the sum of the wrapped fragments in that cell; masked cells stay excluded
and the zero-valid-support refusal is unchanged; `area_total` / the PET denominator are the actual represented physical support
(for a small circle on a fully periodic domain the row sum is `pi r^2` to FP64 roundoff). For a circle wider than half a periodic
span the area per cell is the UNION: the circle clipped to its nearest-image (Voronoi) rectangle `|dx| <= Lx/2` (`|dy| <= Ly/2`),
which is exact because the nearest-image cells of a rectangular torus are axis-aligned rectangles — never the double-counted sum,
so no canopy size limit is needed. This union is the PET weight support only. **The biological demand area stays the nominal
allometric `pi r^2` for every boundary and every radius** (`biocalendar.rebuild_geometry` and the standalone `IntereventSession` use
the same law; no second demand law exists): for an oversized canopy the demand area may exceed the physical union while the PET
weights / denominator are the union. Mixed axes clip only the nonperiodic axis.

**Integrator tangency correction (revision 2).** The shared exact circle-rectangle integrator (`plant_pet`, nonperiodic and periodic,
Python and Numba) lacked the zero breakpoint when a rectangle side lies exactly on the circle (`y0` or `y1 == +-r`), so one midpoint
classified a whole interval: `area(r=.5, x=[-.5,.5], y=[-.5,.5])` returned 1 instead of pi/4 (a circle of radius .25 in a .5 cell: .25
instead of .19634954084936207). The breakpoint test now includes equality; same law, same integrator, every other input unchanged.
This rare legacy output correction changes `plant_pet` and needs fresh source qualification; no old lock is rewritten.

### 3.4 Root footprints (`roots`)
The law is unchanged: circular footprint of radius `lateral_radius_multiplier x canopy radius`; horizontal weight **uniform** over
the UNIQUE positively overlapped cells (closest point of the closed square strictly inside the circle; tangency excluded;
**not** area-weighted, not a measured root density); then multiplied by the species / individual layer fractions; plant-major CSR.
On periodic axes the footprint is the union / nearest-image support of the circle and its images: wrapped across the seam
(corners included), clipped only on nonperiodic axes, normalised ONCE over the unique reached cells. Every physical
plant-cell-layer pair occurs once — also for a root wider than the domain, `nx = 1` / `ny = 1`, or several images hitting one cell
(they saturate at the domain's cells; no image copy is allocated). Roots are NOT subject to any canopy size rule, and roots of
different plants overlap and compete across a seam exactly as inside the domain even when canopy overlap was disallowed.
Cells inside one plant row follow the unwrapped footprint order (south to north, west to east), not ascending cell index.

### 3.5 Water contracts (unchanged)
Root-weighted requests, proportional local competition with an equal fraction of requests, no species priority, no compensation
to wetter cells, accepted summed uptake drives growth, zero demand stays zero. Native SYRUP availability and the debit stay visible
at the MASY call site and authoritative exactly once. `growth.py`, `uptake.py` and `benchmarks/spatial_mapping.py` are unchanged.

### 3.6 Runtime, restart and recruitment
* Growth / season / rebind / restore rebuilds read the SAME `Domain` flags from the state (`biocalendar.rebuild_geometry`).
  `PlantSession.boundary()` (and a `boundary` key of `export_roots` / `export_canopy` for a periodic domain) reports them;
  `pop_sha256` binds the Domain including the flags and `geometry_sha256` the wrapped roots / canopy.
* Biological-only checkpoint (`plant_checkpoint`, format version unchanged): the flags live in `plants.metadata.json`; a periodic
  domain also writes `manifest['boundary']`. `load_biology` refuses a plant file whose flags differ from the manifest (an old
  manifest without `boundary` means nonperiodic — never silently reinterpreted), a caller `ExternalBindings.domain` with other
  flags, and a non-reproducing population / geometry identity. The standalone `leaf.checkpoint` (layout 4, `IntereventSession`)
  likewise refuses `config.domain` vs plant-file flag mismatch.
* `leaf.periodic` is added to both source-hash sets (`plant_checkpoint.SOURCE_MODULES`, `checkpoint.SOURCE_MODULES`); the changed
  modules differ from older checkpoints and need the explicit `SourceOverride` / `allow_source_change=True`; old locks are not rewritten.
* **Recruitment** (`leaf.recruitment`) changed in ONE place: a newly computed daughter coordinate is explicitly wrapped on each
  periodic axis (`Domain.wrap_positions`) before the existing inside-the-closed-domain test, so nonperiodic axes still reject
  outside positions after the bounded attempts and nonperiodic results/draws are bit-identical. Same single daughter ID, age 0,
  current-surface sampling and parent-funded B / E; no new probability, eligibility, seed bank, timing, damage or mortality.
  Seeds are placed inside the closed domain as before. (The periodic caveat text replaces the "never wrapped" one in the report.)

## 4. CPU / GPU / memory
* The periodic horizontal root build and the periodic canopy build run on the CPU with compiled count / fill kernels
  (`leaf.periodic.kernels('numba', ...)`, lazily compiled; `builder="numpy"` selects the identical pure-Python twin, an explicitly
  SLOW exact reference). The map is then resident on the chosen uptake backend (NumPy / Numba / CUDA). Daily demand and uptake do
  no plant-sized host / device transfer; the cold geometry rebuild transfers are the existing counted ones.
* Admission stays metadata-first and precedes allocation: counts (8P) over the live set, then counts + offsets + edges before the
  edge arrays exist; edge / byte caps are never raised and no plant is dropped. Kernels keep O(1) scratch per plant (no bounding
  grid, no image copies, no dense plant-by-grid buffer). `cover_admission` counts 32 bytes per raster index for periodic
  coordinate scratch (16 nonperiodic); the cover rasters are unchanged. The upper-endpoint canonicalisation uses vectorised boolean masks
  (<= 4 bytes per plant, admitted in the root `roots.horizontal_inputs` stage and the PET preflight; two bytes per plant and axis in the
  O(P) canopy export, which has no byte ledger); there is no Python population loop.
* Cell restriction: the compiled root / PET builders still require SQUARE cells (`dx == dy`), refused explicitly otherwise.

## 5. Remaining restrictions
* Square cells only for roots / PET; non-square periodic domains work for sampling, cover and placement but not those builders.
* The standalone `leaf.interevent.IntereventSession` (not edited) rebuilds roots through `leaf.roots` (so they wrap) and, like
  `PlantSession`, uses the nominal `pi r^2` demand area (no gap between the two paths).
* Boundary configuration is JSON only: the `--nx/--ny/--dx-m` CLI flags cannot declare periodicity (`__main__.py` is not part of this
  revision); use the generation config / a `--domain` JSON.
* Roots in a nonperiodic axis stay clipped at the finite edge exactly as before.
* Periodic cell order inside a root row is the unwrapped footprint order, not ascending.
* No MAPLE, SYRUP or MASY file is changed; the existing Jornada case stays finite, unrun and unregenerated.

## 6. Concrete developer handoff (MAPLE / MASY)
1. **MAPLE adapter step (required, upstream; not done here).** Codex's read-only inspection of MAPLE finds
   `src/maple/case_tools/importers/leaf_vegetation.py::leaf_domain` constructing a LEAF `Domain` WITHOUT flags, and
   `vegetation_from_leaf_population` requiring `Domain` equality (which now includes periodicity). MAPLE developers must derive the strict
   `(x, y)` boolean flags from the `GeometrySpec` boundary kinds and pass `periodic=(px, py)` through that helper, the imports and the
   population comparisons; otherwise LEAF periodicity is dropped or the import is rejected. MAPLE already wraps canopy footprints and
   shelter, so no wind / rasterizer work is needed.
2. To remove the artificial border of a periodic MAPLE case, regenerate the population (only when the developers choose to) with the
   FULL-domain periodic Domain: `Domain(nx, ny, dx_m, periodic=(True, True))` or `"periodic": [true, true]` in the generation / domain
   JSON. **Delete the wrapper's 2 m `placement_inset` AND the inner-area cover correction** at that time: the target cover is the periodic
   union over the whole domain and centres are sampled everywhere. Historical populations are not rewritten here.
3. One plant per CSV row; keep `plant_id` / `x_m` / `y_m` as stored. **Canonical export**: a periodic public center is half-open
   `[0, L)`, so a stored center exactly at the upper endpoint (the same place as the lower edge) is exported as 0 (the MAPLE object
   container refuses upper centers); the stored record is unchanged and there is still one object per plant.
   `canopy_view(...)['meta']['boundary']` / `['meta']['domain']['periodic']` confirm the declaration.
4. MASY: pass and check the shared boundary flags (the LEAF `Domain` of the base `PlantState` must equal the flags MAPLE / SYRUP use).
   `PlantSession.start` / growth / season / reproduction / restart rebuilds propagate them; the native lifecycle call sequence of
   `docs/masy-daily-plant-interface.md` §3 is unchanged (no new water step, hydrology, debit, climate file or schedule).
5. Rebuild / restart: checkpoints record and verify the flags; the module source hashes changed, so restoring an OLDER checkpoint
   needs the explicit typed `SourceOverride` (never relaxing identities). Historical populations are not rewritten.

## 7. Tests (NOT RUN by the author)
`tests/test_periodic_boundaries.py` (new; existing tests untouched): Domain serialisation / strictness / sampling, exclusion
(hand-scripted seam, corner, mixed-axis, tangent, awkward-bin and self-overlap cases), generation cover vs an independent tiled
image raster, seam canopy PET vs summed image fragments and a fine-grid union oracle, root footprints vs brute force, cross-seam
uptake vs a scalar allocator oracle, session / growth / checkpoint propagation and wrong-boundary refusals, source hashes,
recruitment daughters, parity (compiled vs reference, numba, CUDA) and an optional real native N15 test (skips only when
`LEAF_NATIVE_*` is not configured). Run: `python -m unittest tests.test_periodic_boundaries -v` plus the full baseline suite
(`python -m unittest discover -s tests -v`).
