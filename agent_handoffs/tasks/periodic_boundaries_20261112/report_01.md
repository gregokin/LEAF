# LEAF periodic canopy and root boundaries — revision 1 — author report (Claude)

Task `periodic_boundaries_20261112`, baseline LEAF `85651c63003fcd0a66ba3a98af104003f64e818e`. Candidate left **uncommitted**.
**Nothing was executed by the author** (file tools only: Read / Edit / Write). Every check below is **NOT RUN**; no acceptance,
validated-ecology or performance claim is made. All source was read from `candidate/**` only; no file outside the 19 allowed paths
was written.

## 1. Changed files (19 allowed paths; all written)

| file | change |
|---|---|
| `leaf/state.py` | `Domain(..., *, periodic=(False, False))` (strict bools, per axis), `check_periodic`, `edge_convention`, `CELL_SAMPLING_PERIODIC`; `periodic` / `any_periodic` / `edge_convention` / `cell_sampling_convention` properties; `wrap_positions`; periodic-aware `cell_indices` (upper edge == lower edge, mod n), `to_dict` / `from_dict` / eq / hash / repr; plant companion `cell_sampling` uses the domain's convention |
| `leaf/periodic.py` (new) | shared boundary geometry: `min_image`, `boundary_record`, `self_overlapping` / `check_no_self_overlap`, exact `canopy_areas_m2` (torus union), exclusion bin layout / keys / neighbours, and the compiled-or-reference footprint kernels (`kernels`: count, root fill, canopy fill) |
| `leaf/initialization.py` | exclusion placement with wrapped distances + exact-tiling periodic bins + explicit self-overlap refusal; periodic ordering / placement / `boundary` report texts (nonperiodic report key set unchanged) |
| `leaf/canopy.py` | torus-union `measure_cover` (per-axis segment painting, no dense P x N), periodic-aware admission (32 B/index scratch), boundary notes / definition, `meta['boundary']` in `canopy_view` / `maple_vegetation_view` (periodic only) |
| `leaf/roots.py` | `horizontal_edges_periodic` (unique-cell toroidal footprint, uniform weights), wired into `build_root_map`; periodic law label + `boundary` provenance |
| `leaf/plant_pet.py` | periodic canopy count / fill through `periodic.kernels` using the existing exact `area` integrator; provenance `boundary` |
| `leaf/biocalendar.py` | canopy area per live plant = `periodic.canopy_areas_m2` (bit-identical `pi r^2` unless a circle exceeds half a periodic span); +32 B/live admission for periodic only |
| `leaf/plant_session.py` | `boundary()`; periodic-only `boundary` key in `export_roots` / `export_canopy`; docs (flags travel through `base.domain`; `pop_sha256` already hashes `domain.to_dict()`) |
| `leaf/plant_checkpoint.py` | `periodic` in `SOURCE_MODULES`; periodic-only `manifest['boundary']`; restore refuses flag mismatch / old manifest vs periodic file / non-strict flags; format version unchanged |
| `leaf/checkpoint.py` | `periodic` in `SOURCE_MODULES`; restore refuses `config.domain` vs plant-file flag mismatch; layout / version unchanged |
| `leaf/recruitment.py` | ONE change: `domain.wrap_positions(x, y)` on the newly computed daughter coordinate before the existing `contains` test; periodic caveat text (`CAVEATS_PERIODIC`) |
| `tests/test_periodic_boundaries.py` (new) | see §5 |
| `docs/periodic-boundaries.md` (new) | API / config / semantics / roots vs canopy / CPU-GPU-memory / restrictions / MAPLE-MASY handoff |
| `docs/phase0-vegetation.md`, `docs/masy-daily-plant-interface.md`, `docs/maple-leaf-integration.md`, `docs/species-root-profiles.md` | no longer claim all domains are nonperiodic; §4.1 concrete MAPLE handoff; §7b MASY propagation; benchmark addendum |
| `examples/periodic/generation.json` (new) | tiny 64 x 64 x 0.25 m periodic exclusion fixture using the existing phase0 species (no duplicated traits) |
| `periodic_report.md` | this file |

No change to `growth.py`, `uptake.py`, `benchmarks/*`, `interevent.py`, existing tests, locks, cases, `__init__`, orchestration or Jornada files.

## 2. Design and compatibility choices
* **One declaration**: `Domain.periodic = (px, py)`. Nonperiodic Domains serialise byte-for-byte as before (no new key, legacy `edges` /
  `cell_sampling`), repr unchanged; a periodic Domain adds `periodic` + per-axis texts. `from_dict` validates `periodic` strictly and
  checks the documentation keys against the DECLARED periodicity (legacy `edges:"periodic"` without a declaration is refused).
* **Upper edge**: periodic axis -> `floor((x-o)/dx) mod n` (x_max samples column 0); nonperiodic keeps closed-upper. Input positions are
  still validated against the closed domain; computed coordinates use `wrap_positions`.
* **Geometry** (all derived from minimum-image arithmetic, no images are materialised): the union of a circle with its images is the
  circle clipped to its nearest-image rectangle; the canopy fill therefore uses the EXISTING exact circle-rectangle integrator on
  (cell ∩ nearest-image rectangle) fragments, summed per wrapped cell (<= 2 fragments per axis), so oversized circles need NO size
  limit and are never double counted. Roots use the legacy positive-overlap predicate on the nearest image, one entry per unique cell.
* **Exclusion bins**: periodic axis uses `nb = max(1, floor(L/size))` bins of width `L/nb` (>= size) with deduplicated modular
  neighbours; nonperiodic axes keep the legacy bin edge and neighbour order; RNG draws are unchanged.
* **Backends**: periodic builds run on the CPU (Numba `kind='numba'`, lazily compiled, no disk cache; pure-Python twin `kind='python'`
  for `builder="numpy"`, slow reference). Maps then live on NumPy / Numba / CUDA exactly as before. No P-sized per-day transfer added.
* **Memory**: counts (8P) admitted over the live set before the count pass; counts + offsets + edges (8P + 8(P+1) + 16E) before the edge
  arrays; O(1) scratch per plant in the kernels; cover admission counts 32 B per raster index for periodic domains; no dense P x N.
* **Checkpoints**: manifest `boundary` only for periodic domains; old manifests mean nonperiodic and are never reinterpreted; source
  hash sets include `periodic`; no lock rewritten.

## 3. Laws preserved
Root law (circle radius = multiplier x canopy radius, UNIFORM weights over UNIQUE cells, layer fractions, plant-major CSR, normalised once),
canopy PET = area-weighted mean with the existing exact integrator (masked cells excluded, zero-support refusal, shared memory cap fix
untouched), proportional local competition / no priority / no compensation / accepted uptake drives growth (no change to `uptake.py` /
`growth.py`), lognormal sizes / independent ages / large-first exclusion / count fitting / bounds / failure-atomic generation, recruitment
probabilities, eligibility, timing, ID, age 0, surface sampling and parent-funded B / E.

## 4. Actual limitations / things to check
* **NOT RUN.** In particular the Numba typing of `periodic._make_kernels` (tuple returns of `axis_range`, bool arguments), the exact
  admission exception classes (`uptake.AdmissionError`) used by the new tests, and convergence of the fixture generation configs
  (`gen_config`, `examples/periodic/generation.json`) are unverified by execution.
* Square cells only for the root / PET builders (explicit refusal kept); non-square periodic domains work for sampling / cover / placement.
* `leaf.interevent.IntereventSession` (not editable here) builds roots through `leaf.roots` (so they wrap) but keeps nominal `pi r^2` canopy
  areas; `PlantSession` uses the union-aware area. `__main__.py` (not editable) cannot declare periodicity with `--nx/--ny/--dx-m`; use the
  config / domain JSON. Any other non-listed module that rebuilds a `Domain` from fields could drop the flags; I could not inspect them.
* Cell order inside a periodic root row is the unwrapped footprint order, not ascending cell index.
* `measure_cover` periodic path is an O(P) Python loop with small NumPy tiles like the legacy one; the periodic root / PET build with
  `builder="numpy"` is pure Python (slow by design).
* Existing-suite sensitivities to verify: `plant_checkpoint` / `checkpoint` source-hash expectations (new `periodic` member), anything
  asserting exact key sets of cover / canopy meta (I added keys ONLY for periodic domains), `repr(Domain)` / `Domain.to_dict()` of
  nonperiodic domains (unchanged by design).
* Old locks / qualification records are not rewritten; the changed modules need fresh qualification by Codex.

## 5. Tests — NOT RUN by me; suggested exact commands
New `tests/test_periodic_boundaries.py` (independent oracles: brute-force image union / min-image distances, the existing integrator on shifted
rectangles, a fine midpoint grid, tiled-image rasters, scalar allocator oracle): Domain declaration / strictness / serialisation / sampling /
CSV companion; scripted exclusion (seam, corner, mixed axes, tangent, awkward bins, self-overlap); generation cover vs tiled raster and no inset;
cover (seam / corner / mixed / oversized / translation / admission / canopy view); canopy PET (seam sums, varying PET, mask, mixed, translation,
oversized union, compiled parity, budgets); roots (brute force for all axis combinations and origins, species + individual layer fractions, unique
cells, wider than domain, nx/ny = 1, limits, no P x N memory, builder / backend parity); cross-seam uptake and competition; session rebuild and
growth flag propagation; checkpoint round trip with pending and wrong-boundary refusals; source hashes; recruitment daughters; optional CUDA
parity; optional real native N15 (skips only when `LEAF_NATIVE_*` is unset).

```sh
python -m unittest tests.test_periodic_boundaries -v
python -m unittest discover -s tests -v                                    # full baseline suite (existing tests untouched)
python -m unittest tests.test_phase0 tests.test_roots tests.test_plant_pet tests.test_plant_session tests.test_plant_checkpoint \
    tests.test_recruitment tests.test_initial_ages tests.test_spatial_mapping -v
python -m leaf generate --species examples/phase0/species.json --config examples/periodic/generation.json \
    --surface-constant 0 --out /tmp/periodic_plants.csv --report /tmp/periodic_report.json
# native (configured env): LEAF_NATIVE_BUNDLE=... LEAF_NATIVE_CASE_DIR=... LEAF_NATIVE_BACKEND=numpy|numba|cuda \
#   python -m unittest tests.test_periodic_boundaries.NativePeriodicN15Tests -v
```

## 6. MAPLE / MASY developer handoff (summary; full text in `docs/periodic-boundaries.md` §6 and `docs/maple-leaf-integration.md` §4.1)
Regenerate (when chosen) with the FULL-domain `Domain(nx, ny, dx, periodic=(True, True))` / `"periodic": [true, true]`; **remove the a2m
`placement_inset` and any inner-region cover correction**; one plant = one CSV row; MAPLE already wraps footprints / shelter (no new MAPLE work);
MASY passes the same flags to the LEAF `Domain`, LEAF rebuilds / restarts propagate them and checkpoints verify them (older checkpoints need the
explicit `SourceOverride`). No MAPLE case file, historical population or Jornada artifact is changed; the Jornada case stays finite, unrun and
unregenerated.
