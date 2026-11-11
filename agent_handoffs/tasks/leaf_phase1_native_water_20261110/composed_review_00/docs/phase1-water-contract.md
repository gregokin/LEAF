# LEAF Phase 1, part A: water contract, root maps, scheduler records, co-located species files

Modules: `leaf/uptake.py`, `leaf/roots.py`, `leaf/forcing.py`, additions to `leaf/traits.py`. Tests:
`tests/test_uptake.py`, `tests/test_roots.py` (incl. multi-file catalog), `tests/test_forcing.py`.
This part is infrastructure only: **no growth law, no temperature response, no demand formula, no driver**
and no change to the legacy coupling / STEP / benchmark code. `import leaf` and the Phase 0 modules remain
NumPy + stdlib only; numba / cupy are imported lazily and only when a backend is explicitly requested.

## 1. Water contract (`leaf.uptake`)

All amounts are **m3 of water for one timestep**. Native depth/volume/area conversion is downstream (SYRUP
side) and is never performed here.

| symbol | meaning | shape / unit |
|---|---|---|
| `D[p]` | plant demand for the timestep | `[P]` m3 |
| `w[p, r]` | root weight of plant `p` on resource `r = layer * n_cells + cell` | CSR `[E]`, row sum 1 |
| `A[l, c]` | water SYRUP makes available on layer `l`, cell `c` | `(L, N)` or `[R]` m3 |
| `Q[r] = sum_p D[p] w[p, r]` | local request | `[R]` m3 |
| `f[r] = min(1, A[r]/Q[r])` (0 if `Q = 0`) | shared fraction: identical for every plant on `r` | `[R]` |
| `proposed[r] = min(A[r], Q[r])` | LEAF's proposal | `[R]` m3 |
| `accepted[r] <= proposed[r]` | SYRUP's single answer (validated once) | `[R]` m3 |
| `delivered[p] = sum_e D[p] w[e] accepted[r_e] / Q[r_e]` | water the plant receives | `[P]` m3 |
| `S[p] = delivered / D` for `D > 0`, else **0** | growth satisfaction (explicit zero-demand policy) | `[P]` |
| `unmet[p] = D[p] - delivered[p]` | shortfall | `[P]` m3 |

Rules enforced: no species / size / order priority (everyone on a resource gets `f[r]`); **no root plasticity
or compensation** (a request on a dry, masked or unavailable resource with `A = 0` stays unmet; weights are never
renormalised or redistributed); water nobody requested stays with SYRUP (deeper residual water unused);
multiple plants share **proportionally to their local requests**, not equal litres. SYRUP owns the sole water
inventory and the accepted debit; LEAF holds no store and never decrements anything.

Required example: `D = 100`, weights `.5/.5` in two layers, `A = 25/200` → `Q = 50/50`,
`proposed = 25/50`, `accepted = 25/50`, `delivered = 75`, `unmet = 25`, `S = 0.75`, residual `0/150` unused.

### API

* `RootMap.from_arrays(ids, offsets, resources, weights, *, n_layers, n_cells, backend, max_edges, max_bytes, provenance=None)`
  validated plant-major CSR: int64 stable plant IDs `[P]` (unique), int64 offsets `[P+1]`, int64 flattened
  resource indices `[E]`, FP64 weights `[E]` (finite, `>= 0`; non-empty rows sum to 1 within
  `WEIGHT_SUM_TOL = 1e-9`, the catalog layer-fraction tolerance, never renormalised). Weights are fully general
  (non-separable per individual). Empty rows are **inactive roots**: positive demand on them is refused.
  Bool / float / unsigned-overflow IDs and indices, shape and index bounds are refused before any lossy cast.
  The map owns copies (host copies frozen read-only); failed construction leaves caller arrays untouched.
  Order of checks: metadata (dtype, ndim, shape) → admission → content (scalar reductions) → copy/transfer.
* `propose(root_map, demand, supply, *, max_bytes=None) -> UptakePlan` (`demand`, `request`, `proposed`,
  `fraction`, `summary`); `plan.proposed_grid()` gives the `(L, N)` view.
* `accept(plan, accepted, *, max_bytes=None) -> UptakeFeedback` (`accepted` clipped copy, `scale`, `delivered`,
  `unmet`, `satisfaction`, `closure`, `summary`); `plan.accept_all()` is the no-shortfall case.
  Acceptance validation: finite, `>= 0`, `<= proposed * (1 + acceptance_tolerance_rel)` with
  `acceptance_tolerance_rel = eps * (max edges on one resource + 2)`; material excess is refused, rounding-level
  excess is clipped and **counted** (`n_rounding_clipped`, `max_rounding_excess_m3`); water on a resource with
  `Q = 0` is refused. Delivered may exceed `D` only by the row-sum tolerance (`unmet` and `S` are clamped at
  that level, an implementation failure beyond it).
* Water closure: `sum(delivered) == sum(accepted)` within `closure_tolerance_m3 = eps * (2E + R + P + 8) * total`
  (counts the roundings in the chain; proportional to machine epsilon, never an enlarged tolerance). A violation
  raises `UptakeError`.
* Both operations are **pure and stateless**: no store, no receipt, no state publication (that is the later
  driver / native authority). Foreign objects (a non-plan passed to `accept`, arrays of the wrong namespace,
  wrong shapes) are refused.

### Backends

`numpy` (vectorised reference: bincount scatter/gather), `numba` (compiled CPU kernels), `cuda` (CuPy
RawKernel, FP64 atomics, compute capability >= 60). One algorithm: the per-edge arithmetic is written once as
plain Python kernels (`_k_scatter`, `_k_proposal`, `_k_scale`, `_k_gather`), compiled by numba, mirrored in
CUDA C (`-fmad=false`) and vectorised in the NumPy reference. CPU/CUDA: scatter Q → proposal → gather accepted
deliveries. No dense `P x L x N` tensor, no `P x P` matrix, no daily edge history; arrays are bounded by `P`,
`E`, `R = L * N`. A `cuda` map takes and returns CuPy arrays only (host demand / supply / accepted arrays are
refused: no implicit transfers); the host sees scalar reductions only (`.get()` / `asnumpy` of P/E/R arrays is
never performed per step; `RootMap.host_arrays()` is a labelled diagnostic helper). `backend_status(name)`
probes availability without raising; `BackendUnavailableError` is raised when an unavailable backend is
requested (no implicit fallback or device switch). The one explicit host → device load is
`RootMap.from_arrays(..., backend='cuda')` from host arrays (validated on host first) or `map.with_backend('cuda')`.

### Memory admission (precise scope)

Explicit `max_edges` / `max_bytes` on map construction; `propose` / `accept` reuse the map budget unless a
tighter `max_bytes` is passed. Every allocating operation estimates BEFORE allocating and raises
`AdmissionError(stage, reason, estimate)`:

| stage | owned | scratch (freed) | staging |
|---|---|---|---|
| `estimate_map_bytes` | ids 8P + offsets 8(P+1) + counts 8P + resources 8E + weights 8E | plant-of-edge 8E + cast copies 16E + row sums 8P + id sort 8P + resource edge counts 8R | owned again for a cuda load from host |
| `estimate_plan_bytes` | demand 8P + request/proposed/fraction 24R | supply copy 8R (+ 16E per-edge temporaries, numpy reference only) | – |
| `estimate_feedback_bytes` | accepted 8R + scale 8R + delivered/unmet/S 24P | accepted cast 8R + mask R (+ 32E, numpy reference only) | – |

The admission total is `resident (map [+ plan]) + estimate`. Caller input arrays are not counted (the caller
holds them); **no host/device mirrors are kept** (`mirror_bytes = 0`). Bounds are never raised and individuals
never discarded to pass; old data is preserved on refusal.

## 2. Root map builder (`leaf.roots`)

`build_root_map(state, catalog, *, max_edges, max_bytes, backend='numpy', builder='compiled', statuses=('live',), layer_fraction_overrides=None, provenance=None)`

* Geometry from the ONE shared allometry `leaf.canopy.derive_geometry` (root radius = multiplier × canopy
  radius); no duplicate allometry.
* **Horizontal law — labelled existing simple geometric approximation**: uniform weight over every cell the root
  circle positively overlaps (closest point of the closed square strictly inside the circle), clipped to the
  finite non-periodic domain, normalised once at build time. It reuses the compiled circle-overlap CSR builder of
  `benchmarks.spatial_mapping` (`builder='compiled'`, numba) or an arithmetic-identical NumPy reference
  (`builder='numpy'`). It is **not** measured root density and **not** area weighting.
* Vertical law: species `roots.layer_fractions` by default; `layer_fraction_overrides={plant_id: [L fractions]}`
  (all layers, finite, nonnegative, unit sum; IDs must exist and be selected). Edge order per plant is
  layer-major; zero-fraction layers produce no edges.
* Selection: `statuses=('live',)` by default (removed / dead_standing do not transpire); any explicit status set
  is mapped consistently; selected plants with zero root radius get inactive rows.
* Weights are fixed for the map's lifetime; growth-driven rebuilds are a later driver operation creating a NEW
  map. No renormalisation for dry / masked cells (see the contract).
* Non-square cells (`dx != dy`) are **refused before any allocation** (the shared helper supports square spacing;
  dx is never silently used for dy). Positions are domain-local (landscape − origin); cell index `row * nx + col`.
* Admission stages: `roots.preflight` (from P, N, L only), `roots.horizontal_compiled` (the benchmark helper's
  own estimate, reported verbatim; it includes a `(layers+1)*N` grid term) or `roots.horizontal_numpy`,
  `roots.expand_layers` (the expanded edge count `sum_p cells_p * positive_layers_p` admitted BEFORE the expanded
  arrays exist), then the `RootMap` admission. Everything is recorded in `map.provenance['admission']`.
* Custom general maps: build them with `RootMap.from_arrays` directly and skip this builder.

## 3. Scheduler timestep records (`leaf.forcing`, stdlib only)

`TimestepRecord(start, end, temperature_c, pet_mm_per_day=None, process_kind='interevent', gate='open', label=None)`

* `start`/`end`: offset-aware datetimes with a **fixed** UTC offset (`datetime.timezone`; naive datetimes and
  DST-capable zones refused), same offset, `end > start`.
* `temperature_c` (degC, finite, above absolute zero) is **required even when PET is explicit**;
  `pet_mm_per_day` is an optional, carried-through PET depth rate (mm/day); no PET or weather physics is computed.
* `process_kind`: only `'interevent'`; wind / rain / storm / water-event / recession kinds are refused (no
  catch-up over them). `gate`: `'open'` / `'closed'` (explicit process gate).
* Exact arithmetic: `dt_seconds_exact`, `dt_days_exact` are `fractions.Fraction` (microsecond exact); leap days,
  year boundaries and partial days come from the calendar itself.
* Helpers: `split_at_local_midnight()` / `local_midnights_between`, `split_at(boundaries)` / `split_record`
  (configured growth-update boundaries; pieces inherit the record values), `check_contiguous(records)` (one
  fixed offset, no gaps, overlaps or reordering), `records_from_weather_rows(rows, utc_offset, date_key='date',
  temperature_key='tmean_c', pet_key=None)` (whole local days from scheduler mapping rows; strictly consecutive
  zero-padded `YYYY-MM-DD` dates; nothing synthesised).
* No synthetic weather, no independent clock starting at 1 January, no temperature-response or growth formula
  (pending the science choice). The real MASY coordinator is unchanged; a future driver delegates to this.

## 4. Co-located species files (`leaf.traits`)

`SpeciesCatalog.from_files(paths)` and `load_catalog(path | [paths])`:

* every file is a complete, independently valid catalog (same schema / units / `root_layers`) holding one or
  more **complete** species records; species are merged whole, never split or overridden across files;
  a species ID in two files is refused even when identical; a path listed twice is refused;
* the merged species table is sorted, so the canonical hash is independent of file order and key order;
  `from_files([one])` is identical to `from_file(one)` (same hash, source, provenance);
* with several files, top-level `catalog_id` / `notes` are kept only when every file carries the same value
  (otherwise omitted, deterministically);
* paths are provenance **metadata only** (`catalog.sources`, `catalog.species_sources`, `metadata()['sources']`),
  never hash input. Per-species content, ownership (deep copies) and unknown-value validation are unchanged;
  no growth / temperature traits are added or activated.

## 5. Native-authority boundary (what this part does NOT do)

* No water store, inventory, debit, receipt or state publication: SYRUP owns water; the later driver publishes.
* No growth, temperature response, demand formula, mortality, plasticity, remapping or event coordination.
* No change to `leaf.coupling`, `leaf.stepwat`, `leaf.runtime`, `benchmarks/*`, old defaults, fixtures or
  existing tests.
