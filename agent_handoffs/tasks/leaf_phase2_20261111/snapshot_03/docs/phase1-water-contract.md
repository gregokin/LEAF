# LEAF Phase 1, part A: water contract, root maps, scheduler records, co-located species files

Modules: `leaf/uptake.py` (v0.2.0), `leaf/roots.py` (v0.2.0), `leaf/forcing.py` (v0.2.0), additions to
`leaf/traits.py`. Tests: `tests/test_uptake.py`, `tests/test_roots.py` (incl. multi-file catalog),
`tests/test_forcing.py`. This part is infrastructure only: **no growth law, no temperature response, no demand
formula, no driver** and no change to the legacy coupling / STEP / benchmark code. `import leaf` and the Phase 0
modules remain NumPy + stdlib only; numba / cupy are imported lazily and only when a backend is requested.
Nothing here is a Phase 1 completion claim; the tests in this tree have not been executed by the writer.

## 1. Water contract (`leaf.uptake`)

All amounts are **m3 of water for one timestep**. Native depth/volume/area conversion is downstream (SYRUP
side) and never performed here.

| symbol | meaning | shape / unit |
|---|---|---|
| `D[p]` | plant demand for the timestep | `[P]` m3 |
| `w[p, r]` | root weight of plant `p` on resource `r = layer * n_cells + cell` | CSR `[E]`, row sum 1 |
| `A[l, c]` | water SYRUP makes available on layer `l`, cell `c` | `(L, N)` or `[R]` m3 |
| `Q[r] = sum_p D[p] w[p, r]` | local request | `[R]` m3 |
| `f[r] = min(1, A[r]/Q[r])` (0 if `Q = 0`) | shared fraction: identical for every plant on `r` | `[R]` |
| `proposed[r] = min(A[r], Q[r])` | LEAF's proposal | `[R]` m3 |
| `accepted[r]` | SYRUP's single answer, `<= proposed` (+ honest roundoff) | `[R]` m3 |
| `delivered[p] = sum_e D[p] w[e] accepted[r_e] / Q[r_e]` | water the plant receives | `[P]` m3 |
| `S[p] = delivered / D` for `D > 0`, else **0** | growth satisfaction (explicit zero-demand policy) | `[P]` |
| `unmet[p] = D[p] - delivered[p]` | shortfall | `[P]` m3 |

Rules enforced: no species / size / order priority; **no root plasticity or compensation** (a request on a dry,
masked or unavailable resource with `A = 0` stays unmet; weights are never renormalised or redistributed); water
nobody requested stays with SYRUP (deeper residual water unused); several plants share **proportionally to their
local requests**, not equal litres. SYRUP owns the sole water inventory and the accepted debit; LEAF holds no
store and never decrements anything. Required example: `D = 100`, weights `.5/.5` in two layers, `A = 25/200` →
`Q = 50/50`, `proposed = accepted = 25/50`, `delivered = 75`, `unmet = 25`, `S = 0.75`, residual `0/150` unused.

### API

* `RootMap.from_arrays(ids, offsets, resources, weights, *, n_layers, n_cells, backend, max_edges, max_bytes, provenance=None)`
  validated plant-major CSR: int64 stable plant IDs `[P]` (unique), int64 offsets `[P+1]`, int64 flattened
  resource indices `[E]`, FP64 weights `[E]` (finite, `>= 0`; non-empty rows sum to 1 within
  `WEIGHT_SUM_TOL = 1e-9`, the catalog layer-fraction tolerance; never renormalised). Weights are fully general
  (non-separable per individual). Empty rows are **inactive roots**: positive demand on them is refused.
  * **Typed arrays only** (`numpy.ndarray`; `cupy.ndarray` on the current device for `cuda`). Python sequences,
    ranges and scalars are refused without being iterated or converted, so no type information is lost (a mixed
    `[True, 2]` list never becomes an integer ID vector). Bool / float / complex / object dtypes and unsigned
    values above the int64 range are refused before any cast.
  * **Offsets are range- and order-checked before any subtraction**: `offsets[0] == 0`, `offsets[-1] == E`,
    every value in `[0, E]`, nondecreasing (direct comparison). Only then are `diff` / `repeat` / kernels run, so
    wrapped counts can never reach a kernel.
  * Order of checks: metadata (dtype, ndim, shape) → **admission** → content (scalar reductions) → owned copies /
    transfer. Nothing is copied, cast or transferred before admission passes.
  * The map owns frozen copies; failed construction leaves caller arrays untouched. `device_id` records the CUDA
    device (cuda maps).
* `propose(root_map, demand, supply, *, max_bytes=None) -> UptakePlan` (`demand`, `request`, `proposed`,
  `fraction`, `summary`); `plan.proposed_grid()` gives the `(L, N)` view. Refused: non-finite / negative demand or
  supply, positive demand on an edge-less row, a non-finite request `Q` (scatter overflow), non-finite totals.
* `accept(plan, accepted, *, max_bytes=None) -> UptakeFeedback` (`accepted`, `scale`, `delivered`, `unmet`,
  `satisfaction`, `closure`, `summary`); `plan.accept_all()` is the no-shortfall case.
* `RootMap.with_backend(backend, *, max_bytes=None)` rebuilds a validated map on another backend.
* Internal `adopt_owned_arrays(...)` (used by `leaf.roots`): validates FRESH owned contiguous host arrays in
  place and freezes them without copying; not for callers that keep references.

### Accepted feedback is the native debit

`accept` never modifies SYRUP's accepted amounts: `fb.accepted` is the supplied array, `scale = accepted / Q`,
and delivered water and the closure are computed against it. Excess over the proposal that is within the FP64
roundoff `acceptance_tolerance_rel = eps * (max edges on one resource + 2)` is **kept and reported**
(`n_roundoff_excess`, `max_roundoff_excess_m3`); a material excess, negative or non-finite values and water on
resources nobody requested are refused (nothing is clipped, no balancing flux is introduced). Per plant, delivered
water may exceed `D` only by `overdelivery_bound_rel = WEIGHT_SUM_TOL + acceptance_tolerance_rel + eps*(max edges
per plant + 4)`; inside that bound `S` is capped at 1 and `unmet` at 0 (`n_plants_clamped`,
`max_overdelivery_m3`), beyond it `accept` fails. Closure `sum(delivered) == sum(accepted as supplied)` must hold
within `closure_tolerance_m3 = eps * (2E + R + P + 8) * total` (eps-proportional, not enlarged) and the flag must
be true.

### Finite results only

Every operation reads back ONE small FP64 packet (`PROPOSE_PACKET`, `ACCEPT_PACKET`: counters, totals, maxima).
A non-finite packet, any non-finite or negative input count, a non-finite request / delivered value or total, or
a failed closure raises before a plan or feedback is returned (two finite `1e308` demands on separate resources
overflow the aggregate; two on one resource overflow the scatter: both refuse).

### Backends and synchronisation

`numpy` (vectorised reference), `numba` (compiled CPU kernels), `cuda` (CuPy RawKernel, FP64 atomics, block
reductions, compute capability >= 60). One algorithm: the per-edge arithmetic and the packet diagnostics are
written once as plain Python kernels `_k_*`, compiled by numba, mirrored in CUDA C (`-fmad=false`) and vectorised
in the NumPy reference. No dense `P x L x N` tensor, no `P x P` matrix, no daily edge history; arrays are bounded
by `P`, `E`, `R = L * N`.

* A **cuda map is bound to its device** (`device_id`). Every supplied CuPy array and the current device are
  checked against it before any allocation or launch; kernels are compiled per device. There is no device
  switch, peer access or fallback. Host arrays on a cuda map (and device arrays on host maps) are refused: no
  implicit transfers. The only transfers are the explicit loads (`from_arrays` / `with_backend`) and the
  diagnostic `host_arrays()`.
* Per operation, validation and diagnostics are batched into the packet, read back by ONE explicit
  synchronisation (`_sync_packet`); no P/E/R-sized readback per step. This does not mean the CUDA path is
  synchronisation-free: kernel launches are asynchronous and the packet read is the (single, explicit) boundary
  where numerical failures become observable.
* `backend_status(name)` probes without raising; `BackendUnavailableError` for an unavailable backend.

### Memory admission (precise scope)

Explicit `max_edges` / `max_bytes` for map construction; operations reuse the map cap, and an operation-level
`max_bytes` may only **tighten** it (a larger value raises). Every allocating public path estimates BEFORE
allocating and raises `AdmissionError(stage, reason, estimate)` with an itemised estimate (`owned`, `scratch`,
`staging`, `exclusions`):

| path | owned | scratch / coexistence |
|---|---|---|
| `estimate_map_bytes` | ids 8P, offsets 8(P+1), counts 8P, resources 8E, weights 8E | order mask, id sort, finite mask, plant-of-edge 8E, weight view 8E, row sums / deviation / mask, resource edge counts 8R (conservative sum); device staging = a second owned set for a host → cuda load |
| `estimate_plan_bytes` | demand 8P, request / proposed / fraction 24R, packet | supply copy 8R; numpy reference temporaries only (numba / cuda kernels are fused and allocate nothing else) |
| `estimate_feedback_bytes` | accepted 8R, scale 8R, delivered / unmet / satisfaction 24P, packet | numpy reference temporaries only |
| `estimate_conversion_bytes` | candidate owned set (source map stays resident) | none (values already validated) |

Admission totals add the resident map (and plan for `accept`). Scope: live algorithm arrays of this module.
EXCLUDED: interpreter objects, NumPy / CuPy allocator-pool overhead, numba JIT / CUDA context / compiled kernels,
and the caller's own input arrays. No host/device mirrors are kept (`mirror_bytes = 0`). Bounds are never
raised; individuals are never discarded to pass; old data survives a refusal. The measured-allocation regression
test (`TestNumbaParity.test_measured_feedback_scratch_stays_within_the_estimate`) checks the compiled paths; the
numpy-reference temporaries are bounded by the stated constants but are not separately measured.

## 2. Root map builder (`leaf.roots`)

`build_root_map(state, catalog, *, max_edges, max_bytes, backend='numpy', builder='compiled', statuses=('live',), layer_fraction_overrides=None, provenance=None)`

* Geometry from the ONE shared allometry `leaf.canopy.derive_geometry` (root radius = multiplier × canopy
  radius); no duplicate allometry.
* **Horizontal law — labelled existing simple geometric approximation**: uniform weight over every cell the root
  circle positively overlaps (closest point of the closed square strictly inside the circle), clipped to the
  finite non-periodic domain, normalised once at build time. It reuses the compiled circle-overlap CSR builder of
  `benchmarks.spatial_mapping` (`builder='compiled'`, numba) or an arithmetic-identical NumPy reference
  (`builder='numpy'`). It is **not** measured root density and **not** area weighting.
* Vertical law: species `roots.layer_fractions`, or `layer_fraction_overrides={plant_id: [L fractions]}` (all
  layers, finite, nonnegative, unit sum; IDs must exist and be selected). Layer-major edges; zero fractions give
  no edges. Weights are fixed for the map's lifetime; rebuilds are a later driver operation; no renormalisation
  for dry / masked cells.
* Selection: `statuses=('live',)` by default; selected plants with zero root radius get inactive rows. Non-square
  cells (`dx != dy`) are **refused before any allocation**.
* **Ownership**: the builder creates fresh contiguous int64 / float64 arrays and hands them to
  `adopt_owned_arrays` (frozen in place, no copy). State arrays, catalog tables and override containers are never
  aliased by the map.
* **Admission ledger** — each stage admits its transient allocations together with everything still live and
  refuses before allocating (reports in `map.provenance['admission']['stages']`): `roots.preflight`
  (mask, `derive_geometry` bound, catalog tables, map resource-scratch lower bound), `roots.select` (selected
  rows, geometry peak, extracted ids / radius / species index / layer table, including object-pointer arrays
  inside the geometry bound of 192 B per plant), `roots.overrides` (python + array scratch, local sorted ID lookup), `roots.horizontal_inputs`, the horizontal builder, `roots.horizontal_map`,
  `roots.expand_count`, `roots.expand_layers` (expanded owned arrays + 16 B per expanded edge of scratch + segment
  arrays + the live horizontal arrays) and finally the map's own validation scratch / device staging.
  * NumPy builder: a scalar-only pass finds the LARGEST bounding box and admits its exact scratch (25 B per
    bounding cell + 128 B per row and column) before any per-plant array; edge arrays are admitted after the count
    and before the fill.
  * Exported `horizontal_edges_compiled` (revision 4): typed, contiguous float64 1-D arrays only (sequences,
    other dtypes and non-contiguous views are refused without conversion); a scalar grid check and an admission of
    the shared estimator plus validation scratch (sort 16 B and masks 6 B per plant) run BEFORE its ids array or
    the shared helper's validation / sorting exist, so a zero-byte cap refuses before any P-sized allocation.
  * Overrides use a local sorted lookup over the selected IDs and never create the state's persistent by-ID index
    cache (`state._order` / `_sorted_ids`), so no newly allocated resident component escapes the ledger; a cache
    already held by the caller's state remains external and uncounted.
  * Compiled builder: counting avoids per-edge arrays; the helper's own pre/post-count admission is given the
    budget left after the other live work (it also counts a `(layers+1)*N` grid term it does not allocate, so it
    can only be stricter) and cells stay int32 (no conversion copy).
  * Scope exclusions: caller-held state / catalog / override containers, interpreter objects, allocator pools,
    numba JIT; `derive_geometry` is a black box bounded by 192 bytes per plant.
* Custom general maps: build them with `RootMap.from_arrays` (typed arrays) and skip this builder.

## 3. Scheduler timestep records (`leaf.forcing`, stdlib only)

`TimestepRecord(start, end, temperature_c, pet_mm_per_day=None, process_kind='interevent', gate='open', label=None)`

* `start`/`end`: offset-aware datetimes with a **fixed** UTC offset (`datetime.timezone`; naive datetimes and
  DST-capable zones refused), the same offset, `end > start`.
* `temperature_c` (degC, finite, above absolute zero) is **required even when PET is explicit**;
  `pet_mm_per_day` is an optional carried-through depth rate; no PET or weather physics is computed.
* `process_kind`: only `'interevent'` (wind / rain / storm / water-event / recession kinds are refused, no
  catch-up). `gate`: `'open'` / `'closed'`.
* **Precision scope**: microsecond-exact (`datetime`). `dt_seconds_exact` / `dt_days_exact` are `Fraction`s.
  MASY's rational **submicrosecond** instants are NOT accepted by this record: a future adapter must preserve them
  itself or refuse them before any mutation; `exact_microseconds(Fraction)` is the refusing conversion (a rational
  number of seconds that is not a whole number of microseconds raises; floats are refused).
* **Fixed UTC offsets with fractional seconds** are preserved exactly: metadata carries
  `utc_offset_seconds_exact` (Fraction text) beside the float `utc_offset_seconds`; nothing is truncated to whole
  seconds. Offsets that `datetime.timezone` cannot represent (|offset| >= 24 h) are refused.
* Helpers: `split_at_local_midnight()` / `local_midnights_between`, `split_at(boundaries)`, `check_contiguous`,
  `records_from_weather_rows(rows, utc_offset, date_key='date', temperature_key='tmean_c', pet_key=None)`.
  `parse_date` accepts exactly zero-padded ASCII Gregorian `YYYY-MM-DD` for years 0001-9999 (verified through
  `date.isoformat`, so `0099-03-01` works; no strftime padding, no `YYYYMMDD` / week dates). Dates whose next
  day is not representable are refused.
* No synthetic weather, no independent clock starting at 1 January, no temperature-response or growth formula.
  The real MASY coordinator is unchanged.

## 4. Co-located species files (`leaf.traits`)

`SpeciesCatalog.from_files(paths)` and `load_catalog(path | [paths])`:

* every file is a complete, independently valid catalog (same schema / units / `root_layers`) holding one or
  more **complete** species records; species are merged whole, never split or overridden; a species ID in two
  files is refused even when identical; a path listed twice is refused;
* the merged species table is sorted, so the canonical hash is independent of file order and key order;
  `from_files([one])` is identical to `from_file(one)` (same hash, source, provenance, numeric spelling);
* with several files, top-level `catalog_id` / `notes` are kept only when every file carries a value with the
  **same canonical serialization** (the text that is hashed). Python equality is not used: `1`, `1.0` and `true`
  compare equal but serialize differently, so keeping "the first equal value" would make the merged hash depend
  on the path order. Differing or missing metadata is omitted deterministically;
* paths are provenance **metadata only** (`catalog.sources`, `catalog.species_sources`, `metadata()['sources']`),
  never hash input. Per-species content, ownership and validation are unchanged; no growth / temperature traits.

## 5. Native-authority boundary (what this part does NOT do)

* No water store, inventory, debit, receipt or state publication: SYRUP owns water; the later driver publishes.
* No growth, temperature response, demand formula, mortality, plasticity, remapping or event coordination.
* No change to `leaf.coupling`, `leaf.stepwat`, `leaf.runtime`, `benchmarks/*`, old defaults, fixtures or
  existing tests.
