# Water-vegetation projection (`leaf.water_vegetation`, Phase 2.5 P1)

> Current independently verified status (2026-11-14 UTC): package 38009872f8969f86b56c720adc1b5d9ffefc554fb9cacf90855d999db604530f, 28 files, external helper SHA-256 333921bb645d984fa8e7449619d03114262ee333c7fea61a22493a878fd9ccc5. Root verified 427 CPU tests (12 hardware skips), four actual-GPU controls (one two-visible-device skip), N=200000 host-peak admission, and the compact-stream quota controls. See water-vegetation-qualification.json. This qualifies the bounded projection/compact-helper interface; it does not accept a coupled water simulation or field calibration. Historical author-only notes below are superseded by this qualification.


LEAF turns the accepted canopy into per-cell **derived vegetation geometry** for SYRUP. It computes no water: the Tietjen matrix factor `g = (a1 + R1)/(a2 + R1)`, the preferential rate/receipt and all water stay in SYRUP and are never applied (or doubled) here.

## Approved rules (user decisions 2026-11-14)
* **Cover is an explicit add-and-cap of exact circle-cell areas** (not a geometric or random-overlap union): per species `f_s,c = min(sum_p area_p,c / A_c, 1)`; total cover `min(sum_s f_s,c, 1)`. Caps are counted in `diagnostics`.
* **Eligibility.** Splash cover: visible `live` + `dead_standing`. Evaporation shading and infiltration inputs: visible `live` only. `removed` never counts. A fully buried plant (burial >= intact height) has no visible cover; partial burial keeps the same circle; an elevated-base (eroded) canopy stays at its X, Y. Optical porosity is unused; no canopy-density factor.
* **Roots** are the fixed species `roots.layer_fractions`. `R1_c = sum_s f_live_s,c * rootfrac_s[0]` (all participating live species, whatever their receiving weights, no cap, no normalisation). A species with an ENABLED (positive-sum) schema-2 receiving-weight vector contributes `c_s,c = f_live_s,c * sum(rootfrac_s[1:])` (0 for one layer); an all-zero vector disables only that contribution. Weights are the RAW species trait (SYRUP normalises). A participating live species without the trait refuses (no default; `require_receiver_traits=False` for splash/shading/R1-only use). The background soil pathway is not a plant term.

## API
```python
p = wv.project_from_plant_session(session, current_surface, active_mask=None, max_edges=<ACTIVE edges>, max_bytes=..., reserve_bytes=0, available_bytes_fn=None,
                                  backend='numpy'|'numba'|'cuda', device=None, max_device_bytes=None, device_reserve_bytes=0, available_device_bytes_fn=None)
p = wv.project_from_initial_state(state, catalog, current_surface, ...)      # labelled `initial_state`, NOT a session receipt
p.splash_cover; p.shading_cover; p.matrix_root_index                         # (N,) sealed read-only views
ids, weights, contributions = p.preferential_inputs(max_bytes=..., reserve_bytes=0, available_bytes_fn=None)   # RAW weights + one-shot iterator of FRESH (N,) arrays, each admitted before it is allocated
p.species_cell_cover('grass', 'live'|'all', max_bytes=...); p.check_session(session, current_surface=..., active_mask=...)
p.identity (deeply frozen); p.identity_record() (detached copy); p.diagnostics; p.omitted_ids; wv.estimate_projection_bytes / estimate_device_bytes; wv.package_source_record()
```
Cell index is `row * nx + col`; `active_mask` is `(ny, nx)` bool, True = active (inactive output 0). Surface: C-contiguous float64 `(ny, nx)`.

## Geometry and masks (active-first)
`canopy.derive_geometry` (the one allometry behind `canopy_view`) gives radius, status and burial; the actual landscape positions feed the existing exact builder `plant_pet.build_canopy_map` (nonzero origin, periodic seams, wide circles via its torus union), unchanged. Before any footprint is built, the existing `plant_pet` count kernels (compiled / NumPy / periodic) count each plant's ACTIVE-cell overlap; plants with none are omitted from THIS projection only (`omitted_ids`; the plant is untouched) and the builder is then called with the mask and only the surviving rows, so no inactive footprint is built and `max_edges` bounds the ACTIVE edges (a second cold count pass is the declared cost). An empty or all-inactive case builds no edge. The required-trait check applies only to participating ACTIVE live species.

## Memory, immutability, identity
Every stage is admitted from metadata (N, P, S, L, E, M) before it allocates, with the reserve counted once per pool: validate (chunked finite check and hashes; the cold package and external-helper digests) -> snapshot (session; admitted before `snapshot_state()`, and dropped right after the selection) -> select -> count_pre -> count -> filter -> build -> aggregate -> reduce -> seal. `estimate_projection_bytes` is a LITERAL listing of the simultaneously live arrays of each stage (original / filtered / sorted / cast / immutable copies; unused references are released, raw outputs are sealed and released one at a time, the caps are applied in place); the measured NumPy peak (tracemalloc, warm, inputs built before the tracer) is bounded by it in the tested empty N=200000, multi-species, masked, periodic, compiled and session cases, and the only itemised slack is `OVERHEAD_SLACK_BYTES` (64 KiB of Python / metadata overhead, never an O(N/P/E/S L) term). Coefficients measured from the actual arrays: snapshot 64 B / row resident and 168 B / row peak, `derive_geometry` 176 B / row, 48 B per active edge of sort / group workspace. Streamed contributions require an explicit quota; each `next()` admits the new (N,) result + its species scratch + the previous yielded array when the caller still holds it (weak-referenced) + the raw weights, under the caller's reserve counted once; nothing is retained (no S x N); the raw weights are one fresh copy and the contribution is scaled in place. The enable test uses the maximum weight (no overflowing sum). Outputs are read-only arrays whose base chain ends in an immutable `bytes` object; identity and diagnostics are deeply frozen (`identity_record()` gives a detached copy); construction is private with a registry fingerprint checked in O(arrays) on every read (no rehash); forged, copied, replaced or uninitialised records and subclasses are refused; exact genuine `PlantState` / `PlantSession` / `SpeciesCatalog` types are required and never mutated.
Identity: session token / sequence / pop / geometry / lineage (or the labelled initial-state hash), catalog, domain, surface and mask hashes, the FULL LEAF package digest (SYRUP `source_tree_digest` algorithm over every regular non-bytecode file of the package: sorted `relative/path NUL file-sha256 LF`, `__pycache__` and `.pyc/.pyo` excluded, chunked and bounded, cold once per process; `file_count` and `algorithm` recorded) and a content hash of the outputs.

## CUDA implementation
Explicit device, no fallback; a separate device pool (`max_device_bytes`, `device_reserve_bytes`, `available_device_bytes_fn`) and the host pool are gated before any work; the CURRENT device must equal the requested one (no implicit switching); the cold H2D is five `asarray` calls of the sealed entry arrays, the three O(N) results live in ONE contiguous (3, N) device array and return by ONE `.get()` (no second host copy; identity records `h2d_calls=5`, `d2h_calls=1`; `estimate_device_bytes`), and the canonical outputs are HOST arrays (not resident GPU maps). One thread per cell, ascending species order, `__d*_rn` intrinsics, no atomics: bit-identical to NumPy / Numba. Hardware definitions: `tests/test_water_vegetation_cuda_definitions.py` (skipped unless `LEAF_RELEASE_GPU=1`); host-side static checks run in `tests/test_water_vegetation.py`.

## Source identity and external helpers
`package_source_record()` is the 28-file LEAF package digest (unchanged algorithm). The compiled `PlantSession` / canopy / root builders also import `benchmarks/spatial_mapping.py`, which lives OUTSIDE the package: `external_source_record()` locates it with `importlib.util.find_spec` (no import), hashes it once per process and `identity['external_helpers']` records its module, file name, SHA-256 and size separately; a missing helper refuses. The package hash is NOT extended over it. Both records are taken once per process (a file changed after first use is observed by the next process).

## Correction03 additions (final public compact API)
* **Session rows.** A genuine `PlantSession` snapshot materialises EVERY base row (live, dead_standing, removed): every stage is admitted from `session.base.n` metadata BEFORE `snapshot_state()`, and the actual snapshot must stay within that bound.
* **Scalar seal.** The producer fingerprint covers every scalar that affects arithmetic, shape, counts, the memory model or the identity export (`cell_area_m2`, `shape`, `n_cells`, `n_entries`, `n_plants`, `resident_bytes`, `species_ids`) by value, next to the sealed arrays and frozen metadata; a replaced scalar is refused on every public path without any content rehash.
* **`check_session`** validates a given surface / mask (exact type, dtype, shape, contiguity; no coercion) BEFORE a chunked memoryview hash (no whole-N copy); it returns a read-only dict of what was checked. With the surface and mask omitted it is BIOLOGY-ONLY and does not validate the current landscape (the SYRUP binding always supplies both).
* **`export_compact()`** gives read-only views (no copy, no S x N / P x N) of the O(N) fields, the cell-major entries, and the tables `root (S, L)`, `lower`, RAW `weights (S, L)`, `has_w`, `omitted_ids`, in the canonical `field_order`, with the scalars and the producer identity. It is an export, not a rehydration factory.
* **Content hash v3** (`VERSION = 3`) covers the canonical header (species, shape, cell area, counts, layers) and every array AND every table in the same order as the export, so changed root fractions, receiving weights or participation change the hash (an independent re-derivation from the export is tested).
* **Stream estimator.** `estimate_stream_bytes(...)` / `projection.stream_estimate(previous_held=...)` is the public model the stream itself admits: new (N,) result + species scratch + the previous yielded array when still held + the raw weights; the caller's reserve is counted once by the quota. Only the most recent yielded array is tracked; a consumer retaining several arrays must reserve them itself (the producer stores no history).

## Correction04 (CUDA host memory) and consumer composition
* **Shared host backing.** The CUDA reduction returns ONE contiguous (3, N) host array (single D2H) and the three raw outputs are its row views, so `_seal_arrays` can release the 24 N backing only after the LAST row is sealed. At the seal peak the base (24 N), the two already sealed rows (16 N) and the temporary copy (8 N) coexist: exactly **16 N bytes more** than the independent-array CPU path. `estimate_projection_bytes(..., backend='cuda')` itemises this literally (`shared_backing`); the NumPy / Numba model is unchanged (the seal stage of the validated N = 200000 empty case stays 8,065,650 B). **`backend` must match the projection** (the build admits the model of its own backend and records it in `diagnostics['declared_peak_bytes']`); a consumer that sizes a budget with `estimate_projection_bytes` must pass the same backend. Host-only controls exercise the real `_seal_arrays` on a real (3, N) NumPy shared base and the real `_build` with the device call replaced by a host function returning the same layout; they do not run or claim a CUDA backend. The hardware host-peak definition is in `tests/test_water_vegetation_cuda_definitions.py` (executed by Root under the qualification above).
* **Composition for S0.** The producer accounts ONLY its own arrays: the raw weights it returned, the new stream result + scratch and, when still held, the previous yielded array (weak-referenced). Every OTHER native buffer belongs in the caller reserve, counted once. A consumer that already counts a held array in its own reserve must release it before the next `next()`, otherwise the producer counts it again.

## Public pure compact helpers (V3 content unchanged)

Added for a restart record that carries `VegetationProjection.export_compact()` and must re-derive without a `PlantSession`, a live projection or a private mint. They are PURE
derived calculations (no producer proof, no source identity), CPU only, and every one takes an explicit host budget (`max_bytes`, `reserve_bytes`, `available_bytes_fn`), admitted
before the allocation. The V3 content algorithm is byte-identical (the producer and the helper call the same `_content_header` / `_content_digest`).

| helper | what it does | checks |
|---|---|---|
| `compact_content_sha256(compact, *, max_bytes, ...)` | H1: the exact V3 content hash of the mapping | COLD: metadata (keys, version, order, dtypes, shapes, counts) then chunked structural validation of every value, then the hash; a recorded `content_sha256` that differs is refused |
| `species_cover_from_compact`, `species_contribution_from_compact`, `preferential_inputs_from_compact` | H2: the per-species cover, `c_s = f_live_s * lower_s` and the one-species-at-a-time stream, with the SAME `_species_cover` / `_preferential_stream` arithmetic as the projection | PER CALL: metadata only; they rely on the same arrays having passed H1 in this process |
| `rederive_fields_from_compact(compact, *, backend, active_mask=None, expected_active_mask_sha256=None, ...)` | H3: splash / shading / R1 through the same `_reduce_numpy` / numba reduction, caps in place; returns a `RederivedFields` receipt (`genuine_projection` is False) | re-validates the structure (cold), the mask only when passed WITH its known sha (every entry must be active) |

The compact arrays' stored covers are never inputs of H3: the caller compares the fresh arrays with its sealed cached values. The LEAF package / external-helper record is NOT part
of the content hash: the owning SYRUP record compares the ACTUAL imported package digest with its external expected pins. Models: `estimate_compact_validation_bytes` (chunk-bounded,
independent of N and M), `estimate_rederive_bytes`, `estimate_stream_bytes`. `cuda` is refused (the CPU reference is for cold file validation). Tests:
`tests/test_water_vegetation_compact_helpers.py`.

### Correction 01 (compact helpers)

- The compact `version` is a closed integer (Python / numpy integer equal to 3): `3.0`, `'3'`, bool, `None` and every other value are refused by all four helpers before any hash or output.
- Nothing O(S) is allocated before the caller's admission: the metadata check only reads counts and loops over types (no set / list / tuple copy of the identifiers); the species-id
  uniqueness (a sort, 12 S bytes) and the species-table checks (row chunks) are part of the admitted cold validation. The compact stream counts the enabled species first and
  admits `weights + COMPACT_STREAM_INDEX_BYTES_PER_SPECIES * n` before it builds the id tuple / index list.
- The content header is streamed (`_content_header` yields byte pieces whose concatenation is exactly the canonical JSON); its transient is 3 copies of the longest identifier
  (`estimate_compact_validation_bytes(..., max_id_chars=...)`). The identifiers and arrays themselves remain caller-owned (no whole-process bound is claimed).

### Correction 02 (one stream model)

`estimate_stream_bytes` is the ONE model admitted by both `VegetationProjection.preferential_inputs` and `preferential_inputs_from_compact` (no context-only surcharge; the compact
`COMPACT_STREAM_INDEX_BYTES_PER_SPECIES` constant is removed). The stream counts the enabled species (allocation-free), admits `weights_stage_bytes` (= weights payload + the typed intp
index 8 n + the transient 8 n pointer list and the 8 n id tuple + `STREAM_FIXED_BYTES`) BEFORE any of them exists, then builds the index array by a cold loop (no Python-int list) and the
weights with a typed advanced index. Each `next()` admits `per_next_bytes` = result + scratch + (previous held) + weights payload + `retained_metadata_bytes` (index + id tuple + fixed).
Key deltas for a cache consumer: `weights_bytes` is the PAYLOAD; `weights_stage_bytes` (> payload) is the call's peak; `retained_metadata_bytes` is new; `per_next_bytes` now includes it.
`VegetationProjection.stream_estimate()` and `species_cell_cover` only COUNT enabled species (no index list). `STREAM_FIXED_BYTES` (8192) is the measured O(1) Python-object allowance
(never an O(S) term). Species-id strings are borrowed, never copied.
