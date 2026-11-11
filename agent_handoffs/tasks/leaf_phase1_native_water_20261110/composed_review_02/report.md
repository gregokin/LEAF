# LEAF Phase 1, bounded part A — candidate report, revision 3 (Claude, implementation writer)

Scope honoured: writes limited to `leaf/uptake.py`, `leaf/roots.py`, `leaf/forcing.py`, `leaf/traits.py`,
`leaf/__init__.py` (unchanged since revision 1), `tests/test_uptake.py`, `tests/test_roots.py`,
`tests/test_forcing.py`, `docs/phase1-water-contract.md`, `report.md`. Nothing in `coupling` / `stepwat` /
`runtime` / `canopy` / `state` / `benchmarks` / existing tests / fixtures / `pyproject.toml` was touched. No growth
law, no temperature response, no native water physics, no driver. Tools used: Read / Edit / Write only. Code is
left uncommitted. **This is not a Phase 1 completion claim.**

## Revision 4 (exported-helper pre-admission) — files changed: `leaf/roots.py`, `tests/test_roots.py`, docs, report

* `horizontal_edges_compiled` (exported) now requires typed C-contiguous float64 1-D arrays (no sequence
  materialisation, cast or copy) and runs scalar / grid / admission preflight (shared `estimate_bytes` total plus
  22 B per plant validation scratch: sort 16 B, masks ~6 B) BEFORE creating its ids `arange` or calling the shared
  helper, so a zero-byte cap refuses before any P-sized allocation. Shared builder physics and upstream benchmark
  untouched. Regression: `test_exported_compiled_helper_preflights_before_any_p_sized_allocation` wraps
  `np.arange` with a real function (no mock object) and asserts no 8P-byte array is created under max_bytes 0 / 1000;
  also non-contiguous / float32 / list inputs refused. (The earlier MagicMock harness failure was Codex's
  instrumentation, not a product failure.)
* `_ensure_index` cache: overrides no longer call `state.indices_of`; `_validate_overrides` uses a local sorted
  lookup over the selected IDs (admitted as `local_sorted_lookup` in `roots.overrides`), so the 16P persistent
  cache is never created. Test `test_overrides_leave_no_persistent_index_cache_on_the_state` checks `state._order`
  and `_sorted_ids` stay `None`. A cache the caller already holds stays external/uncounted (documented).
* Tests: NOT RUN (I cannot execute). Codex's executed revision-3 results (181 CPU pass / 10 skips, 21 CUDA allocator
  tests, 35 bridge tests, four NumPy allocation workloads with no undercount) are Codex's, not mine; my earlier
  desk concern about the NumPy reference estimate is therefore not a confirmed defect. Tool errors in revision 4:
  none.

## Tool-error record (not erased)

* Revision 1: nine ordinary Read tool errors, each literally `File does not exist. Note: your current working
  directory is /home/okin/leaf/agent_handoffs/tasks/leaf_phase1_20261110/candidate.` They were guessed paths
  within scope (e.g. `tests/test_traits.py`, `test_canopy.py`, `test_state.py`, `conftest.py`, `README.md`,
  `benchmarks/__init__.py`, `docs/shared-canopy-view.md`), not auth / network / rate limits. My earlier report's
  "no tool failures" claim was wrong.
* Revision 2: an explicit Fable usage limit interrupted the session after only `leaf/uptake.py` was saved. This
  revision (3) was continued by the Sonnet alias at the user's approval. No other tool, auth, network or rate
  failures occurred in revision 3 (I made no guessed-path reads).

## State of each file

| file | status in revision 3 |
|---|---|
| `leaf/uptake.py` | **Saved in revision 2, NOT rewritten here.** Codex independently verified it (exact rational cases on NumPy / Numba / CUDA, offset overflow, non-finite totals, pre-copy admission, typed inputs, unmodified accepted amounts, CUDA affinity). Internal `adopt_owned_arrays` is its trusted ownership transfer. |
| `leaf/roots.py` | Rewritten: integrates `adopt_owned_arrays` (the removed `_take_ownership` argument is gone), a ledger that admits every stage over the live set, bounded NumPy counting scratch, int32 cells kept for the compiled path, in-place layer expansion with exact scratch accounting. Root physics unchanged. |
| `leaf/traits.py` | `from_files` compares top-level `catalog_id` / `notes` by canonical serialization (omit if they differ or are missing). Docstring updated. |
| `leaf/forcing.py` | `parse_date` via `date.isoformat` + fixed-shape pattern; exact fractional UTC-offset metadata; `exact_microseconds` refusing submicrosecond rationals; overflow-safe weather-row conversion and midnight iteration; documented microsecond scope. |
| `leaf/__init__.py` | Unchanged since revision 1 (lazy exports `uptake`, `roots`, `forcing`). |
| `tests/*`, docs, report | Updated to the actual API (see below). |

## Corrections in this revision

1. **roots.py ownership + admission.** `build_root_map` creates fresh C-contiguous int64 / float64 arrays and
   calls `uptake.adopt_owned_arrays(..., resident_bytes=live + 8K)`; the arrays are frozen in place, no state /
   catalog / override aliases survive. A `_Ledger` tracks live arrays by name and every stage (`roots.preflight`,
   `roots.select`, `roots.overrides`, `roots.horizontal_inputs`, horizontal builder, `roots.horizontal_map`,
   `roots.expand_count`, `roots.expand_layers`, then the map's own scratch/staging) admits `transient + live`
   before allocating. Counted: status mask, geometry bound (192 B per plant including the object-pointer arrays
   `species` / `status_name`), catalog tables, extracted ids / radius / species index / `[K, L]` layer table,
   override python / array / membership scratch and the state's cached by-ID index, active index and local
   coordinates, horizontal arrays, per-plant counts / offsets, expansion arrays and temporaries, old + new maps
   (`adopt` takes the live owned arrays as resident bytes), and cuda staging.
2. **NumPy counting scratch pre-admitted.** A scalar-only pass computes the largest bounding box; its exact scratch
   (25 B per bbox cell + 128 B per row and column) is admitted before any per-plant bounding-grid array; edge
   arrays are admitted after the count and before the fill. Counting uses a mask-only pass (no index grid).
3. **Compiled builder.** Cells remain int32 (removed the 8E conversion copy); the helper is called with the budget
   left after other live work, so its own pre/post-count admission accounts for live arrays (it also counts a
   `(layers+1)*N` grid term it does not allocate: stricter, reported verbatim).
4. **Canonical metadata merge** (`traits.from_files`): `1`, `1.0`, `true` (Python-equal) now omit the metadata
   deterministically; identical canonical text is kept; single-file hash unchanged.
5. **Forcing**: years below 1000; exact fractional offsets (`utc_offset_seconds_exact` + float); `exact_microseconds`
   refusal; `fixed_offset` still refuses |offset| >= 24 h; representability overflow refused as `ForcingError`.
6. **Tests updated** for typed arrays, new diagnostic names (`n_roundoff_excess`, `max_roundoff_excess_m3`),
   unmodified accepted amounts, packet-based plain-Python kernels, op-budget tightening, `adopt` ownership; the CUDA
   excess test now uses `pg.proposed * 1.01` (genuinely above the PROPOSAL), not an already-reduced accepted vector.

## New / retained regression tests (all NOT RUN)

* offsets `[0, INT64_MAX, INT64_MIN+1, -1, 0]` with E = 0 refused at construction (host and device), plus
  out-of-range / decreasing offsets; unsigned-overflow offsets;
* two finite `1e308` demands on separate resources (aggregate overflow) and on one resource (scatter overflow)
  refused on NumPy, Numba, CUDA; inputs preserved; large-but-finite case closes and reports finite totals;
* tracemalloc pre-copy checks: `with_backend(max_bytes=0)`, source+candidate coexistence, `max_edges=0`,
  `max_bytes=0` on a 20 000-plant map allocate far below the ~0.8 MB of arrays; op budget may only tighten;
* typed arrays only: `[True, 2]`, tuples, ranges and a trap sequence that fails if iterated / converted;
* measured compiled `accept` / `propose` allocation (P = E = 100 000, R = 1) stays within the estimate
  (Numba only; skipped without numba);
* honest roundoff: accepted one ulp over the proposal is kept unmodified, counted, closure against it;
  material excess refused; CUDA device-id guards (pure fake-device test + real current-device test; no allocation
  on other devices);
* roots: ownership / no aliasing, NumPy bounding scratch refused before arrays (stage name + tracemalloc),
  edge arrays refused after the count, compiled-builder preflight, cuda builder (skipped without cuda);
* traits: reversed paths with notes `{'value': 1}` / `{'value': 1.0}`, `True` / `1`, `0` / `False`, key-order
  differences, one file without notes, catalog_id case; single-file spelling preserved;
* forcing: `0099-03-01`, `0400-02-29`, strict shape rejections (`20240229`, week dates, full-width digits),
  `9999-12-31` refused, `+05:30:15.5` offset exact, submicrosecond refusal.

## Limits and residual risks (for Codex review)

* The numpy **reference** temporaries of `estimate_plan_bytes` / `estimate_feedback_bytes` (the non-compiled path)
  are bounded by stated constants (`reference_*_temporaries`) that I did not measure and did not touch in this
  revision; my own desk audit suggests the per-resource constant for `accept` (38 B/R) may be slightly low for the
  NumPy `_np_scale` peak. The compiled and CUDA paths are fused and carry no such temporaries (Codex measured them).
  Worth an explicit measurement before relying on the numpy backend for very large R.
* `derive_geometry` is bounded, not measured (192 B per plant): it is a shared Phase 0 routine, unchanged.
* The compiled builder's grid term (`(layers+1)*N*8`) is counted by the shared helper although never allocated here.
* `state._ensure_index` caches a 16 B-per-plant by-ID index on the caller's state when overrides are used; it is
  counted in `roots.overrides` but persists on the state.
* Resource indices are int64 in `uptake`; the compiled horizontal helper keeps its own int32 `nx*ny` guard.
* GPU atomics make `Q` order-nondeterministic; CPU / GPU parity is tolerance based.
* Existing `traits.metadata()` gained keys `sources`, `species_sources`, `provenance_note` in revision 1; unchanged.
* `forcing` refuses zone-based tzinfo by design; submicrosecond MASY instants are explicitly out of scope here.

## Tests: NOT RUN

I cannot execute anything (Read / Edit / Write only). **All tests are reported as NOT RUN**: `tests/test_uptake.py`,
`tests/test_roots.py`, `tests/test_forcing.py` and the existing `tests/test_phase0.py`. Nothing was imported,
compiled or installed. Everything above is desk-checked against the code only. Codex's earlier executed results
(166-test CPU suite, exact rational oracle cases, CUDA allocator checks) belong to Codex and to the previous tree,
not to this revision; the whole completed tree needs a fresh independent freeze and run.
