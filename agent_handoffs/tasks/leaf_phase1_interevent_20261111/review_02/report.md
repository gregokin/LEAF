# LEAF Phase 1 — dated native-water / growth session and complete restart (Claude, implementation writer)

## Revision 2 (guards and memory) — files changed: `leaf/interevent.py`, `leaf/checkpoint.py`, `tests/test_interevent.py`, `docs/phase1-interevent.md`, `report.md`

ALL tests NOT RUN (I cannot execute; Codex's results are Codex's). No tool errors recorded in this revision. Reviewer files
`review_reference/*.py` were read only to align with the reproductions.

1. **Scalar/global admission before copy.** `__init__` now validates `max_edges / max_bytes / reserve_bytes` (strict
   non-boolean ints, `reserve >= 0`) and the argument types, then admits the lower bound
   `(62 + 33) * P + 16 S + tables + native column copies + 2 * 8 L N + reserve` (metadata only; `check_operator` and
   `estimate_step_memory` allocate nothing) BEFORE `PlantState.validate` / `copy`. `checkpoint.load_session` validates the
   same scalars before touching the path. No budget is raised.
2. **Overrides.** `_setup` no longer coerces; after admission it calls `roots._validate_overrides(overrides, live_ids, L)`
   (bool / float / fractional / alias keys, bool fractions, wrong length, non-unit sums, unknown IDs all refused) and
   stores an immutable `{int: tuple(float)}` copy.
3. **Pending finite.** New `_accumulate_checked` (vectorised backend ops, scalar readbacks only, temporaries ledgered as
   `ACCUMULATION_CHECK_BYTES_PER_LIVE = 24` per live plant) validates D, U, A, T, totals, relations and duration before
   publication. NOT a fused numba / CUDA kernel (honest deviation from the preference; the check uses a few backend
   reductions). `_overdelivery_rtol(root_map) = uptake.overdelivery_bound_rel + 16 eps` is passed to `growth.grow`
   (`overdelivery_rtol=`) and used in the pending check and the restore check.
4. **Closure enforced.** `_check_native_step_budget` (finite, non-negative, recomputed residual inside the reported bound
   and equal to the reported residual) right after `ss.step`; the end-to-end residual must lie inside
   `tol_cont + nt.bound + tb.bound` or `IntereventError` is raised before publication. `residual_within_tolerance` is now
   always True in a published receipt (plus `total_tolerance_m3`).
5. **Native clock.** `TIME_TOL_ULPS = 8`: tolerance `8 * eps * max(|t0|, |t1|, dt)` (~1.8e-6 s at 1e9 s) replaces the 1e-9
   relative tolerance. Applies to the external rain / recession handoff only; the calendar cursor is unchanged and exact.
6. **Snapshot / materialisation ledger.** `MATERIALIZE_BYTES_PER_PLANT = 62 + 62 + 17 + 16 = 157` derived from
   `leaf.state` (`_arrays(copy=True)`, constructor `_coerce_*` copies, `np.unique` flatten + output, mask temporaries),
   `_materialize_peak() = 157 P_all + 16 P_live (age temporaries) + 16 P_live on CUDA (B / E host copies)`. Used by
   `snapshot`, `_build_geometry` (separate `_materialize` and `_geometry` stages with `VALIDATE_SCRATCH 33`), the growth
   rebuild preflight and `checkpoint.estimate_save_bytes`. Per-array bound, no fixture padding. At P = 100000 the structural
   additional bound is >= 15.7 MB (CPU) / 18.9 MB (CUDA) vs the reviewer's observed 13.2 / 14.8 MB lower bounds; I did not
   measure.
7. **Accumulation coexistence.** `_stages()['accumulate']` is now the full `estimate_bridge_bytes` total (which lists the
   availability, request-depth, accepted-volume buffers, native outputs, plan, feedback, staged state, demand, resident map,
   native copies and the session's other live bytes as `extra_reserve_bytes`) + new D / U (16 P_live) + check temporaries +
   old / new integrals (32 S). It is a conservative stage sum and labelled so; season carry and root graphs enter through the
   `carry` argument and the bridge ledger. No profiling was done by me.

New tests (class `GuardRegressions`): zero / bool / float / negative budgets refused with `PlantState.copy` and `validate`
observed not called; load scalar validation before path; the four coercion cases plus malformed rows, valid override
survival through rebuild and no alias; the actual one-plant 1e308 overflow fixture (day 1 ok, day 2 refused with the whole
boundary unchanged, native `step` / `withdraw_transpiration` called exactly once, valid zero-PET retry accounted once);
direct `_accumulate_checked` rejections; identical over-delivery tolerance passed to grow; misreported evaporation and
tampered withdrawal actual refused before publication (+ valid retry); late-clock half second and 1 ms refused while the
real native endpoint is accepted; snapshot ledger >= structural bound and refusal one byte short before `_arrays`;
accumulate ledger contains the bridge ledger. One test skips if `TranspirationResult` is not a dataclass.

Known risk: the old (revision 1) sections below describe the earlier state; where they differ (accumulate estimate, clock
tolerance, residual flag, overrides, snapshot ledger) this revision governs.


Status: implemented, **NOTHING WAS EXECUTED BY ME** (Read / Edit / Write only). Every test below is **NOT RUN**.
Left uncommitted. This is the remaining Phase 1 integration only; Phases 2 / 3 were not started. No field
calibration, no storm / wind physics, no full MASY / MAPLE claim.

## Files written (exactly the permitted set)

| file | content |
|---|---|
| `leaf/interevent.py` | `IntereventSession`, errors, boundary, global memory ledger, fingerprint helper |
| `leaf/checkpoint.py` | atomic directory checkpoint `save_session` / `load_session`, hashes, strict validation |
| `tests/test_interevent.py` | unittest cases against the REAL native API (NOT RUN) |
| `docs/phase1-interevent.md` | usage, units, ownership, restart scope, what stays upstream |
| `examples/phase1/run_interevent.py` | UNCALIBRATED example (explicit `--backend`, `--max-bytes`) |
| `README.md` | one added Phase 1 section |
| `report.md` | this file |

All other files (uptake, roots, forcing, traits, growth, state, canopy, syrup_water, native reference, existing tests,
`leaf/__init__.py`) were only read.

## Public API

`IntereventSession(plants, catalog, *, operator, native_state, start, backend, growth_cadence, max_edges, max_bytes,
reserve_bytes=0, root_builder='compiled', layer_fraction_overrides=None, root_map=None, verify_native_sources=True)`

* `step_interevent(record, *, added_water_m=None, potential_evaporation_m_s=None, interval_id=None)` -> receipt dict.
  Open gate: native `ss.step` -> `growth.demand` -> `syrup_water.withdraw_plants` once -> accumulate D / U / A / T ->
  at a boundary `growth.grow` once + root / canopy rebuild -> single publication. Closed gate: pause (forcing must be None).
* `flush_growth(interval_id=None)`; `advance_transport(kind, start, end, *, native_state=None, interval_id=None, label=None)`
  with kinds `wind` / `rain` / `recession`.
* `boundaries()`, `split_record(record)`, `elapsed_days()`, `snapshot()` (canonical `PlantState` at the cursor),
  `export_canopy_view(current_surface, **kw)`, `write_snapshot(writer, which)`, `describe()`, properties `cursor`,
  `native_state`, `root_map`, `live_rows`, `live_ids`, `season_due`, `pending_days`, `last_receipt`, `sequence`.
* `save(path, overwrite=False)` / `IntereventSession.load(path, catalog, *, operator, backend, max_edges, max_bytes,
  reserve_bytes=0, root_builder=None, verify_roots=True, allow_source_change=False, verify_native_sources=True)`.
* `leaf.checkpoint.save_session / load_session / estimate_save_bytes / module_hashes`, `CheckpointError`.
* Errors: `IntereventError`, `IntereventStateError` (gap / overlap / replay / pending growth), `IntereventBoundaryError`
  (unsplit record), `IntereventAdmissionError` (global ledger, carries stage / total / report).

## Design decisions and assumptions (please review)

* **Fixed growth grid.** Boundaries are `start + k * cadence` plus every local new year (year-end forces a boundary with
  `year_end=True`). A flush finishes a partial period WITHOUT moving the grid; transport skips crossed grid points
  (grid preserved, `growth_boundaries_skipped` reported). Alternative (re-anchoring after events) is a science / scheduling
  choice for Codex or the user.
* **Closed gate = pause**: no native step, no T, no growth, calendar and ages advance, pending growth must be flushed
  first (same as transport). This is my reading of "make its native hydrology behavior explicit and preserve mutual
  exclusion"; flagged as a decision, not a resolved science question.
* **Deferred season**: a year end inside transport / pause sets `season_due`; at the next OPEN interval a zero-water,
  zero-integral `growth.grow(year_end=True)` runs first (E removed, B reduced, roots rebuilt) inside the same transaction.
* **Ages** are derived (`age0 + exact elapsed days` for live plants); a restored session is re-anchored at the saved
  cursor, so a split run equals a continuous run to one FP64 addition (tests use `rtol`).
* **External root maps refused** (`root_map` argument raises): the session always builds species-separable roots with the
  shared builder; no silent replacement and no compatible-rebuild policy was invented.
* **Row mapping:** working arrays + root map contain LIVE plants only (state row order); all plants live in the private
  host base arrays (62 B / plant ledgered); dead / removed records are never dropped.
* **Root-map budget caveat:** `RootMap.max_bytes` (immutable) caps allocator operations. A rebuild receives
  `max_bytes - (everything else live incl. the old map)`, so the new map can carry a smaller cap than the first map. The
  session checks (before publication) that the allocator peak and the next interval's bridge ledger fit; if not, the
  rebuild is refused whole and the old boundary is kept. With tight budgets this can make weekly rebuilds refuse; this
  is the truthful consequence of old / new coexistence, not a bug I could remove without changing `uptake`.
* **Water closure:** per open interval the session checks native stage continuity (non-T storage after vs bridge storage
  before) within the MAPLE FP64 bound and reports the end-to-end residual `after - before - (added - return - drain - evap - T)`
  (`residual_within_tolerance` reported, continuity enforced).
* **Native provenance:** `native_provenance(verify=True)` is required to match the pinned hashes at construction
  (`verify_native_sources=False` relaxes); recorded in restart, checked again at load. If the installed SYRUP no longer
  matches `NATIVE_PIN`, construction refuses (a deliberate fail-closed choice; Codex may prefer a warning).
* **Sub-microsecond (MASY rational) instants** are refused by type: only fixed-offset `datetime` is accepted;
  `forcing.exact_microseconds` is the refusing conversion for adapters.
* Receipts keep ONLY current / last metadata (no trajectory list); large arrays are never retained (B / E are copied out
  of grow's shared `[12, P]` buffer; per-species diagnostics are `S`-sized sums).

## Memory ledger and transfers (limits)

One global `max_bytes` (host + device summed) + `reserve_bytes`. Pre-admitted BEFORE any work of an open interval:
native step (`estimate_step_memory`), demand (`growth.memory_estimate`), bridge (`estimate_bridge_bytes` with the
session's other live bytes as `extra_reserve_bytes`), accumulation (plan + feedback + candidates + native result
arrays), growth and the rebuild preflight; the rebuild itself, snapshot, transport validity / copy and checkpoint save /
load are admitted at their own stage. Components receive the actual remaining budget through `extra_reserve_bytes`.
Arrays a component already counts are excluded (D / U are double counted in a flush: labelled conservative). Excluded:
interpreter, JIT, CUDA context, allocator pools, caller-held arrays outside `reserve_bytes`. The weekly geometry path is
host based (explicit transfers of B / E and the CSR on CUDA; derive_geometry bound 192 B / plant is the builder's);
**it is not GPU resident**. The CUDA ledger sums device + host bytes (no separate per-side cap).

## Tests written (all NOT RUN)

`tests/test_interevent.py` (classes `SessionCore`, `RestartTests`, `NoPlantsAndPopulations`, numba subclasses, `CudaParity`):
daily and partial intervals; leap day / new year (365 / 366 `year_days`, month, half-day `dt_days`); weekly growth with
spies checking ONE `growth.grow` per boundary, accumulated D / U, actual satisfaction applied once, copy-out of B / E;
water-limited growth and retained support diagnostics; native closure and an independent native recomputation; species
and ID-keyed individual root profiles through a rebuild; arbitrary power-law allometry and canopy-area demand; unsplit
record / gap / overlap / replay / wrong-offset / wrong-shape / equal-id refusal with state fingerprint equality;
failure injection at native step, demand, bridge, native withdrawal, grow and root build with unchanged fingerprint and
single-accounting retry; global admission refusal before the native step; flush; transport needing a flush, ages, no
event catch-up, stale replay, rain needing the external native result, closed gate; deferred seasonal turnover (also
restored from restart); save / restore mid-week vs a continuous run; canonical CSV; atomic / no-overwrite publication;
corruption (npz bit flip, truncated CSV, edited manifest, missing file, consistent-hash but inconsistent content);
catalog / backend / native-fingerprint / source / builder / budget mismatch; no-live-plants run; CPU-CUDA parity (skips
without CUDA). Numba subclasses skip without numba. Tests use the NumPy root builder so numba is not required.

Known test-design risks (desk-checked only): some exact-equality assertions (restored vs continuous water / B / E) rely on
determinism of the NumPy paths; the admission tests use budgets of 1500 / 4096 bytes that should be below the real
ledgers for this tiny fixture; the failure-injection test for `withdraw_transpiration` assumes `st.TranspirationError`
may not exist (falls back to `RuntimeError`).

## Honest tool record

I recorded no tool errors (no auth, network or rate-limit failures); I made no guessed-path reads (only files listed in
`SOURCE_INDEX.txt` and the permitted writes). I never executed Python, so syntax / import / runtime errors are possible
(e.g. private helper names used from `leaf.syrup_water`: `_native`, `_check_state`, `_check_native_array`, `_bound_fn`,
`_BUDGET_FIELDS`; from `leaf.state`: `_arrays`; from `leaf.uptake`: `RootMap.host_arrays`, `estimate_*`). Codex should run
`python -m unittest tests.test_interevent` (CPU, then numba, then CUDA) plus the independent closure / restart / memory
checks. The example script is also unexecuted.

## Not done / upstream

Storm / wind physics, E partitioning, rational MASY instants, surface remapping, mortality / disturbance, custom
non-separable roots, a MASY-level coupled restart, GPU-resident geometry, field calibration.
