# Report — LEAF Phase 1 native water bridge, revision 2

Writer: Claude (Sonnet 5.5 alias, resumed from the saved Fable-5.1 session). Ordinary Read/Edit/Write only.
This is an independent subtask, not Phase 1 completion. Everything is uncommitted for Codex review.

## Files changed (exactly the four allowed writes)

* `leaf/syrup_water.py` — revision 2 of the saved bridge (same public API and pipeline; ledger, projection and
  conservation corrected).
* `tests/test_syrup_water.py` — new (the missing tests).
* `docs/phase1-native-water.md` — new.
* `report.md` — replaced (the previous text was a stale copy).

Not touched: `leaf/uptake.py` (corrected allocator, read-only), roots / forcing / traits, `native_reference/`,
`review_reference/`, benchmarks, any upstream source.

## What was corrected

1. **Memory ledger / admission (CUDA omission).**
   * Native column and parameter resident copies are now taken from the native authoritative
     `estimate_step_memory` components: mask/thickness copies (`column_bytes`), `SoilLayers` vectors
     (`layer_bytes`, host), and `KernelParameters` copies (host for numba, host AND device for cuda). The saved
     bridge omitted the kernel-parameter copies.
   * Population-/grid-dependent reducer scratch removed rather than allowance-padded: the bridge no longer
     calls CuPy `sum`/`max`/`stack` over `[P]` or `[L, N]`. Totals come from the allocator's packets
     (`plan.summary`, `feedback.summary`) and from a bridge CUDA projection kernel that block-reduces into a
     2-double packet (16 device + 16 host bytes). The NumPy projection no longer builds `[L, N]` temporaries
     (per-layer `[N]` masks, itemised); the Numba one uses `[2, N]` rows (itemised).
   * Ledger entries are tagged host/device/reserve and held/fresh (`fresh_*_bytes`), allocator host packet reads
     and the native host packet are listed, `allocator_peak_bytes` is checked against the root map's budget
     before any allocation, standalone `availability_snapshot` admission now counts the same native resident
     copies. No fixture-specific allowance, no raised cap.
   * I believe the 8,821-byte CUDA gap came from CuPy reduction workspace over `P`-sized `delivered` / `unmet` /
     `demand` in the old `_reduce`. This is an inference from the code; I could not run the hooked profile, so it
     is unverified.
2. **Conservation.** `requested_matches_proposal` is now enforced as one of eight named checks
   (requested == proposal, actual <= request, actual <= availability, snapshot == plan supply, accepted == native actual, delivered ==
   native actual, allocator closure closed, `delivered + unmet = demand` identity), and all finite checks.
   `_validate_native_result` rejects malformed native results (wrong type / dtype / shape / device, aliased
   arrays, changed or non-finite time, non-finite / negative budget fields, budget not closing in its own bound)
   before the allocator gather. Native physics and roundoff policy untouched; the accepted vector is passed to
   the corrected allocator unchanged.

## Revision 3 — physical debit verification

Confirmed hole (Codex): replacing only the returned state with a fresh copy of the original water still gave
`closed=True`, because only reported budget scalars were compared. Added `_verify_native_debit` in
`leaf/syrup_water.py`: a read-only fused pass (`_k_verify`, Numba-compiled; CUDA kernel `verify`; per-layer
NumPy path) over supplied water, returned water and native actual / unmet / scale / request arrays, with one
14-double packet (no `P`/`R` read-back, no library reducer). It rejects missing debit, extra loss, wrong-cell
debit, changed inactive cells, and non-finite / negative / over-saturated or inconsistent arrays; `_conservation`
now also requires the budget's storage / actual / unmet / requested totals to match the arrays and
`sum(before) - sum(after)` to equal the accepted volume. Its scratch/packets are in the public ledger
(`verify_scratch`, `verify_packet*`) before allocation. Existing request/proposal enforcement, the reduction fix
and native component estimates are retained.

Tests: the old drier-state shortfall test (which hid water loss) is replaced by
`test_accepted_shortfall_on_the_original_inventory_conserves_water` (real operator, ONE call, original water,
reduced request, only diagnostics rewritten, removed volume == delivery). New:
`test_physical_debit_is_verified_against_the_supplied_inventory`,
`test_plain_python_verify_kernel_matches_the_numpy_path`, `test_gpu_physical_debit_verification_rejects_tampered_inventory`,
and ledger assertions for the verify entries. Tolerances (`4 eps`, `8 eps`) are my choice from the native
`out = w - min(q, avail)` arithmetic; Codex should confirm they never false-reject real runs.

I did not Read `review_reference/*.py` (not needed for this change); no new tool errors this revision.

## Revision 4 — brittle test expectation corrected (tests only)

`test_native_requested_must_match_the_proposal` asserted the substring `native_requested_matches_proposal`, but its
halved-request fixture is now refused earlier by the physical validator (`bad_unmet`). The assertion now requires
only the refusal (`SyrupWaterError`), exactly one native call and unchanged inputs, independent of check order.
I also added a budget-only variant (only `requested_m3` doubled) that must likewise be refused. The physical
validator and `leaf/syrup_water.py` are unchanged. NOT RUN by me; Codex runs the real checks. No new tool errors.

## Read-only context used

Corrected `leaf/uptake.py` (typed arrays, `device_id`, packets, no accepted clipping, `NP_PACKET`/`NA_PACKET`,
`estimate_*`), `native_reference/subsurface*.py`, the earlier saved bridge. I did not open `review_reference/`
file contents (see Read errors).

## Tests NOT RUN

I cannot execute anything. NONE of the following was run by me; every test in `tests/test_syrup_water.py` is
unrun, as are `python -m unittest tests.test_syrup_water`, the existing LEAF suite, the real native CPU/CUDA
checks, the tracemalloc and CUDA memory-hook regressions (`test_cpu_fresh_allocations_stay_inside_the_declared_host_bound`,
`test_declared_device_bound_covers_the_hooked_live_peak`), numba compilation of `_k_project`, and the CUDA
projection-kernel compile. Syntax and API use were checked only by reading. Expect possible first-run fixes
(for example CuPy `MemoryHook` kwargs, tracemalloc slack of 128 KiB, NumPy/CuPy ufunc corner cases, exact
bit-equality assertions on the availability projection).

Known test-design points Codex should confirm: the tight allocator-budget test tightens a frozen `RootMap`
with `object.__setattr__` (test-only); the shortfall test hands the REAL operator a drier REAL state through a
narrow spy (halving the request is instead used to test rejection); the CUDA hook test allows 512 B pool
rounding per allocation; no test allocates on a second device, so cross-device affinity is covered only by
code review and namespace-mismatch tests.

## Ordinary tool errors (in-tree, not auth/rate/network)

* `Read` on the candidate root directory: `EISDIR: illegal operation on a directory` (revision 1 and again for
  `review_reference` in revision 2).
* `Read` of `README.md` and `docs/phase1-uptake.md`: `File does not exist`.
* `uptake.py` and `native_reference/subsurface.py` Reads were truncated by the tool's 25,000-token cap and
  continued with `offset`; no data was lost.
* Because directories cannot be Read and no listing tool exists, I did not inspect `review_reference/` contents.
* The earlier provider limit that ended revision 1 is separate from these tool errors.

## Remaining / for the dated driver

Atomic publication of native + biological state with an exactly-once interval receipt, combined restart,
weather/PET/temperature, growth, mortality, recruitment, root-map rebuilds and surface remapping are not part of
this subtask and are not claimed.
