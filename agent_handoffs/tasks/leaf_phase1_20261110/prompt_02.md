# LEAF Phase 1 part A — revision 2, independent-review corrections

Resume exact session `fd82b85e-8a07-4982-b621-b132f845eab7`. You remain the sole
implementation writer in candidate/. Codex owns controls, execution, review and
publication. User authorization for automatic relevant sharing with Anthropic
and correction loops remains in force. This is infrastructure, not complete
Phase 1; growth/temperature/response laws remain outside this assignment.

## Unchanged boundaries

READ only this prompt and candidate/**. Ordinary Read/Edit/Write only; no Bash,
execution, agents, network, settings, installs, Git, upstream or archive reads.
Allowed WRITES exactly:

`leaf/uptake.py`, `leaf/roots.py`, `leaf/forcing.py`, `leaf/traits.py`,
`leaf/__init__.py`, `tests/test_uptake.py`, `tests/test_roots.py`,
`tests/test_forcing.py`, `docs/phase1-water-contract.md`, `report.md`.

Preserve all other existing modules/tests/fixtures/benchmarks. Do not implement
growth or native water physics. All scientific water rules in revision 1 remain
binding: normalized fixed roots, proportional local demand sharing, actual
accepted feedback, no compensation or second water inventory. CPU/GPU hot loops
compiled; no population-sized daily readbacks or automatic fallback.

Actual candidate files include tests/test_phase0.py, examples/phase0/species.json,
docs/shared-canopy.md and docs/stepwat-growth.md. There is no tests/test_traits.py,
test_canopy.py, test_state.py, conftest.py, README.md, benchmarks/__init__.py or
docs/shared-canopy-view.md. Do not guess/read nonexistent alternate paths again.

## Independent evidence (Codex executed, not claims for your test report)

The completed tree was frozen: full CPU suite ran 166 tests, 158 passed and 8
CUDA-related skips. Seven exact rational hand-oracle cases passed NumPy and
Numba: user 75% example, unequal roots/competition, native partial acceptance,
nonseparable 3-layer/2-cell roots, unavailable resources, empty/zero demand.

Actual CUDA allocator checks compiled and reached numerical parity, then failed
the test at tests/test_uptake.py:420: `accept(pg, cp.asarray(acc)*1.01)` did not
raise because acc had already been reduced by random fractions, so 1.01 times
it did not necessarily exceed the proposal. Correct the test to use genuinely
excessive proposal-based data. Backend/device-content tests after this point
have not yet completed. Codex will rerun the complete corrected tree.

## Required correctness fixes

1. **Reject overflowing/invalid CSR offsets before subtraction.** With E=0,
   offsets `[0, INT64_MAX, INT64_MIN+1, -1, 0]` currently pass. `diff` wraps to
   positive counts despite negative and out-of-range offsets. Explicitly check
   all offsets in `[0,E]` and direct monotonic comparisons BEFORE subtraction,
   repeat or kernels. A construction-only test must reject this map; never run
   a numerical kernel on malformed bounds.

2. **Reject nonfinite intermediate arrays, totals and closure.** Two plants on
   separate resources, finite D=A=1e308 each, currently return feedback with
   accepted/delivered totals inf, residual NaN and `closure.closed=false` because
   `abs(NaN)>inf` is false. Another case can overflow a shared resource's Q.
   Fail before returning a plan/feedback (and before a proposal could reach a
   native authority). All relevant computed arrays, aggregate packets, bounds
   and closure must be finite; the closure must be true. Add both aggregate and
   scatter-overflow cases and preserve inputs. Do not loosen tolerances.

3. **Truthful memory admission, all allocating public paths.** On warmed Numba,
   P=E=200000 and R=1, `estimate_feedback_bytes` predicts 4,800,025 additional
   bytes, but tracemalloc measured 8,203,504 operation peak (preexisting map,
   plan and caller arrays excluded). Admission accepted a full cap14,400,057
   despite more than3 MB unreserved array-scale scratch. Fuse plant output /
   validation arithmetic or account for simultaneous P-sized temporaries.
   Audit map/plan/feedback across NumPy/Numba/CUDA, non-contiguous arrays and
   conversions; Boolean masks, comparison vectors, casts, sorting/repeat/bin-count
   scratch and old/new array coexistence count. Explicitly label scope and any
   interpreter/JIT/context/allocator exclusions; estimates must cover live
   algorithm arrays, not silently widen a configured cap.

4. **Preflight before copies/conversions.** `map.with_backend(...,max_bytes=0)`
   currently materializes about8 MB of host arrays before refusing. Sequence
   map inputs with zero edge/byte budget likewise materialize6.4 MB first in
   np.asarray. Preflight dimensions/types/admission before copies or transfer,
   or require existing typed arrays and reject unsupported sequences WITHOUT
   materializing them. A device-to-device conversion must not stage through
   host unnecessarily. Conversion admission must account for source+candidate
   coexistence, and optional operation max_bytes may only tighten the map cap.

5. **Type information must survive validation.** `[True,2]` currently becomes
   an accepted integer ID vector. Refuse mixed bool inputs before conversion,
   or enforce typed-array inputs consistently (existing new tests may adapt).
   Apply the same rule to offsets/resources and numerical inputs as appropriate.
   Keep unsigned-overflow refusal before lossy cast and metadata-first checking.

6. **Bind CUDA device identity.** RootMap currently records no device and daily
   operations check only CuPy namespace. Record the exact device, validate every
   supplied CuPy array and current device against it before work, and key compiled
   CUDA kernels/availability by device as needed. Reject mismatches before
   allocation/launch; do not switch devices or silently enable peer access.
   Add guard tests that do not allocate on other people's devices. Codex uses
   only an explicitly selected idle GPU. Preserve one-backend daily residency.

7. **Accepted feedback is the native debit.** Current accept() clips rounding-
   level accepted excess to proposed, then closes against that modified answer.
   Preserve SYRUP's actual accepted amounts as the authoritative input when they
   differ only by honest FP64 conversion/summation roundoff; report any excess
   explicitly and reject material excess. Do not close against silently changed
   water. S/unmet boundary handling may tolerate only the documented small bound;
   record roundoff diagnostics. New modules are not yet accepted, so rename their
   new diagnostic keys/tests/docs if necessary. Do not introduce balancing fluxes.

8. **Root-builder memory admission.** NumPy horizontal counting currently calls
   `_overlap_cells` and materializes per-plant edge/bounding-grid arrays before
   its post-count edge admission. Use a bounded count pass or preflight exact
   scratch before those arrays. Production compiled counting already avoids
   per-edge arrays. Count all simultaneously held shared geometry, layer tables,
   override/index scratch, horizontal and expanded maps in each stage; component
   admissions must account for already-live work. Do not silently delete plants
   or alter the uniform-overlap/root-layer law to reduce memory. Internal trusted
   ownership transfer must not leave caller-writable aliases on public map paths.

9. **Canonical merge order.** `from_files` keeps top-level notes when Python `==`
   says they agree. Notes `{'value':1}` and `{'value':1.0}` compare equal, but the
   retained first value serializes differently, so reversing paths changes the
   merged hash (independently reproduced). Compare canonical representations or
   deterministically omit differing metadata; preserve single-file exact hash,
   species contents and duplicate refusal. Add numerical/bool-equality edge cases.

10. **CUDA synchronization.** Daily code currently performs many separate
    bool/float/int scalar reductions. Batch validation/diagnostic reductions into
    small packets where practical, with explicit synchronization boundaries. No
    full P/E/R readback per day. Do not claim no hidden synchronization merely
    because each transfer is scalar. Keep numerical failures observable.

11. **Forcing precision and strictness.** Part A may remain microsecond-exact;
    explicitly document that it does not yet accept MASY's rational submicrosecond
    instants and a future adapter must preserve or refuse them before mutation.
    `parse_date` uses strftime('%Y'), which can reject valid zero-padded years
    below1000; use date.isoformat for the stated Gregorian contract. Fixed UTC
    offsets with fractional seconds must not be truncated silently in metadata:
    preserve exactly or refuse unsupported offset precision explicitly.

## Report and checks

Update report.md with changes, limits and tests NOT RUN (you cannot execute).
Correct the previous claim 'No tool ... failures occurred': revision1 has nine
ordinary Read tool errors, each literally 'File does not exist. Note: your
current working directory is /home/okin/leaf/agent_handoffs/tasks/leaf_phase1_20261110/candidate.'
They were guessed paths within scope, not auth/network/rate limits. Do not erase
them or claim all tool calls succeeded. Codex retains raw records and dispositions.

Add meaningful regression tests for the confirmed failures, not merely assertions
that repeat an estimate formula. Document actual API/admission/roundoff/precision
scope. Do not implement unrelated features or mark the phase complete. Codex will
independently freeze, execute, check GPU/admission/conservation and correct again
if necessary. No new user approval is needed for these bounded corrections.
