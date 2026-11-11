# Native bridge revision 3 — verify physical debit, not only reported totals

Resume exact session b7e68832-ef32-4098-899b-b6bed02172c3, preserve revision2.
Sole writer in this candidate, automatic relevant source sharing and corrections
authorized. Codex controls records/execution/review/publication. READ this prompt
and candidate/** only; Read/Edit/Write only, no execution/Bash/agents/network/
settings/installs/Git/external reads. WRITES exactly leaf/syrup_water.py,
tests/test_syrup_water.py, docs/phase1-native-water.md and report.md.

Exact read-only reviewer files if useful (do not Read their directory):
review_reference/verify_syrup_bridge.py and review_reference/profile_bridge_memory.py.
They are source-only synthetic verifier scripts. All other baseline/context
files remain unchanged. No growth/driver or upstream edits.

## Independently verified revision 2

All35 native bridge tests pass on actual CUDA; CPU has29 pass/6 CUDA skips.
The independent33 CPU native fixtures and failure guards pass. Actual hooked
CUDA allocations now fit the bound in both workloads: P200000/N1/L2 raw peak
19200482 versus declared19200882; P4096/N65536/L3 raw20971872 versus27263640.
Both finish at0 tracked bytes,0 unknown frees. Retain the reduction fix/native
component estimates. CPU tracer still flags roughly25k interpreter/bookkeeping
beyond the array ledger; that probe includes excluded Python metadata, not proof
of an array undercount. Codex is separating that diagnostic; do not add arbitrary
padding or raise a cap.

## Confirmed conservation hole

Codex wrapped the REAL native withdrawal (one call), then replaced ONLY its
returned state with a FRESH copy of the original water array, keeping all actual
withdrawal/budget fields. The bridge returns conservation.closed=True and plant
delivery75m3 although actual input-minus-output storage is0m3. Input arrays are
unchanged. The current checks compare reported budget scalars, not the returned
physical inventory against the actual supplied inventory.

Your test_accepted_shortfall_feedback_with_real_operator_and_drier_state has
the same issue: it swaps a drier input inside the spy, reports closed, and loses
unaccounted source water relative to the bridge's real input. That is not a
conservative model of an acceptance cap. A legitimate shortfall test should call
the real native operator once on the ORIGINAL water with a reduced request,
then adjust only requested/unmet/scale diagnostics to describe the full original
proposal; physical actual water/storage must remain that real call's result.

Add independent, bounded verification that the returned native water actually
reflects the accepted removal from the supplied soil_state. Check per-resource
debit/finite/nonnegative invariants and unchanged inactive cells, with the pinned
native FP64 policy, and ensure reported storage totals agree with the actual
arrays. This is read-only conservation validation, NOT another subtraction,
withdrawal, water store, native physics fork or dummy native call. Compile CPU/
CUDA hot loops, fixed packets only, no P/R readback or unbounded library reducer
scratch. Include its scratch/packets in the public ledger before allocation.
The projection pass can collect actual before-storage totals if helpful; the
post-validation can fuse actual-after/storage/debit checks without full-size
temporary arrays. Reject missing debit, extra/unaccounted loss, layer/cell
misplacement, changed inactive storage, and malformed budget/returned arrays.
Preserve genuine original-input shortfall and honest area-conversion roundoff.

Update/add meaningful regressions: fresh unchanged-water return must reject;
extra loss or wrong-resource debit must reject; original-input real-operator
shortfall passes with exactly one call and removed volume equals delivery.
Correct the old drier-state test instead of accepting hidden water loss. Inputs
remain unchanged on every refusal. Keep existing request/proposal enforcement.

Update docs/report to actual validation/memory scope. Tests NOT RUN by you;
all independently executed checks belong to Codex. Report ordinary tool errors
honestly. No phase completion, implementation installation, commits or pushes.
