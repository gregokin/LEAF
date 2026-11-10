# Independent read-only review

Reviewer: Codex agent `/root/jornada_cover`. No edits or workloads performed.

Revision 2: found the allocating supplied-input edge bound before admission,
underestimated profile-row audit scratch, and omitted WUE H2D bytes. Corrections
were sent to the real Claude session as revision 3.

Revision 3 verdict: accept the implementation. The scalar `init_scalar_bound`
precedes the allocating edge bound, scratch includes `S*L+32*S`, and setup H2D
totals include the `8P` WUE vector with timing coverage labeled. Tests cover
refusal before edge/profile materialization and explicitly account controlled
water changes. No remaining material numerical, GPU or memory blocker found.

The reviewer also found a stale API function name in the benchmark guide.
Claude revision 4 corrected documentation only; numerical source stayed frozen.
Final task acceptance remains Codex's responsibility after measured-run checks.

## Full-run spot-check

The same read-only reviewer independently checked five completed weighted
records: CPU hectare/roots3 and GPU hectare/roots3, hectare/roots5, two layers
and 512². All execute 10,950 days, 1,564 updates, 1,565 maps, 783 recharges,
two pending days and 30 passing annual audits. Water/biomass budgets, exact
persistent and H2D formulas, profile identities, source/wrapper hashes and exit
receipts reconcile. The full CPU/GPU reference agrees at 1e-12 relative /
1e-9 absolute in the checked scalars. Configuration and memory documentation
match implementation. Remaining cases were not included in this spot-check;
the orchestrator's final checker must cover every case.
