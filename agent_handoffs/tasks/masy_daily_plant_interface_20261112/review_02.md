# Independent review of revision 2 — CORRECT

Frozen candidate: candidate_02_manifest.json, public_evidence/review_02. No code installed.
Claude exit0, structured success, reaped. One rejected Edit request used an out-of-scope path with identical old/new strings; tool_error, no changes or file created. See scope_audit_02.json. All actual candidate writes are in scope.

## Passing evidence

Real N15 CPU and actual CUDA (GPU1) each pass all 12 lifecycle_bridge tests: equal-digest alternate branch refusal, mixed native candidate refusal, source binding, shared PET, one hydrology + one zero-time T, closure and explicitly supplied date/run/ordinal mismatch guards. Qualified C1 growth/PET/calendar code remains byte-identical. Wind process_kind is correctly refused by forcing.check_process_kind; the independent probe confirms this (not a defect).

## Remaining defects, reproduced

1. DayBinding remains optional at allocation. probe_revision02_contract.py: real N15 daily occurrence0 is accepted for nominal year2037 without a mapping. Required day/run binding must be mandatory for coupling. Transport-only construction can remain unbound, but a water-connected proposal must require an explicit mapping and a completed daily lineage.
2. Missing-feedback continuation/save. probe_revision02_unpaired.py: consume T1, add a real extra T2 without biological feedback, then both the next fragment proposal AND save_biology(external=T2) succeed. Physical n_requests=2 but LEAF consumed1. Refuse unhandled T advancement (request_sequence AND n_requests) unless it is precisely a supplied pending-after-debit stage. Later physical hydro/storm/close references with the SAME T counts remain legitimate. On first connection bind the initial physical cursor explicitly so an arbitrary unseen T history is not silently adopted.
3. resource_identity currently records passed lop.sha256 rather than validating parent.lop_sha256. Require actual operator association; retain genuine native instance capabilities.
4. Memory preflight is incomplete. start allocates base.status==LIVE before admission and its lower bound omits root resource scratch. Qualified C1 rebuild_geometry admits root map with a remaining budget but calls the canopy builder with its independent canopy max_bytes and only original reserve, omitting live root map/candidate state/canopy-area coexistence. This can bypass the global cap. A narrow accounting extension to adopted biocalendar is authorized, with explicit requalification; do not alter scientific kernels. Measure actual CPU/CUDA allocation peaks and tight-cap refusal, do not label estimates measurements.

## Test failures (5 failed /64 passed /4 native skipped /29 subtests)

- transpiration_without_growth test publishes a successor then stages a flush on its stale ancestor. Correct production refusal; branch independently for the comparison.
- tight-cap test sees rebuild_geometry called before the missing metadata lower bound rejects; improve preflight and test actual allocation entry points rather than asserting arbitrary private call ordering.
- three reproduction tests use many ecological patches whose requests exceed max_new_plants=5; production correctly refuses 8–20 births. Keep the cap. Use a deliberately small one-patch fixture so existing requests fit; preserve a separate refusal test. Do NOT increase resource limits, drop births, edit shipped catalog/case or recalibrate laws to make tests pass.

All reproduction and trial scientific work remains paused. Fix only this interface, integration evidence and fixtures. No ecological validation or full MASY acceptance claimed.
