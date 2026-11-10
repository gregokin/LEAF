# Independent verification

Task: `species_root_profiles_20261031`. Final numerical version: `0.2.0`.

## Tests and analytical checks

- CPU: `tests_cpu_03` — 47 tests, OK; three expected unavailable-GPU skips.
- Actual GTX 1080 Ti: `tests_gpu_03` — 47 tests, OK; one unavailable-only skip.
- `oracles_03`: independent CPU/GPU analytical checks pass opposite profiles
  with dry upper/lower layers, finite competition, all-dry cells, an explicit
  third profile, a uniform reference and exact persistent bytes.
- Coverage includes generated cases and CLI JSON input, strict rows/IDs,
  refusals before edge-bound/profile allocation, weekly growth/calendar,
  unchanged horizontal mapping, split-run equivalence, fixed profiles,
  heterogeneous-water CPU/GPU parity and complete setup H2D bytes.

The exact commands, environments, raw stdout/stderr and successful/reaped exit
records are in `tasks/species_root_profiles_20261031/`. The suites use the
existing `/home/okin/MAPLE/.venv-cupy/bin/python`; no installs or new environment.

## Complete duration and resources

Seven GPU runs and one one-thread CPU reference each complete 10,950 days,
1,564 weekly biomass/root-map updates, 1,565 maps, 783 recharge pulses and two
pending uptake days. All 30 annual audits and final budgets pass for every run.
No plants were dropped, state discarded or resource limits raised. All hooks
close with zero live allocations and no unknown frees. The exact persistent
formula and 52P+8SL setup H2D bytes reconcile with recorded arrays/transfers.

The full CPU/GPU reference agrees on counters and water/biomass scalars. The new
uniform-profile control agrees with historical v0.1 counters, final connections
and scalar budgets within the checker's tolerances. Weighted-profile timings
are new measurements; the control is not evidence of identical weighted work.

Unchanged limits: 10 million horizontal edges and 1.5 billion accounted bytes;
the latter also bounds the CuPy pool. Live allocation peak, sampled reservation,
ledger estimates and host RSS are labeled separately. GPU numerical state stays
resident during daily work; weekly checks and reports synchronize.

Final checker command:

```sh
/home/okin/MAPLE/.venv-cupy/bin/python -B \
  agent_handoffs/tasks/species_root_profiles_20261031/verify_performance.py
```

It passed under `result_checks_final`, independently reading all eight raw
result records, source/wrapper hashes and process receipts. Exact results:
`tasks/species_root_profiles_20261031/result_verification.json`. Measurements,
configuration and reproduction commands: [profile report](../docs/species-root-profiles.md).

## Review, provenance and preservation

Real Claude session `a91d9d48-394b-427f-97dd-9c347feb80ba` completed four launches:
32, 26 and eight exact numerical/test/document edits, followed by four
documentation-only edits. Every structured result reports success; processes
were reaped. The no-tools, safe-mode session received generic benchmark text
only, in an empty temporary working directory. No private brief or external
scientific source was submitted. Raw reports and launch/exit identities are
preserved. Codex installed exact edits and independently verified them.

A separate read-only reviewer cleared the corrected implementation and
spot-checked five full records, byte formulas and configuration/memory docs.
Numerical and profiling sources stayed frozen throughout measurements.
`final_sources/` preserves the final generic source, and `baseline_sources/`
preserves the preceding version. Root AGENTS.md, geometry source/tests and
181 historical task/result artifacts match their protected hashes. No upstream
edits, Git initialization, commits, pushes, packages or environments changed.

## Classified earlier failures and corrections

- Revision 1 CPU suite: `check_failed`, two test-only historical-water-budget
  assertions after external water changes. Production accounting was preserved;
  corrected tests explicitly track artificial changes or set the initial budget.
- Reviews 1/2: `check_failed`, generated-field projection, early materialization,
  scratch/transfer accounting and unused class names; corrected through revision 3.
- Revision 3 preparation: `tool_error`, bare `python` unavailable and therefore
  prompt absent. No Claude process started. Corrected by using `python3` and
  creating the prompt before dispatch; exact error lines are recorded locally.
- No auth, network or rate-limit failure occurred in the four actual launches.

No numerical/test/resource issue remains. Final provenance/preservation/link
check passes: protected hashes, 181 historical artifacts, 61 local links, source
snapshots, syntax and all recorded reaped processes reconcile. Its exact result
is in the task's `integrity_checks.json`.
