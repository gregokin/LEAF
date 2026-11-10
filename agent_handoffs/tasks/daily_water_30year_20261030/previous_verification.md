# Verification

Task: `root_radius_3_5_20261030`.

[Final acceptance](tasks/root_radius_3_5_20261030/final_acceptance.json) checks the
new capacity arithmetic, baseline invariants and recorded CPU/GPU measurements.

- [Updated capacity](tasks/root_radius_3_5_20261030/capacity_estimates.json):
  216 cases with root multipliers 3/4/5, fixed population assumptions and two/three layers.
- [Execution](tasks/root_radius_3_5_20261030/benchmark.exit.json): four cases passed,
  exit 0, no timeout, child reaped. [Raw results](../benchmarks/results/root_radius_3_5_20261030/profiled_results.json)
  preserve repeats, full-population demand audits, field parity and ledgers.
- Source SHA matches the previous 20-test qualification and prior 5× benchmarks;
  unchanged code/tests were not unnecessarily rerun. Existing caps are unchanged.
- Allocation hooks finish with zero live bytes and zero unknown frees.
- Previous GPU was occupied; selected another idle GTX 1080 Ti by UUID. Existing
  jobs were not altered. Cross-run timings are not controlled speedup ratios.
- Root AGENTS and closed task evidence preserved. No new Claude session or
  implementation edits; no physical model, upstream write, install or Git action.
