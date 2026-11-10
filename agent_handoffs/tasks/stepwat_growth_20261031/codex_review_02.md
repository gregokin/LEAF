# Revision2 independent review — correct

No acceptance. The stable snapshot review_snapshot_02 passed seven independent
scalar kernel cases on CPU and real CUDA, and analytical native-water/E/T/area/
year-end/refusal-retry oracles on NumPy and CUDA. Full CPU snapshot suite ran90
checks:82passed,7skipped,1failed. Raw candidate_cpu_02.* records exact evidence.

Required corrections:

1. Enforce the documented host-only temperature_factor interface. Current
validate_temperature_factor calls .get() on a device-like value before checking
shape, allowing a full P-sized transfer on invalid input. Reject device-backed
values without .get(), inspect NumPy shape/sequence length before conversion,
and validate each scalar before materializing malformed nested input. Valid
scalar and [S] host inputs must retain identical numerical behavior. Add tests
for no transfer/no large conversion and no physical mutation on bad forcing.

2. Fix test_weekly_old_new_peak_counts_growth_state, not the working guard.
Simulation stores the bound admission callback during construction. Patching
only a._vegetation_admission afterwards does not intercept it, leaving stages
empty. Also the test expects init_generated from a capture begun after init.
Capture the real callback during construction and weekly work, retaining all
old+new-peak, before-builder, exact-boundary and retry assertions. Do not weaken
checks, raise limits or alter production physics to make the fixture pass.

3. Summary biomass/gain/loss totals are domain sums in kg, whereas the B array
holds kg per individual. Distinguish these units in the reporting fields and
docs so domain totals cannot be read as per-individual values.

Codex has made no implementation edits. Original six-file writer scope persists;
correction source scope is the same user-approved candidate and numerical review.
