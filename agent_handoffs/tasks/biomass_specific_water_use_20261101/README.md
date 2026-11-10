# Additional biomass-specific water-use evidence

User requests additional literature values per kg aboveground **dry** biomass
per day. Codex performed this research-only task; no Claude or other agent was
launched. [Scope](scope.md) and the immutable [baseline](baseline.json) define
authorized writes and preservation requirements.

Deliverable: [comparison and source qualifications](../../../docs/biomass-specific-water-use.md).
No runtime, candidate parameters, locks, upstream sources, accepted archives,
environments or installed packages changed. No model scenario was run.

## Primary numerical sources

- [Belovitch et al. 2023](https://doi.org/10.1111/nph.18909),
  [public author PDF](public_evidence/belovitch.pdf), public data tag `v1.0.0`,
  commit `862abe68c7a9f1eb6397c36516bca35c9d445298`.
  [Source identities](source_identities.json) record the commit metadata and
  successful Git-blob checks for all seven selected files. The raw hourly table
  contains water, fixed harvest shoot dry mass, leaf mass and soil water potential.
  Stored leaf-normalized rates were not substituted for shoot-normalized rates.
- [Murano et al. 2010](https://doi.org/10.1002/etc.288), primary publisher
  Tables 2 and 4 inspected through web access. Direct HTML retrieval returned
  403, preserved in [retrievals](public_evidence/retrievals_03.json).
  [Transcribed numerical observations](public_evidence/murano_table_readings.json)
  retain means, marginal SEs, units, drying method and experimental context.

Additional archived sources include the [bunchgrass paper](public_evidence/he_bunchgrasses.pdf),
[wheat HTML](public_evidence/el_habti.html) and its figures, and
[poplar HTML](public_evidence/renninger_poplar.html). They are qualified leads;
they do not supply silently interchangeable primary-table coefficients.

## Calculations

Run from the repository root, with a new output location after acceptance:

```sh
python3 -B agent_handoffs/tasks/biomass_specific_water_use_20261101/derive_daily_rates.py
```

The archived script writes `derived/` and refuses an existing directory. To
repeat after this task is frozen, copy the task's script and public evidence to
a separate directory before execution; do not write into the accepted archive.
No dependencies beyond the Python standard library are needed for derivation.

Results:

- [All daily records and exclusions](derived/belovitch_all_daily_qc.csv).
- [Individual/bin means](derived/belovitch_plant_bin_means.csv).
- [Group summaries, units and QC](derived/belovitch_summary.json).
- [Fourteen table-derived rates](derived/murano_rates.json).
- [Independent numerical checks](independent_checks.json) use the existing
  NumPy/pandas environment; no package installation was needed.

Savanna calculations require 24 unique hourly observations, include nighttime
and negative increments, exclude the two ambiguous plant IDs and use daily mean
soil water potential for moisture bins. Fixed harvest dry mass is explicit.
The reported medians/IQRs are descriptive reanalyses, not the paper's mixed
model or a new field calibration. Table ratios are ratios of means; marginal
SEs do not identify ratio confidence intervals.

## Disposition

[Verification](verification.md), [review](codex_review.md),
[preservation/link checks](checks.json) and [accepted artifacts](accepted_artifacts.json)
record final evidence. Public-download network failures, 403 responses and
unusable literature leads remain documented. Automatic collaboration stays
enabled; this completed research task creates no new implementation assignment.
