# Study parameterization — 2026-11-01

Initial inactive STEP-style candidates from Feldman et al. (2018) grass data and
Peterson & Neofotis (2004) mesquite, with **two mesquite seedlings per modeled
plant**. The [assessment](../../../docs/study-stepwat-parameters.md),
[candidate JSON](../../../parameters/candidates/study-stepwat.json) and
[reproduction commands](../../../analysis/study_parameterization/README.md)
describe the results and their scientific limits.

## Scope and provenance

[Scope](scope.md), immutable [baseline](baseline.json),
[environment](environment.json) and [runtime identity](runtime_identity.json)
record the allowed work and actual sources. No usable Git history was available;
the baseline hashes 1,623 regular files and excludes pre-existing symbolic links.
Root instructions, existing runtime/locks and earlier accepted archives are
preserved. No packages, environment, upstream sources or simulation defaults
were changed.

Feldman primary sources are the [paper](https://doi.org/10.1104/pp.18.00146),
[data archive](https://doi.org/10.5281/zenodo.1117062) and
[versioned analysis](https://github.com/maxjfeldman/Feldman_Ellsworth_Setaria_WUE_2017/tree/v1.0).
The [retrieval manifest](public_evidence/retrieval_manifest.json) retains the
successful ZIP download and the initial extraction-guard failure. The
[streaming projection](public_evidence/image_projection.json) preserves all
70,985 image rows while omitting large landmark arrays. Raw weighing/harvest
tables remain in the previous
[public evidence](../paired_growth_et_survey_20261101/public_evidence/selected/).
[Additional retrievals](public_evidence/additional_retrievals.json) pin lens
calibration and genotype metadata. The full ZIP in `/tmp/leaf-study-data/` is
scratch; projected and selected tables suffice to repeat the fit.

Mesquite sources are the [paper](https://doi.org/10.1007/s00442-004-1688-y),
previously archived [public PDF](../paired_growth_et_survey_20261101/public_evidence/followup/mesquite_peterson_2004.pdf),
[current public HTML](public_evidence/peterson_springer.html), losslessly
extracted [figure images](public_evidence/peterson_figures/manifest.json) and
[approximate readings](public_evidence/peterson_figure_readings.json).
Reading resolution is not a statistical confidence interval. Absolute AGB and
a complete water series were not recovered; the shrub values are conditional
priors, not a joint growth fit.

## Codex–Claude implementation and review

Codex supplied a bounded [public contract](public_contract.md).
Claude session `0eff6b97-852d-45dc-ad6c-24503b07a18c` wrote only the four
candidate analysis/test/documentation files, with Read/Edit/Write tools and no
shell. [First prompt](prompt_01.md), [launch](launch_01.json),
[exit](exit_01.json) and separate raw streams preserve the timeout and partial
work. Codex's [initial review](review_initial.md) required corrections for
image-only placeholder weighings, the dispenser fixture, study QC, training-only
priors and actual withheld-harvest evaluation.

The [second prompt](prompt_02.md) resumed the exact session;
[launch](launch_02.json), [exit](exit_02.json),
[structured result](structured_result_02.json) and
[Claude report](candidate/report.md) record successful correction and reaping.
Both [review snapshots](review_snapshot_02.json) are preserved; the original
README was [recovered from its successful Write payload](readme_recovery.json).
Claude did not execute tests or fits. Codex independently executed and reviewed
the final stable snapshot. Installed analysis source is
[byte-identical](installation.json).

## Verification and disposition

- [29 tests passed](tests_02.stdout.txt); [exit and timing](tests_02.exit.json).
- [Full public-data fit](fit_02/candidate_fragments.json) completed in
  [70.36 seconds](fit_02.exit.json); [diagnostics](fit_02/diagnostics.json) and
  [comparison figure](fit_02/study_parameters.png) retain all main outputs.
- [Independent balances](independent_feldman/oracle.json) and
  [18 numerical comparisons](independent_comparison.json) agree.
- [177 independent endpoint replays](independent_replay_checks.json), including
  34 harvests checked against the original raw table, agree to floating-point
  precision. These are short CPU diagnostics with measured water input, not a
  native hydrology, transport or GPU scenario.
- The [actual validator probe](runtime_rate_probe.json) records the expected
  rejection of the seedling rates; no rate bound was altered.

[Verification](verification.md), [Codex disposition](codex_review.md),
[scope/source checks](checks.json) and [accepted artifacts](accepted_artifacts.json)
are the final review record. Acceptance covers the analysis and inactive
candidate evidence, not a complete trait vector or field validation.
