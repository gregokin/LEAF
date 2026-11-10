# Independent verification — study_parameterization_20261101

Reviewer/executor: Codex. Claude authored the isolated pipeline and tests but
executed neither. Final source: `review_snapshot_02/`, installed byte-identically
under `analysis/study_parameterization/` (see [installation](installation.json)).

## Executed checks

The existing `/home/okin/MAPLE/.venv-cupy/bin/python` environment was used with
CPU-only execution, one BLAS/OpenMP thread and no bytecode/cache writes. Exact
commands, PID/start identities, timestamps, separate streams, exits and reaping
are in the numbered launch/exit records. No environment or package was added.

1. `python run_checks.py tests_02`: [29 passed](tests_02.stdout.txt), exit 0,
   [3.16 s including startup](tests_02.exit.json). Tests cover physical weighing
   chains, negative/placeholder handling, QC, units, genotype holdout isolation,
   STEP updates and two-seedling pot aggregation.
2. `python run_checks.py fit_02`: [exit 0 in 70.36 s](fit_02.exit.json).
   [Outputs](fit_02/candidate_fragments.json) include TRAIN-only estimates,
   profiles, capacity/ET sensitivities and withheld harvest evaluation.
3. Independent raw-weight reconstruction in
   [independent_feldman_balance.py](independent_feldman_balance.py) agrees on
   13,970 training periods, 959 plants, 0.873344 m³ loss and the burden integral.
   It shares only the frozen, separately reviewed image calibration, not the
   candidate's water-accounting functions. [18 comparisons](independent_comparison.json)
   pass, including independent mesquite unit/rate calculations.
4. [independent_replay_check.py](independent_replay_check.py) imports no candidate
   functions: all 177 withheld endpoints agree within 1.74e-18 kg. It checks
   34 dry harvests directly against the original raw table: RMSE 1.5247 g and
   R² 0.70297. [Results](independent_replay_checks.json).
5. [Runtime probe](runtime_rate_probe.json): `r_year=1` passes schema validation;
   the mesquite diagnostic lower bound is rejected by the existing rate cap.
   This expected incompatibility is reported, not bypassed.
6. [Final checker](verify_deliverables.py): baseline preservation, allowed paths,
   source/data hashes, inactive JSON, candidate units/numerical identities,
   tool/source scope, process completion, Markdown links and whitespace.
   [Results](checks.json). The installed CLI also passed a `--help` smoke check;
   this is not another parameter fit.

The primary daily-loss QC uses the current weighing/end-day DAP, verified against
the archived author's water-use script. The final pipeline excludes 19,889
image-only `-1` placeholders before constructing physical boundaries, flags 342
negative intervals without clipping, and obtains zero closure residual for all
29,435 complete chains. Fourteen daily balances fail the primary study QC.

## Errors retained and resolved

- `tool_error`: initial image extraction refused the 590 MB member with literal
  `Expected one bounded image-feature CSV`. The 216 MB ZIP was successfully
  retrieved and checked; a streaming scalar projection recovered every image
  row without extracting the full member.
- `interrupted`: first Claude launch exceeded 900 s, exit 143; child reaped and
  partial work preserved. The exact session resumed successfully, exit 0;
  structured result is success with no tool errors. Both sessions retain raw
  stdout and stderr.
- `check_failed`: an incorrect unittest discovery collected zero pytest tests,
  exit 5; this was never accepted as a pass. First proper pytest run had 22 pass,
  1 fail from a fixture exactly at the strict 5 g threshold. Corrected fixture
  and a boundary regression pass in the final 29-test suite.
- `check_failed`: Codex's sentinel counterexample exposed an invalid -106 g daily
  balance instead of 5 g. Revision 2 removes only image placeholders from the
  weighing chain, preserves physical days and adds the regression.
- `tool_error`: one atomic patch missed its expected context; no writes occurred,
  and the corrected patch succeeded. A convenience read under system Python
  lacked pandas; rerunning with the existing project interpreter succeeded.
  A later `python` command was absent (`/bin/bash: line 1: python: command not
  found`, exit 127); its read-only metadata inspection succeeded with `python3`.
- `tool_error`: a closing patch attempted delete/add on the same mutable report
  path; the patch tool rejected the entire operation before writes. Reissued
  allowed updates succeeded. The checker's initial dry run used the wrong
  metadata allowlist and ran before linked final evidence existed. Comparing
  the immutable dispatch prompt corrected the checker to its exact four public
  metadata paths; no Claude scope violation or missing final artifact remained.

No rate-limit/auth error or automatic approval rejection remains. Automatic
collaboration stays enabled. No native hydrology/transport/GPU scenario, install,
upstream edit, commit or push was performed in this task.

## Scientific limits

Acceptance does not make the candidates runtime-ready. Grass pot loss is ET,
not independently separated T. Earlier biomass uses a training-only image
calibration; endpoint timing uses last-image day as a sampling proxy. Mesquite
support additionally assumes incremental allocation approximates standing AGB
allocation; its two-day water is the final interval, not cumulative 39-day use.
Seedling rates exceed the current adapter cap. Adult capacity/allometry,
phenology, persistent/seasonal partition and independent demand remain
unidentified. Seven-day diagnostic blocks plus a final partial update differ
from the complete native calendar. No forced activation or field-validity claim.
