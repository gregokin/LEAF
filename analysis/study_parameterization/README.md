# Study parameter analysis

This isolated research pipeline derives inactive candidates from Feldman et al.
(2018) grass observations and Peterson & Neofotis (2004) mesquite figures.
See [the assessment](../../docs/study-stepwat-parameters.md) and
[parameter record](../../parameters/candidates/study-stepwat.json).
It imports no LEAF/SYRUP runtime source and changes no simulation defaults.

From the repository root, use the existing environment and a new output directory:

```sh
LEAF_STUDY_TASK=agent_handoffs/tasks/study_parameterization_20261101
LEAF_STUDY_RAW=agent_handoffs/tasks/paired_growth_et_survey_20261101/public_evidence/selected
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/leaf-study-data/mpl \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/okin/MAPLE/.venv-cupy/bin/python analysis/study_parameterization/study_parameters.py \
  --images "$LEAF_STUDY_TASK/public_evidence/setaria_image_scalars.csv" \
  --weights "$LEAF_STUDY_RAW/setaria_snapshots.csv" \
  --harvest "$LEAF_STUDY_RAW/setaria_harvest.csv" \
  --zoom "$LEAF_STUDY_TASK/public_evidence/zoom_calibration_data.txt" \
  --peterson "$LEAF_STUDY_TASK/public_evidence/peterson_figure_readings.json" \
  --out /tmp/leaf-study-parameter-repeat --n-boot 100 --qc-day-label end
```

Existing output directories are refused. Selected public input tables and
provenance are archived in the task directories; the full scratch ZIP is not
needed to repeat the fit. Do not write repeat results into an accepted archive.
No package installation or new environment is required. Exact versions are in
[environment.json](../../agent_handoffs/tasks/study_parameterization_20261101/environment.json).

Run the synthetic accounting, holdout, unit-conversion and aggregation tests:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/okin/MAPLE/.venv-cupy/bin/python -m pytest \
  analysis/study_parameterization/test_study_parameters.py -q -p no:cacheprovider
```

The pipeline excludes image-only `-1` placeholders from physical weighing chains,
retains other QC flags, and uses actual irrigation mass plus storage change to
reconstruct pot loss. Published daily-QC rules are primary; flagged-data and
T/ET-partition sensitivities remain explicit. Dry-mass calibration, growth fitting,
capacity selection and bootstrap estimates use training genotypes only.

Outputs include source/QC counts, period/plant tables, candidate fragments,
rate/support profiles, matched withheld-harvest predictions and a PNG comparison.
The normal-only diagnostic applies seven-day updates and a final partial
observation interval; it is not a complete native calendar replay. A successful
fit does not activate parameters or establish field validity. Adult allometry,
annual phenology and the seasonal/persistent split remain unidentified.
