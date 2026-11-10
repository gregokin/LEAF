# Research verification

Task: [paired_growth_et_survey_20261101](tasks/paired_growth_et_survey_20261101/README.md).
See [full verification](tasks/paired_growth_et_survey_20261101/verification.md),
[checks](tasks/paired_growth_et_survey_20261101/checks.json),
[row/ID audit](tasks/paired_growth_et_survey_20261101/setaria_audit.json) and
[references](tasks/paired_growth_et_survey_20261101/references.json).

- Compare all 1,580 baseline regular-file hashes and the exact bounded write list.
  Existing symbolic aliases were excluded by the baseline and are recorded
  separately; all predate this task. The initial checker-population mismatch
  is preserved and corrected without modifying source links.
- Verify 13 nonempty public-response sizes/hashes and six public Git blob
  identities. Parse raw Setaria tables: 87,368 weighing records, 179 harvests,
  176 exact barcode joins (91 wet / 85 dry). No trajectory reconstruction/QC.
- Check inactive candidate JSON, confidence intervals and depth units, local
  Markdown targets and whitespace. Preserve historical alias bytes/link bases.
- Preserve root AGENTS.md, runtime/defaults, locks and accepted task archives.
  Root instruction SHA-256 remains
  `90a8b6217553480283efd0dcba514306c5472809b35b1d5a0e80e3009e48ef31`.

Dryad HTTP403, the misspelled GitHub link's 404 and empty Figshare HTTP202 are
recorded as failed/unusable retrievals. They do not establish absent data.
The corrected Setaria link and the mesquite PDF were retrieved successfully.
No automatic-approval rejection, source upload, investigator contact, Claude
launch, model test/run, calibration, upstream edit, install, environment, commit
or push occurred.
