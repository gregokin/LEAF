# Research verification

Task: paired_growth_et_survey_20261101, Codex research/documentation only.
Evidence: [checks.json](checks.json), [Setaria audit](setaria_audit.json),
[primary references](references.json), and the four public retrieval manifests.
No model tests or runs were needed or performed for these changes.

## Public observations and provenance

`python3 agent_handoffs/tasks/paired_growth_et_survey_20261101/audit_public_tables.py`
independently reads raw CSVs, verifies six public Git blob identities against
the v1.0 listing and checks exact plant-barcode joins. Results: 87,368 raw weight
records, 179 unique endpoint harvests, **176 exact joins (91 wet / 85 dry)**.
All harvested dry weights parse as finite nonnegative numbers; this is a schema
check, not scientific QC. Three unmatched harvest barcodes remain unchanged.
No daily water-loss or biomass trajectory was calculated.

`python3 agent_handoffs/tasks/paired_growth_et_survey_20261101/extract_selected_evidence.py`
reproduces the inspected text extracts without modifying existing matching files.
The original mesquite PDF remains authoritative; extraction is not qualified
for equations/tables. Methods establish two seedlings/pot, initial removed-sample
mass, a 39-day interval and two-day water recording with evaporation controls.
Setaria's separate 216 MB archive was identified from public metadata only.

Primary full text supports Ravi–Breshears measurement limits and the explicit
Figshare link. Its contents remain unverified. Dryad landing-page schemas support
the sapling/milkweed leads, but no XLSX data rows were downloaded.

## Candidate parameters and preservation

`python3 agent_handoffs/tasks/paired_growth_et_survey_20261101/verify_deliverables.py`
checks the 1,580 baseline paths/hashes, the bounded write list, JSON parsing,
source-response sizes/hashes, local Markdown links, trailing whitespace and
the inactive candidate's estimates/CI units. Historical `previous_*.md` links
resolve from their original `agent_handoffs/` directory; their bytes are preserved.

The new numerical values are potential uptake parameters only. Physical layer
boundaries, profile-tail handling and biomass-to-canopy-volume conversion remain
unselected. No active loader/default, numerical source, dependency lock, upstream
file or previous accepted task artifact was modified. Root AGENTS.md retains
`90a8b6217553480283efd0dcba514306c5472809b35b1d5a0e80e3009e48ef31`.

## Retrieval/tool limitations

- `network`: four Dryad file requests returned HTTP 403. The paper's misspelled
  Setaria GitHub link returned 404; the correct archive metadata supplied the
  working repository name, and successful files are pinned and hashed.
- `other`: Figshare share-page attempts produced no usable body. The final
  diagnostic established **HTTP 202, zero bytes**. The public-title search
  succeeded as an HTTP request but returned unrelated records, so it does not
  qualify any matching dataset. Empty responses are excluded from evidence.
- `tool_error`: browser PDF/share-page opening failed; local `pdftotext` was
  unavailable (exit 127). Existing system/MAPLE environment checks found no PDF
  library; none was installed. An initial literal-extraction attempt missed
  CR-only PDF line endings; the corrected reproducible extraction succeeded.
  One closing patch was rejected before writes because it combined delete/add
  operations on the same path; it was reissued with supported operations.
- `check_failed`: the first scope checker enumerated 142 old symbolic source
  links that the baseline intentionally excluded, falsely reporting new files.
  [The failure](checks.initial.json) is preserved. The corrected checker uses
  the baseline's regular-file population, records link targets separately and
  verifies their link modification times predate the task baseline. No source
  or symbolic link was changed to satisfy this check.

These are not automatic-approval rejections or proof of absent observations.
No investigators were contacted, and no local source or project brief was
uploaded. No implementation agent, simulation, calibration, install, new
environment, upstream edit, Git initialization, commit or push was performed.
