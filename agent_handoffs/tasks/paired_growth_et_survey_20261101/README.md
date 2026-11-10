# Paired individual growth and water-use survey

Date: 2026-11-01. Codex research/documentation under [scope](scope.md);
no Claude assignment, additional agent launch or model implementation.

Deliverables:

- [Inactive Jornada uptake candidates](../../../parameters/candidates/jornada-uptake.json)
  and [interpretation](../../../docs/candidate-uptake-parameters.md).
- [Paired-observation survey](../../../docs/paired-growth-water-data.md),
  with [primary references and qualification limits](references.json).
- [Setaria row/ID audit](setaria_audit.json) and [verification](verification.md).
  No growth/water time series was reconstructed or calibrated.

`public_evidence/` preserves four independent retrieval manifests, raw returned
bytes and successful selected public files. Empty/error responses are failures;
no observation content is inferred from them. Source CSVs include 87,368 raw
weighing records and 179 endpoint harvests, with 176 exact barcode matches.
Public download scripts retrieve only the specified public resources; the
Figshare POST is a read-only paper-title search. No local source was uploaded.

The mesquite PDF was retrieved from its author publication record. Local PDF
tools/libraries were absent; literal text was extracted with standard-library
Flate decompression for method inspection. That rough text is not a validated
equation/table transcription. Original PDF bytes remain the source evidence.
The separate DOCX documentation was extracted using ZIP/XML without executing
the R workflow. See [extraction utility](extract_selected_evidence.py).

Baseline: [baseline.json](baseline.json). The `previous_*.md` aliases are exact
historical copies; resolve their links from their original `agent_handoffs/`
base. Root instructions, runtime code, defaults, locks, upstreams and previous
accepted archives are preserved. No tests, model runs, installs, environments,
calibration, investigator contact, Git initialization, commits or pushes.

Acceptance and scope checks: [Codex review](codex_review.md),
[checks](checks.json), and [artifact manifest](accepted_artifacts.json).
