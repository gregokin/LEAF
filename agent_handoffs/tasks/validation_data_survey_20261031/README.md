# Validation-data survey

Read the [prioritized inventory](../../../docs/validation-data.md).
This is dataset discovery and metadata/method review, not model validation.

- `scope.md`, `baseline.json`: bounded user request, permitted documentation
  writes, and content identities for 1,530 existing files before this task.
- `catalog.json`: ten parsed EML packages, three NEON product/site availability
  summaries, and additional primary-source leads. Observation tables were not
  retrieved; reported counts are declarations in metadata.
- `public_metadata/`, `selected_records/`, `followup_records/`,
  `confirmed_versions/`: actual public responses, URL/time/hash manifests,
  successful metadata and explicit failures. DataONE mirrors may lag EDI.
  One empty HTTP-200 response is preserved and excluded from usable metadata.
- `fetch_*.py`: bounded public-GET helpers. These read public metadata and
  published links only; they do not upload local project content.
- `access_notes.json`: initial DNS/HTTP failures and interpretation of manifests.
- `verification.md`, `codex_review.md`, `checks.json`: source preservation,
  schema/link checks and research disposition.
- `previous_*.md`: orchestration aliases saved before the survey.

Codex performed the research directly; no Claude implementation or new agent was
launched. No numerical source, model tests/runs, calibration, upstream edits,
environment/package changes, commits or pushes were part of this task. Automatic
collaboration remains enabled for subsequently requested bounded work.
