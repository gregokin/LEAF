# Verification

Task: `individual_vegetation_feasibility_20261030`.

Assessment verification is documentation/source review, not ecological execution.
Final check output is archived in
[final acceptance](tasks/individual_vegetation_feasibility_20261030/final_acceptance.json).

## Independently checked evidence

- Read actual repository instructions and scoped source; established current
  HEADs and dirty/untracked source hashes. Rehashed all 788 manifested paths and
  compared Git HEAD/status after inspection: unchanged. See
  [before](tasks/individual_vegetation_feasibility_20261030/source_provenance_before.json),
  [after](tasks/individual_vegetation_feasibility_20261030/source_provenance_after.json),
  [preservation](tasks/individual_vegetation_feasibility_20261030/upstream_preservation.json).
- Recomputed all 12 archived STEPWAT2 hashes and 13 selected active ECOTONE
  ecological hashes against prior manifests: match. No wind model evaluated.
  [Output](tasks/individual_vegetation_feasibility_20261030/archived_source_verification.json).
- Read the SYRUP qualification lock and existing environment metadata without
  model/package imports. Lock SHA256:
  `2734e4338245be9e08e36f308eb80a72a498538e6981930220d32cffdbcc6c81`.
  It does not qualify the inspected live MAPLE HEAD.
  [Output](tasks/individual_vegetation_feasibility_20261030/dependency_environment_metadata.json).
- Recomputed memory formula, scenario totals, geometry areas and dense-tensor
  size independently. These are arithmetic checks, not measured GPU memory.
- Parsed separate Claude JSONL streams and launch/exit records: both structured
  successes, no permission denials, prompt hashes match, only allowed web tools,
  exact session/start identities recorded, children reaped and PIDs absent.
  [Output](tasks/individual_vegetation_feasibility_20261030/claude_execution_verification.json).
- Verified root AGENTS.md against both recorded baselines, unchanged guidance
  against the task baseline, source-scope output paths, and local links/whitespace
  in final authored docs. Historical bootstrap evidence was not rewritten.

Primary sources are cited beside claims in the assessment. Original Tietjen
methods, Caylor 2006, STEPWAT2, Hi-sAFe and GRASSMIND methods were checked;
Caylor 2004/2005 comparisons are explicitly limited to publisher abstracts.
No claim of a complete portable reference, runtime ranking or measured GPU
bottleneck is accepted.

## Execution limitations and failures

A sandboxed read-only `nvidia-smi` query returned “GPU access blocked by the
operating system” (`tool_error`, not a usage limit). The same metadata-only
query then succeeded through the approved execution boundary and recorded four
11 GiB GTX 1080 Ti devices. No GPU context/model was allocated.
[Successful query](tasks/individual_vegetation_feasibility_20261030/gpu_metadata.json).

Public page fetches sometimes returned 429/cache errors; primary indexed methods,
author PDFs and authors' source supplied the stated evidence. Unsupported claims
were excluded or labelled. Both Claude calls completed without auth/network or
usage-limit failure; `rate_limit_event` messages reporting allowed status were
not classified as limits.

No scientific model was built or run. No performance, conservation, ecological
calibration or cross-backend restart test was executed; those belong to proposed
implementation phases. No dependency installation or new environment occurred.
