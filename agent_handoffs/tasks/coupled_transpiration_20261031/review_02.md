# Revision 2 — correct

CPU execution: 66 passed, five expected backend skips, two new analytical
fixture failures. Actual diffusion at 1e-15 over 86400 s explains those failures;
set D=0 in isolated uptake tests, not tolerance or production equation changes.

Independent snapshot reproductions confirm inactive-budget mismatch, allocation
before native memory refusal, negative-reserve admission bypass, vegetation
construction before native zero-memory refusal, and float32 gather acceptance.
Read-only review confirms repeated host packets and missing parameter bytes.
GPU test helpers also use the wrong mask namespace and can silently skip the
construction bug. Native per-cell withdrawal law and basic volume/time flow
are otherwise correct. No source has been installed.

Verdict: correct per prompt_03.md. Classification: check_failed. Preserve raw
stdout/stderr, immutable native review snapshot and source manifests.
