# Codex review — revision 2

Verdict: **correct**, focused revision 3 required. Candidate not installed.

Writer returned exit0, is_error=false and was reaped; 21-file revision_02
snapshot/manifest preserve the tree. Independent geometry/count/cover/exclusion
and separate raster/scratch bounds pass. All 12 independent callback/ID/bridge
guards pass. Representative 512²/.25m setup: 51893 plants, 1.95s CPU,
3217366B plant arrays; whole-process peakRSS89616KiB includes SciPy oracle.
128² mixed no-overlap setup improved from >150 CPU seconds to .184s, retaining
sizes/cover/global exclusion and using a LOWER 50000-proposal bound.

The final suite ran43 tests, with3 failures and8 errors. Most are caused by a
whole-document quote ban that rejects its own required JSON metadata; other
issues are uncaught arithmetic overflow and a mathematically wrong inverse-
overflow test. Saved generator config also cannot replay because of derived
unknown keys. Device B/E shapes must be checked before host .get(), and the
seasonal restore helper must enforce its fresh-driver contract. See prompt_03.

No limits raised, no plants/state discarded, no new scientific laws, upstream
edits, installs/environments, or commits. Original numerical source remains
unchanged. GPU availability inside the sandbox is misleading (insufficient-
driver error); outside access was verified on physical GPU1 for bounded checks.
