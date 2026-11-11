# Phase2 revision4 — explicit checkpoint format version

Resume exact session711c5973-64c3-45f8-af8a-379ba0e357a7. Same user authority,
Read/Edit/Write only, source=this prompt and candidate/**. NO execution/outside
reads/settings/agents/Git/network/install tools. Narrowed writes for this revision:
leaf/checkpoint.py, tests/test_interevent.py, docs/phase2-disturbance.md, report.md.
All other code is now read-only. No ecology or numerical-law change.

R3 reaped/frozen, scope/toolerrors0. It restores signed-int64 IDs per Codex's
corrected instruction. R2 full450 CPU, entry-point guards, compiled example and
actual mixed-live CUDA memory pass; final whole-tree checks remain Codex's work.

The Phase2 restart adds five REQUIRED arrays and a REQUIRED phase2 manifest block
to the accepted Phase1 format, but checkpoint.SCHEMA_VERSION remains1. This is a
breaking persisted layout: old v1 cannot be read by this loader even with source-
change permission because those fields are absent. Give the new layout explicit
schema_version2. Keep canonical plantCSV version1 unchanged. Validate the version
as an actual non-boolean integer, not Python numeric equality that accepts2.0.
On mismatch report actual schema/version and expected2 clearly, BEFORE decoding
arrays or allocating the restored population. Do not infer defaults or silently
migrate old saved biological state; source-change permission does not authorize
a format conversion. Original v1 runs can use their pinned Phase1 runtime; any
future migration needs separate qualification, and CSV alone is not a complete
restart of pending growth/random/stress state.

Add focused save-version2 / round-trip tests, and explicit rejection of v1,
float2.0/bool/unknown versions with otherwise internally hashed manifests. Preserve
strict hashes, source/native/root validation, state and all passing current tests.
Document the new format and migration limit accurately; update report with ALL
checks NOT RUN. No other production changes. Codex will run final full CPU/CUDA,
native biology/restart, signed-ID and memory checks before Phase2 publication.
