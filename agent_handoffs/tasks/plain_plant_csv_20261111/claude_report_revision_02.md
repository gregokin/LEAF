# Report — task `plain_plant_csv_20261111`, revision 2 (Claude)

**Nothing was executed.** I had Read / Edit / Write only: no test, import, CLI or syntax check was run, so every statement
below about behaviour is from reading the code. All new/changed tests are UNRUN. Please run the suggested checks in
"Suggested exact checks". Changes are left uncommitted in the candidate tree; no Git, network, package or environment action.

## What changed

### Library
* `leaf/state.py` — the CSV serializer is now plain-only and a companion layer was added.
  * `plant_csv_text(state, *, initial=False)` returns PLAIN CSV (header + nine-field rows). The `metadata=` and
    `provenance=` keywords were **removed**, so no keyword, flag or code path writes embedded metadata (a test asserts the
    signatures and that `metadata=True` raises `TypeError`).
  * New companion API: `metadata_path`, `plant_metadata`, `plant_metadata_text`, `plant_files_text`, `parse_plant_metadata`,
    `read_plant_companion`, `preflight_plant_files`, `publish_plant_files`, `write_plant_files`; constants
    `METADATA_SCHEMA='leaf.plant_state_metadata'`, `METADATA_SCHEMA_VERSION=1`, `METADATA_SUFFIX='.metadata.json'`.
    `csv_metadata` was removed. `write_plant_csv(...)` keeps its signature and return value (CSV path) and now writes both files.
  * `parse_plant_csv(..., metadata=None)` / `read_plant_csv(..., metadata=None)`: file readers auto-load
    `<stem>.metadata.json` and verify it (details below). `metadata=False` is an explicit opt-out (plain list + caller context).
  * Legacy embedded `# {json}` first line is still READ (old checks unchanged); refused together with a companion; a `#`
    first line that is not that JSON is refused (plain files carry no comments).
* `leaf/checkpoint.py` — checkpoint dir = `plants.csv` (plain) + `plants.metadata.json` + `restart_arrays.npz` + `restart.json`.
  `SCHEMA_VERSION` 3 → **4**; `files` table of the manifest lists the companion hash; load verifies it, parses plants with the
  verified companion (`initial` must be false, time must equal the manifest time, row count, catalog, domain). Save ledger got a
  small `plant_metadata` entry; load admission formula includes the companion size. A version-3-looking directory (three files)
  gets an explicit "older layout, not migrated" message.
* `leaf/recording.py` — `SnapshotWriter` writes the pair through `write_plant_files`; each record gains `metadata_path`;
  bookkeeping is still committed only after the write succeeded.
* `leaf/__main__.py` — `init` / `generate` preflight both plant destinations; `generate` also refuses `--report` equal to
  the companion path of `--out`, serialises CSV + companion + report before any write, keeps the report-rollback behaviour and
  additionally the pair's own rollback; JSON output gets `metadata_written`; help/docstring updated. Readers (`inspect`,
  `export-canopy`, `init --plants`) pick up the companion automatically through `read_plant_csv`.
* `cases/.../preview/render_preview.py` — `initial_plants.metadata.json` added to the declared outputs (preflighted with the
  others), provenance lists the SHA-256 of both files plus a `plant_files` block, the written pair is read back via
  `read_plant_csv` and compared with the generated plants (exit 1 if not exact). New `--rerender-from DIR` mode (reads an existing
  preview dir through the companion, draws a new PNG + `rerender_provenance.json` into a NEW `--output`, never overwrites);
  this is new, unrun code — it is my reading of "its rerender path must read correctly".

### Tests
* `tests/test_plant_csv.py` (new, no native deps): plain-CSV strictness (nine fields, no `#`/JSON), signature checks, naming
  convention, real-file round trip with companion/hash, edited / reformatted / stale / mismatched CSV, ~20 edited-companion
  mutations (each must raise `StateError`), invalid JSON / duplicate key / NaN, caller-context conflicts (catalog, domain, time),
  missing companion = plain list, orphan companion, plain initial list with caller context, SURFACE guards (refused before any
  surface is sampled), legacy reads (checks kept, ambiguity refused), no-clobber of CSV and companion, unrelated metadata never
  overwritten, injected failure of each of the two publications (clean/restored), simulated between-publication interruption
  detected, hard-link failure, snapshot writer (+ failed pair leaves bookkeeping), CLI generate/inspect/init chain incl. collisions
  and injected failure rollback, preview helper declares the companion output.
* `tests/test_phase0.py` — adapted honestly (not weakened): tests that read the embedded line now use the companion dict /
  file; added hash-binding, `n_plants`, time/domain/catalog guards; recording and CLI tests assert plain CSV + companion and the
  companion collision; the generate test covers the companion collision and unrelated-companion refusal.
* `tests/test_interevent.py` — checkpoint tests: four-file layout, plain CSV + companion checks, edited/dropped/inconsistent
  companion, version 4 strict (1, 2, 3, 5, 4.0, True, '4', None refused), v3-style directory refused, old versions (1, 2, 3)
  refused, snapshot writer companion.

### Docs
README, `docs/phase0-vegetation.md` (§2.3 rewritten, recording, CLI), `phase1-interevent.md` (restart layout, version 4
compatibility), `phase2-disturbance.md`, `phase3-reproduction.md`, `shared-canopy.md`, `maple-leaf-integration.md`, the case
README and `preview/README.md`. The no-embedded-metadata convention is stated in README and `docs/phase0-vegetation.md` §2.3
(and in `leaf/state.py`'s docstring). Root `AGENTS.md` was not touched (not in scope).

## Design and compatibility choices
* **Companion**: `<stem>.metadata.json` (stem = name without last extension). Content: time, n_plants, catalog hash + species
  table, domain, units, coordinate frame, `initial` / `establishment_resolved`, provenance, and `csv.{sha256, n_bytes, columns, file}`.
  `csv.sha256` binds the exact file bytes (a CRLF conversion or an extra newline is a mismatch). `csv.file` is informational so a
  pair can be moved together. No self-hash on the companion: it stays hand-editable (time, provenance), and the CSV hash plus the
  context checks catch inconsistent edits.
* **Reads** refuse: hash mismatch, wrong schema/version type, missing required key, wrong catalog/table/units/frame, a caller
  domain/time that disagrees, `n_plants` mismatch, `establishment_resolved` contradicting the CSV, SURFACE tokens unless the
  companion declares `initial: true` (checked before any surface is sampled). No companion → handwritten plain list (domain from
  the caller, time default 0, SURFACE allowed) — same as the old no-metadata-line behaviour.
* **Writes**: validate + serialise both, preflight both, publish companion first and CSV second, undo on CSV failure. Interruption
  between the two is *detected* (orphan companion → missing CSV; overwrite → new companion + old CSV → SHA mismatch). I make no
  atomicity claim for the pair. Overwrite refuses a metadata file that is not a LEAF companion (never clobbers unrelated metadata).
  An orphan LEAF companion blocks a non-overwrite write (the interrupted-run remnant must be removed or overwritten explicitly).
* **Checkpoint**: companion lives inside the checkpoint directory (published by one directory rename, so the pair there cannot be
  half-visible) and its hash is in `restart.json`, so plain `read_plant_csv(<dir>/plants.csv, catalog)` stays convenient.
  **Version bumped to 4**, versions 1–3 refused with no migration or reinterpretation; Phase 1–3 manifest blocks, arrays,
  hashes, memory limits and transactional publication are unchanged. Old checkpoints must be loaded with the writer that made them.

## Suggested exact checks (for Codex; none run by me)
1. Syntax/import: `python -c "import leaf.state, leaf.checkpoint, leaf.recording, leaf.__main__"` and
   `python -m unittest tests.test_plant_csv -v`.
2. `python -m unittest tests.test_phase0 -v` and, with native SYRUP on `PYTHONPATH`, `python -m unittest tests.test_interevent -v`
   (RestartTests and the Phase 2/3 restart classes exercise version 4); the whole suite via
   `python -m unittest discover -s tests -v` with the usual env. Other test modules I did not open (e.g. disturbance, demography,
   recruitment, roots, uptake) were assumed not to use the plant CSV API; if one does, it needs the same adaptation.
3. Preview conversion (no regeneration) — suggested recipe, then verify independently:
   ```python
   s = state.parse_plant_csv(old_bytes, cat)                       # reads the legacy embedded first line
   head, body = old_bytes.split(b'\n', 1)
   assert body.decode() == state.plant_csv_text(s, initial=True)   # rows byte-identical to the current table
   meta = state.plant_metadata(s, body, initial=True, provenance=s.provenance, csv_file='initial_plants.csv')
   # write `body` unchanged as initial_plants.csv and state.plant_metadata_text(meta) as initial_plants.metadata.json
   ```
   Then check: row count / row bytes equal to the pre-split rows (`old_bytes` minus its first line), every row has nine fields,
   `read_plant_csv` returns arrays equal to the pre-split parse, companion `csv.sha256` equals the file hash, PNG hash unchanged.
   Note `preview/preview_provenance.json` (historical, immutable) records the SHA-256 of the pre-split CSV and mentions no
   companion; the new layout's hashes belong in whatever provenance addendum you choose — I did not touch that file.
4. Scope: confirm only the listed files changed and `AGENTS.md` is unchanged.

## Limitations / not done / things to review
* Unrun code and tests (see top). Highest-risk items to look at first: `tests/test_plant_csv.py` (large, newly written, expects
  specific error texts such as 'match'/'exists'/'differs'), the `render_preview.py --rerender-from` path, and the
  `test_interevent.py` restart tests that rewrite companion + manifest hashes.
* **Stale text outside my write scope** (report-only, needs a Codex-owned edit): `leaf/interevent.py` module docstring
  ("complete (version 3) checkpoints", "canonical `plants.csv`") and `leaf/__init__.py` ("complete checkpoints are version 3")
  still say version 3. Docs/examples outside the allowed list (e.g. other files under `docs/`, `examples/phase*/README`) were not read
  or edited. `examples/phase1/run_interevent.py` still works unchanged (it now also writes `initial.metadata.json` /
  `final.metadata.json`).
* No CLI option exposes the library's `metadata=False` opt-out; to edit a plant CSV by hand, delete its companion (documented).
* `init --out` equal to its own `--plants` with `--overwrite` is not specially guarded (as before).
* Reading a plain CSV whose companion was deleted silently uses time 0 unless the caller passes `time_days` (inherent to a plain
  file; a snapshot's companion is the only carrier of time).
* The checkpoint ledger constant for the companion (`PLANT_METADATA_LEDGER_BYTES = 49152`) is a conservative guess, not measured.
* Not done by design: no commit/push, no preview regeneration, no touching of `preview_provenance.json`, baseline, prompt,
  `AGENTS.md` or any control file.
