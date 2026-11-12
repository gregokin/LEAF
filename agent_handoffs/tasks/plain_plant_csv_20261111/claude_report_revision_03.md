# Report — task `plain_plant_csv_20261111`, revision 3 (Claude)

**Nothing was executed.** I had Read / Edit / Write only: no test, import, CLI or syntax check was run. Everything below is
from reading the code. The revision-2 independent run (561 tests, 45 CUDA skips, 15 ERRORs all cascading from ONE new
mutation test) is Codex's record; **my corrections are unrun**, so no pass is claimed. Changes are uncommitted in the candidate.

## Corrections made in this revision (numbered as in the task)

1. **Mutation-test fixture bug** — `tests/test_plant_csv.py::TestCompanionVerification::test_edited_companion_fields_are_refused`
   now reads the pristine companion text once (written by `setUp`) and restores it with `write_text` directly in the temp
   directory before EVERY mutation, instead of calling `write_plant_csv(..., overwrite=True)` (whose production guard correctly
   refuses to replace a companion with a corrupted schema). The guard in `leaf/state.py` is unchanged. All previous mutation
   cases are kept; seven new `csv.n_bytes` mutations were added (see 4). The pristine pair is asserted readable at the end.
2. **Actual companion memory accounting** — `PLANT_METADATA_LEDGER_BYTES = 3 * 16384` is gone.
   * `state.plant_metadata_size(state, initial=, provenance=)`: a STREAMING counting pass
     (`json.JSONEncoder(sort_keys=True, indent=2, allow_nan=False).iterencode`, same options as `plant_metadata_text`; the whole
     text is never built, the largest transient is one encoded value). Provenance is counted whole, never truncated. Placeholders:
     fixed-length SHA-256, 20-digit `n_bytes`, longest finite float for `time_days` (so the estimate does not depend on clock or
     CSV size and is >= the real length), plus `METADATA_SIZE_SLACK_BYTES = 512` and the trailing newline.
   * `checkpoint.estimate_companion_bytes(session)` counts the companion of the session's base population with the provenance
     `snapshot()` + `save_session` will attach (base provenance + the small `interevent` record + `checkpoint` tag).
     `estimate_save_bytes` now reports `plant_metadata_text_bytes` and `plant_metadata = COMPANION_COPIES (4) x` that (encoder
     chunks, joined text, UTF-8 bytes, write buffer) and includes it in `total_bytes`. This pre-check runs before the snapshot.
   * `save_session` recounts from the REAL snapshot provenance right after `snapshot()` and re-admits against `max_bytes`
     (`IntereventAdmissionError` stage `checkpoint_save_metadata`) before any CSV/companion text, npz buffer, temp dir or file exists.
   * `max_bytes` / `reserve_bytes` / provenance are never changed or truncated. Load keeps its size admission
     (`4 x` companion size, unchanged). Encoder-object overhead follows the existing documented exclusions (docs say so).
   * Test: `tests/test_interevent.py::RestartTests::test_large_provenance_is_accounted_at_its_real_size_and_round_trips_unchanged`
     (2 MB provenance): estimate grows with the blob; budget one byte short → refused with `snapshot` and `plant_files_text`
     patched to fail (so before the snapshot/serialization) and nothing left; a second-count-inflated run → refused at
     `checkpoint_save_metadata` before serialization; sufficient budget → companion/`read_plant_csv`/restored session carry the
     blob unchanged and `max_bytes` stays `BIG`. Pure test `test_metadata_size_counts_the_actual_companion_json_without_building_it`
     checks the size function is >= the real text and tight to its scalar placeholders, and counts provenance whole.
3. **Preview helper scope** — the `--rerender-from` feature is REMOVED from `render_preview.py` (argument, `preflight_rerender`,
   `rerender`, `RERENDER_*` constants, docstring and `preview/README.md` section). Kept: the companion in `OUTPUTS`
   (preflighted with the other five), provenance SHA-256 of both plant files, the `plant_files` block and the post-write
   read-back via `read_plant_csv` (exit 1 if not exact). A test asserts six outputs and no `rerender` attribute. No rendering,
   initialization or model run was done or is added.
4. **Metadata consistency** — `parse_plant_metadata` now requires `csv.n_bytes` to be a non-boolean non-negative integer;
   `parse_plant_csv` verifies `n_bytes == len(csv bytes)` before the SHA-256 check (both messages contain "match"). Mutation tests:
   wrong, zero, negative, float, bool, string and missing `n_bytes` on disk; an in-memory wrong-size companion with the right hash;
   a checkpoint companion with wrong size (consistent manifest). Tests that deliberately re-bind edited CSV bytes
   (`tests/test_phase0.py::rebound`) now refresh BOTH `sha256` and `n_bytes`, so they reach the intended `n_plants` / SURFACE check.
   The real-file round trip also asserts `n_bytes`.
5. **Version documentation** — `leaf/interevent.py` module docstring (Phase 2 arrays paragraph and the Phase 3 checkpoint
   sentence) and `leaf/__init__.py` docstring now describe layout version 4 / plain `plants.csv` + companion; docstring text only,
   no code touched (I read the rest of `interevent.py` and found no other version/CSV text). README Phase 1–3 "still need an
   independent run" sentences are replaced by pointers to the corresponding task records under `agent_handoffs/tasks/`
   (`codex_review.md`, `verification.md`) and, for Phase 1, a note that this plain-CSV/version-4 change has its own pending
   verification. I do not name individual historical task directories (I cannot see them) and I make no current-pass claim.
6. **Existing preview provenance/docs** — `preview/README.md` now has a section "The existing picture in this directory
   (historical record) versus a new run of the helper": the existing files were produced before the plain-CSV change by an
   earlier operator run (not by the modified helper); `preview_provenance.json` is the preserved original record with the
   pre-split CSV hash and no companion hash; the current layout is a pure metadata split whose pair hashes, original-provenance
   hash and row/image preservation checks live in a separate `plant_files_migration.json` addendum (written by the conversion step,
   not the helper). The "Generate", "Files a NEW run would write" parts are labelled as describing a future newly executed run.
   The case `README.md` paragraph says the same. I wrote/changed no existing preview data and described no rerender.

## Unchanged design from revision 2 (for context)
Plain nine-column CSV; companion `<stem>.metadata.json` (schema `leaf.plant_state_metadata` v1) bound to exact CSV bytes
(`csv.sha256`, now also `csv.n_bytes`); readers load/verify automatically; legacy embedded line still READ only; two-file write =
validate/serialise both, preflight both, companion first then CSV, undo on CSV failure, no atomicity claim for the pair; overwrite
refuses non-LEAF metadata; checkpoint layout version 4 (plain `plants.csv` + `plants.metadata.json` + arrays + manifest, versions
1–3 refused). `AGENTS.md` untouched.

## Suggested exact checks (Codex; none run by me)
1. `python -m unittest tests.test_plant_csv -v` (the mutation test must now reach all its assertions; check no cascade).
2. `python -m unittest tests.test_phase0 -v`; with native SYRUP on `PYTHONPATH`, `python -m unittest tests.test_interevent -v`
   (new large-provenance test is in `RestartTests`, so also runs in `RestartNumba` when numba exists), then the whole CPU suite.
3. Check no `rerender`/`--rerender-from` text remains: `render_preview.py`, `preview/README.md`, docs.
4. Preview split recipe unchanged from revision 2 (parse legacy, assert rows byte-identical to `plant_csv_text(s, initial=True)`,
   `plant_metadata(...)` for the companion): the companion now includes `csv.n_bytes`, which the reader requires.
5. Scope: changed files are the listed ones plus `leaf/interevent.py` and `leaf/__init__.py` (docstrings only).

## Limitations / things to review
* All new/changed code is unrun. Highest risk: the large-provenance test (mock-patched `snapshot`/`plant_files_text`/
  `plant_metadata_size`, relies on `estimate == exact recount`: the worst-case time placeholder is what makes them equal), the
  streaming size function's agreement with `json.dumps` output (same options; check with the new pure test), and the `n_bytes`
  mutation list.
* The counting pass allocates one encoded value at a time (about the size of the largest provenance string) before admission;
  it is far smaller than the 4 accounted copies but not zero.
* README pointers to "task records under `agent_handoffs/tasks/`" are generic; if you want exact directory names, edit them.
* No CLI option exposes `metadata=False`; a CSV without companion still reads as time 0 unless the caller passes `time_days`.
