# Report — task `plain_plant_csv_20261111`, revision 4 (Claude): uniform initial ages (revision 3 report kept below)

**Nothing was executed in revision 4 either** (Read/Edit/Write only): no test, import, generation, rendering or model run. All
code/tests below are unrun; the revision-3 metadata code (state/checkpoint/recording/CLI/preview helper) was NOT modified.

## Revision 4 — what changed
Files written: `leaf/initialization.py`, `leaf/traits.py` (docs only), `tests/test_initial_ages.py` (new), `tests/test_phase0.py`
(one assertion), `examples/phase0/generation.json`, `cases/jornada_grass05_shrub10_256/{generation.json, PARAMETERS.md,
SOURCES.md, README.md, preview/README.md}`, `docs/phase0-vegetation.md`, `README.md`, `report.md`. Trial `species.json`, the preview
files, `preview_provenance.json`, the renderer and all state/checkpoint/recording code are untouched.

* **Rule.** New age distribution `uniform_to_life_expectancy` (`leaf.initialization.AGE_UNIFORM_TO_LIFE`). It is the DEFAULT when
  a species omits `initial_age_days` (that key is now optional; `target_cover`, `diameter_mean_m`, `diameter_std_m` stay required) and
  can be written explicitly as `{"distribution": "uniform_to_life_expectancy"}` (no other keys allowed). Each individual's age is
  an independent draw from a continuous uniform on `[0, L)` days with `L = catalog.life_expectancy_days(sid)` of THAT species
  (mean x 365 for years, existing convention). The bound is read from the catalog only (not duplicated in configs, not from
  allometry/Weibull).
* **Sampling.** Reuses the existing uniform path: `_sample_ages` now reads `spec['age_sampling']` (resolved `{'distribution':
  'uniform', 'min_days': 0.0, 'max_days': L}`) and makes the same single vectorised `rng.uniform(lo, hi, k)` call per species
  batch, so the stream/draw order is the existing uniform one (a test asserts the selector equals an explicit uniform 0..L). Diameter /
  placement / cover acceptance / limits are untouched; no age-biomass relation (a weak-correlation test guards it).
* **Errors.** Missing/null/unusable life expectancy with the default or selector → `CatalogError` naming the species, stating
  whether it was the default, and offering explicit ages; no guessed bound and never zero. Explicit numbers / `constant` /
  bounded `uniform` still work with `life_expectancy: null`. A species not in the config never needs one.
* **Replay/report.** `config_record` keeps the SOURCE rule (selector, written explicitly even if the original omitted the key),
  so JSON round-trip replay with the same catalog is exact and re-resolves the bounds (another catalog → another range, by design).
  The resolved record (rule, rule_source default-vs-explicit, min/max days, life_expectancy mean/unit, days_per_unit, catalog hash,
  catalog origin string, "not a hard death age" note) is in `derived.species.<id>.initial_age`, `species[i].age_requested`, a
  top-level `initial_age_note`, and the state provenance (so it reaches the companion metadata; never the plain CSV).
  `RNG_RECORD` draw-order text updated.
* **Shipped configs.** `examples/phase0/generation.json` and the Jornada `generation.json` use the selector (Jornada: BOER 0..15
  years = 0..5475 d, PROGLA 0..80 years = 0..29 200 d, values read from the unchanged catalog; no new life-expectancy
  estimates). Notes in both files explain this and that it is only the initial range, not a death age.
* **Docs.** `docs/phase0-vegetation.md` §4 ("Initial ages") + `life_expectancy` bullet + example line; README sentence;
  `leaf/traits.py` module/`life_expectancy_days` docstrings (usage described; no mortality change); case `PARAMETERS.md`
  (rows for both species, a row labelling the preview population as the earlier constant-age initialization, §2e wording: it
  was "metadata only / not a cutoff", now also bounds initial sampling, catalog unchanged because its hash is bound to the
  preview; and the explicit initial-sampling-vs-mortality distinction), `SOURCES.md`, case `README.md` and `preview/README.md`
  (existing preview = earlier initialization with constant 730/1825 d, not a realization of the new rule, nothing regenerated;
  a new run would not even reproduce its population because age draws consume random numbers).
* **Tests.** `tests/test_initial_ages.py` (small 32x32 synthetic generations; one 16x16 exclusion-mode case): default bounds
  0..L per species incl. KS test (critical value 2.5/sqrt(n)), min/max spread, mean, non-constant, independence from size;
  default == selector == explicit uniform 0..L, same-seed reproducibility, other seed differs; two species with different units
  (days / years) and a second catalog with other bounds; resolved-record/report/provenance contents, source-rule round trip,
  JSON replay, replay with another catalog, validate/config_record round trip; null life expectancy: default, selector and
  shrub-only selector fail (named species, DEFAULT wording only for the default), explicit ages (numbers, constant, uniform) still
  work, unconfigured species fine; selector parameter/unknown-distribution/None refusals; explicit rules unchanged; exclusion
  mode; plain CSV round trip of ages (rule absent from CSV, present in companion), handwritten ages untouched (even above L);
  shipped example config; Jornada config validated PURELY (no generation) to 0..15 / 0..80 years; CLI generate default rule +
  report, and CLI failure for null life expectancy writing nothing. `tests/test_phase0.py::test_generate_example_config_and_report_collision`
  gained an assertion that the shipped example's ages lie in `[0, 8*365)` / `[0, 40*365)` and vary.
* **Newborn behaviour.** No code touched (`leaf/recruitment.py`, `interevent.py` untouched); existing Phase 3 tests that assert
  age-0 recruits are unchanged. I did not add a native-SYRUP test for it.

## Revision 4 — suggested checks (Codex; unrun by me)
`python -m unittest tests.test_initial_ages -v`; `python -m unittest tests.test_phase0 -v` (generation tests with explicit ages must be
unchanged; the example-config CLI test now uses the default rule); then the full CPU suite. Statistical assertions use fixed seed 11 /
wide margins (KS 2.5/sqrt(n) ~ p 1e-5, min/max margins ~1e-19) so they should not be flaky, but they were not run.

## Revision 4 — limitations / notes
* Any other test or script that omitted `initial_age_days` and expected a validation error would now behave differently; I found none
  in the files I read (not every test module was opened).
* `cases/.../PARAMETERS.md` still lists "restart format 3" in its checkpoint row and `case.toml` (out of scope) may too: stale after
  the layout-4 change, report-only.
* A later hand-edit of `species.json` life expectancies would change both the catalog hash and the age range (documented as by design).
* Ages equal to exactly 0 are possible in principle (uniform on [0, L)); a live plant needs only B > 0, so this is valid.

---

# Report — revision 3 (metadata corrections; unchanged by revision 4)

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
