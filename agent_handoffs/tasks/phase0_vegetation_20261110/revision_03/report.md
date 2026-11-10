# Phase 0 vegetation infrastructure — implementation report, revision 3 (focused correction)

Writer: Claude (bounded LEAF Phase 0 implementer). All work is uncommitted in `candidate/`; only the 17
allowed candidate paths were written; the task-local review artifacts were read, not edited. No shell,
Python execution, installs, network, Git or execution of any check was performed by me. **Everything listed
under "Tests NOT RUN" is for Codex to execute.** No scientific scope was widened; legacy numerical modules
and the build backend are unchanged. The corrected revision-2 algorithms (large-canopy-first exclusion,
separate raster/scratch admission, default-safe canopy publication, ownership/ID/bridge checks, import
isolation) are preserved; no bound was raised.

## Changes in revision 3

1. **CSV quoting regression (the 8 errors + 2 CLI failures).** The whole-document double-quote bans in
   `plant_csv_text` and `parse_plant_csv` are removed (the JSON metadata line necessarily contains quotes).
   The metadata line is parsed separately as strict JSON; rows use `csv.reader(strict=True)`, so ordinary
   quoted numeric/status fields are accepted and an unterminated or stray quote is refused (`csv.Error` →
   `StateError`). Schema, FP64 round trip, clock/catalog/units/domain metadata and the initial-only SURFACE
   checks are unchanged. Tests now cover a successful metadata-bearing round trip (already present), quoted
   status and quoted numeric fields that round-trip exactly, a rejected unterminated quoted last field and a
   rejected stray quote. Docs updated to the standard strict-CSV policy.
2. **Numerical error conversion.** `expected_crown_area_m2` and `lognormal_parameters` convert Python float
   `OverflowError` (and related arithmetic errors) into `CatalogError` before any allocation; finite results
   remain valid (tested with `1e100`). The inverse-overflow test now uses `b_d = 0.5` (inverse exponent 2,
   really overflowing); the `b_d = 2` forward-overflow check is kept and its finite inverse result is
   asserted as valid (allometry unchanged).
3. **Replayable generation config.** `report['generation_config']` is now the accepted SOURCE configuration
   (schema/domain/seed/overlap/tolerance/subsamples/iterations/limits/time/species input rules); derived
   ordering, lognormal μ/σ, expected crown areas and initial counts moved to `report['derived']` (also in
   `state.provenance`). Unknown-key validation is unchanged. Tests regenerate array-for-array from a
   JSON-serialised/deserialised saved config, with and without a surface (overlap and exclusion modes).
4. **Device vector preflight.** `runtime._float_array` checks shape/ndim/dtype metadata of device-backed
   arrays (anything with `.get()` that is not a NumPy array) against the known P **before** calling `.get()`,
   refuses missing/invalid metadata, performs exactly one host read per valid vector and validates the
   received vector again. No CuPy/native import. Tests mirror `transfer_preflight_oracle.py` (bad B, bad E,
   bad dtype, missing metadata, lying `get()`, valid transfer-once).
5. **Fresh-driver restore contract.** `restore_seasonal_mass` now validates the population, requires STEP
   mode, driver `day == 0` (and coupled accepted `days == 0` for a `CoupledSimulation`-like object), the same
   ID/species/live association as capture (shared `associate()` helper) and `E ≤ actual driver B`, all before
   any mutation; a mid-run call raises and leaves driver E unchanged. No clock is reset; no coupled restart.
6. Docs (§2.3, §4, §7) and this report updated; `__all__` lists extended (`derived_record`, `associate`).

## Honest notes

* `_to_host` in `leaf/runtime.py` is no longer used by the read path (kept as a tiny helper); no behaviour
  depends on it.
* Packaging: `pyproject.toml` is unchanged (setuptools backend, readme `docs/phase0-vegetation.md`); no
  wheel/sdist check was run by me.

## Tests NOT RUN by me (for Codex)

`python -m unittest tests.test_phase0 -v` (PYTHONPATH incl. `candidate`; legacy-driver tests skip without
numba/benchmarks): the revision-2 suite plus the focused additions above (quoted-field acceptance and
refusal, inverse-overflow fixture with `b_d = 0.5`, overflow→`CatalogError` conversions, saved-config exact
replay in both modes, device-vector metadata preflight, fresh-driver restore contract incl. coupled days and
association/E≤B refusals). Also not run: Codex's independent oracles (`review_probes_01.py`,
`review_geometry_recording_01.py`, `generation_oracles.py`, `cover_memory_replay_oracles.py`,
`transfer_preflight_oracle.py`), the legacy CPU/GPU suites, the sdist/package inspection, and any timing.
