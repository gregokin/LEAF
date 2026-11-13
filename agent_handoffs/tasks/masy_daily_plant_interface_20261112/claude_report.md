# Interface report — masy_daily_plant_interface_20261112, correction revision 4 (Claude)

**All author checks are NOT RUN** (no execution tool). Nothing is committed. No "all passed" claim, no full-coupled or ecological-validation claim. Independent CPU / CUDA / old-suite evidence is Codex's and will be re-run on this final source.
Same session as revisions 1–3; this is a narrow correction (reviews `review_03.md`, `review_02_rng_addendum.md`). Only the allowed 16 candidate files were written.

## Changes in revision 4
1. **Restart configuration identity (HIGH).** New `SessionConfig.identity()` / `identity_sha256()`: catalog hash, cadence, the ACTUAL `rng_seed`, `patch_size_m`, every per-individual root layer override (also for plants that are not live now) and the canopy valid-cell-mask hash. It is part of the
   session token body (`PlantSession._make`), stored in the checkpoint manifest (`config`, `config_sha256`, plus `laws_active`; the old `rng_seed_set` flag stays only as information) and checked by `load_biology`, which refuses any difference with the list of differing keys. A `SourceOverride` does not apply to it.
   Deliberately caller-adjustable on restore: byte / edge ceilings and the root / canopy builder choice (identical algorithms, protected by the geometry hash); the backend is fixed by the compiled table the caller passes. No ecological law or parameter changed; the unreleased codec stays version 2; layout 4 and plain `plants.csv` are untouched.
2. **Test fix.** `test_transpiration_off_growth_off_is_calendar_only` now asserts the explicitly bound cursor is preserved (equal to the parent's, counts 0 / 0).
3. **Regressions.** `test_plant_checkpoint.py`: identity bound to seed / patch / cadence / overrides / mask / missing seed, ceilings adjustable, `SourceOverride` cannot bypass. `test_plant_boundaries.py`: restart before the annual reproduction opportunity continues with identical keyed draws (births, ids, biomass), and a different seed or patch is refused.
4. **Documentation** (`docs/masy-daily-plant-interface.md`, rewritten): complete public signature / return table (config, day / binding objects, start / connect, prepare, propose, feedback, stage / publish / discard, exports, Phase 2/3 boundaries, ledger, save / load / `rebuild_pending`, bridge read / `request_args` / conversions / accepted result / source binding), symbols P, S, L, N, R, E, dtypes, units, ordering, ownership, backend / device rules,
   the exact daily SYRUP call sequence as in the example with the shared byte cap / reserve, nominal fragments, a crash-window table (pending-after-debit resume, `StageId` pairing, exact external reference), the `snapshot_state` + `leaf.canopy.canopy_view` route for accepted allometry (`export_canopy` is demand map + areas only), memory / transfer notes and the restart configuration rules.
   Stale revision title, the "`request_sequence >=`" sentence and the "CPU/GPU memory / parity future work" paragraph are gone; qualification is pointed to the independent task evidence. The MASY combined scheduler / checkpoint manager is stated as MASY's responsibility.
5. **Example.** All three conversions now pass the shared `reserve_bytes`; the stale `unsupported_boundaries()` call was replaced by `due_boundaries()`.

## Unchanged / preserved
Growth, PET, uptake, the Phase 2/3 evaluators and the real-native sequence code paths were not edited. `biocalendar.py` keeps the revision-3 memory-accounting divergence from C1 (canopy cap = `min(canopy cap, global cap)` with the live reserve); `growth.py` and `plant_pet.py` are byte-identical to C1.

## Remaining limitations (real)
* Unexecuted source: syntax / API-shape slips are possible. Inferred shapes noted earlier remain (`uptake.estimate_map_bytes`, `RootMap`, `PlantState._arrays`, evaluator return keys, Phase 2/3 example catalogs); I never read the bodies of `demography.decide`, `disturbance.evaluate_event`, `recruitment.evaluate`.
* The configuration identity covers only what the session passes into the biology; a caller changing the catalog file without changing its hash cannot be detected. Backend change on restore is governed by the compiled table, not by an explicit manifest field.
* Geometry restore assumes radii do not depend on age beyond the state (hash-checked); memory figures are structural estimates; the interface was exercised only on small fixtures; T off / G on remains refused (C1 rule).


## Revision5 final report (structured successful result)

Revision 5 is done. I wrote only `tests/test_lifecycle_bridge.py` and `docs/masy-daily-plant-interface.md`; no code or other files changed, and nothing was run (author checks are NOT RUN).

**Test fix.** The `dir(obj)` scan that matched `stage_connect_water` is gone. In `test_visible_daily_sequence_counts_water_closure_clock_and_shared_pet`, the check now inspects stored data on the session and on its working population:
- no stored value is a SYRUP state, operator, or the boundary or day state objects;
- every stored array is 1-D (per plant or per species);
- none has the `(L, N)` grid shape.

The real native checks (factory, source, clock, one hydrology plus one T, closure) are unchanged.

**Docs**
- **Status:** now points to `final_review.md` and `qualification.json` for Codex's independent evidence. It separates tested interface mechanics from uncalibrated ecology, does not assume final acceptance exists, and states that the source pair is the published N15 package, not live checkouts.
- **Reserve:** `reserve = session.config.reserve_bytes + session.resident_bytes()` now sits inside the per-day loop, with a note on why.
- **Accounting:** I added that passing the same cap is not accounting by itself. The caller-held native state must go into LEAF's `reserve_bytes`, and LEAF's bytes must reach SYRUP through its preparation reserves.
- **Signature:** `load_biology` now shows `day_binding_check=True`. That keyword has no behavioural effect in the code today, so I added a short note saying not to rely on it.
