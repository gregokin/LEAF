# Phase 0 vegetation infrastructure — implementation report, revision 2 (bounded correction)

Writer: Claude (bounded LEAF Phase 0 implementer). All work is uncommitted in `candidate/`. Only the 17
allowed candidate paths were written; the Codex review artifacts were read, not edited. No shell, Python
execution, installs, network, Git or execution of any check was performed by me. **Every test and probe
listed under "Tests NOT RUN" is for Codex to execute.** No scientific scope was widened: no growth,
mortality, injury or damage law was added; legacy numerical modules are untouched.

## A. Test outcomes fixed for the right reasons

1. Anchor test: the three explicit anchors are left intact; the test now checks
   `delta = sampled_current − stored_anchor` for **all** rows and the exact 0.3 / −0.2 changes only on the
   SURFACE subset (placement semantics unchanged).
2. Transaction test: counts after removing id 1 are three live, one dead_standing, two removed.
3. Cadence: snapshots at 10 then 35 list missed times `[20, 30]`, save only the actual t=35 state,
   `next_due = 40`; a snapshot at 30 after one at 10 lists `[20]` (both tested).
4. Collision test: both writers now target an **actual** matching destination (`veg_{kind}_{time_days}`),
   `FileExistsError` is asserted for the initial and the cadence path, and the failed writer's scheduling,
   records and skipped totals are shown unchanged.

## B–H. Corrections implemented

| area | change |
|---|---|
| state | explicit `establishment_unresolved` mask required (bare NaN/inf refused); metadata-bearing non-initial or undeclared files with `SURFACE` refused **before** any surface sampling; unresolved states cannot `advance_time`; `advance_time` preflights clock+ages (overflow → `StateError`, nothing changes); double quotes rejected anywhere in the CSV (strict reader); constructor copies inputs, all canonical arrays read-only, identity fields unassignable, by-ID index cannot go stale; `Domain` immutable, finite positive area, conflicting metadata (periodic edges / other sampling rule) refused; coordinate labels corrected to landscape frame; `preflight_destination`; atomic no-overwrite never falls back to check-then-replace (refuses when hard links are unsupported) |
| traits | `document` is a deep copy of private content; coefficient tables frozen; species table = sorted IDs (deterministic across key order) + `check_species_table` used by CSV/canopy/bridge; non-finite allometry **outputs** refused; bool/complex/str inputs refused |
| canopy | `maple_vegetation_view(state, catalog, current_surface, ...)` requires the current surface; default `exposed` heights; fully buried and eroded anchors unsupported (error/exclude/flag) with `native_maple_valid` per object and per view; `intact_diagnostic` mode labelled, never native-valid; state + derived outputs validated; `cover_admission` with precisely scoped `raster_bytes` + `working_bytes` (row-tiled footprint temporaries, `max_working_bytes`), `index_bytes` reported; boundary note |
| initialization | **large-canopy-first** exclusion placement (sampled sizes sorted descending, uniform proposals, global canopy exclusion, sparse bins), complete re-placement after count corrections, `max_attempts` = total proposals; incremental counts, compacted per-species sample storage (no dead slots), counts admitted before sampling; `cover_tolerance` mandatory finite positive; non-finite derived quantities refused; complete `generation_config` + RNG/NumPy record in report and provenance; mixed-size 128²/0.25 m regression (seed 7751, `max_attempts=50000`) |
| recording | all scheduling/index/provenance changes staged until the file exists (initial included); missed due times per A.3 |
| events | immutable records (owned read-only arrays, no attribute rebinding), `ReadOnlyView` with frozen copies and the immutable domain, IDs resolved once and re-verified after the handler, redirect/mutation attempts leave state/domain/IDs unchanged |
| runtime | state validity, species table, exact integer ID/profile dtypes and ranges (unsigned overflow), unknown profile names, finite non-negative `rtol`, finite positive STEP density checked; **species-law** consistency (`b_d = 0.5`, `a_d = 2/√(πρ)`, ρ = 1.0 fixture or STEP density) instead of one initial B; nonzero seasonal E refused unless `restore` (+ `restore_seasonal_mass`) or `acknowledge_reset`; non-STEP capture never erases population E silently (`keep`/`zero` explicit); clock/ages documented caller-owned; no restart claim |
| CLI | `export-canopy` requires the current surface, `--height-mode exposed|intact_diagnostic`, warning when not native-valid; `generate --report` aliasing `--out` refused before generation, both destinations preflighted, both outputs serialised before any commit, report removed if the CSV write then fails |
| packaging/docs/examples | `pyproject` readme → `docs/phase0-vegetation.md`; `__all__`/lazy table consistent; docs rewritten for the corrected APIs, safe publication, boundary semantics, bridge policies; `plants.csv` anchors valid against the documented current surface (99.9 / 100.0 / 99.8 vs 100 m), `generation.json` at 128² × 0.25 m with grass 0.15 ± 0.03 m and shrub 2.0 ± 0.3 m |

## Honest notes / limitations

* Nothing was executed. The mixed-size exclusion regression is designed to need roughly a few thousand
  proposals per placement pass (≈3.1 k grasses + ≈16 shrubs); expected total well under 50 000 but
  **unmeasured**.
* The tracemalloc-based test asserts the measured peak ≤ declared `raster_bytes + working_bytes` + 256 KiB
  slack; the slack covers interpreter/bookkeeping objects and is a test tolerance, not part of the admission.
* `cover_admission` keeps `max_cover_bytes` as the raster bound and adds a separate explicit
  `max_working_bytes` (default 1 MiB) for the tiled working set; both terms are reported precisely.
* A handler can still rebind attributes through `object.__setattr__`; the transaction detects any ID change
  and refuses, and proposals are always applied to the IDs resolved before the handler ran.
* `restore_seasonal_mass` writes the population E into a freshly constructed STEP driver's `E` array by ID;
  it does not alter the driver's science and is the only way `seasonal_policy='restore'` completes.
* Exclusion placement is a bounded initialisation algorithm; exhausting `max_attempts` is reported as
  inability within the bounds, not as proof that no packing exists.

## Tests NOT RUN by me (for Codex)

`python -m unittest tests.test_phase0 -v` (PYTHONPATH incl. `candidate`; legacy-driver tests skip without
numba/benchmarks): 7 catalog, 2 domain, 10 plant-state (incl. explicit-mask/ownership, metadata-noninitial
SURFACE refusal, quote rejection, overflow preflight, atomic no-fallback), 5 canopy (safe publication,
admission + tracemalloc scope, tiling equivalence), 5 generation (incl. the 128² mixed-size exclusion
regression and mandatory tolerance), 2 recording (missed times, staged bookkeeping, real collisions),
2 events (immutability, hostile handler rollback, ID redirect refusal), 5 bridge (type checks, seasonal
policies, species-law consistency, restore, real driver capture), 6 CLI/examples/import-isolation/lazy
exports. Also not run: Codex's independent probes (`review_probes_01.py`, `review_geometry_recording_01.py`,
`generation_oracles.py`), the legacy CPU/GPU suites, wheel build/packaging, and any timing.
