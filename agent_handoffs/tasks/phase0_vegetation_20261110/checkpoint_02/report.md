# Phase 0 vegetation infrastructure — implementation report (revision 1)

Writer: Claude (bounded LEAF Phase 0 implementer). All work is uncommitted in `candidate/`. No shell,
Python execution, installs, network, Git or writes outside the allowed paths were used. **No check in this
report was executed by me**; every test listed under "Tests NOT RUN" is for Codex to execute.

## Files written (all within the allowed scope)

| path | purpose |
|---|---|
| `leaf/__init__.py` | replaced eager exports with lazy `__getattr__`; `import leaf` imports only NumPy-free stdlib; `CoupledSimulation`, `CoupledPhaseError`, `stepwat`, `coupling` resolve on first access |
| `leaf/traits.py` | versioned species catalog JSON (`leaf.species_catalog` v1): strict parser (duplicate keys, NaN/Infinity, bool-as-number), power-law allometry + exact inverse, roots/porosity/life-expectancy/law sections, canonical sha256 hash |
| `leaf/state.py` | `Domain` (axes `(ny,nx)`, closed non-periodic, `containing_cell_closed_upper` sampling), `PlantState` SoA with stable int64 IDs, status codes, SURFACE resolution once, calendar ageing, transactional by-ID updates, plant-state CSV (`leaf.plant_state_csv` v1) with exact FP64 round trip, atomic no-overwrite writes |
| `leaf/canopy.py` | derived geometry, burial/exposure diagnostics, MAPLE-compatible view (fields + ID sidecars, explicit unsupported-case policies), bounded midpoint-raster per-species union cover |
| `leaf/initialization.py` | cover-driven multispecies generator: lognormal diameters (E[d²] = mean² + std²), Poisson-Boolean / nominal starting counts, measured-cover count fitting with declared tolerance, overlap-allowed vs global canopy exclusion (sparse hashed bins, random interleave), explicit bounds, full report |
| `leaf/recording.py` | `SnapshotWriter`: explicit initial/final, cadence in model days, skipped due times recorded (never fabricated), validate-then-atomic-write, rollback on failure |
| `leaf/events.py` | `SurfaceEventDiagnostics` (schema v1, by stable ID, explicit times/units, distinct erosion/deposition/abrasion/flux), validated import, diagnostics-by-default response socket with all-or-nothing handler application |
| `leaf/runtime.py` | ID-keyed bridge: `capture_snapshot` (B and STEP E from `Simulation`/`CoupledSimulation`, profile/ID/status checks), `legacy_inputs_from_state` (refuses silent allometry mismatch with the legacy area law; replay ≠ full restart) |
| `leaf/__main__.py` | CLI `validate-species`, `init`, `generate`, `inspect`, `export-canopy`; `.npy`/constant surfaces; atomic outputs; nonzero exit on error |
| `tests/test_phase0.py` | unittest suite (see below) |
| `docs/phase0-vegetation.md`, `docs/maple-leaf-integration.md` | schemas, API/CLI, conventions, coverage approximation, reproducibility, age/cadence/replay, MAPLE handoff |
| `examples/phase0/species.json`, `plants.csv`, `generation.json` | two fixture species (porosity 0.20, allometry, growth/water-use/reproduction/life-expectancy/mortality/disturbance sections), SURFACE + explicit anchors, dead/removed rows, cover-driven generation |
| `pyproject.toml` | installable `leaf` with NumPy-only core; optional `growth` (numba) / `gpu` (cupy) extras; `benchmarks` included only for the lazy legacy exports |

Unchanged (read only): `leaf/coupling.py`, `leaf/stepwat.py`, `benchmarks/*`.

## Behaviour summary

* Independent interface: pure stdlib + NumPy. No numba/cupy/maple_syrup/benchmarks import on
  `import leaf`, `leaf.state`, initialization or CLI (checked by a subprocess test that Codex runs).
* Species catalog validated before use; catalog hash accompanies every CSV, generation report and view.
* One CSV schema for initial input, snapshots and final output; optional `# {json}` metadata line; exact
  FP64 round trip; all-or-nothing reads; SURFACE token resolved once from the initial surface and preserved
  afterwards; unresolved states cannot be recorded or published.
* Age advances with calendar time for live plants only (`advance_time`), including wind-only intervals;
  dead/removed ages frozen; exports never age; life expectancy is metadata only (no automatic death).
* Geometry derived on request (intact allometric height; exposed height and burial/exposure diagnostics
  only when a current surface is supplied); MAPLE view in domain-local metres; eroded/buried cases handled by
  explicit policy, never flattened silently.
* Generator: counts fitted to measured per-species union cover (same-species overlap once, clipped to the
  domain), overlap allowed or global exclusion, bounded, reproducible from seed + config + catalog hash;
  unattainable requests raise with history.
* Recording: write only when due; skipped due times listed; atomic; no overwrite by default; writer state
  unchanged on failure.
* Events: diagnostics by default; handler socket with rollback; unknown IDs/bad data fail before mutation.
* Bridge: by-ID capture; legacy-driver input construction refuses the silent allometry mismatch by default
  and documents that replay is not a coupled restart. The legacy driver's science is untouched.

## Limitations / honest notes

* No growth, water use, recruitment, mortality, injury, dislodgement, surface remapping or calibration
  are implemented; example numbers are fixtures.
* Cover is a sampled midpoint-raster approximation with reported resolution/error basis, not analytic area.
* Generation and cover measurement use Python-level per-plant loops over NumPy bounding boxes (O(P)); the
  ~60 k / 512² target is expected to take seconds per measurement, but **this was not timed** (Phase 0
  bounded performance checks are authorized for Codex).
* `atomic_write_bytes` refuses existing files before writing and again via `os.link` where hard links are
  supported; on filesystems without hard links it falls back to `os.replace` after re-checking existence
  (a narrow race remains there).
* The lazy `from leaf import *` still imports the legacy coupling (numba) for `CoupledSimulation`, exactly as
  the previous eager module did.
* `legacy_inputs_from_state` requires a zero-origin square-cell domain and one lateral root multiplier
  across live species (legacy driver constraints); dead/removed plants are excluded from driver inputs.
* The catalog hash is sensitive to numeric spelling (`1` vs `1.0`), documented.

## Tests NOT RUN (for Codex, `tests/test_phase0.py`, unittest)

`python -m unittest tests.test_phase0 -v` with PYTHONPATH including `candidate` (and SYRUP/src only for
the legacy lazy-export / real-driver tests, which skip when `benchmarks`/numba are unavailable):

* catalog: load, hash stability across whitespace/key order, positive/monotone/inverse allometry,
  section preservation, STEP trait extraction, 17 schema-error cases;
* domain: validation, subcell/offset/non-square containing-cell sampling, closed edges, no wrapping,
  transposed/NaN/bool fields refused;
* state: SURFACE resolved once and anchors preserved under deposition/erosion diagnostics; 17 invalid-row
  cases refused without partial state; empty population round trip; exact FP64 CSV round trip with
  metadata checks (catalog mismatch, domain mismatch, truncated/malformed rows, bad header, bad version,
  missing domain); transactional updates/transitions; calendar ageing in wind-only mode and no automatic
  death; ID resolution independent of row order;
* canopy: purity, porosity, MAPLE view fields/IDs/domain-local coordinates, statuses, unsupported
  elevated-base/fully-buried policies, exposed heights, unresolved refusal; analytic circle and clipped
  quarter-circle cover, same-group vs cross-group union, non-square offset domain, bounds, empty input;
  independent finer-resolution agreement;
* generation: E[d²] count formulas, lognormal moments, reproducibility, overlap-allowed species cover
  within tolerance with sample statistics, global non-overlap (all-pairs check on a small case) with
  surface resolution, bounded/impossible/zero-cover/empty cases;
* recording: cadence initial/final/skip/overwrite/error/rollback behaviour and manifest;
* events: construction/dict validation, future-event and unknown-ID refusal, default diagnostics, handler
  failure/invalid-proposal rollback, read-only view, applied proposals;
* bridge: by-ID capture with reordering, profile/age preservation, 7 refusal cases with state unchanged,
  legacy input refusal/acknowledgement/consistency, real `daily_water_growth.Simulation` capture
  (skipped without numba);
* CLI: help/usage, validate-species, init (SURFACE/explicit anchors, no overwrite, unresolved export
  refused), inspect with surface and cover, transposed surface refused, export JSON/NPZ, example
  generation config end to end; subprocess import isolation and `python -m leaf --help`; legacy lazy exports.

Also not run: existing CPU/GPU legacy suites (must still pass — the only legacy-facing change is the lazy
`leaf/__init__.py`), `pip`/build of `pyproject.toml`, and any performance timing.
