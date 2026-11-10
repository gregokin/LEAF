# Report — shared aboveground canopy view (bounded Phase 0 interface correction, revision 1)

Author: Claude (sole implementation writer, candidate/ scope). Orchestration, review, execution, installs
and Git: Codex. Date: 2026-11-10.

## 1. Files written (all inside candidate/, all on the allowed-write list)

| file | change |
|---|---|
| `leaf/canopy.py` | new public `canopy_view` (schema `leaf.canopy_view` v1); `maple_vegetation_view` refactored into a thin adapter over it; `CANOPY_FIELDS` (+ `MAPLE_FIELDS` alias), schema/unit constants; module docstring; `__all__`. `derive_geometry`, `cover_admission`, `measure_cover`, resource bounds and all numerical code untouched |
| `leaf/__init__.py` | module description line for `leaf.canopy` only |
| `tests/test_phase0.py` | new helper `assert_projection_of` and class `TestSharedCanopyView` (3 tests) inserted before `TestGeneration`; no existing test modified |
| `docs/shared-canopy.md` | NEW shared contract for MAPLE and SYRUP |
| `docs/phase0-vegetation.md` | §3 rewritten (3.1 shared view, 3.2 thin adapter); §10 limitation sentence |
| `docs/maple-leaf-integration.md` | title/intro, §1 table, §2 sequence, §3 adapter description, §6 refresh note |
| `report.md` | this file |

Not touched: `leaf/__main__.py` (export-canopy still calls `maple_vegetation_view`, the MAPLE-compatible
projection), `leaf/state.py`, `leaf/traits.py`, other modules, benchmarks, fixtures, locks, AGENTS.md,
archives, anything outside candidate/.

## 2. Design

* **One implementation.** `derive_geometry` remains the only allometry / root-radius / surface-sampling
  code. `canopy_view` calls it once, selects the standing population (statuses default
  `('live','dead_standing')`, diameter > 0, as before), and is the ONLY place that converts landscape
  positions to domain-local centres and assembles `fields` / `sidecar` / `meta`. O(P) arrays, no
  rasterization, no physics, no state mutation, no cache, NumPy only.
* **Retention.** The shared view keeps fully buried plants (exposed height 0, `fully_buried=True`) and
  elevated-base plants (`elevated_base=True`, `exposure_depth_m>0`) with correct anchor diagnostics.
  Geometric visibility is separated from ecological status in both data (`status` sidecar unchanged) and
  metadata (`visibility_note`). Per-plant `native_maple_valid` is kept as informational metadata
  (`sidecar_units` says so; `meta['native_maple_compatibility']` carries counts and a note); there is no
  whole-view `native_maple_valid` verdict in the shared view.
* **Thin adapter.** `maple_vegetation_view` keeps its signature, defaults, error types, messages (one
  clause added pointing to the shared view) and schema `leaf.maple_vegetation_view` v2. It validates
  `unsupported_policy`, calls `canopy_view` with the same `statuses`/`height_mode`/`drag_coefficient`,
  derives `unsupported = elevated_base | fully_buried` from the shared sidecars, raises / excludes / flags,
  and for `'exclude'` applies ONE `np.flatnonzero(~unsupported)` selection to every field and sidecar
  array. Metadata keys are the legacy ones (values for domain, hash, time, statuses, exclusion counts and
  surface reference are taken from the shared meta) plus `derived_from`. No geometry, sampling or
  coordinate code remains in the adapter.
* **Consumers.** MAPLE: `canopy_view` → `maple_vegetation_view` for its current container; SYRUP:
  `canopy_view` directly, not inheriting MAPLE restrictions. No SYRUP adapter, rasterizer, roughness law or
  porosity interpretation is introduced; raster derivation guidance (consumer-side, keep `plant_id`) is
  documented. PlantState remains authoritative; water ownership unchanged.

## 3. Compatibility reasoning (by reading, not execution)

Walked every existing `maple_vegetation_view` assertion in `tests/test_phase0.py` against the new code:
`test_empty_population_round_trip` (k=0 path, `native_maple_valid` True), `test_maple_view_safe_default_…`
(ValueError on `None` surface, field list == `MAPLE_FIELDS`, counts, domain-local centres, default error,
exclude heights/ids, flag flags and counts, buried height 0, diagnostic height/label, diagnostic default
still refused, unresolved refused), `test_generate_example_config_and_report_collision` (`n_objects ==
len(s)`), CLI `export-canopy` JSON/NPZ paths (`_jsonable` handles the added nested `derived_from` dict;
NPZ sidecar handling unchanged). Expected outputs are value-for-value identical because the shared view
performs exactly the former selection and array construction and the adapter subsets by a single index.
Minor ordering change: an invalid `unsupported_policy` is now rejected before the surface check (both are
`ValueError`); the non-finite field message now reads "canopy view field …" (no test depends on it).

## 4. Limits

* Interface correction only: no ecological physics, allometry, porosity, resource bound or fixture changed.
* Upstream adoption (MAPLE/SYRUP calling `canopy_view`) and hydrological qualification of
  porosity/height/radius are future work; nothing claims integrated MAPLE/SYRUP execution.
* `excluded_zero_diameter` can only be non-zero for a standing plant with zero diameter, which the state
  invariants (B > 0 for standing plants, a_d > 0) prevent; the key is retained for schema stability.
* The adapter's `'error'`/`'flag'` paths return the shared arrays themselves (fresh arrays per call, never
  aliases of the state); callers that mutate a view mutate only their own copy.

## 5. Tests — NOT RUN

I have no execution tools in this assignment. **No test, import check, linter or CLI command was run.**
Codex must execute:

* `python -m unittest tests.test_phase0 -v` (full existing CPU suite + `TestSharedCanopyView`);
* independent exact field/sidecar checks by `plant_id` across `canopy_view` and every
  `maple_vegetation_view` policy, including native-MAPLE exclusions;
* pure-import check (`import leaf, leaf.canopy` needs only NumPy — no new imports were added);
* native MAPLE ABI check of `export-canopy` output (unchanged fields and sidecar set).

Manual checks performed by reading: indentation/syntax of all edits, every referenced helper exists
(`_traits.nonnegative`, `_state.status_codes`, `_state.STATUS_NAMES`, `require_resolved`), empty-population
and boolean-indexing paths, monkeypatch targets in the new tests resolve to module globals looked up at
call time (`canopy_view`, `derive_geometry` inside `leaf.canopy`), and test-local expectations (cell indices
for the chosen plant positions, burial/exposure values).

## 6. Tool/auth/network errors

None encountered. Only Read/Edit/Write were used; no shell, Python, agents, network, Git, installs.
