# LEAF periodic canopy and root boundaries — revision 4 — author report (Claude)

## 00. Revision 4 (cover mask lifetime; revision-3/2 text follows unchanged below, archives preserved)

Reviewed revision 3: CORRECT numerically and on interfaces (CPU/CUDA suites, native N15, endpoint and geometry oracles). One measured-memory
defect: in `canopy._paint_wrapped` each row tile assigned `mask = (xd2[None,:] + yd2[ja:jb,None]) < r2` while the previous tile's `mask` was still
referenced, so the peak held float sum + old bool + new bool (10 B / tile element) against the admitted 9 (reproduced by Codex: 6,705,429 traced
vs 6,291,472 declared on a 256 x 256 x .25 periodic domain, subsamples 4, 4 MiB working budget).
Read: `../review_03.md`, `../verify_periodic_cover_memory.py` (read-only, not edited or executed). **Nothing executed by me; all checks NOT RUN.**
Files written: `leaf/canopy.py`, `tests/test_periodic_boundaries.py`, this report only.

* **Fix:** `del mask` at the END of each row-tile iteration, after all y / x segment ORs (loop body indentation of the tile loop). No change to the
  painting arithmetic, no budget / limit / scratch-estimate change.
* **Regression** (`CoverTests.test_wrapped_row_tiles_do_not_retain_a_mask_beyond_the_declared_memory`): the approved fixture (one oversized synthetic
  circle, `Domain(256,256,.25,periodic=(True,True))`, subsamples 4, 4 MiB working budget) asserts several tiles, cover 1.0, and traced peak
  `<= raster_bytes + working_bytes + index_bytes + 32 KiB`; it should fail before the fix and pass after.
* Still required: fresh source qualification of the final hashes (`canopy.py` changed); earlier statements about the tangency correction, no law
  change and the listed restrictions are unchanged.

## 0. Revision 3 (this revision; revision-2 text follows unchanged below, archived copy preserved)

Reviewed revision 2: CORRECT (727 tests, 674 pass, 53 skips; golden / tangency / union / memory checks pass). Two small endpoint issues remained.
Read this round: `../review_02.md`, `../verify_periodic_mixed_endpoints.py` (read-only, not edited or executed). **Nothing executed by me; all
checks NOT RUN.** Files changed: `leaf/state.py`, `leaf/plant_pet.py`, `tests/test_periodic_boundaries.py`, `docs/periodic-boundaries.md`, this report.

1. **`Domain.wrap_positions` exact upper.** The exact endpoint is now selected on the ORIGINAL input (`v == vmax` -> exactly `origin`), in addition
   to a result that rounds onto `vmax`; before, only the result was tested, so `v - origin` rounding ABOVE the span before the modulo returned
   `origin + 1 ulp` (e.g. `Domain(3,2,.1,origin_x_m=.3,origin_y_m=-.3)`: wrap returned .30000000000000004). Values already in `[min, max)` are
   still returned bit-for-bit; nothing is tolerance-snapped; non-finite refusal unchanged.
2. **Mixed-axis canopy-PET validation.** In the any-periodic branch of `plant_pet.build_canopy_map` BOTH axes are now validated on the ORIGINAL
   landscape coordinates against the Domain bounds (an outside point is refused BEFORE any canonicalisation / clipping); then periodic aliases are
   canonicalised and the closed NONperiodic local coordinates are clipped to the span (`np.minimum(x, W)`), so a legal closed-upper position is no
   longer refused because `x - origin` rounded past `W` (e.g. `Domain(3,7,.1,origin_x_m=.3,origin_y_m=-.3,periodic=(False,True))`). No new
   P-sized scratch beyond the already admitted masks, no Python plant loop; the default nonperiodic branch and its numerics are untouched.
3. **Tests** (only `tests/test_periodic_boundaries.py`): the exact failing tuples (3,.1,.3), (3,.2,.6), (3,.3,.6), (7,.2,.4), (7,.7,1.1) added to the
   upper-alias cases; broad wrap test over 48 decimal grids (lower / interior bit-exact / just-inside unchanged / exact upper == exact lower /
   outside wrapped inside `[min, max)`); mixed-axis mapper test over both flag orders, decimal and large origins with BOTH exact upper edges
   (root and PET maps equal the periodic-alias-only twin, reference and compiled) and just-outside refusal on every side.

Unchanged and still required: fresh source qualification of the changed modules; the exact-integrator tangency correction (rare legacy output
change); no ecological-law change. Remaining restrictions: square cells for the root / PET builders, JSON-only boundary config, unwrapped root-row
order, MAPLE importer flag propagation is an upstream developer step. Suggested commands: `python -m unittest tests.test_periodic_boundaries -v`,
the full `discover` run, and `LEAF_CHECK_ROOT=$PWD python ../verify_periodic_aliases_v2.py` / `../verify_periodic_mixed_endpoints.py`.

---

# Revision 2 report (kept)

Task `periodic_boundaries_20261112`, baseline LEAF `85651c63003fcd0a66ba3a98af104003f64e818e`; same session as revision 1
(independently reviewed CORRECT, not accepted: 721 tests, 667 pass, 1 failure, 53 skips). Candidate left **uncommitted**.
**Nothing was executed by the author** (file tools only: Read / Edit / Write): every check below is **NOT RUN**. No acceptance,
validated-ecology or performance claim. Reads: candidate/**, `../review_01.md` and the two approved read-only Codex oracles
`../verify_periodic_aliases_v2.py`, `../verify_periodic_canopy_union_v2.py` (inspected, not edited or executed). Writes only in the
19 revision-1 paths.

## 1. Exact fixes (revision 2)

1. **Exact upper seam alias (decimal grids / origins).** The exact endpoint `x == Domain.x_max_m` (equality on the LANDSCAPE
   coordinate, no tolerance) is now the lower edge everywhere:
   * `Domain.cell_indices`: periodic axes clip to `[0, n-1]` (a real just-inside value keeps the last cell) and the exact endpoint is
     set to index 0 (replaces the old `mod n`, which failed when `x - origin` rounded below the span); nonperiodic rule unchanged.
   * `Domain.wrap_positions`: values already in `[min, max)` are returned bit-for-bit; the exact upper endpoint, including a wrapped
     result that rounds onto it, becomes the lower edge. Non-finite input is now refused for EVERY domain (including no periodic axis);
     ordinary valid nonperiodic output is unchanged.
   * New `leaf.periodic.localize_inplace` / `canonicalize_local` (vectorised, no Python population loop): domain-local coordinate 0.0
     for the exact endpoint, local values kept in `[0, L)` (an entry that `x - origin` rounded to `>= L` becomes the largest double
     below `L`). Used by `roots.build_root_map` (extra `periodic_alias_masks` 4 B/plant admitted in `roots.horizontal_inputs`, periodic
     only), `plant_pet.build_canopy_map` (PET preflight adds 4 B/plant for periodic; periodic axes validated on the landscape
     coordinate so a legal upper point is never refused because `x - origin` rounded past the span) and `canopy.canopy_view` /
     `maple_vegetation_view` (public periodic centers are half-open `[0, L)`; exact stored upper endpoint exported as 0; the
     `PlantState` / CSV record is unchanged; one ID, one object, no image record; no byte ledger exists in the O(P) export, <= 2 B/plant/axis).
   * Nonperiodic branch, mixed axes and legacy admissions are unchanged (new keys / masks only for periodic domains).
2. **Integrator tangency defect** (`plant_pet._make_core.area`, shared by nonperiodic and periodic, Python and Numba): the y-breakpoint
   test now includes equality (`yy >= -r and yy <= r`), restoring the zero breakpoint when a rectangle side is exactly tangent. Same
   integrator and law; `area(.5,[-.5,.5],[-.5,.5])` = pi/4 (was 1), radius .25 in a .5 cell = 0.19634954084936207 (was .25). Documented
   in the module docstring and `docs/periodic-boundaries.md` §3.3 as a rare legacy output correction requiring fresh source qualification.
3. **Demand normalization restored.** `biocalendar.rebuild_geometry` again uses `np.pi * radius * radius` (identical to the standalone
   `IntereventSession`) for every boundary and radius. `leaf.periodic.canopy_areas_m2` (and its `_anti` / `_quadrant_area`), the
   `biocalendar` periodic import and the `+32 B/live` admission were removed. The `CanopyMap` PET support / denominator remain the exact
   toroidal union (unchanged, including oversized circles). Docstrings (`plant_session`, `biocalendar`, `plant_pet`, `periodic`) and docs
   corrected; the claimed standalone-vs-session gap no longer exists.
4. **Default report compatibility.** The current `canopy.measure_cover` adds `periodic` only when an axis is periodic (the key set of the
   default report equals the legacy set); a regression test pins the exact legacy key set for default and `[False, False]` domains.
5. **MAPLE handoff made precise** (docs only): MAPLE `src/maple/case_tools/importers/leaf_vegetation.py::leaf_domain` builds `Domain`
   without flags and `vegetation_from_leaf_population` requires `Domain` equality, so MAPLE developers must derive strict `(x, y)` flags
   from the `GeometrySpec` boundary kinds and pass them through that helper, the imports and the comparisons (no wind / rasterizer work;
   MAPLE already wraps footprints / shelter). Canonical upper export explained. The wrapper's 2 m `placement_inset` (typo "a2m" fixed) AND
   the inner-area cover correction are removed only when the developers choose to regenerate. MASY passes / checks the shared flags; no new
   water step or debit. No upstream edit, case regeneration or trial run here.

## 2. Files changed in revision 2
`leaf/state.py`, `leaf/periodic.py`, `leaf/roots.py`, `leaf/plant_pet.py`, `leaf/canopy.py`, `leaf/biocalendar.py`, `leaf/plant_session.py`
(docstrings), `tests/test_periodic_boundaries.py`, `docs/periodic-boundaries.md`, `docs/maple-leaf-integration.md`,
`docs/masy-daily-plant-interface.md`, `docs/phase0-vegetation.md`, `periodic_report.md`. Unchanged this revision: `initialization.py`,
`plant_checkpoint.py`, `checkpoint.py`, `recruitment.py`, `docs/species-root-profiles.md`, `examples/periodic/generation.json`. No change to
`growth.py`, `uptake.py`, `interevent.py`, benchmarks, existing tests, species catalogs, locks, cases or upstreams.

## 3. New / changed regressions (all in `tests/test_periodic_boundaries.py`; existing tests untouched)
* `UpperAliasTests`: decimal spacings / origins (incl. the review example `Domain(3,2,.2,origin_x_m=.4,periodic=(True,False))` and
  `origin 100000.1`): exact upper endpoint samples 0, wraps to the lower edge, exports public centers 0 (canopy view and MAPLE view), root
  and canopy maps bit-equal to the lower-endpoint twin (reference and compiled builders), stored record unchanged; a real just-inside value
  (`nextafter`) keeps the last cell / an unchanged wrap / a center strictly in `(0, L)`; mixed axes and nonperiodic edge keep their rules;
  `wrap_positions` non-finite refusal for all domains and unchanged valid nonperiodic draws.
* Tangency: exact circle-rectangle values (Python and compiled) and nonperiodic / periodic canopy maps (circle tangent to its cell sides,
  inscribed circle of a one-cell torus).
* Oversized canopy: demand area is bit-equal to `np.pi * r * r` through `biocalendar.rebuild_geometry` while the PET map total equals the
  independently computed union (`circle_rectangle_area` on the nearest-image rectangle) and is below the nominal area.
* `measure_cover` default key set (legacy) vs periodic-only `periodic` field. The former `canopy_areas_m2` helper test was removed; the
  oversized-canopy test now checks the union against the independent clipped-rectangle area.

## 4. Laws preserved
Root law (uniform weights over unique cells, layer fractions), canopy PET = area-weighted mean with the same integrator (now exact at tangency),
proportional allocation / no priority / no compensation, nominal allometric demand area, lognormal sizes / exclusion / cover fitting, recruitment
probabilities and ledger, SYRUP sole water authority. Seeded nonperiodic generation, default Domain serialization and nonperiodic root / PET arrays
are unchanged by construction (nonperiodic code paths untouched, except the tangency numerical correction which can only change exactly tangent cells).

## 5. Genuine remaining restrictions
* Square cells only for the root / PET builders (explicit refusal kept).
* Boundary configuration is JSON / `Domain` only: `__main__.py` (not editable) has no CLI flag for periodicity.
* Cell order inside a periodic root row is the unwrapped footprint order, not ascending.
* Periodic builds with `builder="numpy"` are pure-Python references (slow by design).
* The MAPLE importer propagation (§1.5) is an upstream developer step, not done here. Any unlisted module that rebuilds a `Domain` from fields
  could drop the flags (not inspectable here).
* The tangency correction and all source changes need fresh Codex qualification (new source hashes; no old lock rewritten).

## 6. Tests — NOT RUN by me; suggested exact commands
```sh
python -m unittest tests.test_periodic_boundaries -v
python -m unittest discover -s tests -v                                    # full baseline suite (existing tests untouched)
LEAF_CHECK_ROOT=$PWD python ../verify_periodic_aliases_v2.py               # Codex oracles (not edited by me)
LEAF_CHECK_ROOT=$PWD python ../verify_periodic_canopy_union_v2.py
python -m leaf generate --species examples/phase0/species.json --config examples/periodic/generation.json \
    --surface-constant 0 --out /tmp/periodic_plants.csv --report /tmp/periodic_report.json
# native (configured env): LEAF_NATIVE_BUNDLE=... LEAF_NATIVE_CASE_DIR=... LEAF_NATIVE_BACKEND=numpy|numba|cuda \
#   python -m unittest tests.test_periodic_boundaries.NativePeriodicN15Tests -v
```
Most likely to need correction on first execution: Numba compilation of the unchanged kernels (never run), exact bit-equality of lower-vs-upper
maps on all listed decimal cases, and the new session-free oversized-canopy test (`rebuild_geometry` arguments).
