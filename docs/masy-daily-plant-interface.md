# MASY daily plant interface (LEAF side) — public handoff

Status: this document describes the interface mechanics only. **The author executed nothing: every author check is NOT RUN.** Independent execution evidence (real N15 CPU / Numba / CUDA daily
harness, unit suites, old standalone suites, measurements) is recorded by Codex in `../agent_handoffs/tasks/masy_daily_plant_interface_20261112/final_review.md` and `qualification.json`; this text does not assume that final
acceptance exists and must be read together with those records. The *interface mechanics* (binding, identities, transactions, bounded memory, checkpoint pairing) are what the independent tests exercise; the *ecology*
(growth, demand, Phase 2/3 laws, parameters) is existing and UNCALIBRATED and is not validated by any of this. No full-coupled-model, field or ecological-validation claim is made. The combined scheduler / checkpoint manager that
pairs SYRUP and LEAF state stays MASY's responsibility. The SYRUP source pair is the explicit published N15 package (`feature/subsurface-daily-recharge` `a5b8b18`, `82e9c011…`), **not** any live checkout.

## 1. Ownership and conventions
| Owner | Owns |
|---|---|
| LEAF (`plant_session`, `growth`, `uptake`, `roots`, `canopy`, `plant_pet`, `biocalendar`, existing `demography` / `disturbance` / `recruitment`) | plants, roots, canopy, potential demand, equal-access allocation, biological accumulators, growth, existing Phase 2/3 population changes |
| SYRUP Phase 1.5 (N15, `a5b8b18`, package `82e9c011…`) | the **sole** water inventory and **every** water update (daily hydrology, T debit) |
| MASY | all forcing / temporal controls, call sequencing, nominal date ↔ SYRUP run mapping, pairing and commit of the two states, combined checkpoints |

The plant interface holds no water, copies no inventory, calls no hydrology / evaporation / standalone debit (neither `syrup_water.withdraw_plants` nor native `withdraw_transpiration`) and hides no operation.

**Symbols.** P = live plants (state row order of live rows), S = species (sorted catalog ids), L = soil layers (top first), N = cells (flat `row * nx + col`, raster `(ny, nx)`), R = L·N resources (layer-major: `layer * N + cell`),
E = root edges (plant-major CSR). **Dtypes.** plant ids int64; species index int32; every mass / water / PET array float64 C-contiguous; flags bool. **Units.** biomass kg dry; demand / supply / actual volumes m³; availability / request / actual
**depth** m of water over the cell area; terrain PET mm day⁻¹ (SYRUP's PET is m s⁻¹: `lifecycle_bridge.pet_m_s_to_mm_per_day` is a pure unit conversion of the SAME field); A, T integrals in elapsed years; times are fixed-offset `datetime`s;
native `t_s` is physical seconds and is **never** a biological date. **Backends / devices.** `numpy`, `numba` (CPU) and `cuda` (CuPy on the table's current device); arrays of one session live in one namespace; no implicit transfer, no device switching,
no silent fallback; the backend must equal the compiled table's and the SYRUP lifecycle's (`lop.xp`); CUDA needs `growth.compile_species(..., 'cuda', max_bytes, device=<current device>)`.

## 2. Public objects and signatures (`leaf.plant_session`, `leaf.lifecycle_bridge`, `leaf.plant_checkpoint`)
| Call | Returns / meaning |
|---|---|
| `growth.compile_species(catalog, backend, max_bytes, *, device=None)` | `CompiledSpecies` table (existing) |
| `SessionConfig(catalog, table, backend, max_edges, max_bytes, cadence_days, reserve_bytes=0, builder='compiled', canopy=None, layer_fraction_overrides=None, rng_seed=None, patch_size_m=None)` | immutable; `cadence_days` int 1..7; `reserve_bytes ≤ max_bytes`; `canopy={'max_edges','max_bytes'[,'valid_cell_mask' bool[N],'builder']}` (needed for per-plant canopy PET); `rng_seed` / `patch_size_m` required when the catalog's active laws need them. `.identity()` / `.identity_sha256()` = declared biology configuration (§7) |
| `DayBinding(run_digest, origin_date)` | 64-hex SYRUP daily run; ISO date of occurrence ordinal 0. `.ordinal_of(date)` |
| `DayContext(start, temperature_c, transpiration_enabled, growth_enabled, process_kind='interevent', duration=None)` | ONE open fragment: `start` fixed-offset instant (= the session cursor), `duration` the ENABLED length (default: to the next local midnight; must not cross it) |
| `TransportDay(start, kind, duration=None)` | `kind ∈ {'wind','rain','recession','closed_gate'}`, any duration (default 1 day) |
| `PlantSession.start(base, config, *, anchor, session_id=None, day_binding=None, initial_resource=None)` | accepted parent at the local-midnight `anchor` (cursor = anchor). Water is connected only with BOTH `day_binding` and `initial_resource` (`ResourceIdentity` of the SYRUP state whose T history is known consumed) |
| `session.stage_connect_water(day_binding, initial_resource)` → `Staged` | later, once, never replaced |
| `session.prepare_daily(day, *, terrain_pet)` → `DailyPlan` | `terrain_pet`: float64 `[N]` mm/day field (canopy-area-weighted mean per plant) **or** a scalar mm/day. `plan.D` `[P]` m³, `plan.report()`, `plan.needs_withdrawal`, random `plan.nonce` |
| `session.propose_withdrawal(plan, read)` → `WithdrawalProposal` | `.proposed_m3` `[R]`, `.identity` (≤128 ASCII), `.retained_feedback_bytes`, `.report()` |
| `session.stage_day(proposal, accepted)` → `Staged` | pure; `staged.feedback` = `DailyFeedback(live_ids[P] int64, requested_m3, delivered_m3, unmet_m3, satisfaction [P], totals{requested,delivered,unmet,accepted,n_plants_unsatisfied,n_plants_zero_demand}, closure)`; `staged.stage_id`, `staged.report`, `staged.status`, `staged.next` |
| `session.stage_dry_day(plan)` · `stage_flush()` · `stage_transport(tday)` · `stage_deferred_season()` | calendar boundaries → `Staged` |
| `stage_surface_event(event, current_surface)` · `stage_annual_decision(year)` · `stage_reproduction(year, current_surface, *, max_new_plants, max_total_records, max_attempts)` · `stage_decline_reproduction(year)` | existing Phase 2/3 evaluators → `Staged` |
| `session.publish(staged)` → new `PlantSession` · `staged.discard()` | exactly once per parent; stale / foreign / duplicate refused |
| `session.export_roots()` | `{'root_map': RootMap (E edges, fixed weights), 'live_ids', 'generation', 'geometry_sha256'}`; `root_map.host_arrays()` is the explicit host export |
| `session.export_canopy()` | `{'canopy_area_m2' [P] (borrowed backend array), 'canopy_map' (sparse demand-PET map or None), 'live_ids', 'generation', 'geometry_sha256'}` — **the demand map and areas only** |
| `session.snapshot_state()` → `PlantState` | canonical state at the cursor (explicit host copy); accepted aboveground allometry for consumers = `leaf.canopy.canopy_view(session.snapshot_state(), catalog, current_surface, **kw)` |
| `session.pending()` / `live_ids()` / `counters()` / `boundaries()` / `due_boundaries()` / `phase2_state()` / `phase3_state()` / `stress_counters()` / `annual_accumulators()` / `needs_flush()` / `flush_required(next_growth_enabled=, next_is_transport=)` | read-only views (explicit host copies where noted); no private field is needed |
| `session.memory_ledger()` · `resident_bytes()` · `candidate_bytes()` | structural byte estimates (not measurements) for comparison with measured peaks |
| attributes | `token`, `sequence`, `lineage`, `cursor`, `n_live`, `geometry_sha256`, `pop_sha256`, `last_stage_id` (`StageId.to_dict()`), `resource_cursor`, `day_binding`, `config` |
| `lb.bind_source(evidence_sha256=None)` → `BridgeSource` | once per run: cold tree-digest check of the imported SYRUP package |
| `lb.read_availability(src, lop, parent, *, max_bytes, reserve_bytes=0, extra_reserve_bytes=0)` → `AvailabilityRead` | `.supply_m3` `[L,N]` m³ = native depth × `cell_area_m2`, `.identity` (`ResourceIdentity`), live parent capability |
| `lb.request_args(read, lop, parent)` → `{'expected_parent': …}` | pass as `**args` to `wl.request_withdrawal` |
| `lb.request_depth_m(read, proposed_m3, *, max_bytes, reserve_bytes=0)` | `[L,N]` depth m = volume / the SAME area |
| `lb.accepted_withdrawal(read, identity, new_state, candidate, *, lop, max_bytes, reserve_bytes=0)` → `AcceptedWithdrawal` | `.actual_m3` / `.unmet_m3` `[L,N]`, `.before` / `.after` identities, totals |
| `lb.resource_identity(lop, parent)` → `ResourceIdentity` | scalars incl. run, LAST consumed ordinal, pending flag (validates the operator association) |
| `pc.save_biology(directory, session, *, external, limits, pending=None, provenance=None, syrup_source=None)` → manifest `dict` | new directory only; `external` = `ResourceIdentity` of the exact SYRUP reference paired with this biology |
| `pc.load_biology(directory, config, *, bindings, limits, day_binding_check=True)` → `Restored(session, pending, manifest)` | `bindings=ExternalBindings(external, expected_stage_id=None, domain=None, source_override=None)`, `limits=CheckpointLimits(max_file_bytes, current_resident_bytes=0)` |
| `pc.rebuild_pending(session, restored.pending)` → `(plan, proposal, accepted)` | then `session.stage_day(proposal, accepted)` + `publish` — no new withdrawal |

## 3. The visible daily call sequence (as in `examples/masy_daily_plants.py`)
```python
src = lb.bind_source(profile.syrup_package_digest)                                   # once
state = dw.start_daily_lineage(lop, boundary, expected_parent=wl.expect_parent(boundary), run_digest=RUN)
session = ps.PlantSession.start(plants, cfg, anchor=midnight, day_binding=ps.DayBinding(RUN, origin_date),
                                initial_resource=lb.resource_identity(lop, state))
for each nominal day:
    # recomputed EVERY day: population, geometry and pending arrays change with growth, births and deaths
    reserve = session.config.reserve_bytes + session.resident_bytes()                # what the caller keeps live besides the call
    # 1. MASY completes the day's wind / precipitation treatment (a storm branch: dw.begin_daily_storm / wl.evolve_storm … / dw.finish_daily_storm)
    plan_s = dw.next_plan(state, p_mm, i_mm_h, t_mm)
    day = dw.daily_water_action(lop, state, expected_parent=wl.expect_parent(state), plan=plan_s, pet_terrain_m_s=pet_m_s).state   # ONE hydrology op
    read = lb.read_availability(src, lop, day, max_bytes=cap, reserve_bytes=reserve)  # pure read, zero time
    plan = session.prepare_daily(ps.DayContext(start, T_c, True, growth_on, duration=enabled), terrain_pet=lb.pet_m_s_to_mm_per_day(pet_m_s))
    prop = session.propose_withdrawal(plan, read)
    new_day, cand = wl.request_withdrawal(lop, day, **lb.request_args(read, lop, day),                                    # ZERO time, T count + 1
                                          depth_m=lb.request_depth_m(read, prop.proposed_m3, max_bytes=cap, reserve_bytes=reserve),
                                          identity=prop.identity, retained_feedback_bytes=prop.retained_feedback_bytes)
    acc = lb.accepted_withdrawal(read, prop.identity, new_day, cand, lop=lop, max_bytes=cap, reserve_bytes=reserve)
    staged = session.stage_day(prop, acc)
    state = wl.close_interevent(lop, new_day, expected_parent=wl.expect_parent(new_day)); session = session.publish(staged)
```
`cap` is the global byte limit (`cfg.max_bytes`), `reserve` what the caller keeps live besides the call. **Passing the same cap is not accounting by itself:** the caller-held physical / native state
(SYRUP lifecycle states, availability and feedback arrays) must be included in LEAF's `SessionConfig.reserve_bytes`, and LEAF's bytes (the value of `session.resident_bytes()`) must be given to SYRUP through its own
preparation reserves (`extra_reserve_bytes`, host / device reserves of the lifecycle factories) so that each side's ledger counts the other's live memory. A nominal day may be split into several fragments (e.g. 6 h wind then 18 h open:
`stage_transport(TransportDay(midnight, 'wind', 6 h))`, then `DayContext(midnight + 6 h)`): only enabled time earns demand, integrals and credit; ages follow the FULL nominal calendar; pending growth is flushed first
(`flush_required`, `stage_flush`). Later fragments of the same SYRUP occurrence read availability from the post-T state (same run, ordinal, hydrology count and T counts) with **no** second hydrology step.
Transpiration on / growth off debits real water and earns no credit; T off / G on is refused (qualified C1 rule). Never call `syrup_water.withdraw_plants` or the native standalone withdrawal.

## 4. Identity and anti-replay rules
* `session.token` binds session id, per-start random lineage, sequence, calendar, counters, Phase 2/3 state, resource cursor, last stage, population / geometry content hashes and the configuration identity. Plans carry a nonce and are consumed at publication.
* **Mandatory connection.** A proposal needs `DayBinding` + initial resource; the read must be of a completed daily lineage of that run, with the ordinal that the nominal date maps to (`origin_date + ordinal`), a non-decreasing resource lineage, and
  **exactly equal** `request_sequence` and `n_requests` to LEAF's consumed cursor (an unseen physical T is refused; hydrology / close / storm states with the same T counts are fine).
* `read.identity` / `request_args` / `accepted_withdrawal` bind the exact parent instance, the candidate (`new_state.payload is candidate.state`), the unchanged native leaves, receipt echo, zero time and one T.

## 5. Transaction crash windows and replay prevention
| Crash after | LEAF accepted state | SYRUP state | MASY recovery |
|---|---|---|---|
| SYRUP daily action, before `request_withdrawal` | unchanged | day state (ordinal advanced) | restore the paired SYRUP state; re-read availability; continue — nothing to replay in LEAF |
| `request_withdrawal` returned (debit applied), before LEAF `publish` | parent without the day | `new_day` (T count + 1) | MASY saved `PendingStage(plan, proposal, accepted)` with `external = accepted.after`; restart: SYRUP restored to exactly that state → `load_biology` → `rebuild_pending` → `stage_day` → `publish`. **No second debit.** A SYRUP state before the debit refuses the resume |
| LEAF `publish`, before the combined commit | new session | `new_day` / closed boundary | compare MASY's recorded `StageId` (`staged.stage_id.to_dict()`) with the checkpoint's `last_stage_id` (`ExternalBindings(expected_stage_id=…)`): equal = already published, do not stage again |
| SYRUP checkpoint written, LEAF checkpoint not (or vice versa) | any | any | `load_biology` requires `external` to EQUAL the saved reference; an older / newer / foreign state is refused, so restore both halves from ONE MASY commit |
A closed boundary after the T is a valid external reference for a no-pending checkpoint; the last consumed T cursor is stored separately inside the session.

## 6. Memory and transfers
Admission is metadata-first: `start` / restore refuse before any population-sized mask or copy; each stage sums reserve + accepted parent (arrays, stress / annual arrays, tables, base, root + canopy maps, retained candidate `PlantState`) + a coexisting candidate +
component estimates; the root and canopy builders share the global `max_bytes` (canopy cap = `min(canopy max_bytes, max_bytes)` with the live reserve; this is the only memory-accounting divergence from the adopted C1 `biocalendar`; no kernel changed). No dense P×N or P×L×N tensor.
Cold, explicit population-sized transfers: weekly / Phase 2-3 rebuild (B, E to host, CSR back), `snapshot_state`, `pending`, hashes, checkpoint. `memory_ledger()` reports estimates; measured CPU / CUDA peaks are independent evidence, not author claims.

## 7. Checkpoint contents and restart configuration identity
`plants.csv` (plain nine columns, all rows) + `plants.metadata.json`, `biology_arrays.npz`, `manifest.json` (codec `leaf.plant_biology_checkpoint` v2; the standalone `leaf.checkpoint` layout 4 and all locks are untouched). Saved: pending D / U / A / T,
stress + annual arrays, all Phase 2/3 counters, day binding, lineage, identities, LEAF source sha256, resource cursor and, separately, the exact external reference, plus the **declared biology configuration** (`config` + `config_sha256`): catalog hash, cadence,
the ACTUAL `rng_seed`, `patch_size_m`, every root layer override (also for currently non-live plants) and the canopy mask hash. The caller must pass the same values to `load_biology`; any difference is refused and a `SourceOverride` never bypasses it. Byte / edge ceilings
are caller-adjustable on restore; the backend is fixed by the compiled table the caller builds (no cross-backend claim beyond the geometry / state identity checks).

## 7b. Periodic boundaries (common boundary config; revision `periodic_boundaries_20261112`)
The plant `Domain` carries a per-axis declaration `periodic=(px, py)` (default `(False, False)` = the finite closed domain used so far; see `periodic-boundaries.md`).
MASY supplies the SAME declaration to LEAF (the `Domain` of the base `PlantState`) and to the rest of the stack; nothing else changes in the call sequence of §3:
* every geometry build of the session — `start`, weekly / year-end growth rebuilds, Phase 2/3 rebinds (births, deaths), restore — reads the flags from the state's `Domain`
  through the shared builders, so roots (uniform unique-cell law, layer fractions unchanged) and canopy-area PET weights (the `CanopyMap` areas / denominator are the torus union) wrap across
  periodic seams, while `canopy_area_m2` (the demand area) stays the nominal allometric `pi r^2` for every boundary and radius, exactly as in the standalone session;
  `session.boundary()` and, for a periodic domain, the `boundary` key of `export_roots()` / `export_canopy()` report the declaration;
* one plant = one record = one ID (no image rows); `canopy_view(session.snapshot_state(), ...)` carries `meta['boundary']` and one object per plant (periodic public centers are half-open
  `[0, L)`: a stored exact upper endpoint is exported as 0, the stored record is unchanged); consumers wrap the footprint themselves. Pass and CHECK the same boundary flags as MAPLE / SYRUP (MAPLE's
  `leaf_domain` import helper must carry them); the exact upper endpoint is the lower edge for sampling, roots and PET;
* water is untouched: SYRUP stays the sole inventory; roots of different plants may overlap across a seam, allocation stays proportional with an equal fraction of requests, no species
  priority and no compensation; the accepted summed uptake drives growth; the native debit stays visible at your call site and authoritative exactly once;
* `pop_sha256` and `geometry_sha256` bind the flags; the checkpoint manifest records `boundary` for a periodic domain and `load_biology` refuses a plant file whose flags differ (an old
  manifest without `boundary` is nonperiodic and is never reinterpreted as periodic); `leaf.periodic` is in `plant_checkpoint.SOURCE_MODULES`, so a checkpoint written before this revision
  needs an explicit `SourceOverride` naming the differing modules (it never relaxes the format, water identity, catalog, geometry or population checks);
* a periodic canopy that overlaps its own image (diameter above a periodic span) keeps working in the session (PET weights = union, demand area = nominal); a no-overlap GENERATION refuses such a canopy explicitly;
* `plant_pet` also carries a numerical correction of the shared exact circle-rectangle integrator for a rectangle side exactly tangent to the circle (rare legacy output change; see `periodic-boundaries.md` §3.3).

## 8. Remaining limits
`load_biology(..., day_binding_check=True)` keeps an existing keyword that currently has no behavioral effect (the saved binding is always restored and verified through the token); do not rely on it.
T off / G on stays refused. Periodic builders require square cells (like the finite ones); the boundary configuration is JSON / `Domain` only (no CLI flag). Geometry restore assumes radii do not depend on age beyond the state (hash-checked). Phase 2/3 laws are the existing uncalibrated ones; they are exposed, not validated. The interface is qualified (by Codex) only on small fixtures.
