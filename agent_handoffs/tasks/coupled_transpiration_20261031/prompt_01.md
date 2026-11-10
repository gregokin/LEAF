# Bounded implementation: native soil-water transpiration and vegetation coupling

You are the sole implementation author. Work ONLY in the supplied isolated candidate directory, with Read/Edit/Write tools. No shell, other files, network tools, private context, external model source, installs, Git actions or permission bypass. Do not inspect parent directories. The candidate contains ONLY generic original benchmark code and its tests. The numerical API contract below is supplied instead of any upstream scientific source. Codex owns actual context, baselines, installation, independent testing/review and task acceptance. Do not claim tests ran: you have no execution tool. Preserve existing numerical defaults and tests. Return a concise report with files, contracts, limitations and unexecuted checks when done.

Allowed writes exactly:
- benchmarks/daily_water_growth.py
- leaf/__init__.py
- leaf/coupling.py
- tests/test_coupling.py
- native/maple_syrup/subsurface_transpiration.py
- native/test_transpiration.py
- docs/coupled-transpiration.md
Read benchmark source/tests as needed. Never edit spatial_mapping.py or original tests. Write compact, practical code. Reuse geometry, plant initialization, demand scatter and weekly growth. No hydrology physics copying or synthetic water in coupled mode.

## Goal and ownership

Connect the existing individual-plant growth model to a separately implemented native layered column. The native column owns all physical water. Vegetation owns plants, roots, potential demand and biomass, then grows from ACTUAL native transpiration. Add an opt-in native withdrawal module plus a real coupled daily driver. Keep the old standalone benchmark unchanged by default. Do not just implement an unattached interface or mock: the orchestrator will run the actual column and coupled model on CPU and CUDA.

## Actual native API contract (use these names)

Installed package `maple_syrup`; module `maple_syrup.subsurface` (import lazily in vegetation coupling). Its first column milestone is fixed geometry/common properties; extension module will be installed alongside it.

`SoilLayers` frozen host config has n_layers, thickness_m[L], residual_m[L], saturation_m[L], capacity_tolerance_m[L], theta_residual, etc. `SoilColumn` frozen eq=False dataclass: layers, backend ('numpy','numba','cuda'), xp (numpy/CuPy), n_layers, n_cells, n_active, cell_area_m2 positive scalar, active[N] bool, active_u8[N] uint8, thickness_xp[L], device_id (None CPU else current bound id), reserve_bytes, available_bytes_fn()->integer available bytes, memory, preflight (callable or None). Columns own copies of masks and parameters.
`SoilColumnState(water_m, t_s)` frozen eq=False dataclass: FP64 C-contiguous [L,N] water-equivalent DEPTH in m; finite time in seconds. Inputs immutable by contract; functions return fresh state.
Public functions in subsurface:
- soil_layers(depth_edges_m, *, theta_residual, theta_field_capacity, theta_saturation, diffusivity_m2_s, theta_evaporation_stress, evaporation_factor=..., gravity_drainage=...)
- prepare_column(layers, active_mask, cell_area_m2, *, backend='numpy', reserve_bytes=0, available_bytes_fn=None)
- uniform_state(column, theta_by_layer, t_s=0); column_state(column, water_m, t_s=0) (makes copy); moisture(column,state); storage_m3(column,state)
- step(column,state,dt_s,*,added_water_m[L,N],potential_evaporation_m_s[N]) -> SubsurfaceStep
- estimate_step_memory(L,N,*,backend,reserve_bytes=0) -> memory dataclass with required_bytes (admitted host for CPU/device for CUDA + reserve), host_bytes,device_bytes,total_bytes, input_bytes,column_bytes,layer_bytes,parameter_bytes,output_bytes,scratch_bytes,step_allocation_bytes
- SubsurfaceError(ValueError), SubsurfaceMemoryError(MemoryError), SubsurfaceBackendUnavailableError(RuntimeError).
Reusable validation helpers (pinned current version): _check_column(column), _check_state(column,state)->finite t_s, _check_dynamic(column,name,array,shape) (exact numpy/cupy ndarray, no subclass, FP64, contiguous, expected shape and CUDA device), _check_dt(dt_s)->finite nonnegative float.
Native conservation helper: `from maple_syrup.conservation import volume_roundoff_bound_m3`; signature `(n_terms:positive int, scale_m3:finite nonnegative)->float`, derived from MAPLE FP64 summation policy. No arbitrary new conservation epsilon.
GPU host transfer helper `from maple.core.backend import to_host`; call once on a stacked small packet to validate/report reductions, never per-layer/full-field downloads.

`SubsurfaceStep` fields: state, dt_s, surface_return_m[N],drainage_m[N],evaporation_m[N],evaporation_unmet_m[N],percolation_m[L-1,N],diffusion_m[L-1,N],budget.
`StepBudget` fields: storage_before_m3,storage_after_m3,added_m3,surface_return_m3,drainage_m3,evaporation_m3,evaporation_unmet_m3,residual_m3,bound_m3,n_terms.
`step` is pure; on rejection inputs unchanged. It validates metadata and time, admits memory before output allocation, runs compiled NumPy/Numba/CUDA column physics, then ONE host reduction packet. Order: actual accepted intake (positive amounts only) -> capacity surface return -> gravity drainage -> soil evaporation -> adjacent-layer diffusion -> closure. It alone advances physical t_s += dt. dt=0 identity, refuses positive intake. Added forcing and evaporation are zero on inactive cells. Antecedent below residual is valid. Negative/NaN/above-saturation water is invalid (capacity_tolerance_m allows roundoff only). Current step has NO transpiration parameter or sink; do not misuse added water or evaporation for T.

## New native module

Implement explicit prepared operator, e.g. `prepare_transpiration(column)`, holding native column and small per-layer parameter copies. Prepare/compile outside daily work. NumPy reference, compiled Numba and CUDA RawKernel (no Python plant/cell loops). No repeated daily layer parameter uploads, no silent CPU fallback, no fixed layer maximum. CUDA keep arrays on the column's device and check current device/preflight before work.

`withdraw_transpiration(operator,state,requested_water_m[L,N])`:
- Native-only physical operation. Requested water is DEPTH over an already defined interval, finite >=0 and zero inactive. Preserve time (do not advance it again).
- Available in each active layer/cell = max(w - layer.residual_m, 0); actual=min(requested,available); output water=w-actual; unmet=requested-actual; scale=actual/requested (0 for zero demand). All nonnegative, finite and bounded. Water initially below residual remains unchanged. No wetter-layer compensation or invented species wilting law.
- Return fresh authoritative state plus actual_m/unmet_m/scale[L,N] and explicit volume budget (area multiplication exactly once). Input arrays untouched. Preserve inactive water exactly. Validate invalid state/forcing on all cells, including inactive; reject inactive positive requests. Capacity checks use supplied tolerances. Do not clip water after subtraction to hide closure errors.
- Preserve water-removed = actual-T and requested=actual+unmet within MAPLE-derived bounds. GPU validation/reporting in a single small packet; no per-layer/full-field reads.
- A scalar memory admission before EVERY bulk allocation including prepare parameters, work/results and any compiled warm-up. Report host/device/peak estimate and reserve. Reject unavailable memory without mutation; do not raise budgets.

`step_with_transpiration(operator,state,dt_s,*,added_water_m,potential_evaporation_m_s,requested_transpiration_m)`:
- Reuse existing native `step`, then withdrawal, documented first-order hydro->T splitting. dt=0 requires zero requested T. Hydro advances time once; T preserves it.
- Return final state and native hydro diagnostics plus actual/unmet/scale and an overall budget: after-before = added-return-drainage-evaporation-transpiration, distinct E and T. Do not publish intermediate water as the authoritative final state. Existing standalone native API and kernels unchanged.
- Count ORIGINAL caller soil state, requested forcing, still-live hydro diagnostics/candidate, T outputs/parameters/scratch and caller reserves in combined admission; do not count only the T operator after ignoring hydro outputs. Keep no histories, release obsolete diagnostics between days. Don't mutate the original column/reserve.

## Existing vegetation reuse and coupled driver

1. Add explicit keyword `vegetation_only=False` and optional external reserve to existing `Simulation`/ledgers as necessary. Defaults must preserve all standalone numerical behavior/tests. In vegetation-only mode SKIP W, scale, removed_cum, overflow_cum, W0, cap/recharge initialization and synthetic water entirely. Retain R cache, trait arrays, roots, weekly uptake/growth. Synthetic step_days/run_case water path must refuse this mode. Provide mode-aware live/array/memory reporting and a plant-only summary or internal helpers; no dummy zero water budget. No copying entire Simulation into a second class.
2. Extract/reuse a compiled CPU gather-only helper matching existing GPU gather; it computes a fresh per-plant actual uptake candidate from native scale, Dp, CSR and species fractions. Don't invoke old monolithic _block_cpu in coupled mode. GPU gather kernel already exists; reuse it or factor it. Additional one-dimensional P candidate buffers allowed and counted; no P*N or H*L tensors.
3. `leaf.coupling.CoupledSimulation(case,column,state,inputs=None)` composes a water-free vegetation Simulation and prepared native operator. Public `.state`/`.soil_state` is the current native SoilColumnState (one physical water authority). Expose vegetation state and roots to callers. Strict N=nx*ny, L=case layers, cell_area=dx² (roundoff-aware equal), flat order row-major j*nx+i, namespace/backend/device alignment. No implicit copies/transfers/resampling. Root profile rows correspond to actual native layers.
4. Supply explicit forcing per day, e.g. `.step_day(added_water_m, potential_evaporation_m_s, *, phase='interevent')` and `.step_days(records, *, phase='interevent')`, records iterable of pairs/native ForcingRecord with dt=86400. DAILY-ONLY interface is acceptable and preferable to ambiguous partial-day weekly calendar. Native state t_s can start at any finite time; own days count intervals since construction, advancing t_s by exactly86400. No synthetic recharge if forcings absent (require explicit forcing rather than guess). Refuse wind/rain/water event phases before work.
5. R is volume m³/day. Each daily request depth = R / area (dt=1 day); actual per-plant volume gathers Dp*w*species_frac*scale (the same piece originally requested). Do not multiply by area twice or allocate a LEAF soil store. Native actual depths * area must equal per-plant volume. Check this and finite nonnegative candidates BEFORE publishing state/counters/plant accumulations. Allocate candidates and admit coupled peak before native work where possible; a forced native/gather/admission error must leave prior accepted soil state, plant accumulators and counters unchanged.
6. Active mask: zero requests on inactive cells, do NOT renormalize horizontal weights silently. Inactive roots' potential demand remains explicitly unmet/unmapped; keep all individuals. Gather zero scale there. Report full potential, mapped demand, native actual, plant actual and inactive/unmapped/unmet separately. No false sum(R)==sum(Dp) check in masked coupled mode.
7. After accepted daily uptake update uptake_week/uptake_cum/day, use existing transactional weekly biomass/radius/CSR/demand refresh every 7 days. All map admission includes native soil/op/state and retained diagnostics reserves. On weekly rebuild refusal, preserve the accepted physical day/uptake and existing geometry with a latched pending weekly boundary, matching prior explicit retry semantics; expose retry, reject further daily stepping until resolved. Document this exception separately from a rejected daily native step (no partial day).
8. Expose combined summary with cumulative added/return/drainage/E/T, native initial/current storage, plant/grid flux closure, biomass budgets and calendar (native time vs accepted days), profile/mask/peak metadata. Stream scalars, no trajectories. Split-run stepping on same state equivalent to uninterrupted run (serialization not requested). Do not infer storm/MASY integration, changing texture or production ecology implemented.
9. Honor original case max_edges/max_bytes and native available/reserve. Admission must account simultaneously live soil, forcing, native outputs, plant arrays/candidates, old/new graphs and scratch; cheap preflight before allocations, grow reserve updates as graphs change. A reading is not reservation/pool bound. Report memory scopes accurately. All hot paths compiled, no daily full-state host/device transfer.

## Required meaningful tests

Add focused tests using actual imported native API (the orchestrator sets PYTHONPATH; don't hard-code private paths). Native isolated tests with parameters constructed through soil_layers/prepare_column: analytic finite/residual-limited uptake; already-below-residual; dry/zero demand; inactive unchanged and invalid inactive forcing; input/state immutability and validation; zero-time combined, no double time; combined E+D+T+intake budgets; memory refusal before allocation; NumPy/Numba/CUDA agreement, real GPU tests explicitly skipped if unavailable.
Coupled tests: two competing individuals with different fractions; dry-top vs dry-bottom; nonunit area0.0625m² AND1m² to catch depth-volume conversion; actual uptake (not demand) drives weekly growth; no own W or synthetic recharge in vegetation mode; weekly/fixed profiles and root rebuilds; daily phase exclusion; partial/refused native/gather day rollback; weekly refusal latch/retry; masks/unmapped demand; split-run equivalence; grid/layer/namespace/device mismatch; CPU/GPU real parity. Keep existing tests unchanged. No fake water budget updates or tolerance loosening.

Document API/configuration/order, units and usage with native package imports, existing environments, branch dependency and limitations; don't invent test outcomes or timings. Codex independently executes checks and measurements then requests bounded corrections if necessary. Leave all files uncommitted.
