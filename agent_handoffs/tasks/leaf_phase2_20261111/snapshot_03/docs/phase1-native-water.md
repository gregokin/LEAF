# Phase 1 — bounded native water bridge (`leaf.syrup_water`)

Status: revision 2 of an independent Phase 1 subtask (not Phase 1 completion). Written with file tools only;
nothing here was executed by its author. Codex runs, reviews, integrates and publishes.

`leaf.syrup_water` is a small PURE bridge between the LEAF sparse uptake allocator (`leaf.uptake`) and the
native SYRUP layered soil column (`maple_syrup.subsurface`, `maple_syrup.subsurface_transpiration`). SYRUP
remains the only owner of physical water. LEAF holds no soil store, no configurable wilting floor, and does not
subtract water anywhere.

## Public API

```python
from leaf import syrup_water as sw

result = sw.withdraw_plants(root_map, demand_m3, operator, soil_state, *, max_bytes, extra_reserve_bytes=0)
snap   = sw.availability_snapshot(operator, soil_state, *, max_bytes, extra_reserve_bytes=0)
ledger = sw.estimate_bridge_bytes(n_plants, n_edges, n_layers, n_cells, *, map_backend, native_backend, ...)
sw.estimate_availability_bytes(...); sw.check_binding(...); sw.check_operator(...)
sw.native_status(); sw.native_provenance(verify=False)
```

* `root_map` — `leaf.uptake.RootMap` (resources `layer * n_cells + cell`, fixed normalised weights).
* `demand_m3` — typed array `[P]`, m3 for the interval the caller already fixed (nothing here scales a rate).
  `numpy.ndarray` for host columns, the same-device `cupy.ndarray` for CUDA columns.
* `operator` — an EXISTING prepared `TranspirationOperator` (from `prepare_transpiration`). The bridge never
  prepares / compiles one (no hidden setup cost) and refuses anything else.
* `soil_state` — native `SoilColumnState` (FP64 `water_m[L, N]` depth in m, finite `t_s`).
* `max_bytes` — required explicit integer budget for the whole call (see Memory).

`PlantWithdrawal` (frozen) holds: `native_state` (the FRESH native inventory, `t_s` preserved), `native` (the
native `TranspirationResult`: `actual_m`, `unmet_m`, `scale`, `budget`, `memory`), `plan` (`UptakePlan`),
`feedback` (`UptakeFeedback`: `delivered`, `unmet`, `satisfaction`, `accepted`, `closure`), `availability`
(`AvailabilitySnapshot`), `request_depth_m`, `accepted_volume_m3`, `conservation`, `memory`, `binding`.
Convenience: `delivered_m3`, `unmet_m3`, `satisfaction`, `t_s`.

Units: availability / proposed / accepted / delivered / unmet are m3; native depths are m; the column cell area
is the scalar `column.cell_area_m2` (m2); time is the native `t_s` (s), preserved.

## What one call does (in order)

1. Validation with NO allocation and NO native call: types, `[L, N]` binding of map / operator / column / state,
   exact array namespace and dtype (`float64`, C-contiguous, no subclass), CUDA device affinity (map, demand,
   column, operator, state must all be the same CuPy module and device), operator-to-column binding,
   `check_device()`, pinned native `VERSION`.
2. Memory ledger and admission (below) against `max_bytes`, the root map's own declared `max_bytes` (the
   allocator's peak) and the native column's `available_bytes_fn`.
3. Availability projection from the native inventory (below), including native validity of every cell.
4. `leaf.uptake.propose(root_map, demand_m3, A)`: `Q = sum(D*w)`, `proposed = min(A, Q)`, shared fraction, fixed
   roots, no compensation, no species priority.
5. `request_depth = proposed / cell_area` (once), then ONE call of the actual
   `maple_syrup.subsurface_transpiration.withdraw_transpiration(operator, state, request_depth,
   extra_reserve_bytes=...)`. The native API alone creates the new inventory.
6. Strict validation of the native result (types, shapes, device, fresh non-aliased arrays, preserved finite
   time, finite non-negative budget closing in its own FP64 bound), then `accepted_volume = actual_m * area`
   (once) and `leaf.uptake.accept(plan, accepted_volume)`. Plants gather from the ACTUAL accepted water; the
   accepted vector is passed through unchanged, so honest area-conversion roundoff is preserved (no clipping,
   no balancing flux, no second native acceptance).
6b. Physical debit verification (revision 3; read-only, between steps 5 and 6, before the gather). Reported budget
   scalars alone are not trusted. One fused pass (`_k_verify`: plain definition compiled by Numba, mirrored in
   CUDA C, vectorised per layer for NumPy) compares the SUPPLIED water, the RETURNED water and the native
   `actual_m` / `unmet_m` / `scale` / request arrays cell by cell with the pinned FP64 policy
   (`|(before - after) - actual| <= 4 eps before`; unmet within `8 eps request`): returned water finite,
   `>= 0`, `<= saturation bound`; actual finite, `>= 0`, `<= max(before - residual, 0)` and `<= request`;
   inactive cells bit-unchanged with zero debit. It also accumulates active-cell depth totals into ONE fixed
   14-double packet (read back as 112 bytes on CUDA; no `P` / `R` / grid read-back, no library reducer).
   Missing debit (fresh copy of the original water), extra or unaccounted loss, a debit on the wrong cell /
   layer, changed inactive storage and non-finite / negative / over-saturated water are rejected. `_conservation`
   then requires the native budget's storage-before / storage-after / actual / unmet / requested totals to
   equal the actual array totals and `sum(before) - sum(after)` to equal the accepted volume
   (`conservation['physical_debit']`). A genuine shortfall (native debited less than requested, on the original
   input) and honest area-conversion roundoff pass; the lower bound `actual = min(request, availability)` is
   deliberately NOT enforced. This is validation only: no subtraction, no water store, no extra native call.
7. Conservation checks (all enforced; a violation is an implementation failure and nothing is returned):
   native requested volume equals the proposal; native actual does not exceed the request or the availability;
   the plan's supply total equals the snapshot; accepted and delivered totals equal the native actual volume
   within the native bound + the area-product bound + the allocator FP64 policy; allocator closure closed;
   `delivered + unmet = demand` (up to documented clamped over-delivery). `result.conservation` itemises the
   tolerance terms, the per-check booleans and the area-conversion roundoff.

## Native source binding and the availability projection

Pinned SYRUP HEAD `c76c603dde4e8f79ec48b18d954a386580d8bc89` (`feature/subsurface-hydrology`),
`subsurface_transpiration.VERSION == '0.2.0'`; source SHA-256:
`subsurface.py` `5e77f14df8af4e473e8a22278700b519ccb2888c722883a9f7da508abdf14ae9`,
`subsurface_transpiration.py` `3b9191872522755924271bbaba4ae2bac290e1cad4ecffebc2d2735c4a0a9fb9`.
`withdraw_plants` refuses any other module VERSION; hashes are checked by `native_provenance(verify=True)` at
initialisation / run-evidence time, never per call (no rereading of sources, no per-day provenance work).

Projection (a read-only view of SYRUP-owned state and floor, NOT a water store and NOT a LEAF wilting floor):

```
A[l, c] = cell_area_m2 * max(water_m[l, c] - operator.residual_xp[l], 0)   for active cells (column.active_u8[c] != 0)
A[l, c] = 0                                                                 for inactive cells
```

This is exactly the native withdrawal law of the pinned version (`available = max(w - residual_m[l], 0)`,
inactive cells carry zero requests). The same pass checks native validity of every cell (finite, `>= 0`,
`<= saturation_m + capacity_tolerance_m`, inactive cells included) without a dummy native withdrawal, and
refuses before any proposal or native call. Implementations: NumPy reference (per-layer `[N]` temporaries, no
`[L, N]` temporaries), compiled Numba CPU kernel (`_k_project`, the authoritative per-cell definition), and a
CuPy `RawModule` kernel that writes `A` and reduces into a 2-double packet (no library reduction).

If SYRUP changes its withdrawal law, the pin and this projection must be re-verified together; there is no
configurable floor to adjust.

## Ownership and publication boundary

`withdraw_plants` is pure. It never writes the input state, root map or demand (also not on failure), returns the
fresh native state as a value, does not publish either inventory or any plant state, does not advance a
scheduler clock and binds no interval receipt. It is NOT a restart, idempotency or event-scheduler mechanism.
The later dated driver must publish native + biological state atomically and bind an exactly-once interval
receipt (and a combined restart format); calling the bridge twice with the same inputs returns the same fresh
state twice and changes nothing.

## Namespaces

CPU NumPy and Numba maps/columns are compatible (either way round). CUDA needs the exact same CuPy module and
device throughout. No implicit fallback, no host/device population read-back (only 16- to ~230-byte scalar
packets), no device switch, no dense `P x L x N` arrays. Importing `leaf.syrup_water` imports only NumPy and
`leaf.uptake`; `maple_syrup`, Numba and CuPy load lazily when the bridge is used.

## Memory scope (revision 2)

All sizes are scalar arithmetic and are admitted BEFORE any allocation, transfer or native call:

* `estimate_bridge_bytes(...)` is a conservative stage-sum ledger (`total_bytes = host + device + reserves`).
  Entries are labelled with a side (`host` / `device` / `reserve`) and whether they are `held_before_call`
  (root map, demand, original native state, native operator parameters, native column mask / thickness copies,
  `SoilLayers` vectors, `KernelParameters` host AND device copies) or fresh (availability buffer + projection
  scratch / packet, allocator plan owned + scratch, request-depth buffer, native outputs / rows / scratch /
  packets, accepted-volume buffer, allocator feedback owned + scratch, allocator host packet reads).
* Native components come from the native authoritative estimates (`estimate_step_memory`:
  `column_bytes`, `layer_bytes`, `parameter_bytes`; `estimate_transpiration_memory`: parameters / state / input /
  output / rows / scratch / packets). Allocator components come from `leaf.uptake.estimate_plan_bytes` and
  `estimate_feedback_bytes`. Nothing is double counted: the request-depth buffer is the native input array.
* `fresh_host_bytes` / `fresh_device_bytes` are the entries allocated by the call; the rest is already held.
* The native withdrawal receives `native_extra_reserve_bytes` so its own admission sees exactly the bridge's
  native-side requirement (`native_required_bytes`), which is also what is compared with `available_bytes_fn`
  first. `allocator_peak_bytes` is checked against `root_map.max_bytes` so the allocator cannot refuse after the
  native call.
* Reductions: the bridge performs NO population- or grid-sized library reductions on the device. Scalar totals
  come from fixed-size packets filled by the allocator kernels or by the projection kernel. The earlier revision
  reduced `delivered`, `unmet`, `demand` (size `P`) with CuPy `sum`/`stack`, whose hidden workspace (probably
  CUB temporary storage; the exact cause was not measured by the author) escaped the ledger and is the most
  likely source of the reported 8,821-byte omission at P=200000, N=1, L=2. That path is gone; the author did
  not run the hooked profile, Codex must.
* Debit verification (revision 3) is in the ledger before allocation: `verify_scratch` (NumPy: 3 float64 + 8
  bool `[N]` equivalents, conservative, plus the packet), `verify_packet` (Numba: 112 host bytes),
  `verify_packet_device` / `verify_packet_host` (CUDA: 112 bytes each). No grid-sized scratch on the compiled
  paths.
* Peak policy: stage buffers freed before the next stage are not reclaimed in the total (labelled upper bound).
* EXCLUDED (declared in `MEMORY_SCOPE`): NumPy/CuPy allocator pools and allocation-size rounding, NumPy ufunc
  casting buffers, the CUDA context, Numba/NVRTC JIT caches and compile workspaces, Python object overhead, the
  native hydro-step arrays (not used here), other model state, and the rest of the process. A reading is not a
  reservation; bounds are never raised and plants are never discarded.

## Tests

`tests/test_syrup_water.py` uses the REAL native API (no fake water inventory). Spies only wrap the real
`withdraw_transpiration`. Covered: the user 75 % case with leftover deep water, proportional competition,
shallow/deep and dry-layer no-compensation, inactive cells without renormalisation, non-unit areas with exact
accepted-vector feedback, random overlaps (finite / non-negative / unit closure, purity), an empty population,
shortfall feedback (real operator called once on the ORIGINAL water with a reduced request; removed volume
equals delivery), physical debit rejection (fresh unchanged water, extra loss, wrong-cell debit, changed inactive
cells, malformed returned arrays, inconsistent budget totals; CPU and CUDA), malformed-native-result rejection, binding / dimension / backend refusals before any
allocation or native call, byte-budget and `available_bytes_fn` refusals at exact bounds, failure atomicity,
ledger scope (native kernel-parameter copies, no population term outside map / demand / allocator),
tracemalloc and CUDA memory-hook regressions, core import isolation, and CUDA parity (skipped only if CUDA is
unavailable). No test allocates on a second device.

## Remaining work (not in this subtask)

Dated growth coupling and the daily driver; atomic publication of native + biological state with an
exactly-once interval receipt and combined restart; weather / PET and temperature laws; mortality, recruitment
and root-map rebuilds; surface remapping. Per-call provenance stays an initialisation concern.
