# LEAF Phase 1, bounded part A — water contract, roots, scheduler records, unified species loading

You are Claude, sole candidate implementation writer. Codex orchestrates, reviews,
executes independent tests, installs, commits/pushes. User explicitly authorized
Phase 1–3, automatic collaboration and relevant sharing with Anthropic/Claude.
This part builds independently testable Phase 1 infrastructure while Codex resolves
a material growth-law interpretation with the user. Do not implement growth yet.

## Governing instructions / scope

May READ only this prompt and candidate/**. Candidate contains existing LEAF core,
tests, two compiled numerical benchmark modules, Phase 0 example fixtures, the
growth/shared-view guides and pyproject.toml. No other repositories, task archives,
briefs, raw studies, credentials/settings or arbitrary external context.
Tools: ordinary Read/Edit/Write only. No Bash, Python execution, agents, network
tools, Git, installs, package changes, upstream writes, or permissions bypass.
Report all tests as NOT RUN. On genuine tool/auth/network/rate failures preserve
work and report exact errors; automatic preference remains enabled.
One writer per scope: your scope is candidate only. Codex owns controls/tests/Git.

Allowed WRITES relative to candidate/ (exactly):
leaf/uptake.py, leaf/roots.py, leaf/forcing.py, leaf/traits.py,
leaf/__init__.py (compatible lazy module exports),
tests/test_uptake.py, tests/test_roots.py, tests/test_forcing.py,
docs/phase1-water-contract.md, report.md.

Do not change legacy coupling/stepwat/runtime/canopy/state/benchmarks/tests/fixtures
or old numerical defaults. Keep Phase 0 NumPy-only imports independent of Numba,
CuPy and native packages. This is part A, not final Phase 1 acceptance/publication.

## Authoritative user water rules

LEAF calculates each plant's timestep demand D[p], m3. Roots have normalized
fractions w[p,c,l], summing to 1 for each rooted plant. Species and individuals
may differ. No root plasticity / compensation: a missed request is never moved
to another cell/layer. SYRUP supplies available water A[l,c] in m3, owns the sole
water inventory and accepted debit; LEAF must not hold/decrement another store.

R[p,c,l] = D[p] * w[p,c,l]
Q[l,c] = sum_p R[p,c,l]
f[l,c] = min(1, A[l,c]/Q[l,c]) when Q>0; otherwise 0
proposed[l,c] = min(A[l,c], Q[l,c])

Every plant sharing a resource receives the SAME fraction, no species priority.
SYRUP validates/accepts the proposed withdrawals ONCE and returns accepted water
and shortfalls. Given accepted[l,c] <= proposed[l,c], final delivered to a plant
is sum_e D[p]*w[e]*accepted[resource[e]]/Q[resource[e]] (zero if Q==0).
Growth satisfaction S[p] = delivered[p]/D[p] for positive D, else 0 (explicit zero
demand policy: no invented growth); unmet[p] = D[p]-delivered[p]. Never use the
least-satisfied root or redistribute deficits. Native depth/volume area conversion
is downstream: all of THIS module's amounts are m3, all same timestep.

Required example: one plant D=100, weights .5/.5 in two layers, A=25/200:
proposed=25/50, accepted=25/50, delivered=75, unmet=25, S=.75. Deeper remaining
water 150 is unused. Multiple plants must share proportional to LOCAL requests,
not equal litres and not species/size order.

## 1. Compiled sparse uptake API (`leaf/uptake.py`)

Provide a validated plant-major CSR root map: stable int64 plant IDs, offsets
length P+1, flattened resource indices l*n_cells+c, FP64 nonnegative weights.
Support general NON-separable weights (different layer/cell distributions for
each individual), finite unit row sums, duplicate-ID refusal, index/shape bounds,
clear inactive/no-edge semantics (positive demand on an empty root row must fail).
Reject bools/fractional/overflowing IDs/indices before lossy cast. Validate input
ownership/aliasing: callers cannot silently mutate accepted host root geometry.

Implement NumPy reference, compiled Numba CPU and CuPy RawKernel GPU paths. Prefer
small shared arithmetic and one authoritative algorithm; no Python per-edge/day
loops in production CPU/GPU. CPU/CUDA scatter Q, compute proposals, then gather
accepted deliveries. NO dense P*L*N tensor, no P*P matrix, no daily edge-value
history; bound arrays by actual P,E,L*N. Reuse existing numerical style where
practical. New modules may use optional lazy imports; do not change old kernels.

Make proposal and accepted feedback separate pure operations. A plan binds its
root map and demand; acceptance validates amounts (finite, >=0, <=proposal within
explicit FP64 summation tolerance, no silent material clipping), and reports
delivered, unmet and S. State/receipt publication belongs to later driver/native
authority, not this stateless allocator. Reject foreign-plan/map/backend misuse.
Permit owned plan/feedback arrays to be consumed on the same backend without
implicit host/device population transfers. GPU uses finite/budget reductions in
small packets; no .get()/asnumpy(P/E/N arrays) per step. Validate metadata before
large transfers/allocations. CPU/GPU equivalence and strict water closure matter.

Memory admission: explicit max_edges/max_bytes; estimate active arrays, scratch,
host/device mirrors before constructing/loading maps and each allocating operation.
Refuse BEFORE large materialization (especially arbitrary supplied root arrays or
huge dimensions), preserve old data, never raise bounds/discard individuals to pass.
Report precise scope of memory estimates. No implicit GPU fallback or device switch.
Handle empty populations, zero demand/supply and many overlapping roots.

## 2. Shared spatial root builder (`leaf/roots.py`)

Build maps from authoritative PlantState + SpeciesCatalog with existing shared
canopy/root allometry (canopy.derive_geometry), preserving stable plant IDs.
Reuse existing compiled circle-overlap CSR builder from benchmarks.spatial_mapping
where feasible. Its horizontal law is uniform across positively overlapping cells,
clipped to finite domain and normalized at initialization; LABEL this as the
existing simple geometric approximation, not measured root density/area weighting.
No canopy/root allometry duplicate. Non-square grid handling must be explicit:
existing helper supports square spacing; reject unsupported geometries clearly
before allocation rather than silently using dx for dy.

Use species layer fractions by default; accept explicitly ID-keyed individual
layer-fraction overrides (all layers/finite/nonnegative/unit sum). Permit callers
to supply custom general root maps instead. Root weights remain fixed during
uptake. Growth-driven rebuild will be a later driver operation; no compensatory
renormalization of dry/masked cells. An unavailable cell gets A=0, its missed
request remains unmet; never redistribute its weight to active/wetter cells.
Allow live-only mapping or selected statuses consistently; removed/dead do not
transpire by default. Admit expanded layer edges before allocation, not after.

## 3. Scheduler records (`leaf/forcing.py`)

Provide a small strict neutral record for an interevent timestep: start/end
offset-aware fixed-offset datetime, actual temperature in degC, optional shared
PET depth rate (mm/day), explicit process kind/gate. No synthetic weather/date,
no independent clock beginning Jan 1, no catch-up over wind/rain/recession.
Positive finite dt, split boundaries at local calendar midnight and configured
growth-update boundaries later. Date/temperature REQUIRED even when PET explicit.
Expose exact/fractional seconds where feasible using stdlib Fraction/timedelta;
support leap days/year boundaries and partial days. UTC offset consistency and
contiguous progression validation helpers; data-only imports NumPy at most.
Provide conversion helper from scheduler-provided weather mapping rows using
`date` YYYY-MM-DD and `tmean_c` degC; do not import/copy MASY weather/PET physics.
Actual MASY delegates to this via future driver; full event coordinator not changed.
No temperature response or demand-growth formula here yet (pending science choice).

## 4. Co-located species files (`leaf/traits.py`)

Add `SpeciesCatalog.from_files(paths)` and compatible load_catalog single path or
sequence support. Each file is a complete existing catalog containing one or more
COMPLETE species records (allometry/roots/growth/water/etc together). Merge species
without splitting/overriding a species across files; duplicate IDs fail even if
identical. Require matching schema/units/root_layers. Deterministic sorted species
table and canonical hash independent of input file order; one-file identity stays
the same as from_file. Preserve per-species content, ownership, unknown-value
validation and existing Phase 0 tests. Paths/provenance are metadata, not scientific
hash input; document source-file list. Do not add active growth/temperature traits
or silently activate earlier study candidates in this part.

## Tests and documentation

Meaningful scalar-oracle tests for supplied 75% example, multi-plant proportional
sharing, multiple cells/layers, zero and unsatisfied roots, deeper residual unused,
mask A=0/no renormalization, custom individual weights, partial acceptance, invalid
arrays/metadata and failure preserving inputs, memory rejection before allocation,
CPU parity and CUDA tests when available (unavailable backend may skip, real
implementation failures may not). Verify finite water closure sum delivered ==
sum accepted at FP64 summation tolerance (not enlarged tolerances). Reordering
individuals cannot change resource priority. Avoid tests that just copy kernels.

Test date leap/year/partial-event gates, complete deterministic multi-file species
loading and compatibility. Existing tests stay unchanged. Document APIs/units,
memory scope, native-authority boundary and root approximation. In report.md give
the exact files/behavior/limits and tests NOT RUN. Codex will independently verify
before later Phase 1 work adds growth and the real native/scheduler driver.
