# LEAF periodic canopy and root boundaries — revision 1

Implement the user-requested periodic-boundary correction in an isolated candidate. Do not restart ecological phases or replace scientific laws. The user explicitly authorized automatic Claude/Anthropic source sharing and Codex–Claude implementation/review. Codex will independently review, execute focused CPU/CUDA/native tests, install accepted exact source and handle publication; leave changes uncommitted.

## Governing workflow / exact scope

Task periodic_boundaries_20261112; baseline LEAF HEAD85651c63003fcd0a66ba3a98af104003f64e818e, clean at start. CWD /home/okin/leaf/agent_handoffs/tasks/periodic_boundaries_20261112/candidate. Exact session in task session.json.
Read ONLY candidate/**, ../baseline.json, this prompt. Candidate contains approved LEAF code/tests/docs/examples, Python benchmarks and dependency locks. No read of upstream checkouts, case data, prior raw sessions, private material, environment credentials or orchestration files. No hidden .claude/settings modification. In safe mode these embedded instructions govern you.
Write ONLY these19 candidate paths:
leaf/state.py; leaf/periodic.py (new common boundary geometry helpers if useful); leaf/initialization.py; leaf/canopy.py; leaf/roots.py; leaf/plant_pet.py; leaf/biocalendar.py; leaf/plant_session.py; leaf/plant_checkpoint.py; leaf/checkpoint.py; leaf/recruitment.py;
tests/test_periodic_boundaries.py (new; existing tests READ ONLY);
docs/periodic-boundaries.md; docs/phase0-vegetation.md; docs/masy-daily-plant-interface.md; docs/maple-leaf-integration.md; docs/species-root-profiles.md;
examples/periodic/generation.json; periodic_report.md.
No edits outside those exact files, including existing tests, __init__, growth, uptake, benchmarks, interevent, old locks, cases, AGENTS, orchestration or accepted archives. If a genuinely necessary additional file is found, describe it in the report; do not silently widen scope. Reuse old tests as evidence, do not weaken/rewrite them. Not all allowed files need changes.
Tools ordinary Read/Edit/Write ONLY. No Bash/execution/network/agents/Git/installs or bypass. Report tests NOT RUN by you. Every write/source access is independently audited. Do not make claims from hypothetical execution.
One writer: Claude owns these candidate files; Codex owns prompt/baseline/session/review/evidence and may read frozen snapshots concurrently. Normal dispatch→implement→review→verify→correct continues until independent acceptance. Keep source readable/modular and changes bounded.

## User intent / context

User found an artificial bare border in a MAPLE case. Its wrapper used a2m inset (60x60 placement in64x64), because LEAF exclusion was finite/nonperiodic. MAPLE developer confirms current MAPLE ALREADY wraps LEAF-imported canopy footprints and shelter across periodic seams; one shrub atx63.8,r0.5 spans both right and left but remains ONE plant. Thus no new MAPLE wind/rasterizer work is needed here. User requests LEAF-generated periodic populations with seam wrapping and no canopy overlap when specified, and root distributions that also wrap. The existing Jornada case is finite/nonperiodic and MUST remain unchanged, unrun and unregenerated. No MAPLE/SYRUP upstream writes. Supply a concrete developer handoff to remove the MAPLE wrapper inset when they choose to regenerate; do not rewrite historical populations.

LEAF owns plants, canopy/root geometry, demand/allocation/biological feedback. SYRUP owns sole physical water and every water update. MASY supplies shared forcing, nominal timing, common boundary config and sequences native lifecycle calls. New water-free PlantSession must keep this; no new hydrology, debit, climate file or schedule.

## Inspect before editing

Read actual Domain serialization/sampling/equality and PlantState CSV+companion/checkpoint logic; initialization exclusion/cover fitting; canopy shared view/measure_cover; roots build/adoption/compiled helper; plant_pet exact-area count/fill kernels and reduction/mask/admission; biocalendar rebuild; PlantSession identity/config/checkpoint; reproduction daughter placement. Read tests/phase0, roots, plant_pet, session, checkpoint and source locks/qualification docs in candidate. Reuse functions and equations, do not infer semantics solely from names. The prior task added qualified C1 PET and calendar; preserve its exact area/demand laws and shared memory cap fix.

## Boundary representation and backward compatibility

Use ONE immutable per-axis boundary declaration on Domain, e.g. optional keyword-only `periodic=(False,False)` inx/y order. Domain.from_dict accepts optional `periodic:[bool,bool]` in generation/domain metadata; strict types, not truthiness of strings/ints. Default remains finite closed nonperiodic. Preserve old nonperiodic coordinate sampling, seeded generation and serialized dictionary convention/bytes as far as possible; preferably emit new periodic metadata only when needed so legacy plain CSV companions/readers and reports remain compatible. Equality/hash/repr and metadata validation MUST distinguish periodicity. Explicit legacy conflicting metadata (edges says finite but periodic true) must refuse. Do not silently reinterpret old `edges:"periodic"` without a valid declaration.

Periodic positions at the upper edge are equivalent to the lower edge for cell sampling/mapping. Define this precisely; preserve old closed-upper sampling on nonperiodic axes. Keep ordinary input position validation (no silent normalization of arbitrary invalid plant records); expose/use explicit coordinate wrapping for newly computed daughter coordinates on periodic axes. Domain origin offsets and mixed periodic/nonperiodic axes must work. Existing square-cell restriction for compiled root/PET builders may stay explicit; do not silently use dx as dy. No coordinate shifts, duplicate plant IDs or plant records to model images. Shared canopy export carries boundary metadata and one object per plant; MAPLE handles physical canopy images itself.

## Initial canopy placement / measured cover

For periodic cases sample centres across the FULL domain, no margin or inset. Keep existing lognormal sizes, independent age draws, sorted large-first exclusion, count fitting, resource/attempt bounds and failure-atomic output.

Exclusion mode: reject overlap using shortest wrapped centre distances on each periodic axis and ordinary distances otherwise; same strict < (ri+rj)^2 tangency rule. Use a sparse spatial index that is correct for seam neighbors and non-integer bin/domain ratios (a shortened last bin can invalidate naive +/-1 modulo searches). Avoid all-pairs population comparisons. Do not let image copies of the SAME plant become separate neighbors or records. A canopy whose diameter exceeds a periodic span self-overlaps: in non-overlap mode refuse explicitly, no silent size rescale/drop/buffer. Exact tangency may be allowed consistently. For overlap-allowed mode retain independent uniform centres and correct periodic union cover.

measure_cover must include wrapped canopy fragments, including corners, and count union correctly once. Keep sampled midpoint cover/tolerance model, bounded row tiles and existing resource admissions. Mixed axes clip only nonperiodic axes. No dense PxN. Nominal pi*r^2 area may exceed physical torus union for oversized circles; do not double-count physical union or claim analytical cover from summed areas. Boundary notes/definition/report provenance must describe actual mode. Nonperiodic branch should remain unchanged numerically and reproducible.

## Canopy-area-weighted PET (not only the plot or generator)

Update build_canopy_map/canopy_map_from_state so periodic canopies read terrain PET across seams. Reuse the exact existing circle–rectangle area integration; do not switch to root weights, centre PET, midpoint sampling or a different demand law. One sparse entry per physical plant-cell pair; sum wrapped fragments within that cell, avoid duplicate-cell weighting. Masked cells stay excluded; zero-valid-support refusal stays. area_total / PET denominator reflect the actual represented physical canopy support. For ordinary small circles in a fully periodic domain area must sum to pi*r^2 within existing FP64 bounds; mixed axes clip only the nonperiodic axis.

For oversized periodic circles handle self-image overlaps as UNION (nearest-image/Voronoi partition is one option), not summing overlapped images twice; root oversized tests below are mandatory. If exact union is genuinely unsupported, reject that canopy size explicitly before allocation with a documented limit—never silently double-count/fallback. Do not impose that canopy limit on roots: larger roots must still work.

## Root mapping and water contracts

Current root law is a circular footprint with radius = species lateral_radius_multiplier * canopy radius. Horizontal root weights are UNIFORM over UNIQUE positively overlapped cells (not area-weighted root density), then multiplied by species/individual vertical layer fractions. Keep that law, normalized weights, positive-overlap predicate/tangent exclusion, sparse plant-major CSR and layer/cell order.

On periodic axes wrap the root footprint across seams/corners, clip only nonperiodic axes. Each physical plant-cell-layer pair occurs ONCE, even for roots wider than a domain, nx=1/ny=1, or duplicate images hitting one cell. Define toroidal footprint as union/nearest-image support, normalize ONCE over unique reached cells; species layer fractions unchanged. Oversized root systems should saturate unique physical cells rather than allocate unbounded image copies. Roots from DIFFERENT plants remain allowed to overlap and compete even if canopy overlap is disallowed.

Use compiled CPU count/fill for production and backend-resident NumPy/Numba/CUDA uptake. A NumPy reference builder can remain slow/reference explicitly. Reuse helpers instead of cloning all existing kernels; leaf/periodic.py can hold common boundary geometry. Benchmarks/spatial_mapping.py is read-only; preserve its legacy nonperiodic path and route periodic builds through an appropriately compiled shared helper. No routine P-sized host/device transfers in daily demand/uptake; existing cold geometry rebuild transfers are allowed and must be counted. Pre-admit P/edge/scratch outputs, coexisting old+new geometry and conversion/checkpoint bytes, no dense plant-by-grid buffers, raised caps or dropped plants to pass tests.

Do not alter actual water/competition laws: root-weighted requests, proportional local competition with equal fraction of requests, no species priority, no compensation to wetter cells, actual accepted summed uptake drives growth, zero-demand behavior unchanged. Native SYRUP availability and debit remain visible at MASY call site and authoritative exactly once. No changes to growth.py or uptake.py permitted.

## Runtime / export / restart / recruitment consistency

Propagate the SAME Domain flags through growth-driven rebuilds (biocalendar/session/standalone paths), shared canopy export and root maps. Both biological-only and standalone checkpoint/plant metadata must persist flags and refuse restoring with mismatched boundary choice. Preserve old default formats and explicit source guards/versioning; never silently reinterpret old nonperiodic checkpoints as periodic. Include new helper sources in relevant source-hash sets. Source/hash changes require explicit qualification, not rewritten old locks.

Recruitment may change ONLY daughter-coordinate boundary handling: explicit wrap each periodic axis, retain existing rejection on nonperiodic axes; same single daughter ID, age0, current-surface sampling and parent-funded B/E. No new probabilities, eligibility, seed bank, recruitment timing, damage, mortality or stress law. Preserve nonperiodic outputs and RNG draws. If no recruitment edit is needed for a documented supported scope, explain clearly rather than advertising a complete periodic lifecycle while daughters still see a hard wall.

## Tests and evidence (Codex executes)

Add tests/test_periodic_boundaries.py with independent expectations, not just repeating implementation. Existing tests read-only, retain full baseline suite.
- Default nonperiodic Domain/serialized companion and fixed-seed generation unchanged; strict bool flags, per-axis metadata/equality/hash/roundtrip, origin offset, upper-edge sampling.
- Hand-placed seam/corner canopy overlap vs non-overlap, mixed axes, tangent and self-overlap cases, awkward bin ratios, generated target cover and no inset, one ID/record perplant. Compare cover with independently tiled images/midpoint raster or minimum-distance oracle, no image duplicates.
- Exact canopy PET seam weighting and areas, spatially varying PET, cell mask, mixed edges, translation equivariance, correct oversized handling.
- Root cells on both sides, corner/mixed/offset cases, species and individual layer fractions, unique resources, unit weight sums, roots larger than domain, nx/ny=1. Independently brute-force smallest grids using minimum distances/closest point to cells.
- Actual uptake of water available only across seam; cross-seam plant competition, inactive/dry cells, no compensation, total accepted=delivered; one plant keeps one record. CPU/Numba/CUDA agree under existing numerical contracts (skip only genuinely unavailable paths).
- Geometry rebuild and growth keep flags, checkpoint roundtrip incl pending, wrong-boundary restore refusal, source hashes. Periodic clonal daughter wraps and donor/daughter ledger closes; nonperiodic unchanged.
- Tight edge/byte limits refuse before large allocations; no all-PxN storage. Small memory/compiled parity/translation checks.
- A small optional real-native N15 test may reuse tests/native_leaf_fixture.py and NativeDailyTests patterns with configured LEAF_NATIVE_*; the published native fixture is periodic in both axes. Do not create fake native water state. An unavailable unconfigured native path skips honestly; configured broken source is failure. Keep native source pins unchanged. Codex will run actual CPU/CUDA native qualification independently too.

Write docs/periodic-boundaries.md: exact API/config, semantics, one-ID wrapped geometry, roots vs canopy overlap, CPU/GPU/memory and remaining restrictions; runnable tiny generator config using existing example species without duplicating traits. Update authorized existing docs to avoid claiming all domains are nonperiodic. Concrete MAPLE/MASY handoff: full-domain Domain(periodic flags), no placement_inset or inner cover correction, oneplant per CSV row, MAPLE already wraps canopy/shelter, shared boundary propagation and source-qualified rebuild/restart. Do not change MAPLE case files or old trial data.

Report in periodic_report.md: changed files, design/compatibility choices, laws preserved, actual limitations, tests NOT RUN by you and suggested exact commands. No acceptance/validated ecology/performance claims. Leave uncommitted; Codex reviews/runs tests/corrects automatically.
