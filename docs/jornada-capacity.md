# Jornada plant counts and initial memory capacity

Date: 2026-10-30. Scope: literature-based geometric population estimates and a
bounded synthetic CPU/GPU root-mapping/canopy benchmark. This is not a vegetation
growth or water solver. Initial water-layer counts are two and three, following
the user's direction.

## Field evidence and interpretation

Ziegler et al. (2023), Table 1, measured a **100 × 100 m Jornada Experimental
Range site**, using three 100 m transects and line-point intercept. Selected
surveys reported:

| Survey | Total shrub cover | Total grass cover |
| --- | ---: | ---: |
| December 2019 | 8.10% | 8.44% |
| March 2020 | 7.51% | 6.92% |
| June 2020 | 17.0% | 6.32% |
| September 2020 | 17.0% | 5.82% |
| December 2020 | 9.36% | 5.23% |

The text attributes the shrub seasonal increase to mesquite leaf emergence.
The table formally reports total shrubs; use mesquite-equivalents rather than
claiming an exact mesquite census. Seasonal foliar cover changes do not imply
plants appeared or disappeared in proportion. [Paper](https://doi.org/10.1016/j.agrformet.2022.109295),
[USDA publication record](https://www.ars.usda.gov/research/publications/publication/?seqNo115=394517).

A wider Jornada field report surveyed twenty pairs of 5-ha plots in autumn
2020. It reports **3–28% shrub cover, largely mesquite**, and **2–43% perennial
grass cover**. These are independent ranges across a degradation gradient;
the two maxima are not a measured joint community. Its short report does not
fully specify cover definitions. Use it to bracket capacity, rather than as a
single representative plot. [Jornada field report](https://lter.jornada.nmsu.edu/plant-community-recovery-following-chemical-control-of-mesquite/).

## Individuals per hectare using the supplied sizes

Assume one modeled grass individual per 10–20 cm clump, mesquite width 2 m,
and fully occupied nonoverlapping circular footprints. For cover fraction C,
diameter d in metres, and area A = 10,000 m²:

`P = A C / (π d² / 4)`.

The June measured cover pair gives:

| Assumed grass diameter | Grass equivalents | Mesquite equivalents | Total |
| --- | ---: | ---: | ---: |
| 10 cm | 80,469 | 541 | 81,010 |
| 15 cm | 35,764 | 541 | 36,305 |
| 20 cm | 20,117 | 541 | 20,658 |

Across the broader marginal cover ranges, grass equivalents span about
**6,366–547,493/ha**, and mesquite equivalents **95–891/ha**. These counts are
calculations using the user's widths, not published observations of density.
If widths describe basal clumps while observations describe foliar cover, the
conversion needs a basal-to-canopy relation. Canopy gaps, overlap, size variation
and connected clones also prevent a unique biological count. A random overlapping
disk model would use `P = −A log(1−C)/area_disk`; it increases the 43% grass-cover
count by about 31%, illustrating sensitivity rather than selecting that model.

## Sparse memory estimate with two or three layers

Use the previous conservative illustrative ledger: eight FP64 soil fields per
cell-layer, 128 bytes per individual, five FP64 native vegetation rasters, and
28 bytes per expanded root connection including a reverse view/flux scratch.
With N cells, L layers, P plants and H horizontal root connections:

`bytes = 64NL + 128P + 40N + 28HL + 4(NL + P + 2)`.

The eight fields are a budget assumption for state/parameters/selected work,
not eight required moisture stores. Actual plant fields and algorithms may need
more. At 0.25 m, a 100 m square is **400² cells**; one FP64 water field takes
2.44 MiB with two layers and 3.66 MiB with three.

The user's working range is **root radius = 3–5 × canopy radius**, superseding
our earlier 2× illustration. For 10–20 cm grass diameters this gives lateral root
radii of **0.15–0.50 m**; 2 m mesquite crowns give **3–5 m** root radii. Circular
root footprint area is therefore 9–25 times canopy area. This changes shared
resource connections, not individual counts. The range is supplied by the user;
it is not a newly verified Jornada rooting measurement or a choice of root depth.

For a circular footprint translated uniformly over an unbounded square grid,
the expected number of cells with positive area overlap is
`k = 1 + 4r/dx + π(r/dx)²`. The perimeter and constant terms matter for subcell
grasses; estimating only disk area divided by cell area undercounts connections.
Finite-domain clipping reduces the expectation near boundaries. The numerical
benchmark measures actual intersections instead.

| Capacity case, 100 × 100 m at 0.25 m | Grasses | Shrubs | Two layers, roots 3–5× | Three layers, roots 3–5× |
| --- | ---: | ---: | ---: | ---: |
| June cover, 10 cm grass | 80,469 | 541 | 71.0–110.7 MiB | 98.4–157.9 MiB |
| June cover, 15 cm grass | 35,764 | 541 | 59.6–97.0 MiB | 84.0–140.1 MiB |
| June cover, 20 cm grass | 20,117 | 541 | 55.0–91.3 MiB | 78.2–132.6 MiB |
| High-grass bound: 43% grass, 3% shrubs, 10 cm grass | 547,493 | 95 | 230.8–340.7 MiB | 308.7–473.5 MiB |
| High-shrub bound: 2% grass, 28% shrubs, 10 cm grass | 25,465 | 891 | 60.2–104.9 MiB | 85.6–152.7 MiB |

The last two cover combinations are constructed capacity cases within marginal
ranges, not reported joint observations. The lower and upper bounds use radius
multipliers 3 and 5 respectively; population, soil layers and other assumptions
are held fixed. These supersede the earlier 2× baseline estimates.

These figures exclude sediment/transport state, graph publication/rebuild copies,
sort work, CUDA context/pools, host/checkpoint mirrors and ecological state beyond
the assumed record. They support feasibility of these synthetic populations on
an 11 GiB device, not a whole-model memory qualification.

A factorized horizontal map plus per-plant vertical profiles can reduce storage
if separability is scientifically valid. The illustrative layer-streamed ledger,
`64NL + 128P + 40N + 28H + 8PL + 4(N + P + 2)`, gives **45.4–64.1 MiB** with two
layers or **55.5–74.2 MiB** with three for the June 15 cm grass case. The high-grass
case gives **171.1–226.0 MiB** or **185.0–239.9 MiB**, respectively. It requires an
allocation algorithm that does not materialize all H×L fluxes simultaneously.

At 512² and 0.25 m, the domain is 128 × 128 m: scale populations and these
uniform-density estimates by 1.6384. With three layers, the expanded ledger is
approximately **138–230 MiB** for the June 15 cm case and **506–776 MiB** for the
high-grass case. A 512² grid at 1–2 m spans much more than one hectare; its
population must scale with area.

Machine-readable [cover evidence](../agent_handoffs/tasks/jornada_capacity_benchmark_20261030/cover_sources.json)
is unchanged. The [updated capacity arithmetic](../agent_handoffs/tasks/root_radius_3_5_20261030/capacity_estimates.json)
uses multipliers 3, 4 and 5; the previous task retains the original 1/2/5 sensitivity
records as historical evidence.

## Original bounded implementation and measurements

The authorized prototype covers synthetic spatial inputs, positive-overlap root
mapping, layered graph traversal and an 8×8 sampled circular canopy projection.
It deliberately does not select root competition, WUE, demography, hydraulic
equations, periodic/elliptical production geometry, or full native physical-field
publication. Uniform root weights and separable vertical profiles are workload
fixtures. Native integration remains a later task.

Synthetic uniform or clumped centres permit overlap: these fixtures preserve
the calculated individual counts, rather than reconstructing observed cover.
For example, a 43% nonoverlapping-equivalent grass count gives about 35% expected
true union cover with independent random overlapping disks before boundary
clipping. The prototype's capped additive coverage is also an approximation to
union cover, so it must not be interpreted as a calibrated vegetation map.

In the original sweep, all **11 cases passed** CPU/GPU field comparisons and complete-population
synthetic-demand audits. All **20 unit tests passed**, including actual GPU
parity, admission before allocation, missing-plant audit failure, clipping,
small roots and canopy aliasing.

Hardware: one GTX 1080 Ti (11 GiB), Intel Core i9-7900X; CPU hot loops use one
thread. Python 3.12.3, NumPy 2.5.3, Numba 0.67.0, CuPy 14.2.0, CUDA runtime 12.9.
Kernel figures below are medians of five warmed repeats. Graph-build figures
are one synchronized build per case, not medians. This single sweep does not
establish a whole-model speedup or control GPU clock variability.

| Three-layer synthetic case | Plants | Horizontal links | GPU build | GPU reduction | GPU canopy | Live CuPy peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| June, 100 m square | 36,305 | 286,381 | 1.52 ms | 0.53 ms | 1.43 ms | 10.4 MiB |
| June density, 512² at 0.25 m | 59,483 | 469,858 | 0.98 ms | 1.26 ms | 1.17 ms | 17.0 MiB |
| High-grass bound | 547,588 | 1,717,321 | 3.65 ms | 2.95 ms | 3.45 ms | 56.0 MiB |
| High-grass, root radii ×5 | 547,588 | 4,564,089 | 6.93 ms | 9.98 ms | 4.46 ms | 88.6 MiB |

For the June 100 m case, single-thread CPU build/reduction/canopy took
3.72/2.00/6.59 ms. For the widest-root high-grass case they took
50.52/37.14/58.50 ms. The same June map with two layers had a 0.35 ms GPU
reduction versus 0.53 ms with three. These operations are small on either
compiled backend; choosing where demography runs still requires full-process
measurement.

Input transfer was 0.82 ms for the June 100 m case; copying the graph and outputs
back for validation took 2.61 ms. In the widest-root high-grass case these were
4.38/8.33 ms. A preceding high-grass transfer took 15.83 ms, showing that individual
allocation/transfer observations vary. Timed reductions kept the graph on the
GPU. Production code should not copy the full graph back daily.

Observed startup included about 0.49 s CPU compile/cache warmup, 0.21 s GPU context
initialization, 0.31 s RawModule compilation/cache loading and 0.12 s GPU operation
warmup. These are cache-dependent observations, not a controlled cold-start test.
The full 11-case profiled process took about 11.2 s including imports.

The independent CuPy malloc/free hook measured **88.6 MiB peak live rounded pool
allocations**, including temporary arrays. All tracked allocations were released
at the end and unknown-free events were zero. The largest observed cached-pool
reservation was **122.7 MiB**. These figures exclude CUDA context and other
allocators. Cumulative CPU process peak RSS was approximately **770 MiB**, including
Python, compiled libraries, CPU graphs and host copies used for GPU validation;
it is not the ecological-state footprint or a per-case RSS measurement.

The original 2×-root June 100 m prototype device arrays total 10.24 MiB. They
store only the geometry fixture, factorized root graph, synthetic reduction and
one cover field. Its historical 64.9 MiB ecological estimate is superseded by the
current 3–5× range above. The fuller ecological ledger includes more plant/soil
state, five physical rasters and expanded graph/reverse/flux storage; the smaller
prototype allocation must not replace that budget.

At 1 m cells, the 15 cm grass fixture missed 2,870 of 59,483 total plants in canopy sampling;
at 2 m it missed 168,037 of 237,928, consistent with the independent aliasing analysis.
All plants retained root connections and synthetic unit demand. Sampling error
is a representation finding, not a reason to delete individuals.

Results support bounded sparse maps for these populations, and show that actual
root extent and residency/transfer policy deserve attention. They do not time
ET, drainage, competing water allocation, biomass, recruitment, injury, full
native vegetation publication or transport.

Source identity: `benchmarks/spatial_mapping.py` SHA256
`333921bb645d984fa8e7449619d03114262ee333c7fea61a22493a878fd9ccc5`.
See [raw results](../benchmarks/results/jornada_20261030/profiled_results.json),
[compact CSV](../benchmarks/results/jornada_20261030/summary.csv),
[benchmark implementation](../benchmarks/spatial_mapping.py),
[commands and limits](../benchmarks/README.md), and
[unit-test receipt](../agent_handoffs/tasks/jornada_capacity_benchmark_20261030/tests_02.exit.json).

## Updated root radii: measured 3× endpoint

Four additional cases passed CPU/GPU field comparisons and full-population demand
audits using the unchanged reviewed implementation. Together with the previously
measured 5× cases, these cover the requested lateral-radius range. No code or
resource limits changed, so the earlier 20-test source qualification remains
applicable without rerunning an unchanged test suite.

The earlier GPU was occupied; these runs used another idle GTX 1080 Ti. These
are workload measurements across separate runs/cards, not controlled timing
ratios between root multipliers. Kernel figures are medians of five warmed
repeats; map construction is one synchronized measurement per case.

| 3× root radius case | Plants | Horizontal links | GPU build | GPU reduction | Live CuPy peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| June hectare, three layers | 36,305 | 519,757 | 3.43 ms | 0.94 ms | 13.1 MiB |
| June hectare, two layers | 36,305 | 519,757 | 2.69 ms | 0.85 ms | 11.8 MiB |
| June density, 512², three layers | 59,483 | 853,849 | 2.50 ms | 1.95 ms | 21.4 MiB |
| High-grass hectare, three layers | 547,588 | 2,520,640 | 5.21 ms | 4.80 ms | 65.2 MiB |

Live rounded pool allocation peaked at **65.2 MiB** for this additional sweep,
with zero unknown frees and zero live tracked bytes remaining at completion.
The 5× high-grass case previously reached **88.6 MiB**. These prototype peaks
exclude CUDA context and the extra fields in the ecological memory estimates.
[Updated raw results](../benchmarks/results/root_radius_3_5_20261030/profiled_results.json),
[execution receipt](../agent_handoffs/tasks/root_radius_3_5_20261030/benchmark.exit.json).

## Coarse-cell canopy sampling finding

The current native reference uses an 8×8 midpoint lattice per cell. An
independent nearest-lattice-point calculation shows that a randomly positioned
10 cm circular grass crown has about **49.7% probability of hitting no sample
at 1 m cells**, and **87.4% at 2 m cells**. At 2 m, a 15 cm crown is missed
about **71.7%** of the time. These percentages concern individual contributions;
many randomly positioned plants can still give a reasonable spatial average.

Root access must use geometric cell intersection rather than this canopy
sampling rule. For future physical publication, evaluate analytic circle/ellipse
cell area or adaptive quadrature before accepting the coarse-cell representation.
The prototype deliberately exposes this limitation and does not modify native
transport physics. [Independent reference](../agent_handoffs/tasks/jornada_capacity_benchmark_20261030/canopy_aliasing_reference.json).
