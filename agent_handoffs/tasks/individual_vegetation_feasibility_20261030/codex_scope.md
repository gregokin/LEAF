# Active assessment scope and clarified design

User authorization: proceed with existing models/code infrastructure and design assessment, simultaneously considering speed, memory and GPU compatibility. This is an assessment, not scientific implementation, model runs, package installation, upstream edits, Git initialization, commit or push.

- Spatial individuals have X/Y coordinates and original ground elevation at establishment, biomass, age, species and functional type. Do not retain plant height at establishment. Current height and canopy/root geometry follow biomass/allometry.
- Individuals may span multiple cells, and plants have subcell locations even on coarse grids.
- Shared MASY landscape/surface supplies authoritative soil conditions, including evolving texture; MAPLE remains the sediment-storage/wind implementation and SYRUP the water/sediment implementation.
- A relatively simple multilayer persistent soil-water component, probably housed in SYRUP, runs regularly (likely daily) between transport events for plant water uptake/transpiration, evaporation and drainage. Wind, water transport, and biological water use/growth occupy exclusive phases. Maintain one physical calendar and budget; assess event crossings/daily boundaries explicitly without silently dropping elapsed time.
- Initially soil moisture has no effect on wind transport; preserve an explicit future interface. Do not duplicate MAPLE/SYRUP/ECOTONE wind physics.
- At the beginning of a growth period establish root fractions linking plants to different soil cells/layers; water access depletes shared stores daily. Biomass updates may be weekly or otherwise coarser, using PFT-specific WUE as a candidate rule, not a selected law.
- Export physical aboveground locations/geometries at phase boundaries for transport models. Assess the cost of rebuilding spatial/root mappings when plants or surfaces change.
- Mortality/injury from burial, erosion/root exposure/dislodgement and abrasion is species-specific, including species sharing a PFT. Compare the evolving surface with establishment reference while assessing additional history/root-state needs; do not claim one ground-height difference fully defines damage.
- Include age/stochastic mortality, temperature/frost sensitivity, simple seeds and vegetative recruitment options; fire is deferred.
- Preserve individuals if that is selected. CPU/GPU memory and runtime are first-class, including persistent state, scratch reuse, transfers, sparse overlap edge counts, spatial indexing, RNG and deterministic/reproducible restart. Sequential phases avoid concurrent kernels but persistent allocations can still add.
- Verify public precedents (Tietjen, STEPWAT2/SOILWAT2, ECOTONE ecology, Caylor 2004/2005/2006 and relevant alternatives), actual local code, source identities and prior review dispositions. Distinguish facts, hypotheses, proposed new science, adapters and unchanged reusable code. No measured speed ranking without benchmarks.
- User may provide target scale asynchronously; proceed with symbolic costs and explicitly illustrative numerical memory examples meanwhile.

Source scope for Codex: relevant repository instructions, source/design files and prior assessment receipts, public papers/official source/documentation, read-only device metadata. Do not inspect or share simulation payloads, secrets or unrelated jobs.

Claude collaboration is restricted to public-literature/public-code/generic algorithm questions using safe mode, public web tools, no file/Bash tools, and an empty temporary cwd. The previously rejected detailed project-outline payload is not to be retransmitted. Local ECOTONE source and user design stay local; Codex independently integrates and verifies the public-source analysis.

Deliverables: a repository assessment with evidence/links, source and dirty/lock provenance, sparse-state memory formulas and labelled scenarios, architecture/coupling contracts, unresolved scientific decisions and bounded next phases; durable task records and concise user-facing findings. No implementation or model benchmark.
