# Survey verification

Status: source/document checks pass; no ecological-model validation claim.

The31protected baseline originals and three additional ecological/build sources
are byte-identical. All12archived STEPWAT files match the earlier manifest.
Three new public source GETs were pinned to STEPWAT2 revision
`003cb9b4adbf13d64a39d6d62fd95280adac27fc`; `ST_indivs.c` matches the prior archive.
Three public author-publication pages were also fetched; all six GETs returned200
and have saved hashes. These files were not compiled/executed.

Actual upstream HEAD/branch/dirty-status commands produce exactly the baseline
results. Root AGENTS.md and all inspected runtime source are unchanged. Pool
memory figures are `8*P*array_count/2**20`, not measurements. Active local report/
README/brief/decision links were checked. See `source_and_document_checks.json`
and `pool_storage_arithmetic.json`. No tests were added or run for documentation.

Code-specific findings were checked against active call paths: ECOTONE
`ecotone.f -> grow.f -> gro_new.f`, annual resource aggregation in `swater.f`,
inverse WUE/maintenance/grass-shrub growth in `gro_new.f`, and actual current
LEAF weekly `_growth_candidate`. The pinned STEPWAT growth routine and resource
conversion were inspected directly; abstract-only claims were not substituted
for those code facts.

Primary literature coverage is recorded per source in `references.json`.
Tietjen, GRASSMIND manual, Williams–Albertson, STEPWAT paper, Caylor2006,
FAO calculation scheme and AquaCrop primary evaluation supplied readable methods.
Caylor2004/2005 conclusions are restricted to verified abstract/author records;
no shared codebase or detailed uninspected physiology is asserted. Peters2002
paper details are not inferred from the later adapted local source.

Access limitations/fallbacks (not model failures or rate limits): some web-tool
public code/author-page fetches returned `Cache miss`; ordinary public GETs then
succeeded. Original Jornada Peters PDF returned403. FAO full-book retrieval was
rejected as too large and chapter3timed out; the official calculation scheme and
primary GMD equation were used. No package installation or environment creation
was used to work around unavailable PDF extraction tools. No approval rejection,
usage/auth limit or private-payload transmission occurred.

Proposed daily/weekly adapter equations and GPU design are identified as LEAF
recommendations, not exact reproductions or measured candidate runtimes. The
existing code and coefficients remain benchmark fixtures.
