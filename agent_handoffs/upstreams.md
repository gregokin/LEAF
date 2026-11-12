# Upstream Sources and Provenance

## Growth-method survey sources — 2026-10-31

[The survey](../docs/water-to-biomass-survey.md) distinguishes primary papers,
active user-owned ECOTONE code and proposed LEAF adaptations. New public
`ST_resgroups.c`/`ST_species.c` were retrieved at the previously inspected
STEPWAT2 revision `003cb9b4adbf13d64a39d6d62fd95280adac27fc`;
the retrieved `ST_indivs.c` exactly matches its archived hash. These selected files
are read-only evidence, not a complete runnable checkout. No model was compiled/run.

Hashes and retrieval URLs are in
`tasks/water_biomass_survey_20261031/public_sources/manifest.json`; literature
access/coverage is in that task's `references.json`. Original ECOTONE-paper PDF
access failed, so specific growth equations are attributed to the actual adapted
local code. Detailed Caylor2004/2005 physiology was not verified from full methods.
No private source or project brief was sent to an external agent; only public GET
requests were used. No upstream source, environment or dependency lock changed.

## Qualified native transpiration integration — 2026-10-31

Current dependency: SYRUP `feature/subsurface-hydrology`, HEAD
`deb60c29834a4aad0bf8cb4cfbb611fc28ebe090`, including its uncommitted native column
and new `subsurface_transpiration.py`. Actual MAPLE runtime HEAD:
`963b06ad1cf0f3e2c420a7d7fb50939e3bd43b7d`. Exact71 SYRUP modules and386 MAPLE
source hashes are pinned by [the local source lock](../dependencies/syrup-subsurface.lock.json).
Read-only verification: existing Python plus
`agent_handoffs/tasks/coupled_transpiration_20261031/verify_sources.py` from LEAF.

This implementation uses the real native column on that branch, not copied
hydrology or the historical MAPLE39 recipe. The existing shared-storage recipe
and checkpoint identities were preserved. The isolated candidate was tested
against symlinks to original sources before installing only the new native
module/test. Original source identities remained unchanged. This qualifies the
daily column/plant integration, not upstream storm/pond or MASY integration.

## Locations and ownership

| Location | Role |
| --- | --- |
| `/home/okin/MAPLE` | Aeolian transport and authoritative sediment storage. |
| `/home/okin/SYRUP` | Water erosion and sediment transport using actual MAPLE storage. |
| `/home/okin/MASY` | Coordination of alternating native wind/water events. |
| `/home/okin/MAHLERAN` | Scientific reference, including adapted Tietjen routines. |
| `/home/okin/Ecotone Wind 9/Ecotone Wind 9` | User-owned ECOTONE ecological/soil-water sources; exclude WEMO wind. |

The actual MAPLE/SYRUP/MASY directory names are uppercase. Read each source
repository's `AGENTS.md` before working there. Keep implementation isolated in
LEAF where practical; preserve existing dirty changes, jobs, and other agents'
work. This setup grants no upstream edits or runtime updates.

## Evidence to establish before reliance

Record current HEAD, relevant dirty/untracked runtime inputs and content hashes,
dependency locks, and actual imported source origins. Use a scoped immutable
manifest where Git metadata is unavailable. Historical identities below do not
establish current checkout contents or compatibility. Do not auto-pull, reset,
initialize Git, or install dependencies to manufacture a baseline.

- Historical SYRUP assessment: `deb60c298` plus uncommitted changes and a
  qualified MAPLE39 dependency.
- Inspected STEPWAT2: `003cb9b4adbf13d64a39d6d62fd95280adac27fc`.
- Pinned SOILWAT2: `b05f410a9e96a5f26857bd37247fe7152e9861d8`.
- The selected STEPWAT2 archive under the previous comparison task is not a
  complete runnable checkout. Verify its `stepwat_source_manifest.json`.
- ECOTONE snapshot identity is recorded by that task's
  `ecotone_source_manifest.json`; `Src/copy` contains backups, not active sources.

Useful SYRUP files: `AGENTS.md`, `claude.md`,
`docs/conservative_water_usage.md`, and
`docs/maple39_shared_storage_qualification.md`. Previous assessments are located
as recorded in [project_brief.md](project_brief.md).

## Runtime and reuse constraints

Existing interpreters are `/home/okin/MAPLE/.venv` and
`/home/okin/MAPLE/.venv-cupy`. Check their contents and governing instructions
before proposing any new environment or installation. No environment or model
execution was needed for this documentation setup.

Prefer actual upstream APIs and bounded adapters. Keep improvements independently
usable in their owning project. Do not copy/fork MAPLE wind physics, SYRUP
sediment physics, or ECOTONE/WEMO wind. Qualification of a source pair, persistent
water policy, ecological adapter, or CPU/GPU path belongs to a later bounded
assignment; no such compatibility is claimed here.

## Individual vegetation assessment — 2026-10-30

Actual HEADs and dirty/untracked source hashes are recorded in
[source provenance](tasks/individual_vegetation_feasibility_20261030/source_provenance_before.json).
The [preservation check](tasks/individual_vegetation_feasibility_20261030/upstream_preservation.json)
rehashes all 788 manifested paths and compares Git HEAD/status after inspection.
No upstream changes were made. MAHLERAN has no root AGENTS.md; the other source
repositories' guidance was read.

The current SYRUP lock qualifies a MAPLE39 source identity, not the live MAPLE
HEAD. See [lock/environment metadata](tasks/individual_vegetation_feasibility_20261030/dependency_environment_metadata.json).
Existing NumPy/Numba and CuPy metadata were inspected without importing model
code or creating an environment. The archived STEPWAT2 file selection and 13
selected active ECOTONE ecological sources match their prior manifests:
[hash comparison](tasks/individual_vegetation_feasibility_20261030/archived_source_verification.json).

Primary literature and current interface limitations are linked in the
[assessment](../docs/individual-vegetation-feasibility.md). No portable ECOTONE
baseline, STEPWAT2 executable, continuous-water integration or GPU performance
qualification was produced by this source-only assessment.


## Literature parameter update — 2026-11-11 local

See case [source table](../cases/jornada_grass05_shrub10_256/SOURCES.md) and [checked evidence](tasks/literature_parameters_20261111/codex_checked_evidence_02.md). New primary inputs: Northup etal2005 DOI10.1016/j.jaridenv.2004.09.019 Tables1/7 (honey mesquite, southern Texas), McClaran etal2013 DOI10.1016/j.jaridenv.2012.08.015 Table2 (velvet mesquite, Santa Rita), FAO56 Chapter7 Table19/Example35. STEP model input `species.in` at003cb9b4adbf13d64a39d6d62fd95280adac27fc SHA256c20a67b0a8095f4c9c920c3f73a711437ecc43039a934a1f4a351bc864bce2f3 supplies blue-grama proxies only. Public source URL/hash receipts archived; raw downloaded PDFs ignored from Git. No upstream repository modified.
