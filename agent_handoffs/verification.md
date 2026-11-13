# Verification

Periodic canopy/root task independently accepted at exact revision4. [Full review](tasks/periodic_boundaries_20261112/final_review.md) · [source/check index](tasks/periodic_boundaries_20261112/qualification.json) · [installed verification](tasks/periodic_boundaries_20261112/installed_verification.json).

| Check | Result |
| --- | --- |
| Frozen full CPU |730 tests:677 passed,53 skipped |
| Frozen full CUDA |730 tests:722 passed,8 skipped |
| Installed full CPU |730 tests:679 passed,51 skipped (case helper checks present) |
| Actual N15 lifecycle |5 native tests each CPU/CUDA; eight-day pipelines each, exactly one daily hydro and zero-time T/day; closure/rollback/restart pass |
| Historical c76 standalone |periodic eight-day split restart and wrong-boundary refusal pass CPU/CUDA with source guards enabled |
| Independent geometry |640 root,32 PET,4 cover cases each NumPy/Numba/CUDA;48 upper aliases,32 mixed endpoints,80 adaptive-quadrature cases pass |
| Default compatibility |seeded plant/root/canopy hashes byte-identical; old tests unchanged |
| Memory |cover6,246,595 traced versus6,291,472 declared; daughter allocation below existing estimate; caps unchanged |
| Preservation |28 case files,8 old locks, AGENTS/existing tests unchanged; no upstream edit, installation or trial run/regeneration |

Separate commands/streams/reaped exits are archived. Revisions1–3 were corrected before acceptance. Recovered Claude tool errors and Codex setup errors are in orchestration_errors.json; the standalone harness indexing correction is explicit. No ecological validation, large-domain runtime ranking or full transport acceptance is claimed. Phase3 science remains paused.
