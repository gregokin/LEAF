# Phase 1 session revision 3 — bound coexistence during root rebuild and restart

Resume EXACT session ecdc127b-6348-4df5-8c50-87404f7afa8e. Same authority, sole
implementation writer, seven-file write scope and Read/Edit/Write tools as revision 2.
Read only this prompt and candidate/**; no execution, network tools, settings,
agents, Git, installs or outside reads. Preserve existing source/physics/limits.
Allowed writes: leaf/interevent.py, leaf/checkpoint.py, tests/test_interevent.py,
docs/phase1-interevent.md, examples/phase1/run_interevent.py, README.md, report.md.
Codex executes checks, integrates and publishes whole phases only.

D2 completed/reaped with scope violations/tool errors 0. Codex independently ran
361 full CPU tests (341 pass/20 GPU skips), all 84 session tests on actual CUDA,
all eight independent failure guards, exact ten-day native/leap/midweek restart,
the CUDA snapshot array-data lower-bound and actual daily-step memory profiles.
All pass. Those are Codex checks; yours remain NOT RUN. Do not rewrite passing
science or other component modules. No whole-phase acceptance yet.

Final code inventory reveals coexistence omissions to correct, even though the
daily/snapshot global profiles now fit. These are source-inventory findings,
NOT a claimed measured global peak overrun during growth or restart:

1. During a normal growth-day step, work.D/U are fresh D_new/U_new while cur.D/U
   remain referenced by the accepted boundary. _growth_update's live_other uses
   _persistent(cur) + ... + 16*Pl: that last term covers B2/E2, leaving the fresh
   pending D/U (16*Pl) unrepresented during _build_geometry's remaining-budget
   calculation. Old/new A/T coexist as well. A flush may alias the accepted D/U;
   conservative double-counts are fine if documented. Count distinct candidates
   or a justified conservative bound, not a guessed padding constant.
2. In _build_geometry, CUDA area_h is still live on the host after xp.asarray;
   budget subtracts only 8*Pl for the device area. Count BOTH host and device area
   through the root builder (or explicitly release the host array before then).
   Bh/Eh host copies and the materialized PlantState are separate live ownership.
   Audit all callers (initialization, growth, deferred season, restart verification)
   and the final reset accumulators / old-new native/root coexistence as well.
3. load_session retains CSV/NPZ bytes, NPZ/BytesIO ownership, the entire decoded
   arrays dict, plants and live_rows while it initializes session arrays, uploads
   water/CSR and rebuilds roots for verification. The early checkpoint_load_read
   check alone doesn't reserve those still-live buffers during later _setup,
   _build_geometry and RootMap allocations. Explicitly release redundant objects
   when safe or carry a conservative serialization/decode reserve through every
   allocating restore stage. Caller reserve and original max_bytes must persist
   unchanged in the final session (no hidden limit increase/reduction of state).
   Count native resident state during loading too. Keep hash/type/root/native
   binding checks strict; restart must remain equivalent with all pending state.
4. memory.peak_estimate_bytes must be the maximum actual admitted STAGE TOTAL,
   not a remaining *_builder_budget capacity. A capacity near max_bytes can mask
   an undercount and falsely report a near-cap peak. Report capacities separately;
   include actual builder admitted peaks from root-map provenance + all external
   resident/candidate bytes in the stage ledger. Clearly label conservative peaks
   and excluded context/pools. Do not replace measurement with a cap-sized number.

Add focused boundary/refusal regressions (before builder/upload work, no old-state
mutation) and a restart with enough serialized state to exercise coexistence.
Do not mirror only arithmetic constants in tests: spy on real allocations / the
passed remaining budget and demonstrate a too-small global cap refuses safely.
Preserve all previous passing results, units, API and single-accounting semantics.
Update docs/report truthfully; no claim tests ran. Save incrementally, uncommitted.
