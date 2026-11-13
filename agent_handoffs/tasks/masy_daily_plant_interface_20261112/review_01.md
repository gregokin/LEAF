# Independent review of revision 1 — CORRECT

Candidate identity: candidate_01_manifest.json; immutable reviewed tree public_evidence/review_01.
Baseline LEAF 9fd4bbb; SYRUP N15 a5b8b18 / 82e9c011; actual MAPLE9f fbc504bb.
Claude revision 1 exited 0, structured success, reaped; scope audit has no violations.
No implementation is installed or accepted. C1 growth/plant_pet/biocalendar are byte-identical to reference.

## Confirmed defects

1. HIGH — lifecycle_bridge.read_availability / accepted_withdrawal lose the actual resource parent instance.
   Actual N15 states A and B from the same pure daily operation have distinct identity but equal digests. Read A,
   withdraw B: feedback accepted. Required wrong-parent refusal fails. Reproducer: probe_branch_identity_01.py,
   stdout JSON records true; deliberately failing assertion exit1. Frozen source hash in early_review_01_source.json.
2. HIGH — accepted_withdrawal does not bind the native candidate to new_state.payload.
   Real withdrawals q and 2q from one parent with the same label; pair successor(q) with feedback(2q): accepted.
   SYRUP accepted T=1.5e-6 m3, LEAF reported accepted=3e-6 m3. Reproducer probe_result_identity_01.py, exit1.
3. HIGH — PlantSession._make token omits plant/geometry identity.
   Two starts with the same explicit session_id, date, catalog and plant count but DIFFERENT IDs/biomass get equal
   tokens. prepare on one, propose on the other: accepted. probe_biology_identity_01/02.stdout.json confirms.
4. HIGH — initialization ignores the caller reserve/global admission.
   SessionConfig reserve_bytes=max_bytes+1 is accepted by PlantSession.start. Geometry construction receives reserve=0.
   probe_biology_identity_02.stdout.json confirms. No complete peak ledger accounts retained candidate geometry,
   all plans/feedback, cold checkpoint buffers and constructor pre-admission yet; these need measured/tight-cap tests.
5. Interface omissions — the wrapper deliberately omits Phase2/3 stress/annual/RNG/reproduction counters and handlers;
   'declared_none' can assert away actual active catalog laws. This does not meet preservation/exposed-boundary and
   biological-checkpoint requirements. Existing mechanics are available: expose/reuse them without new laws or params.
6. Timing/identity omissions — forcing is hardcoded to exactly one complete day, transport only full days, dt=1 in
   demand/integrals; there is no nominal-day/native-daily-ordinal/run binding. This cannot represent excluded subday
   wind/rain/recession time under MASY's required daily sequence and cannot reject a read from the wrong nominal day.
7. Checkpoint gaps — source versions only, not actual sources; external water binding accepts any later counters
   instead of a saved exact pairing; save has no explicit external current-water binding; no save/load byte preflight.
   Keep last consumed T resource distinct from the current physical boundary (which may have been closed after T).

## Actual checks

- Baseline old standalone: 576 run, 531 passed/45 CUDA skipped, exit0.
- New focused CPU: 50 passed, 1 failed, 1 native skipped, 9 subtests passed. The failure is a test using the superseded
  parent after publication and expecting BoundaryError; stale-parent StateError is correct. Test the child for no-pending flush.
- Native CPU: pytest runner first hit the genuine CPU source guard because pytest imported CuPy; environment issue,
  not a physics defect. Isolated unittest in existing .venv: 7 tests passed (includes actual native factory/daily/T/closure).
- Actual GPU1 UUID GPU-b02858c1-29c3-9ea9-33b1-831c33f8cf75, CUDA9f bundle: 7 run, 6 passed, 1 error. The real path
  reached staging/publication and failed only at test np.asarray(CuPy proposal); use explicit test-only host transfer.
- Native harness currently scales only LEAF PET by 1e6: replace with shared physical forcing as required. Scarcity can
  be a separately labeled fixture, not a mismatch of the shared PET.

## Preserved/remaining

No trial case initialization/run/regeneration or parameter changes, no upstream modifications/installations. Phase3 remains
paused as documented. All core Phase2/3 source/tests/locks unchanged. Correct within the new interface scope; do not claim
full coupled-model or ecological validation. Required corrections are specified in prompt_02.md.
