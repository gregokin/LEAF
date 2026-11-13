# Correction revision 5 — test and handoff only (no implementation changes)

Same session 37847909-f17c-4674-af37-323c9f8c6995. Revision4 is reaped, structured success. Read review_04.md.
Allowed READS: prior session roots + review_04.md and this prompt, under task masy_daily_plant_interface_20261112; candidate source is readable. Tools ordinaryRead/Edit/Write ONLY. No execution, Bash, agents, network, installs, Git, upstream/case/orchestration changes.
Allowed WRITES ONLY TWO candidate files:
1 tests/test_lifecycle_bridge.py
2 docs/masy-daily-plant-interface.md
Do not edit implementation, other tests or interface_report.md in this revision; report in final response (Codex archives it).

659-test CPU suite passes (608passes,51skips), eight-day actual CPU/Numba finalharness passes with nonuniform sharedPET, weekly growth, latefailure rollback, pending-debitresume without replay, closedboundarycheckpoint. Final native13-test CPU suite has ONE test-only failure: test_visible_daily_sequence_counts_water_closure_clock_and_shared_pet scans dir(obj) for 'water' and flags the legitimate METHOD stage_connect_water. It is a metadata binding method, not an inventory. Correct the test to inspect stored data/actual behavior instead of banning method names containing water. Keep all real native factory/source/clock/T/closure checks and guards. Do not remove meaningful coverage or change production to satisfy this heuristic.

In docs: move reserve = session.config.reserve_bytes + session.resident_bytes() INSIDE perday loop; population/growth may change. Status should point to ../agent_handoffs/tasks/masy_daily_plant_interface_20261112/final_review.md and qualification.json for independent Codex evidence, distinguish tested interface mechanics vs uncalibrated ecology; no author execution claim, no assumption finalacceptance exists yet. Correct load_biology signature to include existing day_binding_check=True if retaining that argument. Keep all public signatures, ownership, units/shapes/backends/crashwindows. Native Sourcepair is explicitN15 notlivecheckouts. Mention caller-held physical/native state must be reserved on LEAF and LEAF bytes on SYRUP via its preparation reserves; supplied same cap alone is not accounting. No new memory/code/law changes; author checks NOT RUN.

Phase3 scientificwork and trialcase preserved. Leave uncommitted, same automatic workflow. Only2files, no codechanges.
