# Interface verification — PASS

Independent [acceptance](tasks/masy_daily_plant_interface_20261112/final_review.md) and
[machine-readable evidence](tasks/masy_daily_plant_interface_20261112/qualification.json).

- Full frozen implementation: CPU659 run/608passed/51skipped; actual CUDA659 run/652passed/7skipped.
- Final actual N15 native suite:13/13 CPU,13/13 CUDA. Native NumPy/Numba/CUDA eightday harness:
  eight hydro+eight T,691200seconds, one weekly growth, actual uptake/closure, disabled-growth no credit,
  late geometry failure rollback, pending-debit resume without replay and closed-boundary restart.
- Installed focused checks:53passed plus36subtests, including pure case-config/helper checks omitted by
  the isolated snapshot. Installed three-day public example passes and writes a biological-only checkpoint.
- Final probes refuse unbound dates, unseen T and changed RNG/patch. Source hashes, memory measurements,
  parity and exact argv/env/streams/reaped exits are in the task. Trial case never run/generated/recalibrated.

Earlier failures are retained and corrected (check_failed). Native CPU pytest initially triggered the
correct no-CuPy source guard; isolated CPU unittest used the existing CPU environment. One revision2
out-of-scope identical-string Edit was rejected without mutation (tool_error); all actual scope audits pass.
No auth/network/rate-limit error or permission bypass. Publication and final preservation checks follow
in separate task records; no historical evidence was rewritten.

Implementation publication verified at `8d6c8849d219ae7e0449217db40de639bab5b590` on origin/main.
[Receipt](tasks/masy_daily_plant_interface_20261112/publication.json); follow-up commit records completion.
