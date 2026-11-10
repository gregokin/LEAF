# Revision 3 — numerical implementation cleared for measurement

The scalar lower-bound admission now precedes supplied-input edge-bound
allocation; the rejection regression verifies no edge calculation or profile
materialization. Profile scratch reserves S*L+32S bytes. Setup H2D totals include
geometry, profile IDs/table and WUE, with timing coverage labeled explicitly.

Independent CPU suite: 47 tests, OK, three backend skips. Actual-GPU suite:
47 tests, OK, one unavailable-only skip. Independent analytical CPU/GPU checks
pass dry layers, finite competing supply, explicit third-profile IDs, uniform
reference and exact persistent-memory accounting.

No numerical blockers remain. Final acceptance awaits bounded full-duration
measurements, result verification and completed documentation. Freeze numerical
source and the profiling wrapper while measurements execute.
