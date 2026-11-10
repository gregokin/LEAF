# Revision 2 — correct

The corrected CPU suite passes 47 tests (three unavailable-GPU skips); the
actual-GPU suite passes 47 tests (one unavailable-only skip). Independent
analytical CPU/GPU checks also pass. The numerical weighting, generated-case
projection, explicit IDs, profile materialization ordering and class selection
are correct.

Read-only review identified three remaining bounded corrections:

1. A supplied-input edge-bound temporary is still allocated before an aggregate
   state/scratch lower-bound admission check.
2. Profile audit scratch reserves only one FP64 row vector although row sums,
   subtraction and absolute-value expressions overlap. Reserve conservative
   row scratch for arbitrary profile counts.
3. The total setup H2D byte report omits the existing WUE trait vector (8P bytes).
   Include it and label timing coverage accurately.

Classification: check_failed (review). No numerical-equation changes requested.
