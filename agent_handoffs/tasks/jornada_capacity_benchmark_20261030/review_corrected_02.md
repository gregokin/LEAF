# Corrected prototype review

Verdict: accept this bounded synthetic benchmark and capacity follow-up.

Claude authored the standalone draft and 13 correction edits. Codex installed
only the three allowed files, reviewed/verified them, and changed the fixed GPU
warm-up population from 9 to 1,024 under unchanged caps. The separate allocation
measurement wrapper is Codex verification instrumentation.

Independent read-only reviewer confirmed all four blockers are addressed:
admission before generation/H2D, full-population demand audit, complete GPU
operation warm-up, and partial/unavailable status instead of false success.
Root additionally reviewed deliberate clump contention and whole-case timing.
The allocation hook matches the installed CuPy API; observed unknown-free
count is zero and live rounded pool allocation returns to zero.

All 20 tests passed with actual GPU parity. All 11 benchmark cases passed without
rejection, partial backend or numerical failure. Capacity and geometric formulas
were independently recomputed. Source and environment identities are recorded.

Accepted limits: synthetic uniform weights and separable layers; no physical
water allocation or growth law; circular finite geometry; one sampled cover
field rather than complete native publication; single-thread CPU comparison;
five kernel repeats and one build per case; pool peak excludes context and other
allocators. No whole-model memory fit, speedup or ecological validation claimed.

No source implementation changes followed final tests and measurements.
