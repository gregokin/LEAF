# Pure growth revision3 — enforce the aggregate closure flag

Resume exact session f071b8f7-f17b-4046-88f9-25ed8aba25ee. Same isolated candidate,
five-file allowed scope, automatic relevant sharing/corrections authorized.
READ this prompt and candidate/** only; Read/Edit/Write only, no execution,
Bash, agents, network, settings, installs, Git or external reads.

Codex independently ran the revised53 tests on CPU (2CUDA skips) and actual
CUDA (all53 pass). NumPy/Numba array-retention profiles and actual raw CUDA live
profiles up to200000 plants fit their declared array scope. One initial CUDA
profiler forgot the required explicit device argument; corrected review tooling
binds the already-selected current device and preserves the failed harness log.
That was Codex's harness issue, not a product failure. Tests remain NOT RUN by you.

One small review correction remains: grow checks finite aggregate values, then
sets ledger.population_closed = abs(resid)<=tol but returns the result even if
that flag is False. The requirement was to ENFORCE closure before returning.
Add a GrowthDataError refusal when the finite aggregate residual exceeds its
existing bound; never widen the tolerance or return a failed closure as success.
Keep the successful numerical results/signatures/units/backend behavior unchanged.

Add a meaningful fault-injection regression wrapping the REAL tiny NumPy kernel
and perturbing one aggregate packet total while leaving valid per-plant outputs
and error counters unchanged. Verify the wrapper refuses the inconsistent
aggregate and preserves inputs. This tests the public conservation guarantee,
not a claim that a natural scientific input already produced that mismatch.
The normal hand-calculation and CPU/CUDA paths must keep passing.

Actual edits should be only leaf/growth.py, tests/test_growth.py, report.md and
at most the corresponding doc explanation. No unrelated refactor or parameter
change. Another isolated writer is implementing the session against your frozen
revision2; the new guard must not change its API. Save, report all tests NOT RUN,
and leave uncommitted for independent Codex verification.
