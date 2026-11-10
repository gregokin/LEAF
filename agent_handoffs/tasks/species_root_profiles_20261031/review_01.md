# Revision 1 — correct

The numerical CPU/GPU core uses identical profile fractions in requested demand
and accepted-uptake gather. Independent CPU analytical checks pass, including
opposite profiles under heterogeneous supply, finite sharing, all-dry cells,
explicit third-profile IDs, the uniform reference and exact live byte formula.

CPU suite: 43 tests, two failures, three GPU-unavailable skips. Both failures
come from test-only water changes followed by a historical budget assertion.
Classification: check_failed. Production budgets must remain unchanged.

Independent review also found the strict geometry-generator field mismatch,
profile allocations before aggregate admission, omitted profile-sized scratch,
omitted profile transfer accounting and unnecessarily required unused class
names. Revision 2 addresses this bounded list. Raw tests/records preserved.
