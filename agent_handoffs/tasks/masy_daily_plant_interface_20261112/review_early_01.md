# Early independent review — not acceptance

Review is against the frozen first bridge draft in
`public_evidence/early_review_01/lifecycle_bridge.py`. Claude revision 1 is still
writing its candidate; this finding must be checked against its completed tree
and corrected before acceptance.

## F1 — high: actual resource-parent instance is lost

`lifecycle_bridge.AvailabilityRead` retains only `ResourceIdentity` scalars.
`accepted_withdrawal` checks successor scalar counters and the receipt label,
but retains/validates no live capability linking the read to its actual parent.
Creating a new `expect_parent` at each availability read authenticates that read,
but does not authenticate a later withdrawal against that same instance.

Independent reproducer `probe_branch_identity_01.py` uses the actual published
N15 factory and daily lifecycle. Two pure daily transitions from one original
parent produce distinct genuine instances with equal digests. Availability is
read from A; the native withdrawal is performed on B. Bridge feedback succeeds
and reports 1.5e-6 m3 instead of refusing the wrong parent. Output:
`probe_branch_identity_01.stdout.json`; assertion failure and traceback:
`probe_branch_identity_01.stderr.log`. Exec session 47607 completed and was reaped
with exit 1 (a deliberately failing specification assertion).

Required: preserve an actual live parent binding through plan and feedback,
validate the native candidate belongs to the exact supplied successor, and give
MASY a public safe request-binding check before its visible withdrawal call.
Keep serialized identity separate from live capabilities and explicitly rebind
after restore. Add real equal-digest alternate-instance controls, not only
different digest/counter tests.
