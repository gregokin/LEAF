# Floating daily water and accepted LEAF biology

This separate CPU interface connects the genuine Phase 2 floating event/daily
lifecycle to the unchanged `PlantSession`. The N15 `lifecycle_bridge` and its
qualified source pins remain unchanged. New code/physical acceptance is pending
Root's independent review and execution; this document is not a run receipt.

SYRUP owns W, J/I, surface water, hydraulics and all debits. LEAF owns plants,
existing sparse allocation and growth. The bridge reads and validates actual native
transactions and owns only bounded connection bookkeeping. It never steps, debits,
closes, hands back or publishes either model.

## Source selection

```python
source = fb.bind_source(native_source_profile,
    expected_syrup_digest=frozen_runtime_syrup_package_digest,
    expected_syrup_commit=frozen_runtime_syrup_commit,
    expected_maple_commit=frozen_runtime_maple_commit,
    max_bytes=source_budget, reserve_bytes=other_live_bytes)
```

Only a genuine sealed `native_case_profile.NativeSourceProfile` is accepted. Its
CPU capability, actual selected MAPLE commit and package digests must match; the
owning cold `recheck()` runs, and the bridge hashes the actual imported SYRUP tree
and verifies module origins. Arbitrary SHA strings cannot mint a native profile.
The explicit SYRUP commit label is Root's frozen committed provenance; it does not
independently authenticate a Git archive. Root's four-object runtime manifest must
bind that commit to the exact package bytes and separately pin LEAF and its external
`benchmarks.spatial_mapping` helper. No self-referential future SYRUP hash is baked
into LEAF. `source.report()` reports authenticated Source, not qualification.

Cold package reads happen at binding and explicit final
`source.recheck(max_bytes=source_budget, reserve_bytes=other_live_bytes)`. Both
admit the same 4 MiB cold Source allowance before package I/O. Calls between them
check the closed captured module/alias, function/code, class/method and proof-object
roster cheaply: the actual native producer/mint/withdrawal/hold registries, recharge
plan and cursor checks, and the floating estimator/shared transpiration term operands
are included. Registry contents evolve under their owning Native mints; no historical
parent census is retained. This is a finite dependency check, not arbitrary mutable
Python state authentication.
Root keeps the immutable runtime files/currentness controls throughout execution.
Only genuine Dense NumPy CPU is supported; there is no NumPy/Numba/CUDA fallback.

## Visible transaction sequence

```python
binding = ps.DayBinding(run_digest, nominal_origin_date)
connection = fb.connect(source, lineage, accepted_boundary, day_binding=binding)
plants = ps.PlantSession.start(base, config, anchor=nominal_midnight,
    day_binding=binding, initial_resource=connection.resource_cursor)

opening = native_bridge.open_day(lineage, accepted_boundary, ...)
result = native_bridge.day_action(opening, ...)
# If required, complete the genuine storm/recession and use the FINAL opening
# and day_after_storm result. There is no LEAF water operation during transport.
fb.begin_day(connection, opening, leaf_resource_cursor=plants.resource_cursor)
current = result.state                     # genuine completed interevent
plan = plants.prepare_daily(day_context, terrain_pet=shared_pet_mm_day)
if day_context.transpiration_enabled:
    read = fb.read_availability(connection, current,
        leaf_resource_cursor=plants.resource_cursor, max_bytes=..., reserve_bytes=...,
        available_bytes_fn=actual_host_reader)
    proposal = plants.propose_withdrawal(plan, read)
    request = fb.request_depth_m(read, proposal.proposed_m3,
                                 max_bytes=..., reserve_bytes=...)
    outcome = fd.request_floating_withdrawal(current, request_m=request,
        **fb.request_args(read, connection, current),
        available_bytes_fn=actual_host_reader, reserve_bytes=...)
    feedback = fb.accepted_withdrawal(connection, proposal, outcome,
                                      max_bytes=..., reserve_bytes=...)
    staged = plants.stage_day(proposal, feedback)
    current = outcome.state
else:
    staged = plants.stage_dry_day(plan)
closed = fd.close_floating_day(current, expected_parent=fd.expect_parent(current), ...)
accepted_boundary = native_bridge.hand_back(lineage, closed.state)
plants = plants.publish(staged)
fb.commit_day(connection, current, closed.state, accepted_boundary, session=plants)
```

Keep the native boundary live through `begin_day`: Native's own installed-hold to
exact-parent proof uses weak references. Calling `begin_day` after hydrology has
returned `interevent` is supported; pass the opening, not its successor. A matched
full-storm route may replace an opening that had no read/debit. A read or accepted
feedback cannot be silently abandoned. The caller must check `result.outcome` and
complete all actual native storm/recession work, as in the existing owning API.

One full nominal day and at most one T request per day are supported. The original
daily CSV contract applies: the weather date drives biology; native wind/rain/
recession seconds remain additional physical time. They are not rounded into
datetimes or turned into biological catch-up. Growth with T off remains refused by
unchanged `DayContext`; T on/growth off remains a real debit without growth credit.

## Native proof and explicit cumulative bookkeeping

`FloatingDailyState.request_sequence` is per-day and resets to zero. No native
lifetime T counter is invented. `connection.cursor` is a typed `BridgeCursor`:
its connection hash, next nominal ordinal, requests consumed and revision are
explicit bridge accounting. The legacy `ResourceIdentity` fields presented to
`PlantSession` project those bridge labels; `lop_sha256` is the stable connection
identity, not a claim that the evolving native column/operator stayed unchanged.
The actual water parent digest, physical time, native cursor and genuine parent
capability are separately validated. Property contexts may change after wind while
the native lineage, original cold anchor and declared physics configuration remain
the same. Foreign/skipped days and unhandled native requests, including zero-volume
requests, refuse.

Availability depth and actual/unmet depths become volumes exactly once with the
native cell area. Aggregate root requests are converted back to depth once. Shapes
are `[L,N]`, resources are `layer*N+row*nx+col`. The proposal's actual root map must
match the native shape/area/backend. MASY also checks the complete plant Domain,
origin and periodic flags against the actual MAPLE/SYRUP geometry before transport.

Native's public `verify_floating_withdrawal(parent, *, expected_parent, outcome)`
must prove the genuine outcome and exact parent instance, including equal-digest
alternate-branch refusals. The bridge then checks zero time, exact shared J/I and
surface/faces, context/cursor/occurrence, revision and per-day sequence, producer
receipt, unchanged non-T ledger fields, actual native T increments/term counts,
per-resource W debit/request closure, inactive cells, finite actual/unmet/scale and
global accounting. Each per-resource residual uses the same Native global-scale
bound; no stricter per-cell comparison law is claimed. Comparisons use the existing
Native volume-roundoff law and
owning withdrawal term count; no separate tolerance or biological response law is
introduced.

## Memory and failure

`estimate_bridge_bytes` composes the existing LEAF conversion model with bounded
validation scratch. `read_availability` additionally composes the native read's
actual estimate; admission precedes its arrays. Other concurrently live LEAF
populations, old/new native W, request, native feedback arrays and accepted volume
feedback must be in the caller reserve of every operation. Equal cap values in two
models do not count each other's live arrays. Root's coordinator must include the
actual LEAF resident arrays in native operations and actual water state in LEAF's
reserve; no `P*N` or `P*L*N` tensor is allocated here.

The bridge retains one open day's context, scalar opening/plan identities, a weak
read, the actual post-debit state and accepted volume feedback, never a history list.
The accepted feedback holds **two `L*N` float64 arrays (`16*L*N` bytes)** until
`commit_day`; callers must reserve them during staging, native close/handback and
publication. It retains neither the proposal/uptake plan nor the native outcome's
actual/unmet/scale arrays, and releases its converted-request reference on acceptance.
The read-parent proof is weak. An externally retained read still owns its native
ExpectedParent capability, supply array and any not-yet-accepted converted request;
callers retaining reads/proposals/native outcomes own those additional references
and their reserve. The caller must hold the actual pre-debit parent through feedback
validation. A dropped read before acceptance refuses conservatively; there is no
abandon/recovery API in this profile.
`pending_feedback` exposes the actual validated feedback if later biological
staging/close fails. Retry that staged work, never debit again. Only `commit_day`
advances consumed bridge bookkeeping after real close/handback and LEAF publication;
failure retains the previous committed cursor and truthful pending evidence.

There is **no combined physical restart or floating bridge cursor codec**. Connection
records cannot be copied/pickled/restored; connecting to prior daily/T history
refuses. The existing native vegetation checkpoint and LEAF biology checkpoint are
separate owning interfaces, not an accepted combined migration.

## Focused checks (author: not run)

`tests/test_floating_lifecycle_bridge.py` provides seven Native-free metadata methods
and six genuine native methods. Root supplies the actual frozen Source profile,
bundle/digest/commits and unchanged existing SYRUP helper Source closure. Native
methods exercise actual day/debit/feedback/close, second-day native sequence reset,
an explicit no-T day, unhandled zero-volume T, genuine-outcome refusal and exactly
once consumption, Source dependency rebinding refusals and release of the native
outcome/proposal/request while the two accepted feedback arrays remain accessible.
The new fixture explicitly declares both catalog root_layers and root-fraction vector
lengths from the genuine Native column. Root chooses the bounded execution recipe
and records results.
