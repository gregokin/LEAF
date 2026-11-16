# `leaf.floating_lifecycle_bridge_cuda` — Dense CUDA floating-water bridge (unexecuted by its author)

Same protocol as `leaf.floating_lifecycle_bridge` (which stays the strict NumPy CPU default). Call order per nominal day:

1. `source = bind_source(profile, expected_syrup_digest=, expected_syrup_commit=, expected_maple_commit=, max_bytes=)` once; the profile
   must be a genuine `NativeSourceProfile` with `device_runtime_capability=True` (explicit GPU opt-in; CPU factory refuses it).
2. `connection = connect(source, lineage, boundary, day_binding=)`; start the `PlantSession` (backend `cuda`, same device) with
   `initial_resource=connection.resource_cursor`.
3. Native `open_day` / `day_action` (device + host readers), then `begin_day(connection, opening, leaf_resource_cursor=session.resource_cursor)`.
4. `pools = CudaPools(device_max_bytes, device_available_bytes_fn, host_max_bytes, host_available_bytes_fn, device_reserve_bytes, host_reserve_bytes)`
   (reserves = every other live byte of that pool; never sum pools).
   `read = read_availability(connection, interevent, leaf_resource_cursor=, pools=)` → `proposal = session.propose_withdrawal(plan, read)` →
   `depth = request_depth_m(read, proposal.proposed_m3, pools=)` → native `request_floating_withdrawal(interevent, request_m=depth,
   **request_args(read, connection, interevent), available_bytes_fn=<device>, host_available_bytes_fn=<host>)` →
   `accepted = accepted_withdrawal(connection, proposal, outcome, pools=)` → `staged = session.stage_day(proposal, accepted)`.
   A no-T day: skip the read and use `session.stage_dry_day(plan)`.
5. Native `close_floating_day` + `hand_back` (device + host readers), `session = session.publish(staged)`, then
   `commit_day(connection, interevent_after_T, closed, boundary, session=session)`.

Guarantees: current-device gate before any reader; native metadata checks; resident arrays only (bounded 8-byte scalar packets to the host);
one native debit per day; native comparison bound only; refusal of windowed beds, CPU columns, foreign devices, host proposals, stale
cursors/generations and replays. Native memory refusals keep their classes (`MemoryError` subclasses); the LEAF host cap raises
`lifecycle_bridge.BridgeAdmissionError`. Combined persisted restart is not qualified.

Root qualification (MASY `dense_triplet_gpu_20261115`): actual Dense CUDA
availability, per-cell debit feedback, two daily sequence resets, no-T day,
stale/replay/refusal/pool-admission and release tests passed. The existing CPU
bridge's13 metadata/native tests also passed after backend factoring. The
coupled small8day control completed E/T and weekly growth on the device and
matched the accepted CPU control within owning tolerances. This validates the
engineering connection, not scientific parameterization or combined restart.
