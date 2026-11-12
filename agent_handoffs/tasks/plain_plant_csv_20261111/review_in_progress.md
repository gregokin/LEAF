# Independent review in progress (not acceptance)

Reviewed initial candidate state/checkpoint/recording/CLI changes read-only while Claude continues. Design correctly emits header-first plain CSV, preserves legacy read compatibility, verifies companion SHA binding and context, checks report/path collisions, and publishes checkpoint pairs inside the existing atomic directory.

Finding to resolve before acceptance: checkpoint introduces PLANT_METADATA_LEDGER_BYTES = 3 * 16384 as a fixed conservative estimate, but state provenance is not bounded to 16 KiB. Rich user/generator provenance can make companion JSON arbitrarily larger; account its actual serialized size before admission or otherwise bound/check it without silently dropping provenance or increasing caller limits. Preserve small-budget admission and add a meaningful large-provenance check.

Potential minor consistency check: csv.n_bytes is emitted but currently not verified (SHA covers actual CSV). If retained as validated metadata, check its integer type/value against exact bytes; do not let tests merely mask row/SURFACE errors behind a stale size/hash.

Final scope, tests, docs and preview checks remain pending the completed candidate.

Scope correction for next revision: the existing preview helper had no native rerender CLI, only the earlier operator used render() directly. The prompt overstated an existing rerender path. Remove the newly added --rerender-from feature, its extra functions/constants/tests/docs; retain only the metadata-pair output/preflight/provenance/readback updates to the existing initializer. This task changes serialization, not image tooling features. No new image or initialization run is needed.
