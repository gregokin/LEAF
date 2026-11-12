# outputs/ — planned output layout (empty placeholder; nothing generated)

No run has been performed, so this directory holds only this note. The layout below is the intended mapping to existing APIs; it is **not** a loader or CLI contract.

| Planned path | Content | Format |
|---|---|---|
| `outputs/plants_{index:04d}.csv` | plant-state snapshots from `leaf.recording.SnapshotWriter(path_template, cadence_days=30.0)` | plain nine-column CSV (same for initial, cadence, event and final snapshots) |
| `outputs/plants_{index:04d}.metadata.json` | companion context for the matching snapshot | time, domain, catalog identity, units, provenance and CSV hash/size; no metadata is embedded in the CSV |
| `outputs/checkpoints/` | complete session restart (`IntereventSession.save`) | restart **format 4**: plain `plants.csv`, `plants.metadata.json`, restart arrays and manifest |
| `outputs/event_plants_{index:04d}.csv` | optional event snapshots from a SEPARATE caller-created writer | same plain CSV, with `event_plants_{index:04d}.metadata.json` beside it |
| `outputs/checkpoints/boundary_{index:04d}` | optional checkpoint per boundary; planning policy `after_each_accepted_transport_and_final`, `overwrite=false` | restart format 4 |

The event template and checkpoint template/policy are **AUTHOR planning choices / caller policy**, not newly implemented loaders. No snapshot or checkpoint has been created.

Reporting cadence

* `cadence_days = 30.0` is an **exact 30-day cadence** (author choice, user selectable), approximately monthly, **not calendar months**. A calendar-month cadence is not implemented unless the caller handles it explicitly.
* Initial (`write_initial` at day 0) and final (`write_final` at day 365) snapshots are explicit. Event snapshots are written separately by the caller (a second writer/template or `leaf.state.write_plant_csv`); `SnapshotWriter` has no event kind.
* If a due time is crossed **inside** wind/rain/recession transport, no plant state exists at that time: it is listed in `skipped_due_times_days` as missed and is **never interpolated or fabricated**.
* Existing files are not overwritten unless the caller sets `overwrite=True`.
* Paths resolve from the case directory.
