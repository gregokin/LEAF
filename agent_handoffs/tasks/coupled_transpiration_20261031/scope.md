# Coupled transpiration scope

User authorizes connecting current LEAF growth to SYRUP's separately checked
`feature/subsurface-hydrology` branch. HEAD and uncommitted source hashes are
recorded in baseline.json. SYRUP owns physical water; LEAF owns plants and sends
root-weighted potential demand, then grows from actual transpiration.

Claude is sole author for candidate implementations: benchmarks/daily_water_growth.py,
leaf/__init__.py, leaf/coupling.py, tests/test_coupling.py,
native/maple_syrup/subsurface_transpiration.py, native/test_transpiration.py,
and docs/coupled-transpiration.md. Codex owns baselines, dispatch, installation,
independent review/testing/profiling, README and handoff/context records.

The candidate contains only the two generic benchmark modules and their tests;
Claude receives the numerical interface contract, not upstream source, private
briefs, data or any reference ecology. File tools are restricted to the isolated
candidate; no shell, network tools, other files or permission bypass.

After independent qualification, install LEAF files here and the new native
module/test at SYRUP src/maple_syrup/subsurface_transpiration.py and
 tests/subsurface/test_transpiration.py. These are necessary opt-in additions,
not routine upstream setup. Preserve all existing SYRUP code, tests, docs and
handoff controls, its branch/HEAD/dirty work and the other read-only task.
No MAPLE/MASY edits, installs, environments, Git actions, existing historical
artifacts or root AGENTS modifications.

Acceptance: no separate LEAF water store in coupled mode or synthetic recharge;
actual native SYRUP depletion above residual and explicit T budget; species
profiles and competing individuals; correct depth-volume conversion at nonunit
area; daily hydro->T ordering and weekly geometry/growth, no uptake in wind/rain
phases; persistent soil passed across intervals; mask/device/shape/time validation,
transactional refusal, bounded memory, compiled CPU/GPU and real parity; regression
of prior benchmark and SYRUP columns; measured representative grid timing/memory.
Use existing MAPLE environments. No full storm/transport run or new scientific
WUE/climate/demography laws. Restart serialization and moving-ground remapping
remain outside this coupling milestone; verify split-run continuation.
