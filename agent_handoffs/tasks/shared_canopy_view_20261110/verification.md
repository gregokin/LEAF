# Shared canopy verification — 2026-11-10

Executor/reviewer: Codex. **Final revision 2: pass, installed byte-identically.**

## Primary records and commands

Interpreter: `/home/okin/MAPLE/.venv-cupy/bin/python` (existing environment).
`run_check.py NAME ABSOLUTE_CWD COMMAND...` records argv/cwd/PID/start ticks/time,
separate stdout/stderr, exit code and reaping in `NAME.launch.json`,
`NAME.stdout.log`, `NAME.stderr.log`, `NAME.exit.json`.

Environment: candidate or installed absolute cwd plus `/home/okin/SYRUP/src` in
PYTHONPATH; `PYTHONDONTWRITEBYTECODE=1`, `CUDA_VISIBLE_DEVICES=''`; Numba/CuPy
caches under `/tmp/leaf-phase0-*`. No installation/new environment or GPU run.

| Prefix | Command / check | Result |
|---|---|---|
| `revision01_cpu` | `run_cpu_suite.py ABSOLUTE_REVISION_01` | 137 total; 130 pass / 7 CUDA skips |
| `revision01_independent` | `independent_view_check.py verify` | Five legacy fixtures, exact arrays/IDs, single geometry/shared call, purity/import isolation pass |
| `revision01_maple_abi` | Existing independent `phase0_vegetation_20261110/maple_abi_oracle.py` | Native MAPLE validator pass |
| `revision02_cpu` | `run_cpu_suite.py ABSOLUTE_REVISION_02` | 137 total; 130 pass / 7 CUDA skips; 5.460s unittest time |
| `revision02_independent` | `independent_view_check.py verify` | All invariants pass on frozen final tree |
| `revision02_maple_abi` | Same native MAPLE ABI oracle | Pass |
| `installed_independent` | `independent_view_check.py verify` from `/home/okin/leaf` | Installed source passes exact legacy/shared/purity checks |

The suite composes the 90 unchanged root legacy tests with all 47 frozen Phase 0
tests, ensuring candidate changes and new tests are exercised together. The usual
installed reproduction is `PYTHONPATH="$PWD:/home/okin/SYRUP/src"
NUMBA_CACHE_DIR=/tmp/leaf-phase0-numba CUDA_VISIBLE_DEVICES=''
../MAPLE/.venv-cupy/bin/python -m unittest discover -s tests -v` (one shell command
with the environment assignments preceding Python).

`independent_view_check.py freeze` ran against clean baseline code before its
replacement and wrote `legacy_outputs.json`. It covers flat, partial burial,
erosion/full burial with flag and exclude policies, and intact diagnostic height.
Final arrays and machine-readable legacy metadata match those stored outputs;
shared arrays match the adapter by stable ID, including exclusions. Native imports
are blocked during the pure shared-interface check. No plant state is mutated.

## Review corrections and scope

Revision 1 passed every executable check. Codex recorded a **check_failed manual
contract review**: shared metadata described an eroded canopy base as being on the
current surface, while its actual base remains at establishment elevation. The
focused revision 2 clarified metadata/docstrings/guides only. Numerical arrays,
algorithms and all test expectations were unchanged between revisions. Absolute
consistency claims now require matching state/catalog/surface/coupling time;
removed plants are documented as excluded by default.

Both Claude revisions returned structured `success`, `is_error:false`, exit 0,
and were reaped. Scoped Read/Edit/Write tool audits and frozen manifests are in
`scope_revision_01.json` / `scope_revision_02.json`. No tool/auth/network/rate-limit
error occurred. Test children were reaped. `git diff --check` passed after install.

`install_and_lock.py 2` checked all 1,563 original tracked files against the
explicit allowed changes, preserved old locks/archives/AGENTS.md/traits/fixtures/
numerical drivers, and installed six files by frozen hashes. Four geometry/cover
functions and all existing test functions are AST-identical; package initialization
differs only in documentation. All 71 locked SYRUP runtime files and the MAPLE
container hash match previous qualification. See `installation.json`,
`scope_final.json`, baseline and `dependencies/canopy-view.lock.json`.

Compiled CPU/GPU kernels were not changed; no additional GPU performance or
integrated MAPLE/SYRUP run is claimed. Upstream adoption, consumer rasterization
and physical use of canopy geometry remain separate work. The legacy MAPLE CLI
projection is preserved; SYRUP's public geometry source is `canopy_view`.
