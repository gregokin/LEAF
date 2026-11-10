# Verification — daily water uptake and weekly growth

All commands ran from `/home/okin/leaf` using
`/home/okin/MAPLE/.venv-cupy/bin/python`; no installation or environment creation.
The selected GPU was `GPU-b02858c1-29c3-9ea9-33b1-831c33f8cf75` (GTX1080Ti).
Independent process receipts preserve exact argv, cwd, PID/start identity,
environment, hashes, timestamps, separate stdout/stderr, return code and reaping.

## Commands and evidence

- Final tests: `python -m unittest discover -s tests -v`, CPU-only and actual GPU.
  `tests_cpu_final.*`:36tests passed,2expected skips. `tests_gpu_final.*`:36tests
  passed,1unavailability-only skip; CPU/GPU geometry and water/growth parity ran.
- Actual full runs: `run_long_suite.py --backend gpu --tag final` (6cases) and
  `--backend cpu --tag final --select june_hectare_3layers_roots3 june_hectare_3layers_roots5`.
- Same numerical workload with declared pool budget:
  `run_bounded_suite.py --backend gpu --tag bounded` (6cases). The bounded wrapper
  only changes cache policy; it refuses an inherited lower nonzero cap. Exact
  per-case CLI commands are in each `<case>.launch.json`, logs and exit receipt.
- Independent final check: `python -B verify_results.py`, captured by
  `result_checks_final.*`; exit0. `result_verification.json` verifies all14cases,
  all comparisons, all30annual audits/case, budgets, byte/source identities and
  unchanged caps. Report rendered by `write_report.py` from this verified record.
- `preservation_and_process_checks.json`: original root AGENTS hash preserved,
  protected baseline files unchanged; all28 recorded process identities ended.
- Independent read-only reviewer accepts all14raw records and rendered tables.

## Results and limits

GPU pool-budget runs complete30years in23.8–95.2seconds across the fixtures;
observed rounded live peaks50.3–254.8MiB, sampled reservations548.8–1429.9MiB.
GPU host RSS488–541MiB. CPU hectare references139.6/263.5seconds integration.
Largest final absolute residuals: water1.3e-11m3, plant/grid1.46e-11m3,
biomass3.91e-11kg. No balancing corrections. Two pending uptake days retained.

CuPy counters exclude CUDA context/other allocators; reserved counters are not a
VRAM residency trace. Default unlimited caching is preserved as separate evidence.
The report includes setup, whole-process time, stages, rates, source hashes,
fixtures, policy comparisons and reproduction commands. This is a simple
synthetic uptake/growth workload; evaporation, drainage, full demography,
canopy export and MAPLE/SYRUP/MASY event transport are excluded.

## Classified earlier failures

- Revision1: `interrupted`, output-token cutoff followed by900-second execution
  bound; exit143, reaped, raw partial output preserved, no files installed.
  Rate-limit event status was `allowed`, not a usage refusal.
- Revision2 tests: `check_failed`: wrong AdmissionRejected constructor and false
  assertion about wet cells newly reached by growing roots. Corrected through
  revision3. Subsequent final tests pass. No unresolved failure or rate limit.

Primary evidence directory: `agent_handoffs/tasks/daily_water_30year_20261030/`.
Results: `benchmarks/results/daily_water_30year_20261030/`.
