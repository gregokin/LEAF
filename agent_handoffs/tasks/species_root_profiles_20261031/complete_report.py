"""Write measured documentation from independently verified result records."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[3]
TASK = Path(__file__).resolve().parent


def mib(n):
    return n / 1048576


def main():
    v = json.loads((TASK / 'result_verification.json').read_text())
    assert v['status'] == 'pass' and len(v['records']) == 8
    records = {(r['backend'], r['case']): r for r in v['records']}
    labels = [
        ('june_hectare_3layers_roots3', '1 ha, 400²'),
        ('june_hectare_3layers_roots5', '1 ha, 400²'),
        ('june_hectare_2layers_roots5', '1 ha, 400²'),
        ('june_512_3layers_roots5', '1.6384 ha, 512²'),
        ('grass_bound_hectare_3layers_roots3', 'Dense grass, 1 ha, 400²'),
        ('grass_bound_hectare_3layers_roots5', 'Dense grass, 1 ha, 400²'),
    ]
    gpu = [records['gpu', k] for k, _ in labels]
    cpu = records['cpu', labels[0][0]]
    uniform = records['gpu', 'uniform_control_june_hectare_roots3']
    table = ['| Scenario (0.25 m cells) | Plants | Layers | Root radius | 30-year loop | Whole process | Live peak | Reserved peak |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    stage = ['| Scenario | Daily blocks | Weekly updates | Wall/model day | Model years/wall minute |',
             '|---|---:|---:|---:|---:|']
    for (key, label), r in zip(labels, gpu):
        table.append(f"| {label} | {r['plants']:,} | {r['layers']} | {r['roots']}× canopy | {r['integration_s']:.1f} s | {r['whole_process_s']:.1f} s | {mib(r['live_peak_bytes']):.1f} MiB | {mib(r['reserved_peak_bytes']):.1f} MiB |")
        stage.append(f"| {label}, {r['layers']} layers, {r['roots']}× roots | {r['daily_s']:.1f} s | {r['weekly_s']:.1f} s | {r['wall_ms_per_model_day']:.2f} ms | {1800/r['integration_s']:.1f} |")
    residual = {k: max(abs(r[k]) for r in v['records']) for k in ['water_residual', 'flux_residual', 'biomass_residual']}
    report = f'''Final CPU and actual-GPU suites each pass 47 tests, with respectively three
and one expected backend skips. Independent analytical checks cover dry upper
and lower layers, finite competing supply, all-dry cells, an explicit third
profile and a uniform reference. Invalid inputs, early admission refusal,
weekly growth, split-run equivalence and heterogeneous-water CPU/GPU parity are
also tested. The completed 30-year CPU/GPU reference agrees on calendar counts,
water and biomass within the recorded floating-point tolerances.

All rows below are complete 10,950-day runs, performed in fresh processes.
They retain every plant and perform every scheduled weekly rebuild. Populations
come from the [Jornada capacity fixtures](jornada-capacity.md): 36,305 geometric
individual equivalents per hectare in the June example, 59,483 over the larger
512² domain, and a constructed dense-grass bound of 547,588 per hectare. These
are capacity fixtures rather than a census or a measured joint cover maximum.

Hardware/environment match the historical benchmark: one selected idle NVIDIA
GTX 1080 Ti, Intel Core i9-7900X CPU reference using one compiled thread, Python
3.12.3, NumPy 2.5.3, Numba 0.67.0 and CuPy 14.2.0. No environment or package was
created. Numerical source and the measurement wrapper stayed frozen.

{chr(10).join(table)}

Loop time includes daily accounting and weekly growth, admission, graph
replacement and demand-map scattering; annual audits are excluded. Whole-process
time includes imports, startup, initialization, audits and output. Each figure
is one measured run on this workstation, not an extrapolation or a performance
guarantee.

{chr(10).join(stage)}

The one-thread CPU reference for the first row took {cpu['integration_s']:.1f} s
of integration and {cpu['whole_process_s']:.1f} s for the whole process, with
{mib(cpu['rss_bytes']):.1f} MiB peak host RSS. Its GPU integration was
{cpu['integration_s']/gpu[0]['integration_s']:.2f}× faster in these runs. GPU host
processes peaked at {min(mib(r['rss_bytes']) for r in gpu):.1f}–{max(mib(r['rss_bytes']) for r in gpu):.1f} MiB RSS;
RSS includes Python, libraries, compilation and host mirrors.

The equal-layer compatibility control also ran all 30 years. It matches the
historical v0.1 calendar, final connection count and water/biomass totals within
tolerance, and took {uniform['integration_s']:.1f} s with
{mib(uniform['live_peak_bytes']):.1f} MiB live GPU peak. This is close to the historical 23.8 s equal-layer result. The weighted cases
are slower in these measurements and follow different water-competition,
growth and graph trajectories. Earlier equal-layer times should therefore not
be used as estimates for the weighted configurations. These single runs do not
identify the cause of the runtime difference or isolate a stable percentage
implementation overhead; kernel and work-distribution profiling would be needed.

All eight full runs pass annual and final conservation/profile/calendar checks.
Maximum absolute final residuals were {residual['water_residual']:.2e} m³ for the
water budget, {residual['flux_residual']:.2e} m³ for plant/grid uptake agreement,
and {residual['biomass_residual']:.2e} kg for biomass. No population was truncated,
no state was discarded and no resource limit was raised.

### Memory measurement policy

The same 10-million-connection and 1,500,000,000-byte limits remain in force.
The latter also bounds the CuPy pool. **Live peak** comes from malloc/free hooks
and includes observed temporaries; **reserved peak** is sampled after weekly
updates and includes cached free blocks. These are distinct pool measurements,
excluding the CUDA context and other allocators; they are not a trace of physical
VRAM residency. Pool reservations can remain much larger than live arrays.
Host RSS and the conservative admission ledger are reported separately. All
hook allocations were released at process end, with no unknown free events.

### Reproduction and evidence

Use the existing environment and select an idle GPU explicitly. This repeats
the first measured case with the original memory cap:

```sh
CUDA_VISIBLE_DEVICES=GPU-b02858c1-29c3-9ea9-33b1-831c33f8cf75 \\
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \\
NUMBA_CACHE_DIR=/tmp/leaf-root-numba-cache CUPY_CACHE_DIR=/tmp/leaf-root-cupy-cache \\
CUDA_CACHE_PATH=/tmp/leaf-root-cuda-cache \\
../MAPLE/.venv-cupy/bin/python -B \\
  agent_handoffs/tasks/species_root_profiles_20261031/profile_daily_bounded.py \\
  --cases-json agent_handoffs/tasks/species_root_profiles_20261031/case_sets/june_hectare_3layers_roots3.json \\
  --output /tmp/leaf-species-roots.json --backend gpu --years 30 --report-every 365

../MAPLE/.venv-cupy/bin/python -B -m unittest discover -s tests -v
```

Raw outputs are in the [result directory](../benchmarks/results/species_root_profiles_20261031/).
The [independent result verification](../agent_handoffs/tasks/species_root_profiles_20261031/result_verification.json)
contains exact timing/memory values and comparison outcomes. The
[task evidence](../agent_handoffs/tasks/species_root_profiles_20261031/) records
immutable baseline snapshots, exact source hashes, the real Claude session,
corrections, separate stdout/stderr and process launch/exit receipts. The
[measurement manifest](../agent_handoffs/tasks/species_root_profiles_20261031/measurement_source_manifest.json)
identifies the executed numerical sources and wrapper.
'''
    target = ROOT / 'docs/species-root-profiles.md'
    marker = 'Final full-duration measurements and their independent checks are in progress.\nThey are recorded separately from the historical equal-layer benchmark.\n'
    s = target.read_text()
    assert s.count(marker) == 1
    target.write_text(s.replace(marker, report, 1))
    print('Completed measured report from eight independently verified full runs.')


if __name__ == '__main__':
    main()
