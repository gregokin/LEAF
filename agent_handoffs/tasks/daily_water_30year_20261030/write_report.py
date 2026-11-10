"""Render independently verified measurements into the local research report."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[3]
TASK = Path(__file__).resolve().parent


def main():
    data = json.loads((TASK / "result_verification.json").read_text())
    assert data["all_checks_passed"] and len(data["records"]) == 14
    records = {r["file"]: r for r in data["records"]}
    cases = json.loads((TASK / "benchmark_cases.json").read_text())
    label = {
        "june_hectare_3layers_roots3": "1 ha, 400², 3 layers, roots 3×",
        "june_hectare_3layers_roots5": "1 ha, 400², 3 layers, roots 5×",
        "june_hectare_2layers_roots5": "1 ha, 400², 2 layers, roots 5×",
        "june_512_3layers_roots5": "1.6384 ha, 512², 3 layers, roots 5×",
        "grass_bound_hectare_3layers_roots3": "Dense grass, 1 ha, 400², 3 layers, roots 3×",
        "grass_bound_hectare_3layers_roots5": "Dense grass, 1 ha, 400², 3 layers, roots 5×",
    }
    gpu_rows, cpu_rows, cache_rows, stage_rows = [], [], [], []
    for c in cases:
        name = c["name"]
        g = records[f"gpu_{name}_bounded.json"]
        d = records[f"gpu_{name}_final.json"]
        gpu_rows.append(f'| {label[name]} | {c["grass_count"] + c["shrub_count"]:,} | '
                        f'{g["integration_s"]:.1f} s | {g["whole_process_s"]:.1f} s | '
                        f'{g["observed_live_peak_bytes"] / 2**20:.1f} MiB | '
                        f'{g["sampled_reserved_peak_bytes"] / 2**20:.1f} MiB |')
        cache_rows.append(f'| {label[name]} | {d["integration_s"]:.1f} s | {g["integration_s"]:.1f} s | '
                          f'{d["sampled_reserved_peak_bytes"] / 2**20:.1f} MiB | '
                          f'{g["sampled_reserved_peak_bytes"] / 2**20:.1f} MiB |')
        stage_rows.append(f'| {label[name]} | {g["daily_s"]:.1f} s | {g["weekly_s"]:.1f} s | '
                          f'{g["audit_s"]:.2f} s | {g["wall_ms_per_model_day"]:.2f} ms | '
                          f'{g["model_years_per_wall_min"]:.1f} |')
        cf = f"cpu_{name}_final.json"
        if cf in records:
            r = records[cf]
            cpu_rows.append(f'| {label[name]} | {r["integration_s"]:.1f} s | {r["whole_process_s"]:.1f} s | '
                            f'{r["process_peak_rss_bytes"] / 2**20:.1f} MiB | '
                            f'{r["integration_s"] / g["integration_s"]:.1f}× |')
    gpus = [r for r in data["records"] if r["file"].endswith("_bounded.json")]
    rss_min = min(r["process_peak_rss_bytes"] for r in gpus) / 2**20
    rss_max = max(r["process_peak_rss_bytes"] for r in gpus) / 2**20
    residuals = {name: max(abs(r[name]) for r in data["records"])
                 for name in ("water_residual", "flux_residual", "biomass_residual")}
    root = "../agent_handoffs/tasks/daily_water_30year_20261030"
    results = "\n".join([
        "## Results: complete 30-year runs", "",
        "All rows below are actual 10,950-step runs, not extrapolations. Each case ran in a fresh process. "
        "Hardware: one NVIDIA GTX 1080 Ti (11 GiB listed memory), and an Intel Core i9-7900X CPU reference "
        "using one compiled thread. Existing Python 3.12.3, NumPy 2.5.3, Numba 0.67.0 and CuPy 14.2.0 were used.", "",
        "The GPU table uses the existing 1,500,000,000-byte budget for the CuPy pool as well as the array/scratch "
        "admission checks. **Live peak and reserved peak are different measurements:** the first includes all "
        "observed temporary pool allocations; the second includes cached free blocks. Pool limits/counters exclude "
        "CUDA context memory and other allocators. Whole-process time includes imports, startup, initialization, "
        "integration, annual audits and output; loop time includes all weekly growth/maps but excludes annual audits.", "",
        "| Scenario (0.25 m cells) | Plants | 30-year loop | Whole process | Observed live peak | Sampled reserved peak |",
        "|---|---:|---:|---:|---:|---:|", *gpu_rows, "",
        f"GPU host-process RSS peaked at {rss_min:.0f}–{rss_max:.0f} MiB. RSS includes Python, libraries, JIT and "
        "host mirrors; add it separately to device allocation estimates. These are single-run measurements on the "
        "existing workstation, suitable as capacity estimates rather than statistical performance bounds.", "",
        "### Wall time versus model time", "",
        "The daily and weekly columns are accumulated synchronized wall times over all 30 model years. "
        "Weekly time includes growth, invariants, admission, root-map replacement and demand-cache rebuilding.", "",
        "| Scenario | Daily blocks | Weekly updates | Annual audits | Wall/model day | Model years/wall minute |",
        "|---|---:|---:|---:|---:|---:|", *stage_rows, "",
        "### Compiled CPU reference", "",
        "CPU reference runs use identical inputs, numerical fixtures and the full continuous calendar. "
        "The speed ratio compares the measured CPU loop with the GPU loop under the pool budget.", "",
        "| Scenario | 30-year loop | Whole process | Host RSS peak | CPU/GPU wall ratio |",
        "|---|---:|---:|---:|---:|", *cpu_rows, "",
        "### Allocator behavior matters", "",
        "The original runs used CuPy's default unlimited pool cache. Growing root graphs left many differently "
        "sized free blocks cached. Low live-array memory therefore did not establish a small device reservation. "
        "Those original results are preserved separately. The repeat changes only cache retention through "
        "`MemoryPool.set_limit(size=case['max_bytes'])`; it refuses an inherited lower nonzero limit. "
        "No individual, water, biomass, connection or pending uptake is discarded, and the declared cap is unchanged.", "",
        "| Scenario | Default loop | Limited-pool loop | Default reserved counter | Limited reserved counter |",
        "|---|---:|---:|---:|---:|", *cache_rows, "",
        "These are CuPy allocator counters, not a trace of resident VRAM. In particular, the default 512² run's "
        "counter exceeded the card's listed memory on this WSL system; it must not be interpreted as a physical "
        "VRAM measurement. The repeat enforces the pool's allocation budget. Runtime comparisons include normal "
        "run-to-run and host-load variation. Production coupling should explicitly budget persistent state and "
        "reuse scratch/graph buffers or manage unused cache at phase boundaries.", "",
        "### Memory scaling", "",
        "With `N` horizontal cells, `L` layers, `P` plants and `H` horizontal root connections, this prototype's "
        "persistent arrays occupy approximately `40*L*N + 8*N + 184*P + 12*H + 8` bytes, before host mirrors. "
        "The five layered FP64 fields include water, demand, acceptance scale and two cumulative accounting fields. "
        "Plant/candidate buffers and original inputs are included. Graph replacement briefly retains old and new "
        "maps; the admission ledger additionally reserves scratch and host mirrors. Those estimates are separate "
        "from the measured allocation peak in the table. Factorized vertical weights avoid storing a separate "
        "connection list per layer. Memory follows current state and graph size, not 30 years of field histories.", "",
        "These figures cover this uptake/growth workload. Shared MASY soil properties, richer root/allometric "
        "traits, individual demography, canopy export and persistent MAPLE/SYRUP storage need their own additions. "
        "The measurements support feasibility for the specified grids and populations; they do not qualify a "
        "scientific ecohydrology model or establish full coupled-model runtime.", "",
        "## Verification and provenance", "",
        "Fourteen complete runs (six default GPU, six limited-pool GPU and two CPU) represent 420 model years. "
        "Every run retained its full population, completed the required calendar and passed all 30 annual audits. "
        "Independent checks reconcile array byte ledgers, source/wrapper hashes, launch/exit receipts, final "
        "water/biomass budgets and full-run CPU/GPU scalar agreement. Default versus limited-pool repeats match "
        "final water/biomass/flux totals and graph/calendar counters within FP64 tolerances; bitwise identity is "
        "not required. Small CPU/GPU tests compare individual arrays as well.", "",
        f"Largest absolute final residuals across all runs: water `{residuals['water_residual']:.3g} m³`, "
        f"plant/grid uptake `{residuals['flux_residual']:.3g} m³`, biomass `{residuals['biomass_residual']:.3g} kg`. "
        "No residual corrections or balancing fluxes were applied. The final 36-test suite passed in CPU-only "
        "and real-GPU executions; expected unavailable-backend skips are recorded in their logs.", "",
        f"[Verified results]({root}/result_verification.json), [case fixtures]({root}/benchmark_cases.json), "
        f"[runtime identity]({root}/runtime_identity.json), [frozen runtime source hashes]({root}/source_after_correction_04.json), "
        f"[bounded profiler]({root}/profile_daily_bounded.py), "
        "[raw result directory](../benchmarks/results/daily_water_30year_20261030/). "
        "The task directory preserves Claude session records, exact prompts, corrections, stdout/stderr and "
        "per-run process identities/exits. Claude authored the benchmark; Codex reviewed, corrected through "
        "the handoff loop, independently executed and verified it. Root AGENTS.md and upstream models were preserved.", "",
        "## Reproduce", "",
        "From `~/leaf`, use the existing environment. Select an idle GPU UUID for the GPU command. "
        "The bounded profiler preserves the case's byte cap and does not install dependencies.", "",
        "```sh",
        "../MAPLE/.venv-cupy/bin/python -m unittest discover -s tests -v",
        "../MAPLE/.venv-cupy/bin/python -m benchmarks.daily_water_growth \\",
        f"  --cases-json agent_handoffs/tasks/daily_water_30year_20261030/case_sets/june_hectare_3layers_roots3.json \\",
        "  --output /tmp/leaf-30year-cpu.json --backend cpu --years 30",
        "CUDA_VISIBLE_DEVICES=YOUR_IDLE_GPU_UUID ../MAPLE/.venv-cupy/bin/python \\",
        "  agent_handoffs/tasks/daily_water_30year_20261030/profile_daily_bounded.py \\",
        "  --cases-json agent_handoffs/tasks/daily_water_30year_20261030/case_sets/june_hectare_3layers_roots5.json \\",
        "  --output /tmp/leaf-30year-gpu.json --backend gpu --years 30",
        "```", "",
    ])
    path = ROOT / "docs/daily-water-30year.md"
    base = path.read_text().split("## Results", 1)[0]
    path.write_text(base + results)
    print("Wrote report from 14 independently verified full runs.")


if __name__ == "__main__":
    main()
