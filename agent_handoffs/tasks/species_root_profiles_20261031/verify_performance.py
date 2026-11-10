"""Independently check full-run records for the species-profile extension."""
from pathlib import Path
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[3]
TASK = Path(__file__).resolve().parent
OUT = ROOT / "benchmarks/results/species_root_profiles_20261031"


def close(a, b, rtol=1e-9, atol=1e-8):
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def main():
    files = sorted(OUT.glob("*_weighted.json"))
    assert len(files) == 8, f"need 7 GPU and 1 CPU complete runs, found {len(files)}"
    records, raw = [], {}
    for f in files:
        doc = json.loads(f.read_text())
        assert doc["status"] == "ok" and doc["n_results"] == 1
        r = doc["results"][0]
        assert r["status"] == "ok", (f.name, r.get("error"))
        c, k, a, m, t, rp = (r[n] for n in ("case", "counters", "audit", "memory", "timing", "root_profiles"))
        P, N, L, S = c["grass_count"] + c["shrub_count"], c["nx"] * c["ny"], c["layers"], rp["n_profiles"]
        assert (k["day"], k["weeks_done"], k["pending_days"], k["maps_built"], k["demand_map_builds"],
                k["recharge_events"]) == (10950, 1564, 2, 1565, 1565, 783)
        assert k["population"] == P and k["latched_failure"] is None and a["ok"] and all(a["checks"].values())
        assert r["version"] == "0.2.0" and rp["fixed_across_weekly_growth"]
        assert rp["names"] == ["grass", "shrub"] and S == 2 and rp["layers"] == L
        defaults = {2: [[0.8, 0.2], [0.25, 0.75]], 3: [[0.7, 0.2, 0.1], [0.15, 0.25, 0.6]]}
        expected = list(c["root_layer_fractions"].values()) if c.get("root_layer_fractions") else defaults[L]
        assert rp["table"] == expected
        assert all(len(row) == L and all(math.isfinite(v) and v > 0 for v in row)
                   and abs(math.fsum(row) - 1) <= 1e-12 for row in rp["table"])
        assert rp["ids_dtype"] == "int32" and rp["ids_bytes"] == 4 * P
        assert rp["table_dtype"] == "float64" and rp["table_bytes"] == 8 * S * L
        area = N * c["dx"] ** 2
        w, b = a["water"], a["biomass"]
        assert close(w["initial"], 0.075 * area) and close(w["input"], 7.83 * area)
        assert close(w["total"], w["initial"] + w["input"] - w["overflow"] - w["plant_uptake"])
        assert close(w["grid_removed"], w["plant_uptake"])
        assert close(b["total"], b["initial"] + b["growth"] - b["turnover"])
        H = k["n_edges"]
        persistent = 40 * L * N + 188 * P + 8 * S * L + 12 * H + 8
        assert sum(x["nbytes"] for x in m["live_arrays"]) == m["live_bytes"] + m["graph_bytes"] == persistent
        assert not any(x["name"] == "Rh" for x in m["live_arrays"])
        assert m["rebuild_peak_bytes"] <= c["max_bytes"] == 1_500_000_000
        assert H <= c["max_edges"] == 10_000_000
        assert len(r["progress"]) == 30 and all(p["audit_ok"] for p in r["progress"])
        assert [p["day"] for p in r["progress"]] == list(range(365, 10951, 365))
        ip = doc["independent_profile"]
        for name, digest in ip["source_sha256"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
        assert hashlib.sha256((TASK / "profile_daily_bounded.py").read_bytes()).hexdigest() == ip["wrapper_sha256"]
        if r["backend"] == "gpu":
            ts = r["transfer_summary"]
            assert ts["input_arrays_bytes"] == 40 * P
            assert ts["profile_bytes"] == 4 * P + 8 * S * L
            assert ts["trait_bytes"] == 8 * P
            assert ts["total_setup_h2d_bytes"] == 52 * P + 8 * S * L
            assert r["profile_transfer_record"]["bytes"] == ts["profile_bytes"]
            assert ts["profile_wall_s"] >= 0
            assert ip["unknown_free_events"] == 0 and ip["cupy_live_rounded_at_end_bytes"] == 0
            assert ip["allocator_policy"]["pool_limit_bytes"] == c["max_bytes"]
            assert not ip["allocator_policy"]["resource_cap_raised"] and not ip["allocator_policy"]["state_discarded"]
            assert r["allocation_profile"]["weekly_samples"] == 1565
            assert r["allocation_profile"]["pool_reserved_weekly_sample_max_bytes"] <= c["max_bytes"]
        receipt = json.loads((TASK / (f.stem + ".exit.json")).read_text())
        assert receipt["returncode"] == 0 and receipt["child_reaped"] and not receipt["timed_out"]
        raw[f.name] = r
        records.append({"file": f.name, "backend": r["backend"], "case": c["name"], "plants": P, "grid": [c["nx"], c["ny"]],
                        "layers": L, "roots": c["root_multiplier"], "profiles": rp["table"],
                        "integration_s": t["integration_wall_s"], "whole_process_s": receipt["whole_child_wall_seconds"],
                        "daily_s": t["daily_block_wall_s"], "weekly_s": t["weekly_update_wall_s"],
                        "wall_ms_per_model_day": 1000 * t["integration_wall_s"] / 10950,
                        "persistent_bytes": persistent, "rss_bytes": m["process_peak_rss_bytes"],
                        "live_peak_bytes": r.get("allocation_profile", {}).get("cupy_live_rounded_peak_bytes"),
                        "reserved_peak_bytes": r.get("allocation_profile", {}).get("pool_reserved_weekly_sample_max_bytes"),
                        "final_edges": H, "final_water_m3": w["total"], "final_biomass_kg": b["total"],
                        "water_residual": w["residual"], "flux_residual": w["flux_residual"],
                        "biomass_residual": b["residual"], "checks_passed": True})
    pairs = [(raw["cpu_june_hectare_3layers_roots3_weighted.json"], raw["gpu_june_hectare_3layers_roots3_weighted.json"], "cpu_gpu")]
    historical = json.loads((ROOT / "benchmarks/results/daily_water_30year_20261030/gpu_june_hectare_3layers_roots3_bounded.json").read_text())["results"][0]
    pairs.append((raw["gpu_uniform_control_june_hectare_roots3_weighted.json"], historical, "uniform_vs_historical"))
    for x, y, kind in pairs:
        assert x["counters"] == y["counters"], kind
        for group, fields in (("water", ("total", "plant_uptake", "grid_removed", "overflow", "pending_week_uptake")),
                              ("biomass", ("total", "growth", "turnover"))):
            for field in fields:
                assert close(x["audit"][group][field], y["audit"][group][field]), (kind, group, field)
    result = {"status": "pass", "records": records, "full_cpu_gpu_scalar_agreement": True,
              "uniform_matches_historical_equal_layer": True}
    (TASK / "result_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
