"""Independent scalar, calendar, memory and provenance checks of completed records."""
from pathlib import Path
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[3]
TASK = Path(__file__).resolve().parent
OUT = ROOT / "benchmarks/results/daily_water_30year_20261030"


def close(a, b, rtol=1e-9, atol=1e-8):
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def verify(path):
    doc = json.loads(path.read_text())
    assert doc["status"] == "ok" and doc["n_results"] == 1, path.name
    r = doc["results"][0]
    assert r["status"] == "ok", (path.name, r.get("error"))
    c, k, a, m, t = (r[n] for n in ("case", "counters", "audit", "memory", "timing"))
    area = c["nx"] * c["ny"] * c["dx"] ** 2
    assert (k["day"], k["weeks_done"], k["pending_days"], k["maps_built"], k["demand_map_builds"],
            k["recharge_events"]) == (10950, 1564, 2, 1565, 1565, 783), path.name
    assert k["population"] == c["grass_count"] + c["shrub_count"] and k["latched_failure"] is None
    assert r["traits"]["n_grass"] == c["grass_count"] and r["traits"]["n_shrub"] == c["shrub_count"]
    assert a["ok"] and all(a["checks"].values()) and a["graph"]["min_count"] > 0
    w, b = a["water"], a["biomass"]
    assert close(w["initial"], 0.075 * area) and close(w["input"], 783 * 0.010 * area)
    assert close(w["total"], w["initial"] + w["input"] - w["overflow"] - w["plant_uptake"])
    assert close(w["plant_uptake"], w["grid_removed"])
    assert close(b["total"], b["initial"] + b["growth"] - b["turnover"])
    assert w["pending_week_uptake"] >= 0 and w["min"] >= 0 and b["min"] > 0
    assert len(r["progress"]) == 30 and [p["day"] for p in r["progress"]] == list(range(365, 10951, 365))
    assert all(p["audit_ok"] for p in r["progress"])
    assert sum(x["nbytes"] for x in m["live_arrays"]) == m["live_bytes"] + m["graph_bytes"]
    assert m["rebuild_peak_bytes"] <= c["max_bytes"] == 1_500_000_000
    assert k["n_edges"] <= c["max_edges"] == 10_000_000
    assert t["integration_wall_s"] >= t["daily_block_wall_s"] + t["weekly_update_wall_s"]
    assert close(r["throughput"]["model_days_per_wall_s"], 10950 / t["integration_wall_s"])
    ip = doc["independent_profile"]
    for name, digest in ip["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    wrapper = TASK / ("profile_daily_bounded.py" if "_bounded" in path.name else "profile_daily.py")
    assert hashlib.sha256(wrapper.read_bytes()).hexdigest() == ip["wrapper_sha256"], wrapper.name
    if r["backend"] == "gpu":
        p = r["allocation_profile"]
        assert ip["unknown_free_events"] == 0 and ip["cupy_live_rounded_at_end_bytes"] == 0
        assert p["cupy_live_rounded_peak_bytes"] == ip["cupy_live_rounded_peak_bytes"]
        assert p["weekly_samples"] == 1565  # startup fixture + 1564 actual updates
        if "_bounded" in path.name:
            policy = ip["allocator_policy"]
            assert policy["pool_limit_bytes"] == c["max_bytes"] and not policy["state_discarded"]
            assert not policy["resource_cap_raised"]
            assert p["pool_reserved_weekly_sample_max_bytes"] <= c["max_bytes"]
    receipt = json.loads((TASK / (path.stem + ".exit.json")).read_text())
    assert receipt["returncode"] == 0 and receipt["child_reaped"] and not receipt["timed_out"]
    return {"file": path.name, "case": c["name"], "backend": r["backend"], "checks_passed": True,
            "integration_s": t["integration_wall_s"], "loop_including_audits_s": t["loop_wall_s"],
            "whole_process_s": receipt["whole_child_wall_seconds"], "daily_s": t["daily_block_wall_s"],
            "weekly_s": t["weekly_update_wall_s"], "audit_s": t["audit_report_wall_s"],
            "model_years_per_wall_min": r["throughput"]["model_years_per_wall_min"],
            "wall_ms_per_model_day": 1000 * t["integration_wall_s"] / 10950,
            "persistent_bytes": m["live_bytes"] + m["graph_bytes"],
            "admission_estimated_peak_bytes": m["rebuild_peak_bytes"],
            "process_peak_rss_bytes": m["process_peak_rss_bytes"],
            "observed_live_peak_bytes": r.get("allocation_profile", {}).get("cupy_live_rounded_peak_bytes"),
            "sampled_reserved_peak_bytes": r.get("allocation_profile", {}).get("pool_reserved_weekly_sample_max_bytes"),
            "final_water": w["total"], "final_biomass": b["total"], "final_edges": k["n_edges"],
            "water_residual": w["residual"], "flux_residual": w["flux_residual"], "biomass_residual": b["residual"]}


def main():
    records = [verify(p) for p in sorted(OUT.glob("*_final.json"))]
    records += [verify(p) for p in sorted(OUT.glob("*_bounded.json"))]
    assert len(records) == 14, f"expected 6 default GPU, 6 bounded GPU and 2 CPU runs; found {len(records)}"
    comparisons = []
    for bpath in sorted(OUT.glob("gpu_*_bounded.json")):
        dpath = bpath.with_name(bpath.name.replace("_bounded.json", "_final.json"))
        rb, rd = json.loads(bpath.read_text())["results"][0], json.loads(dpath.read_text())["results"][0]
        assert rb["case"] == rd["case"] and rb["counters"] == rd["counters"] and rb["fixture"] == rd["fixture"]
        for group, fields in (("water", ("total", "plant_uptake", "grid_removed", "overflow", "pending_week_uptake")),
                              ("biomass", ("total", "growth", "turnover"))):
            for field in fields:
                assert close(rb["audit"][group][field], rd["audit"][group][field]), (bpath.name, group, field)
        comparisons.append({"case": rb["case"], "default_vs_bounded_state_and_budget_agreement": True})
    for cpath in sorted(OUT.glob("cpu_*_final.json")):
        gpath = cpath.with_name(cpath.name.replace("cpu_", "gpu_", 1))
        rc, rg = json.loads(cpath.read_text())["results"][0], json.loads(gpath.read_text())["results"][0]
        assert rc["case"] == rg["case"] and rc["counters"] == rg["counters"]
        for group, fields in (("water", ("total", "plant_uptake", "overflow", "pending_week_uptake")),
                              ("biomass", ("total", "growth", "turnover"))):
            for field in fields:
                assert close(rc["audit"][group][field], rg["audit"][group][field]), (cpath.name, group, field)
        comparisons.append({"case": rc["case"], "full_30year_cpu_gpu_scalar_agreement": True})
    out = {"records": records, "comparisons": comparisons, "all_checks_passed": True}
    (TASK / "result_verification.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
