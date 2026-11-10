"""Analytical public-API checks; these are not model implementation kernels."""
from pathlib import Path
import json
import math
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from benchmarks import daily_water_growth as dw


def host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a).copy()


def make_case(layers, profiles):
    return {"name": "analytical", "nx": 1, "ny": 1, "dx": 1.0, "grass_count": 2, "shrub_count": 0,
            "grass_diameter": 0.2, "shrub_diameter": 0.2, "root_multiplier": 3.0, "pattern": "uniform",
            "seed": 1, "layers": layers, "max_edges": 1000, "max_bytes": 10_000_000,
            "root_layer_fractions": profiles}


def inputs():
    return {"x": np.array([0.5, 0.5]), "y": np.array([0.5, 0.5]),
            "canopy_radius": np.array([0.1, 0.1]), "root_radius": np.array([0.3, 0.3]),
            "ids": np.array([101, 202], dtype=np.int64),
            "root_profile_id": np.array([0, 1], dtype=np.int32)}


def check_backend(backend):
    demand = 0.003 * math.pi * 0.1 ** 2
    profiles = {"top_species": [0.8, 0.2], "deep_species": [0.2, 0.8], "third_species": [0.5, 0.5]}
    records = []
    for water, expected in (([0.0, 10 * demand], [0.2 * demand, 0.8 * demand]),
                            ([10 * demand, 0.0], [0.8 * demand, 0.2 * demand]),
                            ([0.0, 0.5 * demand], [0.1 * demand, 0.4 * demand]),
                            ([0.0, 0.0], [0.0, 0.0])):
        sim = dw.Simulation(make_case(2, profiles), backend, inputs=inputs())
        expected_bytes = 40 * sim.L * sim.N + 188 * sim.P + 8 * len(profiles) * sim.L + 12 * len(sim.graph["cells"]) + 8
        assert sum(a["nbytes"] for a in sim.array_records()) == expected_bytes
        np.testing.assert_allclose(host(sim.R)[:, 0], [demand, demand], rtol=1e-12)
        sim.step_days(1)  # consume day 0 (recharge), then isolate non-recharge day 1
        sim.W[:, 0] = sim.xp.asarray(water, dtype=sim.xp.float64)
        old_u, old_removed = host(sim.uptake_cum), host(sim.removed_cum)
        sim.step_days(1)
        uptake = host(sim.uptake_cum) - old_u
        removal = host(sim.removed_cum) - old_removed
        np.testing.assert_allclose(uptake, expected, rtol=1e-12, atol=1e-16)
        np.testing.assert_allclose(uptake.sum(), removal.sum(), rtol=1e-12, atol=1e-16)
        assert host(sim.W).min() >= 0 and sim.day == 2 and sim.demand_map_builds == 1
        records.append({"initial_day1_water": water, "uptake_delta": uptake.tolist(), "passed": True})
    third_inputs = inputs()
    third_inputs["root_profile_id"] = np.array([2, 0], dtype=np.int32)
    sim = dw.Simulation(make_case(2, profiles), backend, inputs=third_inputs)
    np.testing.assert_allclose(host(sim.R)[:, 0], [1.3 * demand, 0.7 * demand], rtol=1e-12)
    sim.step_days(1)
    sim.W[:, 0] = sim.xp.asarray([0.0, 10 * demand])
    before = host(sim.uptake_cum)
    sim.step_days(1)
    np.testing.assert_allclose(host(sim.uptake_cum) - before, [0.5 * demand, 0.2 * demand], rtol=1e-12)
    fractions = {"a": [1 / 3] * 3, "b": [1 / 3] * 3}
    sim = dw.Simulation(make_case(3, fractions), backend, inputs=inputs())
    sim.step_days(6)
    expected_water = 0.075 / 3 + 0.010 / 3 - 6 * 2 * demand / 3
    np.testing.assert_allclose(host(sim.W)[:, 0], expected_water, rtol=1e-12)
    np.testing.assert_allclose(host(sim.uptake_cum), [6 * demand, 6 * demand], rtol=1e-12)
    assert sim.summary()["ok"] and sim.weeks_done == 0
    return {"backend": backend, "heterogeneous_supply_cases": records,
            "explicit_third_profile": True, "uniform_analytical_control": True}


def main():
    requested = sys.argv[1:]
    if not requested or any(b not in ("cpu", "gpu") for b in requested):
        raise ValueError("pass explicit cpu and/or gpu backends")
    if "gpu" in requested and not dw.gpu_available():
        raise RuntimeError("requested GPU oracle unavailable; no silent skip")
    print(json.dumps({"independent_analytical_checks": [check_backend(b) for b in requested]}, indent=2))


if __name__ == "__main__":
    main()
