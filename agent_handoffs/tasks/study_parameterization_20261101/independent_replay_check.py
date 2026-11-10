"""Independent scalar replay of all heldout fit endpoints, without candidate code."""
from pathlib import Path
import csv
import hashlib
import json
import math
import re

TASK = Path(__file__).parent
DATA = TASK / "fit_02"


def rows(name):
    with (DATA / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def interpolate(obs, day):
    if day in obs:
        return obs[day]
    lo = max((d for d in obs if d < day), default=None)
    hi = min((d for d in obs if d > day), default=None)
    if lo is None or hi is None or hi - lo > 3:
        return None
    return obs[lo] + (obs[hi] - obs[lo]) * (day - lo) / (hi - lo)


if __name__ == "__main__":
    fit = json.loads((DATA / "candidate_fragments.json").read_text())
    model = fit["step_diagnostic_fit"]
    C, rate = model["fitted_C_unbounded_kg_per_m3"], model["fitted_r_unbounded"]
    M = model["scenarios_train"][model["selected_scenario_unbounded"]]["M_kg"]
    coefficient = fit["dry_mass_calibration"]["coefficient_k_kg_per_normalized_px"]
    mass, water = {}, {}
    for row in rows("daily_sv_area.csv"):
        mass.setdefault(row["plantbarcode"], {})[int(row["dap"])] = coefficient * float(row["area_norm_mean"])
    for row in rows("daily_balances.csv"):
        if row["complete_daily"] == "True":
            water.setdefault(row["barcode"], {})[int(row["dap_start"])] = float(row["balance_loss_g"]) * 1e-6
    predictions = {}
    max_difference = 0.
    for row in rows("heldout_step_unbounded.csv"):
        barcode = row["barcode"]
        geno = re.match(r"^(Dr\d+|Dp[12])", barcode).group(0)
        assert int(hashlib.sha256(geno.encode()).hexdigest(), 16) % 5 == 0
        bm = {d: interpolate(mass[barcode], d) for d in range(17, 34)}
        bm = {d: value for d, value in bm.items() if value is not None}
        choices = []
        for start in range(17, 33):
            if start not in bm:
                continue
            end = start
            while end < 33 and end in water[barcode]:
                end += 1
            while end > start and end not in bm:
                end -= 1
            if end - start >= 7:
                choices.append((start, end))
        start, end = max(choices, key=lambda x: x[1] - x[0])
        B = bm[start]
        for left in range(start, end, 7):
            right = min(left + 7, end)
            dt = right - left
            U = sum(water[barcode][day] for day in range(left, right))
            burden = B * dt / 365.
            accepted_fraction = min(1., C * U / burden)
            increment = .95 * rate * dt / 365. * accepted_fraction * B * (1. - B / M)
            B += min(increment, max(M - B, 0.))
        difference = abs(B - float(row["endpoint_pred_kg"]))
        max_difference = max(max_difference, difference)
        assert math.isclose(B, float(row["endpoint_pred_kg"]), rel_tol=1e-10, abs_tol=1e-12)
        predictions[barcode] = (B, end)
    original_harvest = TASK.parent / "paired_growth_et_survey_20261101/public_evidence/selected/setaria_harvest.csv"
    with original_harvest.open(newline="") as handle:
        actual = {r["plantbarcode"]: float(r["dry_wt"]) * .001 for r in csv.DictReader(handle)}
    errors, actuals = [], []
    for row in rows("heldout_harvest_replay_unbounded.csv"):
        barcode = row["barcode"]
        predicted, day = predictions[barcode]
        assert day == int(row["end_dap"])
        assert math.isclose(actual[barcode], float(row["harvest_dry_wt_kg"]), abs_tol=1e-12)
        errors.append(predicted - actual[barcode]); actuals.append(actual[barcode])
    mean_actual = sum(actuals) / len(actuals)
    rmse = math.sqrt(sum(x*x for x in errors) / len(errors))
    r2 = 1 - sum(x*x for x in errors) / sum((x - mean_actual)**2 for x in actuals)
    reported = model["heldout_harvest_replay_unbounded"]["metrics_kg"]
    assert math.isclose(rmse, reported["rmse"], rel_tol=1e-10)
    assert math.isclose(r2, reported["r2_about_mean"], rel_tol=1e-10)
    result = {"pass": True, "heldout_replays_checked": len(predictions), "harvests_checked_against_original_raw_table": len(actuals),
              "maximum_endpoint_difference_kg": max_difference, "independent_harvest_rmse_g": rmse * 1000,
              "independent_harvest_r2": r2, "candidate_functions_imported": False,
              "scope": "Analytical STEP-style short-window diagnostic only; native hydrology/GPU/runtime source unchanged."}
    (TASK / "independent_replay_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
