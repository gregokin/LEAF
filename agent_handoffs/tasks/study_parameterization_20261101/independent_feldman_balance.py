"""Codex independent vectorized water-balance oracle; no candidate water functions."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd

TASK = Path(__file__).parent
OLD = TASK.parent / "paired_growth_et_survey_20261101/public_evidence/selected"
spec = importlib.util.spec_from_file_location("reviewed_image_functions", TASK / "review_snapshot_01/study_parameters.py")
image_functions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(image_functions)


def linear_mass(observations, day):
    if day in observations:
        return observations[day]
    before = [d for d in observations if d < day]
    after = [d for d in observations if d > day]
    if not before or not after:
        return np.nan
    lo, hi = max(before), min(after)
    if hi - lo > 3:
        return np.nan
    return observations[lo] + (observations[hi] - observations[lo]) * (day - lo) / (hi - lo)


def summarize(frame):
    u = float(frame.loss_g.sum() * 1e-6)
    nominal = float(frame.mass_mean_kg.sum())
    actual = float((frame.mass_mean_kg * frame.actual_days).sum())
    return {"periods": len(frame), "plants": int(frame.barcode.nunique()), "genotypes": int(frame.genotype.nunique()),
            "water_m3": u, "B_integral_nominal_kg_day": nominal, "B_integral_actual_kg_day": actual,
            "q_nominal_m3_kg_day": u / nominal, "C_balance_nominal_kg_m3": nominal / (365 * u),
            "q_actual_m3_kg_day": u / actual, "C_balance_actual_kg_m3": actual / (365 * u)}


if __name__ == "__main__":
    out = TASK / "independent_feldman"
    out.mkdir(exist_ok=False)
    imgs = image_functions.load_images(str(TASK / "public_evidence/setaria_image_scalars.csv"))
    zoom = image_functions.fit_zoom_calibration(image_functions.load_zoom_table(str(TASK / "public_evidence/zoom_calibration_data.txt")))
    daily, audit = image_functions.daily_side_view_area(imgs, zoom["a"], zoom["b"])
    harvest = image_functions.load_harvest(str(OLD / "setaria_harvest.csv"))
    joined, mass_cal = image_functions.calibrate_dry_mass(harvest, daily, 5, 0)
    masses = image_functions.mass_series_from_daily(daily, mass_cal["coefficient_k_g_per_normalized_px"])
    w = pd.read_csv(OLD / "setaria_snapshots.csv")
    valid = (w["weight before"] > 0) & (w["weight after"] > 0) & (w["water amount"] >= 0) & w.completed
    raw_count = len(w)
    w = w[valid].copy().rename(columns={"plant barcode": "barcode", "weight before": "before", "weight after": "after"})
    w["ts"] = pd.to_datetime(w.timestamp)
    w["day"] = (w.ts.dt.normalize() - pd.Timestamp("2014-01-13")).dt.days
    w = w.sort_values(["barcode", "ts"])
    w["irrigation"] = w.after - w.before
    w["interval_loss"] = w.groupby("barcode").after.shift() - w.before
    w["previous_day"] = w.groupby("barcode").day.shift()
    # Actual irrigation after the previous day's last observation through today's last.
    day = w.groupby(["barcode", "day"], sort=True).agg(after=("after", "last"), ts=("ts", "last"),
        irrigation=("irrigation", "sum"), interval_loss_sum=("interval_loss", "sum"),
        negative_intervals=("interval_loss", lambda x: int((x < 0).sum()))).reset_index()
    day["prior_day"] = day.groupby("barcode").day.shift()
    day["prior_after"] = day.groupby("barcode").after.shift()
    day["prior_ts"] = day.groupby("barcode").ts.shift()
    day["loss_g"] = day.irrigation - (day.after - day.prior_after)
    day["actual_days"] = (day.ts - day.prior_ts).dt.total_seconds() / 86400
    day["closure_g"] = day.loss_g - day.interval_loss_sum
    complete = day[(day.day - day.prior_day == 1) & (day.prior_day >= 17) & (day.day <= 33)].copy()
    parsed = complete.barcode.map(image_functions.parse_barcode)
    complete["genotype"] = parsed.map(lambda x: x["genotype"] if x else None)
    complete["treatment"] = parsed.map(lambda x: x["treatment"] if x else None)
    complete = complete[complete.genotype.notna()]
    complete["heldout"] = complete.genotype.map(image_functions.genotype_is_heldout)
    complete["B0"] = [linear_mass(masses.get(row.barcode, {}), int(row.prior_day)) for row in complete.itertuples()]
    complete["B1"] = [linear_mass(masses.get(row.barcode, {}), int(row.day)) for row in complete.itertuples()]
    complete = complete[complete.B0.notna() & complete.B1.notna()].copy()
    complete["mass_mean_kg"] = .5 * (complete.B0 + complete.B1)
    complete["published_exclude"] = ((complete.loss_g <= 0) |
        (complete.day.isin([17, 32, 33]) & (complete.loss_g > 200)) |
        ((complete.treatment == "dry") & (complete.day == 32) & (complete.loss_g > 145)))
    loose = complete[(complete.loss_g > 0) & (complete.loss_g <= 600)]
    primary = complete[~complete.published_exclude]
    primary.to_csv(out / "primary_periods.csv", index=False)
    result = {"raw_weight_rows": raw_count, "actual_weighing_rows": len(w), "nonmeasurement_placeholders": raw_count - len(w),
        "max_abs_primary_closure_g": float(primary.closure_g.abs().max()), "published_QC_removed_periods": int(complete.published_exclude.sum()),
        "dry_mass_calibration": mass_cal,
        "primary_all": summarize(primary), "primary_train": summarize(primary[~primary.heldout]),
        "primary_heldout": summarize(primary[primary.heldout]),
        "primary_train_wet": summarize(primary[(~primary.heldout) & (primary.treatment == "wet")]),
        "primary_train_dry": summarize(primary[(~primary.heldout) & (primary.treatment == "dry")]),
        "loose_all_sensitivity": summarize(loose),
        "actual_duration_quantiles_days": np.quantile(primary.actual_days, [0, .01, .5, .99, 1]).tolist(),
        "scope": "Independent water reconstruction and summaries; image-calibration functions reused from frozen reviewed snapshot. No STEP parameter fit."}
    (out / "oracle.json").write_text(json.dumps(image_functions._jsonable(result), indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["actual_weighing_rows", "nonmeasurement_placeholders", "max_abs_primary_closure_g", "primary_train", "actual_duration_quantiles_days"]}, indent=2))
