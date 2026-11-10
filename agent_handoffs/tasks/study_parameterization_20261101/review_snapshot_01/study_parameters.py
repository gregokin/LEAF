#!/usr/bin/env python3
"""Bounded public-study candidate parameter analysis (study_parameterization_20261101, rev 1).

Inputs are supplied only through CLI arguments (no public-data defaults):
  --images    projected scalar image table (Feldman et al. 2018, vis_setaria_ril.csv projection)
  --weights   original weighing/watering snapshot table (setaria_snapshots.csv columns)
  --harvest   harvest table (plantbarcode, fresh_wt, sat_wt, dry_wt, cartag)
  --zoom      tab-separated zoom calibration table (zoom, camera, rel_area, ...)
  --peterson  JSON of approximate Peterson & Neofotis 2004 figure readings
  --out       NEW output directory (existing directories are refused, never overwritten)

The module is organised as small testable functions.  Nothing here changes any
simulator or field defaults; every number emitted is a *candidate fragment* with
explicit assumptions and audit totals.  Implementer note: this code was written
without executing it (no shell was available); see report.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import date, datetime

import numpy as np
import pandas as pd
from scipy import optimize

# --------------------------------------------------------------------------------------
# Constants (study contract)
# --------------------------------------------------------------------------------------
STUDY_ID = "study_parameterization_20261101"
REVISION = 1
PLANTING_DATE = date(2014, 1, 13)
WINDOW_START_DAP = 17
WINDOW_END_DAP = 33
MAX_IMAGE_GAP_DAYS = 3
G_WATER_TO_M3 = 1e-6          # grams of water -> cubic metres
G_TO_KG = 1e-3
DAYS_PER_YEAR = 365.0
STEP_GAIN_FACTOR = 0.95
STEP_BLOCK_DAYS = 7
T_FRACTIONS = (1.0, 0.9, 0.8)   # explicit conditional T/ET sensitivities (not measured)
DISPENSER_TOL_G = 5.0           # |irrigation mass - dispenser amount| above this is flagged
LARGE_INTERVAL_LOSS_G = 400.0   # flagged as unusual (not excluded at interval level)
LARGE_DAILY_LOSS_G = 600.0      # daily balance above this is treated as a QC artifact
BARCODE_RE = re.compile(r"^(Dp[12]|Dr\d+)(AA|AB)(\d+)$")
TS_FORMATS = (
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
)


# --------------------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------------------
def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _jsonable(obj):
    """Convert numpy / pandas / datetime scalars into JSON-serialisable values."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if obj is pd.NaT:
        return None
    return obj


def parse_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("true")


# --------------------------------------------------------------------------------------
# 1. Calendar days, genotype parsing, deterministic holdout
# --------------------------------------------------------------------------------------
def parse_naive_timestamp(text):
    """Parse a timezone-free timestamp string into a naive datetime (no POSIX conversion)."""
    if not isinstance(text, str):
        return None
    s = text.strip()
    for fmt in TS_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def dap_from_timestamp(ts, planting: date = PLANTING_DATE):
    """Days after planting from the *calendar date* of a naive timestamp."""
    if ts is None or (isinstance(ts, float) and math.isnan(ts)) or ts is pd.NaT:
        return None
    d = ts.date() if hasattr(ts, "date") else ts
    return (d - planting).days


def parse_timestamp_series(series: pd.Series) -> pd.Series:
    parsed = series.map(parse_naive_timestamp)
    return pd.to_datetime(parsed, errors="coerce")  # naive datetime64, no tz


def dap_series(ts: pd.Series) -> pd.Series:
    vals = ts.map(lambda t: dap_from_timestamp(t) if pd.notna(t) else None)
    return pd.array(vals.tolist(), dtype="Int64")


def parse_barcode(barcode):
    """`Dr<number>A[AB]<digits>` or parents `Dp1`/`Dp2`; AA = wet, AB = dry. Exact match only."""
    if not isinstance(barcode, str):
        return None
    m = BARCODE_RE.match(barcode)
    if not m:
        return None
    genotype, trt, replicate = m.groups()
    return {
        "genotype": genotype,
        "treatment": "wet" if trt == "AA" else "dry",
        "replicate": replicate,
        "is_parent": genotype.startswith("Dp"),
    }


def genotype_is_heldout(genotype: str, modulus: int = 5, remainder: int = 0) -> bool:
    """Deterministic holdout: SHA256(genotype) as integer modulo `modulus` == `remainder`."""
    digest = hashlib.sha256(genotype.encode("utf-8")).hexdigest()
    return int(digest, 16) % modulus == remainder


def assign_split(df: pd.DataFrame, barcode_col: str, modulus: int = 5, remainder: int = 0) -> pd.DataFrame:
    """Add genotype / treatment / is_parent / heldout columns keyed on the exact barcode."""
    out = df.copy()
    parsed = out[barcode_col].map(parse_barcode)
    out["genotype"] = parsed.map(lambda p: p["genotype"] if p else None)
    out["treatment"] = parsed.map(lambda p: p["treatment"] if p else None)
    out["is_parent"] = parsed.map(lambda p: bool(p["is_parent"]) if p else False)
    out["heldout"] = out["genotype"].map(
        lambda g: genotype_is_heldout(g, modulus, remainder) if isinstance(g, str) else None
    )
    return out


# --------------------------------------------------------------------------------------
# 2. Zoom calibration and side-view area normalisation
# --------------------------------------------------------------------------------------
def camera_zoom(raw_zoom):
    """Author's basic mapping: camera_zoom = 1 + 5*(raw_zoom-1)/5999."""
    return 1.0 + 5.0 * (np.asarray(raw_zoom, dtype=float) - 1.0) / 5999.0


def parse_zoom_label(label):
    """'z1000' -> 1000 ; accepts ints / numeric strings."""
    if isinstance(label, (int, np.integer)):
        return int(label)
    if isinstance(label, float) and math.isfinite(label):
        return int(label)
    if isinstance(label, str):
        s = label.strip().lower()
        if s.startswith("z"):
            s = s[1:]
        try:
            return int(float(s))
        except ValueError:
            return None
    return None


def zoom_model(cz, a, b):
    return a * np.exp(b * np.asarray(cz, dtype=float))


def fit_zoom_calibration(zoom_table: pd.DataFrame) -> dict:
    """Nonlinear least squares of a*exp(b*camera_zoom) to ALL tabulated relative areas."""
    z = pd.to_numeric(zoom_table["zoom"], errors="coerce").to_numpy(dtype=float)
    rel = pd.to_numeric(zoom_table["rel_area"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(z) & np.isfinite(rel)
    cz = camera_zoom(z[ok])
    popt, _ = optimize.curve_fit(zoom_model, cz, rel[ok], p0=(1.0, 0.8), maxfev=20000)
    pred = zoom_model(cz, *popt)
    resid = rel[ok] - pred
    return {
        "model": "rel_area = a*exp(b*camera_zoom); camera_zoom = 1 + 5*(raw_zoom-1)/5999",
        "a": float(popt[0]),
        "b": float(popt[1]),
        "n_rows": int(ok.sum()),
        "rmse_rel_area": float(np.sqrt(np.mean(resid ** 2))),
        "max_abs_relative_residual": float(np.max(np.abs(resid / rel[ok]))),
        "fitted_to": "all rows (TV and SV, raised and car-top) as the contract requires",
    }


def predicted_rel_area(raw_zoom, a, b):
    return zoom_model(camera_zoom(raw_zoom), a, b)


def daily_side_view_area(images: pd.DataFrame, a: float, b: float):
    """Average valid positive, zoom-normalised SV areas across rotations per barcode/day.

    Returns (daily table, audit dict).  Out-of-bounds images are kept in the primary
    mean but counted, and an in-bounds-only mean is carried for sensitivity.
    Zero / negative / missing areas are never used (counted instead).
    """
    sv = images[images["camera"].astype(str).str.strip().str.upper() == "SV"].copy()
    sv["area_px"] = pd.to_numeric(sv["area"], errors="coerce")
    sv["raw_zoom"] = sv["zoom"].map(parse_zoom_label)
    zoom_ok = sv["raw_zoom"].notna()
    sv["rel_area_pred"] = np.nan
    sv.loc[zoom_ok, "rel_area_pred"] = predicted_rel_area(sv.loc[zoom_ok, "raw_zoom"].astype(float), a, b)
    sv["area_norm"] = sv["area_px"] / sv["rel_area_pred"]
    sv["in_bounds_flag"] = parse_bool_series(sv["in_bounds"])
    has_day = sv["dap"].notna()
    valid_mask = (
        sv["area_px"].notna() & (sv["area_px"] > 0) & sv["area_norm"].notna()
        & np.isfinite(sv["area_norm"].astype(float)) & has_day
    )
    audit = {
        "sv_images_total": int(len(sv)),
        "sv_images_missing_timestamp": int((~has_day).sum()),
        "sv_images_unparsed_zoom": int((~zoom_ok).sum()),
        "sv_images_nonpositive_or_missing_area": int((sv["area_px"].isna() | (sv["area_px"] <= 0)).sum()),
        "sv_images_valid_positive": int(valid_mask.sum()),
        "sv_images_out_of_bounds": int((~sv["in_bounds_flag"]).sum()),
        "sv_images_valid_and_out_of_bounds": int((valid_mask & ~sv["in_bounds_flag"]).sum()),
        "zoom_levels": {str(k): int(v) for k, v in sv["zoom"].astype(str).value_counts().items()},
    }
    valid = sv[valid_mask].copy()
    valid["dap"] = valid["dap"].astype(int)
    grp = valid.groupby(["plantbarcode", "dap"])
    daily = grp.agg(
        area_norm_mean=("area_norm", "mean"),
        n_valid=("area_norm", "size"),
        n_out_of_bounds=("in_bounds_flag", lambda s: int((~s).sum())),
    ).reset_index()
    inb = (
        valid[valid["in_bounds_flag"]]
        .groupby(["plantbarcode", "dap"])["area_norm"].mean()
        .rename("area_norm_mean_in_bounds")
        .reset_index()
    )
    daily = daily.merge(inb, how="left", on=["plantbarcode", "dap"])
    daily["rel_diff_in_bounds_only"] = (
        (daily["area_norm_mean_in_bounds"] - daily["area_norm_mean"]) / daily["area_norm_mean"]
    )
    audit["plant_days"] = int(len(daily))
    audit["plant_days_with_any_out_of_bounds"] = int((daily["n_out_of_bounds"] > 0).sum())
    audit["plant_days_all_out_of_bounds"] = int(daily["area_norm_mean_in_bounds"].isna().sum())
    rd = daily["rel_diff_in_bounds_only"].dropna()
    audit["in_bounds_only_relative_difference"] = {
        "median": float(rd.median()) if len(rd) else None,
        "p05": float(rd.quantile(0.05)) if len(rd) else None,
        "p95": float(rd.quantile(0.95)) if len(rd) else None,
    }
    return daily, audit


# --------------------------------------------------------------------------------------
# 3. Harvest join and zero-intercept dry-mass / area coefficient
# --------------------------------------------------------------------------------------
def fit_zero_intercept_coefficient(area, mass) -> float:
    """Least-squares k for mass = k*area (zero intercept). Raises if not positive."""
    a = np.asarray(area, dtype=float)
    m = np.asarray(mass, dtype=float)
    ok = np.isfinite(a) & np.isfinite(m)
    if ok.sum() == 0 or np.sum(a[ok] ** 2) <= 0:
        raise ValueError("no usable (area, mass) pairs for zero-intercept fit")
    k = float(np.sum(a[ok] * m[ok]) / np.sum(a[ok] ** 2))
    if not k > 0:
        raise ValueError(f"zero-intercept coefficient is not positive: {k}")
    return k


def prediction_metrics(obs, pred) -> dict:
    o = np.asarray(obs, dtype=float)
    p = np.asarray(pred, dtype=float)
    ok = np.isfinite(o) & np.isfinite(p)
    o, p = o[ok], p[ok]
    if len(o) == 0:
        return {"n": 0}
    err = p - o
    ss_tot = float(np.sum((o - o.mean()) ** 2))
    rel = np.abs(err) / np.where(o != 0, np.abs(o), np.nan)
    return {
        "n": int(len(o)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "r2_about_mean": float(1.0 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else None,
        "median_abs_relative_error": float(np.nanmedian(rel)),
    }


def final_day_area(daily: pd.DataFrame, column: str = "area_norm_mean") -> pd.DataFrame:
    """Last imaging day with a valid SV mean per barcode (harvest date is not tabulated)."""
    d = daily[daily[column].notna()]
    idx = d.groupby("plantbarcode")["dap"].idxmax()
    return d.loc[idx, ["plantbarcode", "dap", column]].rename(
        columns={"dap": "final_image_dap", column: "final_area_norm"}
    )


def calibrate_dry_mass(harvest: pd.DataFrame, daily: pd.DataFrame, modulus: int, remainder: int,
                       column: str = "area_norm_mean"):
    """Join final-day SV area to harvest dry mass; fit k on TRAIN genotypes only."""
    join = harvest.merge(final_day_area(daily, column), on="plantbarcode", how="left")
    join = assign_split(join, "plantbarcode", modulus, remainder)
    fitable = join[join["final_area_norm"].notna() & join["dry_wt_g"].notna()
                   & (join["dry_wt_g"] > 0) & join["genotype"].notna()].copy()
    train = fitable[fitable["heldout"] == False]  # noqa: E712
    held = fitable[fitable["heldout"] == True]    # noqa: E712
    k = fit_zero_intercept_coefficient(train["final_area_norm"], train["dry_wt_g"])
    join["dry_wt_pred_g"] = k * join["final_area_norm"]
    train_pred = k * train["final_area_norm"]
    held_pred = k * held["final_area_norm"]
    held_by_geno = {}
    for g, grp in held.groupby("genotype"):
        held_by_geno[g] = prediction_metrics(grp["dry_wt_g"], k * grp["final_area_norm"])
    summary = {
        "coefficient_k_g_per_normalized_px": k,
        "coefficient_k_kg_per_normalized_px": k * G_TO_KG,
        "fit": "zero-intercept least squares on TRAIN genotypes only",
        "n_harvest_rows": int(len(harvest)),
        "n_joined_with_final_area": int(join["final_area_norm"].notna().sum()),
        "n_fitable": int(len(fitable)),
        "n_train": int(len(train)),
        "n_heldout": int(len(held)),
        "train_metrics_g": prediction_metrics(train["dry_wt_g"], train_pred),
        "heldout_metrics_g": prediction_metrics(held["dry_wt_g"], held_pred),
        "heldout_by_genotype_g": held_by_geno,
        "final_image_dap_distribution": {
            str(int(k_)): int(v) for k_, v in join["final_image_dap"].dropna().astype(int).value_counts().sort_index().items()
        },
        "harvest_ids_without_image_join": sorted(join.loc[join["final_area_norm"].isna(), "plantbarcode"].astype(str).tolist()),
        "harvest_ids_unparsed_barcode": sorted(join.loc[join["genotype"].isna(), "plantbarcode"].astype(str).tolist()),
        "note": ("Harvest dry mass is an independent destructive measurement; all trajectory masses "
                 "are image-derived (k * zoom-normalised SV area) with this single calibration. "
                 "Trajectories are not anchored to their own harvest."),
    }
    return join, summary


# --------------------------------------------------------------------------------------
# 4. Water accounting
# --------------------------------------------------------------------------------------
def reconstruct_weighing_intervals(weights: pd.DataFrame, dispenser_tol_g: float = DISPENSER_TOL_G,
                                   large_loss_g: float = LARGE_INTERVAL_LOSS_G) -> pd.DataFrame:
    """Consecutive-weighing intervals per barcode.

    loss_g       = previous weight_after - current weight_before   (never clipped)
    irrigation_g = current weight_after - current weight_before     (actual mass added)
    dispenser_g  = reported 'water amount'                          (kept separately)
    `usable` is False when either record is incomplete or duration is non-positive.
    Negative / large losses, negative irrigation and dispenser mismatches are flagged
    in `reasons` but remain in the table with their actual values.
    """
    cols = ["barcode", "prev_timestamp", "cur_timestamp", "prev_dap", "cur_dap", "dt_days",
            "loss_g", "irrigation_g", "dispenser_g", "usable", "reasons"]
    rows = []
    ordered = weights[weights["timestamp"].notna()].sort_values(["barcode", "timestamp"], kind="mergesort")
    blocking = {"previous_record_incomplete", "current_record_incomplete", "nonpositive_duration"}
    for barcode, grp in ordered.groupby("barcode", sort=False):
        recs = grp.to_dict("records")
        for prev, cur in zip(recs[:-1], recs[1:]):
            reasons = []
            if not prev["successful"]:
                reasons.append("previous_record_incomplete")
            if not cur["successful"]:
                reasons.append("current_record_incomplete")
            dt_days = (cur["timestamp"] - prev["timestamp"]).total_seconds() / 86400.0
            if dt_days <= 0:
                reasons.append("nonpositive_duration")
            loss = prev["weight_after_g"] - cur["weight_before_g"]
            irrigation = cur["weight_after_g"] - cur["weight_before_g"]
            disp = cur["dispenser_water_g"]
            if _finite(loss) and loss < 0:
                reasons.append("negative_loss")
            if _finite(loss) and loss > large_loss_g:
                reasons.append("unusual_large_loss")
            if _finite(irrigation) and irrigation < 0:
                reasons.append("negative_irrigation")
            if _finite(irrigation) and _finite(disp) and abs(irrigation - disp) > dispenser_tol_g:
                reasons.append("dispenser_mismatch")
            usable = not (set(reasons) & blocking)
            rows.append({
                "barcode": barcode,
                "prev_timestamp": prev["timestamp"], "cur_timestamp": cur["timestamp"],
                "prev_dap": prev["dap"], "cur_dap": cur["dap"], "dt_days": dt_days,
                "loss_g": loss, "irrigation_g": irrigation, "dispenser_g": disp,
                "usable": usable, "reasons": ";".join(reasons),
            })
    return pd.DataFrame(rows, columns=cols)


def daily_boundary_balances(weights: pd.DataFrame, intervals: pd.DataFrame,
                            large_daily_loss_g: float = LARGE_DAILY_LOSS_G) -> pd.DataFrame:
    """Water balance between consecutive LAST successful weighings of a day, per barcode.

    balance_loss_g = sum(actual irrigation mass in (t0, t1]) - (W_after(t1) - W_after(t0))
    which must equal the sum of interval losses over the same chain when the chain is
    complete (no incomplete records inside).  `closure_residual_g` records the check.
    """
    cols = ["barcode", "dap_start", "dap_end", "period_days", "duration_days", "ts_start", "ts_end",
            "weight_after_start_g", "weight_after_end_g", "storage_change_g", "irrigation_sum_g",
            "dispenser_sum_g", "balance_loss_g", "interval_loss_sum_g", "closure_residual_g",
            "n_records", "n_unsuccessful_records", "n_intervals", "n_negative_intervals",
            "chain_complete", "complete_daily", "reasons"]
    rows = []
    w = weights[weights["timestamp"].notna()].sort_values(["barcode", "timestamp"], kind="mergesort")
    iv_groups = {k: g for k, g in intervals.groupby("barcode")} if len(intervals) else {}
    for barcode, grp in w.groupby("barcode", sort=False):
        ok = grp[grp["successful"]]
        if ok.empty:
            continue
        last = ok.groupby("dap", sort=True).tail(1).sort_values("timestamp")
        bounds = last.to_dict("records")
        g_iv = iv_groups.get(barcode)
        for b0, b1 in zip(bounds[:-1], bounds[1:]):
            ts0, ts1 = b0["timestamp"], b1["timestamp"]
            between = grp[(grp["timestamp"] > ts0) & (grp["timestamp"] <= ts1)]
            succ = between[between["successful"]]
            n_unsucc = int((~between["successful"]).sum())
            irrigation_sum = float((succ["weight_after_g"] - succ["weight_before_g"]).sum())
            dispenser_sum = float(succ["dispenser_water_g"].sum(skipna=True))
            storage_change = float(b1["weight_after_g"] - b0["weight_after_g"])
            balance_loss = irrigation_sum - storage_change
            if g_iv is not None:
                sel = g_iv[(g_iv["cur_timestamp"] > ts0) & (g_iv["cur_timestamp"] <= ts1)]
            else:
                sel = intervals.iloc[0:0]
            n_int = int(len(sel))
            n_neg = int((sel["loss_g"] < 0).sum())
            chain_complete = bool(n_unsucc == 0 and n_int > 0 and sel["usable"].all())
            interval_loss_sum = float(sel["loss_g"].sum()) if chain_complete else float("nan")
            closure = balance_loss - interval_loss_sum if chain_complete else float("nan")
            d0, d1 = int(b0["dap"]), int(b1["dap"])
            period_days = d1 - d0
            reasons = []
            if not chain_complete:
                reasons.append("chain_incomplete")
            if period_days != 1:
                reasons.append("multi_day_period")
            if balance_loss < 0:
                reasons.append("negative_daily_loss")
            if balance_loss > large_daily_loss_g:
                reasons.append("unusual_large_daily_loss")
            if n_neg > 0:
                reasons.append("contains_negative_interval")
            complete_daily = bool(chain_complete and period_days == 1 and 0 <= balance_loss <= large_daily_loss_g)
            rows.append({
                "barcode": barcode, "dap_start": d0, "dap_end": d1, "period_days": period_days,
                "duration_days": (ts1 - ts0).total_seconds() / 86400.0, "ts_start": ts0, "ts_end": ts1,
                "weight_after_start_g": float(b0["weight_after_g"]), "weight_after_end_g": float(b1["weight_after_g"]),
                "storage_change_g": storage_change, "irrigation_sum_g": irrigation_sum,
                "dispenser_sum_g": dispenser_sum, "balance_loss_g": balance_loss,
                "interval_loss_sum_g": interval_loss_sum, "closure_residual_g": closure,
                "n_records": int(len(between)), "n_unsuccessful_records": n_unsucc,
                "n_intervals": n_int, "n_negative_intervals": n_neg,
                "chain_complete": chain_complete, "complete_daily": complete_daily,
                "reasons": ";".join(reasons),
            })
    return pd.DataFrame(rows, columns=cols)


def water_audit(weights: pd.DataFrame, intervals: pd.DataFrame, balances: pd.DataFrame) -> dict:
    """Counts and totals needed to verify the mass balance before any fitting."""
    reason_counts = {}
    for r in intervals["reasons"]:
        for item in str(r).split(";"):
            if item:
                reason_counts[item] = reason_counts.get(item, 0) + 1
    bal_reasons = {}
    for r in balances["reasons"]:
        for item in str(r).split(";"):
            if item:
                bal_reasons[item] = bal_reasons.get(item, 0) + 1
    complete = balances[balances["chain_complete"]]
    closure = complete["closure_residual_g"].abs()
    return {
        "records_total": int(len(weights)),
        "records_without_timestamp": int(weights["timestamp"].isna().sum()),
        "records_successful": int(weights["successful"].sum()),
        "records_unsuccessful": int((~weights["successful"]).sum()),
        "barcodes": int(weights["barcode"].nunique()),
        "intervals_total": int(len(intervals)),
        "intervals_usable": int(intervals["usable"].sum()),
        "interval_flag_counts": reason_counts,
        "interval_loss_sum_usable_g": float(intervals.loc[intervals["usable"], "loss_g"].sum()),
        "interval_negative_loss_sum_g": float(intervals.loc[intervals["usable"] & (intervals["loss_g"] < 0), "loss_g"].sum()),
        "irrigation_mass_sum_usable_g": float(intervals.loc[intervals["usable"], "irrigation_g"].sum()),
        "dispenser_sum_usable_g": float(intervals.loc[intervals["usable"], "dispenser_g"].sum(skipna=True)),
        "daily_balances_total": int(len(balances)),
        "daily_balances_chain_complete": int(balances["chain_complete"].sum()),
        "daily_balances_complete_daily": int(balances["complete_daily"].sum()),
        "daily_balance_flag_counts": bal_reasons,
        "closure_max_abs_residual_g": float(closure.max()) if len(closure) else None,
        "closure_residuals_above_1e-6_g": int((closure > 1e-6).sum()),
        "balance_loss_sum_complete_chain_g": float(complete["balance_loss_g"].sum()),
        "interval_loss_sum_complete_chain_g": float(complete["interval_loss_sum_g"].sum()),
        "g_water_to_m3": G_WATER_TO_M3,
    }


# --------------------------------------------------------------------------------------
# 5. Window trajectories: interpolation, periods, rates
# --------------------------------------------------------------------------------------
def interpolate_series(obs: dict, days, max_gap: int = MAX_IMAGE_GAP_DAYS) -> dict:
    """Linear interpolation strictly between neighbouring observations; never extrapolate.

    Returns {day: (value, 'observed'|'interpolated')} for requested days.
    """
    out = {}
    obs_days = sorted(d for d, v in obs.items() if _finite(v))
    for d in days:
        if d in obs and _finite(obs[d]):
            out[d] = (float(obs[d]), "observed")
            continue
        prev = max((x for x in obs_days if x < d), default=None)
        nxt = min((x for x in obs_days if x > d), default=None)
        if prev is None or nxt is None or (nxt - prev) > max_gap:
            continue
        wgt = (d - prev) / (nxt - prev)
        out[d] = (float(obs[prev] * (1 - wgt) + obs[nxt] * wgt), "interpolated")
    return out


def mass_series_from_daily(daily: pd.DataFrame, k_g_per_px: float, column: str = "area_norm_mean") -> dict:
    """{barcode: {dap: image-derived dry mass (kg)}}."""
    out = {}
    d = daily[daily[column].notna()]
    for bc, g in d.groupby("plantbarcode"):
        out[bc] = {int(r.dap): float(getattr(r, column)) * k_g_per_px * G_TO_KG for r in g.itertuples()}
    return out


def build_window_periods(balances: pd.DataFrame, mass_by_barcode: dict, window=(WINDOW_START_DAP, WINDOW_END_DAP),
                         max_gap: int = MAX_IMAGE_GAP_DAYS):
    """Complete daily water periods inside the window with image-derived mass at both ends."""
    w0, w1 = window
    rows = []
    excl = {"not_complete_daily": 0, "outside_window": 0, "no_mass_series": 0, "mass_missing_at_period_end": 0}
    days = list(range(w0, w1 + 1))
    for bc, grp in balances.groupby("barcode"):
        obs = mass_by_barcode.get(bc)
        series = interpolate_series(obs, days, max_gap) if obs else None
        for r in grp.itertuples():
            d0, d1 = int(r.dap_start), int(r.dap_end)
            if d0 < w0 or d1 > w1:
                excl["outside_window"] += 1
                continue
            if not r.complete_daily:
                excl["not_complete_daily"] += 1
                continue
            if series is None:
                excl["no_mass_series"] += 1
                continue
            if d0 not in series or d1 not in series:
                excl["mass_missing_at_period_end"] += 1
                continue
            rows.append({
                "barcode": bc, "dap_start": d0, "dap_end": d1, "dt_days": d1 - d0,
                "loss_g": float(r.balance_loss_g), "U_m3": float(r.balance_loss_g) * G_WATER_TO_M3,
                "irrigation_sum_g": float(r.irrigation_sum_g), "dispenser_sum_g": float(r.dispenser_sum_g),
                "B_start_kg": series[d0][0], "B_end_kg": series[d1][0],
                "B_start_source": series[d0][1], "B_end_source": series[d1][1],
            })
    periods = pd.DataFrame(rows, columns=["barcode", "dap_start", "dap_end", "dt_days", "loss_g", "U_m3",
                                          "irrigation_sum_g", "dispenser_sum_g", "B_start_kg", "B_end_kg",
                                          "B_start_source", "B_end_source"])
    return periods, excl


def water_rate_summary(periods: pd.DataFrame) -> dict:
    """q = sum(U)/integral(B dt) [m3/kg/day]; C_balance = integral(B dt)/(365*sum(U)) [kg/m3];
    production ratio = delta_B/sum(U) [kg/m3] over the same periods."""
    if len(periods) == 0:
        return {"n_periods": 0}
    integral = float((0.5 * (periods["B_start_kg"] + periods["B_end_kg"]) * periods["dt_days"]).sum())
    U = float(periods["U_m3"].sum())
    dB = float((periods["B_end_kg"] - periods["B_start_kg"]).sum())
    return {
        "n_periods": int(len(periods)),
        "n_plants": int(periods["barcode"].nunique()),
        "sum_U_m3": U,
        "integral_B_kg_days": integral,
        "delta_B_kg": dB,
        "q_m3_per_kg_day": U / integral if integral > 0 else None,
        "C_balance_kg_per_m3": integral / (DAYS_PER_YEAR * U) if U > 0 else None,
        "production_ratio_kg_per_m3": dB / U if U > 0 else None,
    }


def t_fraction_sensitivity(summary: dict, fractions=T_FRACTIONS) -> dict:
    """Conditional T = f*ET sensitivities (not an ET->T calibration)."""
    out = {}
    q = summary.get("q_m3_per_kg_day")
    c_bal = summary.get("C_balance_kg_per_m3")
    prod = summary.get("production_ratio_kg_per_m3")
    for f in fractions:
        out[str(f)] = {
            "assumed_T_fraction_of_ET": f,
            "q_T_m3_per_kg_day": q * f if q is not None else None,
            "C_balance_T_kg_per_m3": c_bal / f if c_bal is not None else None,
            "production_ratio_T_kg_per_m3": prod / f if prod is not None else None,
        }
    return out


def per_plant_summary(periods: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bc, g in periods.groupby("barcode"):
        s = water_rate_summary(g)
        s.update({"barcode": bc, "genotype": g["genotype"].iloc[0], "treatment": g["treatment"].iloc[0],
                  "heldout": g["heldout"].iloc[0], "dap_first": int(g["dap_start"].min()),
                  "dap_last": int(g["dap_end"].max()), "n_interpolated_ends":
                  int((g["B_start_source"] == "interpolated").sum() + (g["B_end_source"] == "interpolated").sum())})
        rows.append(s)
    return pd.DataFrame(rows)


def bootstrap_by_genotype(periods: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Resample genotypes with replacement (all replicates/treatments/days move together)."""
    genos = sorted(periods["genotype"].dropna().unique().tolist())
    if len(genos) < 2:
        return {"n_genotypes": len(genos), "note": "too few genotypes to bootstrap"}
    rng = np.random.default_rng(seed)
    by_geno = {g: periods[periods["genotype"] == g] for g in genos}
    keys = ["q_m3_per_kg_day", "C_balance_kg_per_m3", "production_ratio_kg_per_m3"]
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.choice(genos, size=len(genos), replace=True)
        samp = pd.concat([by_geno[g] for g in pick], ignore_index=True)
        s = water_rate_summary(samp)
        for k in keys:
            draws[k].append(s.get(k) if s.get(k) is not None else np.nan)
    out = {"n_genotypes": len(genos), "n_boot": n_boot, "seed": seed, "unit": "genotype"}
    for k in keys:
        arr = np.asarray(draws[k], dtype=float)
        arr = arr[np.isfinite(arr)]
        out[k] = {
            "p2.5": float(np.percentile(arr, 2.5)) if len(arr) else None,
            "p50": float(np.percentile(arr, 50)) if len(arr) else None,
            "p97.5": float(np.percentile(arr, 97.5)) if len(arr) else None,
            "n_finite": int(len(arr)),
        }
    return out


# --------------------------------------------------------------------------------------
# 6/7. STEP-style diagnostic growth rule
# --------------------------------------------------------------------------------------
def step_gain(B_kg: float, dt_days: float, U_m3: float, C: float, r_year: float, M_kg: float) -> float:
    """required = B*dt/365; available = C*U; wf = min(1, available/required);
    gain = 0.95*r*(dt/365)*wf*B*(1-B/M), capped at M-B.  Activity=temperature=live fraction=1."""
    required = B_kg * dt_days / DAYS_PER_YEAR
    available = C * U_m3
    wf = 1.0 if required <= 0 else min(1.0, available / required)
    gain = STEP_GAIN_FACTOR * r_year * (dt_days / DAYS_PER_YEAR) * wf * B_kg * (1.0 - B_kg / M_kg)
    return min(gain, M_kg - B_kg)


def simulate_step(B0_kg: float, periods, C: float, r_year: float, M_kg: float) -> list:
    """periods: iterable of (dt_days, U_m3). Returns mass at each period end."""
    B = float(B0_kg)
    out = []
    for dt_days, U_m3 in periods:
        B = B + step_gain(B, dt_days, U_m3, C, r_year, M_kg)
        out.append(B)
    return out


def weekly_euler_periods(days: int, block: int = STEP_BLOCK_DAYS) -> list:
    """7-day updates plus a final partial period."""
    full, rem = divmod(int(days), block)
    return [block] * full + ([rem] if rem else [])


def max_normal_growth_factor(r_year: float, days: int, block: int = STEP_BLOCK_DAYS) -> float:
    """Growth factor with water=temperature=activity=1 and B << M (no cap)."""
    f = 1.0
    for dt in weekly_euler_periods(days, block):
        f *= 1.0 + STEP_GAIN_FACTOR * r_year * dt / DAYS_PER_YEAR
    return f


def min_weekly_euler_rate(factor: float, days: int, block: int = STEP_BLOCK_DAYS) -> float:
    """Smallest r_year whose weekly-Euler normal growth reaches `factor` over `days`."""
    if factor <= 1.0:
        return 0.0
    fn = lambda r: max_normal_growth_factor(r, days, block) - factor  # noqa: E731
    hi = 1.0
    while fn(hi) < 0:
        hi *= 2.0
        if hi > 1e9:
            raise ValueError("rate search did not bracket")
    return float(optimize.brentq(fn, 0.0, hi, xtol=1e-10))


def build_step_trajectory(B_by_day: dict, W_by_day: dict, start: int, end: int, block: int = STEP_BLOCK_DAYS):
    """Blocks of `block` days (final partial) over the longest contiguous run with complete
    daily water (W_by_day[d] covers d->d+1, grams) starting/ending on days with mass.
    B_by_day: {day: (B_kg, source)}. Returns None if no run of >= block days exists."""
    best = None
    for s in range(start, end):
        if s not in B_by_day:
            continue
        e = s
        while e < end and e in W_by_day:
            e += 1
        while e > s and e not in B_by_day:
            e -= 1
        if e - s >= block and (best is None or (e - s) > (best[1] - best[0])):
            best = (s, e)
    if best is None:
        return None
    s, e = best
    bounds = list(range(s, e, block)) + [e]
    periods, obs, sources = [], [], []
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        periods.append((float(b1 - b0), sum(W_by_day[d] for d in range(b0, b1)) * G_WATER_TO_M3))
        if b1 in B_by_day:
            obs.append(B_by_day[b1][0])
            sources.append(B_by_day[b1][1])
        else:
            obs.append(float("nan"))
            sources.append("missing")
    return {"start_dap": s, "end_dap": e, "B0_kg": B_by_day[s][0], "B0_source": B_by_day[s][1],
            "periods": periods, "obs_kg": obs, "obs_sources": sources, "boundaries": bounds[1:]}


def step_residuals(x_log, trajectories, M_kg):
    C, r = float(np.exp(x_log[0])), float(np.exp(x_log[1]))
    res = []
    for tr in trajectories:
        pred = simulate_step(tr["B0_kg"], tr["periods"], C, r, M_kg)
        for p, o in zip(pred, tr["obs_kg"]):
            if _finite(o) and o > 0 and p > 0:
                res.append(math.log(p) - math.log(o))
    return np.asarray(res, dtype=float)


def fit_step_parameters(trajectories, M_kg: float, r_max: float, C_bounds=(1e-4, 1e4), r_min: float = 1e-4,
                        starts=None) -> dict:
    """Least squares in log mass with positive bounds; r_max=1 reproduces the runtime bound."""
    if starts is None:
        starts = [(c0, r0) for c0 in (0.01, 1.0, 100.0) for r0 in (0.5, 5.0, 50.0)]
    lb = np.log([C_bounds[0], r_min])
    ub = np.log([C_bounds[1], r_max])
    best = None
    for c0, r0 in starts:
        x0 = np.clip(np.log([c0, r0]), lb + 1e-9, ub - 1e-9)
        try:
            fit = optimize.least_squares(step_residuals, x0, bounds=(lb, ub), args=(trajectories, M_kg))
        except Exception as exc:  # pragma: no cover - defensive
            continue
        sse = float(np.sum(fit.fun ** 2))
        if best is None or sse < best["sse"]:
            best = {"C": float(np.exp(fit.x[0])), "r_year": float(np.exp(fit.x[1])), "sse": sse,
                    "n_residuals": int(len(fit.fun)), "rmse_log": float(np.sqrt(sse / max(1, len(fit.fun))))}
    if best is None:
        return {"error": "all starts failed"}
    best["r_max_bound"] = r_max
    best["r_at_upper_bound"] = bool(best["r_year"] >= 0.99 * r_max)
    best["C_at_bound"] = bool(best["C"] <= C_bounds[0] * 1.01 or best["C"] >= C_bounds[1] * 0.99)
    return best


def evaluate_step(trajectories, C: float, r_year: float, M_kg: float) -> dict:
    """Trajectory log-RMSE and endpoint relative error, pooled and by genotype."""
    per = []
    for tr in trajectories:
        pred = simulate_step(tr["B0_kg"], tr["periods"], C, r_year, M_kg)
        pairs = [(p, o) for p, o in zip(pred, tr["obs_kg"]) if _finite(o) and o > 0 and p > 0]
        if not pairs:
            continue
        logs = [math.log(p) - math.log(o) for p, o in pairs]
        p_end, o_end = pairs[-1]
        per.append({"barcode": tr["barcode"], "genotype": tr["genotype"], "treatment": tr["treatment"],
                    "trajectory_rmse_log": math.sqrt(np.mean(np.square(logs))),
                    "endpoint_pred_kg": p_end, "endpoint_obs_kg": o_end,
                    "endpoint_relative_error": p_end / o_end - 1.0,
                    "observed_growth_factor": o_end / tr["B0_kg"], "predicted_growth_factor": p_end / tr["B0_kg"]})
    df = pd.DataFrame(per)
    if df.empty:
        return {"n_plants": 0, "per_plant": []}
    by_geno = {}
    for g, grp in df.groupby("genotype"):
        by_geno[g] = {"n": int(len(grp)), "trajectory_rmse_log_mean": float(grp["trajectory_rmse_log"].mean()),
                      "endpoint_relative_error_median": float(grp["endpoint_relative_error"].median())}
    return {"n_plants": int(len(df)),
            "trajectory_rmse_log_mean": float(df["trajectory_rmse_log"].mean()),
            "endpoint_relative_error_median": float(df["endpoint_relative_error"].median()),
            "endpoint_relative_error_p10": float(df["endpoint_relative_error"].quantile(0.10)),
            "endpoint_relative_error_p90": float(df["endpoint_relative_error"].quantile(0.90)),
            "observed_growth_factor_median": float(df["observed_growth_factor"].median()),
            "predicted_growth_factor_median": float(df["predicted_growth_factor"].median()),
            "by_genotype": by_geno, "per_plant": df.to_dict("records")}


def profile_rate(trajectories, M_kg, r_grid, C_bounds=(1e-4, 1e4)) -> list:
    """For each fixed r, best C (1-D) and SSE: exposes the C-r trade-off / flat directions."""
    out = []
    for r in r_grid:
        f = lambda lc: float(np.sum(step_residuals([lc, math.log(r)], trajectories, M_kg) ** 2))  # noqa: E731
        res = optimize.minimize_scalar(f, bounds=(math.log(C_bounds[0]), math.log(C_bounds[1])), method="bounded")
        out.append({"r_year": float(r), "C_best": float(math.exp(res.x)), "sse": float(res.fun)})
    return out


def profile_capacity(trajectories, M_kg, C_grid, r_bounds=(1e-4, 1e4)) -> list:
    out = []
    for C in C_grid:
        f = lambda lr: float(np.sum(step_residuals([math.log(C), lr], trajectories, M_kg) ** 2))  # noqa: E731
        res = optimize.minimize_scalar(f, bounds=(math.log(r_bounds[0]), math.log(r_bounds[1])), method="bounded")
        out.append({"C_kg_per_m3": float(C), "r_best": float(math.exp(res.x)), "sse": float(res.fun)})
    return out


# --------------------------------------------------------------------------------------
# 8. Peterson & Neofotis 2004 pot aggregation
# --------------------------------------------------------------------------------------
def aggregate_peterson_row(row: dict, plants_per_pot: int = 2, modeled_units_per_pot: int = 1,
                           water_window_days: int = 2, growth_days: int = 39) -> dict:
    """One aggregate modeled plant per pot: extensive water scales by plants/units; intensive
    ratios (RGR, transpiration productivity, support proxies) do not."""
    rgr = float(row["whole_plant_rgr_per_day"])
    ln_e = float(row["ln_transpiration_productivity"])
    root = float(row["root_component_rgr_per_day"])
    w_plant = float(row["water_last_two_days_ml_per_reported_plant"])
    scale = plants_per_pot / modeled_units_per_pot
    agg_window_ml = w_plant * scale
    daily_ml = agg_window_ml / water_window_days
    e_prod = math.exp(ln_e)
    proxy_total = water_window_days * 1000.0 * e_prod / (DAYS_PER_YEAR * rgr)
    alloc = 1.0 - root / rgr
    factor = math.exp(rgr * growth_days)
    return {
        "co2_pa": row.get("co2_pa"), "water_treatment": row.get("water_treatment"),
        "whole_plant_rgr_per_day": rgr, "root_component_rgr_per_day": root,
        "ln_transpiration_productivity": ln_e, "transpiration_productivity_g_per_g_day": e_prod,
        "water_last_two_days_ml_per_reported_plant": w_plant,
        "aggregate_water_last_two_days_ml": agg_window_ml,
        "aggregate_water_ml_per_day": daily_ml,
        "aggregate_water_m3_per_day": daily_ml * G_WATER_TO_M3,
        "balance_support_proxy_total_mass_kg_per_m3": proxy_total,
        "increment_allocation_nonroot_fraction": alloc,
        "balance_support_proxy_conditional_agb_kg_per_m3": proxy_total * alloc,
        "observed_growth_factor_39d_whole_plant": factor,
        "max_normal_growth_factor_r1_weekly_euler": max_normal_growth_factor(1.0, growth_days),
        "min_weekly_euler_rate_per_year_for_factor": min_weekly_euler_rate(factor, growth_days),
        "assumptions": [
            "aggregate = source per-seedling water * 2 (two seedlings per pot, one modeled unit)",
            "daily rate = two-day aggregate / 2 days; ml water treated as g (1e-6 m3/g)",
            "total-mass proxy = 2*1000*exp(ln_Eprod)/(365*RGR) under the paper's log-mean identities",
            "AGB proxy multiplies by 1-root_CRGR/RGR: increment allocation under a stable-allocation "
            "approximation, NOT observed standing aboveground fraction",
            "growth factor is whole-plant exp(RGR*39); cap comparison uses water=temperature=activity=1, B<<M",
        ],
    }


def peterson_block(readings: dict) -> dict:
    rows = [aggregate_peterson_row(r, readings.get("reported_plants_per_pot", 2),
                                   readings.get("modeled_units_per_pot", 1),
                                   readings.get("water_observation_window_days", 2),
                                   readings.get("growth_interval_days", 39)) for r in readings["rows"]]
    baseline = [r for r in rows if r["co2_pa"] == 38]
    elevated = [r for r in rows if r["co2_pa"] != 38]
    return {
        "source_doi": readings.get("source_doi"),
        "reading_resolution": readings.get("reading_resolution"),
        "reading_resolution_is_not_sampling_uncertainty": True,
        "baseline_38pa": baseline,
        "co2_sensitivity_76pa": elevated,
        "limitations": readings.get("limitations", []),
        "unidentified": ["absolute mature mass", "canopy density", "seasonal turnover", "annual phenology",
                         "rooting layers", "39-day cumulative water", "absolute aboveground mass"],
    }


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_images(path: str) -> pd.DataFrame:
    need = {"timestamp", "frame", "zoom", "plantbarcode", "camera", "area", "in_bounds"}
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = need - set(raw.columns)
    if missing:
        raise ValueError(f"images table missing columns: {sorted(missing)}")
    raw["timestamp_dt"] = parse_timestamp_series(raw["timestamp"])
    raw["dap"] = dap_series(raw["timestamp_dt"])
    return raw


def load_weights(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    need = ["plant barcode", "timestamp", "weight before", "weight after", "water amount", "completed"]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise ValueError(f"weights table missing columns: {missing}")
    df = pd.DataFrame({
        "barcode": raw["plant barcode"],
        "timestamp_text": raw["timestamp"],
        "weight_before_g": pd.to_numeric(raw["weight before"], errors="coerce"),
        "weight_after_g": pd.to_numeric(raw["weight after"], errors="coerce"),
        "dispenser_water_g": pd.to_numeric(raw["water amount"], errors="coerce"),
        "completed": parse_bool_series(raw["completed"]),
        "cartag": raw["car tag"] if "car tag" in raw.columns else "",
    })
    df["timestamp"] = parse_timestamp_series(df["timestamp_text"])
    df["dap"] = dap_series(df["timestamp"])
    df["successful"] = (df["completed"] & df["weight_before_g"].notna() & df["weight_after_g"].notna()
                        & df["timestamp"].notna())
    return df


def load_harvest(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in ("plantbarcode", "dry_wt"):
        if c not in raw.columns:
            raise ValueError(f"harvest table missing column {c}")
    out = pd.DataFrame({"plantbarcode": raw["plantbarcode"]})
    for c in ("fresh_wt", "sat_wt", "dry_wt"):
        out[c + "_g"] = pd.to_numeric(raw[c], errors="coerce") if c in raw.columns else np.nan
    out["cartag"] = raw["cartag"] if "cartag" in raw.columns else ""
    return out


def load_zoom_table(path: str) -> pd.DataFrame:
    tbl = pd.read_csv(path, sep="\t")
    tbl.columns = [c.strip() for c in tbl.columns]
    if "zoom" not in tbl.columns or "rel_area" not in tbl.columns:
        raise ValueError("zoom table needs 'zoom' and 'rel_area' columns")
    return tbl


# --------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------
def make_figure(path, harvest_join, profile_r, fit_bounded, fit_unbounded, eval_bounded, eval_unbounded):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    ax = axes[0]
    held = harvest_join[(harvest_join["heldout"] == True) & harvest_join["dry_wt_pred_g"].notna()]  # noqa: E712
    for trt, mk in (("wet", "o"), ("dry", "^")):
        sub = held[held["treatment"] == trt]
        ax.scatter(sub["dry_wt_g"], sub["dry_wt_pred_g"], s=14, marker=mk, label=f"heldout {trt}", alpha=0.7)
    if len(held):
        lim = [0, float(max(held["dry_wt_g"].max(), held["dry_wt_pred_g"].max())) * 1.05]
        ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("harvest dry mass (g, independent)")
    ax.set_ylabel("k * final-day SV area (g)")
    ax.set_title("Heldout dry-mass prediction")
    ax.legend(fontsize=7)

    ax = axes[1]
    if profile_r:
        ax.plot([p["r_year"] for p in profile_r], [p["sse"] for p in profile_r], "-o", ms=3)
        ax.set_xscale("log")
    ax.axvline(1.0, color="r", ls=":", label="runtime bound r_year<=1")
    if fit_bounded and "r_year" in fit_bounded:
        ax.scatter([fit_bounded["r_year"]], [fit_bounded["sse"]], c="r", marker="s", label="bounded fit")
    if fit_unbounded and "r_year" in fit_unbounded:
        ax.scatter([fit_unbounded["r_year"]], [fit_unbounded["sse"]], c="g", marker="*", s=60, label="unbounded fit")
    ax.set_xlabel("r_year (C optimised at each r)")
    ax.set_ylabel("TRAIN SSE (log mass)")
    ax.set_title("Rate profile / identifiability")
    ax.legend(fontsize=7)

    ax = axes[2]
    for ev, lab, mk in ((eval_bounded, "bounded r<=1", "s"), (eval_unbounded, "unbounded", "*")):
        pp = ev.get("per_plant", []) if ev else []
        if pp:
            ax.scatter([p["endpoint_obs_kg"] for p in pp], [p["endpoint_pred_kg"] for p in pp], s=16, marker=mk,
                       label=lab, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    xl = ax.get_xlim()
    ax.plot(xl, xl, "k--", lw=0.8)
    ax.set_xlabel("heldout image-derived endpoint mass (kg)")
    ax.set_ylabel("STEP-rule predicted (kg)")
    ax.set_title("Heldout endpoint: growth-cap comparison")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def run_analysis(args) -> dict:
    out = args.out
    os.makedirs(out, exist_ok=False)
    window = (args.window_start, args.window_end)
    notes = []

    images = load_images(args.images)
    weights = load_weights(args.weights)
    harvest = load_harvest(args.harvest)
    zoom_tbl = load_zoom_table(args.zoom)
    with open(args.peterson, "r", encoding="utf-8") as fh:
        peterson = json.load(fh)

    # 2. zoom calibration and daily areas
    zoom_fit = fit_zoom_calibration(zoom_tbl)
    daily, area_audit = daily_side_view_area(images, zoom_fit["a"], zoom_fit["b"])
    daily.to_csv(os.path.join(out, "daily_sv_area.csv"), index=False)

    # 3. harvest join and dry-mass calibration (TRAIN only)
    harvest_join, mass_cal = calibrate_dry_mass(harvest, daily, args.holdout_modulus, args.holdout_remainder)
    weight_barcodes = set(weights["barcode"].astype(str))
    mass_cal["harvest_ids_without_weight_join"] = sorted(
        b for b in harvest["plantbarcode"].astype(str) if b not in weight_barcodes)
    harvest_join.to_csv(os.path.join(out, "harvest_join.csv"), index=False)
    k = mass_cal["coefficient_k_g_per_normalized_px"]
    # sensitivity: in-bounds-only areas
    try:
        _, mass_cal_inb = calibrate_dry_mass(harvest, daily, args.holdout_modulus, args.holdout_remainder,
                                             column="area_norm_mean_in_bounds")
        mass_cal["sensitivity_in_bounds_only"] = {
            "coefficient_k_g_per_normalized_px": mass_cal_inb["coefficient_k_g_per_normalized_px"],
            "heldout_metrics_g": mass_cal_inb["heldout_metrics_g"], "n_train": mass_cal_inb["n_train"]}
    except Exception as exc:
        mass_cal["sensitivity_in_bounds_only"] = {"error": str(exc)}

    # 4. water accounting
    intervals = reconstruct_weighing_intervals(weights, args.dispenser_tol_g, LARGE_INTERVAL_LOSS_G)
    balances = daily_boundary_balances(weights, intervals, args.large_daily_loss_g)
    intervals.to_csv(os.path.join(out, "weighing_intervals.csv"), index=False)
    balances.to_csv(os.path.join(out, "daily_balances.csv"), index=False)
    audit = water_audit(weights, intervals, balances)
    weights_split = assign_split(weights[["barcode"]].drop_duplicates(), "barcode", args.holdout_modulus, args.holdout_remainder)
    unparsed = weights_split.loc[weights_split["genotype"].isna(), "barcode"].astype(str).tolist()
    evap = balances[balances["barcode"].isin(unparsed) & balances["complete_daily"]]
    audit["unparsed_barcodes_possible_empty_pots"] = {
        "barcodes": sorted(unparsed), "n": len(unparsed),
        "mean_complete_daily_loss_g": float(evap["balance_loss_g"].mean()) if len(evap) else None,
        "n_complete_daily": int(len(evap)),
        "note": "identified only by barcode pattern failure; treated as evaporation reference, never plants; "
                "not subtracted from plant losses (paper warns of overcorrection)",
    }

    # 5. window periods
    mass_by_bc = mass_series_from_daily(daily, k)
    periods, excl = build_window_periods(balances, mass_by_bc, window, args.max_image_gap)
    periods = assign_split(periods, "barcode", args.holdout_modulus, args.holdout_remainder)
    periods = periods[periods["genotype"].notna()].copy()
    periods.to_csv(os.path.join(out, "plant_periods.csv"), index=False)
    plants = per_plant_summary(periods) if len(periods) else pd.DataFrame()
    plants.to_csv(os.path.join(out, "plant_summary.csv"), index=False)

    # multi-day sensitivity (complete chains up to 3 days, measured totals, not extrapolated)
    bal_multi = balances.copy()
    bal_multi["complete_daily"] = (bal_multi["chain_complete"] & (bal_multi["period_days"] <= 3)
                                   & (bal_multi["balance_loss_g"] >= 0)
                                   & (bal_multi["balance_loss_g"] <= args.large_daily_loss_g * bal_multi["period_days"]))
    periods_multi, _ = build_window_periods(bal_multi, mass_by_bc, window, args.max_image_gap)
    periods_multi = assign_split(periods_multi, "barcode", args.holdout_modulus, args.holdout_remainder)
    periods_multi = periods_multi[periods_multi["genotype"].notna()]

    # 6. rates and support prior
    pooled = water_rate_summary(periods)
    rates = {
        "window_dap": list(window), "max_image_gap_days": args.max_image_gap,
        "period_exclusions": excl,
        "pooled_all": pooled,
        "pooled_train": water_rate_summary(periods[periods["heldout"] == False]),   # noqa: E712
        "pooled_heldout": water_rate_summary(periods[periods["heldout"] == True]),  # noqa: E712
        "by_treatment": {t: water_rate_summary(periods[periods["treatment"] == t]) for t in ("wet", "dry")},
        "sensitivity_including_multi_day_periods_le3": water_rate_summary(periods_multi),
        "T_fraction_sensitivity": t_fraction_sensitivity(pooled),
        "bootstrap_genotype": bootstrap_by_genotype(periods, args.n_boot, args.seed) if len(periods) else {},
        "definitions": {
            "q_m3_per_kg_day": "sum(U)/integral(B dt); U = pot water loss (ET) in m3, B image-derived dry mass kg",
            "C_balance_kg_per_m3": "integral(B dt)/(365*sum(U)); annual balance SUPPORT threshold, not biomass per water",
            "production_ratio_kg_per_m3": "sum(delta_B)/sum(U) over the same periods (empirical)",
        },
    }
    if len(plants):
        q = plants["q_m3_per_kg_day"].dropna()
        rates["per_plant_q_quantiles"] = {p: float(q.quantile(p / 100)) for p in (5, 25, 50, 75, 95)}
        rates["per_plant_n"] = int(len(plants))
        geno_tbl = plants.groupby(["genotype", "treatment"]).agg(
            n=("barcode", "size"), q_mean=("q_m3_per_kg_day", "mean"),
            C_balance_mean=("C_balance_kg_per_m3", "mean"),
            production_ratio_mean=("production_ratio_kg_per_m3", "mean")).reset_index()
        geno_tbl.to_csv(os.path.join(out, "genotype_treatment_summary.csv"), index=False)

    # 7. STEP diagnostic fit
    W_by_bc = {}
    for r in balances[balances["complete_daily"]].itertuples():
        W_by_bc.setdefault(r.barcode, {})[int(r.dap_start)] = float(r.balance_loss_g)
    days = list(range(window[0], window[1] + 1))
    trajs = []
    traj_excl = {"no_mass": 0, "no_water": 0, "run_shorter_than_block": 0, "unparsed_barcode": 0}
    for bc in sorted(set(W_by_bc) | set(mass_by_bc)):
        info = parse_barcode(bc)
        if info is None:
            traj_excl["unparsed_barcode"] += 1
            continue
        if bc not in mass_by_bc:
            traj_excl["no_mass"] += 1
            continue
        if bc not in W_by_bc:
            traj_excl["no_water"] += 1
            continue
        B_by_day = interpolate_series(mass_by_bc[bc], days, args.max_image_gap)
        tr = build_step_trajectory(B_by_day, W_by_bc[bc], window[0], window[1])
        if tr is None:
            traj_excl["run_shorter_than_block"] += 1
            continue
        tr.update({"barcode": bc, "genotype": info["genotype"], "treatment": info["treatment"],
                   "heldout": genotype_is_heldout(info["genotype"], args.holdout_modulus, args.holdout_remainder)})
        trajs.append(tr)
    train_tr = [t for t in trajs if not t["heldout"]]
    held_tr = [t for t in trajs if t["heldout"]]
    step = {"n_trajectories": len(trajs), "n_train": len(train_tr), "n_heldout": len(held_tr),
            "trajectory_exclusions": traj_excl, "block_days": STEP_BLOCK_DAYS,
            "assumptions": ["activity = temperature = live fraction = 1", "no seasonal extra biomass",
                            "gain capped at M-B", "M scenarios are multiples of the maximum TRAIN endpoint mass, "
                            "not mature mass inferred from a seedling harvest"]}
    fit_b = fit_u = None
    eval_b = eval_u = {}
    prof_r = []
    train_end_masses = [o for t in train_tr for o in t["obs_kg"] if _finite(o)]
    if train_tr and train_end_masses:
        max_end = max(train_end_masses)
        scenarios = {}
        for mult in (2.0, 4.0, 10.0):
            M = mult * max_end
            fb = fit_step_parameters(train_tr, M, r_max=1.0)
            fu = fit_step_parameters(train_tr, M, r_max=1e4)
            scenarios[f"M_{mult:g}x"] = {"M_kg": M, "bounded_r_le_1": fb, "unbounded_positive": fu}
        step["max_train_endpoint_mass_kg"] = max_end
        step["scenarios_train"] = scenarios
        sel_b = min(scenarios, key=lambda s: scenarios[s]["bounded_r_le_1"].get("sse", np.inf))
        sel_u = min(scenarios, key=lambda s: scenarios[s]["unbounded_positive"].get("sse", np.inf))
        fit_b, Mb = scenarios[sel_b]["bounded_r_le_1"], scenarios[sel_b]["M_kg"]
        fit_u, Mu = scenarios[sel_u]["unbounded_positive"], scenarios[sel_u]["M_kg"]
        step["selected_scenario_bounded"] = sel_b
        step["selected_scenario_unbounded"] = sel_u
        step["fitted_C_bounded_kg_per_m3"] = fit_b.get("C")
        step["fitted_r_bounded"] = fit_b.get("r_year")
        step["fitted_C_unbounded_kg_per_m3"] = fit_u.get("C")
        step["fitted_r_unbounded"] = fit_u.get("r_year")
        step["water_balance_C_kg_per_m3_for_comparison"] = pooled.get("C_balance_kg_per_m3")
        eval_b = evaluate_step(held_tr, fit_b["C"], fit_b["r_year"], Mb) if held_tr else {}
        eval_u = evaluate_step(held_tr, fit_u["C"], fit_u["r_year"], Mu) if held_tr else {}
        step["heldout_bounded"] = {k_: v for k_, v in eval_b.items() if k_ != "per_plant"}
        step["heldout_unbounded"] = {k_: v for k_, v in eval_u.items() if k_ != "per_plant"}
        step["train_eval_bounded"] = {k_: v for k_, v in evaluate_step(train_tr, fit_b["C"], fit_b["r_year"], Mb).items() if k_ != "per_plant"}
        step["train_eval_unbounded"] = {k_: v for k_, v in evaluate_step(train_tr, fit_u["C"], fit_u["r_year"], Mu).items() if k_ != "per_plant"}
        prof_r = profile_rate(train_tr, Mu, np.logspace(-1, 3, 21))
        prof_c = profile_capacity(train_tr, Mu, np.logspace(-3, 3, 19))
        step["profile_r_fixed"] = prof_r
        step["profile_C_fixed"] = prof_c
        obs_factor = step["train_eval_unbounded"].get("observed_growth_factor_median")
        run_len = int(np.median([t["end_dap"] - t["start_dap"] for t in train_tr]))
        step["rate_cap_diagnostic"] = {
            "median_train_run_days": run_len,
            "median_observed_growth_factor_train": obs_factor,
            "max_normal_growth_factor_r1": max_normal_growth_factor(1.0, run_len),
            "min_weekly_euler_rate_for_observed_factor": min_weekly_euler_rate(obs_factor, run_len) if obs_factor else None,
            "bounded_fit_hits_r_upper_bound": fit_b.get("r_at_upper_bound"),
        }
        if fit_b.get("r_at_upper_bound"):
            notes.append("Bounded (r_year<=1) STEP fit sits at its upper bound: the runtime rate cap cannot "
                         "reproduce observed seedling growth in the window; reported, not hidden.")
        pd.DataFrame(eval_u.get("per_plant", [])).to_csv(os.path.join(out, "heldout_step_unbounded.csv"), index=False)
        pd.DataFrame(eval_b.get("per_plant", [])).to_csv(os.path.join(out, "heldout_step_bounded.csv"), index=False)
    else:
        notes.append("No TRAIN trajectories satisfied the STEP block requirements; STEP fit skipped.")

    # 8. Peterson
    pet = peterson_block(peterson)

    make_figure(os.path.join(out, "study_parameters.png"), harvest_join, prof_r, fit_b, fit_u, eval_b, eval_u)

    candidate = {
        "study": STUDY_ID, "revision": REVISION, "generated_on": date.today().isoformat(),
        "inputs": {"images": args.images, "weights": args.weights, "harvest": args.harvest,
                   "zoom": args.zoom, "peterson": args.peterson},
        "settings": {"planting_date": PLANTING_DATE.isoformat(), "window_dap": list(window),
                     "holdout": f"sha256(genotype) % {args.holdout_modulus} == {args.holdout_remainder}",
                     "seed": args.seed, "n_boot": args.n_boot, "max_image_gap_days": args.max_image_gap,
                     "large_daily_loss_g": args.large_daily_loss_g, "dispenser_tol_g": args.dispenser_tol_g},
        "zoom_calibration": zoom_fit,
        "dry_mass_calibration": mass_cal,
        "water_rate_and_support": rates,
        "step_diagnostic_fit": {k_: v for k_, v in step.items() if k_ not in ("profile_r_fixed", "profile_C_fixed")},
        "peterson_neofotis_2004": pet,
        "notes": notes,
        "unidentified_or_not_claimed": [
            "ET->T calibration (only conditional T fractions reported)",
            "mature mass M (scenarios only)", "canopy density", "seasonal turnover", "annual phenology",
            "rooting layers", "empty-pot identity beyond barcode-pattern failure"],
    }
    diagnostics = {"image_area_audit": area_audit, "water_accounting_audit": audit,
                   "step_profiles": {"r_fixed": step.get("profile_r_fixed", []), "C_fixed": step.get("profile_C_fixed", [])},
                   "window_period_exclusions": excl, "trajectory_exclusions": traj_excl}
    with open(os.path.join(out, "candidate_fragments.json"), "w", encoding="utf-8") as fh:
        json.dump(_jsonable(candidate), fh, indent=2)
    with open(os.path.join(out, "diagnostics.json"), "w", encoding="utf-8") as fh:
        json.dump(_jsonable(diagnostics), fh, indent=2)
    return candidate


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--harvest", required=True)
    p.add_argument("--zoom", required=True)
    p.add_argument("--peterson", required=True)
    p.add_argument("--out", required=True, help="new output directory (existing directories are refused)")
    p.add_argument("--seed", type=int, default=20261101)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--holdout-modulus", type=int, default=5)
    p.add_argument("--holdout-remainder", type=int, default=0)
    p.add_argument("--window-start", type=int, default=WINDOW_START_DAP)
    p.add_argument("--window-end", type=int, default=WINDOW_END_DAP)
    p.add_argument("--max-image-gap", type=int, default=MAX_IMAGE_GAP_DAYS)
    p.add_argument("--large-daily-loss-g", type=float, default=LARGE_DAILY_LOSS_G)
    p.add_argument("--dispenser-tol-g", type=float, default=DISPENSER_TOL_G)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not (100 <= args.n_boot <= 200):
        print("warning: contract asks for 100-200 bootstrap resamples", file=sys.stderr)
    if os.path.exists(args.out):
        print(f"refusing to overwrite existing output directory: {args.out}", file=sys.stderr)
        return 2
    run_analysis(args)
    print(f"wrote candidate fragments to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
