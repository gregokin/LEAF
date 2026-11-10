"""Unit tests for study_parameters.py using small synthetic inputs (no public data).

Expected values are derived from the study contract (hand computation), not from
re-running the implementation.  Run with:  python -m pytest test_study_parameters.py
"""
import hashlib
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_parameters as sp  # noqa: E402


# ------------------------------------------------------------------ helpers
def _weights(rows):
    """rows: (barcode, timestamp_text, before, after, dispenser, completed)."""
    df = pd.DataFrame(rows, columns=["barcode", "timestamp_text", "weight_before_g", "weight_after_g",
                                     "dispenser_water_g", "completed"])
    df["timestamp"] = sp.parse_timestamp_series(df["timestamp_text"])
    df["dap"] = sp.dap_series(df["timestamp"])
    df["successful"] = (df["completed"] & df["weight_before_g"].notna() & df["weight_after_g"].notna()
                        & df["timestamp"].notna())
    return df


# ------------------------------------------------------------------ 1. calendar / genotype / split
def test_dap_uses_calendar_date_not_posix():
    assert sp.dap_from_timestamp(sp.parse_naive_timestamp("2014-01-13 00:00:00.000")) == 0
    assert sp.dap_from_timestamp(sp.parse_naive_timestamp("2014-01-30 23:59:59.999")) == 17
    assert sp.dap_from_timestamp(sp.parse_naive_timestamp("2014-01-31 00:00:00")) == 18
    assert sp.parse_naive_timestamp("not a date") is None


def test_barcode_parsing_and_treatment():
    p = sp.parse_barcode("Dr33AA001347")
    assert p["genotype"] == "Dr33" and p["treatment"] == "wet" and not p["is_parent"]
    p = sp.parse_barcode("Dr102AB001763")
    assert p["genotype"] == "Dr102" and p["treatment"] == "dry"
    p = sp.parse_barcode("Dp1AB000010")
    assert p["genotype"] == "Dp1" and p["treatment"] == "dry" and p["is_parent"]
    assert sp.parse_barcode("EMPTY01") is None
    assert sp.parse_barcode(" Dr33AA001347") is None   # exact barcodes only
    assert sp.parse_barcode(None) is None


def test_holdout_is_deterministic_sha256_modulo():
    g = "Dr33"
    expected = int(hashlib.sha256(g.encode()).hexdigest(), 16) % 5 == 0
    assert sp.genotype_is_heldout(g, 5, 0) == expected
    assert sp.genotype_is_heldout(g, 5, 0) == sp.genotype_is_heldout(g, 5, 0)
    # every residue class is used exactly once per genotype
    assert sum(sp.genotype_is_heldout(g, 5, r) for r in range(5)) == 1


def test_split_keeps_genotype_together_and_has_no_leakage():
    barcodes = []
    for n in range(1, 60):
        for trt in ("AA", "AB"):
            for rep in range(3):
                barcodes.append(f"Dr{n}{trt}{n * 10 + rep:06d}")
    df = sp.assign_split(pd.DataFrame({"plantbarcode": barcodes}), "plantbarcode", 5, 0)
    train = set(df.loc[df["heldout"] == False, "genotype"])  # noqa: E712
    held = set(df.loc[df["heldout"] == True, "genotype"])    # noqa: E712
    assert train.isdisjoint(held)
    assert held, "expected some heldout genotypes among 59"
    assert df.groupby("genotype")["heldout"].nunique().max() == 1
    assert df.groupby("genotype")["treatment"].nunique().min() == 2  # wet and dry travel together


# ------------------------------------------------------------------ 2. zoom calibration
def test_camera_zoom_mapping_endpoints():
    assert sp.camera_zoom(1) == pytest.approx(1.0)
    assert sp.camera_zoom(6000) == pytest.approx(6.0)
    assert sp.parse_zoom_label("z1000") == 1000 and sp.parse_zoom_label("z1") == 1
    assert sp.parse_zoom_label("bogus") is None


def test_zoom_fit_recovers_synthetic_exponential():
    zooms = np.array([1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000])
    rel = 0.9 * np.exp(0.7 * sp.camera_zoom(zooms))
    fit = sp.fit_zoom_calibration(pd.DataFrame({"zoom": zooms, "rel_area": rel}))
    assert fit["a"] == pytest.approx(0.9, rel=1e-4)
    assert fit["b"] == pytest.approx(0.7, rel=1e-4)
    assert sp.predicted_rel_area(2000, fit["a"], fit["b"]) == pytest.approx(rel[4], rel=1e-4)


def test_daily_area_excludes_nonpositive_and_counts_out_of_bounds():
    img = pd.DataFrame({
        "timestamp": ["2014-02-01 08:00:00.000"] * 5,
        "frame": ["0", "90", "180", "270", "0"],
        "zoom": ["z1", "z1", "z1", "z1", "z1"],
        "plantbarcode": ["Dr1AA000001"] * 4 + ["Dr1AA000001"],
        "camera": ["SV", "SV", "SV", "SV", "TV"],
        "area": ["100", "300", "0", "-5", "999"],
        "in_bounds": ["True", "False", "True", "True", "True"],
    })
    img["timestamp_dt"] = sp.parse_timestamp_series(img["timestamp"])
    img["dap"] = sp.dap_series(img["timestamp_dt"])
    a, b = 1.0, 0.0   # predicted rel area == 1 at every zoom
    daily, audit = sp.daily_side_view_area(img, a, b)
    assert len(daily) == 1
    assert daily["area_norm_mean"].iloc[0] == pytest.approx(200.0)      # mean of 100 and 300 only
    assert daily["n_valid"].iloc[0] == 2
    assert daily["n_out_of_bounds"].iloc[0] == 1
    assert daily["area_norm_mean_in_bounds"].iloc[0] == pytest.approx(100.0)
    assert audit["sv_images_nonpositive_or_missing_area"] == 2
    assert audit["sv_images_total"] == 4   # TV image never enters


# ------------------------------------------------------------------ 3. dry-mass coefficient
def test_zero_intercept_fit_uses_train_only():
    rng = np.random.default_rng(1)
    train_area = rng.uniform(1e4, 1e5, 20)
    k_true = 2.5e-4
    # mass = k*area exactly on train; heldout rows have a very different slope
    barcodes = [f"Dr{n}AA{n:06d}" for n in range(1, 400)]
    split = sp.assign_split(pd.DataFrame({"plantbarcode": barcodes}), "plantbarcode", 5, 0)
    train_bcs = split.loc[split["heldout"] == False, "plantbarcode"].tolist()[:20]  # noqa: E712
    held_bcs = split.loc[split["heldout"] == True, "plantbarcode"].tolist()[:5]     # noqa: E712
    daily_rows, harvest_rows = [], []
    for bc, a in zip(train_bcs, train_area):
        daily_rows.append({"plantbarcode": bc, "dap": 30, "area_norm_mean": a})
        daily_rows.append({"plantbarcode": bc, "dap": 25, "area_norm_mean": a * 0.5})  # earlier day ignored
        harvest_rows.append({"plantbarcode": bc, "dry_wt_g": k_true * a, "fresh_wt_g": np.nan, "sat_wt_g": np.nan, "cartag": ""})
    for bc in held_bcs:
        daily_rows.append({"plantbarcode": bc, "dap": 30, "area_norm_mean": 5e4})
        harvest_rows.append({"plantbarcode": bc, "dry_wt_g": 10 * k_true * 5e4, "fresh_wt_g": np.nan, "sat_wt_g": np.nan, "cartag": ""})
    harvest_rows.append({"plantbarcode": "Dr999AA000999", "dry_wt_g": 1.0, "fresh_wt_g": np.nan, "sat_wt_g": np.nan, "cartag": ""})
    join, summary = sp.calibrate_dry_mass(pd.DataFrame(harvest_rows), pd.DataFrame(daily_rows), 5, 0)
    assert summary["coefficient_k_g_per_normalized_px"] == pytest.approx(k_true, rel=1e-9)
    assert summary["n_train"] == 20 and summary["n_heldout"] == 5
    assert summary["harvest_ids_without_image_join"] == ["Dr999AA000999"]
    assert summary["heldout_metrics_g"]["median_abs_relative_error"] == pytest.approx(0.9, rel=1e-9)
    assert summary["train_metrics_g"]["rmse"] == pytest.approx(0.0, abs=1e-9)


def test_zero_intercept_rejects_nonpositive():
    with pytest.raises(ValueError):
        sp.fit_zero_intercept_coefficient([1.0, 2.0], [-1.0, -2.0])


# ------------------------------------------------------------------ 4. water accounting
def test_irrigation_storage_closure_uses_actual_mass_not_dispenser():
    bc = "Dr1AA000001"
    w = _weights([
        (bc, "2014-02-02 10:00:00.000", 500.0, 600.0, 100.0, True),
        (bc, "2014-02-02 16:00:00.000", 580.0, 640.0, 60.0, True),
        (bc, "2014-02-03 10:00:00.000", 520.0, 620.0, 94.0, True),   # dispenser says 94, scale says 100 (>5 g)
    ])
    iv = sp.reconstruct_weighing_intervals(w)
    assert len(iv) == 2
    assert iv["loss_g"].tolist() == pytest.approx([20.0, 120.0])
    assert iv["irrigation_g"].tolist() == pytest.approx([60.0, 100.0])
    assert iv["dt_days"].iloc[0] == pytest.approx(0.25)
    assert "dispenser_mismatch" in iv["reasons"].iloc[1]
    assert iv["usable"].all()
    bal = sp.daily_boundary_balances(w, iv)
    assert len(bal) == 1
    row = bal.iloc[0]
    # W_after(day20 last)=640, irrigation in (t0,t1] = 100 (actual mass), W_after(day21 last)=620
    assert row["irrigation_sum_g"] == pytest.approx(100.0)
    assert row["storage_change_g"] == pytest.approx(-20.0)
    assert row["balance_loss_g"] == pytest.approx(120.0)
    assert row["interval_loss_sum_g"] == pytest.approx(120.0)
    assert abs(row["closure_residual_g"]) < 1e-9
    assert row["dispenser_sum_g"] == pytest.approx(94.0)     # kept separately, never used as loss
    assert bool(row["complete_daily"]) and row["period_days"] == 1
    audit = sp.water_audit(w, iv, bal)
    assert audit["balance_loss_sum_complete_chain_g"] == pytest.approx(audit["interval_loss_sum_complete_chain_g"])


def test_dispenser_mismatch_is_strictly_greater_than_tolerance():
    bc = "Dr1AA000001"
    w = _weights([
        (bc, "2014-02-02 10:00:00.000", 500.0, 600.0, 100.0, True),
        (bc, "2014-02-03 10:00:00.000", 520.0, 620.0, 95.0, True),   # exactly 5 g: not a mismatch
        (bc, "2014-02-04 10:00:00.000", 520.0, 620.0, 94.9, True),   # 5.1 g: mismatch
    ])
    iv = sp.reconstruct_weighing_intervals(w, dispenser_tol_g=5.0)
    assert "dispenser_mismatch" not in iv["reasons"].iloc[0]
    assert "dispenser_mismatch" in iv["reasons"].iloc[1]
    assert iv["loss_g"].tolist() == pytest.approx([80.0, 100.0])      # physical losses unaffected


def test_image_placeholder_rows_do_not_form_boundaries(tmp_path):
    """Independent 3-row counterexample: real -> image placeholder (-1,-1,-1) -> real.
    Physical loss is 110 - 105 = 5 g; the placeholder must neither break the chain nor
    become the last-day boundary (which would yield -106 g)."""
    csv = tmp_path / "snap.csv"
    csv.write_text(
        "plant barcode,timestamp,weight before,weight after,water amount,completed,car tag\n"
        "Dr5AA000005,2014-02-02 10:00:00.000,100.0,110.0,10.0,true,1001\n"
        "Dr5AA000005,2014-02-02 14:00:00.000,-1,-1,-1,true,1001\n"
        "Dr5AA000005,2014-02-03 10:00:00.000,105.0,115.0,10.0,true,1001\n"
    )
    w = sp.load_weights(str(csv))
    assert w["placeholder"].tolist() == [False, True, False]
    assert w["successful"].tolist() == [True, False, True]
    iv = sp.reconstruct_weighing_intervals(w)
    assert len(iv) == 1 and iv["usable"].iloc[0]
    assert iv["loss_g"].iloc[0] == pytest.approx(5.0)
    assert iv["dt_days"].iloc[0] == pytest.approx(1.0)                # real -> real, placeholder skipped
    bal = sp.daily_boundary_balances(w, iv)
    assert len(bal) == 1
    row = bal.iloc[0]
    assert (row["dap_start"], row["dap_end"]) == (20, 21)
    assert row["weight_after_start_g"] == pytest.approx(110.0)         # not the -1 placeholder
    assert row["weight_after_end_g"] == pytest.approx(115.0)
    assert row["irrigation_sum_g"] == pytest.approx(10.0)
    assert row["storage_change_g"] == pytest.approx(5.0)
    assert row["balance_loss_g"] == pytest.approx(5.0)
    assert row["interval_loss_sum_g"] == pytest.approx(5.0) and abs(row["closure_residual_g"]) < 1e-9
    assert row["n_records"] == 1 and row["n_unsuccessful_records"] == 0
    assert bool(row["chain_complete"]) and bool(row["complete_daily"])
    audit = sp.water_audit(w, iv, bal)
    assert audit["records_image_placeholders_excluded_from_chains"] == 1
    assert audit["records_physical"] == 2 and audit["records_unsuccessful"] == 0
    assert audit["daily_balances_primary_study_qc"] == 1


def test_partial_sentinel_is_a_genuine_incomplete_record(tmp_path):
    csv = tmp_path / "snap.csv"
    csv.write_text(
        "plant barcode,timestamp,weight before,weight after,water amount,completed\n"
        "Dr6AA000006,2014-02-02 10:00:00.000,100.0,110.0,10.0,true\n"
        "Dr6AA000006,2014-02-03 10:00:00.000,105.0,-1,-1,true\n"
        "Dr6AA000006,2014-02-04 10:00:00.000,100.0,112.0,12.0,true\n"
    )
    w = sp.load_weights(str(csv))
    assert w["sentinel_partial"].tolist() == [False, True, False]
    assert not w["successful"].iloc[1]
    iv = sp.reconstruct_weighing_intervals(w)
    assert len(iv) == 2 and not iv["usable"].any()                     # stays in chain, blocks it
    bal = sp.daily_boundary_balances(w, iv)
    assert len(bal) == 1 and not bal.iloc[0]["chain_complete"]         # no extrapolation across it
    assert "multi_day_period" in bal.iloc[0]["reasons"]


def test_study_qc_policy_reasons_and_sensitivity_selection():
    assert sp.study_qc_reasons(17, 250.0, "wet") == ["qc_dap17_loss_gt_200g"]
    assert sp.study_qc_reasons(20, 250.0, "wet") == []
    assert sp.study_qc_reasons(32, 150.0, "dry") == ["qc_dry_dap32_loss_gt_145g"]
    assert sp.study_qc_reasons(32, 150.0, "wet") == []
    assert sp.study_qc_reasons(32, 210.0, "dry") == ["qc_dap32_loss_gt_200g", "qc_dry_dap32_loss_gt_145g"]
    assert sp.study_qc_reasons(33, 0.0, "wet") == ["qc_nonpositive_daily_loss"]
    assert sp.study_qc_reasons(25, -3.0, None) == ["qc_nonpositive_daily_loss"]
    # DAP 17 = 2014-01-30 (local calendar). A 250 g day labelled by its weighing day (DAP 17)
    # is excluded from the primary selection, retained (unclipped) in the sensitivity selection.
    bc = "Dr7AB000007"
    w = _weights([
        (bc, "2014-01-29 10:00:00.000", 500.0, 800.0, 300.0, True),
        (bc, "2014-01-30 10:00:00.000", 550.0, 800.0, 250.0, True),
        (bc, "2014-01-31 10:00:00.000", 550.0, 800.0, 250.0, True),
    ])
    iv = sp.reconstruct_weighing_intervals(w)
    bal = sp.daily_boundary_balances(w, iv, qc_day_label="end")
    assert bal["qc_dap_label"].tolist() == [17, 18]
    assert bal["balance_loss_g"].tolist() == pytest.approx([250.0, 250.0])   # never clipped
    assert bal["study_qc_reasons"].tolist() == ["qc_dap17_loss_gt_200g", ""]
    assert bal["complete_daily"].tolist() == [False, True]
    assert bal["complete_daily_sensitivity"].tolist() == [True, True]
    bal_s = sp.daily_boundary_balances(w, iv, qc_day_label="start")
    assert bal_s["qc_dap_label"].tolist() == [16, 17]
    assert bal_s["complete_daily"].tolist() == [True, False]
    audit = sp.water_audit(w, iv, bal)
    assert audit["study_qc_flag_counts_one_day_complete_chains"] == {"qc_dap17_loss_gt_200g": 1}
    assert audit["daily_balances_primary_study_qc"] == 1
    assert audit["daily_balances_sensitivity_retained_flagged"] == 2
    assert audit["study_qc_excluded_loss_sum_g"] == pytest.approx(250.0)


def test_multi_record_day_closure_identity_holds_generally():
    rng = np.random.default_rng(7)
    bc = "Dr2AB000002"
    rows, after = [], 700.0
    t = pd.Timestamp("2014-02-05 08:00:00")
    for i in range(12):
        before = after - rng.uniform(0, 80)
        new_after = before + rng.uniform(0, 150)
        rows.append((bc, t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], before, new_after, rng.uniform(0, 150), True))
        after = new_after
        t += pd.Timedelta(hours=float(rng.choice([4, 8, 12, 24])))
    w = _weights(rows)
    iv = sp.reconstruct_weighing_intervals(w)
    bal = sp.daily_boundary_balances(w, iv)
    assert bal["chain_complete"].all()
    assert (bal["closure_residual_g"].abs() < 1e-9).all()
    # total of daily balances equals the sum of interval losses between first and last boundary
    assert bal["balance_loss_g"].sum() == pytest.approx(bal["interval_loss_sum_g"].sum())


def test_incomplete_and_negative_intervals_are_flagged_not_clipped():
    bc = "Dr3AA000003"
    w = _weights([
        (bc, "2014-02-02 10:00:00.000", 500.0, 600.0, 100.0, True),
        (bc, "2014-02-03 10:00:00.000", None, None, 0.0, False),     # failed weighing
        (bc, "2014-02-04 10:00:00.000", 480.0, 600.0, 120.0, True),
        (bc, "2014-02-05 10:00:00.000", 680.0, 700.0, 20.0, True),   # mass appeared: negative loss
        (bc, "2014-02-06 10:00:00.000", 650.0, 700.0, 50.0, True),
    ])
    w["weight_before_g"] = pd.to_numeric(w["weight_before_g"])
    w["weight_after_g"] = pd.to_numeric(w["weight_after_g"])
    w["successful"] = w["completed"] & w["weight_before_g"].notna() & w["weight_after_g"].notna()
    iv = sp.reconstruct_weighing_intervals(w)
    assert len(iv) == 4
    assert not iv["usable"].iloc[0] and "current_record_incomplete" in iv["reasons"].iloc[0]
    assert not iv["usable"].iloc[1] and "previous_record_incomplete" in iv["reasons"].iloc[1]
    assert iv["loss_g"].iloc[2] == pytest.approx(-80.0)            # retained, not clipped to zero
    assert "negative_loss" in iv["reasons"].iloc[2] and iv["usable"].iloc[2]
    assert iv["loss_g"].iloc[3] == pytest.approx(50.0)
    bal = sp.daily_boundary_balances(w, iv)
    # boundaries at DAP 20 -> 22 (spanning failed record), 22 -> 23, 23 -> 24
    assert bal["dap_start"].tolist() == [20, 22, 23]
    first = bal.iloc[0]
    assert not first["chain_complete"] and "chain_incomplete" in first["reasons"]
    assert "multi_day_period" in first["reasons"] and not first["complete_daily"]
    assert math.isnan(first["interval_loss_sum_g"])
    neg = bal.iloc[1]
    assert neg["balance_loss_g"] == pytest.approx(-80.0) and "negative_daily_loss" in neg["reasons"]
    assert not neg["complete_daily"]
    assert bal.iloc[2]["complete_daily"] and bal.iloc[2]["balance_loss_g"] == pytest.approx(50.0)
    audit = sp.water_audit(w, iv, bal)
    assert audit["records_unsuccessful"] == 1
    assert audit["interval_flag_counts"]["negative_loss"] == 1
    assert audit["daily_balance_flag_counts"]["chain_incomplete"] == 1
    assert audit["daily_balances_complete_daily"] == 1


# ------------------------------------------------------------------ 5/6. interpolation, units
def test_interpolation_between_neighbours_only_no_extrapolation():
    obs = {17: 1.0, 20: 4.0, 25: 9.0}
    s = sp.interpolate_series(obs, range(15, 28), max_gap=3)
    assert 15 not in s and 16 not in s and 26 not in s           # no extrapolation
    assert s[18][0] == pytest.approx(2.0) and s[18][1] == "interpolated"
    assert s[19][0] == pytest.approx(3.0)
    assert s[20] == (4.0, "observed")
    assert 22 not in s                                          # gap of 5 days > 3
    assert s[25] == (9.0, "observed")


def test_rate_and_support_units_kg_m3_day():
    periods = pd.DataFrame({
        "barcode": ["Dr1AA000001"], "dap_start": [20], "dap_end": [21], "dt_days": [1],
        "loss_g": [1000.0], "U_m3": [1000.0 * 1e-6], "B_start_kg": [1.0], "B_end_kg": [1.0],
    })
    s = sp.water_rate_summary(periods)
    assert s["q_m3_per_kg_day"] == pytest.approx(1e-3)             # 1 litre per kg per day
    assert s["C_balance_kg_per_m3"] == pytest.approx(1.0 / (365 * 1e-3))
    assert s["production_ratio_kg_per_m3"] == pytest.approx(0.0)
    # a growing plant: 1 -> 3 kg over one day, 2 litres: integral = 2 kg*day, delta_B/U = 2/0.002
    periods2 = periods.assign(B_end_kg=3.0, loss_g=2000.0, U_m3=2e-3)
    s2 = sp.water_rate_summary(periods2)
    assert s2["q_m3_per_kg_day"] == pytest.approx(2e-3 / 2.0)
    assert s2["production_ratio_kg_per_m3"] == pytest.approx(1000.0)
    sens = sp.t_fraction_sensitivity(s2)
    assert sens["0.8"]["q_T_m3_per_kg_day"] == pytest.approx(0.8 * s2["q_m3_per_kg_day"])
    assert sens["0.8"]["C_balance_T_kg_per_m3"] == pytest.approx(s2["C_balance_kg_per_m3"] / 0.8)


def test_bootstrap_resamples_genotypes_as_units():
    rows = []
    for g in ("Dr1", "Dr2", "Dr3", "Dr4"):
        for rep in range(2):
            rows.append({"barcode": f"{g}AA00000{rep}", "genotype": g, "treatment": "wet", "heldout": False,
                         "dap_start": 20, "dap_end": 21, "dt_days": 1, "loss_g": 100.0 * (1 + int(g[2:])),
                         "U_m3": 1e-4 * (1 + int(g[2:])), "B_start_kg": 0.001, "B_end_kg": 0.002})
    periods = pd.DataFrame(rows)
    b1 = sp.bootstrap_by_genotype(periods, 100, 3)
    b2 = sp.bootstrap_by_genotype(periods, 100, 3)
    assert b1 == b2 and b1["n_genotypes"] == 4 and b1["n_boot"] == 100
    lo, hi = b1["q_m3_per_kg_day"]["p2.5"], b1["q_m3_per_kg_day"]["p97.5"]
    full = sp.water_rate_summary(periods)["q_m3_per_kg_day"]
    assert lo <= full <= hi


def test_candidate_block_ignores_heldout_periods():
    rows = []
    for n in range(1, 400):
        g = f"Dr{n}"
        held = sp.genotype_is_heldout(g, 5, 0)
        rows.append({"barcode": f"{g}AA{n:06d}", "genotype": g, "treatment": "wet", "heldout": held,
                     "dap_start": 20, "dap_end": 21, "dt_days": 1,
                     "loss_g": 100.0 if not held else 5000.0, "U_m3": 1e-4 if not held else 5e-3,
                     "B_start_kg": 0.001, "B_end_kg": 0.0012})
    periods = pd.DataFrame(rows)
    block = sp.candidate_rate_block(periods, 100, 11)
    train_only = sp.water_rate_summary(periods[periods["heldout"] == False])  # noqa: E712
    assert block["pooled"] == train_only
    assert block["pooled"]["q_m3_per_kg_day"] == pytest.approx(1e-4 / 0.0011)
    # perturbing heldout rows must not move any candidate number
    periods2 = periods.copy()
    periods2.loc[periods2["heldout"] == True, "U_m3"] *= 10  # noqa: E712
    assert sp.candidate_rate_block(periods2, 100, 11) == block
    assert "conditions on the single fitted image calibration" in block["conditioning_note"]


# ------------------------------------------------------------------ 7. STEP rule and rate cap
def test_step_gain_water_limit_and_cap():
    # abundant water: wf=1 ; B=1, dt=365, r=1, M=inf-like -> gain = 0.95
    assert sp.step_gain(1.0, 365.0, 1e9, 1.0, 1.0, 1e12) == pytest.approx(0.95, rel=1e-6)
    # required = 1*7/365 kg; available = C*U = 0.5*required -> wf=0.5
    req = 7.0 / 365.0
    g_full = sp.step_gain(1.0, 7.0, req, 1.0, 1.0, 1e12)
    g_half = sp.step_gain(1.0, 7.0, 0.5 * req, 1.0, 1.0, 1e12)
    assert g_half == pytest.approx(0.5 * g_full)
    # cap at M-B
    assert sp.step_gain(0.9, 365.0, 1e9, 1.0, 50.0, 1.0) == pytest.approx(0.1)


def test_rate_cap_cannot_reproduce_seedling_growth():
    # 16-day window, r_year = 1 -> at most a few percent of growth
    assert sp.max_normal_growth_factor(1.0, 16) < 1.05
    assert sp.weekly_euler_periods(16) == [7, 7, 2]
    assert sp.weekly_euler_periods(39) == [7, 7, 7, 7, 7, 4]
    # observed Setaria-like growth: RGR 0.1/day over 16 days -> factor ~5
    factor = math.exp(0.1 * 16)
    r_needed = sp.min_weekly_euler_rate(factor, 16)
    assert r_needed > 10
    assert sp.max_normal_growth_factor(r_needed, 16) == pytest.approx(factor, rel=1e-8)
    # simulation with r<=1, unlimited water and huge M cannot reach the factor
    traj = sp.simulate_step(1e-3, [(7.0, 1.0), (7.0, 1.0), (2.0, 1.0)], C=1e3, r_year=1.0, M_kg=1e3)
    assert traj[-1] / 1e-3 < 1.05 < factor
    # bounded fit on a trajectory that grows 5x sits at the r upper bound; unbounded fit finds r >> 1
    tr = {"barcode": "x", "genotype": "Dr1", "treatment": "wet", "B0_kg": 1e-3,
          "periods": [(7.0, 5e-4), (7.0, 5e-4), (2.0, 2e-4)],
          "obs_kg": sp.simulate_step(1e-3, [(7.0, 5e-4), (7.0, 5e-4), (2.0, 2e-4)], 1e3, r_needed, 1e3)}
    fb = sp.fit_step_parameters([tr], 1e3, r_max=1.0)
    fu = sp.fit_step_parameters([tr], 1e3, r_max=1e4)
    assert fb["r_at_upper_bound"] and fb["sse"] > 1.0
    assert fu["r_year"] == pytest.approx(r_needed, rel=2e-2) and fu["sse"] < 1e-4


def test_profile_exposes_flat_C_direction_when_water_is_not_limiting():
    tr = {"barcode": "x", "genotype": "Dr1", "treatment": "wet", "B0_kg": 1e-3,
          "periods": [(7.0, 5e-4), (7.0, 5e-4)], "obs_kg": None}
    tr["obs_kg"] = sp.simulate_step(1e-3, tr["periods"], 1e3, 20.0, 1e3)
    prof = sp.profile_capacity([tr], 1e3, [1.0, 10.0, 100.0, 1000.0])
    # once C is large enough that wf == 1, SSE no longer changes with C (unidentified direction)
    assert prof[-1]["sse"] == pytest.approx(prof[-2]["sse"], abs=1e-8)


def test_build_step_trajectory_blocks_and_partial_period():
    B = {d: (float(d), "observed") for d in range(17, 34)}
    W = {d: 10.0 for d in range(17, 33)}   # covers 17->18 ... 32->33
    tr = sp.build_step_trajectory(B, W, 17, 33)
    assert tr["start_dap"] == 17 and tr["end_dap"] == 33
    assert [p[0] for p in tr["periods"]] == [7.0, 7.0, 2.0]
    assert tr["periods"][0][1] == pytest.approx(70.0 * 1e-6)
    assert tr["obs_kg"] == [24.0, 31.0, 33.0]
    # a water gap at day 25 shortens the usable run; runs shorter than 7 days are rejected
    W2 = {d: 10.0 for d in range(17, 33) if d != 25}
    tr2 = sp.build_step_trajectory(B, W2, 17, 33)
    assert tr2["start_dap"] == 17 and tr2["end_dap"] == 25
    assert sp.build_step_trajectory(B, {d: 10.0 for d in range(17, 22)}, 17, 33) is None


def test_harvest_replay_only_when_endpoint_dap_matches_last_image():
    periods = [(7.0, 5e-4), (7.0, 5e-4), (2.0, 2e-4)]
    base = {"B0_kg": 1e-3, "periods": periods, "obs_kg": [float("nan")] * 3, "treatment": "wet"}
    trajs = [dict(base, barcode="Dr1AA000001", genotype="Dr1", end_dap=33),   # matches, eligible
             dict(base, barcode="Dr2AA000002", genotype="Dr2", end_dap=33),   # last image on DAP 35: mismatch
             dict(base, barcode="Dr3AA000003", genotype="Dr3", end_dap=33),   # no harvest row
             dict(base, barcode="Dr4AA000004", genotype="Dr4", end_dap=33)]   # dry_wt missing
    hj = pd.DataFrame({"plantbarcode": ["Dr1AA000001", "Dr2AA000002", "Dr4AA000004"],
                       "dry_wt_g": [2.0, 3.0, np.nan], "final_image_dap": [33, 35, 33]})
    ev = sp.evaluate_step_against_harvest(trajs, 1e3, 20.0, 1e3, hj)
    assert ev["eligibility"] == {"n_trajectories": 4, "n_without_harvest_row": 1, "n_harvest_dry_wt_missing": 1,
                                 "n_final_image_dap_missing": 0, "n_endpoint_dap_mismatch": 1, "n_eligible": 1}
    assert len(ev["per_plant"]) == 1 and ev["per_plant"][0]["barcode"] == "Dr1AA000001"
    expected = sp.simulate_step(1e-3, periods, 1e3, 20.0, 1e3)[-1]
    assert ev["per_plant"][0]["predicted_endpoint_kg"] == pytest.approx(expected)
    assert ev["per_plant"][0]["harvest_dry_wt_kg"] == pytest.approx(2e-3)
    assert ev["metrics_kg"]["n"] == 1
    assert ev["by_genotype"] == {"Dr1": {"n": 1, "relative_error_median": pytest.approx(expected / 2e-3 - 1)}}


# ------------------------------------------------------------------ 8. Peterson aggregation
def test_pot_aggregation_doubles_water_but_preserves_intensive_ratios():
    row = {"co2_pa": 38, "water_treatment": "low", "whole_plant_rgr_per_day": 0.1,
           "root_component_rgr_per_day": 0.03, "ln_transpiration_productivity": -4.97,
           "water_last_two_days_ml_per_reported_plant": 15.9}
    two = sp.aggregate_peterson_row(row, plants_per_pot=2, modeled_units_per_pot=1)
    one = sp.aggregate_peterson_row(row, plants_per_pot=1, modeled_units_per_pot=1)
    assert two["aggregate_water_last_two_days_ml"] == pytest.approx(31.8)
    assert two["aggregate_water_ml_per_day"] == pytest.approx(15.9)
    assert two["aggregate_water_m3_per_day"] == pytest.approx(15.9e-6)
    assert one["aggregate_water_last_two_days_ml"] == pytest.approx(15.9)
    expected_total = 2 * 1000 * math.exp(-4.97) / (365 * 0.1)
    assert two["balance_support_proxy_total_mass_kg_per_m3"] == pytest.approx(expected_total)
    assert one["balance_support_proxy_total_mass_kg_per_m3"] == pytest.approx(expected_total)
    assert two["whole_plant_rgr_per_day"] == one["whole_plant_rgr_per_day"] == 0.1
    assert two["balance_support_proxy_conditional_agb_kg_per_m3"] == pytest.approx(expected_total * 0.7)
    assert two["observed_growth_factor_39d_whole_plant"] == pytest.approx(math.exp(3.9))
    assert two["max_normal_growth_factor_r1_weekly_euler"] < 1.2
    assert two["min_weekly_euler_rate_per_year_for_factor"] > 10


def test_peterson_block_separates_co2_levels():
    readings = {"rows": [
        {"co2_pa": 38, "water_treatment": "low", "whole_plant_rgr_per_day": 0.1, "root_component_rgr_per_day": 0.03,
         "ln_transpiration_productivity": -5.0, "water_last_two_days_ml_per_reported_plant": 10.0},
        {"co2_pa": 76, "water_treatment": "high", "whole_plant_rgr_per_day": 0.1, "root_component_rgr_per_day": 0.03,
         "ln_transpiration_productivity": -4.0, "water_last_two_days_ml_per_reported_plant": 10.0}]}
    blk = sp.peterson_block(readings)
    assert len(blk["baseline_38pa"]) == 1 and len(blk["co2_sensitivity_76pa"]) == 1
    assert blk["reading_resolution_is_not_sampling_uncertainty"] is True


# ------------------------------------------------------------------ CLI safety
def test_cli_refuses_existing_output_directory(tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    rc = sp.main(["--images", "a", "--weights", "b", "--harvest", "c", "--zoom", "d", "--peterson", "e",
                  "--out", str(out)])
    assert rc == 2
    assert list(out.iterdir()) == []   # nothing written


def test_jsonable_handles_numpy_and_nan():
    out = sp._jsonable({"a": np.float64(1.5), "b": np.int64(2), "c": float("nan"), "d": np.bool_(True),
                        "e": [np.float32(0.5)]})
    assert out == {"a": 1.5, "b": 2, "c": None, "d": True, "e": [0.5]}
