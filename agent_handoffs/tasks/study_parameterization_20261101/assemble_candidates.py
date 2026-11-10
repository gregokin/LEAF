"""Assemble inactive, explicitly qualified candidates from reviewed study outputs."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[2]
fit = json.loads((TASK / "fit_02/candidate_fragments.json").read_text())
oracle = json.loads((TASK / "independent_feldman/oracle.json").read_text())
independent_mesh = json.loads((TASK / "mesquite_independent.json").read_text())
rates = fit["water_rate_and_support"]["candidate_train"]
normal = fit["step_diagnostic_fit"]
selected = normal["scenarios_train"][normal["selected_scenario_unbounded"]]
coeff = normal["fitted_C_unbounded_kg_per_m3"]
rate = normal["fitted_r_unbounded"]
best_sse = selected["unbounded_positive"]["sse"]
profiles = json.loads((TASK / "fit_02/diagnostics.json").read_text())["step_profiles"]["r_fixed"]
similar = [dict(row, excess_train_sse_fraction=row["sse"] / best_sse - 1)
           for row in profiles if row["sse"] <= 1.10 * best_sse]
mesh = fit["peterson_neofotis_2004"]
ambient = mesh["baseline_38pa"]

result = {
    "schema": "leaf-study-stepwat-candidates-v1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "active": False,
    "runtime_ready": False,
    "purpose": "Initial controlled-environment seedling candidates and constraints; not complete field trait vectors.",
    "runtime_model": "stepwat_weekly_aboveground_v0.1",
    "runtime_source_sha256": hashlib.sha256((ROOT / "leaf/stepwat.py").read_bytes()).hexdigest(),
    "evidence_task": "agent_handoffs/tasks/study_parameterization_20261101",
    "mass_convention": "kg total dry aboveground mass per modeled plant unit",
    "water_convention": "m3 actual plant transpiration; Feldman pot loss is measured ET and requires an explicit T/ET assumption",
    "normal_rule": "required=B*dt_days/365; available=C*U; wf=min(1,available/required); gain=.95*r*(dt_days/365)*wf*B*(1-B/M), capped at M-B",
    "study_assumptions": {
        "live_fraction": 1.0,
        "temperature_factor": 1.0,
        "activity_during_observed_period": 1.0,
        "extra_growth_fraction": 0.0,
        "status": "Fixed diagnostic assumptions, not estimated annual traits or an identified seasonal/persistent split.",
        "update_clock": "Seven-day blocks relative to each valid experimental segment, plus a final partial observation interval; not a full native calendar replay."
    },
    "grass_setaria_c4_seedling": {
        "source_paper": "https://doi.org/10.1104/pp.18.00146",
        "data_archive": "https://doi.org/10.5281/zenodo.1117062",
        "source_revision": "8c88886b7f6ce3e0453eaba74ccadb0f319d3b1d",
        "biological_plants_per_model_unit": 1,
        "window_dap": [17, 33],
        "temperature_day_night_C": [31, 21],
        "normal_only_joint_fit": {
            "status": "Conditional experimental fit; use this C and r as a pair, not with the separate balance prior.",
            "biomass_support_kg_per_m3": coeff,
            "intrinsic_rate_per_year": rate,
            "capacity_scenario_kg": selected["M_kg"],
            "capacity_status": "2x largest TRAIN image-derived update-boundary mass; not measured mature biomass.",
            "assumed_T_fraction_of_ET": 1.0,
            "resource_limited_coefficient_0_95_r_C": .95 * rate * coeff,
            "coefficient_interpretation": "Controls gain per water in the resource-limited branch with unit live/activity/temperature factors; not an independently fitted constant-WUE law.",
            "n_train_trajectories": normal["n_train"],
            "n_heldout_trajectories": normal["n_heldout"],
            "not_runtime_compatible_reason": "Installed validator caps intrinsic_rate_per_year at 1; this diagnostic fit requires a larger value."
        },
        "empirical_balance_prior": {
            **rates["pooled"],
            "status": "TRAIN-only PR=1 burden/water prior under unit live/activity; not the joint-fit C and not a production yield.",
            "q_liters_per_kg_day": rates["pooled"]["q_m3_per_kg_day"] * 1000,
            "definition": "C_balance=integral(B dt)/(365*sum(U)); q=sum(U)/integral(B dt)",
            "nominal_calendar_days": True,
            "actual_weighing_duration_sensitivity": oracle["primary_train"],
            "by_treatment": rates["by_treatment"],
            "bootstrap_95_percentiles_fixed_allometry": rates["bootstrap_genotype"],
            "uncertainty_note": rates["conditioning_note"]
        },
        "T_ET_sensitivity": [
            {"assumed_T_fraction_of_ET": alpha, "joint_fit_C_kg_per_m3": coeff / alpha,
             "joint_fit_r_per_year": rate, "q_T_m3_per_kg_day": rates["pooled"]["q_m3_per_kg_day"] * alpha,
             "balance_prior_C_kg_per_m3": rates["pooled"]["C_balance_kg_per_m3"] / alpha,
             "status": "Illustrative uniform partition; not an observed evaporation fraction or uncertainty bound. C*U remains unchanged."}
            for alpha in [1., .9, .8]
        ],
        "similar_fit_profile_pairs": similar,
        "profile_status": "Alternatives within 10% of minimum TRAIN SSE on the coarse rate profile, not confidence limits.",
        "capacity_sensitivity": normal["scenarios_train"],
        "heldout_actual_harvest_evaluation": normal["heldout_harvest_replay_unbounded"],
        "heldout_actual_harvest_evaluation_with_current_rate_cap": normal["heldout_harvest_replay_bounded"],
        "allometry_holdout": fit["dry_mass_calibration"]["heldout_metrics_g"],
        "limitations": [
            "Seedlings of Setaria recombinant inbred lines and parents are not field black grama or a calibrated perennial-grass PFT.",
            "Pot loss is ET; no independent whole-plant transpiration partition was established.",
            "Image-derived trajectories use one training-only dry-mass calibration; they are not repeated destructive dry harvests.",
            "Last-image day is the source sampling proxy for harvest date, not independently confirmed destructive timing.",
            "Numerical rates depend on the explicit seven-day diagnostic and final-partial observation convention.",
            "The normal-only fit does not identify persistent versus seasonal biomass."
        ]
    },
    "shrub_mesquite_two_seedlings_per_model_unit": {
        "source_paper": "https://doi.org/10.1007/s00442-004-1688-y",
        "species": "Prosopis velutina",
        "biological_plants_per_model_unit": 2,
        "model_units_per_source_pot": 1,
        "extensive_scale_from_reported_per_seedling": 2.0,
        "intensive_scale": 1.0,
        "temperature_day_night_C": [36, 22],
        "status": "Conditional figure-derived priors, not a jointly fitted shrub trait vector.",
        "ambient_38Pa_prior": {
            "C_balance_conditional_agb_kg_per_m3": sum(x["balance_support_proxy_conditional_agb_kg_per_m3"] for x in ambient) / len(ambient),
            "C_balance_conditional_agb_low_high_water_range": [min(x["balance_support_proxy_conditional_agb_kg_per_m3"] for x in ambient), max(x["balance_support_proxy_conditional_agb_kg_per_m3"] for x in ambient)],
            "q_conditional_m3_per_kg_day_range": sorted([1/(365*x["balance_support_proxy_conditional_agb_kg_per_m3"]) for x in ambient]),
            "aggregate_endpoint_T_m3_per_day_range": sorted([x["aggregate_water_m3_per_day"] for x in ambient]),
            "observed_whole_plant_log_RGR_per_day_range": sorted([x["whole_plant_rgr_per_day"] for x in ambient]),
            "weekly_Euler_rate_lower_bound_whole_plant_per_year_range": sorted([x["min_weekly_euler_rate_per_year_for_factor"] for x in ambient])
        },
        "ambient_rows": ambient,
        "elevated_76Pa_sensitivity_rows": mesh["co2_sensitivity_76pa"],
        "unmeasured_initial_and_adult_agb_kg": None,
        "minimum_AGB_growth_factor_without_stable_allocation_assumption": [
            {"co2_pa": x["co2_pa"], "water_treatment": x["water_treatment"],
             "minimum_factor": x["minimum_agb_growth_factor_with_any_initial_agb_fraction_at_most_one"]}
            for x in independent_mesh["rows"]
        ],
        "limitations": [
            "Water values are the final two days, not cumulative 39-day transpiration.",
            "Published figure means are approximate readings; reading resolution is not sampling uncertainty.",
            "The total-mass support proxy uses the paper's log-mean identity; its AGB proxy assumes increment allocation approximates standing AGB allocation.",
            "The source lacks a retrieved raw water series and tabulated absolute starting/final AGB; no fictitious time series or full shrub calibration is supplied.",
            "Pair aggregation leaves relative growth and mass/water coefficients unchanged.",
            "Velvet mesquite seedlings are not mature Jornada honey mesquite."
        ]
    },
    "traits_not_identified_by_these_experiments": [
        "adult mature aboveground biomass", "canopy biomass density and biomass-to-geometry allometry",
        "independent potential-transpiration depth", "field live/leaf fraction and its size dependence",
        "annual monthly activity and temperature-response curve", "seasonal extra-growth fraction and turnover",
        "root depth/layer fractions and lateral extent", "demography, seed recruitment or event injury"
    ],
    "demand_mapping": "For adequate water and known canopy density rho, q_T=(potential_transpiration_depth/rho)*live_fraction*activity. Observed q alone does not identify both depth and rho.",
    "runtime_changes": "none; no loader/default/rate bound/equation or upstream source changed"
}
output = ROOT / "parameters/candidates/study-stepwat.json"
output.parent.mkdir(parents=True, exist_ok=True)
if output.exists():
    raise FileExistsError("Preserve assembled candidates; create a versioned revision instead")
output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print("Created", output.relative_to(ROOT), "as inactive study candidates.")
