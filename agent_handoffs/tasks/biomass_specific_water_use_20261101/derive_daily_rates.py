"""Derive descriptive water rates from public observations; no model execution."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

TASK = Path(__file__).resolve().parent
SOURCE = TASK / "public_evidence/belovitch_Compiled_hourly_transpiration_WP_weather_data.csv"
OUT = TASK / "derived"
OUT.mkdir(exist_ok=False)


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def percentile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    lo = math.floor(position)
    hi = math.ceil(position)
    return values[lo] + (position - lo) * (values[hi] - values[lo])


with SOURCE.open(newline="") as stream:
    raw = list(csv.DictReader(stream))
day_rows = defaultdict(list)
hour_keys = Counter()
metadata = defaultdict(set)
negative_hours = 0
normalization_mismatches = Counter()
unplaced = []
for row in raw:
    identity = row["ID"]
    metadata[identity].add((row["Species"], row["Group"], number(row["Shoot"])))
    day = number(row["Day"])
    hour = number(row["Hour"])
    if day is None or hour is None or not hour.is_integer() or not 0 <= hour < 24:
        unplaced.append(identity)
        continue
    key = identity, int(day), int(hour)
    hour_keys[key] += 1
    day_rows[key[:2]].append(row)
    loss = number(row["deltaM"])
    shoot = number(row["Shoot"])
    supplied = number(row["deltaM.Shoot"])
    if loss is not None and loss < 0:
        negative_hours += 1
    if loss is not None and shoot is not None and supplied is not None:
        if abs(loss / shoot - supplied) > 1e-4:
            normalization_mismatches[identity] += 1
ambiguous_ids = sorted({key[0] for key, count in hour_keys.items() if count > 1})
changing_metadata = sorted(identity for identity, values in metadata.items() if len(values) != 1)
excluded_ids = set(ambiguous_ids) | set(changing_metadata) | set(normalization_mismatches)

daily = []
eligible = []
for (identity, day), rows in sorted(day_rows.items()):
    species, group, shoot = next(iter(metadata[identity]))
    losses = [number(row["deltaM"]) for row in rows]
    water_potentials = [number(row["WP"]) for row in rows]
    hours = {int(float(row["Hour"])) for row in rows}
    reasons = []
    if identity in excluded_ids:
        reasons.append("ambiguous_individual_or_inconsistent_normalization")
    if len(rows) != 24 or hours != set(range(24)):
        reasons.append("not_one_record_for_each_of_24_hours")
    if any(value is None for value in losses):
        reasons.append("missing_water_loss")
    if any(value is None for value in water_potentials):
        reasons.append("missing_water_potential")
    if shoot is None or shoot <= 0:
        reasons.append("invalid_shoot_dry_mass")
    water = math.fsum(value for value in losses if value is not None)
    psi = statistics.mean(value for value in water_potentials if value is not None) if any(value is not None for value in water_potentials) else None
    if water < 0:
        reasons.append("negative_daily_net_loss")
    q = water / shoot if shoot is not None and shoot > 0 else None
    if psi is None:
        moisture = "unclassified"
    elif psi > -0.5:
        moisture = "wet_gt_minus0.5_MPa"
    elif psi > -1:
        moisture = "moderate_minus1_to_minus0.5_MPa"
    elif psi > -1.5:
        moisture = "dry_minus1.5_to_minus1_MPa"
    else:
        moisture = "very_dry_le_minus1.5_MPa"
    result = {"ID": identity, "species_code": species, "functional_group": group,
              "experimental_day": day, "row_count": len(rows), "unique_hours": len(hours),
              "net_water_g": water, "harvested_shoot_dry_mass_g": shoot,
              "mean_soil_water_potential_MPa": psi, "moisture_bin": moisture,
              "q_L_per_kg_aboveground_DRY_biomass_per_day": q,
              "primary_eligible": not reasons, "exclusion_reasons": ";".join(reasons)}
    daily.append(result)
    if not reasons:
        eligible.append(result)

with (OUT / "belovitch_all_daily_qc.csv").open("x", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(daily[0]))
    writer.writeheader()
    writer.writerows(daily)

by_plant_bin = defaultdict(list)
for row in eligible:
    key = row["functional_group"], row["moisture_bin"], row["ID"]
    by_plant_bin[key].append(row["q_L_per_kg_aboveground_DRY_biomass_per_day"])
plant_rows = [{"functional_group": key[0], "moisture_bin": key[1], "ID": key[2],
               "eligible_days": len(values), "q_L_per_kg_AGB_DRY_per_day": statistics.mean(values)}
              for key, values in sorted(by_plant_bin.items())]
with (OUT / "belovitch_plant_bin_means.csv").open("x", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(plant_rows[0]))
    writer.writeheader()
    writer.writerows(plant_rows)
summary = []
for group, moisture in sorted({key[:2] for key in by_plant_bin}):
    selected = [row for row in plant_rows if row["functional_group"] == group and row["moisture_bin"] == moisture]
    values = [row["q_L_per_kg_AGB_DRY_per_day"] for row in selected]
    summary.append({"functional_group": group, "moisture_bin": moisture,
                    "individuals": len(values), "complete_24h_days": sum(row["eligible_days"] for row in selected),
                    "median_L_per_kg_AGB_DRY_per_day": statistics.median(values),
                    "interquartile_range_L_per_kg_AGB_DRY_per_day": [percentile(values, 0.25), percentile(values, 0.75)],
                    "range_L_per_kg_AGB_DRY_per_day": [min(values), max(values)],
                    "mean_L_per_kg_AGB_DRY_per_day": statistics.mean(values)})
audit = {"raw_rows": len(raw), "raw_individual_ids": len(metadata),
         "raw_functional_group_species_codes": {group: len({row['Species'] for row in raw if row['Group'] == group}) for group in {row['Group'] for row in raw}},
         "source_count_note": "Raw species-code counts differ from the paper's stated biological species counts; codes are not independently reidentified.",
         "unplaced_rows": len(unplaced), "duplicate_hour_excess_rows": sum(count - 1 for count in hour_keys.values() if count > 1),
         "ambiguous_individual_ids": ambiguous_ids, "changing_metadata_ids": changing_metadata,
         "supplied_normalization_mismatch_rows_by_id": dict(normalization_mismatches),
         "excluded_individual_ids": sorted(excluded_ids), "negative_hourly_losses_retained": negative_hours,
         "plant_days_total": len(daily), "primary_complete_days": len(eligible),
         "primary_individuals": len({row['ID'] for row in eligible}),
         "excluded_days_by_reason": dict(Counter(reason for row in daily for reason in row['exclusion_reasons'].split(';') if reason)),
         "policy": "Exclude whole individuals with ambiguous duplicate hours or inconsistent supplied normalization; require complete unique 24-hour observations and finite water/potential; retain negative hourly increments without clipping; recompute normalization from raw water and shoot mass."}
result = {"source": "Belovitch et al. 2023, doi:10.1111/nph.18909; released data v1.0.0",
          "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          "quantity": "Litres transpired per kg harvested aboveground DRY shoot biomass per 24-hour day",
          "mass_timing": "One fixed end-of-drydown harvested shoot dry mass per plant; no daily biomass observations or assumed biomass trajectory.",
          "conversion": "sum hourly water losses (g)/shoot dry mass (g) is numerically L/kg/day at water density approximately 1 kg/L.",
          "aggregation": "Mean daily rate within each individual/moisture bin, then median and IQR across individuals; no mixed-model refit.",
          "uncertainty": "Descriptive IQR, not confidence limits; includes species, plant and environment variation; no field validation.",
          "audit": audit, "summary": summary}
with (OUT / "belovitch_summary.json").open("x") as stream:
    stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")

murano = json.loads((TASK / "public_evidence/murano_table_readings.json").read_text())
murano_rows = []
for table in murano["tables"]:
    for row in table["rows"]:
        murano_rows.append({"table": table["number"], "environment": table["environment"],
                            **row, "q_L_per_kg_aboveground_DRY_biomass_per_day": row["T_mL_per_pot_per_day"] / row["shoot_DRY_g_per_pot"]})
with (OUT / "murano_rates.json").open("x") as stream:
    stream.write(json.dumps({"source": murano["source"], "rows": murano_rows,
                            "calculation": "Ratio of treatment means; mL/g/day equals L/kg/day. No ratio confidence interval inferred from marginal SEs."}, indent=2) + "\n")
print(json.dumps({"belovitch_audit": audit, "wet_bin_summary": [row for row in summary if row['moisture_bin'] == 'wet_gt_minus0.5_MPa'],
                  "murano_soil_rates": [{"species": row['species'], "q": row['q_L_per_kg_aboveground_DRY_biomass_per_day']} for row in murano_rows if row['table'] == 2]}, indent=2))
