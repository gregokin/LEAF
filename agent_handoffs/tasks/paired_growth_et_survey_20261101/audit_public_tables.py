"""Inspect public table identities and schemas; no fitting or model execution."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import math

TASK = Path(__file__).parent
DATA = TASK / "public_evidence" / "selected"


def read_csv(name):
    with (DATA / name).open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return reader.fieldnames, rows


def group(plant_id):
    return "wet" if "AA" in plant_id else "dry" if "AB" in plant_id else "other"


weight_fields, weights = read_csv("setaria_snapshots.csv")
mass_fields, masses = read_csv("setaria_harvest.csv")
weight_ids = {row["plant barcode"] for row in weights}
mass_counts = Counter(row["plantbarcode"] for row in masses)
matched = [row for row in masses if row["plantbarcode"] in weight_ids]
tree = json.loads((TASK / "public_evidence" / "followup" / "setaria_tag_tree.json").read_text())
blob_shas = {item["path"]: item["sha"] for item in tree["tree"] if item["type"] == "blob"}
raw_paths = {
    "setaria_snapshots.csv": "data/SnapshotInfo_RIL_allday.csv",
    "setaria_harvest.csv": "data/measured_biomass_setaria_ril.csv",
    "setaria_fetch_data.R": "script/fetch_data.R",
    "setaria_water_use.R": "script/ril_water_use.R",
    "setaria_biomass_modeling.R": "script/ril_biomass_modeling.R",
    "setaria_documentation.docx": "Documentation.docx",
}
source_hashes = []
for name, upstream_path in raw_paths.items():
    data = (DATA / name).read_bytes()
    git_sha = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    source_hashes.append({"file": name, "upstream_path": upstream_path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "git_blob_sha1": git_sha, "matches_pinned_listing": git_sha == blob_shas[upstream_path]})

bad_mass = []
for row in masses:
    try:
        value = float(row["dry_wt"])
        if not math.isfinite(value) or value < 0:
            bad_mass.append(row["plantbarcode"])
    except ValueError:
        bad_mass.append(row["plantbarcode"])

result = {
    "recorded_utc": datetime.now(timezone.utc).isoformat(),
    "revision_from_tag": json.loads((DATA / "setaria_tag.json").read_text())["object"],
    "weight_schema": weight_fields,
    "harvest_schema": mass_fields,
    "raw_weight_rows": len(weights),
    "raw_weight_barcodes_including_controls": len(weight_ids),
    "weight_timestamp_start": min(row["timestamp"] for row in weights),
    "weight_timestamp_end": max(row["timestamp"] for row in weights),
    "weight_completed_values": dict(Counter(row["completed"] for row in weights)),
    "raw_harvest_rows": len(masses),
    "unique_harvest_barcodes": len(mass_counts),
    "duplicate_harvest_barcodes": {key: value for key, value in mass_counts.items() if value > 1},
    "exact_barcode_matched_harvest_rows": len(matched),
    "matched_treatment_codes": dict(Counter(group(row["plantbarcode"]) for row in matched)),
    "unmatched_harvest_barcodes": sorted(set(mass_counts) - weight_ids),
    "nonfinite_negative_or_nonnumeric_dry_mass_barcodes": bad_mass,
    "source_hashes": source_hashes,
    "limits": [
        "Raw counts precede scientific QC, date selection and evaporation corrections.",
        "An exact barcode join is established; no daily water loss, growth series, genotype analysis or calibration was computed.",
        "Three unmatched harvest barcodes are retained without automatic correction.",
        "The separate 216 MB image-feature/analysis archive was identified through metadata, not downloaded.",
        "The paper's main 17-DAP analysis window and selected script's intermediate 15-33-DAP window need explicit reconciliation before reuse."
    ]
}
assert len(matched) == 176
assert result["matched_treatment_codes"] == {"wet": 91, "dry": 85}
assert not result["duplicate_harvest_barcodes"] and not bad_mass
assert all(item["matches_pinned_listing"] for item in source_hashes)
(TASK / "setaria_audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(f"Raw weight rows: {len(weights)}; harvests: {len(masses)}; exact joins: {len(matched)} (91 wet, 85 dry).")
print(f"Pinned public Git blob checks: {len(source_hashes)} passed.")
