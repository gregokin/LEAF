"""Read-only scientific/source checks; optionally write the first check record."""
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit
import argparse
import hashlib
import json
import math
import os
import re

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[2]
PREFIX = TASK.relative_to(ROOT).as_posix() + "/"
ALLOWED = {
    "README.md", "docs/paired-growth-water-data.md", "docs/stepwat-growth.md",
    "docs/study-stepwat-parameters.md", "parameters/candidates/study-stepwat.json",
    "agent_handoffs/current_prompt.md", "agent_handoffs/orchestration_state.md",
    "agent_handoffs/codex_review.md", "agent_handoffs/verification.md",
    "agent_handoffs/project_brief.md", "agent_handoffs/decisions.md",
    "agent_handoffs/claude_report.md",
}
parser = argparse.ArgumentParser()
parser.add_argument("--read-only", action="store_true")
args = parser.parse_args()
errors = []


def load(path):
    def reject(value):
        raise ValueError(f"Non-finite JSON token: {value}")
    return json.loads(path.read_text(), parse_constant=reject)


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def check(condition, reason):
    if not condition:
        errors.append(reason)


baseline = load(TASK / "baseline.json")
original = {row["path"]: row for row in baseline["files"]}
excluded = set(baseline["excluded_paths"])
baseline_time = datetime.fromisoformat(baseline["recorded_utc"]).timestamp()
present = {}
symlinks = []
for base, dirs, files in os.walk(ROOT):
    dirs[:] = [name for name in dirs if name not in excluded]
    for name in files:
        path = Path(base) / name
        rel = path.relative_to(ROOT).as_posix()
        if path.is_symlink():
            predates = path.lstat().st_mtime < baseline_time
            symlinks.append({"path": rel, "target": os.readlink(path), "predates_baseline": predates})
            check(predates, f"New symbolic link requires scope review: {rel}")
        else:
            present[rel] = path


def allowed(name):
    return name in ALLOWED or name.startswith(PREFIX) or name.startswith("analysis/study_parameterization/")


changes = []
for name, row in original.items():
    path = present.get(name)
    if path is None:
        errors.append(f"Deleted baseline file: {name}")
        continue
    after = digest(path)
    if after != row["sha256"]:
        changes.append({"path": name, "before_sha256": row["sha256"], "after_sha256": after})
        check(allowed(name), f"Out-of-scope change: {name}")
new = sorted(set(present) - set(original))
for name in new:
    check(allowed(name), f"Out-of-scope new file: {name}")

json_count = 0
for name in sorted(set(new) | ALLOWED):
    path = ROOT / name
    if path.is_file() and path.suffix == ".json":
        try:
            load(path)
            json_count += 1
        except Exception as exc:
            errors.append(f"Invalid JSON {name}: {exc}")

candidate = load(ROOT / "parameters/candidates/study-stepwat.json")
check(candidate["active"] is False and candidate["runtime_ready"] is False,
      "Study parameters must remain inactive and not runtime-ready")
runtime_hash = digest(ROOT / "leaf/stepwat.py")
check(candidate["runtime_source_sha256"] == runtime_hash, "Runtime identity mismatch")
grass = candidate["grass_setaria_c4_seedling"]
fit = grass["normal_only_joint_fit"]
prior = grass["empirical_balance_prior"]
check(math.isclose(prior["C_balance_kg_per_m3"], 1 / (365 * prior["q_m3_per_kg_day"])),
      "Grass balance/support units do not agree")
check(math.isclose(fit["resource_limited_coefficient_0_95_r_C"],
                   .95 * fit["intrinsic_rate_per_year"] * fit["biomass_support_kg_per_m3"]),
      "Grass paired growth/support coefficient mismatch")
check(fit["intrinsic_rate_per_year"] > 1, "Expected diagnostic rate incompatibility missing")
shrub = candidate["shrub_mesquite_two_seedlings_per_model_unit"]
check(shrub["biological_plants_per_model_unit"] == 2 and shrub["model_units_per_source_pot"] == 1,
      "Mesquite aggregate plant identity mismatch")
check(shrub["extensive_scale_from_reported_per_seedling"] == 2 and shrub["intensive_scale"] == 1,
      "Mesquite extensive/intensive scaling mismatch")
independent = load(TASK / "independent_comparison.json")
check(independent["pass"] and all(independent["checks"].values()), "Independent balance comparisons failed")
replay = load(TASK / "independent_replay_checks.json")
check(replay["pass"] and replay["heldout_replays_checked"] == 177 and
      replay["harvests_checked_against_original_raw_table"] == 34,
      "Independent harvest/replay checks failed")
metrics = grass["heldout_actual_harvest_evaluation"]["metrics_kg"]
check(math.isclose(1000 * metrics["rmse"], replay["independent_harvest_rmse_g"]),
      "Recorded harvest RMSE does not agree with independent raw-data check")

for row in load(TASK / "installation.json"):
    check(row["source_identical"] and digest(ROOT / row["file"]) == row["sha256"] == digest(ROOT / row["source"]),
          f"Installed source differs from tested snapshot: {row['file']}")
for row in load(TASK / "review_snapshot_02.json"):
    check(digest(TASK / "review_snapshot_02" / row["file"]) == row["sha256"],
          f"Frozen snapshot changed: {row['file']}")

public_hashes = 0
for row in load(TASK / "public_evidence/retrieval_manifest.json"):
    path = Path("/tmp/leaf-study-data") / row["file"] if row["file"] == "feldman_data.zip" else TASK / "public_evidence" / row["file"]
    if row["file"] == "feldman_data.zip" and not path.exists():
        continue  # Reproduction uses the archived projected table, not scratch.
    check(path.exists() and path.stat().st_size == row["bytes"] and digest(path) == row["sha256"],
          f"Retrieved public source identity mismatch: {row['file']}")
    public_hashes += 1
for row in load(TASK / "public_evidence/additional_retrievals.json"):
    path = TASK / "public_evidence" / row["file"]
    check(path.stat().st_size == row["bytes"] and digest(path) == row["sha256"],
          f"Public calibration identity mismatch: {row['file']}")
    public_hashes += 1
projection = load(TASK / "public_evidence/image_projection.json")
projected = TASK / "public_evidence" / Path(projection["output_file"]).name
check(projected.stat().st_size == projection["output_bytes"] and digest(projected) == projection["output_sha256"],
      "Projected image source identity mismatch")
check(projection["rows_retained"] == 70985 and projection["rows_excluded"] == 0,
      "Projection did not retain all source rows")
public_hashes += 1
figures = load(TASK / "public_evidence/peterson_figures/manifest.json")
pdf = TASK.parent / "paired_growth_et_survey_20261101/public_evidence/followup/mesquite_peterson_2004.pdf"
check(digest(pdf) == figures["source_sha256"], "Mesquite primary PDF identity mismatch")
for row in figures["figures"]:
    path = TASK / "public_evidence/peterson_figures" / row["file"]
    check(path.stat().st_size == row["bytes"] and digest(path) == row["sha256"],
          f"Extracted figure identity mismatch: {row['file']}")
    public_hashes += 1

tool_count = 0
for name in ("claude_01.stdout.jsonl", "claude_02.stdout.jsonl"):
    for line in (TASK / name).read_text().splitlines():
        event = json.loads(line)
        message = event.get("message", {})
        content = message.get("content", []) if isinstance(message, dict) else []
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tool_count += 1
            tool = block["name"]
            path = Path(block.get("input", {}).get("file_path", ""))
            allowed_candidate = {TASK / "candidate" / f for f in
                                 ("study_parameters.py", "test_study_parameters.py", "README.md", "report.md")}
            allowed_metadata = {TASK / "public_evidence" / f for f in
                                ("input_schemas.json", "image_projection.json", "zoom_calibration_data.txt", "peterson_figure_readings.json")}
            check(tool in {"Read", "Write", "Edit"}, f"Unpermitted Claude tool: {tool}")
            check(path in allowed_candidate or (tool == "Read" and path in allowed_metadata),
                  f"Claude source/write scope violation: {tool} {path}")
result = load(TASK / "structured_result_02.json")
check(result["results"][-1]["subtype"] == "success" and
      result["results"][-1]["is_error"] is False and not result["tool_errors"],
      "Final Claude structured result not successful")
for name in ("exit_01.json", "exit_02.json", "tests_02.exit.json", "fit_02.exit.json"):
    row = load(TASK / name)
    check(row.get("child_reaped") is True, f"Child not recorded as reaped: {name}")

link_count = 0
markdown = {ROOT / name for name in ALLOWED if name.endswith(".md") and (ROOT / name).exists()}
markdown.update(TASK.glob("*.md"))
markdown.add(ROOT / "analysis/study_parameterization/README.md")
planned = {TASK / "checks.json", TASK / "accepted_artifacts.json"} if not args.read_only else set()
for path in sorted(markdown):
    text = path.read_text()
    historical = path.parent == TASK and path.name.startswith("previous_")
    link_base = ROOT / "agent_handoffs" if historical else path.parent
    for match in re.finditer(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", text):
        target = match.group(1).strip().strip("<>")
        split = urlsplit(target)
        if split.scheme or not split.path:
            continue
        dest = (link_base / unquote(split.path)).resolve()
        check(dest.exists() or dest in planned,
              f"Broken local link {path.relative_to(ROOT)} -> {target}")
        link_count += 1
    if not historical:
        check(text.endswith("\n"), f"Missing final newline: {path}")
        for number, line in enumerate(text.splitlines(), 1):
            check(line.rstrip() == line, f"Trailing whitespace: {path}:{number}")

root_agents = digest(ROOT / "AGENTS.md")
check(root_agents == "90a8b6217553480283efd0dcba514306c5472809b35b1d5a0e80e3009e48ef31",
      "Root AGENTS.md modified")
manifest_path = TASK / "accepted_artifacts.json"
if manifest_path.exists():
    manifest = load(manifest_path)
    for row in manifest["files"]:
        path = TASK / row["path"]
        check(path.exists() and path.stat().st_size == row["bytes"] and digest(path) == row["sha256"],
              f"Accepted artifact changed: {row['path']}")

record = {"checked_utc": datetime.now(timezone.utc).isoformat(), "pass": not errors,
          "baseline_file_count": len(original), "changes": changes, "new_files": new,
          "excluded_preexisting_symlinks": symlinks, "root_agents_sha256": root_agents,
          "runtime_stepwat_sha256": runtime_hash, "json_documents_parsed": json_count,
          "public_source_hashes_checked": public_hashes, "claude_tool_calls_scope_checked": tool_count,
          "independent_numerical_checks": len(independent["checks"]),
          "independent_endpoint_replays": replay["heldout_replays_checked"],
          "local_markdown_links_checked": link_count, "errors": errors}
if not args.read_only:
    with (TASK / "checks.json").open("x") as out:
        out.write(json.dumps(record, indent=2) + "\n")
print(json.dumps({key: record[key] for key in
                  ("pass", "baseline_file_count", "public_source_hashes_checked",
                   "claude_tool_calls_scope_checked", "local_markdown_links_checked", "errors")}, indent=2))
raise SystemExit(bool(errors))
