"""Bounded documentation/source-preservation checks; no model execution."""
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit
import argparse
import hashlib
import json
import os
import re

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[2]
PREFIX = TASK.relative_to(ROOT).as_posix() + "/"
ALLOWED = {
    "README.md", "docs/validation-data.md", "docs/species-root-profiles.md",
    "docs/candidate-uptake-parameters.md", "docs/paired-growth-water-data.md",
    "parameters/candidates/jornada-uptake.json",
    "agent_handoffs/current_prompt.md", "agent_handoffs/orchestration_state.md",
    "agent_handoffs/codex_review.md", "agent_handoffs/verification.md",
    "agent_handoffs/project_brief.md", "agent_handoffs/decisions.md",
}
parser = argparse.ArgumentParser()
parser.add_argument("--read-only", action="store_true")
args = parser.parse_args()
baseline = json.loads((TASK / "baseline.json").read_text())
original = {item["path"]: item for item in baseline["files"]}
excluded = set(baseline["excluded_paths"])
present = {}
excluded_symlinks = []
baseline_time = datetime.fromisoformat(baseline["recorded_utc"]).timestamp()
for base, dirs, files in os.walk(ROOT):
    dirs[:] = [name for name in dirs if name not in excluded]
    for name in files:
        path = Path(base) / name
        if path.is_symlink():
            # The immutable baseline deliberately excludes symbolic links.
            # Match that population; do not dereference old upstream aliases.
            excluded_symlinks.append({"path": path.relative_to(ROOT).as_posix(), "target": os.readlink(path), "predates_baseline": path.lstat().st_mtime < baseline_time})
            continue
        present[path.relative_to(ROOT).as_posix()] = path

errors = []
for item in excluded_symlinks:
    if not item["predates_baseline"]:
        errors.append(f"Symbolic link needs separate scope review: {item['path']}")
changes = []
for name, item in original.items():
    path = present.get(name)
    if path is None:
        errors.append(f"Deleted baseline path: {name}")
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != item["sha256"]:
        changes.append({"path": name, "before_sha256": item["sha256"], "after_sha256": digest})
        if name not in ALLOWED and not name.startswith(PREFIX):
            errors.append(f"Out-of-scope change: {name}")
new = sorted(set(present) - set(original))
for name in new:
    if name not in ALLOWED and not name.startswith(PREFIX):
        errors.append(f"Out-of-scope new file: {name}")

json_count = 0
for name in sorted(ALLOWED | set(new)):
    path = ROOT / name
    if path.is_file() and path.suffix == ".json":
        try:
            json.loads(path.read_text())
            json_count += 1
        except Exception as exc:
            errors.append(f"Invalid JSON {name}: {exc}")

candidate = json.loads((ROOT / "parameters/candidates/jornada-uptake.json").read_text())
if candidate["active"] is not False or candidate["status"] != "candidate_only":
    errors.append("Candidate record is not explicitly inactive")
if candidate["profile_form"]["z_unit"] != "cm":
    errors.append("Candidate depth units changed")
if len(candidate["beta_estimates"]) != 6:
    errors.append("Expected six reported beta estimates")
for row in candidate["beta_estimates"]:
    lo, hi = row["reported_95_percent_ci"]
    if not 0 < lo <= row["beta"] <= hi < 1:
        errors.append(f"Invalid candidate beta/CI: {row}")

response_count = 0
retrieval_failures = []
for manifest in sorted((TASK / "public_evidence").rglob("manifest.json")):
    for item in json.loads(manifest.read_text()):
        if "error" in item:
            retrieval_failures.append({"manifest": manifest.relative_to(TASK).as_posix(), "file": item["file"], "error": item["error"], "status": item.get("status")})
        if "sha256" not in item:
            continue
        path = manifest.parent / item["file"]
        if not path.exists():
            errors.append(f"Missing public response: {path}")
            continue
        data = path.read_bytes()
        if not data or len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            errors.append(f"Public response integrity failed: {path}")
        response_count += 1

audit = json.loads((TASK / "setaria_audit.json").read_text())
if audit["exact_barcode_matched_harvest_rows"] != 176 or audit["matched_treatment_codes"] != {"wet": 91, "dry": 85}:
    errors.append("Setaria audit does not establish the reported joins")
if not all(item["matches_pinned_listing"] for item in audit["source_hashes"]):
    errors.append("Public Git source identity check failed")

link_count = 0
generated_links = []
markdown = {ROOT / name for name in ALLOWED if name.endswith(".md") and (ROOT / name).exists()}
markdown.update(TASK.glob("*.md"))
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
        if dest in planned and not dest.exists():
            generated_links.append(str(dest.relative_to(ROOT)))
        elif not dest.exists():
            errors.append(f"Broken local link {path.relative_to(ROOT)} -> {target}")
        link_count += 1
    if not historical:
        if not text.endswith("\n"):
            errors.append(f"Missing final newline: {path}")
        for line_number, line in enumerate(text.splitlines(), 1):
            if line.rstrip() != line:
                errors.append(f"Trailing whitespace: {path}:{line_number}")

root_agents = hashlib.sha256((ROOT / "AGENTS.md").read_bytes()).hexdigest()
if root_agents != "90a8b6217553480283efd0dcba514306c5472809b35b1d5a0e80e3009e48ef31":
    errors.append("Root AGENTS.md was modified")

result = {
    "checked_utc": datetime.now(timezone.utc).isoformat(),
    "pass": not errors,
    "baseline_file_count": len(original),
    "excluded_preexisting_symlinks": excluded_symlinks,
    "changes": changes,
    "new_files": new,
    "root_agents_sha256": root_agents,
    "json_documents_parsed": json_count,
    "nonempty_public_responses_hash_checked": response_count,
    "retrieval_failures_preserved": retrieval_failures,
    "local_markdown_links_checked": link_count,
    "generated_evidence_links_pending_creation": sorted(set(generated_links)),
    "errors": errors,
    "model_runs_or_tests": "none; research and documentation checks only"
}
if not args.read_only:
    (TASK / "checks.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: result[key] for key in ["pass", "baseline_file_count", "nonempty_public_responses_hash_checked", "local_markdown_links_checked", "errors"]}, indent=2))
if errors:
    raise SystemExit(1)
