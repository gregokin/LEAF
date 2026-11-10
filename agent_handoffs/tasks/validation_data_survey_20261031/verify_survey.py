"""Document/source audit only; does not import or run scientific code."""
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit
import hashlib
import json
import re
import xml.etree.ElementTree as ET

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[2]
baseline = json.loads((TASK / "baseline.json").read_text())
allowed = {
    "README.md", "agent_handoffs/current_prompt.md",
    "agent_handoffs/orchestration_state.md", "agent_handoffs/project_brief.md",
    "agent_handoffs/decisions.md", "agent_handoffs/codex_review.md",
    "agent_handoffs/verification.md",
}
changed, missing = [], []
for entry in baseline["files"]:
    path = ROOT / entry["path"]
    if not path.is_file():
        missing.append(entry["path"])
    elif hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        changed.append(entry["path"])
original = {entry["path"] for entry in baseline["files"]}
new_outside_scope = []
for path in ROOT.rglob("*"):
    relative = path.relative_to(ROOT)
    if not path.is_file() or path.is_symlink() or any(x in baseline["excluded_paths"] for x in relative.parts):
        continue
    name = str(relative)
    if name not in original and name != "docs/validation-data.md" and not name.startswith("agent_handoffs/tasks/validation_data_survey_20261031/"):
        new_outside_scope.append(name)
metadata_errors, response_checks = [], 0
for folder in ["public_metadata", "selected_records", "followup_records", "confirmed_versions"]:
    for entry in json.loads((TASK / folder / "manifest.json").read_text()):
        if "file" not in entry:
            continue
        path = TASK / folder / entry["file"]
        data = path.read_bytes()
        response_checks += 1
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            metadata_errors.append(str(path.relative_to(TASK)))
catalog = json.loads((TASK / "catalog.json").read_text())
for entry in catalog["packages_with_parsed_eml"]:
    path = TASK / entry["metadata_path"]
    root = ET.fromstring(path.read_bytes())
    if root.attrib.get("packageId") != entry["package_id"] or hashlib.sha256(path.read_bytes()).hexdigest() != entry["metadata_sha256"]:
        metadata_errors.append(str(path.relative_to(TASK)))
documents = [ROOT / "README.md", ROOT / "docs/validation-data.md"]
documents += [ROOT / "agent_handoffs" / x for x in [
    "current_prompt.md", "orchestration_state.md", "project_brief.md",
    "decisions.md", "codex_review.md", "verification.md",
]]
documents += list(TASK.glob("*.md"))
bad_links, trailing, link_count = [], [], 0
for path in documents:
    content = path.read_text()
    for line_no, line in enumerate(content.splitlines(), 1):
        if line.rstrip() != line:
            trailing.append([str(path.relative_to(ROOT)), line_no])
    # Saved aliases are byte-exact historical files with their original base.
    link_base = ROOT / "agent_handoffs" if path.name.startswith("previous_") else path.parent
    for target in re.findall(r"\]\(([^)]+)\)", content):
        if urlsplit(target).scheme or target.startswith("#"):
            continue
        link_count += 1
        local = unquote(target.split("#")[0].split("?")[0]).strip("<>")
        if not (link_base / local).resolve().exists():
            bad_links.append([str(path.relative_to(ROOT)), target])
agents_hash = hashlib.sha256((ROOT / "AGENTS.md").read_bytes()).hexdigest()
expected = "90a8b6217553480283efd0dcba514306c5472809b35b1d5a0e80e3009e48ef31"
result = {
    "checked_utc": datetime.now(timezone.utc).isoformat(),
    "baseline_files": len(baseline["files"]),
    "unchanged_baseline_files": len(baseline["files"]) - len(changed) - len(missing),
    "changed_baseline_paths": changed,
    "out_of_scope_changes": [x for x in changed if x not in allowed],
    "missing_baseline_files": missing,
    "new_paths_outside_scope": new_outside_scope,
    "root_agents_sha256": agents_hash,
    "root_agents_expected": expected,
    "parsed_eml_packages": len(catalog["packages_with_parsed_eml"]),
    "neon_products": len(catalog["neon_product_metadata"]),
    "archived_response_hashes_checked": response_checks,
    "metadata_errors": metadata_errors,
    "local_links_checked": link_count,
    "snapshot_link_base": "Original agent_handoffs directory",
    "bad_local_links": bad_links,
    "trailing_whitespace": trailing,
    "observation_rows_retrieved": False,
    "model_tests_or_runs": False,
    "claude_sessions": 0,
}
result["passed"] = not any([
    result["out_of_scope_changes"], missing, new_outside_scope,
    metadata_errors, bad_links, trailing, agents_hash != expected,
])
destination = TASK / "checks.json"
if destination.exists():
    previous = TASK / "checks.pre_close.json"
    if previous.exists():
        raise FileExistsError(previous)
    destination.rename(previous)
destination.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
if not result["passed"]:
    raise SystemExit("Research audit failed")
