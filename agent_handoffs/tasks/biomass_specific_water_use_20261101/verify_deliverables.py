"""Check research sources and baseline preservation, without running models."""
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
ALLOWED = {"README.md", "docs/biomass-specific-water-use.md",
           "agent_handoffs/current_prompt.md", "agent_handoffs/orchestration_state.md",
           "agent_handoffs/codex_review.md", "agent_handoffs/verification.md"}
parser = argparse.ArgumentParser()
parser.add_argument("--read-only", action="store_true")
args = parser.parse_args()
errors = []


def load(path):
    def reject(value):
        raise ValueError(f"Non-finite JSON token: {value}")
    return json.loads(path.read_text(), parse_constant=reject)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


baseline = load(TASK / "baseline.json")
original = {row["path"]: row for row in baseline["files"]}
excluded = set(baseline["excluded_paths"])
present = {}
symlinks = []
for base, dirs, files in os.walk(ROOT):
    dirs[:] = [name for name in dirs if name not in excluded]
    for name in files:
        path = Path(base) / name
        if path.is_symlink():
            symlinks.append(path.relative_to(ROOT).as_posix())
        else:
            present[path.relative_to(ROOT).as_posix()] = path
changes = []
for name, row in original.items():
    path = present.get(name)
    if path is None:
        errors.append(f"Deleted baseline file: {name}")
    elif digest(path) != row["sha256"]:
        changes.append(name)
        if name not in ALLOWED and not name.startswith(PREFIX):
            errors.append(f"Out-of-scope change: {name}")
new = sorted(set(present) - set(original))
for name in new:
    if name not in ALLOWED and not name.startswith(PREFIX):
        errors.append(f"Out-of-scope new file: {name}")

json_count = 0
for name in new:
    path = ROOT / name
    if path.suffix == ".json":
        try:
            load(path)
            json_count += 1
        except Exception as exc:
            errors.append(f"Invalid JSON {name}: {exc}")
hash_count = 0
retrieval_errors = []
for manifest in sorted((TASK / "public_evidence").glob("retrievals_*.json")):
    for row in load(manifest):
        if "error" in row:
            retrieval_errors.append({"manifest": manifest.name, "file": row["file"], "error": row["error"]})
        if "sha256" not in row:
            continue
        path = TASK / "public_evidence" / row["file"]
        if not path.exists() or path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            errors.append(f"Public source identity failed: {row['file']}")
        hash_count += 1
        if "matches_pinned_tree" in row and not row["matches_pinned_tree"]:
            errors.append(f"Git blob source mismatch: {row['file']}")
independent = load(TASK / "independent_checks.json")
if not independent["pass"] or independent["independent_days"] != 759 or independent["murano_rows_checked"] != 14:
    errors.append("Independent numerical checks failed")
summary = load(TASK / "derived/belovitch_summary.json")
source = TASK / "public_evidence/belovitch_Compiled_hourly_transpiration_WP_weather_data.csv"
if summary["source_sha256"] != digest(source) or summary["audit"]["primary_complete_days"] != 759:
    errors.append("Derived data identity or day count mismatch")
if "DRY" not in summary["quantity"] or "fixed" not in summary["mass_timing"]:
    errors.append("Dry biomass/timing definitions missing")
identities = load(TASK / "source_identities.json")
if identities["commit_sha"] != "862abe68c7a9f1eb6397c36516bca35c9d445298" or not all(row["matches_pinned_tree"] for row in identities["selected_git_files"]):
    errors.append("Pinned public revision identity failed")

markdown = [ROOT / name for name in ALLOWED if name.endswith(".md")]
markdown += list(TASK.glob("*.md"))
links = 0
planned = {TASK / "checks.json", TASK / "accepted_artifacts.json"} if not args.read_only else set()
for path in sorted(set(markdown)):
    if not path.exists():
        errors.append(f"Missing document: {path}")
        continue
    content = path.read_text()
    historical = path.parent == TASK and path.name.startswith("previous_")
    link_base = ROOT / "agent_handoffs" if historical else path.parent
    for match in re.finditer(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", content):
        target = match.group(1).strip().strip("<>")
        split = urlsplit(target)
        if split.scheme or not split.path:
            continue
        destination = (link_base / unquote(split.path)).resolve()
        if not destination.exists() and destination not in planned:
            errors.append(f"Broken link: {path.relative_to(ROOT)} -> {target}")
        links += 1
    if not historical:
        if not content.endswith("\n"):
            errors.append(f"Missing final newline: {path}")
        for number, line in enumerate(content.splitlines(), 1):
            if line.rstrip() != line:
                errors.append(f"Trailing whitespace: {path}:{number}")
root_agents = digest(ROOT / "AGENTS.md")
if root_agents != "90a8b6217553480283efd0dcba514306c5472809b35b1d5a0e80e3009e48ef31":
    errors.append("Root AGENTS.md changed")
manifest = TASK / "accepted_artifacts.json"
if manifest.exists():
    for row in load(manifest)["files"]:
        path = TASK / row["path"]
        if not path.exists() or path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            errors.append(f"Accepted artifact changed: {row['path']}")
record = {"checked_utc": datetime.now(timezone.utc).isoformat(), "pass": not errors,
          "baseline_files": len(original), "changed_paths": changes, "new_paths": new,
          "excluded_preexisting_symlinks": symlinks, "root_agents_sha256": root_agents,
          "strict_json_files_checked": json_count, "retrieved_sources_hash_checked": hash_count,
          "retrieval_errors_preserved": retrieval_errors, "local_links_checked": links,
          "independent_numerical_checks_pass": independent["pass"], "errors": errors}
if not args.read_only:
    with (TASK / "checks.json").open("x") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
print(json.dumps({key: record[key] for key in
                  ("pass", "baseline_files", "retrieved_sources_hash_checked", "local_links_checked", "errors")}, indent=2))
raise SystemExit(bool(errors))
