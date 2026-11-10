"""Fetch two metadata versions positively identified in the DataONE index."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import urllib.parse
import fetch_followup

DEST = Path(__file__).parent / "confirmed_versions"
DEST.mkdir(exist_ok=True)
fetch_followup.DEST = DEST
items = []
for name, identifier, revision in [
    ("seasonal_biomass", "210011001", 2),
    ("plant_dimensions", "210011002", 108),
]:
    object_id = f"https://pasta.lternet.edu/package/metadata/eml/knb-lter-jrn/{identifier}/{revision}"
    items.append((name, "https://cn.dataone.org/cn/v2/object/" + urllib.parse.quote(object_id, safe="")))
with ThreadPoolExecutor(max_workers=2) as executor:
    records = list(executor.map(fetch_followup.fetch, items))
path = DEST / "manifest.json"
if path.exists():
    raise FileExistsError(path)
path.write_text(json.dumps(records, indent=2) + "\n")
for record in records:
    print(record["name"], record.get("status"), record.get("bytes"), record.get("error", ""))
