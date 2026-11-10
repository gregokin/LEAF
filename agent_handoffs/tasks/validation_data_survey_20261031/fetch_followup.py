"""Follow published public metadata links; small selected evidence only."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

BASE = Path(__file__).parent
DEST = BASE / "followup_records"
DEST.mkdir(exist_ok=True)
URLS = {}
for name, identifier, revision in [
    ("tracer", "210469002", 1),
    ("permanent_quadrats", "210351001", 49),
]:
    object_id = f"https://pasta.lternet.edu/package/metadata/eml/knb-lter-jrn/{identifier}/{revision}"
    URLS[name] = "https://cn.dataone.org/cn/v2/object/" + urllib.parse.quote(object_id, safe="")
for name, query in {
    "seasonal_biomass_catalog": 'title:"Seasonal aboveground plant biomass" AND -obsoletedBy:*',
    "dimensions_catalog": 'title:"Seasonal non-destructive vegetation" AND -obsoletedBy:*',
    "rain_experiment_catalog": 'title:Jornada AND title:precipitation AND -scope:ecotrends AND -obsoletedBy:*',
}.items():
    URLS[name] = "https://cn.dataone.org/cn/v2/query/solr/?" + urllib.parse.urlencode({
        "q": query, "fl": "id,title,beginDate,endDate,datePublished,formatId", "wt": "json", "rows": 100
    })
for name in ["reference_harvest", "annual_anpp"]:
    root = ET.fromstring((BASE / "selected_records" / (name + ".xml")).read_bytes())
    for entity in list(root.find("dataset")):
        if entity.tag in ["dataTable", "otherEntity"]:
            url = entity.findtext(".//distribution/online/url")
            if url and (entity.tag == "dataTable" or "Harvestprotocol" in " ".join(entity.itertext())):
                key = name + ("_table" if entity.tag == "dataTable" else "_protocol")
                URLS[key] = "https://cn.dataone.org/cn/v2/object/" + urllib.parse.quote(url, safe="")
URLS["neon_clip_manifest"] = "https://data.neonscience.org/api/v0/data/DP1.10023.001/JORN/2022-09"


def fetch(item):
    name, url = item
    record = {"name": name, "url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat()}
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            data = response.read(4_000_001)
            if not data:
                raise ValueError("Empty HTTP response")
            if len(data) > 4_000_000:
                raise ValueError("Selected evidence exceeds 4 MB limit")
            if data.startswith(b"%PDF"):
                suffix = ".pdf"
            elif data.lstrip().startswith(b"<"):
                ET.fromstring(data)
                suffix = ".xml"
            else:
                try:
                    json.loads(data)
                    suffix = ".json"
                except json.JSONDecodeError:
                    suffix = ".csv"
            path = DEST / (name + suffix)
            if path.exists():
                raise FileExistsError(path)
            path.write_bytes(data)
            record.update(status=response.status, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), file=path.name)
    except Exception as exc:
        record.update(error_type=type(exc).__name__, error=str(exc))
    return record


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(fetch, URLS.items()))
    path = DEST / "manifest.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(records, indent=2) + "\n")
    for r in records:
        print(r["name"], r.get("status"), r.get("bytes"), r.get("error", ""))
