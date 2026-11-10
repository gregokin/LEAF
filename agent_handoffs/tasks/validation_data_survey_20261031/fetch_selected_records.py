"""Retrieve explicitly selected public records; bounded metadata, no model runs."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import urllib.parse
import urllib.request

DEST = Path(__file__).parent / "selected_records"
DEST.mkdir(exist_ok=True)
URLS = {}
for name, identifier, revision in [
    ("seasonal_biomass", "210011001", 4),
    ("plant_dimensions", "210011002", 109),
    ("annual_anpp", "210011003", 106),
    ("reference_harvest", "210011004", 85),
    ("monthly_soil_water", "210013003", 17),
    ("daily_precipitation", "210425001", 76),
    ("daily_climate", "210126001", 128),
    ("variable_rain_biomass", "210328001", 5),
    ("variable_rain_soil_water", "210328003", 6),
]:
    object_id = f"https://pasta.lternet.edu/package/metadata/eml/knb-lter-jrn/{identifier}/{revision}"
    URLS[name] = "https://cn.dataone.org/cn/v2/object/" + urllib.parse.quote(object_id, safe="")
for name, query in {
    "quadrat_catalog": 'title:"Plant Cover at Permanent Quad" AND -obsoletedBy:*',
    "tracer_catalog": 'title:"Shrub and grass soil-resource partitioning"',
    "experiment_catalog": 'title:precipitation AND title:Jornada AND -obsoletedBy:*',
}.items():
    URLS[name] = "https://cn.dataone.org/cn/v2/query/solr/?" + urllib.parse.urlencode({
        "q": query, "fl": "id,title,beginDate,endDate,datePublished,formatId", "wt": "json", "rows": 25
    })
URLS.update({
    "neon_clip_202209": "https://data.neonscience.org/api/v0/data/DP1.10023.001/JORN/2022-09?release=RELEASE-2026",
    "neon_woody_202310": "https://data.neonscience.org/api/v0/data/DP1.10098.001/JORN/2023-10?release=RELEASE-2026",
    "neon_water_202209": "https://data.neonscience.org/api/v0/data/DP1.00094.001/JORN/2022-09?release=RELEASE-2026",
})


def fetch(item):
    name, url = item
    record = {"name": name, "url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat()}
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            data = response.read(2_000_001)
            if len(data) > 2_000_000:
                raise ValueError("Record exceeds 2 MB limit")
            suffix = ".xml" if data.lstrip().startswith(b"<") else ".json"
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
    manifest = DEST / "manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    manifest.write_text(json.dumps(records, indent=2) + "\n")
    for record in records:
        print(record["name"], record.get("status"), record.get("bytes"), record.get("error", ""))
