"""Bounded public GETs only; never reads or uploads project sources."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import urllib.parse
import urllib.request

DEST = Path(__file__).parent / "public_metadata"
DEST.mkdir(exist_ok=True)
QUERIES = {
    "dataone_harvest": 'title:"Seasonal reference harvest measurements"',
    "dataone_soil_water": 'title:"Soil volumetric water content calculated from neutron"',
    "dataone_rain_experiment": 'title:"Aboveground vegetation cover and biomass" AND title:"altered precipitation"',
}
URLS = {name: "https://cn.dataone.org/cn/v2/query/solr/?" + urllib.parse.urlencode({
    "q": query, "fl": "id,title,beginDate,endDate,datePublished,replicaMN,size,formatId,seriesId,obsoletedBy", "wt": "json", "rows": 8
}) for name, query in QUERIES.items()}
URLS.update({
    "neon_site_jorn": "https://data.neonscience.org/api/v0/sites/JORN",
    "neon_herbaceous": "https://data.neonscience.org/api/v0/products/DP1.10023.001",
    "neon_woody": "https://data.neonscience.org/api/v0/products/DP1.10098.001",
    "neon_soil_water": "https://data.neonscience.org/api/v0/products/DP1.00094.001",
})


def fetch(item):
    name, url = item
    record = {"name": name, "url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat()}
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            data = response.read(2_000_001)
            if len(data) > 2_000_000:
                raise ValueError("Metadata exceeds 2 MB limit")
            json.loads(data)
            path = DEST / (name + ".json")
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
    print(json.dumps(records, indent=2))
