"""Bounded public GETs for the user-requested data survey; no local uploads."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import urllib.request

DEST = Path(__file__).parent / "public_evidence"
DEST.mkdir(exist_ok=True)
SOURCES = [
    ("sunny_traits.xlsx", "https://datadryad.org/downloads/file_stream/3647145", 1_000_000),
    ("sunny_water.xlsx", "https://datadryad.org/downloads/file_stream/3647146", 1_000_000),
    ("sunny_readme.md", "https://datadryad.org/downloads/file_stream/3647148", 100_000),
    ("milkweed_drought.xlsx", "https://datadryad.org/downloads/file_stream/3724660", 1_000_000),
    ("setaria_zenodo.json", "https://zenodo.org/api/records/1340752", 2_000_000),
    ("setaria_github_tree.json", "https://api.github.com/repos/maxjfeldman/Feldman_Elsworth_Setaria_WUE_2017/git/trees/HEAD?recursive=1", 2_000_000),
    ("mesquite_nasa.html", "https://www.giss.nasa.gov/pubs/abs/pe02200i.html", 1_000_000),
]


def fetch(source):
    name, url, limit = source
    result = {"file": name, "url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat()}
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "LEAF-public-data-survey"})
        with urllib.request.urlopen(request, timeout=20) as response:
            data = response.read(limit + 1)
            if not data:
                raise ValueError("Empty response")
            if len(data) > limit:
                raise ValueError("Public evidence exceeds size limit")
            if name.endswith(".xlsx") and not data.startswith(b"PK"):
                raise ValueError("Response is not an XLSX archive")
            if name.endswith(".json"):
                json.loads(data)
            path = DEST / name
            if path.exists():
                raise FileExistsError(path)
            path.write_bytes(data)
            result.update(status=response.status, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), final_url=response.url)
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
    return result


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(fetch, SOURCES))
    manifest = DEST / "manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    manifest.write_text(json.dumps(results, indent=2) + "\n")
    for result in results:
        print(result["file"], result.get("status"), result.get("bytes"), result.get("error", ""))
