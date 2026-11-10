"""Retrieve selected public methods/metadata; never upload local files."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import urllib.request

DEST = Path(__file__).parent / "public_evidence" / "followup"
SOURCES = [
    ("mesquite_peterson_2004.pdf", "https://www.giss.nasa.gov/pubs/docs/2004/2004_Peterson_pe02200i.pdf", 5_000_000),
    ("setaria_tag_tree.json", "https://api.github.com/repos/maxjfeldman/Feldman_Ellsworth_Setaria_WUE_2017/git/trees/v1.0?recursive=1", 2_000_000),
    ("ravi_figshare.html", "https://figshare.com/s/7cf2cd975decff746bb3", 2_000_000),
]


def fetch(source):
    name, url, limit = source
    result = {"file": name, "url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat()}
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "LEAF-public-data-survey"})
        with urllib.request.urlopen(request, timeout=25) as response:
            data = response.read(limit + 1)
            if not data or len(data) > limit:
                raise ValueError("Empty or oversized public response")
            if name.endswith(".json"):
                json.loads(data)
            if name.endswith(".pdf") and not data.startswith(b"%PDF"):
                raise ValueError("Response is not a PDF")
            path = DEST / name
            if path.exists():
                raise FileExistsError(path)
            path.write_bytes(data)
            result.update(status=response.status, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), final_url=response.url)
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
    return result


if __name__ == "__main__":
    DEST.mkdir(exist_ok=True)
    if (DEST / "manifest.json").exists():
        raise FileExistsError("Followup already recorded")
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(fetch, SOURCES))
    (DEST / "manifest.json").write_text(json.dumps(results, indent=2) + "\n")
    for result in results:
        print(result["file"], result.get("status"), result.get("bytes"), result.get("error", ""))
