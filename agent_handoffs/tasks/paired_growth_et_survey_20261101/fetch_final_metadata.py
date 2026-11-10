"""Read public archive metadata; POST is a read-only Figshare search."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import urllib.request

DEST = Path(__file__).parent / "public_evidence" / "final_metadata"
SOURCES = [
    ("setaria_data_zenodo.json", "https://zenodo.org/api/records/1117062", None),
    ("ravi_figshare_search.json", "https://api.figshare.com/v2/articles/search", {"search_for": "Biological invasions and climate change amplify", "limit": 10}),
    ("ravi_share_response.json", "https://figshare.com/s/7cf2cd975decff746bb3", None),
]


def fetch(source):
    name, url, body = source
    item = {"file": name, "url": url, "request_body": body, "retrieved_utc": datetime.now(timezone.utc).isoformat()}
    try:
        headers = {"User-Agent": "LEAF-public-data-survey"}
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode()
        request = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read(1_000_001)
            item.update(status=response.status, bytes_read=len(raw), final_url=response.url)
            if not raw:
                raise ValueError("HTTP success with empty body; no archive listing retrieved")
            if len(raw) > 1_000_000:
                raise ValueError("Response exceeded metadata limit")
            if name == "ravi_share_response.json":
                obj = {"headers": dict(response.headers), "body_prefix": raw[:1000].decode("utf-8", "replace")}
                raw = (json.dumps(obj, indent=2) + "\n").encode()
            else:
                json.loads(raw)
            (DEST / name).write_bytes(raw)
            item.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    except Exception as exc:
        item.update(error_type=type(exc).__name__, error=str(exc))
    return item


if __name__ == "__main__":
    DEST.mkdir(exist_ok=True)
    if (DEST / "manifest.json").exists():
        raise FileExistsError("Final metadata already recorded")
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(fetch, SOURCES))
    (DEST / "manifest.json").write_text(json.dumps(results, indent=2) + "\n")
    for item in results:
        print(item["file"], item.get("status"), item.get("bytes_read"), item.get("error", ""))
