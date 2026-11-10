"""Bounded public downloads for the requested study parameter derivation."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile

TASK = Path(__file__).parent
DEST = TASK / "public_evidence"
SCRATCH = Path("/tmp/leaf-study-data")
OLD = TASK.parent / "paired_growth_et_survey_20261101"
metadata = json.loads((OLD / "public_evidence/final_metadata/setaria_data_zenodo.json").read_text())
file = metadata["files"][0]


def fetch(source):
    name, url, limit, destination = source
    item = {"file": name, "url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat()}
    path = destination / name
    try:
        if path.exists():
            raise FileExistsError(path)
        request = urllib.request.Request(url, headers={"User-Agent": "LEAF-public-study-parameterization"})
        sha = hashlib.sha256()
        md5 = hashlib.md5()
        size = 0
        with urllib.request.urlopen(request, timeout=30) as response, path.open("xb") as handle:
            while True:
                chunk = response.read(1_048_576)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise ValueError("Public response exceeds bounded size")
                handle.write(chunk)
                sha.update(chunk)
                md5.update(chunk)
            if not size:
                raise ValueError(f"Empty HTTP {response.status} response")
            item.update(status=response.status, bytes=size, sha256=sha.hexdigest(), md5=md5.hexdigest(), final_url=response.url)
        if name.endswith(".zip"):
            if size != file["size"] or "md5:" + md5.hexdigest() != file["checksum"]:
                raise ValueError("Archive does not match pinned Zenodo size/checksum")
            with zipfile.ZipFile(path) as archive:
                listing = [{"name": x.filename, "bytes": x.file_size, "compressed_bytes": x.compress_size} for x in archive.infolist()]
                (DEST / "setaria_archive_listing.json").write_text(json.dumps(listing, indent=2) + "\n")
                selected = [x for x in archive.infolist() if Path(x.filename).name == "vis_setaria_ril.csv"]
                if len(selected) != 1 or selected[0].file_size > 120_000_000:
                    raise ValueError("Expected one bounded image-feature CSV")
                data = archive.read(selected[0])
                output = DEST / "vis_setaria_ril.csv"
                with output.open("xb") as handle:
                    handle.write(data)
                item["selected_member"] = {"member": selected[0].filename, "file": output.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        item["local_path"] = str(path)
    except Exception as exc:
        item.update(error_type=type(exc).__name__, error=str(exc))
    return item


if __name__ == "__main__":
    DEST.mkdir(exist_ok=True)
    SCRATCH.mkdir(exist_ok=True)
    if (DEST / "retrieval_manifest.json").exists():
        raise FileExistsError("Preserve previous public retrievals")
    sources = [
        ("feldman_data.zip", file["links"]["self"], 220_000_000, SCRATCH),
        ("peterson_springer.html", "https://link.springer.com/article/10.1007/s00442-004-1688-y", 3_000_000, DEST),
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(fetch, sources))
    (DEST / "retrieval_manifest.json").write_text(json.dumps(results, indent=2) + "\n")
    for item in results:
        print(item["file"], item.get("status"), item.get("bytes"), item.get("error", ""), item.get("selected_member", {}))
