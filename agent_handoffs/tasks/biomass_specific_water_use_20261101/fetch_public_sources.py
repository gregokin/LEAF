"""Bounded, read-only retrieval of public primary literature and data metadata."""
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

DEST = Path(__file__).resolve().parent / "public_evidence"
ITEMS = [
    ("belovitch.pdf", "https://www.k-state.edu/ecophyslab/pdf%27s/Belowvitch_2023_NewPhyt.pdf"),
    ("belovitch_github_tree.json", "https://api.github.com/repos/mbelovitch/NPH-MS-2023-42507/git/trees/v1.0.0?recursive=1"),
    ("old_transpiration.pdf", "https://core.ac.uk/download/pdf/29395353.pdf"),
    ("he_bunchgrasses.pdf", "https://journal.hep.com.cn/fase/EN/PDF/10.15302/J-FASE-2017175"),
    ("el_habti.html", "https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2020.568693/full"),
    ("el_habti_fig5.webp", "https://www.frontiersin.org/files/Articles/568693/xml-images/fpls-11-568693-g005.webp"),
    ("el_habti_fig3.webp", "https://www.frontiersin.org/files/Articles/568693/xml-images/fpls-11-568693-g003.webp"),
    ("renninger_poplar.html", "https://www.frontiersin.org/journals/forests-and-global-change/articles/10.3389/ffgc.2021.704799/full"),
]


def fetch(item):
    name, url = item
    row = {"file": name, "url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat()}
    try:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0 (scientific literature review)"})
        with urlopen(request, timeout=25) as response:
            data = response.read(20_000_001)
            row.update(status=response.status, final_url=response.url, content_type=response.headers.get("Content-Type"))
        if len(data) > 20_000_000:
            raise ValueError("Public response exceeded bounded retrieval limit")
        with (DEST / name).open("xb") as out:
            out.write(data)
        row.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    except Exception as exc:
        row.update(error_type=type(exc).__name__, error=str(exc))
    return row


with ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(fetch, ITEMS))
with (DEST / "retrievals_02.json").open("x") as out:
    out.write(json.dumps(results, indent=2) + "\n")
print(json.dumps([{key: row[key] for key in ("file", "bytes", "status", "error") if key in row} for row in results], indent=2))
