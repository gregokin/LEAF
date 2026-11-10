"""Bounded public source/data follow-up, separate from earlier retrieval records."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import fetch_followup

DEST = Path(__file__).parent / "public_evidence" / "selected"
BASE = "https://raw.githubusercontent.com/maxjfeldman/Feldman_Ellsworth_Setaria_WUE_2017/v1.0/"
SOURCES = [
    ("setaria_tag.json", "https://api.github.com/repos/maxjfeldman/Feldman_Ellsworth_Setaria_WUE_2017/git/ref/tags/v1.0", 100_000),
    ("setaria_snapshots.csv", BASE + "data/SnapshotInfo_RIL_allday.csv", 12_000_000),
    ("setaria_harvest.csv", BASE + "data/measured_biomass_setaria_ril.csv", 100_000),
    ("setaria_fetch_data.R", BASE + "script/fetch_data.R", 100_000),
    ("setaria_water_use.R", BASE + "script/ril_water_use.R", 100_000),
    ("setaria_biomass_modeling.R", BASE + "script/ril_biomass_modeling.R", 100_000),
    ("setaria_documentation.docx", BASE + "Documentation.docx", 200_000),
    ("ravi_figshare.html", "https://figshare.com/s/7cf2cd975decff746bb3", 8_000_000),
]

if __name__ == "__main__":
    DEST.mkdir(exist_ok=True)
    if (DEST / "manifest.json").exists():
        raise FileExistsError("Selected retrieval already recorded")
    fetch_followup.DEST = DEST
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(fetch_followup.fetch, SOURCES))
    (DEST / "manifest.json").write_text(json.dumps(results, indent=2) + "\n")
    for result in results:
        print(result["file"], result.get("status"), result.get("bytes"), result.get("error", ""))
