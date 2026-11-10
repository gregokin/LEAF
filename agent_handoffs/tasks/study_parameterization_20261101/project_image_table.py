"""Stream all public image rows, retaining relevant scalar metadata/features."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import io
import json
import zipfile

TASK = Path(__file__).parent
DEST = TASK / "public_evidence"
ARCHIVE = Path("/tmp/leaf-study-data/feldman_data.zip")
COLUMNS = ["timestamp", "frame", "lifter", "cartag", "zoom", "plantbarcode", "camera",
           "area", "hull-area", "solidity", "perimeter", "width", "height", "longest_axis",
           "in_bounds", "height_above_bound", "height_below_bound", "above_bound_area",
           "percent_above_bound_area", "exposure", "measurementlabel", "imgtype"]


class DigestReader(io.RawIOBase):
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.digest = hashlib.sha256()
        self.count = 0

    def readable(self):
        return True

    def readinto(self, target):
        data = self.wrapped.read(len(target))
        target[:len(data)] = data
        self.digest.update(data)
        self.count += len(data)
        return len(data)


if __name__ == "__main__":
    output = DEST / "setaria_image_scalars.csv"
    if output.exists():
        raise FileExistsError(output)
    with zipfile.ZipFile(ARCHIVE) as archive:
        selected = [x for x in archive.infolist() if Path(x.filename).name == "vis_setaria_ril.csv"]
        assert len(selected) == 1
        item = selected[0]
        digest_reader = DigestReader(archive.open(item))
        camera = Counter()
        zoom = Counter()
        lifter = Counter()
        examples = []
        rows = 0
        with io.TextIOWrapper(io.BufferedReader(digest_reader), encoding="utf-8", newline="") as handle, output.open("x", newline="") as out:
            reader = csv.DictReader(handle, delimiter=";")
            assert all(key in reader.fieldnames for key in COLUMNS)
            writer = csv.DictWriter(out, fieldnames=COLUMNS)
            writer.writeheader()
            for row in reader:
                writer.writerow({key: row[key] for key in COLUMNS})
                rows += 1
                camera[row["camera"]] += 1
                zoom[row["zoom"]] += 1
                lifter[row["lifter"]] += 1
                if len(examples) < 3 or row["camera"] == "SV" and not any(x["camera"] == "SV" for x in examples):
                    examples.append({key: row[key] for key in COLUMNS})
        assert digest_reader.count == item.file_size
        record = {"source_archive_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(), "source_member": item.filename,
                  "source_member_bytes": digest_reader.count, "source_member_sha256": digest_reader.digest.hexdigest(),
                  "projected_columns": COLUMNS, "rows_retained": rows, "rows_excluded": 0,
                  "cameras": dict(camera), "zooms": dict(zoom), "lifters": dict(lifter), "examples": examples,
                  "output_file": output.name, "output_bytes": output.stat().st_size,
                  "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                  "note": "Every source row retained. Coordinate-list landmark columns omitted; no biomass or water calculations."}
        (DEST / "image_projection.json").write_text(json.dumps(record, indent=2) + "\n")
        print("Streamed", rows, "rows from", item.file_size, "bytes to", output.stat().st_size, "bytes; no rows dropped.")
        print("Cameras", dict(camera), "zooms", dict(zoom), "lifters", dict(lifter))
