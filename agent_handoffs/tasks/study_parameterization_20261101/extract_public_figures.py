"""Losslessly expose the paper's embedded CCITT figures using existing Pillow."""
from pathlib import Path
from io import BytesIO
import hashlib
import json
import re
import struct
from PIL import Image

TASK = Path(__file__).parent
SOURCE = TASK.parent / "paired_growth_et_survey_20261101/public_evidence/followup/mesquite_peterson_2004.pdf"
DEST = TASK / "public_evidence" / "peterson_figures"


def tiff_header(width, height, count):
    values = [(256, 4, width), (257, 4, height), (258, 3, 1), (259, 3, 4),
              (262, 3, 0), (273, 4, 8 + 2 + 9 * 12 + 4), (278, 4, height),
              (279, 4, count), (293, 4, 0)]
    data = struct.pack("<2sHIH", b"II", 42, 8, len(values))
    for tag, kind, value in values:
        data += struct.pack("<HHI", tag, kind, 1)
        data += struct.pack("<H", value) + b"\0\0" if kind == 3 else struct.pack("<I", value)
    return data + struct.pack("<I", 0)


if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    data = SOURCE.read_bytes()
    manifest = []
    for match in re.finditer(rb"(\d+)\s+0\s+obj(.*?)endobj", data, re.S):
        obj, content = match.group(1), match.group(2)
        if b"/Subtype/Image" not in content:
            continue
        assert b"/CCITTFaxDecode" in content and b"/K -1" in content
        width = int(re.search(rb"/Width\s+(\d+)", content).group(1))
        height = int(re.search(rb"/Height\s+(\d+)", content).group(1))
        length = int(re.search(rb"/Length\s+(\d+)", content).group(1))
        start = re.search(rb"stream(?:\r\n|\r|\n)", content).end()
        compressed = content[start:start + length]
        assert len(compressed) == length
        output = DEST / f"figure_{len(manifest) + 1}_object_{obj.decode()}.png"
        if output.exists():
            raise FileExistsError(output)
        with Image.open(BytesIO(tiff_header(width, height, length) + compressed)) as image:
            image.save(output)
        raw = output.read_bytes()
        manifest.append({"file": output.name, "pdf_object": int(obj), "width": width, "height": height,
                         "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "transformation": "CCITT Group 4 decode only; no resampling"})
    (DEST / "manifest.json").write_text(json.dumps({"source_sha256": hashlib.sha256(data).hexdigest(), "figures": manifest}, indent=2) + "\n")
    print("Extracted", len(manifest), "embedded figures without installing a PDF package.")
