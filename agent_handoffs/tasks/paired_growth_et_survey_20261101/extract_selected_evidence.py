"""Reproduce rough text extracts for inspection; not a general PDF parser."""
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile
import zlib

BASE = Path(__file__).parent / "public_evidence"


def save_or_compare(path, text):
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f"Existing text differs: {path}")
    else:
        path.write_text(text)


pdf = BASE / "followup" / "mesquite_peterson_2004.pdf"
streams = []
for match in re.finditer(rb"(?<!end)stream(?:\r\n|\r|\n)(.*?)endstream", pdf.read_bytes(), re.S):
    try:
        content = zlib.decompress(match.group(1))
    except zlib.error:
        continue
    if b"BT" in content and (b"Tj" in content or b"TJ" in content):
        text = []
        for token in re.findall(rb"\((?:\\.|[^\\()])*\)", content):
            token = re.sub(rb"\\([0-7]{1,3})", lambda m: bytes([int(m.group(1), 8)]), token[1:-1])
            token = re.sub(rb"\\([()\\])", rb"\1", token)
            text.append(token.decode("latin1"))
        streams.append("".join(text))
save_or_compare(pdf.with_suffix(".literal-text.txt"), "\n\n".join(streams) + "\n")

docx = BASE / "selected" / "setaria_documentation.docx"
with zipfile.ZipFile(docx) as archive:
    root = ET.fromstring(archive.read("word/document.xml"))
    text = "\n".join("".join(p.itertext()) for p in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"))
save_or_compare(docx.with_suffix(".txt"), text + "\n")
print(f"PDF text streams: {len(streams)}; DOCX text characters: {len(text)}")
