"""Explicit source image + frozen MinerU Markdown interface. No Gold fields."""
from pathlib import Path
import re

from .io import inside, publish, read, sha

SCHEMA = "btsl-source-pages/1"
FIELDS = {"id", "image", "image_sha256", "raw", "raw_sha256"}
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".ppm"}


def valid_id(value):
    return (isinstance(value, str) and bool(value) and not value.startswith(".")
            and len(value.encode("utf-8")) <= 251 and not re.search(r"[/\\\x00-\x1f\x7f]", value))


def validate(manifest):
    manifest = Path(manifest).resolve()
    value = read(manifest)
    if set(value) != {"schema", "pages"} or value["schema"] != SCHEMA or not value["pages"]:
        raise ValueError("Expected nonempty source-only manifest")
    rows, ids = [], set()
    for item in value["pages"]:
        if not isinstance(item, dict) or set(item) != FIELDS:
            raise ValueError("Manifest accepts image/Raw bindings only, never labels or answers")
        identity = item["id"]
        if not valid_id(identity) or identity in ids:
            raise ValueError("Invalid or duplicate page id")
        ids.add(identity)
        row = dict(item)
        for key in ("image", "raw"):
            path = inside(manifest.parent, item[key])
            if not path.is_file() or sha(path) != item[key + "_sha256"]:
                raise ValueError("Missing/changed input: " + str(path))
            row[key + "_path"] = path
        row["raw_path"].read_bytes().decode("utf-8", errors="strict")
        rows.append(row)
    return rows


def prepare(root, images="images", raw="raw", name="manifest.json"):
    root = Path(root).resolve()
    image_root, raw_root = inside(root, images), inside(root, raw)
    output = inside(root, name)
    if output.exists():
        raise ValueError("Manifest already exists")
    rows = []
    for image in sorted(image_root.iterdir()):
        if image.suffix.lower() not in EXTENSIONS or not image.is_file():
            continue
        md = raw_root / (image.stem + ".md")
        if not md.is_file():
            raise ValueError("Every image needs matching MinerU Markdown: " + image.name)
        if not valid_id(image.stem):
            raise ValueError("Unsafe or overlong page identifier: " + image.stem)
        rows.append({"id": image.stem, "image": str(image.relative_to(root)),
                     "image_sha256": sha(image), "raw": str(md.relative_to(root)), "raw_sha256": sha(md)})
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("No images or duplicate stems")
    if {p.stem for p in raw_root.glob("*.md")} != {r["id"] for r in rows}:
        raise ValueError("Image/Markdown page sets differ")
    publish(output, {"schema": SCHEMA, "pages": rows})
    validate(output)
    return {"manifest": str(output), "pages": len(rows), "sha256": sha(output)}
