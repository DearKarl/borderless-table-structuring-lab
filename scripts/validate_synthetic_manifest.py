from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FORBIDDEN_PATH_PARTS = {"..", "~"}


def iter_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        records = list(value) if isinstance(value, list) else list(value.values()) if isinstance(value, dict) else []
    if not records or not all(isinstance(record, dict) for record in records):
        raise ValueError("manifest must contain one or more object records")
    return records


def validate_manifest(manifest: Path, asset_root: Path) -> int:
    records = iter_records(manifest)
    checked = 0
    for index, record in enumerate(records):
        image = record.get("image", record.get("file"))
        if not isinstance(image, str) or not image:
            raise ValueError(f"record {index} has no image/file reference")
        image_path = Path(image)
        if image_path.is_absolute() or any(part in FORBIDDEN_PATH_PARTS for part in image_path.parts):
            raise ValueError(f"record {index} contains an unsafe image path: {image}")
        if not (asset_root / image_path).is_file():
            raise ValueError(f"record {index} references a missing asset: {image}")
        checked += 1
    return checked


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a synthetic-data manifest without reading model outputs.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("asset_root", type=Path)
    args = parser.parse_args()
    checked = validate_manifest(args.manifest, args.asset_root)
    print(f"validated_records={checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
