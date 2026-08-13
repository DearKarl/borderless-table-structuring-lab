from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from PIL import Image


FINGERPRINT_KINDS = ("image_sha256", "structure", "text", "geometry", "source")


def _load(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dhash(path: Path) -> int:
    with Image.open(path) as image:
        resized = image.convert("L").resize((9, 8))
        get_pixels = getattr(resized, "get_flattened_data", resized.getdata)
        pixels = list(get_pixels())
    bits = 0
    for row in range(8):
        start = row * 9
        for col in range(8):
            bits = (bits << 1) | int(pixels[start + col] > pixels[start + col + 1])
    return bits


def _phash(path: Path) -> int:
    size = 32
    low = 8
    with Image.open(path) as image:
        resized = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
        get_pixels = getattr(resized, "get_flattened_data", resized.getdata)
        pixels = [float(value) for value in get_pixels()]
    cosines = [
        [math.cos((2 * position + 1) * frequency * math.pi / (2 * size)) for position in range(size)]
        for frequency in range(low)
    ]
    coefficients = []
    for vertical in range(low):
        for horizontal in range(low):
            coefficient = 0.0
            for row in range(size):
                vertical_weight = cosines[vertical][row]
                offset = row * size
                coefficient += vertical_weight * sum(
                    pixels[offset + col] * cosines[horizontal][col]
                    for col in range(size)
                )
            coefficients.append(coefficient)
    values = sorted(coefficients[1:])
    median = values[len(values) // 2]
    bits = 0
    for value in coefficients[1:]:
        bits = (bits << 1) | int(value > median)
    return bits


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _gold_fingerprints(record: dict[str, Any]) -> dict[str, str]:
    table = record["gold_record"]["canonical_table"]
    cells = sorted(
        table.get("cells", []),
        key=lambda cell: (
            int(cell.get("row", 0)),
            int(cell.get("col", 0)),
            str(cell.get("cell_id", "")),
        ),
    )
    structure = {
        "rows": table.get("rows"),
        "cols": table.get("cols"),
        "cells": [
            {
                "row": cell.get("row"),
                "col": cell.get("col"),
                "rowspan": cell.get("rowspan"),
                "colspan": cell.get("colspan"),
                "tag": cell.get("tag"),
            }
            for cell in cells
        ],
    }
    text = [str(cell.get("text", "")).strip() for cell in cells]
    geometry = [cell.get("bbox", cell.get("geometry", {}).get("bbox")) for cell in cells]
    return {
        "structure": _hash(structure),
        "text": _hash(text),
        "geometry": _hash(geometry),
        "source": str(record["image"]["source_document_hash"]),
    }


def audit(records: Iterable[dict[str, Any]], image_root: Path, perceptual_distance: int) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    record_count = 0
    for record in records:
        record_count += 1
        group_id = str(record["pair_group_id"])
        fingerprints = _gold_fingerprints(record)
        if group_id not in groups:
            groups[group_id] = {
                "roles": set(),
                "paths": set(),
                "fingerprints": {key: set() for key in FINGERPRINT_KINDS if key != "image_sha256"},
            }
        group = groups[group_id]
        for key, value in fingerprints.items():
            group["fingerprints"][key].add(value)
        group["roles"].add(str(record["role"]))
        group["paths"].add(str(record["image"]["relative_path"]))

    units = []
    blockers = []
    for group_id, group in groups.items():
        roles = group["roles"]
        paths = group["paths"]
        if len(roles) != 1 or len(paths) != 1 or any(
            len(values) != 1 for values in group["fingerprints"].values()
        ):
            blockers.append({
                "code": "GROUP_INPUT_INCONSISTENT",
                "pair_group_id": group_id,
                "roles": sorted(roles),
                "paths": sorted(paths),
                "fingerprint_cardinality": {
                    key: len(values) for key, values in group["fingerprints"].items()
                },
            })
            continue
        path = image_root / next(iter(paths))
        if not path.is_file():
            blockers.append({"code": "IMAGE_MISSING", "path": str(path)})
            continue
        fingerprints = {
            key: next(iter(values)) for key, values in group["fingerprints"].items()
        }
        fingerprints["image_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        units.append({
            "pair_group_id": group_id,
            "role": next(iter(roles)),
            "path": str(path),
            "dhash": _dhash(path),
            **fingerprints,
        })

    exact_overlaps: dict[str, list[dict[str, Any]]] = {}
    for kind in FINGERPRINT_KINDS:
        by_value: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for unit in units:
            by_value[str(unit[kind])].append(unit)
        hits = []
        for value, values in by_value.items():
            roles = sorted({item["role"] for item in values})
            if len(roles) > 1:
                hits.append({
                    "fingerprint": value,
                    "roles": roles,
                    "pair_group_ids": sorted(item["pair_group_id"] for item in values),
                })
        exact_overlaps[kind] = hits
        if hits:
            blockers.append({
                "code": "CROSS_ROLE_EXACT_OVERLAP",
                "kind": kind,
                "count": len(hits),
            })

    dhash_candidates = []
    perceptual_hits = []
    phash_cache: dict[str, int] = {}
    phash_distance = 6
    for left, right in itertools.combinations(units, 2):
        if left["role"] == right["role"]:
            continue
        distance = _hamming(int(left["dhash"]), int(right["dhash"]))
        if distance <= perceptual_distance:
            for unit in (left, right):
                path = str(unit["path"])
                if path not in phash_cache:
                    phash_cache[path] = _phash(Path(path))
            phash_value = _hamming(phash_cache[str(left["path"])], phash_cache[str(right["path"])])
            candidate = {
                "left_group": left["pair_group_id"],
                "left_role": left["role"],
                "right_group": right["pair_group_id"],
                "right_role": right["role"],
                "dhash_distance": distance,
                "phash_distance": phash_value,
            }
            dhash_candidates.append(candidate)
            if phash_value <= phash_distance:
                perceptual_hits.append(candidate)
    if perceptual_hits:
        blockers.append({
            "code": "CROSS_ROLE_PERCEPTUAL_OVERLAP",
            "count": len(perceptual_hits),
            "threshold": {
                "dhash": perceptual_distance,
                "phash": phash_distance,
            },
        })

    return {
        "schema_version": "mpr-tsr/raw-preserving-overlap-audit-v1",
        "records": record_count,
        "pair_groups": len(groups),
        "audited_units": len(units),
        "perceptual_hash": "dhash-9x8-screened-by-phash-32x32",
        "perceptual_hamming_threshold": {
            "dhash": perceptual_distance,
            "phash": phash_distance,
        },
        "exact_cross_role_overlaps": exact_overlaps,
        "dhash_cross_role_candidate_count": len(dhash_candidates),
        "dhash_cross_role_candidate_examples": dhash_candidates[:100],
        "perceptual_cross_role_overlaps": perceptual_hits,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "status": "PASS" if not blockers else "BLOCKED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed cross-role overlap audit.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--perceptual-distance", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(_load(args.input), args.image_root, args.perceptual_distance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "pair_groups": report["pair_groups"],
        "audited_units": report["audited_units"],
        "blocker_count": report["blocker_count"],
    }, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
