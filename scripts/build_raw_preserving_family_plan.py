from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TEMPLATE_CATALOG = (
    {"role": "train", "tables": 1, "layout": "stacked", "border": "none", "merged": False, "header_depth": 1, "density": "medium", "orientation": "portrait", "columns": 4, "body_rows": [9, 12]},
    {"role": "train", "tables": 1, "layout": "stacked", "border": "three_line", "merged": True, "header_depth": 2, "density": "dense", "orientation": "portrait", "columns": 5, "body_rows": [16, 20]},
    {"role": "train", "tables": 1, "layout": "stacked", "border": "full", "merged": True, "header_depth": 2, "density": "sparse", "orientation": "landscape", "columns": 6, "body_rows": [5, 8]},
    {"role": "train", "tables": 2, "layout": "stacked", "border": "none", "merged": True, "header_depth": 1, "density": "medium", "orientation": "portrait", "columns": 5, "body_rows": [10, 14]},
    {"role": "train", "tables": 2, "layout": "side_by_side", "border": "three_line", "merged": False, "header_depth": 1, "density": "medium", "orientation": "landscape", "columns": 4, "body_rows": [9, 13]},
    {"role": "train", "tables": 3, "layout": "stacked", "border": "none", "merged": True, "header_depth": 2, "density": "dense", "orientation": "portrait", "columns": 6, "body_rows": [16, 21]},
    {"role": "train", "tables": 2, "layout": "stacked", "border": "full", "merged": False, "header_depth": 2, "density": "sparse", "orientation": "portrait", "columns": 5, "body_rows": [5, 8]},
    {"role": "train", "tables": 1, "layout": "stacked", "border": "none", "merged": True, "header_depth": 2, "density": "dense", "orientation": "landscape", "columns": 6, "body_rows": [18, 24]},
    {"role": "development", "tables": 1, "layout": "stacked", "border": "three_line", "merged": True, "header_depth": 1, "density": "medium", "orientation": "landscape", "columns": 7, "body_rows": [11, 15]},
    {"role": "development", "tables": 2, "layout": "side_by_side", "border": "none", "merged": True, "header_depth": 2, "density": "dense", "orientation": "landscape", "columns": 7, "body_rows": [16, 22]},
    {"role": "holdout", "tables": 3, "layout": "stacked", "border": "three_line", "merged": False, "header_depth": 1, "density": "medium", "orientation": "portrait", "columns": 8, "body_rows": [10, 14]},
    {"role": "holdout", "tables": 1, "layout": "stacked", "border": "full", "merged": False, "header_depth": 2, "density": "dense", "orientation": "portrait", "columns": 9, "body_rows": [17, 23]},
    {"role": "holdout", "tables": 2, "layout": "stacked", "border": "three_line", "merged": True, "header_depth": 2, "density": "medium", "orientation": "portrait", "columns": 8, "body_rows": [11, 16]},
    {"role": "holdout", "tables": 2, "layout": "side_by_side", "border": "full", "merged": True, "header_depth": 1, "density": "sparse", "orientation": "landscape", "columns": 9, "body_rows": [5, 9]},
)

RENDERER_CATALOG = (
    {"role": "train", "profile": "clean"},
    {"role": "train", "profile": "jpeg_light"},
    {"role": "train", "profile": "blur_light"},
    {"role": "development", "profile": "downsample_light"},
    {"role": "holdout", "profile": "low_contrast"},
    {"role": "holdout", "profile": "jpeg_medium"},
)


def _role(index: int, groups: int, train_ratio: float, development_ratio: float) -> str:
    train_end = int(groups * train_ratio)
    development_end = train_end + int(groups * development_ratio)
    if index < train_end:
        return "train"
    if index < development_end:
        return "development"
    return "holdout"


def _catalog_entries(catalog: tuple[dict, ...], role: str) -> list[tuple[int, dict]]:
    return [(index, value) for index, value in enumerate(catalog) if value["role"] == role]


def build_plan(groups: int, seed: int, train_ratio: float, development_ratio: float) -> list[dict]:
    if groups <= 0:
        raise ValueError("groups must be positive")
    if not 0 < train_ratio < 1 or not 0 < development_ratio < 1:
        raise ValueError("split ratios must be between zero and one")
    if train_ratio + development_ratio >= 1:
        raise ValueError("train and development ratios must leave holdout groups")

    per_role_index = {"train": 0, "development": 0, "holdout": 0}
    plan = []
    for index in range(groups):
        role = _role(index, groups, train_ratio, development_ratio)
        role_index = per_role_index[role]
        per_role_index[role] += 1
        template_entries = _catalog_entries(TEMPLATE_CATALOG, role)
        renderer_entries = _catalog_entries(RENDERER_CATALOG, role)
        template_index, template_spec = template_entries[role_index % len(template_entries)]
        renderer_index, renderer_spec = renderer_entries[role_index % len(renderer_entries)]
        language = "en" if role_index % 4 == 0 else "cn"
        generation_seed = seed * 1_000_003 + index
        document_cluster_id = f"raw-preserving-v2-doc-{index:06d}"
        template_family_id = f"raw-preserving-v2-template-{template_index:02d}"
        content_family_id = f"raw-preserving-v2-content-{index:06d}"
        renderer_family_id = f"raw-preserving-v2-renderer-{renderer_index:02d}"
        identity = {
            "group_index": index,
            "role": role,
            "document_cluster_id": document_cluster_id,
            "template_family_id": template_family_id,
            "content_family_id": content_family_id,
            "renderer_family_id": renderer_family_id,
            "generation_seed": generation_seed,
            "language": language,
            "template_spec": template_spec,
            "renderer_spec": renderer_spec,
        }
        plan.append({
            "plan_version": "mpr-tsr/raw-preserving-family-plan-v2",
            "group_index": index,
            "pair_group_id": f"raw-preserving-v2-group-{index:06d}",
            **identity,
            "plan_entry_sha256": hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        })
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a terminal-blind pre-render family split plan.")
    parser.add_argument("--groups", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--development-ratio", type=float, default=0.10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan(args.groups, args.seed, args.train_ratio, args.development_ratio)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for entry in plan:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    role_counts = {}
    for entry in plan:
        role_counts[entry["role"]] = role_counts.get(entry["role"], 0) + 1
    print(json.dumps({
        "groups": len(plan),
        "role_counts": role_counts,
        "template_families": len({entry["template_family_id"] for entry in plan}),
        "renderer_families": len({entry["renderer_family_id"] for entry in plan}),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
