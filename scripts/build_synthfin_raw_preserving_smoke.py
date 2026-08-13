from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

from mpr_tsr_splitmerge_v2.counterfactual import build_candidate
from mpr_tsr_splitmerge_v2.raw_preserving import KEEP, label_oracle_action


def _provenance(sample_id: str, source_hash: str, purpose: str) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "producer": "raw-preserving-synthfin-adapter",
        "producer_version": "v1",
        "purpose": purpose,
        "input_image_sha256": source_hash,
        "terminal_benchmarks_visible": False,
    }


def _cell_token_count(text: str) -> int:
    return max(1, len(text)) if text else 0


def _canonical_table(table: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cells = []
    tokens = []
    for index, source in enumerate(table.get("cells", [])):
        row = int(source["row_start"])
        col = int(source["col_start"])
        row_end = int(source["row_end"])
        col_end = int(source["col_end"])
        text = str(source.get("text", "") or "")
        bbox = [float(value) for value in source["bbox"]]
        token_indexes = list(range(len(tokens), len(tokens) + _cell_token_count(text)))
        if text:
            token_width = (bbox[2] - bbox[0]) / len(text)
            for char_index, char in enumerate(text):
                token_bbox = [
                    bbox[0] + char_index * token_width,
                    bbox[1],
                    bbox[0] + (char_index + 1) * token_width,
                    bbox[3],
                ]
                tokens.append({"text": char, "bbox": token_bbox})
        cells.append({
            "cell_id": f"gold-{index:05d}",
            "row": row,
            "col": col,
            "rowspan": row_end - row + 1,
            "colspan": col_end - col + 1,
            "text": text,
            "tag": "th" if source.get("is_header") else "td",
            "bbox": bbox,
            "geometry": {"status": "present", "bbox": bbox},
            "ocr_token_indexes": token_indexes,
        })
    return {
        "rows": int(table["n_rows"]),
        "cols": int(table["n_cols"]),
        "cells": cells,
    }, tokens


def _record(
    table: dict[str, Any],
    *,
    sample_id: str,
    source_hash: str,
    purpose: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    canonical, tokens = _canonical_table(table)
    return {
        "canonical_table": canonical,
        "ocr_tokens": tokens,
        "non_table_state_sha256": hashlib.sha256(
            f"{source_hash}:non-table".encode("utf-8")
        ).hexdigest(),
        "provenance": _provenance(sample_id, source_hash, purpose),
    }, tokens


def _clone_with_provenance(record: dict[str, Any], purpose: str) -> dict[str, Any]:
    value = copy.deepcopy(record)
    value["provenance"]["purpose"] = purpose
    return value


def _load_tables(root: Path, limit: int, one_table_per_page: bool = False) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    manifests = sorted(root.glob("shard_*/manifest.jsonl"))
    root_manifest = root / "manifest.jsonl"
    if root_manifest.is_file():
        manifests = [root_manifest, *manifests]
    for manifest in manifests:
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                page = json.loads(line)
                for table in page.get("tables") or []:
                    if not table.get("cells") or not table.get("n_rows") or not table.get("n_cols"):
                        continue
                    entries.append({"page": page, "table": table, "manifest": str(manifest)})
                    if len(entries) >= limit:
                        return entries
                    if one_table_per_page:
                        break
    return entries


def _validate_generator_policy(page: dict[str, Any]) -> None:
    if page.get("terminal_data_used") is not False:
        raise ValueError("formal corpus requires terminal_data_used=false")
    if page.get("generator_policy") != "terminal_blind_v1":
        raise ValueError("formal corpus requires generator_policy=terminal_blind_v1")
    identity = " ".join(
        str(page.get(key, ""))
        for key in ("renderer_version", "generator_policy", "source_dataset")
    ).lower()
    forbidden = ("eval50", "eval_50", "omnidoc", "customer50", "customer_50")
    matches = [token for token in forbidden if token in identity]
    if matches:
        raise ValueError(f"formal corpus generator identity is terminal-derived: {matches}")


def _family_fields(page: dict[str, Any], require_family_plan: bool) -> dict[str, Any]:
    if require_family_plan:
        _validate_generator_policy(page)
    keys = (
        "role",
        "pair_group_id",
        "document_cluster_id",
        "template_family_id",
        "content_family_id",
        "renderer_family_id",
        "generation_seed",
        "family_plan_entry_sha256",
    )
    missing = [key for key in keys if page.get(key) in (None, "")]
    if require_family_plan and missing:
        raise ValueError(f"manifest page lacks family-plan fields: {missing}")
    source_hash = str(page["source_document_hash"])
    legacy_seed = int(source_hash[:16], 16)
    return {
        "role": page.get("role", "development"),
        "pair_group_id": page.get("pair_group_id"),
        "document_cluster_id": page.get("document_cluster_id", f"source-{source_hash}"),
        "template_family_id": page.get("template_family_id", source_hash),
        "content_family_id": page.get("content_family_id", source_hash),
        "renderer_family_id": page.get(
            "renderer_family_id", page.get("renderer_version", "unknown")
        ),
        "generation_seed": page.get("generation_seed", legacy_seed),
        "family_plan_entry_sha256": page.get("family_plan_entry_sha256"),
    }


def _make_pair(
    entry: dict[str, Any],
    group_index: int,
    local_index: int,
    *,
    dataset_root: Path,
    raw: dict[str, Any],
    gold: dict[str, Any],
    candidate: dict[str, Any] | None,
    ocr_tokens: list[dict[str, Any]],
    operator: str,
    tag: str,
    require_family_plan: bool,
) -> dict[str, Any]:
    page = entry["page"]
    source_hash = str(page["source_document_hash"])
    family = _family_fields(page, require_family_plan)
    table_index = int(entry["table"]["table_index"])
    pair_group_id = family["pair_group_id"]
    if pair_group_id:
        pair_group_id = f"{pair_group_id}-table-{table_index:02d}"
    else:
        pair_group_id = f"synthfin-v34-group-{group_index:04d}"
    sample_id = (
        f"{pair_group_id}-cf-{local_index:02d}"
        if require_family_plan
        else f"synthfin-v34-rp-{group_index:04d}-{local_index:02d}"
    )
    decision = label_oracle_action(raw, candidate, gold, ocr_tokens=ocr_tokens)
    image = Path(str(page["image"]))
    image_path = image if image.is_absolute() else dataset_root / image
    if not image_path.is_file():
        raise ValueError(f"image does not exist: {image_path}")
    if not image.is_absolute():
        relative_image = image.as_posix()
    else:
        relative_image = os.path.relpath(str(image), str(dataset_root)).replace(os.sep, "/")
    if require_family_plan and (Path(relative_image).is_absolute() or ".." in Path(relative_image).parts):
        raise ValueError("formal image path must be relative to the dataset root")
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    declared_image_sha256 = page.get("image_sha256")
    if require_family_plan and declared_image_sha256 != image_sha256:
        raise ValueError("formal image SHA256 does not match the manifest")
    return {
        "schema_version": "mpr-tsr/raw-preserving-pair-v1",
        "sample_id": sample_id,
        "pair_group_id": pair_group_id,
        "role": family["role"],
        "image": {
            "relative_path": relative_image,
            "sha256": image_sha256,
            "source_document_hash": source_hash,
            "table_index": table_index,
            "image_size": page.get("image_size"),
        },
        "raw_record": raw,
        "gold_record": gold,
        "candidate_record": candidate,
        "oracle_decision": decision.as_dict(),
        "phenomenon_tags": [tag, str(entry["table"].get("meta", {}).get("border", "unknown"))],
        "operator": {"name": operator, "version": "v1", "terminal_data_used": False},
        "provenance": {
            "generator": str(page.get("source_dataset", "unknown")),
            "generator_version": str(page.get("renderer_version", "unknown")),
            **(
                {"generator_policy": str(page["generator_policy"])}
                if page.get("generator_policy") is not None else {}
            ),
            "generation_seed": family["generation_seed"],
            "terminal_data_used": False,
            "document_cluster_id": family["document_cluster_id"],
            "template_family_id": family["template_family_id"],
            "content_family_id": family["content_family_id"],
            "renderer_family_id": family["renderer_family_id"],
            **(
                {"family_plan_entry_sha256": str(family["family_plan_entry_sha256"])}
                if family["family_plan_entry_sha256"] is not None else {}
            ),
            "source_dataset": str(page.get("source_dataset", "SynthFin")),
            "license": str(page.get("license", "synthetic")),
            **({"font": page["font"]} if page.get("font") is not None else {}),
        },
    }


def build_group(
    entry: dict[str, Any],
    group_index: int,
    *,
    dataset_root: Path,
    require_family_plan: bool,
) -> list[dict[str, Any]]:
    page = entry["page"]
    table = entry["table"]
    source_hash = str(page["source_document_hash"])
    gold, ocr_tokens = _record(table, sample_id=f"gold-{group_index:04d}", source_hash=source_hash, purpose="offline_gold")
    raw_good = _clone_with_provenance(gold, "raw_like_identity")
    harmful = build_candidate(raw_good, gold, operator="over_merge")
    if harmful is None:
        harmful = build_candidate(raw_good, gold, operator="over_split")
    if harmful is None:
        harmful = _clone_with_provenance(raw_good, "raw_like_identity_fallback")
    raw_bad = build_candidate(
        gold,
        gold,
        operator="assignment_swap" if require_family_plan else "over_merge",
    )
    if raw_bad is None:
        raw_bad = build_candidate(gold, gold, operator="over_merge")
    if raw_bad is None:
        raw_bad = copy.deepcopy(gold)
        cells = raw_bad["canonical_table"]["cells"]
        if cells:
            cells[0]["text"] = f"{cells[0]['text']}__raw_error"
    records = []
    specifications = [
        (raw_good, _clone_with_provenance(raw_good, "identity_candidate"), "identity", "raw_good_identity"),
        (raw_good, harmful, "over_edit", "raw_good_harmful_candidate"),
        (raw_good, harmful, "over_edit_repeat", "raw_good_harmful_candidate_repeat"),
        (raw_good, _clone_with_provenance(raw_good, "identity_candidate_repeat"), "identity_repeat", "raw_good_identity_repeat"),
        (raw_bad, _clone_with_provenance(raw_bad, "identity_candidate"), "identity", "raw_bad_identity"),
        (raw_bad, harmful, "over_edit", "raw_bad_harmful_candidate"),
        (raw_bad, _clone_with_provenance(raw_bad, "identity_candidate_tie"), "identity_tie", "raw_bad_tie"),
        (raw_bad, harmful, "partial_candidate", "raw_bad_partial_or_harmful"),
        (raw_bad, _clone_with_provenance(raw_bad, "non_table_change_candidate"), "non_table_change", "raw_bad_non_table_control"),
        (raw_bad, _clone_with_provenance(gold, "gold_candidate"), "gold_candidate", "raw_bad_gold_candidate"),
    ]
    for index, (raw_value, candidate, operator, tag) in enumerate(specifications):
        if operator == "non_table_change":
            candidate = copy.deepcopy(candidate)
            candidate["non_table_state_sha256"] = "changed-non-table-state"
        records.append(_make_pair(
            entry, group_index, index, dataset_root=dataset_root, raw=raw_value, gold=gold,
            candidate=candidate, ocr_tokens=ocr_tokens, operator=operator, tag=tag,
            require_family_plan=require_family_plan,
        ))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a 256-group SynthFin Raw-preserving smoke manifest.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=256)
    parser.add_argument("--require-family-plan", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    entries = _load_tables(
        args.root,
        args.groups,
        one_table_per_page=args.require_family_plan,
    )
    if len(entries) != args.groups:
        raise RuntimeError(f"requested {args.groups} table groups, found {len(entries)}")
    records = [
        record
        for index, entry in enumerate(entries)
        for record in build_group(
            entry,
            index,
            dataset_root=args.root,
            require_family_plan=args.require_family_plan,
        )
    ]
    counts: dict[str, int] = {}
    for record in records:
        action = record["oracle_decision"]["action"]
        counts[action] = counts.get(action, 0) + 1
    expected = {KEEP: args.groups * 9, "ACCEPT_EDIT": args.groups}
    if counts != expected:
        raise RuntimeError(f"oracle distribution mismatch: expected={expected} observed={counts}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"groups": args.groups, "records": len(records), "action_counts": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
