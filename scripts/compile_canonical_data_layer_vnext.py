#!/usr/bin/env python3
"""Compile the non-overwriting Canonical Data Layer vNext target layer.

The compiler streams Formal20k v1, preserves every source record by hash and
line reference, and emits direct Canonical Table state plus order-invariant
route targets. It never rewrites the source manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import types
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PACKAGE = types.ModuleType("mpr_tsr_splitmerge_v2")
PACKAGE.__path__ = [str(ROOT / "src/mpr_tsr_splitmerge_v2")]
sys.modules.setdefault("mpr_tsr_splitmerge_v2", PACKAGE)

from mpr_tsr_splitmerge_v2.canonical import (  # noqa: E402
    canonical_cells,
    normalize_text,
    table_shape,
    topology_key,
    validate_cells,
)


SCHEMA = "mpr-tsr/canonical-data-layer-vnext-v1"
EXPECTED_MANIFEST_SHA256 = (
    "716ad18e1c26a58ced016dcf8424049a4081c92e39215ad7b56268b3fd43aa99"
)
EXPECTED_RECORDS = 20_000
EXPECTED_QUARANTINE = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def jsonl_rows(path: Path) -> Iterable[tuple[int, dict[str, Any], str]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record is not an object")
            yield line_number, value, hashlib.sha256(line.encode("utf-8")).hexdigest()


def geometry(value: Mapping[str, Any]) -> dict[str, Any]:
    bbox = value.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        return {
            "status": "present",
            "bbox": [round(float(item), 6) for item in bbox[:4]],
        }
    return {"status": "missing", "bbox": None}


def topology_record(value: Mapping[str, Any]) -> dict[str, int]:
    row, col, rowspan, colspan = topology_key(dict(value))
    return {
        "row_start": row,
        "row_end": row + rowspan,
        "col_start": col,
        "col_end": col + colspan,
    }


def topology_id(value: Mapping[str, Any]) -> str:
    item = topology_record(value)
    return (
        f"r{item['row_start']}:{item['row_end']}"
        f"c{item['col_start']}:{item['col_end']}"
    )


def merge_path_ambiguity(record: Mapping[str, Any]) -> dict[str, int]:
    residual = record.get("residual_edit_labels") or {}
    per_gold: dict[int, set[int]] = defaultdict(set)
    for edge in residual.get("merge_edges", []):
        gold_index = int(edge["gold_index"])
        per_gold[gold_index].add(int(edge["source_raw_index"]))
        per_gold[gold_index].add(int(edge["target_raw_index"]))
    components = sum(len(nodes) >= 3 for nodes in per_gold.values())
    raw_cells = sum(len(nodes) for nodes in per_gold.values() if len(nodes) >= 3)
    return {"components": components, "raw_cells": raw_cells}


def ownership(record: Mapping[str, Any], gold_count: int) -> list[list[int]]:
    """Return Canonical Gold cell ownership from audited text-edit labels.

    The historical ``ocr_gold_pointer`` indexes primitive-grid owners, not
    Canonical Gold cells, and therefore must not be interpreted as a Gold
    cell index. The text-edit labels expose the replayed Gold index and the
    exact OCR token indexes for each Canonical cell.
    """
    owners: list[list[int]] = [[] for _ in range(gold_count)]
    tokens = list(record.get("ocr_tokens") or [])
    valid_token_indexes = {
        int(token.get("token_index", position))
        for position, token in enumerate(tokens)
    }
    assigned: set[int] = set()
    text_cells = list((record.get("text_edit_labels") or {}).get("cells") or [])
    if len(text_cells) != gold_count:
        raise ValueError(
            f"text-label cell count {len(text_cells)} != Gold count {gold_count}"
        )
    for item in text_cells:
        gold_index = int(item.get("gold_index", -1))
        if gold_index < 0 or gold_index >= gold_count:
            raise ValueError(f"text-label Gold index {gold_index} is outside range")
        for value in item.get("ocr_token_indexes", []):
            token_index = int(value)
            if token_index not in valid_token_indexes:
                raise ValueError(f"OCR token index {token_index} does not exist")
            if token_index in assigned:
                raise ValueError(f"OCR token index {token_index} has multiple owners")
            assigned.add(token_index)
            owners[gold_index].append(token_index)
    return [sorted(values) for values in owners]


def canonical_state(
    record: Mapping[str, Any], gold: list[dict[str, Any]]
) -> dict[str, Any]:
    owners = ownership(record, len(gold))
    cells: list[dict[str, Any]] = []
    semantic_cells: list[dict[str, Any]] = []
    for gold_index, cell in enumerate(gold):
        topo = topology_record(cell)
        original_text = str(cell.get("text", "") or "").strip()
        normalized = normalize_text(original_text)
        tag = "th" if str(cell.get("tag", "td")).lower() == "th" else "td"
        semantic = {
            "cell_id": topology_id(cell),
            **topo,
            "normalized_text": normalized,
            "tag": tag,
        }
        semantic_cells.append(semantic)
        cells.append(
            {
                **semantic,
                "gold_index": gold_index,
                "text": original_text,
                "ocr_token_indexes": owners[gold_index],
                "geometry": geometry(cell),
            }
        )
    rows, cols = table_shape(gold)
    full = {"rows": rows, "cols": cols, "cells": cells}
    return {
        **full,
        "semantic_state_sha256": stable_hash(
            {"rows": rows, "cols": cols, "cells": semantic_cells}
        ),
        "full_state_sha256": stable_hash(full),
    }


def explicit_target(
    raw: list[dict[str, Any]], gold: list[dict[str, Any]]
) -> dict[str, Any]:
    raw_by_id = {topology_id(cell): topology_record(cell) for cell in raw}
    gold_by_id = {topology_id(cell): topology_record(cell) for cell in gold}
    raw_ids = set(raw_by_id)
    gold_ids = set(gold_by_id)
    return {
        "schema_version": "mpr-tsr/explicit-topology-difference-set-v1",
        "target_type": "order_invariant_partition_difference",
        "default_action": "KEEP",
        "text_policy": "PRESERVE_RAW_BY_DEFAULT",
        "preserve": [raw_by_id[key] for key in sorted(raw_ids & gold_ids)],
        "remove": [raw_by_id[key] for key in sorted(raw_ids - gold_ids)],
        "add": [gold_by_id[key] for key in sorted(gold_ids - raw_ids)],
    }


def historical_hashes(record: Mapping[str, Any]) -> dict[str, str]:
    keys = (
        "structure_labels",
        "residual_edit_labels",
        "text_candidates",
        "text_edit_labels",
        "raw_to_gold_diff",
    )
    return {key: stable_hash(record.get(key)) for key in keys}


def load_quarantine(path: Path) -> set[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "complete":
        raise ValueError("grid-hole classification is not complete")
    sample_ids = {str(row["sample_id"]) for row in value.get("records_detail", [])}
    if len(sample_ids) != EXPECTED_QUARANTINE:
        raise ValueError(
            f"expected {EXPECTED_QUARANTINE} grid-hole IDs, found {len(sample_ids)}"
        )
    return sample_ids


def atomic_writer(path: Path):
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    return temporary, temporary.open("w", encoding="utf-8")


def write_line(handle, value: Mapping[str, Any]) -> None:
    handle.write(stable_json(value) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--grid-hole-classification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    output = args.output_dir.resolve()
    working = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    classification = args.grid_hole_classification.resolve()
    if output.exists():
        raise SystemExit(f"non-overwrite refusal: output directory exists: {output}")
    if working.exists():
        raise SystemExit(f"temporary output directory exists: {working}")
    if sha256(manifest) != EXPECTED_MANIFEST_SHA256:
        raise SystemExit("frozen Formal20k manifest SHA256 mismatch")
    quarantine_ids = load_quarantine(classification)
    working.mkdir(parents=True)

    paths = {
        "targets": working / "canonical_targets.jsonl",
        "replay": working / "replay_map.jsonl",
        "included": working / "included_manifest.jsonl",
        "quarantined": working / "quarantined_manifest.jsonl",
    }
    writers = {name: atomic_writer(path) for name, path in paths.items()}
    handles = {name: pair[1] for name, pair in writers.items()}
    counts: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    geometry_status: Counter[str] = Counter()
    ambiguity_by_source: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []

    try:
        for line_number, record, record_sha in jsonl_rows(manifest):
            counts["records"] += 1
            sample_id = str(record.get("sample_id") or "")
            if not sample_id:
                raise ValueError(f"line {line_number}: missing sample_id")
            source = str((record.get("metadata") or {}).get("source_dataset") or "unknown")
            sources[source] += 1
            raw = canonical_cells(
                (record.get("raw_mineru_prior") or {}).get("cells", []),
                include_geometry=False,
            )
            gold = canonical_cells((record.get("gold") or {}).get("canonical_cells", []))
            raw_errors = validate_cells(raw, require_complete=True)
            gold_errors = validate_cells(gold, require_complete=True)
            is_quarantined = sample_id in quarantine_ids
            if gold_errors:
                raise ValueError(f"{sample_id}: invalid Canonical Gold: {gold_errors}")
            if raw_errors and not is_quarantined:
                raise ValueError(f"{sample_id}: unexpected invalid Raw grid: {raw_errors}")
            if is_quarantined and not raw_errors:
                raise ValueError(f"{sample_id}: frozen grid-hole record is now Raw-valid")

            ambiguity = merge_path_ambiguity(record)
            if ambiguity["components"]:
                counts["records_with_historical_action_path_nonuniqueness"] += 1
                counts["historical_nonunique_components"] += ambiguity["components"]
                counts["historical_nonunique_raw_cells"] += ambiguity["raw_cells"]
                ambiguity_by_source[source] += 1

            state = canonical_state(record, gold)
            geometry_status.update(
                cell["geometry"]["status"] for cell in state["cells"]
            )
            status = "quarantined" if is_quarantined else "included"
            source_record = {
                "manifest_path": str(manifest),
                "manifest_sha256": EXPECTED_MANIFEST_SHA256,
                "line_number": line_number,
                "record_sha256": record_sha,
                "image_sha256": str((record.get("image") or {}).get("input_image_sha256") or ""),
            }
            active_target = not is_quarantined
            target = {
                "schema_version": SCHEMA,
                "sample_id": sample_id,
                "document_id": str(record.get("document_id") or sample_id),
                "split": str(record.get("split") or "unknown"),
                "source_dataset": source,
                "record_status": status,
                "source_record": source_record,
                "canonical_table": state,
                "explicit_topology_target": explicit_target(raw, gold) if active_target else None,
                "lora_canonical_candidate_target": (
                    {
                        "schema_version": "mpr-tsr/lora-canonical-candidate-target-v1",
                        "scope": "TABLE_ONLY",
                        "canonical_state_sha256": state["full_state_sha256"],
                        "geometry_policy": "PARALLEL_OR_SHARED_SIDECAR",
                        "text_policy": "OCR_COPY_PREFERRED",
                    }
                    if active_target
                    else None
                ),
                "historical_supervision": {
                    "primary_truth": False,
                    "historical_action_path_nonunique": bool(ambiguity["components"]),
                    "ambiguity": ambiguity,
                    "artifact_sha256": historical_hashes(record),
                },
                "quarantine": (
                    {
                        "policy": "PRESERVE_AND_QUARANTINE",
                        "reason": "RAW_GRID_HOLE",
                        "raw_validation_errors": raw_errors,
                        "requires_separate_repair_evidence": True,
                    }
                    if is_quarantined
                    else None
                ),
            }
            write_line(handles["targets"], target)
            replay = {
                "sample_id": sample_id,
                "source_record_sha256": record_sha,
                "historical_artifact_sha256": historical_hashes(record),
                "canonical_semantic_state_sha256": state["semantic_state_sha256"],
                "canonical_full_state_sha256": state["full_state_sha256"],
                "historical_action_path_nonunique": bool(ambiguity["components"]),
            }
            write_line(handles["replay"], replay)
            manifest_row = {
                "sample_id": sample_id,
                "document_id": str(record.get("document_id") or sample_id),
                "source_dataset": source,
                "record_status": status,
                "canonical_full_state_sha256": state["full_state_sha256"],
            }
            write_line(handles["quarantined" if is_quarantined else "included"], manifest_row)
            counts[status] += 1
            counts["records_processed"] += 1
    except Exception as error:
        failures.append({"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        for handle in handles.values():
            handle.close()

    if counts["records"] != EXPECTED_RECORDS:
        raise SystemExit(f"record count mismatch: {counts['records']}")
    if counts["quarantined"] != EXPECTED_QUARANTINE:
        raise SystemExit(f"quarantine count mismatch: {counts['quarantined']}")
    if counts["included"] != EXPECTED_RECORDS - EXPECTED_QUARANTINE:
        raise SystemExit(f"included count mismatch: {counts['included']}")

    for name, (temporary, _) in writers.items():
        temporary.replace(paths[name])

    report = {
        "schema_version": "mpr-tsr/canonical-data-layer-vnext-compile-report-v1",
        "status": "PASS",
        "contract_sha256": args.contract_sha256,
        "source_manifest": str(manifest),
        "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "source_manifest_modified": False,
        "terminal_customer50_or_omnidoc_inputs_read": False,
        "primary_truth": "DIRECT_CANONICAL_TABLE_STATE",
        "grid_hole_policy": "PRESERVE_AND_QUARANTINE",
        "counts": dict(sorted(counts.items())),
        "sources": dict(sorted(sources.items())),
        "geometry_status": dict(sorted(geometry_status.items())),
        "historical_action_path_nonuniqueness_by_source": dict(
            sorted(ambiguity_by_source.items())
        ),
        "failures": failures,
        "outputs": {
            name: {"path": str(output / path.name), "sha256": sha256(path)}
            for name, path in paths.items()
        },
    }
    report_path = working / "compile_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sum_paths = [*paths.values(), report_path]
    (working / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in sum_paths),
        encoding="utf-8",
    )
    working.replace(output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
