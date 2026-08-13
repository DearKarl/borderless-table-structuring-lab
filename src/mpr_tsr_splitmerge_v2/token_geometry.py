from __future__ import annotations

import copy
from typing import Any


def _bbox(value: dict[str, Any]) -> list[float] | None:
    geometry = value.get("geometry")
    box = geometry.get("bbox") if isinstance(geometry, dict) else value.get("bbox")
    if not isinstance(box, list) or len(box) != 4:
        return None
    return [float(item) for item in box]


def _contains(box: list[float], center: tuple[float, float]) -> bool:
    return box[0] <= center[0] <= box[2] and box[1] <= center[1] <= box[3]


def topology_signature(record: dict[str, Any]) -> tuple[Any, ...]:
    table = record.get("canonical_table", record)
    cells = table.get("cells", []) if isinstance(table, dict) else []
    return (
        table.get("rows"),
        table.get("cols"),
        tuple(sorted(
            (
                cell.get("row", cell.get("row_start")),
                cell.get("col", cell.get("col_start")),
                cell.get("rowspan", cell.get("row_end")),
                cell.get("colspan", cell.get("col_end")),
                cell.get("tag", "td"),
            )
            for cell in cells
        )),
    )


def ownership_violations(
    record: dict[str, Any],
    ocr_tokens: list[dict[str, Any]],
) -> int | None:
    table = record.get("canonical_table", record)
    cells = table.get("cells", []) if isinstance(table, dict) else []
    violations = 0
    seen = set()
    for cell in cells:
        cell_box = _bbox(cell)
        if cell_box is None:
            return None
        indexes = cell.get("ocr_token_indexes")
        if not isinstance(indexes, list):
            return None
        for index in indexes:
            if not isinstance(index, int) or index < 0 or index >= len(ocr_tokens) or index in seen:
                return None
            token_box = _bbox(ocr_tokens[index])
            if token_box is None:
                return None
            seen.add(index)
            center = ((token_box[0] + token_box[2]) / 2, (token_box[1] + token_box[3]) / 2)
            if not _contains(cell_box, center):
                violations += 1
    return violations


def reassign_tokens_by_geometry(
    raw_record: dict[str, Any],
    ocr_tokens: list[dict[str, Any]],
) -> dict[str, Any] | None:
    candidate = copy.deepcopy(raw_record)
    table = candidate.get("canonical_table", candidate)
    cells = table.get("cells", []) if isinstance(table, dict) else []
    cell_boxes = [_bbox(cell) for cell in cells]
    if any(box is None for box in cell_boxes):
        return None
    assignments: list[list[int]] = [[] for _ in cells]
    for index, token in enumerate(ocr_tokens):
        token_box = _bbox(token)
        if token_box is None:
            return None
        center = ((token_box[0] + token_box[2]) / 2, (token_box[1] + token_box[3]) / 2)
        owners = [cell_index for cell_index, box in enumerate(cell_boxes) if _contains(box, center)]
        if len(owners) != 1:
            return None
        assignments[owners[0]].append(index)
    for cell, indexes in zip(cells, assignments):
        indexes.sort()
        cell["ocr_token_indexes"] = indexes
        cell["text"] = "".join(str(ocr_tokens[index].get("text", "")) for index in indexes)
    provenance = candidate.setdefault("provenance", {})
    provenance["producer"] = "token-geometry-reassignment"
    provenance["producer_version"] = "v1"
    provenance["purpose"] = "observable_token_assignment_candidate"
    return candidate
