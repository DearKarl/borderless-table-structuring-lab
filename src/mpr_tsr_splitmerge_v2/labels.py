from __future__ import annotations

from collections import defaultdict, deque
from statistics import median
from typing import Any

from .canonical import (
    canonical_cells,
    normalize_text,
    occupied_slots,
    strict_key,
    table_shape,
    topology_key,
    validate_cells,
)


CONTROL_POINTS = 4
HEADER_POLICY_VERSION = "gold-functional-th-td/v1"
RESIDUAL_EDIT_POLICY_VERSION = "logical-topology-minimal-edit/v1"


def _logical_box(
    cell: dict[str, Any], rows: int, cols: int
) -> tuple[float, float, float, float]:
    return (
        float(cell["col"]) / max(1, cols),
        float(cell["row"]) / max(1, rows),
        float(cell["col"] + cell.get("colspan", 1)) / max(1, cols),
        float(cell["row"] + cell.get("rowspan", 1)) / max(1, rows),
    )


def _intersection_over_smaller(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    return intersection / max(1e-9, min(first_area, second_area))


def _merge_tree_edges(
    indexes: list[int], cells: list[dict[str, Any]]
) -> list[tuple[int, str, int]]:
    """Return one acyclic directed edge per primitive absorbed by a merge."""

    nodes = sorted(set(indexes))
    if len(nodes) < 2:
        return []
    candidates: list[tuple[int, str, int]] = []
    for left in nodes:
        first = cells[left]
        first_slots = occupied_slots(first)
        for right in nodes:
            if left == right:
                continue
            second = cells[right]
            second_slots = occupied_slots(second)
            if any(
                row == other_row and col + 1 == other_col
                for row, col in first_slots
                for other_row, other_col in second_slots
            ):
                candidates.append((left, "merge_right", right))
            if any(
                col == other_col and row + 1 == other_row
                for row, col in first_slots
                for other_row, other_col in second_slots
            ):
                candidates.append((left, "merge_down", right))
    adjacency: dict[int, list[tuple[int, str, int]]] = defaultdict(list)
    for edge in candidates:
        adjacency[edge[0]].append(edge)
        adjacency[edge[2]].append(edge)
    root = min(nodes, key=lambda index: (cells[index]["row"], cells[index]["col"]))
    queue = deque([root])
    visited = {root}
    selected: list[tuple[int, str, int]] = []
    while queue:
        current = queue.popleft()
        for edge in sorted(adjacency[current]):
            other = edge[2] if edge[0] == current else edge[0]
            if other in visited:
                continue
            visited.add(other)
            queue.append(other)
            selected.append(edge)
    return selected


def compile_residual_edit_labels(
    raw_values: list[dict[str, Any]],
    gold_values: list[dict[str, Any]],
    structure_labels: dict[str, Any],
) -> dict[str, Any]:
    """Compile explicit, cell-anchored edits without raw physical geometry.

    Raw MinerU HTML provides topology and text but no trustworthy cell boxes or
    confidence.  Alignment therefore uses normalized logical topology only.
    Physical target boundaries remain a separate Gold/visual supervision path.
    """

    raw = canonical_cells(raw_values, include_geometry=False)
    gold = canonical_cells(gold_values, include_geometry=False)
    raw_rows, raw_cols = table_shape(raw)
    gold_rows, gold_cols = table_shape(gold)
    raw_boxes = [_logical_box(cell, raw_rows, raw_cols) for cell in raw]
    gold_boxes = [_logical_box(cell, gold_rows, gold_cols) for cell in gold]
    overlap = [
        [_intersection_over_smaller(first, second) for second in gold_boxes]
        for first in raw_boxes
    ]
    raw_matches: list[list[int]] = []
    for raw_index, scores in enumerate(overlap):
        matches = [index for index, score in enumerate(scores) if score >= 0.55]
        if not matches and gold:
            raw_text = normalize_text(raw[raw_index]["text"])
            exact_text = [
                index
                for index, cell in enumerate(gold)
                if raw_text and normalize_text(cell["text"]) == raw_text
            ]
            if exact_text:
                matches = [max(exact_text, key=lambda index: scores[index])]
            elif scores and max(scores) > 0.0:
                matches = [max(range(len(scores)), key=scores.__getitem__)]
        raw_matches.append(sorted(set(matches)))

    # A Gold primitive with no raw overlap represents a MinerU hole/missing
    # separator.  Attach it to the closest raw logical cell so the dense split
    # field receives an explicit local edit anchor instead of silently dropping
    # the error.  This is topology alignment only; no physical bbox is inferred.
    covered_gold = {index for matches in raw_matches for index in matches}
    for gold_index, gold_cell in enumerate(gold):
        if gold_index in covered_gold or not raw:
            continue
        gold_text = normalize_text(gold_cell["text"])
        exact_text = [
            index
            for index, cell in enumerate(raw)
            if gold_text and normalize_text(cell["text"]) == gold_text
        ]
        if exact_text:
            raw_index = max(
                exact_text, key=lambda index: overlap[index][gold_index]
            )
        else:
            gx = (gold_boxes[gold_index][0] + gold_boxes[gold_index][2]) / 2.0
            gy = (gold_boxes[gold_index][1] + gold_boxes[gold_index][3]) / 2.0
            raw_index = min(
                range(len(raw)),
                key=lambda index: (
                    (
                        (raw_boxes[index][0] + raw_boxes[index][2]) / 2.0
                        - gx
                    )
                    ** 2
                    + (
                        (raw_boxes[index][1] + raw_boxes[index][3]) / 2.0
                        - gy
                    )
                    ** 2,
                    index,
                ),
            )
        raw_matches[raw_index].append(gold_index)
        raw_matches[raw_index] = sorted(set(raw_matches[raw_index]))
        covered_gold.add(gold_index)

    gold_to_raw: dict[int, list[int]] = defaultdict(list)
    for raw_index, matches in enumerate(raw_matches):
        for gold_index in matches:
            gold_to_raw[gold_index].append(raw_index)

    per_raw = [
        {
            "raw_index": index,
            "anchor": f"{cell['row']}:{cell['col']}",
            "keep_structure": True,
            "split_row": False,
            "split_column": False,
            "merge_right": False,
            "merge_down": False,
            "header_update": False,
            "text_update": False,
            "matched_gold_indexes": raw_matches[index],
        }
        for index, cell in enumerate(raw)
    ]
    for raw_index, matches in enumerate(raw_matches):
        if len(matches) > 1:
            row_bands = {
                (gold[index]["row"], gold[index]["row"] + gold[index]["rowspan"])
                for index in matches
            }
            col_bands = {
                (gold[index]["col"], gold[index]["col"] + gold[index]["colspan"])
                for index in matches
            }
            per_raw[raw_index]["split_row"] = len(row_bands) > 1
            per_raw[raw_index]["split_column"] = len(col_bands) > 1
        if not matches:
            per_raw[raw_index]["keep_structure"] = False

    merge_edges: list[dict[str, Any]] = []
    for gold_index, raw_indexes in sorted(gold_to_raw.items()):
        for source, action, target in _merge_tree_edges(raw_indexes, raw):
            per_raw[source][action] = True
            merge_edges.append(
                {
                    "gold_index": gold_index,
                    "source_raw_index": source,
                    "target_raw_index": target,
                    "action": action,
                }
            )

    gold_by_topology = {topology_key(cell): cell for cell in gold}
    for raw_index, cell in enumerate(raw):
        target = gold_by_topology.get(topology_key(cell))
        if target is not None:
            per_raw[raw_index]["header_update"] = cell["tag"] != target["tag"]
            per_raw[raw_index]["text_update"] = (
                normalize_text(cell["text"]) != normalize_text(target["text"])
            )
        structural_action = any(
            per_raw[raw_index][name]
            for name in ("split_row", "split_column", "merge_right", "merge_down")
        )
        if structural_action:
            per_raw[raw_index]["keep_structure"] = False

    raw_topology = {topology_key(cell) for cell in raw}
    gold_topology = {topology_key(cell) for cell in gold}
    topology_edit = raw_topology != gold_topology
    text_edit = any(item["text_update"] for item in per_raw)
    header_edit = any(item["header_update"] for item in per_raw)
    return {
        "policy_version": RESIDUAL_EDIT_POLICY_VERSION,
        "alignment_basis": "normalized_logical_topology_only",
        "physical_bbox_used": False,
        "confidence_used": False,
        "table_action": "EDIT" if topology_edit or text_edit or header_edit else "KEEP",
        "needs_structure_edit": topology_edit,
        "needs_text_edit": text_edit,
        "needs_header_edit": header_edit,
        "raw_cells": per_raw,
        "merge_edges": merge_edges,
        "alignment": {
            "raw_unmatched": [
                index for index, matches in enumerate(raw_matches) if not matches
            ],
            "gold_unmatched": [
                index for index in range(len(gold)) if index not in gold_to_raw
            ],
        },
        "target_program": {
            "rows": structure_labels["rows"],
            "cols": structure_labels["cols"],
            "row_boundaries": structure_labels["row_boundaries"],
            "column_boundaries": structure_labels["column_boundaries"],
            "merge_right": structure_labels["merge_right"],
            "merge_down": structure_labels["merge_down"],
            "header": structure_labels["header"],
            "cell_payload": structure_labels["cell_payload"],
        },
        "header_policy": {
            "version": HEADER_POLICY_VERSION,
            "target_source": "public_gold_only",
            "classes": ["td", "th"],
            "raw_mineru_tag_is_observation_not_target": True,
        },
    }


def _interpolate(values: list[float | None]) -> list[float]:
    known = [index for index, value in enumerate(values) if value is not None]
    if not known:
        return [
            index / max(1, len(values) - 1) for index in range(len(values))
        ]
    output = [0.0] * len(values)
    for index in range(0, known[0] + 1):
        output[index] = float(values[known[0]])
    for left, right in zip(known, known[1:]):
        for index in range(left, right + 1):
            alpha = (index - left) / max(1, right - left)
            output[index] = float(values[left]) * (1 - alpha) + float(
                values[right]
            ) * alpha
    for index in range(known[-1], len(values)):
        output[index] = float(values[known[-1]])
    for index in range(1, len(output)):
        output[index] = max(output[index], output[index - 1] + 1e-5)
    return [min(1.0, max(0.0, value)) for value in output]


def boundary_labels(
    cells: list[dict[str, Any]],
    count: int,
    *,
    axis: str,
    width: float,
    height: float,
) -> dict[str, Any]:
    denominator = height if axis == "row" else width
    start_bbox = 1 if axis == "row" else 0
    end_bbox = 3 if axis == "row" else 2
    index_key = "row" if axis == "row" else "col"
    span_key = "rowspan" if axis == "row" else "colspan"
    candidates: list[list[float]] = [[] for _ in range(count + 1)]
    for cell in cells:
        bbox = cell.get("bbox")
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        start = int(cell[index_key])
        end = start + int(cell.get(span_key, 1) or 1)
        candidates[start].append(float(bbox[start_bbox]) / max(1.0, denominator))
        candidates[end].append(float(bbox[end_bbox]) / max(1.0, denominator))
    direct: list[float | None] = [
        median(values) if values else None for values in candidates
    ]
    reliable = [value is not None for value in direct]
    positions = _interpolate(direct)
    positions[0] = 0.0
    positions[-1] = 1.0
    sources = [
        "physical_bbox" if is_reliable else "logical_interpolated"
        for is_reliable in reliable
    ]
    sources[0] = sources[-1] = "crop_boundary_completion"
    return {
        "positions": positions,
        "physical_reliable": reliable,
        "sources": sources,
        "missing_physical_indexes": [
            index
            for index, is_reliable in enumerate(reliable)
            if not is_reliable and index not in {0, count}
        ],
    }


def _owner_grid(
    cells: list[dict[str, Any]], rows: int, cols: int
) -> list[list[int]]:
    owner = [[-1 for _ in range(cols)] for _ in range(rows)]
    for index, cell in enumerate(cells):
        for row, col in occupied_slots(cell):
            if row >= rows or col >= cols:
                raise ValueError("Gold cell exceeds logical table shape")
            if owner[row][col] >= 0:
                raise ValueError(f"Gold cells overlap at {(row, col)}")
            owner[row][col] = index
    holes = [
        (row, col)
        for row in range(rows)
        for col in range(cols)
        if owner[row][col] < 0
    ]
    if holes:
        raise ValueError(f"Gold grid has {len(holes)} holes")
    return owner


def assign_ocr_tokens(
    ocr_tokens: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    cols: int,
) -> list[int]:
    assignments: list[int] = []
    for token in ocr_tokens:
        bbox = token.get("bbox", [0, 0, 0, 0])
        cx = (float(bbox[0]) + float(bbox[2])) / 2.0
        cy = (float(bbox[1]) + float(bbox[3])) / 2.0
        best_score = 0.0
        best_anchor = -1
        for cell in cells:
            cell_bbox = cell.get("bbox", [0, 0, 0, 0])
            ix = max(
                0.0,
                min(float(bbox[2]), float(cell_bbox[2]))
                - max(float(bbox[0]), float(cell_bbox[0])),
            )
            iy = max(
                0.0,
                min(float(bbox[3]), float(cell_bbox[3]))
                - max(float(bbox[1]), float(cell_bbox[1])),
            )
            inside = (
                float(cell_bbox[0]) <= cx <= float(cell_bbox[2])
                and float(cell_bbox[1]) <= cy <= float(cell_bbox[3])
            )
            score = ix * iy + (1e12 if inside else 0.0)
            if score > best_score:
                best_score = score
                best_anchor = int(cell["row"]) * cols + int(cell["col"])
        assignments.append(best_anchor)
    return assignments


def compile_structure_labels(
    gold_values: list[dict[str, Any]],
    ocr_tokens: list[dict[str, Any]],
    image: dict[str, Any],
) -> dict[str, Any]:
    gold = canonical_cells(gold_values)
    errors = validate_cells(gold, require_complete=True)
    if errors:
        raise ValueError("; ".join(errors))
    rows, cols = table_shape(gold)
    owner = _owner_grid(gold, rows, cols)
    merge_right = [
        [
            int(owner[row][col] == owner[row][col + 1])
            for col in range(cols - 1)
        ]
        for row in range(rows)
    ]
    merge_down = [
        [
            int(owner[row][col] == owner[row + 1][col])
            for col in range(cols)
        ]
        for row in range(rows - 1)
    ]
    header = [
        [
            int(gold[owner[row][col]]["tag"] == "th")
            for col in range(cols)
        ]
        for row in range(rows)
    ]
    row_boundary = boundary_labels(
        gold,
        rows,
        axis="row",
        width=float(image["width"]),
        height=float(image["height"]),
    )
    col_boundary = boundary_labels(
        gold,
        cols,
        axis="col",
        width=float(image["width"]),
        height=float(image["height"]),
    )
    row_separators = [
        {
            "logical_index": index,
            "coordinate": row_boundary["positions"][index],
            "control_points": [
                row_boundary["positions"][index]
            ]
            * CONTROL_POINTS,
            "physical_reliable": row_boundary["physical_reliable"][index],
            "source": row_boundary["sources"][index],
        }
        for index in range(1, rows)
    ]
    col_separators = [
        {
            "logical_index": index,
            "coordinate": col_boundary["positions"][index],
            "control_points": [
                col_boundary["positions"][index]
            ]
            * CONTROL_POINTS,
            "physical_reliable": col_boundary["physical_reliable"][index],
            "source": col_boundary["sources"][index],
        }
        for index in range(1, cols)
    ]
    pointer = assign_ocr_tokens(ocr_tokens, gold, cols)
    payload = {
        f"{cell['row']}:{cell['col']}": {
            "text": cell["text"],
            "tag": cell["tag"],
            "bbox": cell.get("bbox"),
        }
        for cell in gold
    }
    return {
        "rows": rows,
        "cols": cols,
        "row_boundaries": row_boundary["positions"],
        "column_boundaries": col_boundary["positions"],
        "row_separators": row_separators,
        "column_separators": col_separators,
        "primitive_grid": {"rows": rows, "cols": cols},
        "owner": owner,
        "merge_right": merge_right,
        "merge_down": merge_down,
        "header": header,
        "header_policy": {
            "version": HEADER_POLICY_VERSION,
            "target_source": "public_gold_only",
            "classes": ["td", "th"],
        },
        "ocr_gold_pointer": pointer,
        "cell_crops": [
            {
                "row": cell["row"],
                "col": cell["col"],
                "bbox": cell.get("bbox"),
            }
            for cell in gold
        ],
        "cell_payload": payload,
        "missing_physical_row_separators": row_boundary[
            "missing_physical_indexes"
        ],
        "missing_physical_column_separators": col_boundary[
            "missing_physical_indexes"
        ],
    }


def classify_raw_to_gold(
    raw_values: list[dict[str, Any]], gold_values: list[dict[str, Any]]
) -> dict[str, bool]:
    raw = canonical_cells(raw_values)
    gold = canonical_cells(gold_values)
    raw_rows, raw_cols = table_shape(raw)
    gold_rows, gold_cols = table_shape(gold)
    raw_slots = [occupied_slots(cell) for cell in raw]
    gold_slots = [occupied_slots(cell) for cell in gold]
    split = any(
        sum(bool(slots & other) for other in gold_slots) > 1
        for slots in raw_slots
    )
    merge = any(
        sum(bool(slots & other) for other in raw_slots) > 1
        for slots in gold_slots
    )
    raw_strict = {strict_key(cell) for cell in raw}
    gold_strict = {strict_key(cell) for cell in gold}
    gold_tags_by_topology = {
        topology_key(cell): cell["tag"] for cell in gold
    }
    return {
        "identity": raw_strict == gold_strict,
        "missing_row_separator": raw_rows < gold_rows,
        "extra_row_separator": raw_rows > gold_rows,
        "missing_column_separator": raw_cols < gold_cols,
        "extra_column_separator": raw_cols > gold_cols,
        "split_needed": split,
        "merge_needed": merge,
        "mixed_split_merge": split and merge,
        "empty_cells": any(not normalize_text(cell["text"]) for cell in gold),
        "spanning_cells": any(
            cell["rowspan"] > 1 or cell["colspan"] > 1 for cell in gold
        ),
        "large_table": len(gold) > 50,
        "long_table": gold_rows > 32,
        "header_change": any(
            topology_key(cell) in gold_tags_by_topology
            and cell["tag"]
            != gold_tags_by_topology[topology_key(cell)]
            for cell in raw
        ),
    }


def raw_to_gold_diff(
    raw_values: list[dict[str, Any]], gold_values: list[dict[str, Any]]
) -> dict[str, Any]:
    raw = canonical_cells(raw_values, include_geometry=False)
    gold = canonical_cells(gold_values, include_geometry=False)
    raw_top = {topology_key(cell): cell for cell in raw}
    gold_top = {topology_key(cell): cell for cell in gold}
    keep = sum(
        strict_key(raw_top[key]) == strict_key(gold_top[key])
        for key in raw_top.keys() & gold_top.keys()
    )
    update = sum(
        strict_key(raw_top[key]) != strict_key(gold_top[key])
        for key in raw_top.keys() & gold_top.keys()
    )
    categories = classify_raw_to_gold(raw, gold)
    return {
        "raw_cells": len(raw),
        "gold_cells": len(gold),
        "keep_same_topology_and_content": keep,
        "update_same_topology": update,
        "removed_or_restructured_raw_cells": len(raw_top.keys() - gold_top.keys()),
        "added_or_restructured_gold_cells": len(gold_top.keys() - raw_top.keys()),
        "categories": categories,
    }
