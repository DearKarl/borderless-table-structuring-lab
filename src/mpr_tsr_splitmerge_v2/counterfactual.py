from __future__ import annotations

import copy
from typing import Any


OPERATORS = (
    "identity",
    "over_merge",
    "over_split",
    "assignment_swap",
    "gold_candidate",
)


def _table(record: dict[str, Any]) -> dict[str, Any]:
    table = record.get("canonical_table", record)
    if not isinstance(table, dict) or not isinstance(table.get("cells"), list):
        raise ValueError("record has no canonical table")
    return table


def _new_record(raw: dict[str, Any], cells: list[dict[str, Any]], suffix: str) -> dict[str, Any]:
    candidate = copy.deepcopy(raw)
    table = _table(candidate)
    table["cells"] = cells
    provenance = candidate.setdefault("provenance", {})
    provenance["producer"] = "raw-preserving-counterfactual-bank"
    provenance["producer_version"] = "v1"
    provenance["purpose"] = suffix
    return candidate


def _cell_bounds(cell: dict[str, Any]) -> tuple[int, int, int, int]:
    if "row_start" in cell:
        row = int(cell["row_start"])
        row_end = int(cell["row_end"])
        col = int(cell["col_start"])
        col_end = int(cell["col_end"])
    else:
        row = int(cell.get("row", 0))
        row_end = row + int(cell.get("rowspan", 1) or 1)
        col = int(cell.get("col", 0))
        col_end = col + int(cell.get("colspan", 1) or 1)
    return row, row_end, col, col_end


def _set_bounds(cell: dict[str, Any], bounds: tuple[int, int, int, int]) -> None:
    row, row_end, col, col_end = bounds
    if "row_start" in cell:
        cell.update(row_start=row, row_end=row_end, col_start=col, col_end=col_end)
    else:
        cell.update(row=row, col=col, rowspan=row_end - row, colspan=col_end - col)


def _merge_pair(cells: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    for left_index, left in enumerate(cells):
        lr, lre, lc, lce = _cell_bounds(left)
        if lre - lr != 1 or lce - lc != 1:
            continue
        for right_index in range(left_index + 1, len(cells)):
            right = cells[right_index]
            rr, rre, rc, rce = _cell_bounds(right)
            if (rr, rre, rc) != (lr, lre, lce):
                continue
            merged = copy.deepcopy(left)
            _set_bounds(merged, (lr, lre, lc, rce))
            merged["text"] = f"{left.get('text', '')}{right.get('text', '')}"
            if "ocr_token_indexes" in left or "ocr_token_indexes" in right:
                merged["ocr_token_indexes"] = sorted(
                    list(left.get("ocr_token_indexes", []))
                    + list(right.get("ocr_token_indexes", []))
                )
            left_box = left.get("bbox")
            right_box = right.get("bbox")
            if isinstance(left_box, list) and isinstance(right_box, list):
                merged["bbox"] = [left_box[0], min(left_box[1], right_box[1]), right_box[2], max(left_box[3], right_box[3])]
                if isinstance(merged.get("geometry"), dict):
                    merged["geometry"]["bbox"] = list(merged["bbox"])
            return [cell for index, cell in enumerate(cells) if index not in {left_index, right_index}] + [merged]
    return None


def _split_cell(cells: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    for index, cell in enumerate(cells):
        row, row_end, col, col_end = _cell_bounds(cell)
        tokens = list(cell.get("ocr_token_indexes", []))
        if row_end - row != 1 or col_end - col != 1 or len(tokens) < 2:
            continue
        midpoint = len(tokens) // 2
        first = copy.deepcopy(cell)
        second = copy.deepcopy(cell)
        first["cell_id"] = f"{cell.get('cell_id', index)}-left"
        second["cell_id"] = f"{cell.get('cell_id', index)}-right"
        _set_bounds(first, (row, row_end, col, col + 1))
        _set_bounds(second, (row, row_end, col + 1, col + 2))
        first["text"] = str(cell.get("text", ""))[: max(1, len(str(cell.get("text", ""))) // 2)]
        second["text"] = str(cell.get("text", ""))[max(1, len(str(cell.get("text", ""))) // 2):]
        first["ocr_token_indexes"] = tokens[:midpoint]
        second["ocr_token_indexes"] = tokens[midpoint:]
        return cells[:index] + [first, second] + cells[index + 1:]
    return None


def _swap_assignment(cells: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    candidates = [
        index
        for index, cell in enumerate(cells)
        if str(cell.get("text", "")) and cell.get("ocr_token_indexes")
    ]
    if len(candidates) < 2:
        return None
    left_index, right_index = candidates[0], candidates[1]
    output = copy.deepcopy(cells)
    left = output[left_index]
    right = output[right_index]
    left["text"], right["text"] = right["text"], left["text"]
    left["ocr_token_indexes"], right["ocr_token_indexes"] = (
        list(right["ocr_token_indexes"]),
        list(left["ocr_token_indexes"]),
    )
    return output


def build_candidate(
    raw_record: dict[str, Any],
    gold_record: dict[str, Any],
    *,
    operator: str,
) -> dict[str, Any] | None:
    """Build a deterministic nonterminal candidate for smoke and QA only."""
    if operator not in OPERATORS:
        raise ValueError(f"unknown operator: {operator}")
    if operator == "identity":
        return copy.deepcopy(raw_record)
    if operator == "gold_candidate":
        candidate = copy.deepcopy(gold_record)
        candidate.setdefault("provenance", {})["purpose"] = "nonterminal_oracle_candidate_smoke_only"
        return candidate
    cells = copy.deepcopy(_table(raw_record)["cells"])
    if operator == "over_merge":
        changed = _merge_pair(cells)
    elif operator == "over_split":
        changed = _split_cell(cells)
    else:
        changed = _swap_assignment(cells)
    if changed is None:
        return None
    candidate = _new_record(raw_record, changed, f"nonterminal_{operator}_smoke_only")
    if operator == "over_split":
        candidate["canonical_table"]["cols"] = int(candidate["canonical_table"].get("cols", 0)) + 1
    return candidate
