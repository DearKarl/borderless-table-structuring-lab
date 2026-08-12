from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import Any


def normalize_text(value: Any) -> str:
    value = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", "", value).strip()


def canonical_cell(
    value: dict[str, Any], *, include_geometry: bool = True
) -> dict[str, Any]:
    cell = {
        "row": int(value.get("row", 0)),
        "col": int(value.get("col", 0)),
        "rowspan": max(1, int(value.get("rowspan", 1) or 1)),
        "colspan": max(1, int(value.get("colspan", 1) or 1)),
        "text": str(value.get("text", "") or "").strip(),
        "tag": "th" if str(value.get("tag", "td")).lower() == "th" else "td",
    }
    if include_geometry and isinstance(value.get("bbox"), list):
        cell["bbox"] = [round(float(item), 6) for item in value["bbox"][:4]]
    # Missing confidence means "the producer did not expose confidence".  Do
    # not silently turn that into a perfect score: downstream residual gates
    # must be able to distinguish absent evidence from model confidence.
    if include_geometry and value.get("confidence") is not None:
        cell["confidence"] = round(float(value["confidence"]), 6)
    return cell


def canonical_cells(
    values: Iterable[dict[str, Any]], *, include_geometry: bool = True
) -> list[dict[str, Any]]:
    return sorted(
        (
            canonical_cell(value, include_geometry=include_geometry)
            for value in values
        ),
        key=lambda cell: (
            cell["row"],
            cell["col"],
            cell["rowspan"],
            cell["colspan"],
            cell["text"],
            cell["tag"],
        ),
    )


def topology_key(value: dict[str, Any]) -> tuple[int, int, int, int]:
    cell = canonical_cell(value, include_geometry=False)
    return (
        cell["row"],
        cell["col"],
        cell["rowspan"],
        cell["colspan"],
    )


def strict_key(value: dict[str, Any]) -> tuple[Any, ...]:
    cell = canonical_cell(value, include_geometry=False)
    return (*topology_key(cell), normalize_text(cell["text"]), cell["tag"])


def table_shape(values: Iterable[dict[str, Any]]) -> tuple[int, int]:
    cells = list(values)
    return (
        max(
            (
                int(cell["row"]) + int(cell.get("rowspan", 1) or 1)
                for cell in cells
            ),
            default=0,
        ),
        max(
            (
                int(cell["col"]) + int(cell.get("colspan", 1) or 1)
                for cell in cells
            ),
            default=0,
        ),
    )


def occupied_slots(value: dict[str, Any]) -> set[tuple[int, int]]:
    row, col, rowspan, colspan = topology_key(value)
    return {
        (r, c)
        for r in range(row, row + rowspan)
        for c in range(col, col + colspan)
    }


def validate_cells(
    values: Iterable[dict[str, Any]], *, require_complete: bool = True
) -> list[str]:
    cells = canonical_cells(values, include_geometry=False)
    errors: list[str] = []
    occupied: dict[tuple[int, int], int] = {}
    for index, cell in enumerate(cells):
        if cell["row"] < 0 or cell["col"] < 0:
            errors.append(f"cell[{index}] has negative index")
        for slot in occupied_slots(cell):
            if slot in occupied:
                errors.append(
                    f"cell[{index}] overlaps cell[{occupied[slot]}] at {slot}"
                )
            occupied[slot] = index
    if require_complete and cells:
        rows, cols = table_shape(cells)
        missing = [
            (row, col)
            for row in range(rows)
            for col in range(cols)
            if (row, col) not in occupied
        ]
        if missing:
            errors.append(f"grid has {len(missing)} holes")
    return errors
