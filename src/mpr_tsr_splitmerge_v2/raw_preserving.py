from __future__ import annotations

import copy
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from .canonical import normalize_text
from .safety_layer import SafetyPolicy, validate_candidate


KEEP = "KEEP_RAW"
EDIT = "ACCEPT_EDIT"

REASONS = {
    "RAW_ALREADY_CORRECT",
    "CANDIDATE_MISSING",
    "CANDIDATE_INVALID",
    "NO_STRICT_GAIN",
    "TEXT_REGRESSION",
    "GEOMETRY_REGRESSION",
    "NON_TABLE_STATE_CHANGED",
    "STRICTLY_BETTER_CANDIDATE",
}


@dataclass(frozen=True)
class OracleDecision:
    action: str
    reason: str
    raw_good: bool
    candidate_valid: bool
    gold_cell_denominator: int
    raw_exact_cells: int
    candidate_exact_cells: int
    raw_text_exact: int
    candidate_text_exact: int
    raw_geometry_coverage: int
    candidate_geometry_coverage: int
    delta_exact_cells: int
    delta_pp: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cells(record: dict[str, Any]) -> list[dict[str, Any]]:
    table = record.get("canonical_table", record)
    if not isinstance(table, dict) or not isinstance(table.get("cells"), list):
        return []
    output = []
    for value in table["cells"]:
        if not isinstance(value, dict):
            continue
        if "row_start" in value:
            row = int(value.get("row_start", 0))
            col = int(value.get("col_start", 0))
            rowspan = int(value.get("row_end", row + 1)) - row
            colspan = int(value.get("col_end", col + 1)) - col
        else:
            row = int(value.get("row", 0))
            col = int(value.get("col", 0))
            rowspan = int(value.get("rowspan", 1) or 1)
            colspan = int(value.get("colspan", 1) or 1)
        output.append({
            "row": row,
            "col": col,
            "rowspan": rowspan,
            "colspan": colspan,
            "text": str(value.get("text", "") or ""),
            "tag": "th" if str(value.get("tag", "td")).lower() == "th" else "td",
            "bbox": value.get("bbox", value.get("geometry", {}).get("bbox") if isinstance(value.get("geometry"), dict) else None),
        })
    return output


def _strict_key(cell: dict[str, Any]) -> tuple[Any, ...]:
    return (
        cell["row"], cell["col"], cell["rowspan"], cell["colspan"],
        normalize_text(cell["text"]), cell["tag"],
    )


def _cell_exact_count(
    predicted: list[dict[str, Any]], gold: list[dict[str, Any]]
) -> int:
    remaining = Counter(_strict_key(cell) for cell in gold)
    correct = 0
    for cell in predicted:
        key = _strict_key(cell)
        if remaining[key] > 0:
            remaining[key] -= 1
            correct += 1
    return correct


def _text_exact_count(
    predicted: list[dict[str, Any]], gold: list[dict[str, Any]]
) -> int:
    remaining = Counter(normalize_text(cell.get("text", "")) for cell in gold)
    correct = 0
    for cell in predicted:
        text = normalize_text(cell.get("text", ""))
        if remaining[text] > 0:
            remaining[text] -= 1
            correct += 1
    return correct


def _geometry_coverage(cells: list[dict[str, Any]]) -> int:
    count = 0
    for cell in cells:
        geometry = cell.get("geometry")
        bbox = geometry.get("bbox") if isinstance(geometry, dict) else cell.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            try:
                if float(bbox[2]) > float(bbox[0]) and float(bbox[3]) > float(bbox[1]):
                    count += 1
            except (TypeError, ValueError):
                pass
    return count


def _non_table_state_changed(
    raw_record: dict[str, Any], candidate_record: dict[str, Any]
) -> bool:
    raw_state = raw_record.get("non_table_state_sha256")
    if raw_state is None:
        return False
    return candidate_record.get("non_table_state_sha256") != raw_state


def _validate(
    raw_record: dict[str, Any],
    candidate_record: dict[str, Any] | None,
    *,
    ocr_tokens: list[dict[str, Any]] | None,
    policy: SafetyPolicy,
) -> bool:
    if candidate_record is None:
        return False
    return validate_candidate(
        raw_record,
        candidate_record,
        policy=policy,
        ocr_tokens=ocr_tokens,
    )["status"] == "PASS"


def _decision(
    *,
    action: str,
    reason: str,
    raw_good: bool,
    candidate_valid: bool,
    denominator: int,
    raw_cells: list[dict[str, Any]],
    candidate_cells: list[dict[str, Any]],
    gold_cells: list[dict[str, Any]],
) -> OracleDecision:
    raw_exact = _cell_exact_count(raw_cells, gold_cells)
    candidate_exact = _cell_exact_count(candidate_cells, gold_cells)
    raw_text = _text_exact_count(raw_cells, gold_cells)
    candidate_text = _text_exact_count(candidate_cells, gold_cells)
    delta = candidate_exact - raw_exact
    return OracleDecision(
        action=action,
        reason=reason,
        raw_good=raw_good,
        candidate_valid=candidate_valid,
        gold_cell_denominator=denominator,
        raw_exact_cells=raw_exact,
        candidate_exact_cells=candidate_exact,
        raw_text_exact=raw_text,
        candidate_text_exact=candidate_text,
        raw_geometry_coverage=_geometry_coverage(raw_cells),
        candidate_geometry_coverage=_geometry_coverage(candidate_cells),
        delta_exact_cells=delta,
        delta_pp=100.0 * delta / denominator if denominator else 0.0,
    )


def label_oracle_action(
    raw_record: dict[str, Any],
    candidate_record: dict[str, Any] | None,
    gold_record: dict[str, Any],
    *,
    ocr_tokens: list[dict[str, Any]] | None = None,
    policy: SafetyPolicy | None = None,
) -> OracleDecision:
    """Create an offline KEEP/EDIT label from Raw, candidate, and Gold.

    Gold is required only for this offline function. The runtime selector view
    deliberately excludes Gold and every Gold-derived metric.
    """
    raw_cells = _cells(raw_record)
    gold_cells = _cells(gold_record)
    candidate_cells = _cells(candidate_record or {})
    denominator = len(gold_cells)
    raw_exact = _cell_exact_count(raw_cells, gold_cells)
    raw_good = bool(gold_cells) and len(raw_cells) == denominator and raw_exact == denominator
    validation_policy = policy or SafetyPolicy(
        policy_id="offline-oracle-v1",
        minimum_expected_gain=0.0,
        threshold_source="NONTERMINAL_PREREGISTERED",
        text_policy="OCR_GROUNDED" if ocr_tokens is not None else "FROZEN_RAW",
    )
    candidate_valid = _validate(
        raw_record,
        candidate_record,
        ocr_tokens=ocr_tokens,
        policy=validation_policy,
    )
    if raw_good:
        return _decision(
            action=KEEP,
            reason="RAW_ALREADY_CORRECT",
            raw_good=True,
            candidate_valid=candidate_valid,
            denominator=denominator,
            raw_cells=raw_cells,
            candidate_cells=candidate_cells,
            gold_cells=gold_cells,
        )

    if candidate_record is None:
        return _decision(
            action=KEEP,
            reason="CANDIDATE_MISSING",
            raw_good=False,
            candidate_valid=False,
            denominator=denominator,
            raw_cells=raw_cells,
            candidate_cells=[],
            gold_cells=gold_cells,
        )

    validation = validate_candidate(
        raw_record,
        candidate_record,
        policy=validation_policy,
        ocr_tokens=ocr_tokens,
    )
    if validation["status"] != "PASS":
        return _decision(
            action=KEEP,
            reason="CANDIDATE_INVALID",
            raw_good=False,
            candidate_valid=False,
            denominator=denominator,
            raw_cells=raw_cells,
            candidate_cells=candidate_cells,
            gold_cells=gold_cells,
        )

    raw_text = _text_exact_count(raw_cells, gold_cells)
    candidate_text = _text_exact_count(candidate_cells, gold_cells)
    raw_geometry = _geometry_coverage(raw_cells)
    candidate_geometry = _geometry_coverage(candidate_cells)
    raw_exact = _cell_exact_count(raw_cells, gold_cells)
    candidate_exact = _cell_exact_count(candidate_cells, gold_cells)
    if _non_table_state_changed(raw_record, candidate_record):
        reason = "NON_TABLE_STATE_CHANGED"
    elif candidate_text < raw_text:
        reason = "TEXT_REGRESSION"
    elif candidate_geometry < raw_geometry:
        reason = "GEOMETRY_REGRESSION"
    elif candidate_exact <= raw_exact:
        reason = "NO_STRICT_GAIN"
    else:
        reason = "STRICTLY_BETTER_CANDIDATE"
    return _decision(
        action=EDIT if reason == "STRICTLY_BETTER_CANDIDATE" else KEEP,
        reason=reason,
        raw_good=False,
        candidate_valid=True,
        denominator=denominator,
        raw_cells=raw_cells,
        candidate_cells=candidate_cells,
        gold_cells=gold_cells,
    )


def _runtime_cell(value: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "cell_id", "row", "col", "rowspan", "colspan", "row_start",
        "row_end", "col_start", "col_end", "text", "tag",
        "ocr_token_indexes", "geometry", "bbox",
    )
    return {key: copy.deepcopy(value[key]) for key in allowed if key in value}


def _runtime_record(record: dict[str, Any]) -> dict[str, Any]:
    table = record.get("canonical_table", record)
    if not isinstance(table, dict):
        return {"canonical_table": {"cells": []}}
    output = {"canonical_table": {
        key: copy.deepcopy(table[key])
        for key in ("rows", "cols")
        if key in table
    }}
    output["canonical_table"]["cells"] = [
        _runtime_cell(value)
        for value in table.get("cells", [])
        if isinstance(value, dict)
    ]
    return output


def _runtime_image(image: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(image[key])
        for key in ("path", "width", "height", "sha256", "image_sha256")
        if key in image
    }


def selector_runtime_view(
    *,
    sample_id: str,
    image: dict[str, Any],
    raw_record: dict[str, Any],
    candidate_record: dict[str, Any],
) -> dict[str, Any]:
    """Build the inference-time whitelist; Gold-derived fields cannot enter."""
    raw_cells = _cells(raw_record)
    candidate_cells = _cells(candidate_record)
    return {
        "schema_version": "mpr-tsr/raw-preserving-selector-view-v1",
        "sample_id": sample_id,
        "image": _runtime_image(image),
        "raw_record": _runtime_record(raw_record),
        "candidate_record": _runtime_record(candidate_record),
        "candidate_diff": {
            "raw_cell_count": len(raw_cells),
            "candidate_cell_count": len(candidate_cells),
            "cell_count_delta": len(candidate_cells) - len(raw_cells),
            "raw_rows": raw_record.get("canonical_table", {}).get("rows"),
            "candidate_rows": candidate_record.get("canonical_table", {}).get("rows"),
            "raw_cols": raw_record.get("canonical_table", {}).get("cols"),
            "candidate_cols": candidate_record.get("canonical_table", {}).get("cols"),
        },
    }
