from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

from .canonical import normalize_text


Decision = Literal["PASS_THROUGH", "ACCEPT", "ROLLBACK"]
TextPolicy = Literal["FROZEN_RAW", "OCR_GROUNDED"]


@dataclass(frozen=True)
class SafetyPolicy:
    """Frozen policy for table-only candidate validation and selection.

    ``minimum_expected_gain`` is deliberately supplied by the caller.  This
    module does not estimate or calibrate it, which prevents a terminal
    benchmark from silently becoming an acceptance-threshold source.
    """

    policy_id: str
    minimum_expected_gain: float
    threshold_source: Literal["NONTERMINAL_PREREGISTERED"]
    text_policy: TextPolicy = "FROZEN_RAW"
    require_complete_grid: bool = True
    require_geometry: bool = True
    require_provenance: bool = True


@dataclass(frozen=True)
class ExpectedGainEvidence:
    value: float
    protocol_id: str
    source: Literal["NONTERMINAL_DEVELOPMENT"]


def stable_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _interval(cell: dict[str, Any]) -> tuple[int, int, int, int] | None:
    try:
        if "row_start" in cell:
            row_start = int(cell["row_start"])
            row_end = int(cell["row_end"])
            col_start = int(cell["col_start"])
            col_end = int(cell["col_end"])
        else:
            row_start = int(cell["row"])
            row_end = row_start + int(cell.get("rowspan", 1))
            col_start = int(cell["col"])
            col_end = col_start + int(cell.get("colspan", 1))
    except (KeyError, TypeError, ValueError):
        return None
    return row_start, row_end, col_start, col_end


def _bbox(cell: dict[str, Any]) -> list[float] | None:
    geometry = cell.get("geometry")
    value = geometry.get("bbox") if isinstance(geometry, dict) else cell.get("bbox")
    if not isinstance(value, list) or len(value) != 4:
        return None
    if not all(
        isinstance(item, (int, float)) and math.isfinite(float(item))
        for item in value
    ):
        return None
    box = [float(item) for item in value]
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _tokens(cell: dict[str, Any]) -> list[int] | None:
    value = cell.get("ocr_token_indexes")
    if value is None:
        return None
    if not isinstance(value, list) or not all(
        isinstance(item, int) and item >= 0 for item in value
    ):
        return None
    return value


def _table_cells(record: dict[str, Any]) -> list[dict[str, Any]] | None:
    table = record.get("canonical_table", record)
    cells = table.get("cells") if isinstance(table, dict) else None
    if not isinstance(cells, list) or not all(isinstance(cell, dict) for cell in cells):
        return None
    return cells


def _table_shape(record: dict[str, Any], cells: list[dict[str, Any]]) -> tuple[int, int] | None:
    table = record.get("canonical_table", record)
    try:
        declared_rows = int(table.get("rows", -1))
        declared_cols = int(table.get("cols", -1))
    except (TypeError, ValueError):
        return None
    if declared_rows <= 0 or declared_cols <= 0:
        return None
    maximum_row = maximum_col = 0
    for cell in cells:
        interval = _interval(cell)
        if interval is None:
            return None
        maximum_row = max(maximum_row, interval[1])
        maximum_col = max(maximum_col, interval[3])
    if (maximum_row, maximum_col) != (declared_rows, declared_cols):
        return None
    return declared_rows, declared_cols


def _provenance_issues(record: dict[str, Any]) -> list[str]:
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        return ["PROVENANCE_MISSING"]
    issues: list[str] = []
    for field in ("sample_id", "producer", "producer_release", "purpose", "input_image_sha256"):
        if not str(provenance.get(field, "")).strip():
            issues.append(f"PROVENANCE_{field.upper()}_MISSING")
    if provenance.get("restricted_evaluation_visible") is not False:
        issues.append("PROVENANCE_RESTRICTED_EVALUATION_VISIBILITY_NOT_FALSE")
    terminal_words = ("private-evaluation", "restricted-evaluation")
    serialized = json.dumps(provenance, ensure_ascii=False).lower()
    if any(word in serialized for word in terminal_words):
        issues.append("PROVENANCE_RESTRICTED_EVALUATION_REFERENCE_FORBIDDEN")
    return issues


def _text_token_issues(
    raw_record: dict[str, Any],
    candidate_record: dict[str, Any],
    *,
    policy: SafetyPolicy,
    ocr_tokens: list[dict[str, Any]] | None,
) -> list[str]:
    raw_cells = _table_cells(raw_record) or []
    candidate_cells = _table_cells(candidate_record) or []
    issues: list[str] = []
    raw_token_lists = [_tokens(cell) for cell in raw_cells]
    candidate_token_lists = [_tokens(cell) for cell in candidate_cells]
    token_contract_active = any(value is not None for value in raw_token_lists + candidate_token_lists)

    if token_contract_active:
        if any(value is None for value in raw_token_lists + candidate_token_lists):
            issues.append("TOKEN_OWNERSHIP_PARTIAL")
            return issues
        raw_flat = [item for values in raw_token_lists for item in values or []]
        candidate_flat = [item for values in candidate_token_lists for item in values or []]
        if len(raw_flat) != len(set(raw_flat)):
            issues.append("RAW_TOKEN_OWNERSHIP_DUPLICATED")
        if len(candidate_flat) != len(set(candidate_flat)):
            issues.append("CANDIDATE_TOKEN_OWNERSHIP_DUPLICATED")
        if sorted(raw_flat) != sorted(candidate_flat):
            issues.append("TOKEN_COVERAGE_CHANGED")
        if ocr_tokens is not None:
            for cell in candidate_cells:
                indexes = _tokens(cell) or []
                if any(index >= len(ocr_tokens) for index in indexes):
                    issues.append("TOKEN_INDEX_OUT_OF_RANGE")
                    continue
                expected = normalize_text("".join(str(ocr_tokens[index].get("text", "")) for index in indexes))
                observed = normalize_text(cell.get("text", ""))
                if expected != observed:
                    issues.append("CANDIDATE_TEXT_NOT_OCR_GROUNDED")
    elif policy.text_policy == "OCR_GROUNDED":
        issues.append("TOKEN_OWNERSHIP_REQUIRED")

    if policy.text_policy == "FROZEN_RAW":
        raw_text = Counter(normalize_text(cell.get("text", "")) for cell in raw_cells)
        candidate_text = Counter(normalize_text(cell.get("text", "")) for cell in candidate_cells)
        if raw_text != candidate_text:
            issues.append("RAW_TEXT_MULTISET_CHANGED")
    return issues


def validate_candidate(
    raw_record: dict[str, Any],
    candidate_record: dict[str, Any],
    *,
    policy: SafetyPolicy,
    ocr_tokens: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate a table-only candidate without reading any performance metric."""

    cells = _table_cells(candidate_record)
    issues: list[str] = []
    if cells is None or not cells:
        issues.append("CANONICAL_CELLS_MISSING")
        cells = []
    shape = _table_shape(candidate_record, cells) if cells else None
    if shape is None:
        issues.append("CANONICAL_SHAPE_INVALID")

    occupied: dict[tuple[int, int], int] = {}
    cell_ids: set[str] = set()
    for index, cell in enumerate(cells):
        cell_id = str(cell.get("cell_id", index))
        if cell_id in cell_ids:
            issues.append("CELL_ID_DUPLICATED")
        cell_ids.add(cell_id)
        interval = _interval(cell)
        if interval is None:
            issues.append("CELL_INTERVAL_INVALID")
            continue
        row_start, row_end, col_start, col_end = interval
        if row_start < 0 or col_start < 0 or row_end <= row_start or col_end <= col_start:
            issues.append("CELL_INTERVAL_INVALID")
            continue
        if shape is not None and (row_end > shape[0] or col_end > shape[1]):
            issues.append("CELL_INTERVAL_OUT_OF_BOUNDS")
        for row in range(row_start, row_end):
            for col in range(col_start, col_end):
                slot = (row, col)
                if slot in occupied:
                    issues.append("CANONICAL_GRID_OVERLAP")
                occupied[slot] = index
        if policy.require_geometry and _bbox(cell) is None:
            issues.append("GEOMETRY_MISSING_OR_INVALID")

    if policy.require_complete_grid and shape is not None:
        expected_slots = shape[0] * shape[1]
        if len(occupied) != expected_slots:
            issues.append("CANONICAL_GRID_INCOMPLETE")
    if policy.require_provenance:
        issues.extend(_provenance_issues(candidate_record))
    issues.extend(
        _text_token_issues(
            raw_record,
            candidate_record,
            policy=policy,
            ocr_tokens=ocr_tokens,
        )
    )
    unique_issues = sorted(set(issues))
    return {
        "schema_release": "shared-candidate-validation-2026.08.12",
        "status": "PASS" if not unique_issues else "FAIL",
        "issues": unique_issues,
        "cell_count": len(cells),
        "occupied_slot_count": len(occupied),
        "candidate_state_sha256": stable_sha256(candidate_record),
        "policy_id": policy.policy_id,
    }


def select_candidate_or_rollback(
    raw_record: dict[str, Any],
    candidate_record: dict[str, Any] | None,
    *,
    policy: SafetyPolicy,
    expected_gain: ExpectedGainEvidence | None = None,
    ocr_tokens: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Select a candidate or return a byte-equivalent deep copy of Raw.

    Identity candidates pass through Raw before the expected-gain gate.  Every
    failure path uses the same rollback output, so missing, invalid, or
    insufficient-gain candidates cannot partially mutate the baseline.
    """

    raw_hash = stable_sha256(raw_record)
    if candidate_record is None:
        output = copy.deepcopy(raw_record)
        return _selection("ROLLBACK", output, raw_hash, ["CANDIDATE_MISSING"])
    candidate_hash = stable_sha256(candidate_record)
    if candidate_hash == raw_hash:
        output = copy.deepcopy(raw_record)
        return _selection("PASS_THROUGH", output, raw_hash, ["IDENTICAL_STATE"])

    validation = validate_candidate(
        raw_record,
        candidate_record,
        policy=policy,
        ocr_tokens=ocr_tokens,
    )
    if validation["status"] != "PASS":
        output = copy.deepcopy(raw_record)
        return _selection(
            "ROLLBACK",
            output,
            raw_hash,
            ["VALIDATION_FAILED", *validation["issues"]],
            validation=validation,
        )
    if expected_gain is None:
        output = copy.deepcopy(raw_record)
        return _selection("ROLLBACK", output, raw_hash, ["EXPECTED_GAIN_MISSING"], validation=validation)
    if (
        expected_gain.source != "NONTERMINAL_DEVELOPMENT"
        or not expected_gain.protocol_id.strip()
        or not math.isfinite(expected_gain.value)
    ):
        output = copy.deepcopy(raw_record)
        return _selection("ROLLBACK", output, raw_hash, ["EXPECTED_GAIN_EVIDENCE_INVALID"], validation=validation)
    if expected_gain.value <= policy.minimum_expected_gain:
        output = copy.deepcopy(raw_record)
        return _selection("ROLLBACK", output, raw_hash, ["EXPECTED_GAIN_INSUFFICIENT"], validation=validation)
    output = copy.deepcopy(candidate_record)
    return _selection("ACCEPT", output, raw_hash, ["VALID_AND_POSITIVE_EXPECTED_GAIN"], validation=validation)


def assemble_table_only(
    page_record: dict[str, Any],
    *,
    table_block_index: int,
    selection: dict[str, Any],
) -> dict[str, Any]:
    """Assemble one selected table while freezing every non-table page block.

    The page contract stores the table candidate under ``table_record``.  A
    candidate containing page-level ``blocks`` is rejected, so neither route
    can use this function to rewrite titles, paragraphs, formulas, figures, or
    reading order.
    """

    blocks = page_record.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError("Page blocks are missing")
    if table_block_index < 0 or table_block_index >= len(blocks):
        raise IndexError("Table block index is out of range")
    source_block = blocks[table_block_index]
    if not isinstance(source_block, dict) or source_block.get("type") != "table":
        raise ValueError("Selected page block is not a table")
    output_record = selection.get("output")
    if not isinstance(output_record, dict) or "canonical_table" not in output_record:
        raise ValueError("Selection output is not a Canonical Table record")
    if "blocks" in output_record:
        raise ValueError("Full-page candidate output is forbidden")

    before_non_table = [
        block for index, block in enumerate(blocks) if index != table_block_index
    ]
    output_page = copy.deepcopy(page_record)
    output_page["blocks"][table_block_index]["table_record"] = copy.deepcopy(output_record)
    after_non_table = [
        block
        for index, block in enumerate(output_page["blocks"])
        if index != table_block_index
    ]
    frozen_non_table = stable_sha256(before_non_table) == stable_sha256(after_non_table)
    if not frozen_non_table:
        raise RuntimeError("Non-table page state changed during table assembly")
    return {
        "schema_release": "table-only-assembly-2026.08.12",
        "status": "PASS",
        "decision": selection.get("decision"),
        "output_page": output_page,
        "non_table_state_frozen": frozen_non_table,
        "non_table_state_sha256": stable_sha256(after_non_table),
        "output_page_sha256": stable_sha256(output_page),
    }


def _selection(
    decision: Decision,
    output: dict[str, Any],
    raw_hash: str,
    reasons: list[str],
    *,
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_hash = stable_sha256(output)
    return {
        "schema_release": "shared-candidate-selection-2026.08.12",
        "decision": decision,
        "reason_codes": sorted(set(reasons)),
        "output": output,
        "raw_state_sha256": raw_hash,
        "output_state_sha256": output_hash,
        "rollback_exact": decision not in {"ACCEPT"} and output_hash == raw_hash,
        "validation": validation,
    }
