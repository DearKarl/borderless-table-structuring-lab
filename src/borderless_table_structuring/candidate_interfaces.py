from __future__ import annotations

import copy
from typing import Any, Iterable

from .safety_layer import (
    ExpectedGainEvidence,
    SafetyPolicy,
    select_candidate_or_rollback,
    stable_sha256,
)


EXPLICIT_POLICY = SafetyPolicy(
    policy_id="explicit-topology-only-2026.08.12",
    minimum_expected_gain=0.0,
    threshold_source="NONTERMINAL_PREREGISTERED",
    text_policy="OCR_GROUNDED",
)

LORA_POLICY = SafetyPolicy(
    policy_id="lora-table-only-2026.08.12",
    minimum_expected_gain=0.0,
    threshold_source="NONTERMINAL_PREREGISTERED",
    text_policy="OCR_GROUNDED",
)


def _canonical_table(record: dict[str, Any]) -> dict[str, Any]:
    table = record.get("canonical_table")
    if not isinstance(table, dict):
        raise ValueError("Canonical Table is missing")
    cells = table.get("cells")
    if not isinstance(cells, list) or not all(isinstance(cell, dict) for cell in cells):
        raise ValueError("Canonical cells are missing")
    return table


def _cell_id(cell: dict[str, Any], index: int) -> str:
    value = str(cell.get("cell_id", f"cell-{index:04d}"))
    if not value:
        raise ValueError("Cell ID is empty")
    return value


def _tokens(cell: dict[str, Any]) -> list[int]:
    values = cell.get("ocr_token_indexes")
    if not isinstance(values, list) or not all(
        isinstance(value, int) and value >= 0 for value in values
    ):
        raise ValueError("OCR token ownership is missing or invalid")
    return values


def _bbox(cell: dict[str, Any]) -> list[float]:
    geometry = cell.get("geometry")
    values = geometry.get("bbox") if isinstance(geometry, dict) else None
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError("Physical geometry is missing")
    return [float(value) for value in values]


def _union_bbox(cells: Iterable[dict[str, Any]]) -> list[float]:
    boxes = [_bbox(cell) for cell in cells]
    if not boxes:
        raise ValueError("A topology partition must reference at least one Raw cell")
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _token_bbox(indexes: list[int], ocr_tokens: list[dict[str, Any]]) -> list[float]:
    boxes: list[list[float]] = []
    for index in indexes:
        if index >= len(ocr_tokens):
            raise ValueError("OCR token index is outside the supplied sidecar")
        value = ocr_tokens[index].get("bbox")
        if not isinstance(value, list) or len(value) != 4:
            raise ValueError("Split token geometry is missing")
        box = [float(item) for item in value]
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Split token geometry is invalid")
        boxes.append(box)
    if not boxes:
        raise ValueError("A split candidate cell must own at least one OCR token")
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _ocr_text(indexes: list[int], ocr_tokens: list[dict[str, Any]]) -> str:
    if any(index >= len(ocr_tokens) for index in indexes):
        raise ValueError("OCR token index is outside the supplied sidecar")
    return "".join(str(ocr_tokens[index].get("text", "")) for index in indexes)


def _route_provenance(
    raw_record: dict[str, Any], *, producer: str, purpose: str
) -> dict[str, Any]:
    raw = raw_record.get("provenance")
    if not isinstance(raw, dict):
        raise ValueError("Raw provenance is missing")
    return {
        "sample_id": str(raw.get("sample_id", "")),
        "producer": producer,
        "producer_release": "2026.08.12",
        "purpose": purpose,
        "input_image_sha256": str(raw.get("input_image_sha256", "")),
        "restricted_evaluation_visible": False,
        "raw_state_sha256": stable_sha256(raw_record),
    }


def build_explicit_topology_candidate(
    raw_record: dict[str, Any],
    *,
    partitions: list[dict[str, Any]] | None,
    ocr_tokens: list[dict[str, Any]],
    rows: int | None = None,
    cols: int | None = None,
) -> dict[str, Any]:
    """Build a reversible topology-only candidate with OCR-copy text.

    ``None`` is the preregistered default ``KEEP`` action and returns an exact
    Raw copy. A changed proposal assigns each Raw OCR token to exactly one
    candidate cell. A Raw cell may therefore be split into several candidate
    cells, and several Raw cells may be merged into one candidate cell. Text is
    always a deterministic projection of frozen OCR tokens.
    """

    if partitions is None:
        return copy.deepcopy(raw_record)
    raw_table = _canonical_table(raw_record)
    raw_cells = raw_table["cells"]
    indexed: dict[str, dict[str, Any]] = {}
    for index, cell in enumerate(raw_cells):
        cell_id = _cell_id(cell, index)
        if cell_id in indexed:
            raise ValueError("Raw cell IDs are not unique")
        indexed[cell_id] = cell

    raw_tokens_by_cell: dict[str, list[int]] = {
        cell_id: _tokens(cell) for cell_id, cell in indexed.items()
    }
    raw_token_owner: dict[int, str] = {}
    for cell_id, token_indexes in raw_tokens_by_cell.items():
        for token_index in token_indexes:
            if token_index >= len(ocr_tokens):
                raise ValueError("Raw OCR token index is outside the supplied sidecar")
            if token_index in raw_token_owner:
                raise ValueError("Raw OCR token ownership is duplicated")
            raw_token_owner[token_index] = cell_id
    if set(raw_token_owner) != set(range(len(ocr_tokens))):
        raise ValueError("Raw OCR token ownership must cover the sidecar exactly")

    used_tokens: list[int] = []
    candidate_cells: list[dict[str, Any]] = []
    replay_partitions: list[dict[str, Any]] = []
    for index, partition in enumerate(partitions):
        source_ids = [str(value) for value in partition.get("source_cell_ids", [])]
        if not source_ids or len(source_ids) != len(set(source_ids)):
            raise ValueError("Each topology partition needs unique source cell IDs")
        try:
            sources = [indexed[source_id] for source_id in source_ids]
        except KeyError as error:
            raise ValueError(f"Unknown Raw source cell ID: {error.args[0]}") from error
        declared_tokens = partition.get("ocr_token_indexes")
        if declared_tokens is None:
            token_indexes = sorted(
                token_index
                for source_id in source_ids
                for token_index in raw_tokens_by_cell[source_id]
            )
        elif not isinstance(declared_tokens, list) or not all(
            isinstance(token_index, int) and token_index >= 0
            for token_index in declared_tokens
        ):
            raise ValueError("Candidate OCR token assignment is invalid")
        else:
            token_indexes = sorted(declared_tokens)
        if not token_indexes or len(token_indexes) != len(set(token_indexes)):
            raise ValueError("A candidate cell needs unique OCR token ownership")
        allowed_tokens = {
            token_index
            for source_id in source_ids
            for token_index in raw_tokens_by_cell[source_id]
        }
        if not set(token_indexes).issubset(allowed_tokens):
            raise ValueError("Candidate OCR token is not owned by its Raw sources")
        if set(source_ids) != {raw_token_owner[token_index] for token_index in token_indexes}:
            raise ValueError("Candidate Raw sources do not exactly bind assigned tokens")
        used_tokens.extend(token_indexes)
        candidate_id = str(partition.get("cell_id", f"explicit-{index:04d}"))
        geometry = (
            _union_bbox(sources)
            if set(token_indexes) == allowed_tokens
            else _token_bbox(token_indexes, ocr_tokens)
        )
        candidate_cells.append(
            {
                "cell_id": candidate_id,
                "row_start": int(partition["row_start"]),
                "row_end": int(partition["row_end"]),
                "col_start": int(partition["col_start"]),
                "col_end": int(partition["col_end"]),
                "text": _ocr_text(token_indexes, ocr_tokens),
                "tag": "th" if all(str(cell.get("tag", "td")) == "th" for cell in sources) else "td",
                "ocr_token_indexes": token_indexes,
                "geometry": {"status": "present", "bbox": geometry},
                "source_raw_cell_ids": source_ids,
            }
        )
        replay_partitions.append(
            {
                "candidate_cell_id": candidate_id,
                "source_raw_cell_ids": source_ids,
                "ocr_token_indexes": token_indexes,
            }
        )

    if len(used_tokens) != len(set(used_tokens)):
        raise ValueError("Candidate OCR token ownership is duplicated")
    if set(used_tokens) != set(raw_token_owner):
        raise ValueError("Candidate OCR token ownership must cover Raw exactly")
    candidate_rows = int(rows if rows is not None else raw_table["rows"])
    candidate_cols = int(cols if cols is not None else raw_table["cols"])
    return {
        "schema_release": "explicit-topology-candidate-2026.08.13.1",
        "canonical_table": {
            "rows": candidate_rows,
            "cols": candidate_cols,
            "cells": candidate_cells,
        },
        "candidate_interface": {
            "route": "EXPLICIT_LAYOUT_TRANSFORMER",
            "scope": "TABLE_ONLY",
            "default_action": "KEEP",
            "text_policy": "FROZEN_TOKEN_PROJECTION",
            "full_page_rewrite": False,
        },
        "replay": {
            "schema_release": "explicit-reversible-replay-2026.08.13.1",
            "raw_state_sha256": stable_sha256(raw_record),
            "partitions": replay_partitions,
        },
        "provenance": _route_provenance(
            raw_record,
            producer="explicit-layout-transformer-interface",
            purpose="nonterminal_explicit_topology_candidate",
        ),
    }


def replay_explicit_to_raw(
    raw_record: dict[str, Any], candidate_record: dict[str, Any]
) -> dict[str, Any]:
    replay = candidate_record.get("replay")
    if not isinstance(replay, dict):
        raise ValueError("Explicit replay metadata is missing")
    if replay.get("raw_state_sha256") != stable_sha256(raw_record):
        raise ValueError("Explicit replay Raw binding does not match")
    return copy.deepcopy(raw_record)


def build_lora_table_candidate(
    raw_record: dict[str, Any], *, candidate_table: dict[str, Any]
) -> dict[str, Any]:
    """Wrap one complete table-only LoRA candidate behind the shared safety layer."""

    if "blocks" in candidate_table or "page" in candidate_table:
        raise ValueError("LoRA full-page output is forbidden")
    table = copy.deepcopy(candidate_table.get("canonical_table", candidate_table))
    if not isinstance(table, dict) or not isinstance(table.get("cells"), list):
        raise ValueError("LoRA output must contain one complete Canonical Table")
    return {
        "schema_release": "lora-canonical-table-candidate-2026.08.12",
        "canonical_table": table,
        "candidate_interface": {
            "route": "LORA_TABLE_MODEL",
            "scope": "TABLE_ONLY",
            "output": "COMPLETE_CANONICAL_TABLE",
            "text_policy": "OCR_COPY_PREFERRED_AND_VALIDATED",
            "geometry_policy": "PARALLEL_OR_SIDECAR_REQUIRED",
            "full_page_rewrite": False,
        },
        "provenance": _route_provenance(
            raw_record,
            producer="lora-table-model-interface",
            purpose="nonterminal_lora_canonical_table_candidate",
        ),
    }


def select_explicit_candidate(
    raw_record: dict[str, Any],
    candidate_record: dict[str, Any] | None,
    *,
    expected_gain: ExpectedGainEvidence | None,
    ocr_tokens: list[dict[str, Any]],
) -> dict[str, Any]:
    return select_candidate_or_rollback(
        raw_record,
        candidate_record,
        policy=EXPLICIT_POLICY,
        expected_gain=expected_gain,
        ocr_tokens=ocr_tokens,
    )


def select_lora_candidate(
    raw_record: dict[str, Any],
    candidate_record: dict[str, Any] | None,
    *,
    expected_gain: ExpectedGainEvidence | None,
    ocr_tokens: list[dict[str, Any]],
) -> dict[str, Any]:
    return select_candidate_or_rollback(
        raw_record,
        candidate_record,
        policy=LORA_POLICY,
        expected_gain=expected_gain,
        ocr_tokens=ocr_tokens,
    )
