from __future__ import annotations

from copy import deepcopy

from borderless_table_structuring.safety_layer import (
    ExpectedGainEvidence,
    SafetyPolicy,
    assemble_table_only,
    select_candidate_or_rollback,
    stable_sha256,
    validate_candidate,
)


def _record(cells=None):
    cells = cells or [
        {
            "cell_id": "c0",
            "row_start": 0,
            "row_end": 1,
            "col_start": 0,
            "col_end": 1,
            "text": "A",
            "ocr_token_indexes": [0],
            "geometry": {"status": "present", "bbox": [0, 0, 10, 10]},
        },
        {
            "cell_id": "c1",
            "row_start": 0,
            "row_end": 1,
            "col_start": 1,
            "col_end": 2,
            "text": "B",
            "ocr_token_indexes": [1],
            "geometry": {"status": "present", "bbox": [10, 0, 20, 10]},
        },
    ]
    return {
        "canonical_table": {"rows": 1, "cols": 2, "cells": cells},
        "provenance": {
            "sample_id": "dev-fixture-001",
            "producer": "unit-fixture",
            "producer_release": "2026.08.12",
            "purpose": "nonterminal_correctness_fixture",
            "input_image_sha256": "0" * 64,
            "restricted_evaluation_visible": False,
        },
    }


POLICY = SafetyPolicy(
    policy_id="synthetic-preregistered-2026.08.12",
    minimum_expected_gain=0.0,
    threshold_source="NONTERMINAL_PREREGISTERED",
    text_policy="OCR_GROUNDED",
)
OCR = [{"text": "A"}, {"text": "B"}]


def test_identity_is_exact_pass_through_without_gain_evidence():
    raw = _record()
    result = select_candidate_or_rollback(raw, deepcopy(raw), policy=POLICY, ocr_tokens=OCR)
    assert result["decision"] == "PASS_THROUGH"
    assert result["rollback_exact"] is True
    assert result["output_state_sha256"] == stable_sha256(raw)


def test_overlap_fails_and_rolls_back_exactly():
    raw = _record()
    candidate = deepcopy(raw)
    candidate["canonical_table"]["cells"][1]["col_start"] = 0
    result = select_candidate_or_rollback(
        raw,
        candidate,
        policy=POLICY,
        expected_gain=ExpectedGainEvidence(1.0, "development-protocol-2026.08.12", "NONTERMINAL_DEVELOPMENT"),
        ocr_tokens=OCR,
    )
    assert result["decision"] == "ROLLBACK"
    assert result["rollback_exact"] is True
    assert "CANONICAL_GRID_OVERLAP" in result["reason_codes"]


def test_token_loss_and_text_change_fail_closed():
    raw = _record()
    candidate = deepcopy(raw)
    candidate["canonical_table"]["cells"][1]["ocr_token_indexes"] = []
    candidate["canonical_table"]["cells"][1]["text"] = "hallucination"
    validation = validate_candidate(raw, candidate, policy=POLICY, ocr_tokens=OCR)
    assert validation["status"] == "FAIL"
    assert "TOKEN_COVERAGE_CHANGED" in validation["issues"]
    assert "CANDIDATE_TEXT_NOT_OCR_GROUNDED" in validation["issues"]


def test_valid_changed_candidate_needs_positive_nonterminal_gain():
    raw = _record()
    candidate = deepcopy(raw)
    candidate["canonical_table"]["cells"] = [
        {
            "cell_id": "merged",
            "row_start": 0,
            "row_end": 1,
            "col_start": 0,
            "col_end": 2,
            "text": "AB",
            "ocr_token_indexes": [0, 1],
            "geometry": {"status": "present", "bbox": [0, 0, 20, 10]},
        }
    ]
    missing = select_candidate_or_rollback(raw, candidate, policy=POLICY, ocr_tokens=OCR)
    insufficient = select_candidate_or_rollback(
        raw,
        candidate,
        policy=POLICY,
        expected_gain=ExpectedGainEvidence(0.0, "development-protocol-2026.08.12", "NONTERMINAL_DEVELOPMENT"),
        ocr_tokens=OCR,
    )
    accepted = select_candidate_or_rollback(
        raw,
        candidate,
        policy=POLICY,
        expected_gain=ExpectedGainEvidence(0.1, "development-protocol-2026.08.12", "NONTERMINAL_DEVELOPMENT"),
        ocr_tokens=OCR,
    )
    assert missing["decision"] == "ROLLBACK"
    assert insufficient["decision"] == "ROLLBACK"
    assert accepted["decision"] == "ACCEPT"
    assert accepted["rollback_exact"] is False


def test_missing_geometry_and_bad_provenance_fail():
    raw = _record()
    candidate = deepcopy(raw)
    candidate["canonical_table"]["cells"][0].pop("geometry")
    candidate["provenance"]["restricted_evaluation_visible"] = True
    validation = validate_candidate(raw, candidate, policy=POLICY, ocr_tokens=OCR)
    assert validation["status"] == "FAIL"
    assert "GEOMETRY_MISSING_OR_INVALID" in validation["issues"]
    assert "PROVENANCE_RESTRICTED_EVALUATION_VISIBILITY_NOT_FALSE" in validation["issues"]


def test_table_only_assembler_freezes_every_other_page_block():
    raw = _record()
    candidate = deepcopy(raw)
    candidate["canonical_table"]["cells"] = [
        {
            "cell_id": "merged",
            "row_start": 0,
            "row_end": 1,
            "col_start": 0,
            "col_end": 2,
            "text": "AB",
            "ocr_token_indexes": [0, 1],
            "geometry": {"status": "present", "bbox": [0, 0, 20, 10]},
        }
    ]
    selection = select_candidate_or_rollback(
        raw,
        candidate,
        policy=POLICY,
        expected_gain=ExpectedGainEvidence(0.1, "development-protocol-2026.08.12", "NONTERMINAL_DEVELOPMENT"),
        ocr_tokens=OCR,
    )
    page = {
        "blocks": [
            {"type": "paragraph", "text": "frozen title"},
            {"type": "table", "table_record": raw},
            {"type": "formula", "latex": "x+y"},
        ]
    }
    assembled = assemble_table_only(page, table_block_index=1, selection=selection)
    assert assembled["status"] == "PASS"
    assert assembled["non_table_state_frozen"] is True
    assert assembled["output_page"]["blocks"][0] == page["blocks"][0]
    assert assembled["output_page"]["blocks"][2] == page["blocks"][2]
    assert assembled["output_page"]["blocks"][1]["table_record"] == candidate


def test_table_only_assembler_rejects_full_page_candidate():
    raw = _record()
    forbidden = deepcopy(raw)
    forbidden["blocks"] = [{"type": "paragraph", "text": "rewrite"}]
    selection = {"decision": "ACCEPT", "output": forbidden}
    page = {"blocks": [{"type": "table", "table_record": raw}]}
    try:
        assemble_table_only(page, table_block_index=0, selection=selection)
    except ValueError as error:
        assert "Full-page candidate" in str(error)
    else:
        raise AssertionError("Full-page candidate was not rejected")
