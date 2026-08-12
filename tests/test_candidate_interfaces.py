from __future__ import annotations

from copy import deepcopy

import pytest

from borderless_table_structuring.candidate_interfaces import (
    build_explicit_topology_candidate,
    build_lora_table_candidate,
    replay_explicit_to_raw,
    select_explicit_candidate,
    select_lora_candidate,
)
from borderless_table_structuring.safety_layer import ExpectedGainEvidence, stable_sha256


OCR = [{"text": "A"}, {"text": "B"}]
GAIN = ExpectedGainEvidence(0.1, "interface-fixture-2026.08.12", "NONTERMINAL_DEVELOPMENT")


def _raw_record():
    return {
        "canonical_table": {
            "rows": 1,
            "cols": 2,
            "cells": [
                {
                    "cell_id": "raw-a",
                    "row_start": 0,
                    "row_end": 1,
                    "col_start": 0,
                    "col_end": 1,
                    "text": "A",
                    "tag": "td",
                    "ocr_token_indexes": [0],
                    "geometry": {"status": "present", "bbox": [0, 0, 10, 10]},
                },
                {
                    "cell_id": "raw-b",
                    "row_start": 0,
                    "row_end": 1,
                    "col_start": 1,
                    "col_end": 2,
                    "text": "B",
                    "tag": "td",
                    "ocr_token_indexes": [1],
                    "geometry": {"status": "present", "bbox": [10, 0, 20, 10]},
                },
            ],
        },
        "provenance": {
            "sample_id": "isolated-dev-fixture-001",
            "producer": "raw-mineru-fixture",
            "producer_release": "2026.08.12",
            "purpose": "synthetic_interface_fixture",
            "input_image_sha256": "0" * 64,
            "restricted_evaluation_visible": False,
        },
    }


def test_explicit_default_keep_is_exact_raw_pass_through():
    raw = _raw_record()
    candidate = build_explicit_topology_candidate(
        raw, partitions=None, ocr_tokens=OCR
    )
    result = select_explicit_candidate(
        raw, candidate, expected_gain=None, ocr_tokens=OCR
    )
    assert stable_sha256(candidate) == stable_sha256(raw)
    assert result["decision"] == "PASS_THROUGH"
    assert result["rollback_exact"] is True


def test_explicit_changed_topology_preserves_tokens_and_is_reversible():
    raw = _raw_record()
    candidate = build_explicit_topology_candidate(
        raw,
        partitions=[
            {
                "cell_id": "merged",
                "row_start": 0,
                "row_end": 1,
                "col_start": 0,
                "col_end": 2,
                "source_cell_ids": ["raw-a", "raw-b"],
            }
        ],
        ocr_tokens=OCR,
    )
    result = select_explicit_candidate(
        raw, candidate, expected_gain=GAIN, ocr_tokens=OCR
    )
    cell = candidate["canonical_table"]["cells"][0]
    assert result["decision"] == "ACCEPT"
    assert cell["text"] == "AB"
    assert cell["ocr_token_indexes"] == [0, 1]
    assert cell["geometry"]["bbox"] == [0.0, 0.0, 20.0, 10.0]
    replayed = replay_explicit_to_raw(raw, candidate)
    assert stable_sha256(replayed) == stable_sha256(raw)


def test_explicit_rejects_duplicate_or_incomplete_source_coverage():
    raw = _raw_record()
    with pytest.raises(ValueError, match="cover every Raw cell"):
        build_explicit_topology_candidate(
            raw,
            partitions=[
                {
                    "row_start": 0,
                    "row_end": 1,
                    "col_start": 0,
                    "col_end": 1,
                    "source_cell_ids": ["raw-a"],
                }
            ],
            ocr_tokens=OCR,
        )


def test_lora_complete_table_candidate_accepts_only_ocr_grounded_text():
    raw = _raw_record()
    table = {
        "rows": 1,
        "cols": 2,
        "cells": deepcopy(raw["canonical_table"]["cells"]),
    }
    table["cells"][0]["cell_id"] = "lora-a"
    candidate = build_lora_table_candidate(raw, candidate_table=table)
    result = select_lora_candidate(
        raw, candidate, expected_gain=GAIN, ocr_tokens=OCR
    )
    assert result["decision"] == "ACCEPT"
    assert candidate["candidate_interface"]["scope"] == "TABLE_ONLY"
    assert "blocks" not in candidate


def test_lora_hallucinated_text_rolls_back_exactly():
    raw = _raw_record()
    table = deepcopy(raw["canonical_table"])
    table["cells"][0]["cell_id"] = "lora-a"
    table["cells"][0]["text"] = "HALLUCINATED"
    candidate = build_lora_table_candidate(raw, candidate_table=table)
    result = select_lora_candidate(
        raw, candidate, expected_gain=GAIN, ocr_tokens=OCR
    )
    assert result["decision"] == "ROLLBACK"
    assert result["rollback_exact"] is True
    assert "CANDIDATE_TEXT_NOT_OCR_GROUNDED" in result["reason_codes"]


def test_lora_full_page_output_is_rejected_at_interface_boundary():
    with pytest.raises(ValueError, match="full-page output is forbidden"):
        build_lora_table_candidate(
            _raw_record(),
            candidate_table={"blocks": [{"type": "paragraph", "text": "rewrite"}]},
        )
