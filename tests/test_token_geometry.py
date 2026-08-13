from __future__ import annotations

from copy import deepcopy

from mpr_tsr_splitmerge_v2.counterfactual import build_candidate
from mpr_tsr_splitmerge_v2.token_geometry import (
    ownership_violations,
    reassign_tokens_by_geometry,
    topology_signature,
)


def _record() -> dict:
    tokens = [
        {"text": "A", "bbox": [0.0, 0.0, 10.0, 10.0]},
        {"text": "B", "bbox": [10.0, 0.0, 20.0, 10.0]},
    ]
    return {
        "canonical_table": {
            "rows": 1,
            "cols": 2,
            "cells": [
                {
                    "cell_id": "c0",
                    "row": 0,
                    "col": 0,
                    "rowspan": 1,
                    "colspan": 1,
                    "text": "A",
                    "tag": "td",
                    "bbox": [0.0, 0.0, 10.0, 10.0],
                    "geometry": {"bbox": [0.0, 0.0, 10.0, 10.0]},
                    "ocr_token_indexes": [0],
                },
                {
                    "cell_id": "c1",
                    "row": 0,
                    "col": 1,
                    "rowspan": 1,
                    "colspan": 1,
                    "text": "B",
                    "tag": "td",
                    "bbox": [10.0, 0.0, 20.0, 10.0],
                    "geometry": {"bbox": [10.0, 0.0, 20.0, 10.0]},
                    "ocr_token_indexes": [1],
                },
            ],
        },
        "ocr_tokens": tokens,
        "non_table_state_sha256": "same",
        "provenance": {},
    }


def test_assignment_swap_preserves_coarse_structure_and_text_multiset() -> None:
    gold = _record()
    swapped = build_candidate(gold, gold, operator="assignment_swap")
    assert swapped is not None
    assert topology_signature(swapped) == topology_signature(gold)
    assert len(swapped["canonical_table"]["cells"]) == len(gold["canonical_table"]["cells"])
    assert sorted(cell["text"] for cell in swapped["canonical_table"]["cells"]) == ["A", "B"]
    assert swapped["canonical_table"] != gold["canonical_table"]


def test_geometry_reassignment_preserves_good_and_repairs_swapped_assignment() -> None:
    gold = _record()
    tokens = gold["ocr_tokens"]
    identity = reassign_tokens_by_geometry(gold, tokens)
    assert identity is not None
    assert identity["canonical_table"] == gold["canonical_table"]
    swapped = build_candidate(gold, gold, operator="assignment_swap")
    assert swapped is not None
    assert ownership_violations(swapped, tokens) == 2
    repaired = reassign_tokens_by_geometry(swapped, tokens)
    assert repaired is not None
    assert ownership_violations(repaired, tokens) == 0
    assert repaired["canonical_table"] == gold["canonical_table"]
