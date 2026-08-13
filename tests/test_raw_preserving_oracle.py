from __future__ import annotations

from copy import deepcopy

from mpr_tsr_splitmerge_v2.raw_preserving import (
    EDIT,
    KEEP,
    label_oracle_action,
    selector_runtime_view,
)


def _record(values):
    cells = []
    for index, (row, col, text) in enumerate(values):
        cells.append({
            "cell_id": f"c{index}",
            "row": row,
            "col": col,
            "rowspan": 1,
            "colspan": 1,
            "text": text,
            "tag": "td",
            "bbox": [col * 10, row * 10, col * 10 + 9, row * 10 + 9],
            "geometry": {"bbox": [col * 10, row * 10, col * 10 + 9, row * 10 + 9]},
            "ocr_token_indexes": [index],
        })
    return {
        "canonical_table": {"rows": 1, "cols": len(cells), "cells": cells},
        "non_table_state_sha256": "same-page-state",
        "provenance": {
            "sample_id": "fixture",
            "producer": "fixture",
            "producer_version": "v1",
            "purpose": "nonterminal_fixture",
            "input_image_sha256": "0" * 64,
            "terminal_benchmarks_visible": False,
        },
    }


def test_raw_good_forces_keep_even_with_a_valid_candidate():
    gold = _record([(0, 0, "A"), (0, 1, "B")])
    candidate = deepcopy(gold)
    decision = label_oracle_action(gold, candidate, gold)
    assert decision.action == KEEP
    assert decision.reason == "RAW_ALREADY_CORRECT"
    assert decision.raw_good is True


def test_strictly_better_candidate_is_accept_edit():
    raw = _record([(0, 0, "A"), (0, 1, "WRONG")])
    gold = _record([(0, 0, "A"), (0, 1, "B")])
    candidate = deepcopy(gold)
    decision = label_oracle_action(raw, candidate, gold, ocr_tokens=[{"text": "A"}, {"text": "B"}])
    assert decision.action == EDIT
    assert decision.reason == "STRICTLY_BETTER_CANDIDATE"
    assert decision.delta_exact_cells == 1


def test_legal_but_tied_candidate_stays_keep():
    raw = _record([(0, 0, "A"), (0, 1, "WRONG")])
    gold = _record([(0, 0, "A"), (0, 1, "B")])
    candidate = deepcopy(raw)
    decision = label_oracle_action(raw, candidate, gold)
    assert decision.action == KEEP
    assert decision.reason == "NO_STRICT_GAIN"


def test_runtime_view_excludes_gold_and_gold_derived_fields():
    raw = _record([(0, 0, "A")])
    candidate = deepcopy(raw)
    view = selector_runtime_view(
        sample_id="s1",
        image={"path": "sample.jpg"},
        raw_record=raw,
        candidate_record=candidate,
    )
    serialized = str(view)
    assert "gold" not in serialized.lower()
    assert "raw_score" not in serialized
    assert "candidate_score" not in serialized
    assert "decision_target" not in serialized
