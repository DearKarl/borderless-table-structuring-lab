from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compile_canonical_data_layer_vnext",
    ROOT / "scripts" / "compile_canonical_data_layer_vnext.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_explicit_target_is_order_invariant_and_defaults_to_keep() -> None:
    raw = [
        {"row": 0, "col": 1, "rowspan": 1, "colspan": 1},
        {"row": 0, "col": 0, "rowspan": 1, "colspan": 1},
    ]
    gold = [
        {"row": 0, "col": 0, "rowspan": 1, "colspan": 2},
    ]
    first = MODULE.explicit_target(raw, gold)
    second = MODULE.explicit_target(list(reversed(raw)), list(reversed(gold)))
    assert first == second
    assert first["default_action"] == "KEEP"
    assert len(first["remove"]) == 2
    assert len(first["add"]) == 1


def test_ownership_rejects_out_of_range_gold_index() -> None:
    record = {
        "ocr_tokens": [{"token_index": 0}],
        "text_edit_labels": {
            "cells": [{"gold_index": 3, "ocr_token_indexes": [0]}]
        },
    }
    try:
        MODULE.ownership(record, 1)
    except ValueError as error:
        assert "outside range" in str(error)
    else:
        raise AssertionError("out-of-range pointer was accepted")


def test_canonical_state_hash_is_independent_of_input_cell_order() -> None:
    record = {
        "ocr_tokens": [],
        "text_edit_labels": {
            "cells": [
                {"gold_index": 0, "ocr_token_indexes": []},
                {"gold_index": 1, "ocr_token_indexes": []},
            ]
        },
    }
    cells = [
        {"row": 0, "col": 0, "rowspan": 1, "colspan": 1, "text": "A", "tag": "th"},
        {"row": 0, "col": 1, "rowspan": 1, "colspan": 1, "text": "B", "tag": "td"},
    ]
    first = MODULE.canonical_state(record, MODULE.canonical_cells(cells))
    second = MODULE.canonical_state(record, MODULE.canonical_cells(reversed(cells)))
    assert first["semantic_state_sha256"] == second["semantic_state_sha256"]
    assert first["full_state_sha256"] == second["full_state_sha256"]
