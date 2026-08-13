from __future__ import annotations

from copy import deepcopy

from borderless_table_structuring.explicit import (
    build_explicit_topology_candidate,
    replay_explicit_to_raw,
)


def _raw_record() -> dict[str, object]:
    return {
        "canonical_table": {
            "rows": 1,
            "cols": 2,
            "cells": [
                {
                    "cell_id": "left",
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
                    "cell_id": "right",
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
            "sample_id": "synthetic-explicit-fixture",
            "input_image_sha256": "0" * 64,
        },
    }


def test_public_explicit_interface_preserves_identity() -> None:
    raw = _raw_record()
    candidate = build_explicit_topology_candidate(
        raw,
        partitions=None,
        ocr_tokens=[
            {"text": "A", "bbox": [0, 0, 10, 10]},
            {"text": "B", "bbox": [10, 0, 20, 10]},
        ],
    )
    assert candidate == raw
    assert candidate is not raw


def test_public_explicit_interface_is_reversible() -> None:
    raw = _raw_record()
    raw_snapshot = deepcopy(raw)
    candidate = build_explicit_topology_candidate(
        raw,
        partitions=[
            {
                "cell_id": "merged",
                "source_cell_ids": ["left", "right"],
                "row_start": 0,
                "row_end": 1,
                "col_start": 0,
                "col_end": 2,
            }
        ],
        ocr_tokens=[
            {"text": "A", "bbox": [0, 0, 10, 10]},
            {"text": "B", "bbox": [10, 0, 20, 10]},
        ],
    )
    assert candidate["canonical_table"]["cells"][0]["text"] == "AB"
    assert replay_explicit_to_raw(raw, candidate) == raw_snapshot
