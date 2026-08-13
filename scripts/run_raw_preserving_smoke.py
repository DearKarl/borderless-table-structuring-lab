from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from mpr_tsr_splitmerge_v2.counterfactual import build_candidate
from mpr_tsr_splitmerge_v2.raw_preserving import KEEP, label_oracle_action


def record(cells: list[tuple[int, int, str, list[int]]], *, cols: int) -> dict:
    return {
        "canonical_table": {
            "rows": 1,
            "cols": cols,
            "cells": [
                {
                    "cell_id": f"c{index}",
                    "row": row,
                    "col": col,
                    "rowspan": 1,
                    "colspan": 1,
                    "text": text,
                    "tag": "td",
                    "bbox": [col * 10, 0, col * 10 + 9, 9],
                    "geometry": {"bbox": [col * 10, 0, col * 10 + 9, 9]},
                    "ocr_token_indexes": token_indexes,
                }
                for index, (row, col, text, token_indexes) in enumerate(cells)
            ],
        },
        "non_table_state_sha256": "smoke-page-state",
        "provenance": {
            "sample_id": "raw-preserving-smoke",
            "producer": "raw-preserving-smoke",
            "producer_version": "v1",
            "purpose": "nonterminal_logic_smoke",
            "input_image_sha256": "0" * 64,
            "terminal_benchmarks_visible": False,
        },
    }


def build_cases(group_index: int) -> list[dict]:
    token_text = [{"text": "A"}, {"text": "B"}]
    raw_good = record([(0, 0, "A", [0]), (0, 1, "B", [1])], cols=2)
    gold_good = deepcopy(raw_good)
    raw_bad = record([(0, 0, "A", [0]), (0, 1, "WRONG", [1])], cols=2)
    gold_bad = record([(0, 0, "A", [0]), (0, 1, "B", [1])], cols=2)
    raw_merged = record([(0, 0, "AB", [0, 1])], cols=1)
    gold_split = record([(0, 0, "A", [0]), (0, 1, "B", [1])], cols=2)
    cases = []
    specifications = [
        (raw_good, gold_good, "identity", "raw_good_identity"),
        (raw_good, gold_good, "over_merge", "raw_good_over_merge"),
        (raw_good, gold_good, "over_merge", "raw_good_over_merge_2"),
        (raw_bad, gold_bad, "identity", "raw_bad_identity"),
        (raw_bad, gold_bad, "over_merge", "raw_bad_over_merge"),
        (raw_bad, gold_bad, "over_merge", "raw_bad_over_merge_2"),
        (raw_bad, gold_bad, "identity", "raw_bad_tie"),
        (raw_merged, gold_split, "identity", "raw_bad_merged_identity"),
        (raw_merged, gold_split, "over_split", "raw_bad_over_split"),
        (raw_merged, gold_split, "over_merge", "raw_bad_over_merge_noop"),
    ]
    for local_index, (raw, gold, operator, tag) in enumerate(specifications):
        candidate = build_candidate(raw, gold, operator=operator)
        decision = label_oracle_action(raw, candidate, gold, ocr_tokens=token_text)
        cases.append({
            "sample_id": f"smoke-{group_index:04d}-{local_index:02d}",
            "pair_group_id": f"group-{group_index:04d}",
            "image": {"path": f"synthetic://raw-preserving-smoke/{group_index:04d}.json"},
            "operator": operator,
            "phenomenon_tags": [tag],
            "oracle_decision": decision.as_dict(),
        })
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the non-image Raw-preserving oracle smoke.")
    parser.add_argument("--groups", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.groups <= 0:
        raise ValueError("groups must be positive")
    cases = [case for group in range(args.groups) for case in build_cases(group)]
    counts = {}
    for case in cases:
        action = case["oracle_decision"]["action"]
        counts[action] = counts.get(action, 0) + 1
    expected_total = args.groups * 10
    if len(cases) != expected_total:
        raise AssertionError(f"expected {expected_total} cases, got {len(cases)}")
    if counts.get(KEEP, 0) != args.groups * 9:
        raise AssertionError(f"expected 90% KEEP, got {counts}")
    if counts.get("ACCEPT_EDIT", 0) != args.groups:
        raise AssertionError(f"expected 10% ACCEPT_EDIT, got {counts}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({
            "schema_version": "mpr-tsr/raw-preserving-logic-smoke-v1",
            "image_payload_included": False,
            "groups": args.groups,
            "records": len(cases),
            "action_counts": counts,
            "records_detail": cases,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"groups": args.groups, "records": len(cases), "action_counts": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
