from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


NUMERIC_FEATURES = (
    "cell_count_delta",
    "row_delta",
    "col_delta",
)
CATEGORICAL_FEATURES = (
    "identity_candidate",
    "text_multiset_equal",
    "structure_equal",
)


def _load(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _cells(record: dict[str, Any]) -> list[dict[str, Any]]:
    return list(record.get("canonical_table", {}).get("cells", []))


def _structure(cell: dict[str, Any]) -> tuple[Any, ...]:
    return (
        cell.get("row", cell.get("row_start")),
        cell.get("col", cell.get("col_start")),
        cell.get("rowspan", cell.get("row_end")),
        cell.get("colspan", cell.get("col_end")),
        cell.get("tag", "td"),
    )


def _features(record: dict[str, Any]) -> dict[str, Any]:
    raw = record["raw_record"]
    candidate = record["candidate_record"]
    raw_table = raw["canonical_table"]
    candidate_table = candidate["canonical_table"]
    raw_cells = _cells(raw)
    candidate_cells = _cells(candidate)
    return {
        "cell_count_delta": len(candidate_cells) - len(raw_cells),
        "row_delta": int(candidate_table.get("rows", 0)) - int(raw_table.get("rows", 0)),
        "col_delta": int(candidate_table.get("cols", 0)) - int(raw_table.get("cols", 0)),
        "identity_candidate": raw_table == candidate_table,
        "text_multiset_equal": sorted(str(cell.get("text", "")) for cell in raw_cells) == sorted(str(cell.get("text", "")) for cell in candidate_cells),
        "structure_equal": sorted(_structure(cell) for cell in raw_cells) == sorted(_structure(cell) for cell in candidate_cells),
    }


def _metrics(labels: list[bool], predictions: list[bool]) -> dict[str, float | int]:
    tp = sum(label and prediction for label, prediction in zip(labels, predictions))
    fp = sum(not label and prediction for label, prediction in zip(labels, predictions))
    tn = sum(not label and not prediction for label, prediction in zip(labels, predictions))
    fn = sum(label and not prediction for label, prediction in zip(labels, predictions))
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / len(labels) if labels else 0.0,
        "balanced_accuracy": (tpr + tnr) / 2,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tpr,
    }


def audit(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    labels = []
    values = []
    for record in records:
        labels.append(record["oracle_decision"]["action"] == "ACCEPT_EDIT")
        values.append(_features(record))
    rules = []

    for feature in NUMERIC_FEATURES:
        candidates = sorted({float(value[feature]) for value in values})
        thresholds = sorted(set(candidates + [(left + right) / 2 for left, right in zip(candidates, candidates[1:])]))
        for threshold in thresholds:
            for direction in ("greater", "less"):
                if direction == "greater":
                    predictions = [float(value[feature]) > threshold for value in values]
                else:
                    predictions = [float(value[feature]) < threshold for value in values]
                rules.append({
                    "feature": feature,
                    "rule": f"{direction}_than_{threshold:g}",
                    **_metrics(labels, predictions),
                })

    for feature in CATEGORICAL_FEATURES:
        for positive_value in (True, False):
            predictions = [bool(value[feature]) is positive_value for value in values]
            rules.append({
                "feature": feature,
                "rule": f"equals_{str(positive_value).lower()}",
                **_metrics(labels, predictions),
            })

    rules.sort(key=lambda value: (float(value["balanced_accuracy"]), float(value["precision"]), float(value["recall"])), reverse=True)
    all_keep = _metrics(labels, [False] * len(labels))
    best = rules[0] if rules else None
    blocked = bool(best and best["balanced_accuracy"] >= 0.90)
    contingency = {
        feature: {
            str(feature_value): dict(collections.Counter(
                "ACCEPT_EDIT" if label else "KEEP_RAW"
                for value, label in zip(values, labels)
                if value[feature] == feature_value
            ))
            for feature_value in sorted({value[feature] for value in values}, key=str)
        }
        for feature in (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
    }
    return {
        "schema_version": "mpr-tsr/selector-shortcut-audit-v1",
        "records": len(labels),
        "keep_records": sum(not label for label in labels),
        "accept_records": sum(labels),
        "all_keep": all_keep,
        "best_single_feature_rule": best,
        "top_rules": rules[:10],
        "feature_label_contingency": contingency,
        "block_threshold_balanced_accuracy": 0.90,
        "status": "BLOCKED_SHORTCUT" if blocked else "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit synthetic KEEP/EDIT pairs for observable shortcut labels.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(_load(args.input))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "records": report["records"],
        "best_single_feature_rule": report["best_single_feature_rule"],
        "all_keep": report["all_keep"],
    }, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
