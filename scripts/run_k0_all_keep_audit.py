from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from mpr_tsr_splitmerge_v2.safety_layer import (
    SafetyPolicy,
    select_candidate_or_rollback,
)


def _load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(records: list[dict[str, Any]]) -> dict[str, Any]:
    policy = SafetyPolicy(
        policy_id="k0-all-keep-v1",
        minimum_expected_gain=0.0,
        threshold_source="NONTERMINAL_PREREGISTERED",
        text_policy="OCR_GROUNDED",
    )
    decision_counts: collections.Counter[str] = collections.Counter()
    reason_counts: collections.Counter[str] = collections.Counter()
    failures = []

    for record in records:
        raw = record["raw_record"]
        candidate = record.get("candidate_record")
        ocr_tokens = raw.get("ocr_tokens")
        selection = select_candidate_or_rollback(
            raw,
            candidate,
            policy=policy,
            expected_gain=None,
            ocr_tokens=ocr_tokens,
        )
        decision_counts[selection["decision"]] += 1
        reason_counts.update(selection["reason_codes"])
        checks = {
            "not_accept": selection["decision"] != "ACCEPT",
            "rollback_exact": selection["rollback_exact"] is True,
            "state_hash_equal": (
                selection["output_state_sha256"] == selection["raw_state_sha256"]
            ),
            "non_table_state_equal": (
                selection["output"].get("non_table_state_sha256")
                == raw.get("non_table_state_sha256")
            ),
            "output_equal_raw": selection["output"] == raw,
        }
        if not all(checks.values()):
            failures.append({
                "sample_id": record.get("sample_id"),
                "decision": selection["decision"],
                "reason_codes": selection["reason_codes"],
                "checks": checks,
            })

    return {
        "schema_version": "mpr-tsr/k0-all-keep-audit-v1",
        "records": len(records),
        "decision_counts": dict(decision_counts),
        "reason_counts": dict(reason_counts),
        "exact_raw_outputs": len(records) - len(failures),
        "failure_count": len(failures),
        "failures": failures,
        "status": "PASS" if not failures and decision_counts.get("ACCEPT", 0) == 0 else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the K0 all-KEEP exact Raw bypass audit.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(_load(args.input))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "records": report["records"],
        "exact_raw_outputs": report["exact_raw_outputs"],
        "decision_counts": report["decision_counts"],
        "failure_count": report["failure_count"],
    }, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
