from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator

from mpr_tsr_splitmerge_v2.safety_layer import (
    ExpectedGainEvidence,
    SafetyPolicy,
    select_candidate_or_rollback,
    stable_sha256,
    validate_candidate,
)
from mpr_tsr_splitmerge_v2.token_geometry import ownership_violations, topology_signature


def _load(path: Path, role: str) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if record["role"] == role:
                    yield record


def _upper_zero_failures(trials: int, alpha: float = 0.05) -> float:
    return 1.0 - alpha ** (1.0 / trials) if trials else 1.0


def _lower_all_successes(successes: int, alpha: float = 0.05) -> float:
    return alpha ** (1.0 / successes) if successes else 0.0


def _token_indexes(record: dict[str, Any]) -> list[int] | None:
    table = record.get("canonical_table", record)
    values = []
    for cell in table.get("cells", []):
        indexes = cell.get("ocr_token_indexes")
        if not isinstance(indexes, list):
            return None
        values.extend(indexes)
    return sorted(values)


def decide(
    raw: dict[str, Any],
    candidate: dict[str, Any],
    ocr_tokens: list[dict[str, Any]],
    policy: SafetyPolicy,
) -> tuple[bool, dict[str, Any]]:
    raw_violations = ownership_violations(raw, ocr_tokens)
    candidate_violations = ownership_violations(candidate, ocr_tokens)
    topology_equal = topology_signature(raw) == topology_signature(candidate)
    token_conserved = _token_indexes(raw) == _token_indexes(candidate)
    non_table_equal = raw.get("non_table_state_sha256") == candidate.get("non_table_state_sha256")
    validation = validate_candidate(raw, candidate, policy=policy, ocr_tokens=ocr_tokens)
    accept = (
        raw_violations is not None
        and candidate_violations is not None
        and topology_equal
        and token_conserved
        and candidate_violations < raw_violations
        and candidate_violations == 0
        and non_table_equal
        and validation["status"] == "PASS"
    )
    return accept, {
        "raw_violations": raw_violations,
        "candidate_violations": candidate_violations,
        "topology_equal": topology_equal,
        "token_conserved": token_conserved,
        "non_table_equal": non_table_equal,
        "validator_status": validation["status"],
    }


def evaluate(records: Iterable[dict[str, Any]], role: str) -> dict[str, Any]:
    selected = records
    policy = SafetyPolicy(
        policy_id="k1-geometry-selector-v1",
        minimum_expected_gain=0.0,
        threshold_source="NONTERMINAL_PREREGISTERED",
        text_policy="OCR_GROUNDED",
    )
    route_counts: collections.Counter[str] = collections.Counter()
    final_counts: collections.Counter[str] = collections.Counter()
    reason_counts: collections.Counter[str] = collections.Counter()
    per_template: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    group_routes: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    failures = []

    selected_count = 0
    for record in selected:
        selected_count += 1
        raw = record["raw_record"]
        candidate = record["candidate_record"]
        tokens = raw.get("ocr_tokens", [])
        predicted_edit, features = decide(raw, candidate, tokens, policy)
        true_edit = record["oracle_decision"]["action"] == "ACCEPT_EDIT"
        route_key = (
            "tp" if predicted_edit and true_edit else
            "fp" if predicted_edit else
            "fn" if true_edit else
            "tn"
        )
        route_counts[route_key] += 1
        group_routes[str(record["pair_group_id"])][route_key] += 1
        template = str(record["provenance"]["template_family_id"])
        per_template[template][route_key] += 1
        evidence = (
            ExpectedGainEvidence(
                value=float(features["raw_violations"] - features["candidate_violations"]),
                protocol_id="k1-geometry-selector-v1",
                source="NONTERMINAL_DEVELOPMENT",
            )
            if predicted_edit else None
        )
        selection = select_candidate_or_rollback(
            raw,
            candidate,
            policy=policy,
            expected_gain=evidence,
            ocr_tokens=tokens,
        )
        output_is_raw = selection["output_state_sha256"] == selection["raw_state_sha256"]
        output_is_candidate = selection["output_state_sha256"] == stable_sha256(candidate)
        if true_edit:
            final_counts["true_edit_candidate"] += int(output_is_candidate)
            final_counts["true_edit_raw"] += int(output_is_raw)
        else:
            final_counts["true_keep_raw"] += int(output_is_raw)
            final_counts["harmful_accepted"] += int(not output_is_raw)
        reason_counts.update(selection["reason_codes"])
        if predicted_edit != (selection["decision"] == "ACCEPT"):
            failures.append({
                "sample_id": record["sample_id"],
                "predicted_edit": predicted_edit,
                "selection_decision": selection["decision"],
                "features": features,
                "reason_codes": selection["reason_codes"],
            })

    tp, fp, tn, fn = (route_counts[key] for key in ("tp", "fp", "tn", "fn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    keep_preservation = tn / (tn + fp) if tn + fp else 0.0
    accepted_families = sum(values["tp"] > 0 for values in per_template.values())
    minimum_accepted_families = 2 if role == "development" else 4
    group_keep_trials = sum(values["tn"] + values["fp"] > 0 for values in group_routes.values())
    group_accept_trials = sum(values["tp"] + values["fn"] > 0 for values in group_routes.values())
    group_false_edits = sum(values["fp"] > 0 for values in group_routes.values())
    group_true_accepts = sum(values["tp"] > 0 for values in group_routes.values())
    harmful_upper = (
        _upper_zero_failures(group_keep_trials)
        if group_false_edits == 0 else None
    )
    precision_lower = (
        _lower_all_successes(group_true_accepts)
        if group_false_edits == 0 else None
    )
    gates = {
        "raw_good_or_keep_preservation_ge_0_995": keep_preservation >= 0.995,
        "harmful_takeover_group_upper_95_le_0_01": harmful_upper is not None and harmful_upper <= 0.01,
        "takeover_precision_group_lower_95_ge_0_95": precision_lower is not None and precision_lower >= 0.95,
        "nonzero_accept_coverage_meets_role_catalog": accepted_families >= minimum_accepted_families,
        "selector_execution_consistent": not failures,
    }
    return {
        "schema_version": "mpr-tsr/k1-geometry-selector-eval-v1",
        "candidate_source": "controlled_counterfactual_bank_with_offline_gold_positive",
        "candidate_generation_evaluated": False,
        "role": role,
        "records": selected_count,
        "route_counts": dict(route_counts),
        "route_precision": precision,
        "route_recall": recall,
        "keep_preservation": keep_preservation,
        "harmful_takeover_group_upper_95": harmful_upper,
        "all_success_accept_group_lower_95": precision_lower,
        "independent_group_counts": {
            "keep_trials": group_keep_trials,
            "accept_trials": group_accept_trials,
            "false_edit_groups": group_false_edits,
            "true_accept_groups": group_true_accepts,
        },
        "accepted_template_families": accepted_families,
        "minimum_accepted_template_families": minimum_accepted_families,
        "final_counts": dict(final_counts),
        "selection_reason_counts": dict(reason_counts),
        "gates": gates,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "execution_failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the preregistered K1 token-geometry selector.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--role", choices=("development", "holdout"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(_load(args.input, args.role), args.role)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "role": report["role"],
        "records": report["records"],
        "route_counts": report["route_counts"],
        "keep_preservation": report["keep_preservation"],
        "all_success_accept_group_lower_95": report["all_success_accept_group_lower_95"],
        "harmful_takeover_group_upper_95": report["harmful_takeover_group_upper_95"],
        "accepted_template_families": report["accepted_template_families"],
    }, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
