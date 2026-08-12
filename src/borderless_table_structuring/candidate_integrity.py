from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


FORBIDDEN_TARGET_FIELDS = {"gold_text", "target_text"}
EXPECTED_VIEW_NAMES = ["tight", "padded"]
EXPECTED_RECOGNIZER = "PaddleOCR/PP-OCRv5_server_rec"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _has_forbidden_target_field(value: Any) -> bool:
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            if FORBIDDEN_TARGET_FIELDS & set(current):
                return True
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return False


def _finite_box(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    if not all(
        isinstance(item, (int, float)) and math.isfinite(float(item))
        for item in value[:4]
    ):
        return None
    return [float(item) for item in value[:4]]


def _boxes_equal(left: list[float], right: list[float]) -> bool:
    return all(abs(a - b) <= 1e-6 for a, b in zip(left, right))


def _index_sidecars(
    sidecar_dir: Path, issues: Counter[str]
) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in sorted(sidecar_dir.glob("*.json")):
        try:
            value = _read_json(path)
        except (OSError, json.JSONDecodeError, TypeError):
            issues["invalid_sidecar_json"] += 1
            continue
        sample_id = str(value.get("sample_id", ""))
        if not sample_id:
            issues["missing_sample_id"] += 1
            continue
        if sample_id in indexed:
            issues["duplicate_sample_id"] += 1
            continue
        indexed[sample_id] = path
    return indexed


def audit_candidate_integrity(
    manifest: Path,
    sidecar_dir: Path,
    candidate_audit: dict[str, Any],
    *,
    expected_tables: int,
) -> dict[str, Any]:
    issues: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    view_count_distribution: Counter[int] = Counter()
    indexed = _index_sidecars(sidecar_dir, issues)
    unmatched = set(indexed)
    seen_manifest: set[str] = set()
    threshold = float(candidate_audit.get("token_confidence_threshold", -1.0))
    maximum_candidates = int(
        candidate_audit.get("maximum_candidates_per_table", -1)
    )
    expected_policy = candidate_audit.get("candidate_proposal_policy")
    expected_mode = candidate_audit.get("candidate_proposal_mode")
    expected_gate_threshold = candidate_audit.get("proposal_gate_threshold")
    expected_config_sha256 = candidate_audit.get("frozen_config_sha256")
    expected_model_sha256 = candidate_audit.get("model_artifact_sha256")
    expected_geometry_sha256 = candidate_audit.get(
        "geometry_checkpoint_sha256"
    )

    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            totals["manifest_tables"] += 1
            if sample_id in seen_manifest:
                issues["duplicate_manifest_sample_id"] += 1
            seen_manifest.add(sample_id)
            path = indexed.get(sample_id)
            if path is None:
                issues["missing_sidecar"] += 1
                continue
            unmatched.discard(sample_id)
            value = _read_json(path)
            provenance = value.get("provenance", {})
            expected_provenance = {
                "gold_trigger_visible_to_proposal": False,
                "gold_text_visible_to_recognizer": False,
                "gold_geometry_visible_to_recognizer": False,
                "candidate_text_frozen_before_gold_matching": True,
                "restricted_evaluation_visible": False,
                "crop_geometry_source": "frozen_paddleocr_token_bbox_only",
                "logical_projection_bbox_forbidden": True,
                "candidate_proposal_policy": expected_policy,
                "candidate_proposal_mode": expected_mode,
                "proposal_gate_threshold": expected_gate_threshold,
                "token_confidence_threshold": threshold,
                "maximum_candidates_per_table": maximum_candidates,
                "role": "public_training_candidates_only",
                "trigger_source": "frozen_ocr_token_confidence",
                "frozen_config_sha256": expected_config_sha256,
                "model_artifact_sha256": expected_model_sha256,
                "geometry_checkpoint_sha256": expected_geometry_sha256,
                "input_image_sha256": row.get("image", {}).get(
                    "input_image_sha256"
                ),
            }
            for name, expected in expected_provenance.items():
                if provenance.get(name) != expected:
                    issues[f"provenance_{name}_mismatch"] += 1
            if _has_forbidden_target_field(value):
                issues["forbidden_target_field"] += 1
            token_candidates = value.get("token_candidates")
            if not isinstance(token_candidates, list):
                issues["token_candidates_not_list"] += 1
                continue
            if int(value.get("proposal_count", -1)) != len(token_candidates):
                issues["proposal_count_mismatch"] += 1
            if len(token_candidates) > maximum_candidates:
                issues["maximum_candidates_exceeded"] += 1
            totals["candidate_tokens"] += len(token_candidates)
            totals["zero_candidate_tables"] += int(not token_candidates)
            source_tokens = row.get("ocr_tokens", [])
            width = float(row.get("image", {}).get("width", 0))
            height = float(row.get("image", {}).get("height", 0))
            token_indexes: set[int] = set()
            ordered_token_indexes: list[int] = []
            for candidate in token_candidates:
                token_index = candidate.get("token_index")
                if (
                    not isinstance(token_index, int)
                    or token_index < 0
                    or token_index >= len(source_tokens)
                ):
                    issues["invalid_token_index"] += 1
                    continue
                if token_index in token_indexes:
                    issues["duplicate_token_index"] += 1
                token_indexes.add(token_index)
                ordered_token_indexes.append(token_index)
                source = source_tokens[token_index]
                source_confidence = candidate.get("source_confidence")
                frozen_source_confidence = source.get("confidence")
                if not isinstance(source_confidence, (int, float)) or not (
                    math.isfinite(float(source_confidence))
                    and float(source_confidence) < threshold
                ):
                    issues["source_confidence_not_below_threshold"] += 1
                elif not isinstance(frozen_source_confidence, (int, float)) or not (
                    math.isfinite(float(frozen_source_confidence))
                ):
                    issues["invalid_frozen_source_confidence"] += 1
                elif abs(
                    float(source_confidence) - float(frozen_source_confidence)
                ) > 1e-6:
                    issues["source_confidence_mismatch"] += 1
                if str(candidate.get("source_text", "")) != str(
                    source.get("text", "")
                ):
                    issues["source_text_mismatch"] += 1
                box = _finite_box(candidate.get("absolute_bbox"))
                source_box = _finite_box(source.get("bbox"))
                if box is None:
                    issues["invalid_bbox"] += 1
                else:
                    x0, y0, x1, y1 = box
                    if x1 <= x0 or y1 <= y0:
                        issues["degenerate_bbox"] += 1
                    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
                        issues["bbox_out_of_image"] += 1
                    if source_box is None or not _boxes_equal(box, source_box):
                        issues["bbox_source_mismatch"] += 1
                candidates = candidate.get("candidates")
                if not isinstance(candidates, list):
                    issues["candidate_views_not_list"] += 1
                    continue
                view_count_distribution[len(candidates)] += 1
                if [item.get("view") for item in candidates] != EXPECTED_VIEW_NAMES:
                    issues["view_contract_mismatch"] += 1
                for item in candidates:
                    totals["candidate_views"] += 1
                    totals["nonempty_candidate_views"] += int(
                        bool(str(item.get("text", "")).strip())
                    )
                    confidence = item.get("confidence")
                    if not isinstance(confidence, (int, float)) or not math.isfinite(
                        float(confidence)
                    ):
                        issues["invalid_recognition_confidence"] += 1
                    if item.get("model_id") != EXPECTED_RECOGNIZER:
                        issues["recognizer_model_mismatch"] += 1
            expected_token_indexes = [
                index
                for index, source in enumerate(source_tokens)
                if isinstance(source.get("confidence"), (int, float))
                and math.isfinite(float(source["confidence"]))
                and float(source["confidence"]) < threshold
                and _finite_box(source.get("bbox")) is not None
            ][:maximum_candidates]
            if ordered_token_indexes != expected_token_indexes:
                issues["eligible_token_sequence_mismatch"] += 1

    if unmatched:
        issues["sidecar_not_in_manifest"] += len(unmatched)
    checks = {
        "candidate_audit_pass": candidate_audit.get("status") == "PASS",
        "manifest_count": totals["manifest_tables"] == expected_tables,
        "unique_sidecar_count": len(indexed) == expected_tables,
        "all_sidecars_match_manifest": not unmatched
        and issues["missing_sidecar"] == 0,
        "zero_semantic_issues": not any(issues.values()),
        "candidate_pool_nonempty": totals["candidate_tokens"] > 0,
        "two_views_per_candidate": (
            set(view_count_distribution) <= {2}
            and totals["candidate_views"] == 2 * totals["candidate_tokens"]
        ),
    }
    return {
        "schema_release": "candidate-integrity-2026.08.12",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "manifest": str(manifest.resolve()),
        "sidecar_dir": str(sidecar_dir.resolve()),
        "expected_tables": expected_tables,
        "unique_sidecars": len(indexed),
        "totals": dict(sorted(totals.items())),
        "view_count_distribution": {
            str(name): count
            for name, count in sorted(view_count_distribution.items())
        },
        "issues": dict(sorted(issues.items())),
        "restricted_evaluation_visible": False,
    }
