from __future__ import annotations

import json
from pathlib import Path

from borderless_table_structuring.candidate_integrity import (
    audit_candidate_integrity,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    manifest = tmp_path / "manifest.jsonl"
    row = {
        "sample_id": "table-1",
        "image": {
            "width": 100,
            "height": 50,
            "input_image_sha256": "image-sha",
        },
        "ocr_tokens": [
            {"text": "12.3", "confidence": 0.8, "bbox": [1, 2, 30, 12]}
        ],
    }
    _write_json(manifest, row)
    sidecar_dir = tmp_path / "sidecars"
    sidecar = {
        "sample_id": "table-1",
        "proposal_count": 1,
        "token_candidates": [
            {
                "token_index": 0,
                "source_text": "12.3",
                "source_confidence": 0.8,
                "absolute_bbox": [1, 2, 30, 12],
                "candidates": [
                    {
                        "view": "tight",
                        "text": "12.3",
                        "confidence": 0.9,
                        "model_id": "PaddleOCR/PP-OCRv5_server_rec",
                    },
                    {
                        "view": "padded",
                        "text": "12.3",
                        "confidence": 0.95,
                        "model_id": "PaddleOCR/PP-OCRv5_server_rec",
                    },
                ],
            }
        ],
        "provenance": {
            "gold_trigger_visible_to_proposal": False,
            "gold_text_visible_to_recognizer": False,
            "gold_geometry_visible_to_recognizer": False,
            "candidate_text_frozen_before_gold_matching": True,
            "restricted_evaluation_visible": False,
            "crop_geometry_source": "frozen_paddleocr_token_bbox_only",
            "logical_projection_bbox_forbidden": True,
            "candidate_proposal_policy": "proposal-policy-2026.08.12",
            "candidate_proposal_mode": "low_confidence_ocr_token_lines",
            "proposal_gate_threshold": 0.05,
            "token_confidence_threshold": 0.95,
            "maximum_candidates_per_table": 1180,
            "role": "public_training_candidates_only",
            "trigger_source": "frozen_ocr_token_confidence",
            "frozen_config_sha256": "config-sha",
            "model_artifact_sha256": "model-sha",
            "geometry_checkpoint_sha256": "checkpoint-sha",
            "input_image_sha256": "image-sha",
        },
    }
    _write_json(sidecar_dir / "sidecar.json", sidecar)
    audit = {
        "status": "PASS",
        "token_confidence_threshold": 0.95,
        "maximum_candidates_per_table": 1180,
        "candidate_proposal_policy": "proposal-policy-2026.08.12",
        "candidate_proposal_mode": "low_confidence_ocr_token_lines",
        "proposal_gate_threshold": 0.05,
        "frozen_config_sha256": "config-sha",
        "model_artifact_sha256": "model-sha",
        "geometry_checkpoint_sha256": "checkpoint-sha",
    }
    return manifest, sidecar_dir, audit


def test_candidate_integrity_accepts_complete_gold_free_sidecar(
    tmp_path: Path,
) -> None:
    manifest, sidecar_dir, audit = _fixture(tmp_path)
    report = audit_candidate_integrity(
        manifest, sidecar_dir, audit, expected_tables=1
    )
    assert report["status"] == "PASS"
    assert report["totals"]["candidate_views"] == 2
    assert report["issues"] == {}


def test_candidate_integrity_rejects_gold_field_and_bbox_misbinding(
    tmp_path: Path,
) -> None:
    manifest, sidecar_dir, audit = _fixture(tmp_path)
    path = sidecar_dir / "sidecar.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["gold_text"] = "leak"
    value["token_candidates"][0]["absolute_bbox"] = [2, 2, 30, 12]
    _write_json(path, value)
    report = audit_candidate_integrity(
        manifest, sidecar_dir, audit, expected_tables=1
    )
    assert report["status"] == "FAIL"
    assert report["issues"]["forbidden_target_field"] == 1
    assert report["issues"]["bbox_source_mismatch"] == 1


def test_candidate_integrity_rejects_missing_eligible_token(
    tmp_path: Path,
) -> None:
    manifest, sidecar_dir, audit = _fixture(tmp_path)
    path = sidecar_dir / "sidecar.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["token_candidates"] = []
    value["proposal_count"] = 0
    _write_json(path, value)
    report = audit_candidate_integrity(
        manifest, sidecar_dir, audit, expected_tables=1
    )
    assert report["status"] == "FAIL"
    assert report["issues"]["eligible_token_sequence_mismatch"] == 1
