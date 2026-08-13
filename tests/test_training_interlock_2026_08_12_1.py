from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from borderless_table_structuring.training_interlock import require_train_ready_corpus


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generated_directory_without_seal_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="TRAIN_READY_SEAL_MISSING"):
        require_train_ready_corpus(tmp_path)


def test_failed_status_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "TRAIN_READY_SEAL.json").write_text(
        json.dumps({"status": "FAILED_NOT_TRAINABLE"}), encoding="utf-8"
    )
    with pytest.raises(PermissionError, match="TRAIN_READY_STATUS_INVALID"):
        require_train_ready_corpus(tmp_path)


def test_train_ready_manifest_hash_is_verified(tmp_path: Path) -> None:
    manifest = tmp_path / "records.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    (tmp_path / "TRAIN_READY_SEAL.json").write_text(
        json.dumps(
            {
                "status": "TRAIN_READY_SEALED",
                "training_authorized": True,
                "manifest": {"path": "records.jsonl", "sha256": _sha256(manifest)},
                "builder_verifier_agree": True,
                "all_hard_gates_passed": True,
            }
        ),
        encoding="utf-8",
    )
    seal = require_train_ready_corpus(tmp_path)
    assert seal["status"] == "TRAIN_READY_SEALED"


def test_changed_manifest_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "records.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    expected = _sha256(manifest)
    (tmp_path / "TRAIN_READY_SEAL.json").write_text(
        json.dumps(
            {
                "status": "TRAIN_READY_SEALED",
                "training_authorized": True,
                "manifest": {"path": "records.jsonl", "sha256": expected},
                "builder_verifier_agree": True,
                "all_hard_gates_passed": True,
            }
        ),
        encoding="utf-8",
    )
    manifest.write_text("{\"changed\":true}\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="TRAIN_READY_MANIFEST_HASH_MISMATCH"):
        require_train_ready_corpus(tmp_path)
