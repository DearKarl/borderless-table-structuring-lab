from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


TRAIN_READY_STATUS = "TRAIN_READY_SEALED"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_train_ready_corpus(corpus: Path) -> dict[str, Any]:
    seal_path = corpus / "TRAIN_READY_SEAL.json"
    if not seal_path.is_file():
        raise PermissionError("TRAIN_READY_SEAL_MISSING")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("status") != TRAIN_READY_STATUS:
        raise PermissionError("TRAIN_READY_STATUS_INVALID")
    if seal.get("training_authorized") is not True:
        raise PermissionError("TRAINING_AUTHORIZATION_MISSING")
    manifest_relative = seal.get("manifest", {}).get("path")
    manifest_sha256 = seal.get("manifest", {}).get("sha256")
    if not isinstance(manifest_relative, str) or not isinstance(manifest_sha256, str):
        raise PermissionError("TRAIN_READY_MANIFEST_REFERENCE_INVALID")
    manifest_path = corpus / manifest_relative
    if not manifest_path.is_file():
        raise PermissionError("TRAIN_READY_MANIFEST_MISSING")
    if _sha256(manifest_path) != manifest_sha256:
        raise PermissionError("TRAIN_READY_MANIFEST_HASH_MISMATCH")
    if seal.get("builder_verifier_agree") is not True:
        raise PermissionError("INDEPENDENT_VERIFIER_NOT_CONFIRMED")
    if seal.get("all_hard_gates_passed") is not True:
        raise PermissionError("HARD_GATES_NOT_CONFIRMED")
    return seal
