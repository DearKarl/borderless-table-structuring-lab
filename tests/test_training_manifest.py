import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("training_verifier", ROOT / "training/verify_checkpoint.py")
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def test_manifest_truthful():
    manifest = json.loads((ROOT / "training/model_manifest.json").read_text())
    assert manifest["availability"] == "EXACT_WEIGHTS_GITHUB_RELEASE"
    assert manifest["download_url"].endswith('/training-explicit-v2-original-2026.09.13.1/model.safe-state')
    assert set(manifest['files']) == {'model.safe-state'}
    assert manifest['saved_metadata']['update'] == 894
    assert manifest["observed_result"]["adopted_operations"] == 0
    assert not manifest["observed_result"]["objective_strictly_above_95_met"]


def test_inference_source_bytes_match_archive():
    manifest = json.loads((ROOT / 'training/model_manifest.json').read_text())
    for name, expected in manifest['inference_source_files'].items():
        assert hashlib.sha256((ROOT / 'training' / name).read_bytes()).hexdigest() == expected


def test_wrong_or_missing_file_rejected(tmp_path):
    manifest = {"files": {"model.safe-state": hashlib.sha256(b"expected").hexdigest()}}
    with pytest.raises(ValueError, match="Missing regular"):
        VERIFIER.verify(tmp_path, manifest)
    (tmp_path / "model.safe-state").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="identity mismatch"):
        VERIFIER.verify(tmp_path, manifest)


def test_matching_file_only_proves_identity(tmp_path):
    (tmp_path / "model.safe-state").write_bytes(b"fixture")
    result = VERIFIER.verify(tmp_path, {"files": {"model.safe-state": hashlib.sha256(b"fixture").hexdigest()}})
    assert result["status"] == "EXACT_FILES_VERIFIED_NOT_INFERENCE_OR_SCORE"
