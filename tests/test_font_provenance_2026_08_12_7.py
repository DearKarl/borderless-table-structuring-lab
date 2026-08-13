from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from borderless_table_structuring.synthetic_data import (
    FONT_INVENTORY_PATH,
    REGISTERED_FONT_FAMILIES,
    CoverageRequest,
    _font_candidates,
    _make_record,
    _render_profile,
    _sha256,
    _stable_json,
    _validate_font_inventory,
    _validate_record_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/corpus_release_parameters_2026.08.12.7.json"
SCHEMA_PATH = ROOT / "schemas/synthetic_table_record_2026.08.12.7.json"


def _record() -> tuple[dict[str, object], bytes]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return _make_record(
        CoverageRequest("exact_keep", "KEEP", "simple_regular", "development"),
        0,
        0,
        None,
        config,
        "a" * 64,
        schema,
        dataset_release="shared-corpus-2026.08.12.3",
        schema_release="synthetic-table-record-2026.08.12.7",
        generator_release="2026.08.12.7",
        base_seed_override=202608121234567,
    )


def test_inventory_contains_only_three_registered_ofl_families() -> None:
    assets = _validate_font_inventory()
    assert {asset["font_family_id"] for asset in assets} == REGISTERED_FONT_FAMILIES
    assert all(asset["license_id"] == "OFL-1.1" for asset in assets)
    assert all(not Path(asset["font_source"]).is_absolute() for asset in assets)
    assert all(asset["upstream_commit"] == "038b637da7b3fd956a4ed93ffc607c3d5e4ce172" for asset in assets)


def test_font_selection_is_deterministic_and_host_independent() -> None:
    _font_candidates.cache_clear()
    first = _render_profile(202608121234567, "weak_borders")
    _font_candidates.cache_clear()
    second = _render_profile(202608121234567, "weak_borders")
    assert first == second
    assert first["font_family_id"] in REGISTERED_FONT_FAMILIES
    assert first["font_source"].startswith("assets/fonts/")


def test_missing_font_asset_is_a_hard_failure(tmp_path: Path) -> None:
    inventory = json.loads(FONT_INVENTORY_PATH.read_text(encoding="utf-8"))
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="FONT_REGISTERED_ASSET_MISSING"):
        _validate_font_inventory(path, tmp_path)


def test_modified_font_asset_is_a_hard_failure(tmp_path: Path) -> None:
    inventory = json.loads(FONT_INVENTORY_PATH.read_text(encoding="utf-8"))
    for asset in inventory["assets"]:
        for key in ("font_source", "license_path", "metadata_path"):
            source = ROOT / asset[key]
            target = tmp_path / asset[key]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    target_font = tmp_path / inventory["assets"][0]["font_source"]
    target_font.write_bytes(target_font.read_bytes() + b"tamper")
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    with pytest.raises(ValueError, match="FONT_ASSET_SIZE_MISMATCH"):
        _validate_font_inventory(inventory_path, tmp_path)


def test_absolute_host_font_path_is_rejected() -> None:
    inventory = json.loads(FONT_INVENTORY_PATH.read_text(encoding="utf-8"))
    inventory["assets"][0]["font_source"] = "/System/Library/Fonts/Arial.ttf"
    temporary = ROOT / "assets/fonts/TEST_HOST_PATH_REJECTION_2026.08.12.1.json"
    try:
        temporary.write_text(json.dumps(inventory), encoding="utf-8")
        with pytest.raises(ValueError, match="FONT_HOST_PATH_PROHIBITED"):
            _validate_font_inventory(temporary, ROOT)
    finally:
        temporary.unlink(missing_ok=True)


def test_unknown_family_is_rejected() -> None:
    inventory = json.loads(FONT_INVENTORY_PATH.read_text(encoding="utf-8"))
    inventory["assets"][0]["font_family_id"] = "unknown-font"
    temporary = ROOT / "assets/fonts/TEST_UNKNOWN_FAMILY_2026.08.12.1.json"
    try:
        temporary.write_text(json.dumps(inventory), encoding="utf-8")
        with pytest.raises(ValueError, match="FONT_FAMILY_SET_MISMATCH"):
            _validate_font_inventory(temporary, ROOT)
    finally:
        temporary.unlink(missing_ok=True)


def test_record_contains_complete_registered_font_provenance() -> None:
    record, image = _record()
    assert image.startswith(b"\x89PNG")
    font = record["rendering"]["font_sources"][0]
    registered = {asset["font_family_id"]: asset for asset in _font_candidates()}
    assert font == registered[font["font_family_id"]]
    assert record["identity"]["font_family_id"] == font["font_family_id"]
    _validate_record_contract(record)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert not list(Draft202012Validator(schema).iter_errors(record))


def test_record_schema_rejects_host_font_path_and_missing_license() -> None:
    record, _ = _record()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    bad_path = copy.deepcopy(record)
    bad_path["rendering"]["parameters"]["font_source"] = "/tmp/font.ttf"
    assert list(Draft202012Validator(schema).iter_errors(bad_path))
    missing_license = copy.deepcopy(record)
    del missing_license["rendering"]["font_sources"][0]["license_sha256"]
    assert list(Draft202012Validator(schema).iter_errors(missing_license))


def test_record_contract_rejects_forged_registered_hash() -> None:
    record, _ = _record()
    forged = copy.deepcopy(record)
    forged["rendering"]["font_sources"][0]["font_source_sha256"] = _sha256(
        _stable_json("forged")
    )
    with pytest.raises(ValueError, match="FONT_PROVENANCE_NOT_REGISTERED"):
        _validate_record_contract(forged)
