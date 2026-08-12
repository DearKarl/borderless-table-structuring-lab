from __future__ import annotations

import json
from pathlib import Path

import pytest

from borderless_table_structuring.canonical import validate_cells
from borderless_table_structuring.synthetic_data import (
    CoverageRequest,
    _difference,
    _gold_table,
    _license_manifest,
    _make_record,
    _negative_grid_fixture,
    _overlap_report,
    _prior_for_edit,
    _sha256,
    _stable_json,
    _token_ownership_valid,
)


def test_keep_table_has_empty_order_invariant_difference() -> None:
    gold = _gold_table(2026081204, 6, 5, "mixed_two_dimensional_spans", "hard_keep")
    assert not validate_cells(gold["cells"])
    assert _difference(gold, gold) == []
    assert _token_ownership_valid(gold)


@pytest.mark.parametrize(
    "phenomenon",
    [
        "extra_split",
        "missing_split",
        "incorrect_merge",
        "missing_merge",
        "span_extent_error",
        "token_ownership_error",
        "geometry_inconsistency",
        "row_or_column_assignment_error",
    ],
)
def test_single_edit_priors_remain_legal_and_differ_from_gold(phenomenon: str) -> None:
    gold = _gold_table(2026081204, 7, 6, phenomenon, "single_minimal_edit")
    prior, operations = _prior_for_edit(gold, phenomenon, "single_minimal_edit")
    assert operations
    assert not validate_cells(gold["cells"])
    assert not validate_cells(prior["cells"])
    assert prior["semantic_state_sha256"] != gold["semantic_state_sha256"]
    assert _difference(prior, gold)
    assert _token_ownership_valid(prior)


def test_complex_correction_contains_multiple_operations() -> None:
    gold = _gold_table(2026081204, 8, 6, "joint_split_and_merge", "complex_correction")
    prior, operations = _prior_for_edit(gold, "joint_split_and_merge", "complex_correction")
    assert len(operations) >= 2
    assert not validate_cells(prior["cells"])


@pytest.mark.parametrize("columns", [2, 3, 5, 14])
def test_extra_split_is_non_identity_across_configured_column_range(
    columns: int,
) -> None:
    gold = _gold_table(
        2026081205 + columns,
        5,
        columns,
        "weak_borders extra_split",
        "single_minimal_edit",
    )
    prior, operations = _prior_for_edit(
        gold, "extra_split", "single_minimal_edit"
    )
    assert operations == ["split_spanning_cell"]
    assert prior["semantic_state_sha256"] != gold["semantic_state_sha256"]


def test_incomplete_grid_is_rejection_fixture_only() -> None:
    assert _negative_grid_fixture() == "GRID_INCOMPLETE"


def test_stable_json_hash_is_order_independent() -> None:
    assert _sha256(_stable_json({"a": 1, "b": 2})) == _sha256(
        _stable_json({"b": 2, "a": 1})
    )


def test_counterfactual_pair_shares_edit_capable_gold_and_image() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs" / "generation_parameters_2026.08.12.4.json").read_text()
    )
    schema = json.loads(
        (root / "schemas" / "synthetic_table_record_2026.08.12.4.json").read_text()
    )
    license_sha = _sha256(_stable_json(_license_manifest()))
    shared = "weak_borders extra_split"
    keep, keep_image = _make_record(
        CoverageRequest("hard_keep", "KEEP", "weak_borders", "development"),
        0,
        0,
        0,
        config,
        license_sha,
        schema,
        shared,
    )
    edit, edit_image = _make_record(
        CoverageRequest(
            "single_minimal_edit", "EDIT", "extra_split", "development"
        ),
        1,
        0,
        0,
        config,
        license_sha,
        schema,
        shared,
    )
    assert keep["gold"]["full_state_sha256"] == edit["gold"]["full_state_sha256"]
    assert keep_image == edit_image
    assert keep["gold"]["semantic_state_sha256"] == keep["prior"]["semantic_state_sha256"]
    assert edit["gold"]["semantic_state_sha256"] != edit["prior"]["semantic_state_sha256"]


def test_single_cross_role_structure_signal_is_not_a_hard_overlap() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs" / "generation_parameters_2026.08.12.4.json").read_text()
    )
    schema = json.loads(
        (root / "schemas" / "synthetic_table_record_2026.08.12.4.json").read_text()
    )
    license_sha = _sha256(_stable_json(_license_manifest()))
    first, first_image = _make_record(
        CoverageRequest("exact_keep", "KEEP", "ordinary_grid", "train"),
        0,
        1,
        None,
        config,
        license_sha,
        schema,
    )
    second, second_image = _make_record(
        CoverageRequest("exact_keep", "KEEP", "ordinary_grid", "holdout"),
        1,
        2,
        None,
        config,
        license_sha,
        schema,
    )
    report = _overlap_report(
        [first, second],
        {first["sample_id"]: first_image, second["sample_id"]: second_image},
    )
    assert report["status"] == "PASS"
