from __future__ import annotations

import json
from pathlib import Path

from borderless_table_structuring.corpus_release import (
    _family_specs,
    _load_coverage,
    audit_latent_plan,
    create_latent_plan,
    verify_latent_plan,
    write_latent_plan,
)
from borderless_table_structuring.synthetic_data import (
    CoverageRequest,
    _deterministic_label,
    _generative_identity,
    _gold_table,
    _sample_shape,
    _sha256,
    _stable_json,
)


ROOT = Path(__file__).resolve().parents[1]


def _small_config() -> dict[str, object]:
    return {
        "seed_start": 202608120000000,
        "requested_records": 4,
        "categories": {
            "exact_keep": {"count": 1},
            "hard_keep": {"count": 1},
            "single_minimal_edit": {"count": 1},
            "complex_correction": {"count": 1},
        },
        "roles_per_category": {"development": 1},
        "structure": {
            "rows": {"minimum": 3, "maximum": 10},
            "columns": {"minimum": 2, "maximum": 8},
            "maximum_cells": 80,
        },
    }


def _small_coverage(path: Path) -> None:
    path.write_text(
        "category,gate_label,phenomenon,requested_count,train_count,development_count,holdout_count,counterfactual_pair_required,primary_risk\n"
        "exact_keep,KEEP,simple_regular,1,0,1,0,false,false_edit\n"
        "hard_keep,KEEP,weak_borders,1,0,1,0,true,false_edit\n"
        "single_minimal_edit,EDIT,extra_split,1,0,1,0,true,cell_inflation\n"
        "complex_correction,EDIT,joint_split_and_merge,1,0,1,0,false,cell_inflation\n",
        encoding="utf-8",
    )


def test_content_does_not_repeat_at_the_previous_modulo_boundary() -> None:
    assert _deterministic_label(10, 0, 0) != _deterministic_label(10017, 0, 0)


def test_family_identity_is_independent_of_role_and_record_id() -> None:
    config = _small_config()
    base_seed = 202608121234567
    rows, columns = _sample_shape(base_seed, config)
    gold = _gold_table(base_seed, rows, columns, "weak_borders", "hard_keep")
    structure = {
        "rows": rows,
        "columns": columns,
        "cells": [
            {
                "row": cell["row"],
                "col": cell["col"],
                "rowspan": cell["rowspan"],
                "colspan": cell["colspan"],
                "tag": cell["tag"],
            }
            for cell in gold["cells"]
        ],
    }
    first = _generative_identity(
        base_seed,
        rows,
        columns,
        "weak_borders",
        "hard_keep",
        _sha256(_stable_json(structure)),
    )
    second = _generative_identity(
        base_seed,
        rows,
        columns,
        "weak_borders",
        "hard_keep",
        _sha256(_stable_json(structure)),
    )
    assert first == second
    assert "train" not in json.dumps(first)
    assert "holdout" not in json.dumps(first)


def test_counterfactual_family_is_never_split_across_roles() -> None:
    requests = [
        CoverageRequest("hard_keep", "KEEP", "weak_borders", "train"),
        CoverageRequest("single_minimal_edit", "EDIT", "extra_split", "train"),
        CoverageRequest("exact_keep", "KEEP", "simple_regular", "development"),
        CoverageRequest("complex_correction", "EDIT", "joint_split_and_merge", "development"),
    ]
    families = _family_specs(requests)
    paired = [family for family in families if len(family["members"]) == 2]
    assert len(paired) == 1
    assert "role" not in paired[0]


def test_family_construction_is_independent_of_requested_roles() -> None:
    first = [
        CoverageRequest("hard_keep", "KEEP", "weak_borders", "train"),
        CoverageRequest("single_minimal_edit", "EDIT", "extra_split", "train"),
        CoverageRequest("exact_keep", "KEEP", "simple_regular", "development"),
        CoverageRequest(
            "complex_correction", "EDIT", "joint_split_and_merge", "holdout"
        ),
    ]
    second = [
        CoverageRequest("hard_keep", "KEEP", "weak_borders", "holdout"),
        CoverageRequest(
            "single_minimal_edit", "EDIT", "extra_split", "holdout"
        ),
        CoverageRequest("exact_keep", "KEEP", "simple_regular", "train"),
        CoverageRequest(
            "complex_correction", "EDIT", "joint_split_and_merge", "development"
        ),
    ]
    assert _family_specs(first) == _family_specs(second)


def test_small_plan_builder_and_independent_verifier(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    coverage_path = tmp_path / "coverage.csv"
    config_path.write_text(json.dumps(_small_config()), encoding="utf-8")
    _small_coverage(coverage_path)
    output = tmp_path / "latent-plan"
    builder = write_latent_plan(output, config_path, coverage_path)
    verifier = verify_latent_plan(output, config_path, coverage_path)
    assert builder["status"] == "PASS"
    assert verifier["status"] == "PASS"
    assert verifier["builder_verifier_agree"] is True
    plan = [
        json.loads(line)
        for line in (output / "latent_plan.jsonl").read_text().splitlines()
    ]
    assert {family["role"] for family in plan} == {"development"}
    assert all(family["duplicate_component_size"] == 1 for family in plan)


def test_full_coverage_families_are_role_free_before_split() -> None:
    requests = _load_coverage(
        ROOT / "docs/corpus/SHARED_CORPUS_COVERAGE_2026.08.12.csv"
    )
    families = _family_specs(requests)
    assert sum(len(family["members"]) for family in families) == 40_000
    assert all("role" not in family for family in families)
    paired = [family for family in families if len(family["members"]) == 2]
    assert len(paired) == 10_000
    assert all(
        {member["category"] for member in family["members"]}
        == {"hard_keep", "single_minimal_edit"}
        for family in paired
    )


def test_cross_role_signature_collision_is_a_hard_failure() -> None:
    config = _small_config()
    realized = {
        "generative_family_key": "same",
        "structure_sha256": "same",
        "geometry_sha256": "same",
        "text_sha256": "same",
        "renderer_sha256": "same",
        "base_seed_sha256": "same",
    }
    plan = [
        {
            "family_id": "one",
            "counterfactual_group_id": None,
            "role": "train",
            "members": [{"category": "exact_keep", "phenomenon": "simple"}],
            "realized": realized,
            "terminal_inputs_used": False,
        },
        {
            "family_id": "two",
            "counterfactual_group_id": None,
            "role": "development",
            "members": [{"category": "complex_correction", "phenomenon": "complex"}],
            "realized": realized,
            "terminal_inputs_used": False,
        },
    ]
    report = audit_latent_plan(plan, config)
    assert report["status"] == "FAIL"
    assert report["cross_role_overlaps"]
