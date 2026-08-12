from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from borderless_table_structuring.shared_corpus import (
    DATASET_RELEASE,
    GENERATOR_RELEASE,
    SCHEMA_RELEASE,
    _audit_signatures,
    _prepare_assignments,
    build_shared_corpus,
    verify_shared_corpus,
)
from borderless_table_structuring.synthetic_data import (
    CoverageRequest,
    _expand_coverage,
    _shape_pool,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_shared_corpus_distribution() -> None:
    config = json.loads(
        (ROOT / "configs" / "shared_corpus_parameters_2026.08.12.json").read_text()
    )
    requests = _expand_coverage(
        ROOT / "docs" / "corpus" / "SHARED_CORPUS_COVERAGE_2026.08.12.csv"
    )
    categories = Counter(request.category for request in requests)
    roles = Counter(request.role for request in requests)
    category_roles: dict[str, Counter[str]] = defaultdict(Counter)
    for request in requests:
        category_roles[request.category][request.role] += 1

    assert len(requests) == config["requested_records"] == 40000
    assert categories == Counter(
        {
            "exact_keep": 10000,
            "hard_keep": 10000,
            "single_minimal_edit": 10000,
            "complex_correction": 10000,
        }
    )
    assert roles == Counter({"train": 28000, "development": 8000, "holdout": 4000})
    assert all(
        counts == Counter({"train": 7000, "development": 2000, "holdout": 1000})
        for counts in category_roles.values()
    )


def test_frozen_schema_and_config_identifiers_agree() -> None:
    config = json.loads(
        (ROOT / "configs" / "shared_corpus_parameters_2026.08.12.json").read_text()
    )
    schema = json.loads(
        (ROOT / "schemas" / "synthetic_table_record_2026.08.12.5.json").read_text()
    )
    assert config["dataset_release"] == DATASET_RELEASE
    assert config["generator_release"] == GENERATOR_RELEASE
    assert config["schema_release"] == SCHEMA_RELEASE
    assert schema["properties"]["dataset_release"]["const"] == DATASET_RELEASE
    assert schema["properties"]["schema_release"]["const"] == SCHEMA_RELEASE
    assert schema["properties"]["provenance"]["properties"]["generator_release"]["const"] == GENERATOR_RELEASE


def test_counterfactual_assignments_are_complete_and_same_role() -> None:
    requests = _expand_coverage(
        ROOT / "docs" / "corpus" / "SHARED_CORPUS_COVERAGE_2026.08.12.csv"
    )
    assignments = _prepare_assignments(requests)
    paired: dict[int, list[CoverageRequest]] = defaultdict(list)
    for request in requests:
        family_index, pair_index, shared = assignments[id(request)]
        assert family_index >= 0
        if pair_index is not None:
            assert shared
            paired[pair_index].append(request)
    assert len(assignments) == len(requests)
    assert len(paired) == 10000
    assert all(
        len(members) == 2
        and {member.category for member in members}
        == {"hard_keep", "single_minimal_edit"}
        and len({member.role for member in members}) == 1
        for members in paired.values()
    )


def test_shape_pool_is_cached_and_supports_reuse() -> None:
    first = _shape_pool(2026081205)
    second = _shape_pool(2026081205)
    assert first is second
    assert len(first) > 100
    assert all(3 <= rows <= 28 and 2 <= columns <= 14 for rows, columns in first)
    assert all(rows * columns <= 240 for rows, columns in first)


def _signature(sample: str, role: str, phash: str, *, text: str, structure: str, geometry: str) -> dict[str, object]:
    return {
        "sample_id": sample,
        "role": role,
        "image_sha256": f"image-{sample}",
        "pixel_sha256": f"pixel-{sample}",
        "table_sha256": f"table-{sample}",
        "structure_sha256": structure,
        "text_sha256": text,
        "geometry_sha256": geometry,
        "perceptual_hash": phash,
        "families": {
            "document_cluster_id": f"document-{role}-{sample}",
            "source_family_id": f"source-{role}-{sample}",
            "template_family_id": f"template-{role}-{sample}",
            "content_family_id": f"content-{role}-{sample}",
            "renderer_family_id": f"renderer-{role}-{sample}",
        },
    }


def test_near_overlap_audit_is_not_limited_to_a_phash_prefix() -> None:
    # These hashes differ in the first hexadecimal digit but have Hamming
    # distance one. Matching text is the preregistered supporting signal.
    first = _signature(
        "first", "train", "0000000000000000", text="same", structure="a", geometry="b"
    )
    second = _signature(
        "second", "holdout", "8000000000000000", text="same", structure="c", geometry="d"
    )
    report = _audit_signatures([first, second])
    assert report["status"] == "FAIL"
    assert report["cross_role_unresolved_near_overlaps"][0]["distance"] == 1


def test_single_structure_signature_is_reported_but_not_rejected() -> None:
    first = _signature(
        "first", "train", "0000000000000000", text="a", structure="same", geometry="b"
    )
    second = _signature(
        "second", "holdout", "ffffffffffffffff", text="c", structure="same", geometry="d"
    )
    report = _audit_signatures([first, second])
    assert report["status"] == "PASS"
    assert report["within_and_cross_role_duplicate_signature_members"]["structure_sha256"] == 2


def test_small_non_overwriting_build_and_independent_verification(tmp_path: Path) -> None:
    config = json.loads(
        (ROOT / "configs" / "shared_corpus_parameters_2026.08.12.json").read_text()
    )
    config["requested_records"] = 4
    config["categories"] = {
        "exact_keep": {"count": 1, "gate_label": "KEEP"},
        "hard_keep": {"count": 1, "gate_label": "KEEP"},
        "single_minimal_edit": {"count": 1, "gate_label": "EDIT"},
        "complex_correction": {"count": 1, "gate_label": "EDIT"},
    }
    config["roles_per_category"] = {"development": 1}
    config["counterfactual_pairs"]["minimum_pairs"] = 1
    config["execution"]["progress_interval_records"] = 1
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    coverage_path = tmp_path / "coverage.csv"
    with coverage_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "category",
                "gate_label",
                "phenomenon",
                "requested_count",
                "train_count",
                "development_count",
                "holdout_count",
                "counterfactual_pair_required",
                "primary_risk",
            ],
        )
        writer.writeheader()
        for category, label, phenomenon in (
            ("exact_keep", "KEEP", "simple_regular"),
            ("hard_keep", "KEEP", "weak_borders"),
            ("single_minimal_edit", "EDIT", "extra_split"),
            ("complex_correction", "EDIT", "joint_split_and_merge"),
        ):
            writer.writerow(
                {
                    "category": category,
                    "gate_label": label,
                    "phenomenon": phenomenon,
                    "requested_count": 1,
                    "train_count": 0,
                    "development_count": 1,
                    "holdout_count": 0,
                    "counterfactual_pair_required": str(
                        category in {"hard_keep", "single_minimal_edit"}
                    ).lower(),
                    "primary_risk": "bounded_test",
                }
            )
    schema_path = ROOT / "schemas" / "synthetic_table_record_2026.08.12.5.json"
    output = tmp_path / "shared-corpus-test"
    acceptance = build_shared_corpus(output, config_path, coverage_path, schema_path)
    verified = verify_shared_corpus(output, schema_path, expected_records=4)
    assert acceptance["status"] == verified["status"] == "PASS"
    assert acceptance["complete_counterfactual_pairs"] == 1
    assert (output / "SHA256SUMS").is_file()
