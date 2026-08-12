from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .synthetic_data import (
    CoverageRequest,
    _expand_coverage,
    _hamming,
    _license_manifest,
    _make_record,
    _perceptual_hash,
    _sha256,
    _stable_json,
    _validate_record_contract,
)


GENERATOR_RELEASE = "2026.08.12.5"
DATASET_RELEASE = "shared-corpus-2026.08.12"
SCHEMA_RELEASE = "synthetic-table-record-2026.08.12.5"
ZERO_SHA256 = "0" * 64


def _prepare_assignments(
    requests: list[CoverageRequest],
) -> dict[int, tuple[int, int | None, str | None]]:
    grouped: dict[str, list[CoverageRequest]] = defaultdict(list)
    for request in requests:
        grouped[request.category].append(request)
    assignments: dict[int, tuple[int, int | None, str | None]] = {}
    base_family_index = 0
    hard = grouped["hard_keep"]
    single = grouped["single_minimal_edit"]
    if len(hard) != len(single):
        raise ValueError("counterfactual category counts do not match")
    for pair_index, (keep_request, edit_request) in enumerate(zip(hard, single)):
        if keep_request.role != edit_request.role:
            raise ValueError("counterfactual pair roles do not match")
        shared = f"{keep_request.phenomenon} {edit_request.phenomenon}"
        assignments[id(keep_request)] = (base_family_index, pair_index, shared)
        assignments[id(edit_request)] = (base_family_index, pair_index, shared)
        base_family_index += 1
    for category in ("exact_keep", "complex_correction"):
        for request in grouped[category]:
            assignments[id(request)] = (base_family_index, None, None)
            base_family_index += 1
    return assignments


def _compact_signature(record: dict[str, Any], image: bytes) -> dict[str, Any]:
    gold = record["gold"]
    structure = {
        "rows": gold["rows"],
        "columns": gold["columns"],
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
    structure_sha256 = _sha256(_stable_json(structure))
    text_sha256 = _sha256(
        _stable_json([token["text"] for token in gold["tokens"]])
    )
    geometry_sha256 = _sha256(
        _stable_json([cell["bbox"] for cell in gold["cells"]])
    )
    return {
        "sample_id": record["sample_id"],
        "role": record["role"],
        "image_sha256": record["payloads"]["image_sha256"],
        "pixel_sha256": record["rendering"]["normalized_pixel_sha256"],
        "perceptual_hash": _perceptual_hash(image),
        "structure_sha256": structure_sha256,
        "text_sha256": text_sha256,
        "geometry_sha256": geometry_sha256,
        "table_sha256": _sha256(
            _stable_json(
                {
                    "structure": structure_sha256,
                    "text": text_sha256,
                    "geometry": geometry_sha256,
                }
            )
        ),
        "families": {
            key: record["identity"][key]
            for key in (
                "document_cluster_id",
                "source_family_id",
                "template_family_id",
                "content_family_id",
                "renderer_family_id",
            )
        },
    }


def _audit_signatures(signatures: list[dict[str, Any]]) -> dict[str, Any]:
    exact: list[dict[str, Any]] = []
    near: list[dict[str, Any]] = []
    fields = (
        "image_sha256",
        "pixel_sha256",
        "table_sha256",
        "structure_sha256",
        "text_sha256",
        "geometry_sha256",
    )
    duplicate_statistics: dict[str, int] = {}
    for field in fields:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in signatures:
            buckets[item[field]].append(item)
        duplicate_statistics[field] = sum(
            len(members) for members in buckets.values() if len(members) > 1
        )
        for value, members in buckets.items():
            roles = sorted({member["role"] for member in members})
            if len(roles) > 1 and field in (
                "image_sha256",
                "pixel_sha256",
                "table_sha256",
            ):
                exact.append(
                    {
                        "signal": field,
                        "value": value,
                        "roles": roles,
                        "sample_ids": [member["sample_id"] for member in members[:8]],
                    }
                )
    for family_key in next(iter(signatures))["families"] if signatures else ():
        buckets: dict[str, set[str]] = defaultdict(set)
        for item in signatures:
            buckets[item["families"][family_key]].add(item["role"])
        for value, roles in buckets.items():
            if len(roles) > 1:
                exact.append(
                    {
                        "signal": family_key,
                        "value": value,
                        "roles": sorted(roles),
                    }
                )
    # The near-overlap policy requires either identical normalized text or an
    # identical structure-and-geometry pair.  Blocking on those required
    # supporting signals is exhaustive and avoids an unsafe all-pairs scan.
    supporting_buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in signatures:
        supporting_buckets[("text", item["text_sha256"])].append(item)
        supporting_buckets[
            ("structure_geometry", item["structure_sha256"], item["geometry_sha256"])
        ].append(item)
    seen_near_pairs: set[tuple[str, str]] = set()
    for members in supporting_buckets.values():
        by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for member in members:
            by_role[member["role"]].append(member)
        roles = sorted(by_role)
        for role_index, first_role in enumerate(roles):
            for second_role in roles[role_index + 1 :]:
                for first in by_role[first_role]:
                    for second in by_role[second_role]:
                        pair = tuple(sorted((first["sample_id"], second["sample_id"])))
                        if pair in seen_near_pairs:
                            continue
                        distance = _hamming(
                            first["perceptual_hash"], second["perceptual_hash"]
                        )
                        supporting = [
                            field
                            for field in (
                                "text_sha256",
                                "structure_sha256",
                                "geometry_sha256",
                            )
                            if first[field] == second[field]
                        ]
                        if distance <= 2 and (
                            "text_sha256" in supporting
                            or all(
                                field in supporting
                                for field in (
                                    "structure_sha256",
                                    "geometry_sha256",
                                )
                            )
                        ):
                            seen_near_pairs.add(pair)
                            near.append(
                                {
                                    "first": first["sample_id"],
                                    "second": second["sample_id"],
                                    "distance": distance,
                                    "supporting_signals": supporting,
                                }
                            )
    return {
        "release": GENERATOR_RELEASE,
        "dataset_release": DATASET_RELEASE,
        "signature_count": len(signatures),
        "cross_role_exact_overlaps": exact,
        "cross_role_unresolved_near_overlaps": near,
        "within_and_cross_role_duplicate_signature_members": duplicate_statistics,
        "status": "PASS" if not exact and not near else "FAIL",
    }


def build_shared_corpus(
    output: Path,
    config_path: Path,
    coverage_path: Path,
    schema_path: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"non-overwrite path already exists: {output}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    requests = _expand_coverage(coverage_path)
    if len(requests) != config["requested_records"]:
        raise ValueError("coverage count does not match configuration")
    output.mkdir(parents=True)
    for name in ("images", "manifests", "records", "audits", "reports", "quarantine"):
        (output / name).mkdir()

    input_hashes = {
        "config_sha256": _sha256(config_path.read_bytes()),
        "coverage_sha256": _sha256(coverage_path.read_bytes()),
        "schema_sha256": _sha256(schema_path.read_bytes()),
    }
    requested_records = int(config["requested_records"])
    (output / "PREREGISTRATION.md").write_text(
        "# Shared Corpus Build Preregistration 2026.08.12\n\n"
        f"This CPU-first run requests exactly {requested_records:,} new synthetic records. "
        "It performs no model training and no benchmark evaluation. Every "
        "record must pass the frozen schema, Canonical, replay, ownership, "
        "geometry, source/license, pairing, isolation, and checksum gates. "
        "A failed seed is not replaced.\n",
        encoding="utf-8",
    )
    run_config = {
        "release": GENERATOR_RELEASE,
        "dataset_release": DATASET_RELEASE,
        "purpose": "FROZEN_SHARED_CORPUS_BUILD",
        "training": False,
        "benchmark_evaluation": False,
        "terminal_inputs_used": False,
        "output_path": str(output.resolve()),
        **input_hashes,
    }
    (output / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "input_manifest.json").write_text(
        json.dumps(
            {
                "release": GENERATOR_RELEASE,
                "inputs": [
                    {"path": str(config_path), "sha256": input_hashes["config_sha256"]},
                    {"path": str(coverage_path), "sha256": input_hashes["coverage_sha256"]},
                    {"path": str(schema_path), "sha256": input_hashes["schema_sha256"]},
                ],
                "terminal_inputs": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    license_manifest = _license_manifest()
    license_manifest["release"] = GENERATOR_RELEASE
    license_bytes = (
        json.dumps(license_manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    license_sha = _sha256(license_bytes)
    (output / "manifests" / "source_license.json").write_bytes(license_bytes)

    assignments = _prepare_assignments(requests)
    validator = Draft202012Validator(schema)
    category_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    pair_members: dict[str, dict[str, Any]] = {}
    signatures: list[dict[str, Any]] = []
    manifest_hasher = hashlib.sha256()
    role_hasher = hashlib.sha256()
    manifest_path = output / "records" / "manifest.jsonl"
    role_path = output / "manifests" / "role_assignment.jsonl"
    with manifest_path.open("wb") as manifest, role_path.open("wb") as role_file:
        for sample_index, request in enumerate(requests):
            family_index, pair_index, shared = assignments[id(request)]
            record, image = _make_record(
                request,
                sample_index,
                family_index,
                pair_index,
                config,
                license_sha,
                schema,
                shared,
                dataset_release=DATASET_RELEASE,
                schema_release=SCHEMA_RELEASE,
                generator_release=GENERATOR_RELEASE,
                shape_shuffle_seed=2026081205,
            )
            replay, replay_image = _make_record(
                request,
                sample_index,
                family_index,
                pair_index,
                config,
                license_sha,
                schema,
                shared,
                dataset_release=DATASET_RELEASE,
                schema_release=SCHEMA_RELEASE,
                generator_release=GENERATOR_RELEASE,
                shape_shuffle_seed=2026081205,
            )
            if record != replay:
                raise ValueError(f"NONDETERMINISTIC_SEMANTICS: {record['sample_id']}")
            if image != replay_image:
                raise ValueError(f"NONDETERMINISTIC_PIXELS: {record['sample_id']}")
            _validate_record_contract(record)
            (output / record["payloads"]["image_path"]).write_bytes(image)
            line = json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
            manifest.write(line)
            manifest_hasher.update(line)
            role_line = (
                json.dumps(
                    {
                        "sample_id": record["sample_id"],
                        "role": record["role"],
                        "template_family_id": record["identity"]["template_family_id"],
                        "content_family_id": record["identity"]["content_family_id"],
                        "renderer_family_id": record["identity"]["renderer_family_id"],
                        "counterfactual_pair_id": record["identity"]["counterfactual_pair_id"],
                    },
                    sort_keys=True,
                ).encode("utf-8")
                + b"\n"
            )
            role_file.write(role_line)
            role_hasher.update(role_line)
            category_counts[record["category"]] += 1
            role_counts[record["role"]] += 1
            signatures.append(_compact_signature(record, image))
            pair_id = record["identity"]["counterfactual_pair_id"]
            if pair_id is not None:
                state = pair_members.setdefault(
                    pair_id,
                    {
                        "count": 0,
                        "role": record["role"],
                        "gold": record["gold"]["full_state_sha256"],
                        "image": record["payloads"]["image_sha256"],
                        "valid": True,
                    },
                )
                state["count"] += 1
                state["valid"] = state["valid"] and all(
                    (
                        state["role"] == record["role"],
                        state["gold"] == record["gold"]["full_state_sha256"],
                        state["image"] == record["payloads"]["image_sha256"],
                    )
                )
            if (sample_index + 1) % config["execution"]["progress_interval_records"] == 0:
                (output / "reports" / "progress.json").write_text(
                    json.dumps(
                        {
                            "release": GENERATOR_RELEASE,
                            "dataset_release": DATASET_RELEASE,
                            "generated": sample_index + 1,
                            "requested": len(requests),
                            "failed": 0,
                            "training": False,
                            "terminal_inputs_used": False,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )

    overlap = _audit_signatures(signatures)
    (output / "audits" / "overlap_report.json").write_text(
        json.dumps(overlap, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "audits" / "signatures.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in signatures),
        encoding="utf-8",
    )
    complete_pairs = sum(
        state["count"] == 2 and state["valid"] for state in pair_members.values()
    )
    expected_categories = {
        key: value["count"] for key, value in config["categories"].items()
    }
    expected_roles = {
        role: count * 4 for role, count in config["roles_per_category"].items()
    }
    checks = {
        "count": sum(category_counts.values()) == config["requested_records"],
        "categories": dict(category_counts) == expected_categories,
        "roles": dict(role_counts) == expected_roles,
        "pairs": complete_pairs >= config["counterfactual_pairs"]["minimum_pairs"],
        "overlap": overlap["status"] == "PASS",
        "terminal_nonuse": True,
    }
    acceptance = {
        "release": GENERATOR_RELEASE,
        "dataset_release": DATASET_RELEASE,
        "requested": len(requests),
        "generated": sum(category_counts.values()),
        "accepted": sum(category_counts.values()),
        "quarantined": 0,
        "failed": 0,
        "category_counts": dict(sorted(category_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "complete_counterfactual_pairs": complete_pairs,
        "manifest_sha256": manifest_hasher.hexdigest(),
        "role_assignment_sha256": role_hasher.hexdigest(),
        "overlap_status": overlap["status"],
        "terminal_inputs_used": False,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    (output / "reports" / "acceptance_report.json").write_text(
        json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "quarantine" / "quarantine.jsonl").write_text("", encoding="utf-8")
    if acceptance["status"] != "PASS":
        raise RuntimeError(f"shared corpus build failed: {checks}")
    files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{_sha256(path.read_bytes())}  {path.relative_to(output)}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    return acceptance


def verify_shared_corpus(
    output: Path, schema_path: Path, expected_records: int = 40000
) -> dict[str, Any]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    count = 0
    with (output / "records" / "manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            validator.validate(record)
            _validate_record_contract(record)
            hashed = copy.deepcopy(record)
            expected_record_sha = hashed["payloads"]["record_sha256"]
            hashed["payloads"]["record_sha256"] = ZERO_SHA256
            if _sha256(_stable_json(hashed)) != expected_record_sha:
                raise ValueError(f"record hash mismatch: {record['sample_id']}")
            image_path = output / record["payloads"]["image_path"]
            if _sha256(image_path.read_bytes()) != record["payloads"]["image_sha256"]:
                raise ValueError(f"image hash mismatch: {record['sample_id']}")
            count += 1
    if count != expected_records:
        raise ValueError(f"record count mismatch: {count} != {expected_records}")
    for line in (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected_sha, relative_path = line.split("  ", 1)
        if _sha256((output / relative_path).read_bytes()) != expected_sha:
            raise ValueError(f"sealed payload hash mismatch: {relative_path}")
    acceptance = json.loads(
        (output / "reports" / "acceptance_report.json").read_text(encoding="utf-8")
    )
    if acceptance["status"] != "PASS":
        raise ValueError("shared corpus acceptance is not PASS")
    return acceptance
