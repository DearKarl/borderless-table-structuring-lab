from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Iterable, Iterator


ROLE_VALUES = ("train", "development", "holdout", "terminal")
FAMILY_KEYS = (
    "document_cluster_id",
    "template_family_id",
    "content_family_id",
    "renderer_family_id",
    "generation_seed",
)


def _load(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number}: record is not an object")
            yield value


def _family_value(record: dict[str, Any], key: str) -> Any:
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        return None
    return provenance.get(key)


def audit(records: Iterable[dict[str, Any]], expected_roles: set[str]) -> dict[str, Any]:
    blockers: list[dict[str, Any]] = []
    groups: dict[str, dict[str, Any]] = {}
    sample_ids: set[str] = set()
    duplicate_sample_ids = 0
    record_count = 0
    role_counts: collections.Counter[str] = collections.Counter()
    missing_fields: collections.Counter[str] = collections.Counter()
    family_roles: dict[str, dict[str, set[str]]] = {
        key: collections.defaultdict(set) for key in FAMILY_KEYS
    }
    source_hash_by_family: dict[str, set[str]] = {
        key: set() for key in ("template_family_id", "content_family_id")
    }

    for index, record in enumerate(records):
        record_count += 1
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            blockers.append({"code": "MISSING_SAMPLE_ID", "record_index": index})
        elif sample_id in sample_ids:
            duplicate_sample_ids += 1
        else:
            sample_ids.add(sample_id)
        role = record.get("role")
        if role not in ROLE_VALUES:
            blockers.append({
                "code": "INVALID_ROLE",
                "sample_id": sample_id,
                "role": role,
            })
        else:
            role_counts[role] += 1
        group_id = record.get("pair_group_id")
        if not isinstance(group_id, str) or not group_id:
            blockers.append({"code": "MISSING_PAIR_GROUP_ID", "sample_id": sample_id})
        else:
            group = groups.setdefault(group_id, {
                "size": 0,
                "roles": set(),
                "families": {key: set() for key in FAMILY_KEYS},
            })
            group["size"] += 1
            group["roles"].add(role)
            for key in FAMILY_KEYS:
                group["families"][key].add(_family_value(record, key))
        source_hash = record.get("image", {}).get("source_document_hash")
        for key in FAMILY_KEYS:
            value = _family_value(record, key)
            if value is None or value == "":
                missing_fields[key] += 1
            elif role in ROLE_VALUES:
                family_roles[key][str(value)].add(role)
            if key in source_hash_by_family and value is not None and source_hash is not None:
                if str(value) == str(source_hash):
                    source_hash_by_family[key].add(str(value))
        provenance = record.get("provenance")
        if not isinstance(provenance, dict) or provenance.get("terminal_data_used") is not False:
            blockers.append({"code": "TERMINAL_DATA_FLAG_NOT_FALSE", "sample_id": sample_id})

    if duplicate_sample_ids:
        blockers.append({"code": "DUPLICATE_SAMPLE_ID", "count": duplicate_sample_ids})

    missing_roles = sorted(expected_roles - set(role_counts))
    if missing_roles:
        blockers.append({"code": "EXPECTED_ROLE_MISSING", "roles": missing_roles})

    for key, count in missing_fields.items():
        if count:
            blockers.append({"code": "FAMILY_FIELD_MISSING", "field": key, "records": count})

    for key, values in family_roles.items():
        crossed = {value: sorted(roles) for value, roles in values.items() if len(roles) > 1}
        if crossed:
            blockers.append({
                "code": "FAMILY_CROSSES_ROLES",
                "field": key,
                "values": crossed,
            })
        if expected_roles and len(values) < len(expected_roles):
            blockers.append({
                "code": "INSUFFICIENT_FAMILY_CARDINALITY",
                "field": key,
                "distinct_values": len(values),
                "required_roles": len(expected_roles),
            })

    for key, values in source_hash_by_family.items():
        if values:
            blockers.append({
                "code": "FAMILY_ALIASES_SOURCE_DOCUMENT",
                "field": key,
                "values": len(values),
            })

    group_sizes = collections.Counter(group["size"] for group in groups.values())
    inconsistent_groups = []
    for group_id, group in groups.items():
        if len(group["roles"]) != 1 or any(
            len(snapshot) != 1 for snapshot in group["families"].values()
        ):
            inconsistent_groups.append(group_id)
    if inconsistent_groups:
        blockers.append({
            "code": "PAIR_GROUP_SPLIT_OR_INCONSISTENT",
            "groups": inconsistent_groups,
        })

    return {
        "schema_version": "mpr-tsr/raw-preserving-split-audit-v1",
        "records": record_count,
        "pair_groups": len(groups),
        "role_counts": dict(role_counts),
        "group_size_histogram": dict(group_sizes),
        "missing_family_fields": dict(missing_fields),
        "family_cardinality": {
            key: len(values) for key, values in family_roles.items()
        },
        "family_cross_role_values": {
            key: {
                value: sorted(roles)
                for value, roles in values.items()
                if len(roles) > 1
            }
            for key, values in family_roles.items()
        },
        "source_hash_alias_counts": {
            key: len(values) for key, values in source_hash_by_family.items()
        },
        "expected_roles": sorted(expected_roles),
        "missing_expected_roles": missing_roles,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "status": "PASS" if not blockers else "BLOCKED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed Raw-preserving family split audit.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--expected-role",
        action="append",
        choices=ROLE_VALUES,
        default=["train", "development", "holdout"],
    )
    args = parser.parse_args()
    report = audit(_load(args.input), set(args.expected_role))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "records": report["records"],
        "pair_groups": report["pair_groups"],
        "blocker_count": report["blocker_count"],
    }, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
