from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

from .synthetic_data import (
    CoverageRequest,
    _generative_identity,
    _gold_table,
    _render_profile,
    _sample_shape,
    _sha256,
    _stable_json,
)


GENERATOR_RELEASE = "2026.08.12.6"
DATASET_RELEASE = "shared-corpus-2026.08.12.2"
PLAN_RELEASE = "latent-plan-2026.08.12.1"
ROLE_ORDER = ("train", "development", "holdout")
CATEGORY_ORDER = (
    "exact_keep",
    "hard_keep",
    "single_minimal_edit",
    "complex_correction",
)
SIGNATURE_FIELDS = (
    "generative_family_key",
    "structure_sha256",
    "geometry_sha256",
    "text_sha256",
    "renderer_sha256",
    "base_seed_sha256",
)


def _load_coverage(path: Path) -> list[CoverageRequest]:
    requests: list[CoverageRequest] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for role in ROLE_ORDER:
                key = f"{role}_count"
                for _ in range(int(row[key])):
                    requests.append(
                        CoverageRequest(
                            category=row["category"],
                            gate_label=row["gate_label"],
                            phenomenon=row["phenomenon"],
                            role=role,
                        )
                    )
    return requests


def _family_specs(requests: Iterable[CoverageRequest]) -> list[dict[str, Any]]:
    request_list = list(requests)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for request in request_list:
        token = {
            "category": request.category,
            "gate_label": request.gate_label,
            "phenomenon": request.phenomenon,
        }
        grouped[request.category].append(token)
    for category, members in grouped.items():
        grouped[category] = _stable_member_order(members, category)
    families: list[dict[str, Any]] = []
    hard = grouped["hard_keep"]
    edit = grouped["single_minimal_edit"]
    if len(hard) != len(edit):
        raise ValueError("counterfactual category totals differ")
    for keep_request, edit_request in _pair_counterfactual_members(
        hard, edit, request_list
    ):
        families.append(
            {
                "members": [keep_request, edit_request],
                "shared_phenomenon": (
                    f"{keep_request['phenomenon']} {edit_request['phenomenon']}"
                ),
            }
        )
    for category in ("exact_keep", "complex_correction"):
        for request in grouped[category]:
            families.append(
                {
                    "members": [request],
                    "shared_phenomenon": request["phenomenon"],
                }
            )
    return families


def _coverage_quotas(
    requests: Iterable[CoverageRequest],
) -> Counter[tuple[str, str, str]]:
    return Counter(
        (request.category, request.phenomenon, request.role)
        for request in requests
    )


def _stable_member_order(
    members: list[dict[str, str]], category: str
) -> list[dict[str, str]]:
    occurrences: Counter[str] = Counter()
    decorated: list[tuple[str, dict[str, str]]] = []
    for member in members:
        phenomenon = member["phenomenon"]
        ordinal = occurrences[phenomenon]
        occurrences[phenomenon] += 1
        key = _sha256(
            _stable_json(
                {
                    "namespace": "component-first-family-pairing-2026.08.12.6",
                    "category": category,
                    "phenomenon": phenomenon,
                    "ordinal": ordinal,
                }
            )
        )
        decorated.append((key, member))
    return [member for _, member in sorted(decorated)]


def _pair_counterfactual_members(
    hard_members: list[dict[str, str]],
    edit_members: list[dict[str, str]],
    requests: Iterable[CoverageRequest],
) -> list[tuple[dict[str, str], dict[str, str]]]:
    hard_by_role: dict[str, list[dict[str, str]]] = defaultdict(list)
    edit_by_role: dict[str, list[dict[str, str]]] = defaultdict(list)
    for request in requests:
        member = {
            "category": request.category,
            "gate_label": request.gate_label,
            "phenomenon": request.phenomenon,
        }
        if request.category == "hard_keep":
            hard_by_role[request.role].append(member)
        elif request.category == "single_minimal_edit":
            edit_by_role[request.role].append(member)
    role_pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    for role in ROLE_ORDER:
        hard = _stable_member_order(hard_by_role[role], "hard_keep")
        edit = _stable_member_order(edit_by_role[role], "single_minimal_edit")
        if len(hard) != len(edit):
            raise ValueError(f"counterfactual counts differ in role {role}")
        role_pairs.extend(zip(hard, edit))
    pair_counts = Counter(
        (
            hard["phenomenon"],
            edit["phenomenon"],
        )
        for hard, edit in role_pairs
    )
    hard_queues: dict[str, deque[dict[str, str]]] = defaultdict(deque)
    edit_queues: dict[str, deque[dict[str, str]]] = defaultdict(deque)
    for member in hard_members:
        hard_queues[member["phenomenon"]].append(member)
    for member in edit_members:
        edit_queues[member["phenomenon"]].append(member)
    pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    for edge, amount in sorted(pair_counts.items()):
        hard_phenomenon, edit_phenomenon = edge
        for _ in range(amount):
            pairs.append(
                (
                    hard_queues[hard_phenomenon].popleft(),
                    edit_queues[edit_phenomenon].popleft(),
                )
            )
    if any(hard_queues.values()) or any(edit_queues.values()):
        raise ValueError("counterfactual pairing did not consume all members")
    return pairs


def _bounded_bipartite_selection(
    edge_counts: Counter[tuple[str, str]],
    left_demands: Counter[str],
    right_demands: Counter[str],
) -> Counter[tuple[str, str]]:
    total = sum(left_demands.values())
    if total != sum(right_demands.values()):
        raise ValueError("counterfactual role margins disagree")
    source = ("source", "")
    sink = ("sink", "")
    capacity: dict[tuple[tuple[str, str], tuple[str, str]], int] = {}
    adjacency: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)

    def add_edge(
        start: tuple[str, str], end: tuple[str, str], amount: int
    ) -> None:
        capacity[(start, end)] = int(amount)
        capacity.setdefault((end, start), 0)
        adjacency[start].add(end)
        adjacency[end].add(start)

    for phenomenon, amount in sorted(left_demands.items()):
        add_edge(source, ("left", phenomenon), amount)
    for (left, right), amount in sorted(edge_counts.items()):
        add_edge(("left", left), ("right", right), amount)
    for phenomenon, amount in sorted(right_demands.items()):
        add_edge(("right", phenomenon), sink, amount)

    flow = 0
    while flow < total:
        parent: dict[tuple[str, str], tuple[str, str] | None] = {source: None}
        queue = deque([source])
        while queue and sink not in parent:
            node = queue.popleft()
            for neighbor in sorted(adjacency[node]):
                if neighbor in parent or capacity[(node, neighbor)] <= 0:
                    continue
                parent[neighbor] = node
                queue.append(neighbor)
        if sink not in parent:
            raise ValueError("counterfactual components cannot meet role quotas")
        amount = total - flow
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            amount = min(amount, capacity[(previous, node)])
            node = previous
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            capacity[(previous, node)] -= amount
            capacity[(node, previous)] += amount
            node = previous
        flow += amount

    selected: Counter[tuple[str, str]] = Counter()
    for edge, available in edge_counts.items():
        left, right = edge
        residual = capacity[(("left", left), ("right", right))]
        selected[edge] = available - residual
    return selected


def _build_duplicate_components(plan: list[dict[str, Any]]) -> None:
    parent = list(range(len(plan)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for field in SIGNATURE_FIELDS:
        first_seen: dict[str, int] = {}
        for index, family in enumerate(plan):
            value = family["realized"][field]
            if value in first_seen:
                union(index, first_seen[value])
            else:
                first_seen[value] = index
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(plan)):
        groups[find(index)].append(index)
    for members in groups.values():
        component_key = _sha256(
            _stable_json(sorted(plan[index]["family_id"] for index in members))
        )
        for index in members:
            plan[index]["duplicate_component_id"] = f"component-{component_key}"
            plan[index]["duplicate_component_size"] = len(members)


def _assign_component_roles(
    plan: list[dict[str, Any]], requests: Iterable[CoverageRequest]
) -> None:
    if any(family.get("duplicate_component_size") != 1 for family in plan):
        raise ValueError("unresolved latent duplicate component before split")
    quotas = _coverage_quotas(requests)
    paired: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    single: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for family in plan:
        if len(family["members"]) == 2:
            by_category = {member["category"]: member for member in family["members"]}
            edge = (
                by_category["hard_keep"]["phenomenon"],
                by_category["single_minimal_edit"]["phenomenon"],
            )
            paired[edge].append(family)
        else:
            member = family["members"][0]
            single[(member["category"], member["phenomenon"])].append(family)
    for families in paired.values():
        families.sort(key=lambda family: family["family_id"])
    remaining = Counter({edge: len(families) for edge, families in paired.items()})
    offsets: Counter[tuple[str, str]] = Counter()
    for role in ROLE_ORDER[:-1]:
        left = Counter(
            {
                phenomenon: amount
                for (category, phenomenon, quota_role), amount in quotas.items()
                if category == "hard_keep" and quota_role == role
            }
        )
        right = Counter(
            {
                phenomenon: amount
                for (category, phenomenon, quota_role), amount in quotas.items()
                if category == "single_minimal_edit" and quota_role == role
            }
        )
        selected = _bounded_bipartite_selection(remaining, left, right)
        for edge, amount in selected.items():
            start = offsets[edge]
            for family in paired[edge][start : start + amount]:
                family["role"] = role
            offsets[edge] += amount
            remaining[edge] -= amount
    final_role = ROLE_ORDER[-1]
    for edge, amount in remaining.items():
        start = offsets[edge]
        for family in paired[edge][start : start + amount]:
            family["role"] = final_role

    for key, families in single.items():
        category, phenomenon = key
        families.sort(key=lambda family: family["family_id"])
        offset = 0
        for role in ROLE_ORDER:
            amount = quotas[(category, phenomenon, role)]
            for family in families[offset : offset + amount]:
                family["role"] = role
            offset += amount
        if offset != len(families):
            raise ValueError(f"single-component role quota mismatch: {key}")
    if any(family.get("role") not in ROLE_ORDER for family in plan):
        raise ValueError("one or more components were not assigned to a role")


def _realized_signatures(
    base_seed: int,
    shared_phenomenon: str,
    gold_category: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    rows, columns = _sample_shape(base_seed, config)
    gold = _gold_table(
        base_seed, rows, columns, shared_phenomenon, gold_category
    )
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
    generative = _generative_identity(
        base_seed,
        rows,
        columns,
        shared_phenomenon,
        gold_category,
        structure_sha256,
    )
    return {
        "rows": rows,
        "columns": columns,
        "generative_family_key": generative["generative_family_key"],
        "generative_components": generative["generative_components"],
        "structure_sha256": structure_sha256,
        "geometry_sha256": _sha256(
            _stable_json([cell["bbox"] for cell in gold["cells"]])
        ),
        "text_sha256": _sha256(
            _stable_json([token["text"] for token in gold["tokens"]])
        ),
        "renderer_sha256": _sha256(
            _stable_json(_render_profile(base_seed, shared_phenomenon))
        ),
        "base_seed_sha256": _sha256(str(base_seed).encode("utf-8")),
    }


def create_latent_plan(
    config: dict[str, Any], coverage_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requests = _load_coverage(coverage_path)
    if len(requests) != int(config["requested_records"]):
        raise ValueError("coverage count does not match configuration")
    family_specs = _family_specs(requests)
    used: dict[str, set[str]] = {field: set() for field in SIGNATURE_FIELDS}
    plan: list[dict[str, Any]] = []
    sample_index = 0
    resample_count = 0
    seed_start = int(config["seed_start"]) + 1_000_000
    for family_index, family in enumerate(family_specs):
        gold_category = (
            "single_minimal_edit"
            if len(family["members"]) == 2
            else family["members"][0]["category"]
        )
        realized: dict[str, Any] | None = None
        base_seed = 0
        family_nonce = 0
        for family_nonce in range(100_000):
            base_seed = seed_start + family_index * 100_000 + family_nonce
            candidate = _realized_signatures(
                base_seed,
                family["shared_phenomenon"],
                gold_category,
                config,
            )
            if all(candidate[field] not in used[field] for field in SIGNATURE_FIELDS):
                realized = candidate
                break
            resample_count += 1
        if realized is None:
            raise RuntimeError(f"unable to isolate latent family {family_index}")
        for field in SIGNATURE_FIELDS:
            used[field].add(realized[field])
        family_id = f"family-{realized['generative_family_key']}"
        counterfactual_group_id = (
            f"counterfactual-{realized['generative_family_key']}"
            if len(family["members"]) == 2
            else None
        )
        members: list[dict[str, Any]] = []
        for member in family["members"]:
            members.append(
                {
                    **member,
                    "sample_index": sample_index,
                    "sample_id": f"{DATASET_RELEASE}-{sample_index:06d}",
                }
            )
            sample_index += 1
        plan.append(
            {
                "plan_release": PLAN_RELEASE,
                "generator_release": GENERATOR_RELEASE,
                "dataset_release": DATASET_RELEASE,
                "family_index": family_index,
                "family_id": family_id,
                "counterfactual_group_id": counterfactual_group_id,
                "role": None,
                "base_seed": base_seed,
                "family_nonce": family_nonce,
                "shared_phenomenon": family["shared_phenomenon"],
                "gold_category": gold_category,
                "members": members,
                "realized": realized,
                "terminal_inputs_used": False,
            }
        )
    _build_duplicate_components(plan)
    _assign_component_roles(plan, requests)
    summary = audit_latent_plan(plan, config, requests)
    summary["resample_count"] = resample_count
    return plan, summary


def audit_latent_plan(
    plan: list[dict[str, Any]],
    config: dict[str, Any],
    requests: Iterable[CoverageRequest] | None = None,
) -> dict[str, Any]:
    if any("duplicate_component_id" not in family for family in plan):
        _build_duplicate_components(plan)
    cross_role_overlaps: list[dict[str, Any]] = []
    for field in SIGNATURE_FIELDS:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for family in plan:
            buckets[family["realized"][field]].append(family)
        for value, members in buckets.items():
            roles = sorted({member["role"] for member in members})
            if len(roles) > 1:
                cross_role_overlaps.append(
                    {
                        "signal": field,
                        "value": value,
                        "roles": roles,
                        "family_ids": [member["family_id"] for member in members[:8]],
                    }
                )
    family_key_roles: dict[str, set[str]] = defaultdict(set)
    component_roles: dict[str, set[str]] = defaultdict(set)
    category_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    category_role_counts: dict[str, Counter[str]] = defaultdict(Counter)
    phenomenon_role_counts: dict[str, Counter[str]] = defaultdict(Counter)
    counterfactual_errors: list[str] = []
    record_count = 0
    for family in plan:
        family_key_roles[family["family_id"]].add(family["role"])
        component_roles[family["duplicate_component_id"]].add(family["role"])
        if family["counterfactual_group_id"] is not None:
            categories = {member["category"] for member in family["members"]}
            if categories != {"hard_keep", "single_minimal_edit"}:
                counterfactual_errors.append(family["family_id"])
        for member in family["members"]:
            record_count += 1
            category_counts[member["category"]] += 1
            role_counts[family["role"]] += 1
            category_role_counts[member["category"]][family["role"]] += 1
            phenomenon_role_counts[
                f"{member['category']}::{member['phenomenon']}"
            ][family["role"]] += 1
    expected_category = {
        key: int(value["count"]) for key, value in config["categories"].items()
    }
    expected_role = {
        role: int(count) * len(CATEGORY_ORDER)
        for role, count in config["roles_per_category"].items()
    }
    expected_phenomenon_role: dict[str, Counter[str]] | None = None
    if requests is not None:
        expected_phenomenon_role = defaultdict(Counter)
        for request in requests:
            expected_phenomenon_role[
                f"{request.category}::{request.phenomenon}"
            ][request.role] += 1
    checks = {
        "record_count": record_count == int(config["requested_records"]),
        "category_counts": dict(category_counts) == expected_category,
        "role_counts": dict(role_counts) == expected_role,
        "phenomenon_role_counts": expected_phenomenon_role is None
        or phenomenon_role_counts == expected_phenomenon_role,
        "family_single_role": all(len(roles) == 1 for roles in family_key_roles.values()),
        "component_single_role": all(
            len(roles) == 1 for roles in component_roles.values()
        ),
        "counterfactual_groups": not counterfactual_errors,
        "cross_role_signature_overlap": not cross_role_overlaps,
        "terminal_nonuse": all(not family["terminal_inputs_used"] for family in plan),
    }
    return {
        "plan_release": PLAN_RELEASE,
        "generator_release": GENERATOR_RELEASE,
        "dataset_release": DATASET_RELEASE,
        "requested_records": int(config["requested_records"]),
        "planned_records": record_count,
        "planned_families": len(plan),
        "category_counts": dict(sorted(category_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "category_role_counts": {
            key: dict(sorted(value.items()))
            for key, value in sorted(category_role_counts.items())
        },
        "phenomenon_role_counts": {
            key: dict(sorted(value.items()))
            for key, value in sorted(phenomenon_role_counts.items())
        },
        "cross_role_overlaps": cross_role_overlaps,
        "counterfactual_errors": counterfactual_errors,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "terminal_inputs_used": False,
        "training": False,
    }


def write_latent_plan(
    output: Path, config_path: Path, coverage_path: Path
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"non-overwrite path already exists: {output}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    plan, summary = create_latent_plan(config, coverage_path)
    output.mkdir(parents=True)
    plan_bytes = "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in plan
    ).encode("utf-8")
    (output / "latent_plan.jsonl").write_bytes(plan_bytes)
    summary["latent_plan_sha256"] = _sha256(plan_bytes)
    summary["config_sha256"] = _sha256(config_path.read_bytes())
    summary["coverage_sha256"] = _sha256(coverage_path.read_bytes())
    summary_bytes = (
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    (output / "latent_plan_audit.json").write_bytes(summary_bytes)
    files = ("latent_plan.jsonl", "latent_plan_audit.json")
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{_sha256((output / name).read_bytes())}  {name}\n" for name in files
        ),
        encoding="utf-8",
    )
    if summary["status"] != "PASS":
        raise RuntimeError(f"latent plan failed: {summary['checks']}")
    return summary


def read_latent_plan(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def verify_latent_plan(
    output: Path, config_path: Path, coverage_path: Path
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for line in (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if _sha256((output / relative).read_bytes()) != expected:
            raise ValueError(f"latent-plan hash mismatch: {relative}")
    plan = read_latent_plan(output / "latent_plan.jsonl")
    independent = audit_latent_plan(plan, config, _load_coverage(coverage_path))
    builder = json.loads((output / "latent_plan_audit.json").read_text(encoding="utf-8"))
    comparable_keys = (
        "requested_records",
        "planned_records",
        "planned_families",
        "category_counts",
        "role_counts",
        "category_role_counts",
        "phenomenon_role_counts",
        "cross_role_overlaps",
        "counterfactual_errors",
        "checks",
        "status",
    )
    disagreements = [
        key for key in comparable_keys if independent[key] != builder[key]
    ]
    independent["builder_verifier_disagreements"] = disagreements
    independent["builder_verifier_agree"] = not disagreements
    independent["status"] = (
        "PASS"
        if independent["status"] == "PASS" and not disagreements
        else "FAIL"
    )
    return independent
