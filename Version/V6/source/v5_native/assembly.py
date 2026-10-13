"""Independent ablation assembly from one immutable native parse and candidate bank."""
from __future__ import annotations

import copy


def targets(middle):
    result = []
    def walk(value, path):
        if isinstance(value, dict):
            kind, key = ("table", "html") if value.get("type") == "table" else ("equation", "content")
            if value.get("type") in ("table", "interline_equation") and key in value and "bbox" in value:
                result.append({"index": len(result), "path": path, "kind": kind, "key": key,
                               "bbox": list(value["bbox"]), "native": str(value.get(key) or "")})
            for name, child in value.items():
                if name != "dapp":
                    walk(child, path + [name])
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, path + [index])
    walk(middle["pdf_info"][0]["para_blocks"], ["pdf_info", 0, "para_blocks"])
    return result


def overlap(a, b):
    ax0, ay0, ax1, ay1 = a["bbox"]
    bx0, by0, bx1, by1 = b["bbox"]
    area = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    nested = a["path"] == b["path"][:len(a["path"])] or b["path"] == a["path"][:len(b["path"])]
    return area > 0 or nested


def assemble(middle, candidates, arm):
    """Table priority in DTF; then larger area and native traversal order.

    Only accepted, content-changing candidates compete. DT and DF are assembled
    independently from native. DTF reuses their exact candidates. A suppressed
    conflict is recorded rather than applying two replacements to shared content.
    """
    allowed = {"D0": set(), "DT": {"table"}, "DF": {"equation"}, "DTF": {"table", "equation"}}[arm]
    regions = targets(middle)
    selected = []
    audit = []
    def priority(row):
        x0, y0, x1, y1 = row["bbox"]
        return (row["kind"] != "table", -max(0, x1 - x0) * max(0, y1 - y0), row["index"])
    for region in sorted(regions, key=priority):
        candidate = candidates.get(region["index"])
        if region["kind"] not in allowed or not candidate or not candidate["accepted"] or candidate["output"] == region["native"]:
            continue
        conflicts = [old["index"] for old in selected if overlap(region, old)]
        audit.append({"index": region["index"], "kind": region["kind"], "applied": not conflicts,
                      "conflicts_with": conflicts, "rule": "table_then_larger_area_then_native_order"})
        if not conflicts:
            selected.append(region)
    result = copy.deepcopy(middle)
    for region in selected:
        obj = result
        for element in region["path"]:
            obj = obj[element]
        obj[region["key"]] = candidates[region["index"]]["output"]
    return result, audit
