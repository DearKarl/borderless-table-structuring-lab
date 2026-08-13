from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from mpr_tsr_splitmerge_v2.counterfactual import build_candidate
from mpr_tsr_splitmerge_v2.token_geometry import reassign_tokens_by_geometry


LEVELS = {
    "none": 0.0,
    "light": 0.02,
    "medium": 0.08,
    "heavy": 0.20,
}


def _load_groups(path: Path) -> list[dict[str, Any]]:
    groups = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                groups.setdefault(record["pair_group_id"], record)
    return list(groups.values())


def _jitter(tokens: list[dict[str, Any]], fraction: float, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    output = copy.deepcopy(tokens)
    for token in output:
        box = token.get("bbox")
        if not isinstance(box, list) or len(box) != 4:
            continue
        width = float(box[2]) - float(box[0])
        height = float(box[3]) - float(box[1])
        dx = rng.uniform(-fraction, fraction) * width
        dy = rng.uniform(-fraction, fraction) * height
        token["bbox"] = [
            float(box[0]) + dx,
            float(box[1]) + dy,
            float(box[2]) + dx,
            float(box[3]) + dy,
        ]
    return output


def evaluate(records: list[dict[str, Any]]) -> dict[str, Any]:
    results = {}
    for level, fraction in LEVELS.items():
        good_preserved = bad_fixed = rejected = 0
        for record in records:
            gold = record["gold_record"]
            seed = int(hashlib.sha256(f"{record['pair_group_id']}:{level}".encode()).hexdigest()[:16], 16)
            tokens = _jitter(gold["ocr_tokens"], fraction, seed)
            good_candidate = reassign_tokens_by_geometry(gold, tokens)
            if good_candidate is None:
                rejected += 1
            else:
                good_preserved += good_candidate["canonical_table"] == gold["canonical_table"]
            raw_bad = build_candidate(gold, gold, operator="assignment_swap")
            bad_candidate = reassign_tokens_by_geometry(raw_bad, tokens)
            if bad_candidate is None:
                rejected += 1
            else:
                bad_fixed += bad_candidate["canonical_table"] == gold["canonical_table"]
        groups = len(records)
        results[level] = {
            "fraction": fraction,
            "groups": groups,
            "raw_good_preserved": good_preserved,
            "raw_good_preservation": good_preserved / groups if groups else 0.0,
            "raw_bad_fixed": bad_fixed,
            "raw_bad_fix_rate": bad_fixed / groups if groups else 0.0,
            "rejected_candidates": rejected,
        }
    light = results["light"]
    gates = {
        "light_raw_good_preservation_ge_0_995": light["raw_good_preservation"] >= 0.995,
        "light_raw_bad_fix_rate_ge_0_95": light["raw_bad_fix_rate"] >= 0.95,
    }
    return {
        "schema_version": "mpr-tsr/token-geometry-jitter-stress-v1",
        "results": results,
        "gates": gates,
        "status": "PASS" if all(gates.values()) else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stress token-geometry reassignment under preregistered bbox jitter.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(_load_groups(args.input))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "results": report["results"]}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
