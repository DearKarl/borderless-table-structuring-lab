"""Aggregate publication consistency only; no model, dataset or evaluator call."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "artifacts/results/native-tables"


def load(name):
    return json.loads((RESULT / name).read_text())


def test_exact_paired_summary_ready_binding():
    summary_bytes = (RESULT / "PAIRED_TABLE_SUMMARY.json").read_bytes()
    ready_bytes = (RESULT / "READY.json").read_bytes()
    aggregate = load("AGGREGATE.json")
    assert hashlib.sha256(summary_bytes).hexdigest() == "5ccecac587dc99528e6e0881453c682dac7fc72089ed2d9901f18b9b064b1542"
    assert hashlib.sha256(ready_bytes).hexdigest() == "15069ed43f1d89dd562fa307e07f95c36e68efa403383304f48436d086ad5e4d"
    assert json.loads(ready_bytes)["summary_sha256"] == aggregate["evidence"]["paired_summary_sha256"]


def test_full_metric_scope_and_truthful_result_projection():
    summary, aggregate = load("PAIRED_TABLE_SUMMARY.json"), load("AGGREGATE.json")
    assert summary["pages"] == aggregate["pages"] == 1651
    for arm in ("baseline", "candidate"):
        value = summary["arms"][arm]
        assert value["TEDS"]["source_key"] == "table.page.TEDS.ALL"
        assert value["TEDS"]["table_page_denominator"] == 458
        assert value["evaluator_counts"] == {"sample_count": 665, "error_case_count": 0, "timeout_case_count": 0}
    full = summary["arms"]["candidate"]["TEDS"]["percent_0_to_100"]
    structure = summary["arms"]["candidate"]["TEDS_structure"]["percent_0_to_100"]
    assert full == aggregate["full_table_teds"]
    assert structure - full == aggregate["structure_minus_full_gap_points"]
    assert aggregate["scored_local_target"]["achieved"] == (full > 95.0)
    assert aggregate["public_leaderboard_acceptance_verified"] is False
    assert aggregate["boundaries"]["portable_package_separately_scored"] is False


def test_public_result_projection_contains_no_private_paths_or_payloads():
    for name in ("PAIRED_TABLE_SUMMARY.json", "READY.json", "AGGREGATE.json"):
        text = (RESULT / name).read_text()
        for forbidden in ("/Users/", "/home/", "210.28.", "decoded_text", "ground_truth", "private_key", "password"):
            assert forbidden not in text
