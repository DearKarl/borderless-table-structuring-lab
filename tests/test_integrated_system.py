import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

from btsl.inputs import prepare, validate, valid_id
from btsl.io import Lease, publish, read, sha, write_once
from btsl.pipeline import page, verify_ready
from btsl.evaluate import config, verify_source

ROOT = Path(__file__).resolve().parents[1]
RAW = b"Heading\n<table><tr><td>old</td></tr></table>\nNote\n"
LAYOUT = "<box:0 0 1000 1000><label:table><up>"


class Engine:
    def __init__(self, *results):
        self.results, self.calls = iter(results), []

    def generate(self, image, task):
        self.calls.append((image.size, task))
        value = next(self.results)
        if isinstance(value, Exception):
            raise value
        return {"decoded_text": value, "terminal_eos_present": True, "truncated": False}


def test_coordinated_layout_crop_assembly():
    engine = Engine(LAYOUT, "<fcel>new<nl>")
    result, receipt, record = page(RAW, Image.new("RGB", (100, 60)), "0" * 64, engine)
    assert engine.calls == [((1036, 1036), "layout"), ((100, 60), "table")]
    assert b"old" not in result and b"new" in result
    assert result.startswith(b"Heading\n\nNote\n")
    assert receipt["status"] == "NATIVE_COLLECTION_ASSEMBLED"
    assert len(record["tables"]) == 1


def test_zero_tables_is_complete_not_missing_work():
    engine = Engine("<box:0 0 1000 1000><label:text><up>")
    result, receipt, _ = page(RAW, Image.new("RGB", (100, 60)), "0" * 64, engine)
    assert b"<table>" not in result
    assert receipt["status"] == "NATIVE_COLLECTION_ASSEMBLED"


def test_finish_all_regions_then_fallback_whole_page():
    engine = Engine(LAYOUT + "\n" + LAYOUT, "invalid", "<fcel>new<nl>")
    result, receipt, record = page(RAW, Image.new("RGB", (100, 60)), "0" * 64, engine)
    assert result == RAW and len(engine.calls) == 3 and len(record["tables"]) == 2
    assert receipt["used_native_tables"] == 0


def test_runtime_failure_never_becomes_raw_completion():
    with pytest.raises(RuntimeError, match="OOM"):
        page(RAW, Image.new("RGB", (100, 60)), "0" * 64, Engine(LAYOUT, RuntimeError("OOM")))


def test_exact_consumer_format_gate():
    raw = RAW + b"\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    result, receipt, _ = page(raw, Image.new("RGB", (100, 60)), "0" * 64, Engine(LAYOUT, "<fcel>new<nl>"))
    assert result == raw
    assert receipt["status"] == "EXACT_RAW_ASSEMBLY_ABSTENTION"


def inputs(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "raw").mkdir()
    Image.new("RGB", (50, 30)).save(tmp_path / "images/a.png")
    (tmp_path / "raw/a.md").write_bytes(RAW)
    prepare(tmp_path)
    return tmp_path / "manifest.json"


@pytest.mark.parametrize("fault", ["labels", "escape", "modified"])
def test_source_only_manifest_rejects_unbound_input(tmp_path, fault):
    manifest = inputs(tmp_path)
    value = read(manifest)
    assert len(validate(manifest)) == 1
    if fault == "labels":
        value["pages"][0]["gold"] = "secret"
    elif fault == "escape":
        value["pages"][0]["image"] = "../a.png"
    else:
        (tmp_path / "raw/a.md").write_bytes(b"changed")
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        validate(manifest)


def test_atomic_receipt_and_tamper(tmp_path):
    write_once(tmp_path / "a.md", b"one")
    write_once(tmp_path / "a.md", b"one")
    publish(tmp_path / "READY.json", {"files": {"a.md": sha(tmp_path / "a.md")}})
    verify_ready(tmp_path)
    with pytest.raises(ValueError):
        write_once(tmp_path / "a.md", b"two")
    (tmp_path / "a.md").write_bytes(b"two")
    with pytest.raises(ValueError):
        verify_ready(tmp_path)


def test_single_owner(tmp_path):
    with Lease(tmp_path / "lock"):
        with pytest.raises(RuntimeError):
            with Lease(tmp_path / "lock"):
                pass


def test_frozen_evaluator_and_config():
    manifest = verify_source(ROOT / "evaluation/omnidocbench")
    assert len(manifest["files"]) == 54
    settings = config("/predictions", "/gold")["end2end_eval"]
    assert settings["metrics"]["table"]["metric"] == ["TEDS", "Edit_dist"]
    assert settings["dataset"]["match_method"] == "quick_match"
    assert settings["dataset"]["match_timeout_sec"] == 420


def test_cli_lightweight():
    subprocess.run([sys.executable, "-m", "btsl", "--help"], cwd=ROOT, check=True, capture_output=True)
    subprocess.run([sys.executable, "-c", "import btsl.cli, sys; assert 'torch' not in sys.modules"], cwd=ROOT, check=True)


def test_realistic_safe_identifiers():
    assert valid_id("docstructbench_Character%20Sheet.pdf_2")
    assert valid_id("表格 页面_1")
    assert not valid_id("../escape") and not valid_id("hidden\nline")


def test_evaluation_plan_does_not_invoke_docker_or_read_gold_answers(tmp_path, monkeypatch):
    from btsl.evaluate import evaluate
    run = tmp_path / "run"
    (run / "pages").mkdir(parents=True)
    (run / "pages/a.md").write_bytes(RAW)
    publish(run / "READY.json", {"pages": 1, "files": {"pages/a.md": sha(run / "pages/a.md")}})
    gold = tmp_path / "opaque.json"
    gold.write_bytes(b"intentionally not JSON: planner may hash but not parse answers")
    def forbidden(*args, **kwargs):
        raise AssertionError("Plan-only must never start Docker")
    monkeypatch.setattr(subprocess, "check_output", forbidden)
    result = evaluate(run, gold, tmp_path / "evaluation", "unused", "custom", 1)
    assert result["status"] == "PLAN_ONLY_NO_EVALUATOR_CALL"
    assert not (tmp_path / "evaluation").exists()
    with pytest.raises(ValueError, match="exact frozen Official"):
        evaluate(run, gold, tmp_path / "evaluation", "unused", "official", 1)


def test_evaluator_orchestration_with_fake_container(tmp_path, monkeypatch):
    """Exercise wrapper contracts, not a real Docker/scoring run."""
    from btsl.evaluate import evaluate
    run, output = tmp_path / "run", tmp_path / "evaluation"
    (run / "pages").mkdir(parents=True)
    (run / "pages/a.md").write_bytes(RAW)
    publish(run / "READY.json", {"pages": 1, "files": {"pages/a.md": sha(run / "pages/a.md")}})
    gold = tmp_path / "synthetic.json"
    gold.write_bytes(b"[]")
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: '[{"Id":"sha256:fixture-not-a-real-image"}]')
    def fake_container(command, **kwargs):
        assert command[:3] == ["docker", "run", "--rm"]
        assert command[command.index("--network") + 1] == "none"
        assert (output / "frozen_evaluator/result").is_dir()
        publish(output / "results/candidate_quick_match_metric_result.json", {
            "match_debug": {"page_count": 1}, "table": {"page": {
                "TEDS": {"ALL": 0.25}, "TEDS_structure_only": {"ALL": 0.50}}}})
        return 0
    monkeypatch.setattr(subprocess, "call", fake_container)
    result = evaluate(run, gold, output, "fixture", "custom", 1, execute=True)
    assert result["arms"]["candidate"]["full_table_teds"] == 25
    assert not result["historical_runtime_identity"]
    verify_ready(output)
    with pytest.raises(ValueError, match="never overwrite"):
        evaluate(run, gold, output, "fixture", "custom", 1, execute=True)
