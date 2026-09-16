import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from hybrid.native_tables.assemble import assemble_record
from hybrid.native_tables.assembly import digest
from hybrid.native_tables.formats import parse_otsl
from hybrid.native_tables.verify_model import verify

ROOT = Path(__file__).resolve().parents[1]
RAW = "Heading繁體\n<table><tr><td>old</td></tr></table>\nNote\n".encode()
IMAGE = b"synthetic user-owned byte binding"


def record():
    return {"schema": "native-table-page-input/1", "status": "GENERATIONS_COMPLETE",
            "source_image_sha256": digest(IMAGE), "raw_markdown_sha256": digest(RAW),
            "layout": {"decoded_text": "<box:0 0 1000 1000><label:table><up>",
                       "terminal_eos_present": True, "truncated": False},
            "tables": [{"layout_index": 0, "decoded_text": "<fcel>new & safer<nl>",
                        "terminal_eos_present": True, "truncated": False}]}


def test_complete_collection_and_non_table_bytes():
    output, receipt = assemble_record(RAW, digest(IMAGE), record())
    assert output == "Heading繁體\n\nNote\n\n\n<table><tr><td>new &amp; safer</td></tr></table>\n".encode()
    assert receipt["status"] == "NATIVE_COLLECTION_ASSEMBLED"
    assert receipt["policy_details"]["structure_may_change"]
    assert not receipt["policy_details"]["original_table_interleaving_preserved"]


def test_successful_zero_tables_is_not_fallback():
    data = record()
    data["layout"]["decoded_text"] = "<box:0 0 1000 1000><label:text><up>"
    data["tables"] = []
    output, receipt = assemble_record(RAW, digest(IMAGE), data)
    assert output == "Heading繁體\n\nNote\n".encode()
    assert receipt["used_native_tables"] == 0
    assert receipt["status"] == "NATIVE_COLLECTION_ASSEMBLED"


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_order", "resource", "wrong_hash"])
def test_incomplete_work_or_binding_error_is_not_raw_completion(change):
    data = record()
    if change == "missing": data["tables"] = []
    if change == "duplicate": data["tables"] *= 2
    if change == "wrong_order": data["tables"][0]["layout_index"] = 2
    if change == "resource": data["status"] = "RESOURCE_PAUSED"
    if change == "wrong_hash": data["source_image_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        assemble_record(RAW, digest(IMAGE), data)


@pytest.mark.parametrize("fault", ["bad_otsl", "truncated", "no_eos", "bad_layout"])
def test_invalid_completed_page_returns_exact_raw(fault):
    data = record()
    if fault == "bad_otsl": data["tables"][0]["decoded_text"] = "<lcel><nl>"
    if fault == "truncated": data["tables"][0]["truncated"] = True
    if fault == "no_eos": data["tables"][0]["terminal_eos_present"] = False
    if fault == "bad_layout": data["layout"]["decoded_text"] = "not native layout"
    output, receipt = assemble_record(RAW, digest(IMAGE), data)
    assert output == RAW
    assert receipt["used_native_tables"] == 0


def test_failed_second_table_does_not_keep_first():
    data = record()
    data["layout"]["decoded_text"] += "\n<box:0 0 1000 1000><label:table><up>"
    second = copy.deepcopy(data["tables"][0])
    second.update(layout_index=1, decoded_text="<xcel><nl>")
    data["tables"].append(second)
    output, receipt = assemble_record(RAW, digest(IMAGE), data)
    assert output == RAW
    assert receipt["used_native_tables"] == 0


def test_duplicate_raw_literal_is_ambiguous_and_preserved():
    raw = RAW + b"<table><tr><td>old</td></tr></table>"
    data = record()
    data["raw_markdown_sha256"] = digest(raw)
    output, receipt = assemble_record(raw, digest(IMAGE), data)
    assert output == raw
    assert receipt["status"] == "EXACT_RAW_ASSEMBLY_ABSTENTION"


def test_owner_grid_spans_and_no_padding():
    good = parse_otsl("<fcel>A<lcel><nl><ucel><xcel><nl>", terminal_eos_present=True)
    assert good["valid"]
    assert 'rowspan="2" colspan="2"' in good["html"]
    assert not parse_otsl("<fcel>A<lcel><nl><fcel>B<nl>")["valid"]


def test_model_verifier_streams_size_and_hash(tmp_path):
    file = tmp_path / "model.safetensors"
    file.write_bytes(b"synthetic tensors, not a real model")
    manifest = {"files": [{"path": file.name, "bytes": file.stat().st_size,
                           "sha256": digest(file.read_bytes())}]}
    assert verify(tmp_path, manifest) == 1
    file.write_bytes(b"wrong")
    with pytest.raises(ValueError): verify(tmp_path, manifest)


def test_parser_byte_identity_and_manifest_scope():
    root = ROOT / "hybrid/native_tables"
    assert digest((root / "formats.py").read_bytes()) == "40fed19f79ed3e82a8176712ca2224aa88571bf783c457d19d30a5ad87bfcc78"
    manifest = json.loads((root / "model_manifest.json").read_text())
    assert len(manifest["files"]) == 14
    assert manifest["revision"] == "710ea2e26d794fe89cbf3ece0402707c332a8671"
    architecture = json.loads((root / "architecture.json").read_text())
    assert architecture["evaluation_snapshot"]["full_table_teds"] is None
    assert architecture["generation"]["max_new_tokens"] == 4096


def test_cli_synthetic_roundtrip_no_overwrite_no_torch(tmp_path):
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    subprocess.run([sys.executable, "-m", "hybrid.native_tables.example",
                    "--output-dir", str(input_dir)], cwd=ROOT, check=True)
    command = [sys.executable, "-m", "hybrid.native_tables.assemble",
               "--raw", str(input_dir / "raw.md"), "--image", str(input_dir / "page.ppm"),
               "--native", str(input_dir / "native.json"), "--output-dir", str(output_dir)]
    subprocess.run(command, cwd=ROOT, check=True)
    ready = json.loads((output_dir / "READY.json").read_text())
    receipt = json.loads((output_dir / "receipt.json").read_text())
    assert receipt["official_consumer_format_verified"] is False
    assert ready["page_sha256"] == hashlib.sha256((output_dir / "page.md").read_bytes()).hexdigest()
    assert subprocess.run(command, cwd=ROOT, capture_output=True).returncode != 0
    subprocess.run([sys.executable, "-c", "import sys; import hybrid.native_tables.assemble; assert 'torch' not in sys.modules"], cwd=ROOT, check=True)
