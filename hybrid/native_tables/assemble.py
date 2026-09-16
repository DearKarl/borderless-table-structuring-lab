"""Validate and assemble one user-owned page; never run a model or score."""
import argparse
import json
from pathlib import Path

from .assembly import AssemblyAbstain, TableSpans, assemble_page, digest
from .formats import parse_layout, parse_otsl

SCHEMA = "native-table-page-input/1"


def raw_records(raw):
    text = raw.decode("utf-8", errors="strict")
    return [{"table_id": f"raw-{i}", "raw_html": text[start:end],
             "raw_html_sha256": digest(text[start:end].encode("utf-8")),
             "start_char": start, "end_char": end,
             "mapping_reason": "UNIQUE_EXACT_RAW_SPAN"}
            for i, (start, end) in enumerate(TableSpans(text).spans)]


def generation(value):
    if (not isinstance(value, dict) or not isinstance(value.get("decoded_text"), str)
            or type(value.get("terminal_eos_present")) is not bool
            or type(value.get("truncated")) is not bool):
        raise ValueError("MISSING_OR_INVALID_GENERATION_RECORD_NOT_ABSTENTION")
    return {"terminal_eos_present": value["terminal_eos_present"],
            "truncated": value["truncated"]}


def assemble_record(raw, image_sha256, record):
    """Only completed generations may lead to a prospective page abstention.

    A missing table generation, unknown/global error or input hash mismatch raises
    without a result. Invalid completed native syntax returns the entire Raw.
    """
    if (record.get("schema") != SCHEMA or record.get("status") != "GENERATIONS_COMPLETE"
            or record.get("raw_markdown_sha256") != digest(raw)
            or record.get("source_image_sha256") != image_sha256):
        raise ValueError("INPUT_BINDING_OR_COMPLETION_INVALID_NOT_ABSTENTION")
    layout = record.get("layout")
    layout_flags = generation(layout)
    tables = record.get("tables")
    if not isinstance(tables, list):
        raise ValueError("MISSING_TABLE_GENERATIONS_NOT_ABSTENTION")
    for table in tables:
        if not isinstance(table, dict) or type(table.get("layout_index")) is not int:
            raise ValueError("INVALID_TABLE_GENERATION_RECORD")
        generation(table)
    parsed_layout = parse_layout(layout["decoded_text"], **layout_flags)
    if not parsed_layout["valid"]:
        return assemble_page(raw, [], "PAGE_ABSTAIN_EXACT_RAW", [],
                             "INVALID_COMPLETED_LAYOUT: " + str(parsed_layout["error"]))
    indices = [b["index"] for b in parsed_layout["blocks"] if b["type"] == "table"]
    if [t["layout_index"] for t in tables] != indices:
        raise ValueError("MISSING_EXTRA_OR_REORDERED_TABLE_GENERATIONS_NOT_ABSTENTION")
    native = []
    for table in tables:
        parsed = parse_otsl(table["decoded_text"], **generation(table))
        if not parsed["valid"]:
            return assemble_page(raw, [], "PAGE_ABSTAIN_EXACT_RAW", [],
                                 "INVALID_COMPLETED_OTSL: " + str(parsed["error"]))
        native.append({"layout_index": table["layout_index"], "html": parsed["html"],
                       "html_sha256": digest(parsed["html"].encode("utf-8"))})
    try:
        originals = raw_records(raw)
    except (AssemblyAbstain, UnicodeError) as exc:
        return assemble_page(raw, [], "PAGE_ABSTAIN_EXACT_RAW", [],
                             "INVALID_RAW_TABLE_COLLECTION: " + str(exc))
    return assemble_page(raw, originals, "NATIVE_PAGE_COMPLETE", native)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw, image = args.raw.read_bytes(), args.image.read_bytes()
    record = json.loads(args.native.read_text(encoding="utf-8"))
    final, receipt = assemble_record(raw, digest(image), record)
    receipt.update(schema="native-table-portable-assembly/1",
                   source_image_sha256=digest(image),
                   native_record_sha256=digest(args.native.read_bytes()),
                   model_executed=False, scored=False,
                   official_consumer_format_verified=False,
                   scope="User-owned outputs; not a verified benchmark result")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "page.md").open("xb") as stream:
        stream.write(final)
    with (args.output_dir / "receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    # Completion marker is last. Existing output directories are never reused.
    with (args.output_dir / "READY.json").open("x", encoding="utf-8") as stream:
        json.dump({"page_sha256": digest(final), "receipt_sha256": digest(
            (args.output_dir / "receipt.json").read_bytes())}, stream)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "scored": False}))


if __name__ == "__main__":
    main()
