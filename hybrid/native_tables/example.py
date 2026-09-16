"""Create a synthetic model-free assembly example; never an inference result."""
import argparse
import json
from pathlib import Path

from .assembly import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = b"Synthetic heading\n<table><tr><td>old</td></tr></table>\nSynthetic note\n"
    image = b"P3\n2 2\n255\n255 255 255 255 255 255\n255 255 255 255 255 255\n"
    record = {"schema": "native-table-page-input/1", "status": "GENERATIONS_COMPLETE",
              "raw_markdown_sha256": digest(raw), "source_image_sha256": digest(image),
              "layout": {"decoded_text": "<box:0 0 1000 1000><label:table><up>",
                         "terminal_eos_present": True, "truncated": False},
              "tables": [{"layout_index": 0,
                          "decoded_text": "<fcel>Item<fcel>Total<nl><fcel>A<fcel>10<nl>",
                          "terminal_eos_present": True, "truncated": False}],
              "example_only": True}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "raw.md").write_bytes(raw)
    (args.output_dir / "page.ppm").write_bytes(image)
    (args.output_dir / "native.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
