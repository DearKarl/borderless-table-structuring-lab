"""Synthetic scientific-contract checks; no model or benchmark access."""
import copy
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from .io_utils import commit_stage, finalize_page, read_json, require_common_runtime
from .regions import crop_of, guard, reread, runaway, typed_spans


def fixture():
    return {"pdf_info": [{"page_size": [100, 100], "para_blocks": [
        {"type": "text", "bbox": [0, 0, 10, 10], "lines": [{"spans": [{"type": "text", "content": "first"}]}]},
        {"type": "interline_equation", "bbox": [10, 10, 20, 20], "content": "x"},
        {"type": "table", "bbox": [20, 20, 40, 40], "html": "<table><tr><td>one</td></tr></table>"},
        {"type": "text", "bbox": [50, 50, 60, 60], "lines": [{"spans": [{"type": "text", "content": "last"}]}]},
    ]}]}


class Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def batch_content_extract(self, images, types):
        self.calls.append((types, [image.size for image in images]))
        return [next(self.responses) for _ in images]


class ContractTests(unittest.TestCase):
    def test_runtime_json_roundtrip_preserves_tuple_grid_but_rejects_change(self):
        expected = {"helper": {"layout_image_size": [1036, 1036], "max_image_edge_ratio": 50}}
        require_common_runtime({"helper": {"layout_image_size": (1036, 1036), "max_image_edge_ratio": 50}}, expected)
        with self.assertRaisesRegex(RuntimeError, "helper"):
            require_common_runtime({"helper": {"layout_image_size": (1024, 1024), "max_image_edge_ratio": 50}}, expected)

    def test_reference_validator_boundary(self):
        for text, kind, expected in [(None, "text", "empty"), ("a" * 255, "text", None),
                ("a" * 256, "text", "repetition"), ("<nl>", "equation", "token_leak"),
                ("<table><tr><td>x</td></tr>", "table", "truncated"),
                ("<table></table>", "table", "no_rows"),
                ("<table><tr><td>x</td></tr></table>", "table", None)]:
            self.assertEqual(runaway(text, kind), expected)

    def test_crop_reference_clamps_only_top_left(self):
        im = Image.new("RGB", (10, 10), "white")
        cropped = crop_of(im, (-2.7, -3.2, 12.9, 13.9))
        self.assertEqual(cropped.size, (12, 13))
        self.assertEqual(cropped.getpixel((11, 12)), (0, 0, 0))

    def test_rejected_reread_keeps_original_and_order(self):
        original = fixture()
        output, changed = reread(original, Image.new("RGB", (100, 100)), Client([None]), "table", 1, lambda _: None)
        self.assertFalse(changed)
        self.assertEqual(output, original)

    def test_formula125_only_updates_formula(self):
        original = fixture()
        client = Client(["y"])
        output, changed = reread(original, Image.new("RGB", (100, 100)), client, "equation", 1.25, lambda _: None)
        self.assertTrue(changed)
        self.assertEqual(client.calls, [(["equation"], [(12, 12)])])
        expected = copy.deepcopy(original)
        expected["pdf_info"][0]["para_blocks"][1]["content"] = "y"
        self.assertEqual(output, expected)
        self.assertEqual(original, fixture())

    def test_guard_bound_scales_first_valid_and_empty_text_exclusion(self):
        original = fixture()
        blocks = original["pdf_info"][0]["para_blocks"]
        blocks[0]["lines"][0]["spans"][0]["content"] = ""
        blocks[1]["content"] = "<nl>"
        client = Client(["<nl>", "", "z"])
        output, changed = guard(original, Image.new("RGB", (100, 100)), client, lambda _: None)
        self.assertTrue(changed)
        self.assertEqual([sizes for _, sizes in client.calls], [[(28, 28)], [(21, 21)], [(42, 42)]])
        self.assertEqual(output["pdf_info"][0]["para_blocks"][1]["content"], "z")
        self.assertEqual(len(typed_spans(output["pdf_info"][0]["para_blocks"])), 2)

    def test_guard_stops_at_first_valid(self):
        original = fixture()
        original["pdf_info"][0]["para_blocks"][1]["content"] = "<nl>"
        client = Client(["z"])
        guard(original, Image.new("RGB", (100, 100)), client, lambda _: None)
        self.assertEqual(len(client.calls), 1)

    def test_exception_does_not_mutate_completed_stage(self):
        original = fixture()
        with self.assertRaises(StopIteration):
            reread(original, Image.new("RGB", (100, 100)), Client([]), "table", 1, lambda _: None)
        self.assertEqual(original, fixture())

    def test_atomic_fallback_and_empty_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            folder = out / "pages" / "p0"
            commit_stage(folder, "native", fixture(), "first\n<table></table>\nlast")
            result = finalize_page(folder, out, "p0", {"status": "timeout"})
            self.assertEqual(result["retained_stage"], "native")
            self.assertEqual((out / "markdown/p0.md").read_text(), "first\n<table></table>\nlast")
            result = finalize_page(out / "pages/p1", out, "p1", {"status": "timeout"})
            self.assertTrue(result["empty_prediction"])
            self.assertEqual((out / "markdown/p1.md").read_bytes(), b"")
            self.assertEqual(len(list((out / "receipts").glob("*.json"))), 2)


if __name__ == "__main__":
    unittest.main()
