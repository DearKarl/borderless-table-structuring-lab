"""CPU contracts for feature isolation, exported inference and shared assembly."""
import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from .controller import ResponseModel
from .features import FEATURE_NAMES, extract, sample_scales, scaled
from .io_utils import digest
from .train import fit
from .assembly import assemble, targets


class CoreContracts(unittest.TestCase):
    def test_shared_candidates_and_overlap_do_not_mutate_native(self):
        native = {"pdf_info": [{"para_blocks": [
            {"type": "table", "html": "old table", "bbox": [0, 0, 100, 100]},
            {"type": "interline_equation", "content": "old formula", "bbox": [10, 10, 50, 30]},
            {"type": "text", "content": "untouched", "bbox": [0, 110, 100, 140]}]}]}
        original = copy.deepcopy(native)
        candidates = {0: {"accepted": True, "output": "new table"}, 1: {"accepted": True, "output": "new formula"}}
        table, _ = assemble(native, candidates, "DT")
        formula, _ = assemble(native, candidates, "DF")
        combined, audit = assemble(native, candidates, "DTF")
        self.assertEqual(native, original)
        self.assertEqual(table["pdf_info"][0]["para_blocks"][0]["html"], "new table")
        self.assertEqual(formula["pdf_info"][0]["para_blocks"][1]["content"], "new formula")
        self.assertEqual(combined, table)
        self.assertEqual(audit[1]["conflicts_with"], [0])
        self.assertEqual(assemble(native, candidates, "D0")[0], original)

    def test_features_are_finite_without_source_or_labels(self):
        image = Image.new("RGB", (140, 28), "white")
        f = extract(image, "x_i", "equation")
        self.assertEqual(len(f), len(FEATURE_NAMES))
        self.assertTrue(all(math.isfinite(x) for x in f))
        self.assertFalse(any("source" in n or "score" in n for n in FEATURE_NAMES))

    def test_real_pixel_sizes_and_seeded_bounds(self):
        a = sample_scales(12)
        self.assertEqual(a, sample_scales(12))
        self.assertEqual(a[:2], [1, 1.25])
        self.assertEqual(len(a), 8)
        self.assertTrue(all(.5 <= x <= 3 for x in a))
        image = Image.new("RGB", (100, 40), "white")
        self.assertEqual(scaled(image, 1.25).size, (125, 50))
        with self.assertRaises(ValueError):
            scaled(image, 200 / 72 * 1.5)

    def test_deployment_loads_and_matches_fitted_checkpoint(self):
        rng = np.random.default_rng(123)
        rows = []
        for kind in ["equation", "table"]:
            for i in range(64):
                features = rng.normal(size=len(FEATURE_NAMES)).tolist()
                scale = float(rng.uniform(.5, 3))
                rows.append({"kind": kind, "label_valid": True, "features": features,
                             "scale": scale, "gain": features[0] * math.log(scale),
                             "source_group": str(i), "region_id": str(i)})
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            bundle = root / "synthetic_unit_fixture.json"
            bundle.write_text(json.dumps({"provenance": {"scope": "external_only"}, "split": "train", "rows": rows}))
            receipt = fit(bundle, root / "model")
            for kind in ["equation", "table"]:
                self.assertLess(receipt["models"][kind]["deployment_max_abs_error"], 1e-10)
                path = root / "model" / (kind + ".json")
                model = ResponseModel(path, digest(path))
                action = model.choose([0] * len(FEATURE_NAMES), 100, 50, 1000)
                self.assertEqual(action["action"], "no_change")
                self.assertEqual(model.calls, 1)
                self.assertTrue(all(.5 <= s <= 3.0 for s in model.numeric_scales()))
                with self.assertRaises(RuntimeError):
                    ResponseModel(path, "0" * 64)

    def test_training_refuses_non_training_bundles(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path = Path(folder) / "invalid.json"
            path.write_text(json.dumps({"provenance": {"scope": "benchmark"}, "split": "test", "rows": []}))
            with self.assertRaises(ValueError):
                fit(path, Path(folder) / "models")


if __name__ == "__main__":
    unittest.main()
