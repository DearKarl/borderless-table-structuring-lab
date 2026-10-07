"""CPU-only default-loader/capture check; not an OCR functional smoke."""
import argparse
import hashlib
import time
from pathlib import Path

from .io_utils import atomic_json, read_json, utc
from .native import native_page, pixel_identity
from .regions import region_crop


class ObserveOnlyPredictor:
    def __init__(self):
        self.calls = 0
        self.input_identity = None

    def batch_two_step_extract(self, images):
        # The native logger divides by an inference duration rounded to 0.01 s.
        # An observation-only stub must not produce an artificial zero duration.
        time.sleep(0.02)
        self.calls += 1
        if len(images) != 1:
            raise ValueError("Expected one default-rendered image")
        self.input_identity = pixel_identity(images[0])
        return [[]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rows = read_json(args.manifest)["rows"]
    output = Path(args.output)
    observations = []
    # Four predetermined samples from each task, fixed before any inference.
    for row in [r for r in rows if int(r["region_id"].rsplit("-", 1)[1]) in (0, 16, 32, 48)]:
        predictor = ObserveOnlyPredictor()
        middle, markdown, image = native_page(predictor, Path(args.inputs) / row["file"], output / row["region_id"])
        identity = pixel_identity(image)
        assert predictor.calls == 1 and identity == predictor.input_identity
        assert max(image.size) <= 3501  # PDFium ceil rounding can add one pixel.
        page = middle["pdf_info"][0]
        box = [0, 0, *page["page_size"]]
        crop = region_crop(image, page, {"bbox": box}, 1.0)
        assert pixel_identity(crop) == identity
        observations.append({"region_id": row["region_id"], "image": identity,
                             "capture_matches_consumed_image": True, "whole_frame_crop_identical": True,
                             "native_predictor_calls": predictor.calls, "native_markdown_empty": markdown == ""})
    atomic_json(output / "CPU_CAPTURE_CHECK.json", {"at": utc(), "gpu_used": False,
                "recognition": "Observation-only predictor returning no blocks; real 16-page OCR smoke still required",
                "passed": len(observations) == 8, "observations": observations})
    print("CPU_CAPTURE_CHECK", len(observations), "passed")


if __name__ == "__main__":
    main()
