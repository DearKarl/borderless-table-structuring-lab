"""Frozen pre-reread numeric features; no GT, IDs, scores or candidate outputs."""
from __future__ import annotations

import math
import re

import numpy as np
from PIL import Image

FEATURE_NAMES = ["log_width", "log_height", "log_area", "aspect_ratio",
                 "gray_mean", "gray_std", "ink_fraction", "gradient_mean",
                 "horizontal_ink_runs", "vertical_ink_runs", "native_log_length",
                 "native_repeat_fraction", "native_empty", "native_brace_imbalance",
                 "native_table_rows", "native_table_cells", "native_invalid"]


def extract(image, native_text, kind):
    from .regions import runaway
    # A fixed, aspect-preserving thumbnail bounds CPU work. It never enters OCR.
    gray = image.convert("L")
    gray.thumbnail((512, 512), Image.Resampling.BILINEAR)
    a = np.asarray(gray, dtype=np.float32) / 255
    ink = a < 0.8
    row, col = ink.mean(axis=1) > 0.02, ink.mean(axis=0) > 0.02
    runs = lambda x: float(np.count_nonzero(x & ~np.r_[False, x[:-1]]))
    text = str(native_text or "")
    grams = [text[i:i + 16] for i in range(max(0, len(text) - 15))]
    repeat = 1 - len(set(grams)) / len(grams) if grams else 0
    dx = np.abs(np.diff(a, axis=1)).mean() if a.shape[1] > 1 else 0
    dy = np.abs(np.diff(a, axis=0)).mean() if a.shape[0] > 1 else 0
    values = [math.log1p(image.width), math.log1p(image.height), math.log1p(image.width * image.height),
              image.width / image.height, float(a.mean()), float(a.std()), float(ink.mean()), float(dx + dy),
              runs(row), runs(col), math.log1p(len(text)), repeat, float(not text.strip()),
              abs(text.count("{") - text.count("}")), len(re.findall(r"<tr[\s>]", text)),
              len(re.findall(r"<t[dh][\s>]", text)), float(runaway(text, kind) is not None)]
    assert len(values) == len(FEATURE_NAMES) and all(math.isfinite(v) for v in values)
    return values


def sample_scales(seed, maximum=8):
    rng = np.random.default_rng(seed)
    return [1.0, 1.25] + np.exp(rng.uniform(math.log(0.5), math.log(3.0), maximum - 2)).tolist()


def scaled(image, factor):
    if not 0.5 <= factor <= 3.0:
        raise ValueError("Adaptive scale outside frozen feasible interval")
    size = (max(1, round(image.width * factor)), max(1, round(image.height * factor)))
    return image.copy() if size == image.size else image.resize(size, Image.Resampling.LANCZOS)
