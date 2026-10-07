"""Read-only deployment of fitted numerical-response trees; no label access."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .features import FEATURE_NAMES


class ResponseModel:
    def __init__(self, checkpoint, expected_sha256):
        raw = Path(checkpoint).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        if self.sha256 != expected_sha256:
            raise RuntimeError("Checkpoint identity differs from policy freeze")
        self.model = json.loads(raw)
        if self.model["features"] != FEATURE_NAMES + ["log_scale"]:
            raise RuntimeError("Checkpoint feature schema mismatch")
        self.calls = 0
        self.rows_predicted = 0

    def predict(self, rows):
        self.calls += 1
        self.rows_predicted += len(rows)
        if any(len(row) != len(FEATURE_NAMES) + 1 for row in rows):
            raise ValueError("Invalid inference feature vector length")
        result = []
        pre = self.model.get("preprocessor")
        if pre:
            rows = [[((x if math.isfinite(x) else median) - center) / scale
                     for x, median, center, scale in zip(row, pre["median"], pre["mean"], pre["scale"])]
                    for row in rows]
        if self.model.get("family") == "mlp":
            import numpy as np
            values = np.asarray(rows, dtype=np.float64)
            if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES) + 1 or not np.isfinite(values).all():
                raise ValueError("Invalid inference feature vector")
            for i, layer in enumerate(self.model["layers"]):
                values = values @ np.asarray(layer["weight"]) + np.asarray(layer["bias"])
                if i < len(self.model["layers"]) - 1:
                    values = np.maximum(values, 0)
            return (values[:, 0] * self.model["target_scale"] + self.model["target_mean"]).tolist()
        for row in rows:
            if len(row) != len(FEATURE_NAMES) + 1 or not all(math.isfinite(x) for x in row):
                raise ValueError("Invalid inference feature vector")
            score = self.model["baseline"]
            for tree in self.model["trees"]:
                index = 0
                while not tree[index]["is_leaf"]:
                    node = tree[index]
                    index = node["left"] if row[node["feature_idx"]] <= node["num_threshold"] else node["right"]
                score += tree[index]["value"]
            result.append(float(score))
        return result

    def numeric_scales(self):
        """The same finite surrogate search is used for GBDT and MLP."""
        lower, upper = math.log(0.5), math.log(3.0)
        candidates = {lower + (upper - lower) * i / 256 for i in range(257)} | {0.0, math.log(1.25)}
        return [min(3.0, max(0.5, math.exp(x))) for x in sorted(candidates)]

    def choose(self, features, width, height, margin, mode="numeric"):
        if mode not in ("numeric", "menu"):
            raise ValueError("Unknown controller mode")
        scales = self.numeric_scales() if mode == "numeric" else [1.0, 1.25]
        # Limits bound realized crop work, without raising any native processor cap.
        candidates = [s for s in scales if 0.5 <= s <= 3.0
                      and max(1, round(width * s)) * max(1, round(height * s)) <= 64000000]
        if not candidates:
            return {"action": "no_change", "reason": "no_feasible_scale", "scale": None}
        gains = self.predict([list(features) + [math.log(s)] for s in candidates])
        index = max(range(len(gains)), key=lambda i: (gains[i], -abs(math.log(candidates[i])), -candidates[i]))
        score, scale = gains[index], candidates[index]
        return {"action": "reread" if score > margin else "no_change",
                "reason": "predicted_gain" if score > margin else "margin",
                "scale": scale if score > margin else None, "predicted_gain": score,
                "margin": margin, "model_sha256": self.sha256,
                "surrogate_candidates": len(candidates), "selection_mode": mode}
