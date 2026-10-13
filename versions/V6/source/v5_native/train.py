"""Legacy synthetic-fixture trainer; use matched_train for authorized research fits."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import sklearn

from .controller import ResponseModel
from .features import FEATURE_NAMES
from .io_utils import atomic_json, digest, utc

PARAMETERS = dict(loss="squared_error", max_iter=100, max_leaf_nodes=15,
                  min_samples_leaf=20, learning_rate=0.05, l2_regularization=1.0,
                  random_state=0, early_stopping=False)


def fit(bundle, output):
    data = json.loads(Path(bundle).read_text(encoding="utf-8"))
    if data["provenance"]["scope"] != "external_only" or data["split"] != "train":
        raise ValueError("Training only accepts an explicitly isolated external train bundle")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"at": utc(), "training_bundle_sha256": digest(bundle), "parameters": PARAMETERS,
               "sklearn_version": sklearn.__version__, "models": {}, "scope": "exploratory_128_region_pilot"}
    for kind in ("equation", "table"):
        rows = [r for r in data["rows"] if r["kind"] == kind and r["label_valid"]]
        if not rows:
            raise ValueError("No usable training rows for " + kind)
        x = np.array([r["features"] + [math.log(r["scale"])] for r in rows], dtype=np.float64)
        y = np.array([r["gain"] for r in rows], dtype=np.float64)
        model = HistGradientBoostingRegressor(**PARAMETERS).fit(x, y)
        trees = []
        for predictors in model._predictors:
            if len(predictors) != 1:
                raise RuntimeError("Unexpected regression tree shape")
            nodes = predictors[0].nodes
            trees.append([{key: node[key].item() for key in
                           ("value", "feature_idx", "num_threshold", "left", "right", "is_leaf")}
                          for node in nodes])
        checkpoint = {"schema_version": 1, "kind": kind, "features": FEATURE_NAMES + ["log_scale"],
                      "baseline": float(model._baseline_prediction[0, 0]), "trees": trees,
                      "parameters": PARAMETERS, "training_bundle_sha256": digest(bundle),
                      "sklearn_version": sklearn.__version__}
        path = output / (kind + ".json")
        atomic_json(path, checkpoint)
        # Validate the actual file-loading deployment implementation, including unseen scales.
        deployed = ResponseModel(path, digest(path))
        probe = np.vstack([x, np.column_stack([x[:, :-1], np.linspace(math.log(.5), math.log(3), len(x))])])
        delta = np.max(np.abs(np.array(deployed.predict(probe.tolist())) - model.predict(probe)))
        if delta > 1e-10:
            raise RuntimeError("Exported deployment predictions differ from sklearn")
        receipt["models"][kind] = {"checkpoint_sha256": digest(path), "usable_rows": len(rows),
                                   "source_groups": len({r["source_group"] for r in rows}),
                                   "regions": len({r["region_id"] for r in rows}),
                                   "deployment_max_abs_error": float(delta),
                                   "deployment_predict_calls": deployed.calls}
    atomic_json(output / "TRAINING_RECEIPT.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-bundle", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(fit(args.train_bundle, args.output), indent=2))
