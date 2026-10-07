"""Matched CPU response regression with explicit external validation and exports.

The small neural regressor uses NumPy float64 and a specified Adam optimizer.
No OCR model, image loader, benchmark binding or test bundle is accepted here.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from .controller import ResponseModel
from .features import FEATURE_NAMES
from .io_utils import atomic_json, digest, utc

GBDT = dict(loss="squared_error", max_iter=100, max_leaf_nodes=15,
            min_samples_leaf=20, learning_rate=.05, l2_regularization=1.,
            random_state=0, early_stopping=False)
MLP = dict(hidden_layers=[64, 32], activation="relu", output="linear", optimizer="Adam",
           learning_rate=.001, weight_decay=.0001, beta1=.9, beta2=.999, epsilon=1e-8,
           batch_size=128, max_epochs=200, patience=20, seed=0,
           initialization="Glorot_uniform_weights_zero_bias",
           weight_decay_scope="coupled_L2_all_weights_and_biases",
           validation_min_delta=0., device="cpu", dtype="float64")


def load_bundle(path, split):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data["provenance"]["scope"] != "external_only" or data["split"] != split:
        raise ValueError("Expected an isolated external " + split + " bundle")
    if data["provenance"].get("unit_contract") != "native_predicted_region_v1":
        raise ValueError("Unverified label-to-native-region unit contract")
    return data


def arrays(rows):
    if not rows or len({r["kind"] for r in rows}) != 1:
        raise ValueError("A nonempty single-task set is required")
    counts = Counter(r["region_id"] for r in rows)
    identities = [(r["region_id"], r["realized_identity"]) for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Realized-view aliases must not get duplicate training weight")
    if any(not .5 <= r["scale"] <= 3 for r in rows):
        raise ValueError("Scale outside the frozen interval")
    x = np.asarray([r["features"] + [math.log(r["scale"])] for r in rows], dtype=np.float64)
    y = np.asarray([r["gain"] for r in rows], dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) + 1 or not np.isfinite(y).all():
        raise ValueError("Invalid response rows")
    w = np.asarray([1 / counts[r["region_id"]] for r in rows], dtype=np.float64)
    return x, y, w


def preprocessing(x, w):
    median = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=0)
    if not np.isfinite(median).all():
        raise ValueError("Entirely missing training feature")
    clean = np.where(np.isfinite(x), x, median)
    mean = np.average(clean, axis=0, weights=w)
    scale = np.sqrt(np.average((clean - mean) ** 2, axis=0, weights=w))
    scale[scale < 1e-12] = 1
    return {"median": median.tolist(), "mean": mean.tolist(), "scale": scale.tolist(),
            "fitted_on": "train_only", "moment_weighting": "equal_total_weight_per_region"}


def transform(x, pre):
    return (np.where(np.isfinite(x), x, pre["median"]) - pre["mean"]) / pre["scale"]


def initialize(dim, rng):
    layers = []
    for a, b in zip([dim, 64, 32], [64, 32, 1]):
        bound = math.sqrt(6 / (a + b))
        layers.extend([rng.uniform(-bound, bound, (a, b)), np.zeros(b)])
    return layers


def forward(parameters, x):
    a = [x]
    z = []
    for i in range(0, len(parameters), 2):
        value = a[-1] @ parameters[i] + parameters[i + 1]
        z.append(value)
        a.append(np.maximum(value, 0) if i < len(parameters) - 2 else value)
    return a[-1][:, 0], a, z


def loss_gradient(parameters, x, y, w):
    pred, activations, logits = forward(parameters, x)
    residual = pred - y
    loss = float(np.sum(w * residual ** 2) / np.sum(w))
    delta = (2 * w * residual / np.sum(w))[:, None]
    gradients = [None] * len(parameters)
    for layer in range(len(logits) - 1, -1, -1):
        gradients[2 * layer] = activations[layer].T @ delta
        gradients[2 * layer + 1] = delta.sum(axis=0)
        if layer:
            delta = (delta @ parameters[2 * layer].T) * (logits[layer - 1] > 0)
    return loss, gradients


def mse(parameters, x, y, w):
    return float(np.average((forward(parameters, x)[0] - y) ** 2, weights=w))


def fit_mlp(x, y, w, vx, vy, vw, deadline, parameters=None):
    config = dict(MLP if parameters is None else parameters)
    rng = np.random.default_rng(config["seed"])
    net = initialize(x.shape[1], rng)
    first = [np.zeros_like(p) for p in net]
    second = [np.zeros_like(p) for p in net]
    best, best_loss, best_epoch, stale, step = None, float("inf"), None, 0, 0
    history = []
    for epoch in range(1, config["max_epochs"] + 1):
        if time.monotonic() > deadline:
            raise TimeoutError("Controller fit exceeded the two-hour ceiling")
        order = rng.permutation(len(x))
        for start in range(0, len(x), config["batch_size"]):
            indices = order[start:start + config["batch_size"]]
            _, gradients = loss_gradient(net, x[indices], y[indices], w[indices])
            step += 1
            for i, gradient in enumerate(gradients):
                gradient = gradient + config["weight_decay"] * net[i]
                first[i] = config["beta1"] * first[i] + (1 - config["beta1"]) * gradient
                second[i] = config["beta2"] * second[i] + (1 - config["beta2"]) * gradient ** 2
                corrected1 = first[i] / (1 - config["beta1"] ** step)
                corrected2 = second[i] / (1 - config["beta2"] ** step)
                net[i] -= config["learning_rate"] * corrected1 / (np.sqrt(corrected2) + config["epsilon"])
        train_loss, val_loss = mse(net, x, y, w), mse(net, vx, vy, vw)
        if not math.isfinite(train_loss + val_loss):
            raise ValueError("Nonfinite MLP training loss")
        history.append({"epoch": epoch, "train_weighted_mse": train_loss, "validation_weighted_mse": val_loss})
        if val_loss < best_loss - config["validation_min_delta"]:
            best, best_loss, best_epoch, stale = copy.deepcopy(net), val_loss, epoch, 0
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    return best, {"history": history, "selected_epoch": best_epoch, "best_validation_weighted_mse": best_loss,
                  "epochs_completed": epoch, "early_stopped": stale >= config["patience"]}


def trees(model):
    return [[{key: node[key].item() for key in
              ("value", "feature_idx", "num_threshold", "left", "right", "is_leaf")}
             for node in predictors[0].nodes] for predictors in model._predictors]


def fit(train_bundle, validation_bundle, output, phase_events):
    train, validation = load_bundle(train_bundle, "train"), load_bundle(validation_bundle, "validation")
    if {r["source_group"] for r in train["rows"]} & {r["source_group"] for r in validation["rows"]}:
        raise ValueError("Source groups overlap train and validation")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"at": utc(), "device": "cpu", "scope": "matched_external_response_regression",
               "train_sha256": digest(train_bundle), "validation_sha256": digest(validation_bundle),
               "numpy_version": np.__version__, "sklearn_version": sklearn.__version__, "models": {}}
    def event(name, family, kind, **extra):
        with open(phase_events, "a", encoding="utf-8") as f:
            f.write(json.dumps({"at": utc(), "event": name, "family": family, "kind": kind,
                                "device": "cpu", **extra}) + "\n")
    for kind in ("equation", "table"):
        rows = [r for r in train["rows"] if r["kind"] == kind and r["label_valid"]]
        vrows = [r for r in validation["rows"] if r["kind"] == kind and r["label_valid"]]
        raw, y, w = arrays(rows); vraw, vy, vw = arrays(vrows)
        pre = preprocessing(raw, w); x, vx = transform(raw, pre), transform(vraw, pre)
        target_mean = float(np.average(y, weights=w))
        target_scale = max(float(np.sqrt(np.average((y-target_mean)**2, weights=w))), 1e-12)
        identity = {"schema_version": 2, "kind": kind, "features": FEATURE_NAMES + ["log_scale"],
                    "preprocessor": pre, "training_bundle_sha256": digest(train_bundle),
                    "validation_bundle_sha256": digest(validation_bundle),
                    "training_region_weights": "1 / distinct realized views for each region",
                    "usable_rows": len(rows), "regions": len({r["region_id"] for r in rows}),
                    "training_row_order_sha256": __import__('hashlib').sha256(json.dumps(
                        [(r["region_id"],r["realized_identity"]) for r in rows]).encode()).hexdigest()}
        for family in ("gbdt", "mlp"):
            started = time.monotonic(); deadline = started + 2 * 3600
            event("controller_fit_started", family, kind, rows=len(rows))
            def hard_stop():
                event("controller_fit_failed", family, kind, error="Hard two-hour CPU fit ceiling")
                os._exit(124)
            timer = threading.Timer(2 * 3600, hard_stop)
            timer.daemon = True
            timer.start()
            try:
                if family == "gbdt":
                    model = HistGradientBoostingRegressor(**GBDT).fit(x, y, sample_weight=w)
                    checkpoint = {**identity, "family": family, "parameters": GBDT,
                                  "baseline": float(model._baseline_prediction[0, 0]), "trees": trees(model)}
                    expected = model.predict(vx)
                    log = {"train_weighted_mse": float(np.average((model.predict(x)-y)**2, weights=w)),
                           "validation_weighted_mse": float(np.average((expected-vy)**2, weights=vw))}
                else:
                    net, log = fit_mlp(x, (y-target_mean)/target_scale, w,
                                      vx, (vy-target_mean)/target_scale, vw, deadline)
                    checkpoint = {**identity, "family": family, "parameters": MLP,
                                  "target_mean": target_mean, "target_scale": target_scale,
                                  "layers": [{"weight":net[i].tolist(), "bias":net[i+1].tolist()}
                                             for i in range(0,len(net),2)]}
                    expected = forward(net, vx)[0] * target_scale + target_mean
                if time.monotonic() > deadline:
                    raise TimeoutError("Controller fit exceeded the two-hour ceiling")
                path = output / (family + "_" + kind + ".json")
                atomic_json(path, checkpoint)
                deployed = ResponseModel(path, digest(path))
                delta = float(np.max(np.abs(np.asarray(deployed.predict(vraw.tolist()))-expected)))
                if delta > 1e-10:
                    raise RuntimeError("Saved/loaded deployment differs from fitted model")
                atomic_json(output / (family + "_" + kind + "_LOG.json"), log)
                receipt["models"][family + "_" + kind] = {"sha256":digest(path),"usable_rows":len(rows),
                    "regions":identity["regions"],"elapsed_seconds":time.monotonic()-started,
                    "deployment_max_abs_error":delta,"training_row_order_sha256":identity["training_row_order_sha256"]}
                event("controller_fit_completed", family, kind, **receipt["models"][family + "_" + kind])
            except BaseException as exc:
                event("controller_fit_failed", family, kind, error=type(exc).__name__+': '+str(exc))
                raise
            finally:
                timer.cancel()
    atomic_json(output / "TRAINING_RECEIPT.json", receipt)
    return receipt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-bundle", required=True)
    ap.add_argument("--validation-bundle", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--phase-events", required=True)
    args = ap.parse_args()
    print(json.dumps(fit(args.train_bundle, args.validation_bundle, args.output, args.phase_events), indent=2))
