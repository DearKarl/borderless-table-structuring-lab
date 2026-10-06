"""Reference-free NJU-Scale policy. Only load bundles from a trusted local writer.

Hashes detect corruption, not a malicious replacement of both pickle and manifest.
The feature API intentionally cannot accept references or source/split metadata.
Image/PDF preparation is supplied by the existing input-feature integration.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
import re
import sys

from .schema import ACTION_CODES, FEATURES

SEED = 20261003
FEATURE_DEFINITION = "v4_input_features_candidate_001"
PARAMETERS = dict(loss="squared_error", learning_rate=.05, max_iter=100,
                  max_leaf_nodes=7, min_samples_leaf=10, l2_regularization=1,
                  random_state=SEED, early_stopping=False, warm_start=False)
BINDING_KEYS = ("source_sha256", "model_sha256", "processor_sha256",
                "actions_sha256", "score_sha256", "plan_sha256", "addendum_sha256")
MODEL_FILE = "nju_scale_gbdt_v0_1.pkl"
MAX_JSON = 16 * 1024 * 1024
MAX_MODEL = 64 * 1024 * 1024


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def is_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def read_bytes(path, limit):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("Missing, linked or excessive bundle/input file: " + str(path))
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("File exceeded read limit")
    return data


def read_json(path):
    return json.loads(read_bytes(path, MAX_JSON))


def validate_bindings(bindings):
    if not isinstance(bindings, dict) or any(not is_hash(bindings.get(k)) for k in BINDING_KEYS):
        raise ValueError("Verified bindings required: " + ", ".join(BINDING_KEYS))
    # No arbitrary audit text or private metadata in deployment configuration.
    return {key: bindings[key] for key in BINDING_KEYS}


def valid_vector(action, vector):
    return (isinstance(vector, (list, tuple)) and len(vector) == len(FEATURES)
            and all(finite(v) for v in vector)
            and vector[FEATURES.index("action_code")] == ACTION_CODES[action]
            and vector[FEATURES.index("is_original_pdf")] in (0, 1))


def fallback_action(fixed, available):
    if fixed in available:
        return fixed
    if "A" in available:
        return "A"
    return next(iter(sorted(available)), None)


class ScalePolicy:
    """One fitted estimator and a frozen conservative operational policy."""

    def __init__(self, estimator, config, identity):
        self.estimator = estimator
        self.config = config
        self.identity = identity

    def predict(self, action_features):
        available = sorted(set(action_features) & set(ACTION_CODES)) if isinstance(action_features, dict) else []
        fixed = fallback_action(self.config["fixed_action"], available)

        def result(action, reason, gains=None):
            return dict(action=action, reason=reason, predicted_gains=gains or {},
                        policy_identity=self.identity)

        if not available:
            return result(None, "no_available_actions")
        if set(action_features) != set(available):
            return result(fixed, "invalid_action_keys")
        if any(not valid_vector(a, action_features[a]) for a in available):
            return result(fixed, "missing_or_invalid_features")
        pdf_flags = {action_features[a][FEATURES.index("is_original_pdf")] for a in available}
        if len(pdf_flags) != 1 or ("C" in available and pdf_flags != {1}):
            return result(fixed, "inconsistent_source_features")
        if any(action_features[a][FEATURES.index("glyph_reliability")] != 1 for a in available):
            return result(fixed, "unreliable_features")
        values = self.estimator.predict([action_features[a] for a in available])
        gains = {a: float(v) for a, v in zip(available, values)}
        if len(gains) != len(available) or not all(math.isfinite(v) for v in gains.values()):
            return result(fixed, "nonfinite_prediction")
        if not self.config["learned_enabled"]:
            return result(fixed, "fixed_policy_default", gains)
        best = max(available, key=lambda a: gains[a])  # Stable A/B/C tie order.
        if gains[best] > self.config["margin"]:
            return result(best, "learned_gain", gains)
        return result(fixed, "below_margin", gains)


def verified_bundle(directory, *, expected_manifest_sha256=None):
    """Read/check bytes without unpickling. Accept output root or policy/ itself."""
    root = Path(directory)
    if root.is_symlink():
        raise ValueError("Linked policy directory")
    if (root / "policy").is_dir():
        root = root / "policy"
    if root.is_symlink():
        raise ValueError("Linked policy directory")
    manifest_raw = read_bytes(root / "manifest.json", MAX_JSON)
    identity = sha256(manifest_raw)
    if expected_manifest_sha256 is not None and identity != expected_manifest_sha256:
        raise ValueError("Manifest hash mismatch")
    manifest = json.loads(manifest_raw)
    if manifest.get("schema") != "nju_scale_bundle_v1" or manifest.get("trusted_local_only") is not True:
        raise ValueError("Unsupported or untrusted bundle contract")
    if set(manifest.get("files", {})) != {MODEL_FILE, "config.json"}:
        raise ValueError("Unexpected bundle inventory")
    blobs = {}
    for name, expected in manifest["files"].items():
        if not is_hash(expected):
            raise ValueError("Invalid bundle hash")
        blobs[name] = read_bytes(root / name, MAX_MODEL if name == MODEL_FILE else MAX_JSON)
        if sha256(blobs[name]) != expected:
            raise ValueError("Bundle file hash mismatch: " + name)
    config = json.loads(blobs["config.json"])
    if (config.get("schema") != "nju_scale_policy_v1"
            or config.get("feature_names") != list(FEATURES)
            or config.get("feature_definition_id") != FEATURE_DEFINITION
            or config.get("parameters") != PARAMETERS
            or config.get("fixed_action") not in ACTION_CODES
            or type(config.get("learned_enabled")) is not bool
            or not finite(config.get("margin")) or config["margin"] < 0
            or not is_hash(config.get("roster_sha256"))):
        raise ValueError("Invalid policy schema/configuration")
    validate_bindings(config.get("bindings"))
    return root, identity, config, blobs[MODEL_FILE]


def load_policy(directory, *, expected_manifest_sha256=None):
    """Load a trusted-local sklearn pickle after integrity and version checks."""
    _, identity, config, model_bytes = verified_bundle(
        directory, expected_manifest_sha256=expected_manifest_sha256)
    import sklearn
    from sklearn.ensemble import HistGradientBoostingRegressor
    if config.get("sklearn_version") != sklearn.__version__:
        raise ValueError("sklearn version differs from fitted bundle")
    estimator = pickle.loads(model_bytes)
    if (type(estimator) is not HistGradientBoostingRegressor
            or estimator.n_features_in_ != len(FEATURES)
            or estimator.n_iter_ != PARAMETERS["max_iter"]
            or any(estimator.get_params().get(k) != v for k, v in PARAMETERS.items())):
        raise ValueError("Estimator is unfitted or differs from frozen contract")
    return ScalePolicy(estimator, config, identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="Trusted-local policy bundle directory")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--deny-read", action="append", default=[], help=argparse.SUPPRESS)
    args = parser.parse_args()
    # Used by the fresh-process delivery check: input is feature-only JSON stdin.
    denied = [Path(p).resolve() for p in args.deny_read]

    def guard(event, values):
        if event == "open" and isinstance(values[0], (str, bytes)):
            p = Path(values[0].decode() if isinstance(values[0], bytes) else values[0]).resolve()
            if any(p == d or d in p.parents for d in denied):
                raise PermissionError("Reference/label store inaccessible")
        if event == "import" and values[0].endswith(("public_fit", "public_score", "public_outcomes")):
            raise PermissionError("Host-only module unavailable in prediction process")

    sys.addaudithook(guard)
    # Prove the guard is active before making predictions, not just configured.
    blocked = 0
    for path in denied:
        try:
            with open(path / "__denied_probe__", "rb"):
                pass
        except PermissionError:
            blocked += 1
    request = json.loads(sys.stdin.read(MAX_JSON + 1))
    policy = load_policy(args.directory, expected_manifest_sha256=args.expected_manifest_sha256)
    predictions = [policy.predict(row) for row in request["action_features"]]
    print(json.dumps(dict(predictions=predictions, denied_paths_verified=blocked,
                          policy_identity=policy.identity), allow_nan=False))


if __name__ == "__main__":
    main()
