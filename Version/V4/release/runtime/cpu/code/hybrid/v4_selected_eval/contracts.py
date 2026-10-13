"""Strict deployment identities; no reference/outcome reads during inference."""
import json
from dataclasses import dataclass
import hashlib
from pathlib import Path
import pickle
from hybrid.v4_input_selector.scale_policy import MODEL_FILE, PARAMETERS, ScalePolicy, verified_bundle
from hybrid.v4_input_selector.schema import FEATURES
from hybrid.v4_input_selector.worker_contract import bound_file, file_sha, safe_id

MODEL_SHA = '78a8454b8d763f622e5d0ea9caa43695207a20c07461ef36e905753054a42518'
ORIGINAL_MANIFEST_SHA = '7b4d0487724c7f1f1f37b33c39528862442cf9e498e4fac51e60d4db68f5aa61'
ORIGINAL_CONFIG_SHA = 'e67f20276b71b076a968946afbbf66803c39c3bcfe982dd91bd1f194739975d9'
MARGINS = (0.0, 0.0009133843864069113, 0.0018267687728138226)
REVISION = 'd3c045014fac170a5af53d953a6d72071a023c60'
DATASET = 'allenai/olmOCR-mix-1025'


def checked_json(path, expected):
    if file_sha(path) != expected:
        raise ValueError('Evidence hash mismatch: ' + str(path))
    return json.loads(Path(path).read_bytes())


def source_files(root):
    root = Path(root)
    return sorted(p for package in ('v4_input_selector', 'v4_selected_eval')
                  for p in (root/'hybrid'/package).rglob('*.py')
                  if 'tests' not in p.parts and not p.name.startswith('test_'))


def verify_code(root, code_files):
    actual = {p.relative_to(root).as_posix(): file_sha(p) for p in source_files(root)}
    if actual != code_files:
        raise ValueError('Code inventory or hash differs from frozen calibration code')


@dataclass
class PreparedPolicy:
    """Verified bytes retained across guard installation; consumed exactly once."""
    identity: str
    config_json: bytes
    model_bytes: bytes | None
    deployment_json: bytes | None = None


def policy_from_memory(prepared):
    """No file reads here. Native entry invokes this only after load_native()."""
    if not isinstance(prepared, PreparedPolicy) or prepared.model_bytes is None:
        raise ValueError('Verified model bytes missing or already consumed')
    raw = prepared.model_bytes
    prepared.model_bytes = None  # Never retry an attempted model construction.
    if hashlib.sha256(raw).hexdigest() != MODEL_SHA:
        raise ValueError('Verified in-memory model bytes changed')
    config = json.loads(prepared.config_json)
    import sklearn
    from sklearn.ensemble import HistGradientBoostingRegressor
    if config.get('sklearn_version') != sklearn.__version__:
        raise ValueError('sklearn version differs from fitted bundle')
    estimator = pickle.loads(raw)
    if (type(estimator) is not HistGradientBoostingRegressor
            or estimator.n_features_in_ != len(FEATURES)
            or estimator.n_iter_ != PARAMETERS['max_iter']
            or any(estimator.get_params().get(k) != v for k, v in PARAMETERS.items())):
        raise ValueError('Estimator is unfitted or differs from frozen contract')
    policy = ScalePolicy(estimator, config, prepared.identity)
    if prepared.deployment_json is not None:
        policy.deployment_binding = json.loads(prepared.deployment_json)
    return policy


def prepare_frozen_model(directory):
    root, identity, config, raw = verified_bundle(directory, expected_manifest_sha256=ORIGINAL_MANIFEST_SHA)
    if (hashlib.sha256(raw).hexdigest() != MODEL_SHA or file_sha(root/'config.json') != ORIGINAL_CONFIG_SHA
            or config['fixed_action'] != 'B' or config['sklearn_version'] != '1.9.1'
            or config['learned_enabled'] is not False):
        raise ValueError('Original frozen bundle differs')
    return PreparedPolicy(identity, json.dumps(config).encode('utf-8'), raw)


def frozen_model(directory):
    # CPU-only callers retain their existing interface; native entry splits it.
    return policy_from_memory(prepare_frozen_model(directory))


def prepare_adaptive_model(directory, manifest_sha, certificate, certificate_sha, *, code_root):
    root, identity, config, raw = verified_bundle(directory, expected_manifest_sha256=manifest_sha)
    if (hashlib.sha256(raw).hexdigest() != MODEL_SHA or config['learned_enabled'] is not True
            or config['fixed_action'] != 'B' or config['margin'] not in MARGINS
            or config['sklearn_version'] != '1.9.1'):
        raise ValueError('Adaptive startup requires the frozen model and a passed policy')
    cert = checked_json(certificate, certificate_sha)
    binding = config.get('raster_calibration')
    if (cert.get('schema') != 'v4_raster_calibration_certificate_v1'
            or cert.get('accepted') is not True or cert.get('valid_paired_pages') != 8
            or cert.get('model_sha256') != MODEL_SHA or cert.get('fixed_action') != 'B'
            or cert.get('margin') != config['margin'] or binding != {
                'certificate_sha256': certificate_sha, 'plan_sha256': cert.get('plan_sha256'),
                'evidence_sha256': cert.get('evidence_sha256')}
            or not cert['improvement'] > cert['tolerance']
            or cert['added_terminal_failures'] != 0
            or not cert['selected_seconds'] <= 1.1 * cert['baseline_seconds']):
        raise ValueError('Missing or invalid calibration certificate binding')
    verify_code(Path(code_root), cert['code_files'])
    deployment = dict(model_sha256=MODEL_SHA, config_sha256=file_sha(root/'config.json'),
        manifest_sha256=identity, calibration_sha256=certificate_sha, code_files=cert['code_files'],
        rounding_evidence_sha256=cert.get('rounding_evidence_sha256'))
    return PreparedPolicy(identity, json.dumps(config).encode('utf-8'), raw,
                          json.dumps(deployment).encode('utf-8'))


def load_adaptive(directory, manifest_sha, certificate, certificate_sha, *, code_root):
    return policy_from_memory(prepare_adaptive_model(directory, manifest_sha, certificate,
                              certificate_sha, code_root=code_root))


def selected_items(pages):
    """Input-only top-level manifest: exactly one stable item per page, no arms."""
    if not 1 <= len(pages) <= 1651:
        raise ValueError('Single-version page cap exceeded')
    seen, items = set(), []
    for p in pages:
        pid = p['page_id']
        if not safe_id(pid) or pid in seen:
            raise ValueError('Duplicate or unsafe page identity')
        seen.add(pid)
        if set(p) != {'page_id', 'file', 'input_sha256', 'source'}:
            raise ValueError('Input-only page projection required')
        items.append(dict(p, item_id=pid, action='policy_selected'))
    return dict(schema='v4_policy_selected_items_v1', mode='policy_selected', items=items)
