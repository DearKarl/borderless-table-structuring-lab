"""Portable I/O and immutable source checks; no model or benchmark imports."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
VENDOR_HASHES = {
    'mineru-frozen-ocr-sidecar-2026.09.09.1/core.py':
        '07c5fc2e82ae4136d36dc52b722c1dda686eee7a83ac79a54e609ef3a39c5334',
    'mineru-frozen-ocr-sidecar-2026.09.09.2/grounding.py':
        '66c69d5846cf4e8a39b2f69a5ea05a11f51a655fef9c748fc43c94e0f3bf75f7',
    'mineru-frozen-ocr-sidecar-2026.09.09.2/apply_consensus.py':
        '55cc100f1392536309603f0c5ce183e6fcd9bf35a670321419fe2bdaaf6a50a0',
    'frozen_paddleocr_locator.py':
        '270df01fb05fd7386966740dffada6f262ad3d1727f776e61939748ecc5fb180',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_bytes(path, payload):
    with Path(path).open('xb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path, value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         indent=2, allow_nan=False).encode('utf-8') + b'\n'
    write_bytes(path, payload)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_vendor():
    for relative, expected in VENDOR_HASHES.items():
        if sha(ROOT / 'vendor' / relative) != expected:
            raise ValueError('FROZEN_SOURCE_SHA_MISMATCH: ' + relative)
    return dict(VENDOR_HASHES)


def load_policy():
    verify_vendor()
    base = ROOT / 'vendor'
    core = load_module('core', base / 'mineru-frozen-ocr-sidecar-2026.09.09.1/core.py')
    grounding = load_module('grounding', base / 'mineru-frozen-ocr-sidecar-2026.09.09.2/grounding.py')
    apply = load_module('_portable_apply_consensus', base / 'mineru-frozen-ocr-sidecar-2026.09.09.2/apply_consensus.py')
    return core, grounding, apply


def failure(output, exc):
    write_json(output / 'FAILURE.json', {
        'status': 'FAILED_NO_ADMITTED_RESULT',
        'error_type': type(exc).__name__, 'error': str(exc),
        'benchmark_score': None,
    })
