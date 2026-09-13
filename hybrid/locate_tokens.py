#!/usr/bin/env python3
"""Produce image-only tokens with the frozen PaddleOCR v5 locator (unscored)."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

from common import ROOT, failure, load_module, sha, verify_vendor, write_json


LOCATOR_CONFIG_SHA256 = '80a2b7ea3ca73d6da4d8ba7a4fbd6a16716bd566e0afe7e9383e53863080937e'


def relocated_config(model_dir):
    if sha(ROOT / 'locator_config.json') != LOCATOR_CONFIG_SHA256:
        raise ValueError('Frozen portable locator configuration SHA mismatch')
    config = json.loads((ROOT / 'locator_config.json').read_text(encoding='utf-8'))
    for model in config['models'].values():
        model['directory'] = str(model_dir / model['directory'])
    return config


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        started = time.monotonic()
        hashes = verify_vendor()
        config = relocated_config(Path(args.model_dir).resolve())
        # Must precede importing native Paddle/NumPy runtimes.
        for key, value in config['runtime_controls']['environment'].items():
            os.environ[key] = value
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        from PIL import Image
        source = load_module('_portable_frozen_locator', ROOT / 'vendor/frozen_paddleocr_locator.py')
        runtime = source.runtime_versions()
        source.verify_runtime(config, runtime)
        model_hashes = source.verify_models(config)
        image_path = Path(args.table_image).resolve()
        image_sha = sha(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
            image.verify()
        policy = source.effective_inference_policy(config, {
            'image_width': width, 'image_height': height})
        write_json(output / 'INPUT_FREEZE.json', {'image_sha256': image_sha,
            'image_width': width, 'image_height': height, 'runtime': runtime,
            'model_artifact_sha256': model_hashes, 'vendor_sha256': hashes,
            'portable_config_sha256': sha(ROOT / 'locator_config.json'),
            'source_config_sha256': config['source_config_sha256'],
            'effective_inference': policy, 'adapter_sha256': sha(__file__),
            'gold_read': False, 'metric_read': False, 'training': False,
            'portable_adapter_score': None})
        engine = source.create_pipeline(config)
        results = engine.predict(str(image_path), **policy['predict_overrides'])
        if len(results) != 1:
            raise RuntimeError('Expected exactly one PaddleOCR image result')
        raw = source.unwrap_result(results[0])
        tokens, counts = source.parse_ocr_tokens(raw, width, height)
        if sha(image_path) != image_sha:
            raise RuntimeError('Image changed during locator inference')
        verify_vendor()
        write_json(output / 'tokens.json', {
            'schema': 'hybrid-image-only-tokens/v1', 'status': 'ok',
            'input_image_sha256': image_sha, 'image_width': width, 'image_height': height,
            'coordinate_space': 'absolute_xyxy_in_table_crop', 'ocr_tokens': tokens,
            'locator': {'policy': policy, 'model_artifact_sha256': model_hashes,
                        'source_sha256': hashes['frozen_paddleocr_locator.py']}})
        write_json(output / 'RECEIPT.json', {
            'status': 'IMAGE_ONLY_TOKENS_COMPLETE_UNSCORED', 'counts': counts,
            'raw_result_sha256': source.stable_json_hash(raw),
            'tokens_sha256': sha(output / 'tokens.json'),
            'elapsed_seconds': time.monotonic() - started, 'benchmark_score': None,
            'input_freeze_sha256': sha(output / 'INPUT_FREEZE.json')})
        write_json(output / 'READY.json', {'receipt_sha256': sha(output / 'RECEIPT.json'),
            'tokens_sha256': sha(output / 'tokens.json')})
        print(json.dumps({'status': 'COMPLETE_UNSCORED', 'tokens': len(tokens),
                          'output': str(output), 'benchmark_score': None}))
    except Exception as exc:
        failure(output, exc)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table-image', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output', required=True, help='NEW output directory')
    try:
        run(parser.parse_args())
    except Exception as exc:
        print(type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
