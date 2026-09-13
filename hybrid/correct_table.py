#!/usr/bin/env python3
"""Apply the historical anchored OCR policy to one user-owned table (unscored)."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from common import failure, load_policy, sha, verify_vendor, write_bytes, write_json


POLICY = 'ANCHORED_SINGLE_CELL_TWO_ENGINE_ASCII_CONSENSUS_V1'
MODEL_HASHES = {
    'ch_PP-OCRv5_rec_server_infer.pth':
        '4767ddc90c1532ec01d881a980dae0a0b92679f4f82f88c4e9f92563de69e740',
    'ppocrv5_dict.txt':
        'd1979e9f794c464c0d2e0b70a7fe14dd978e9dc644c0e71f14158cdf8342af1b',
    'eng.traineddata':
        '7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2',
}
VERSIONS = {'mineru': '3.1.15', 'torch': '2.12.0', 'numpy': '2.2.6',
            'opencv-python': '4.13.0.92', 'Pillow': '12.2.0'}


def read_tokens(path, image_sha, dimensions):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    allowed = {'schema', 'status', 'input_image_sha256', 'coordinate_space',
               'image_width', 'image_height', 'ocr_tokens', 'locator'}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError('Tokens must use the documented image-only envelope')
    def no_reference_fields(item):
        if isinstance(item, dict):
            for key, child in item.items():
                if str(key).lower() in {'gold', 'labels', 'annotation', 'annotations',
                                       'reference', 'ground_truth', 'target', 'targets'}:
                    raise ValueError('Reference/label fields are forbidden')
                no_reference_fields(child)
        elif isinstance(item, list):
            for child in item:
                no_reference_fields(child)
    no_reference_fields(value)
    if value.get('schema') != 'hybrid-image-only-tokens/v1' or value.get('status') != 'ok':
        raise ValueError('Tokens are not a successful image-only locator result')
    if value.get('input_image_sha256') != image_sha:
        raise ValueError('Detector/table image SHA mismatch')
    if value.get('coordinate_space') != 'absolute_xyxy_in_table_crop':
        raise ValueError('Token coordinates must refer to the original table image')
    if [value.get('image_width'), value.get('image_height')] != list(dimensions):
        raise ValueError('Detector/table image dimensions mismatch')
    tokens = value.get('ocr_tokens')
    if not isinstance(tokens, list):
        raise ValueError('ocr_tokens must be a list')
    allowed_token = {'token_index', 'source_detection_index', 'text', 'confidence',
                     'bbox', 'polygon', 'alternatives'}
    for token in tokens:
        if not isinstance(token, dict) or set(token) - allowed_token:
            raise ValueError('Unknown token field; labels/references are not inputs')
    return value


def integer_crop_box(box, width, height):
    if (not isinstance(box, (list, tuple)) or len(box) != 4
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in box)
            or not 0 <= box[0] < box[2] <= width
            or not 0 <= box[1] < box[3] <= height):
        raise ValueError('Detected text bbox outside original image')
    return [math.floor(box[0]), math.floor(box[1]),
            math.ceil(box[2]), math.ceil(box[3])]


def verify_runtime(model_dir):
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    if versions != VERSIONS:
        raise RuntimeError('Use the pinned correction environment: ' + str(versions))
    if platform.python_version() != '3.10.20':
        raise RuntimeError('Correction environment requires Python 3.10.20')
    models = {}
    for name, expected in MODEL_HASHES.items():
        path = model_dir / 'ocr' / name
        if sha(path) != expected:
            raise ValueError('Model/dictionary SHA mismatch: ' + name)
        models[name] = {'path': str(path), 'sha256': expected}
    binary = shutil.which('tesseract')
    if binary is None:
        raise RuntimeError('Install native Tesseract and put tesseract on PATH')
    binary = str(Path(binary).resolve())
    version = subprocess.run([binary, '--version'], check=True, capture_output=True,
                             text=True, timeout=30)
    return {'python': platform.python_version(), 'packages': versions,
            'tesseract': {'path': binary, 'sha256': sha(binary),
                          'version': version.stdout.strip(),
                          'historical_binary_equivalence_claimed': False},
            'models': models, 'device': 'cpu'}


def recognize_proposals(image, grounded, runtime, output):
    """Exact original crop, recognizer setup and two-engine observations."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import cv2
    import numpy as np
    from PIL import ImageOps
    import torch
    from mineru.model.utils.tools.infer import pytorchocr_utility
    from mineru.model.utils.tools.infer.predict_rec import TextRecognizer
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    args = pytorchocr_utility.init_args().parse_args([])
    args.device = 'cpu'
    args.rec_model_path = runtime['models']['ch_PP-OCRv5_rec_server_infer.pth']['path']
    args.rec_char_dict_path = runtime['models']['ppocrv5_dict.txt']['path']
    args.rec_batch_num = 1
    args.rec_image_shape = '3, 48, 320'
    recognizer = TextRecognizer(args)
    engine_a = 'paddle:' + MODEL_HASHES['ch_PP-OCRv5_rec_server_infer.pth']
    engine_b = 'tesseract-eng:' + MODEL_HASHES['eng.traineddata']
    crops = output / 'crops'
    crops.mkdir()
    observations, records = {}, []
    with torch.inference_mode():
        for proposal in grounded['proposals']:
            box = integer_crop_box(proposal['bbox'], image.width, image.height)
            crop = ImageOps.expand(image.crop(box), border=3, fill='white')
            path = crops / ('cell-' + str(proposal['cell_index']) + '.png')
            crop.save(path)
            prediction, _ = recognizer([cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR)])
            if not prediction or not prediction[0] or not isinstance(prediction[0][0], str):
                raise RuntimeError('Invalid PP-OCR recognition output')
            tess = subprocess.run([
                runtime['tesseract']['path'], str(path), 'stdout', '--tessdata-dir',
                str(Path(runtime['models']['eng.traineddata']['path']).parent),
                '-l', 'eng', '--psm', '7'], check=True, capture_output=True,
                text=True, timeout=60, env=dict(os.environ, OMP_THREAD_LIMIT='1'))
            obs = [{'engine_id': engine_a, 'text': prediction[0][0]},
                   {'engine_id': engine_b, 'text': tess.stdout.strip()}]
            observations[proposal['cell_index']] = obs
            records.append(dict(proposal, crop='crops/' + path.name,
                                crop_sha256=sha(path), integer_box=box,
                                white_border_pixels=3, ocr=obs,
                                tesseract_stderr=tess.stderr))
    return observations, records


def run(args):
    output = Path(args.output).resolve()
    # No overwrite or implicit resume. A failure remains inspectable in this root.
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        html_path, image_path, tokens_path = (Path(args.html).resolve(),
            Path(args.table_image).resolve(), Path(args.ocr_tokens).resolve())
        source_hashes = verify_vendor()
        from PIL import Image
        raw_bytes = html_path.read_bytes()
        raw = raw_bytes.decode('utf-8', errors='strict')
        with Image.open(image_path) as original:
            original.load()
            image = original.convert('RGB')
        inputs = {str(p): sha(p) for p in (html_path, image_path, tokens_path)}
        envelope = read_tokens(tokens_path, inputs[str(image_path)], image.size)
        core, grounding, apply = load_policy()
        # Token invalidity is an input failure, not a successful abstention.
        grounding._tokens(envelope['ocr_tokens'])
        for token in envelope['ocr_tokens']:
            integer_crop_box(token['bbox'], image.width, image.height)
        runtime = verify_runtime(Path(args.model_dir).resolve())
        write_json(output / 'INPUT_FREEZE.json', {
            'policy': POLICY, 'inputs': inputs, 'vendor_sha256': source_hashes,
            'runtime': runtime, 'portable_adapter_sha256': sha(__file__),
            'input_scope': 'User-owned image, raw HTML and image-only OCR tokens',
            'training': False, 'gold_read': False, 'metric_read': False,
            'portable_adapter_score': None,
        })
        write_bytes(output / 'raw.html', raw_bytes)
        grounded = grounding.ground_table(raw, envelope['ocr_tokens'])
        observations, crops = {}, []
        if grounded['proposals']:
            observations, crops = recognize_proposals(image, grounded, runtime, output)
        try:
            result = apply.apply_table(raw, envelope['ocr_tokens'], observations)
            status = 'APPLY_COMPLETE_UNSCORED'
        except core.InvalidInput as exc:
            result = {'final': raw, 'post_commit_receipts': [], 'grounding': grounded,
                      'unsupported_html_reason': str(exc), 'raw_sha256': hashlib.sha256(raw_bytes).hexdigest()}
            status = 'UNSUPPORTED_HTML_RAW_PRESERVED_UNSCORED'
        if any(sha(path) != digest for path, digest in inputs.items()):
            raise RuntimeError('Input changed during correction; no output admitted')
        verify_vendor()
        final = result.pop('final').encode('utf-8')
        write_bytes(output / 'final.html', final)
        write_json(output / 'RECEIPT.json', dict(result, status=status, policy=POLICY,
            final_sha256=hashlib.sha256(final).hexdigest(), crop_records=crops,
            fresh_engine_calls=2 * len(crops), committed_cells=len(result['post_commit_receipts']),
            elapsed_seconds=time.monotonic() - started, benchmark_score=None,
            historical_score_not_reproduced=True,
            input_freeze_sha256=sha(output / 'INPUT_FREEZE.json')))
        write_json(output / 'READY.json', {'receipt_sha256': sha(output / 'RECEIPT.json'),
            'final_sha256': sha(output / 'final.html'), 'status': status})
        print(json.dumps({'status': status, 'committed_cells': len(result['post_commit_receipts']),
                          'output': str(output), 'benchmark_score': None}))
    except Exception as exc:
        failure(output, exc)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--html', required=True, help='One exact raw HTML table fragment')
    parser.add_argument('--table-image', required=True, help='Matching original table crop image')
    parser.add_argument('--ocr-tokens', required=True, help='Image-only token JSON from locate_tokens.py')
    parser.add_argument('--model-dir', required=True, help='Downloaded hybrid model directory')
    parser.add_argument('--output', required=True, help='NEW directory; never overwritten')
    try:
        run(parser.parse_args())
    except Exception as exc:
        print(type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
