"""GT-free region identity, lossless RGB transport, and explicit failure policy."""
import base64
import hashlib
from io import BytesIO
import json
import math
from PIL import Image

CONFIG = {'task': 'equation', 'query': 'Formula Recognition:', 'min_pixels': 112896,
          'max_pixels': 1003520, 'max_new_tokens': 4096, 'use_cache': True,
          'skip_special_tokens': True}
CONFIG_SHA = hashlib.sha256(json.dumps(CONFIG, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
IDENTITY_FIELDS = ('run_id', 'page_id', 'region_id', 'crop_sha256', 'image_mode', 'shape', 'config_sha256')
EXPECTED_FAILURES = {'empty', 'length', 'other', 'decode_error'}


class RunBlocked(RuntimeError): pass
class ExpertTimeout(RuntimeError): pass
class PauseAfterFallback(RunBlocked): pass


def image_binding(image):
    return {'mode': image.mode, 'size': list(image.size),
            'pixel_sha256': hashlib.sha256(image.tobytes()).hexdigest()}


def eligible(block, crop):
    if block.type != 'equation': return False, 'not_equation'
    try:
        def flatten(items):
            for item in items:
                if isinstance(item, (list, tuple)): yield from flatten(item)
                else: yield float(item)
        values = list(flatten(block.bbox))
        if len(values) != 4 or not all(math.isfinite(v) for v in values):
            return False, 'unsupported_bbox'
        x0, y0, x1, y1 = values
        if not 0 <= x0 < x1 <= 1 or not 0 <= y0 < y1 <= 1:
            return False, 'invalid_bbox'
        if block.angle not in (0, 90, 180, 270): return False, 'unknown_angle'
        if not isinstance(crop, Image.Image) or crop.mode != 'RGB' or min(crop.size) < 1:
            return False, 'unsupported_crop'
    except (TypeError, ValueError, AttributeError):
        return False, 'unsupported_metadata'
    return True, 'eligible_equation'


def request(run_id, page_id, input_sha, page_ordinal, block_index, image):
    if image.mode != 'RGB' or min(image.size) < 1: raise RunBlocked('Invalid prepared RGB crop')
    buffer = BytesIO(); image.save(buffer, format='PNG')
    return {'run_id': run_id, 'page_id': page_id,
            'region_id': f'{input_sha}:{page_ordinal}:{block_index}',
            'crop_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
            'image_mode': 'RGB', 'shape': [image.height, image.width, 3],
            'config_sha256': CONFIG_SHA, 'task': 'equation',
            'png_base64': base64.b64encode(buffer.getvalue()).decode('ascii')}


def decode_request(data):
    if data['config_sha256'] != CONFIG_SHA or data['task'] != 'equation' or data['image_mode'] != 'RGB':
        raise RunBlocked('Task/config/color identity mismatch')
    raw = base64.b64decode(data['png_base64'], validate=True)
    with Image.open(BytesIO(raw)) as loaded:
        if loaded.mode != 'RGB' or [loaded.height, loaded.width, 3] != data['shape']:
            raise RunBlocked('Transport image identity mismatch')
        loaded.load(); image = loaded.copy()
    if hashlib.sha256(image.tobytes()).hexdigest() != data['crop_sha256']:
        image.close(); raise RunBlocked('Transport crop hash mismatch')
    return image


def termination(token_ids, eos_ids, cap=4096):
    if not isinstance(token_ids, list) or any(type(x) is not int for x in token_ids):
        raise RunBlocked('Invalid native token audit')
    if len(token_ids) > cap: raise RunBlocked('Native generation exceeded explicit budget')
    if not eos_ids or any(type(x) is not int for x in eos_ids): raise RunBlocked('Missing EOS identity')
    first = next((i for i, value in enumerate(token_ids) if value in eos_ids), None)
    return {'token_ids': token_ids, 'token_count': len(token_ids), 'eos_ids': eos_ids,
            'first_eos_index': first, 'budget': cap,
            'stop': 'eos' if first is not None else 'length' if len(token_ids) == cap else 'other'}


def validate_response(sent, received):
    if any(received.get(k) != sent[k] for k in IDENTITY_FIELDS):
        raise RunBlocked('Response region/config/crop identity mismatch')
    status = received.get('status')
    if status == 'fatal': raise RunBlocked('Paddle fatal: ' + str(received.get('error')))
    if status in EXPECTED_FAILURES: return None, status
    if status != 'ok': raise RunBlocked('Unknown Paddle response status')
    audit = received.get('generation', {})
    verified = termination(audit.get('token_ids'), audit.get('eos_ids'), CONFIG['max_new_tokens'])
    if verified['stop'] != 'eos': return None, verified['stop']
    text = received.get('raw_text')
    if not isinstance(text, str): raise RunBlocked('Paddle output is not text')
    if not text.strip(): return None, 'empty'
    return text, None  # Preserve original whitespace and math delimiters.
