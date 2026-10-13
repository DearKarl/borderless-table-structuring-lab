"""One native Paddle recognizer, region-only stdin/stdout service. No layout pipeline."""
import argparse
from contextlib import redirect_stdout
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import uuid
from .region_protocol import CONFIG, CONFIG_SHA, IDENTITY_FIELDS, decode_request, RunBlocked
from .audit import install_paddle_audit, audit_snapshot
from .runtime_resources import ResourceAudit


class FormulaPredictor:
    def __init__(self, model_dir):
        import paddle
        from paddlex import create_predictor
        from paddlex.inference.models.doc_vlm.predictor import DocVLMLocalPredictor
        if not paddle.is_compiled_with_cuda(): raise RunBlocked('CUDA Paddle required')
        if paddle.device.cuda.device_count() != 1: raise RunBlocked('Expected exactly one visible Paddle GPU')
        paddle.set_device('gpu:0')
        self.predictor = create_predictor(model_name='PaddleOCR-VL-1.6-0.9B', model_dir=str(model_dir),
            engine='paddle_dynamic', device='gpu:0', batch_size=1)
        if type(self.predictor) is not DocVLMLocalPredictor or self.predictor.dtype != 'bfloat16':
            raise RunBlocked('Unexpected Paddle predictor or dtype')
        if type(self.predictor.infer).__name__ != 'PaddleOCRVLForConditionalGeneration':
            raise RunBlocked('Unexpected Paddle model class')
        self.audits, self.calls, self.paddle = [], 0, paddle
        self.restore = install_paddle_audit(self.predictor.infer, self.audits.append)
        self.session = uuid.uuid4().hex
        dtype_counts = {}
        for parameter in self.predictor.infer.parameters():
            key = str(parameter.dtype)
            dtype_counts[key] = dtype_counts.get(key,0)+int(parameter.numel())
        if not any('bfloat16' in key for key in dtype_counts): raise RunBlocked('No actual BF16 Paddle parameters')
        self.load = {'worker_session': self.session, 'pid': os.getpid(), 'model_loads': 1,
                     'predictor_class': type(self.predictor).__name__, 'model_class': type(self.predictor.infer).__name__,
                     'dtype': self.predictor.dtype, 'config_sha256': CONFIG_SHA,
                     'parameter_dtype_counts':dtype_counts,
                     'generation_config': audit_snapshot(self.predictor.infer.generation_config.to_dict()),
                     'packages': {name: importlib.metadata.version(name) for name in
                                  ('paddlepaddle-gpu', 'paddlex', 'paddleocr', 'Pillow', 'numpy')},
                     'Paddle_layout_loaded': False}

    def recognize(self, sent):
        result = {key: sent[key] for key in IDENTITY_FIELDS}
        result.update(worker_session=self.session, request_number=self.calls+1)
        image = decode_request(sent)
        self.calls += 1
        first = len(self.audits); started = time.monotonic()
        try:
            returned = list(self.predictor.predict([{'image': image, 'query': CONFIG['query']}],
                min_pixels=CONFIG['min_pixels'], max_pixels=CONFIG['max_pixels'],
                max_new_tokens=CONFIG['max_new_tokens'], use_cache=True, skip_special_tokens=True))
            audits = self.audits[first:]
            if len(returned) != 1 or len(audits) != 1: raise RunBlocked('Paddle request/result/audit arity mismatch')
            audit = audits[0]
            if not audit['returned']: raise RunBlocked('Paddle generation failed')
            text = returned[0].get('result')
            if not isinstance(text, str): raise RunBlocked('Paddle result is not string')
            status = audit['stop'] if audit['stop'] != 'eos' else 'ok' if text.strip() else 'empty'
            result.update(status=status, raw_text=text, generation=audit)
        except UnicodeDecodeError as exc:
            result.update(status='decode_error', error=str(exc), generation_records=self.audits[first:])
        except BaseException as exc:
            result.update(status='fatal', error_type=type(exc).__name__, error=str(exc), generation_records=self.audits[first:])
        finally:
            result['elapsed_seconds'] = time.monotonic()-started
            result['peak_allocated_bytes'] = self.paddle.device.cuda.max_memory_allocated()
            result['peak_reserved_bytes'] = self.paddle.device.cuda.max_memory_reserved()
            image.close()
        return result

    def close(self):
        self.restore()
        self.predictor.close()

    def native_control(self, sent):
        """Smoke-only direct predictor control on the identical lossless source crop."""
        image = decode_request(sent)
        self.calls += 1
        start = len(self.audits)
        try:
            raw = list(self.predictor.predict([{'image': image, 'query': 'Formula Recognition:'}],
                min_pixels=112896, max_pixels=1003520, max_new_tokens=4096,
                use_cache=True, skip_special_tokens=True))
            if len(raw) != 1 or len(self.audits)-start != 1:
                raise RunBlocked('Direct native control arity mismatch')
            return {**{key:sent[key] for key in IDENTITY_FIELDS}, 'status':'native_control',
                    'raw_text':raw[0].get('result'), 'generation':self.audits[-1],
                    'worker_session':self.session, 'request_number':self.calls}
        finally: image.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--audit-dir', required=True)
    parser.add_argument('--smoke-controls', action='store_true')
    parser.add_argument('--manifest')
    args = parser.parse_args()
    protocol = sys.stdout
    def reply(data):
        protocol.write(json.dumps(data, ensure_ascii=False) + '\n'); protocol.flush()
    folder = Path(args.audit_dir); folder.mkdir(exist_ok=False)
    resources = None
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_bytes())
        for name,version in manifest['paddle_packages'].items():
            if importlib.metadata.version(name) != version: raise RunBlocked('Paddle package mismatch: '+name)
        resources = ResourceAudit(manifest['identity_files'])
    with redirect_stdout(sys.stderr):
        started = time.monotonic(); predictor = FormulaPredictor(args.model_dir)
        predictor.load['initialization_seconds'] = time.monotonic()-started
        (folder / 'load.json').write_text(json.dumps(predictor.load, ensure_ascii=False, indent=2), encoding='utf-8')
        reply({'event': 'ready', **predictor.load})
        try:
            while True:
                line = sys.stdin.buffer.readline(96 * 1024 * 1024 + 1)
                if not line: break
                if len(line) > 96 * 1024 * 1024: raise RunBlocked('Request exceeds protocol byte bound')
                sent = json.loads(line)
                if sent.get('op') == 'close': break
                if sent.get('op') == 'native_control':
                    if not args.smoke_controls: raise RunBlocked('Native control disabled outside smoke')
                    response = predictor.native_control(sent)
                else: response = predictor.recognize(sent)
                if resources:
                    response['resource_audit'] = resources.result()
                    if response['resource_audit']['unknown_resources']:
                        response.update(status='fatal',error='Unbound Paddle runtime auxiliary resources')
                with (folder / ('response-%06d.json' % predictor.calls)).open('x', encoding='utf-8') as stream:
                    json.dump(response, stream, ensure_ascii=False, indent=2)
                reply(response)
                if response['status'] == 'fatal': return 1
        finally:
            if resources:
                (folder/'RESOURCE_AUDIT.json').write_text(json.dumps(resources.result(),indent=2),encoding='utf-8')
            predictor.close()
    return 0


if __name__ == '__main__': sys.exit(main())
