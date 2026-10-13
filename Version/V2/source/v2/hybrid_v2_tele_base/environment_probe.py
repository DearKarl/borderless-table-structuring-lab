"""CPU-only dependency/import and auxiliary capacity probe, inside network-none Docker."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import struct
import traceback


def fasttext_header(raw):
    # Upstream v0.9.2 fasttext.cc signModel, args.cc save, dictionary.cc save.
    if len(raw) < 92: raise ValueError('Truncated fasttext header')
    magic, version = struct.unpack_from('<2i', raw)
    if (magic, version) != (793712314, 12): raise ValueError('Unsupported fasttext format')
    args = struct.unpack_from('<12id', raw, 8)
    size, words, labels, tokens, pruned = struct.unpack_from('<3i2q', raw, 64)
    dim, bucket = args[0], args[8]
    if not (1 <= dim <= 4096 and 0 <= bucket <= 100000000 and 0 < words <= size <= 100000000
            and 0 < labels <= size and words+labels == size and tokens > 0 and -1 <= pruned <= bucket
            and args[7] == 3): raise ValueError('Invalid supervised fasttext header bounds')
    return dict(magic=magic,version=version,dimension=dim,bucket=bucket,words=words,labels=labels,
                dictionary_size=size,tokens=tokens,pruned=pruned)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise RuntimeError('CPU-only probe required')
    sys.path.insert(0, a.source)
    result = {'complete': False, 'imports': [], 'auxiliary': [], 'source': a.source,
              'errors': [], 'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}}
    try:
        import fasttext
        import fast_langdetect
        root = Path(fast_langdetect.__file__).parent
        result['language_sources'] = {str(f): f.read_text() for f in root.rglob('*.py') if f.stat().st_size < 100000}
        models = list(root.rglob('*.ftz'))
        if len(models) != 1: raise RuntimeError('Expected one bundled fasttext model')
        f = models[0]; header = fasttext_header(f.read_bytes())
        model = fasttext.load_model(str(f))
        result['fasttext_prediction_test'] = model.predict('This is a document.')
        dimension, nwords, nlabels, bucket = [header[k] for k in ('dimension','words','labels','bucket')]
        # Expanded input/output matrices plus a conservative bit-level storage allowance
        # for quantizer codebooks and all auxiliary serialized data. Never ftz bytes alone.
        upper = dimension * (nwords + bucket + nlabels) + 8 * f.stat().st_size
        result['auxiliary'].append({'path': str(f), 'sha256': hashlib.sha256(f.read_bytes()).hexdigest(),
            'bytes': f.stat().st_size, 'dimension': dimension, 'words': nwords, 'labels': nlabels,
            'bucket': bucket, 'header': header, 'conservative_scalar_upper_bound': upper,
            'method': 'expanded input/output matrices plus serialized bits allowance'})
    except BaseException as exc:
        result['errors'].append({'stage':'fasttext_capacity','error':str(exc),'traceback':traceback.format_exc()})
    for name in ['magika', 'TeleOCR.tools.read_file', 'TeleOCR.engine',
                 'TeleOCR.vlm_utils.TeleOCR_client', 'transformers']:
        try:
            module = importlib.import_module(name)
            result['imports'].append({'name': name, 'file': module.__file__})
        except BaseException as exc:
            result['errors'].append({'stage':name,'error':str(exc),'traceback':traceback.format_exc()})
    try:
        from TeleOCR.tools.language import detect_lang
        result['language_test'] = [detect_lang('This is a technical document.'), detect_lang('这是中文技术文档。')]
        import magika
        for f in Path(magika.__file__).parent.rglob('*.onnx'):
            result['auxiliary'].append({'path': str(f), 'bytes': f.stat().st_size,
                'sha256': hashlib.sha256(f.read_bytes()).hexdigest(),
                'conservative_scalar_upper_bound': 8 * f.stat().st_size,
                'method': 'serialized bits upper bound; runtime model path must match'})
    except BaseException as exc:
        result['errors'].append({'stage':'language_and_magika','error':str(exc),'traceback':traceback.format_exc()})
    result['complete'] = not result['errors']
    with Path(a.output).open('x', encoding='utf-8') as stream: json.dump(result, stream, indent=2, ensure_ascii=False)
    return 0 if result['complete'] else 1


if __name__ == '__main__': raise SystemExit(main())
