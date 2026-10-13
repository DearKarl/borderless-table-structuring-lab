"""No GPU/model imports, installs, downloads or inference. Only frozen files and metadata."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8*1024*1024): h.update(chunk)
    return h.hexdigest()


def command(args, timeout=60):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'))
        return {'command': args, 'exit_code': result.returncode, 'stdout': result.stdout[-150000:], 'stderr': result.stderr[-5000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'command': args, 'error': type(exc).__name__ + ': ' + str(exc)}


def tensor_header(path):
    with Path(path).open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        if length > 64*1024*1024: raise ValueError('Header bound')
        header = json.loads(stream.read(length))
    total = sum(math.prod(v['shape']) for k,v in header.items() if k != '__metadata__')
    return {'path': str(path), 'stored_tensor_elements': total, 'tensors': len(header)-('__metadata__' in header),
            'classification': 'stored tensor upper bound only; runtime aux audit still required'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--locks', required=True)
    parser.add_argument('--locks-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if sha(args.locks) != args.locks_sha256: raise ValueError('Input lock digest mismatch')
    lock = json.loads(Path(args.locks).read_bytes()); rr = Path(lock['remote_root'])
    result = {'schema': 'hybrid-v2-tele-preflight-readonly/1', 'started_at': time.time(),
              'contract_sha256': lock['contract_sha256'], 'locks_sha256': args.locks_sha256,
              'GPU_allocated': False, 'model_inference': False, 'files': [], 'tensor_headers': [], 'environments': []}
    for expected in lock['files']:
        path = Path(expected['path'])
        if not path.resolve().is_relative_to(rr / 'inference'): raise ValueError('Unexpected source/model root')
        row = dict(expected)
        if not path.is_file(): row.update(exists=False, matches=False)
        else:
            actual = sha(path); size = path.stat().st_size
            row.update(exists=True, actual_sha256=actual, actual_bytes=size,
                       matches=actual == expected['sha256'] and size == expected['bytes'])
            if path.suffix == '.safetensors': result['tensor_headers'].append(tensor_header(path))
        result['files'].append(row)
    probe = """import importlib.metadata as m,json,sys
from pathlib import Path
names=['torch','torchvision','transformers','Pillow','numpy','scipy','safetensors','accelerate','pypdfium2','pymupdf','fast-langdetect','fasttext-predict','magika','onnxruntime','paddlex','paddleocr','paddlepaddle-gpu']
out={'python':sys.version,'executable':sys.executable,'packages':{}}
for name in names:
 try:
  d=m.distribution(name); out['packages'][name]={'version':d.version,'metadata_path':str(d._path)}
 except m.PackageNotFoundError: out['packages'][name]=None
print(json.dumps(out))
"""
    for name in ('tele-v2','tele-v3','paddle-v2'):
        env = rr / 'inference/environments' / name
        result['environments'].append(command([str(env / 'bin/python'), '-B', '-c', probe]))
        aux = []
        site = env / 'lib/python3.12/site-packages'
        for path in site.rglob('*'):
            if path.is_file() and path.suffix.lower() in ('.onnx','.bin','.ftz','.safetensors','.pdiparams','.pth'):
                aux.append({'path': str(path), 'bytes': path.stat().st_size})
                if len(aux) > 1000: raise ValueError('Aux inventory bound exceeded')
        result.setdefault('aux_candidates', {})[name] = aux
        result.setdefault('aux_loader_sources', {})[name] = []
        for relative in ('fast_langdetect/infer.py','magika/magika.py'):
            path = site / relative
            if path.is_file() and path.stat().st_size < 200000:
                result['aux_loader_sources'][name].append({'path': str(path), 'sha256': sha(path), 'text': path.read_text()})
    result['gpu_inventory'] = command(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'])
    result['gpu_processes'] = command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'])
    result['paddle_image'] = command(['docker','image','inspect','sha256:787dd9a15d41c442e78a5faa9cdc721f154cbf3d4cf8624edb41026c3052b75a'])
    result['source_model_identity_passed'] = all(row['matches'] for row in result['files'])
    result['full_capacity_gate_passed'] = False
    result['finished_at'] = time.time()
    with Path(args.output).open('x', encoding='utf-8') as stream: json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'output': args.output, 'sha256': sha(args.output),
                      'source_model_identity_passed': result['source_model_identity_passed']}))


if __name__ == '__main__': main()
