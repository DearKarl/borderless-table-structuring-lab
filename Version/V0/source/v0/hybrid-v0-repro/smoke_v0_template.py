"""Frozen one-page CUDA smoke and conservative complete-system parameter gate."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

PINS = __PINS__
BUDGET_CHILD = __BUDGET_CHILD__
ROOT = Path('/srv/hybrid-research')
PREP = ROOT / 'inference/hybrid-v0-repro-v2'
OUT = ROOT / 'artifacts/official-full-20260921-v1/hybrid_v0/smoke-v1'
PYTHON = PREP / 'venv/bin/python'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_bytes())


def save(name, value):
    with (OUT / name).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def check(item):
    p = Path(item['path'])
    if not p.is_absolute():
        p = ROOT / p
    assert p.is_file() and sha(p) == item['sha256'], str(p)
    return p


def query(args):
    return subprocess.check_output(args, text=True, timeout=30).strip()


def acquire_gpu():
    active = query(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'])
    inventory = query(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'])
    for line in inventory.splitlines():
        index, uid, memory, utilization = [s.strip() for s in line.split(',')]
        if int(index) not in (4, 5, 6, 7) or uid in active or int(memory) > 10 or int(utilization) != 0:
            continue
        handle = (ROOT / 'receipts/REMOTE_NATIVE_SMOKE_v1' / (uid + '.lease')).open('a')
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close(); continue
        if uid in query(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader']):
            handle.close(); continue
        return uid, handle
    raise RuntimeError('No idle allowlisted GPU lease; no automatic retry')


def execute(args, label, env, lease=None, timeout=2400):
    with (OUT / (label + '.log')).open('xb') as log:
        child = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 env=env, cwd=PREP / 'release', start_new_session=True,
                                 pass_fds=() if lease is None else (lease.fileno(),))
        save(label + '-START.json', {'pid': child.pid, 'command': args, 'time': time.time()})
        result = None
        try:
            result = child.wait(timeout=timeout)
        finally:
            if result != 0:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
            save(label + '-EXIT.json', {'exit_code': child.returncode, 'interrupted_or_timeout': result is None, 'time': time.time()})
    assert result == 0, label + ' failed; preserve logs and do not retry'


def main():
    assert sys.argv[1:] == ['--smoke-once'] and sys.platform == 'linux' and not sys.flags.optimize
    assert not OUT.exists(), 'Smoke already claimed; no duplicate or automatic retry'
    prepared = read(check(PINS['prepared']))
    assert prepared['complete'] and not prepared['model_inference_started']
    for row in prepared['model_assets'] + prepared['base_distribution_metadata']:
        check(row)
    for row in PINS['release_files']:
        assert sha(PREP / 'release' / row['path']) == row['sha256']
    raw_lock = read(check(PINS['raw_lock']))
    manifest = read(check(PINS['input_manifest']))
    page = manifest['pages'][0]
    assert page == PINS['smoke_page']
    raw_row = next(p for p in raw_lock['pages'] if p['page_id'] == page['page_id'])
    raw = Path(raw_lock['prediction_directory']) / (page['page_id'] + '.md')
    assert sha(raw) == raw_row['prediction_sha256'] and sha(Path(page['image_path'])) == page['input_sha256']
    capacity = read(check(PINS['mineru_capacity']))
    assert capacity['complete'] and not capacity['unknown_components']
    for row in capacity['runtime_bindings']['mineru']['model_artifacts']:
        check(row)
    components = [x for x in capacity['unique_deployed_components'] if x['identity'].startswith('mineru:')]
    seen = set()
    for component in components:
        for row in component['evidence_artifacts'] + [x['artifact'] for x in component['covered_artifacts']]:
            key = (row['path'], row['sha256'])
            if key not in seen:
                check(row); seen.add(key)
    check(capacity['environment_coverage_artifacts']['mineru'])
    mineru_upper = sum(x['tensor_numel_upper_bound'] for x in components)
    assert mineru_upper == 1380147260
    OUT.mkdir(parents=True)
    environment = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
                   'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'CUDA_VISIBLE_DEVICES': '',
                   'LD_LIBRARY_PATH': str(ROOT / 'inference/driver-bundle-v1/attempt-v2')}
    environment.pop('PYTHONPATH', None)
    # Freeze actual files addressed by the inherited wheel RECORDs, not only version strings.
    dependency_files = {}
    for row in prepared['base_distribution_metadata']:
        record = Path(row['path'])
        if record.name != 'RECORD':
            continue
        for name, expected, size in csv.reader(record.read_text().splitlines()):
            path = (record.parent.parent / name).resolve()
            assert path.is_relative_to(ROOT / 'inference')
            if path.is_file() and path.suffix != '.pyc':
                dependency_files[str(path)] = sha(path)
    local_site = PREP / 'venv/lib/python3.12/site-packages'
    for path in local_site.rglob('*'):
        if path.is_file() and path.suffix != '.pyc':
            dependency_files[str(path)] = sha(path)
    save('DEPENDENCY_FILES.json', dependency_files)
    save('INPUT_FREEZE.json', {'pins': PINS, 'dependency_files_sha256': sha(OUT / 'DEPENDENCY_FILES.json'),
                              'entry_sha256': sha(Path(__file__)), 'Raw': 'frozen MinerU2605, not historical2604', 'GT_read': False})
    budget = OUT / 'budget_child.py'; budget.write_text(BUDGET_CHILD, encoding='utf-8')
    execute([str(PYTHON), '-B', str(budget), str(PREP / 'model'), str(OUT / 'NAVIDC_BUDGET.json')],
            'budget', environment, timeout=900)
    navidc = read(OUT / 'NAVIDC_BUDGET.json')
    total = mineru_upper + navidc['conservative_upper_bound']
    assert total <= 4000000000
    save('PARAMETER_BUDGET.json', {'passed': True, 'limit': 4000000000, 'total_upper_bound': total,
        'mineru_including_aux_upper_bound': mineru_upper, 'navidc': navidc,
        'exact_total_parameters': None, 'scope': 'MinerU complete conservative deployment inventory + same NaviDC model used for layout and tables; no additional v0 model'})
    inputs = OUT / 'input'; (inputs / 'images').mkdir(parents=True); (inputs / 'raw').mkdir()
    shutil.copyfile(page['image_path'], inputs / 'images' / Path(page['image_path']).name)
    shutil.copyfile(raw, inputs / 'raw' / raw.name)
    execute([str(PYTHON), '-B', '-m', 'btsl', 'prepare', '--input-dir', str(inputs)], 'prepare-input', environment)
    uid, lease = acquire_gpu()
    try:
        environment['CUDA_VISIBLE_DEVICES'] = uid
        save('GPU_LEASE.json', {'uuid': uid, 'lease_path': lease.name, 'shared_lease': True})
        execute([str(PYTHON), '-B', '-m', 'btsl', 'run', '--manifest', str(inputs / 'manifest.json'),
                 '--model-dir', str(PREP / 'model'), '--device', 'cuda', '--output', str(OUT / 'run')], 'btsl-run', environment, lease)
        execute([str(PYTHON), '-B', '-m', 'btsl', 'verify', '--run', str(OUT / 'run')], 'btsl-verify', environment, lease, 180)
    finally:
        lease.close()
    ready = read(OUT / 'run/READY.json')
    assert ready['pages'] == 1 and ready['gold_model_facing'] == 0
    for row in PINS['release_files']:
        assert sha(PREP / 'release' / row['path']) == row['sha256']
    for row in prepared['model_assets']:
        check(row)
    native = read(OUT / 'run/page_receipts' / page['page_id'] / 'native.json')
    save('SMOKE_COMPLETE.json', {'complete': True, 'page_id': page['page_id'], 'selection': 'first original manifest entry, no GT',
        'ready_sha256': sha(OUT / 'run/READY.json'), 'parameter_budget_sha256': sha(OUT / 'PARAMETER_BUDGET.json'),
        'statuses': ready['statuses'], 'native_table_calls': len(native.get('tables', [])),
        'full_run_started': False, 'table_path_covered': bool(native.get('tables'))})
    print(json.dumps(read(OUT / 'SMOKE_COMPLETE.json')), flush=True)


if __name__ == '__main__':
    main()
