"""One fresh ordered 1651-page released CUDA pipeline, single idle GPU lease."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

PINS = __PINS__


def identity(pid):
    return {'pid': pid, 'start_ticks': (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19],
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def import_prior(path):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PINS['prior_entry_sha256']
    spec = importlib.util.spec_from_file_location('prior_smoke', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def verify_ready(m, folder):
    ready = m.read(folder / 'READY.json')
    for name, expected in ready['files'].items():
        p = (folder / name).resolve()
        assert not Path(name).is_absolute() and p.is_relative_to(folder.resolve())
        assert m.sha(p) == expected
    return ready


def artifact(m, p):
    return {'path': str(p), 'sha256': m.sha(p), 'bytes': p.stat().st_size}


def work(m, old):
    out = m.OUT
    m.save('SUPERVISOR.json', identity(os.getpid()))
    for name, digest in PINS['smoke_receipts'].items():
        assert m.sha(old / name) == digest
    synthetic = old.with_name('synthetic-table-smoke-v1')
    assert m.sha(synthetic / 'SMOKE_COMPLETE.json') == PINS['synthetic_complete_sha256']
    done = m.read(synthetic / 'SMOKE_COMPLETE.json')
    assert done['complete'] and done['native_table_calls'] >= 1 and done['status'] == 'NATIVE_COLLECTION_ASSEMBLED'
    assert m.sha(synthetic / 'run/READY.json') == done['ready_sha256']
    verify_ready(m, synthetic / 'run')
    budget = m.read(old / 'PARAMETER_BUDGET.json')
    assert budget['passed'] and budget['total_upper_bound'] <= 4000000000
    prepared = m.read(m.check(m.PINS['prepared']))
    for item in prepared['model_assets'] + prepared['base_distribution_metadata']:
        m.check(item)
    for item in m.PINS['release_files']:
        assert m.sha(m.PREP / 'release' / item['path']) == item['sha256']
    depfile = old / 'DEPENDENCY_FILES.json'
    assert m.sha(depfile) == m.read(old / 'INPUT_FREEZE.json')['dependency_files_sha256']
    dependencies = m.read(depfile)
    for path, digest in dependencies.items():
        assert m.sha(Path(path)) == digest
    rawlock = m.read(m.check(m.PINS['raw_lock']))
    manifest = m.read(m.check(m.PINS['input_manifest']))
    pages = manifest['pages']; rawrows = {p['page_id']: p for p in rawlock['pages']}
    assert rawlock['complete'] and len(pages) == len(rawrows) == 1651
    assert len({p['page_id'] for p in pages}) == 1651 and set(rawrows) == {p['page_id'] for p in pages}
    inputs = out / 'input'; (inputs / 'images').mkdir(parents=True); (inputs / 'raw').mkdir()
    rows = []
    for page in pages:
        pid = page['page_id']; rawrow = rawrows[pid]
        assert rawrow['input_sha256'] == page['input_sha256']
        source = Path(page['image_path']); raw = Path(rawlock['prediction_directory']) / (pid + '.md')
        assert source.stem == pid and source.is_file() and m.sha(source) == page['input_sha256']
        assert raw.is_file() and m.sha(raw) == rawrow['prediction_sha256']
        target_image = inputs / 'images' / source.name; target_raw = inputs / 'raw' / raw.name
        shutil.copyfile(source, target_image); shutil.copyfile(raw, target_raw)
        assert m.sha(target_image) == page['input_sha256'] and m.sha(target_raw) == rawrow['prediction_sha256']
        rows.append({'id': pid, 'image': 'images/' + source.name, 'raw': 'raw/' + raw.name,
                     'image_sha256': page['input_sha256'], 'raw_sha256': rawrow['prediction_sha256']})
    with (inputs / 'manifest.json').open('x') as f:
        json.dump({'schema': 'btsl-source-pages/1', 'pages': rows}, f, ensure_ascii=False, indent=2)
    m.save('RUN_FREEZE.json', {'schema': 'hybrid-v0-cuda-repro/1', 'pages': 1651, 'order': 'original input manifest order',
        'smoke_gates': PINS, 'release_pins': m.PINS, 'input_manifest_sha256': m.sha(inputs / 'manifest.json'),
        'dependency_file_inventory': artifact(m, depfile), 'parameter_budget': artifact(m, old / 'PARAMETER_BUDGET.json'),
        'Raw_policy': 'frozen MinerU2605 complete baseline; not freshly regenerated and not historical2604',
        'NaviDC_policy': 'all 1651 pages fresh in new run; no smoke/historical generation adopted',
        'scientific_policy': 'unmodified btsl0.2.0 source-image + Raw native table collection',
        'GPUs_max': 1, 'automatic_retry': False, 'GT_read': False, 'entry_sha256': m.sha(Path(__file__))})
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'LD_LIBRARY_PATH': str(m.ROOT / 'inference/driver-bundle-v1/attempt-v2')}
    env.pop('PYTHONPATH', None)
    uid, lease = m.acquire_gpu()
    try:
        env['CUDA_VISIBLE_DEVICES'] = uid
        m.save('GPU_LEASE.json', {'uuid': uid, 'lease_path': lease.name, 'exclusive_single_card': True})
        m.execute([str(m.PYTHON), '-B', '-m', 'btsl', 'run', '--manifest', str(inputs / 'manifest.json'),
                   '--model-dir', str(m.PREP / 'model'), '--device', 'cuda', '--output', str(out / 'run')], 'btsl-full-run', env, lease, None)
        m.execute([str(m.PYTHON), '-B', '-m', 'btsl', 'verify', '--run', str(out / 'run')], 'btsl-full-verify', env, lease, None)
    finally:
        lease.close()
    ready = verify_ready(m, out / 'run')
    assert ready['pages'] == 1651 and ready['gold_model_facing'] == 0
    predictions = out / 'run/pages'
    assert {p.name for p in predictions.iterdir()} == {p['page_id'] + '.md' for p in pages}
    for item in m.PINS['release_files']:
        assert m.sha(m.PREP / 'release' / item['path']) == item['sha256']
    for item in prepared['model_assets']:
        m.check(item)
    for path, digest in dependencies.items():
        assert m.sha(Path(path)) == digest
    m.check(m.PINS['raw_lock']); m.check(m.PINS['input_manifest'])
    key = m.sha(out / 'RUN_FREEZE.json'); entries = []
    common = [artifact(m, out / 'RUN_FREEZE.json'), artifact(m, out / 'run/FREEZE.json'), artifact(m, out / 'run/READY.json')]
    for page in pages:
        pid = page['page_id']; folder = out / 'run/page_receipts' / pid
        receipt = m.read(folder / 'receipt.json'); page_ready = verify_ready(m, folder)
        prediction = predictions / (pid + '.md')
        digest = m.sha(prediction)
        assert digest == receipt['final_sha256'] == page_ready['prediction_sha256']
        entries.append({'page_id': pid, 'input_sha256': page['input_sha256'], 'prediction_sha256': digest,
            'status': 'success', 'status_semantics': 'completed released pipeline; scientific Raw abstention remains an explicit native_outcome',
            'native_outcome': receipt['status'], 'raw_baseline_status': rawrows[pid]['status'],
            'raw_baseline_prediction_sha256': rawrows[pid]['prediction_sha256'],
            'native_artifacts': common + [artifact(m, folder / n) for n in ('native.json', 'receipt.json', 'READY.json')],
            'source_runtime_key': key, 'source_terminal': artifact(m, folder / 'READY.json')})
    # Missing/failed calls never reach here; no invented empty/Raw page completion.
    m.save('COMPLETE_INPUT_LOCK.json', {'arm': 'hybrid_v0', 'complete': True, 'runtime_key': key,
        'prediction_directory': str(predictions), 'pages': entries, 'underlying_raw_input_lock': m.PINS['raw_lock'],
        'generation_scope': 'NaviDC fresh all1651, frozen MinerU2605 Raw', 'released_pipeline_statuses': ready['statuses']})
    return {'complete': True, 'pages': 1651, 'statuses': ready['statuses'],
            'input_lock': artifact(m, out / 'COMPLETE_INPUT_LOCK.json'), 'evaluation_started': False}


def supervise(m, old):
    try:
        result = work(m, old)
    except BaseException as exc:
        traceback.print_exc()
        result = {'complete': False, 'error_type': type(exc).__name__, 'error': str(exc), 'automatic_retry': False}
    m.save('ENTRY_EXIT.json', result)
    return 0 if result['complete'] else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior-entry', type=Path, required=True)
    parser.add_argument('--run-once', action='store_true', required=True)
    args = parser.parse_args()
    assert sys.platform == 'linux' and not sys.flags.optimize
    m = import_prior(args.prior_entry); old = m.OUT
    m.OUT = old.with_name('full-cuda-v1')
    assert not m.OUT.exists(), 'Full run already claimed; no repeat/resume by this entry'
    m.OUT.mkdir()
    m.save('CLAIM.json', {'owner': identity(os.getpid()), 'entry_sha256': m.sha(Path(__file__)), 'time': time.time(), 'pins': PINS})
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    pid = os.fork()
    if pid:
        print(json.dumps({'supervisor_pid': pid, 'output': str(m.OUT), 'inference_start_confirmed': False}), flush=True)
        return
    os.setsid()
    with open(os.devnull, 'rb') as stdin, (m.OUT / 'supervisor.log').open('xb') as log:
        os.dup2(stdin.fileno(), 0); os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
        status = supervise(m, old)
    os._exit(status)


if __name__ == '__main__':
    main()
