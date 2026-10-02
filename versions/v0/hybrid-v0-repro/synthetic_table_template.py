"""Fixed invented business table through unmodified btsl; no benchmark selection."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

PINS = __PINS__
GENERATOR = __GENERATOR__


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior-entry', type=Path, required=True)
    args = parser.parse_args()
    assert sys.platform == 'linux' and not sys.flags.optimize
    assert hashlib.sha256(args.prior_entry.read_bytes()).hexdigest() == PINS['prior_entry_sha256']
    spec = importlib.util.spec_from_file_location('prior_smoke', args.prior_entry)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    old = module.OUT
    for name, digest in PINS['prior_receipts'].items():
        assert module.sha(old / name) == digest
    budget = module.read(old / 'PARAMETER_BUDGET.json')
    assert budget['passed'] and budget['total_upper_bound'] <= 4000000000
    previous = module.read(old / 'SMOKE_COMPLETE.json')
    assert previous['complete'] and previous['native_table_calls'] == 0
    prepared = module.read(module.check(module.PINS['prepared']))
    for row in prepared['model_assets'] + prepared['base_distribution_metadata']:
        module.check(row)
    for row in module.PINS['release_files']:
        assert module.sha(module.PREP / 'release' / row['path']) == row['sha256']
    freeze = module.read(old / 'INPUT_FREEZE.json')
    assert module.sha(old / 'DEPENDENCY_FILES.json') == freeze['dependency_files_sha256']
    for path, digest in module.read(old / 'DEPENDENCY_FILES.json').items():
        assert module.sha(Path(path)) == digest
    out = old.with_name('synthetic-table-smoke-v1')
    assert not out.exists(), 'Already attempted; no duplicate or automatic retry'
    out.mkdir(); module.OUT = out
    module.save('REUSED_FREEZE.json', {'pins': PINS, 'budget_reused': True, 'environment_retested': False,
                                     'original_input_freeze': module.sha(old / 'INPUT_FREEZE.json'),
                                     'entry_sha256': module.sha(Path(__file__)), 'benchmark_GT_used': False})
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'CUDA_VISIBLE_DEVICES': '',
           'LD_LIBRARY_PATH': str(module.ROOT / 'inference/driver-bundle-v1/attempt-v2')}
    env.pop('PYTHONPATH', None)
    generator = out / 'generate_fixture.py'; generator.write_text(GENERATOR, encoding='utf-8')
    inputs = out / 'input'
    module.execute([str(module.PYTHON), '-B', str(generator), str(inputs)], 'generate-fixture', env)
    module.execute([str(module.PYTHON), '-B', '-m', 'btsl', 'prepare', '--input-dir', str(inputs)], 'prepare-input', env)
    uid, lease = module.acquire_gpu()
    try:
        env['CUDA_VISIBLE_DEVICES'] = uid
        module.save('GPU_LEASE.json', {'uuid': uid, 'lease_path': lease.name})
        module.execute([str(module.PYTHON), '-B', '-m', 'btsl', 'run', '--manifest', str(inputs / 'manifest.json'),
                        '--model-dir', str(module.PREP / 'model'), '--device', 'cuda', '--output', str(out / 'run')], 'btsl-run', env, lease)
        module.execute([str(module.PYTHON), '-B', '-m', 'btsl', 'verify', '--run', str(out / 'run')], 'btsl-verify', env, lease, 180)
    finally:
        lease.close()
    page = out / 'run/page_receipts/synthetic-business-table-001'
    native = module.read(page / 'native.json'); receipt = module.read(page / 'receipt.json')
    ready = module.read(out / 'run/READY.json')
    assert ready['pages'] == 1 and len(native.get('tables', [])) >= 1
    assert receipt['status'] == 'NATIVE_COLLECTION_ASSEMBLED', 'Table inference completed but valid full collection assembly not proven'
    module.save('SMOKE_COMPLETE.json', {'complete': True, 'invented_fixture': True, 'benchmark_GT_used': False,
        'native_table_calls': len(native['tables']), 'status': receipt['status'],
        'ready_sha256': module.sha(out / 'run/READY.json'), 'receipt_sha256': module.sha(page / 'receipt.json'),
        'budget_source_sha256': PINS['prior_receipts']['PARAMETER_BUDGET.json'],
        'original_dependency_freeze_sha256': freeze['dependency_files_sha256'], 'full_run_started': False})
    print(json.dumps(module.read(out / 'SMOKE_COMPLETE.json')), flush=True)


if __name__ == '__main__':
    main()
