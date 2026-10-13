"""Independent CPU evaluation of all frozen predictions, including empty failures."""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .io_utils import atomic_bytes, atomic_json, digest, process_identity, read_json, utc
from .supervisor import docker

SCORER_PYTHON = '/opt/miniconda310/envs/omnidocbench_v16_smoke_20260408_py310/bin/python'


def metrics(raw, pages=1651):
    if raw['match_debug']['page_count'] != pages:
        raise ValueError('Evaluator denominator differs')
    paths = {'text_edit': ('text_block', 'all', 'Edit_dist', 'ALL_page_avg'),
             'cdm': ('display_formula', 'page', 'CDM', 'ALL'),
             'teds': ('table', 'page', 'TEDS', 'ALL'),
             'structure_teds': ('table', 'page', 'TEDS_structure_only', 'ALL'),
             'reading_order': ('reading_order', 'all', 'Edit_dist', 'ALL_page_avg')}
    result = {}
    for name, path in paths.items():
        value = raw
        for key in path:
            value = value[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid official metric: ' + name)
        result[name] = value
    result['overall'] = 100 * ((1 - result['text_edit']) + result['cdm'] + result['teds']) / 3
    return result


def guard(deadline):
    command = [SCORER_PYTHON, '/source/pdf_validation.py', '--config', '/config/end2end-full.yaml']
    child = subprocess.Popen(command, start_new_session=True)
    atomic_json('/work/SCORER_PROCESS.json', {'identity': process_identity(child.pid), 'command': command,
                                          'absolute_deadline_unix': deadline, 'started_at': utc()})
    code, error = None, None
    try:
        code = child.wait(timeout=max(0.001, deadline - time.time()))
    except subprocess.TimeoutExpired:
        error = 'absolute_evaluation_deadline'
        os.killpg(child.pid, signal.SIGKILL)
        code = child.wait(timeout=5)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            code = child.wait(timeout=5)
        atomic_json('/work/SCORER_TERMINAL.json', {'exit_code': code, 'error': error, 'finished_at': utc(),
                                               'absolute_deadline_unix': deadline})
    return 0 if code == 0 and error is None else 1


def prepare_predictions(manifest, arm, target):
    source = Path(manifest['round_root']) / arm / 'output'
    target.mkdir(parents=True, exist_ok=False)
    rows = []
    for page in manifest['pages']:
        page_id = page['page_id']
        receipt = read_json(source / 'receipts' / (page_id + '.json'))
        file = source / 'markdown' / (page_id + '.md')
        if receipt['run_id'] != manifest['run_id'] or digest(file) != receipt['prediction_sha256']:
            raise ValueError('Prediction lineage differs: ' + page_id)
        name = page['original_page_id'] + '.md'
        if Path(name).name != name or (target / name).exists():
            raise ValueError('Invalid or duplicate benchmark output name')
        atomic_bytes(target / name, file.read_bytes())
        rows.append({'page_id': page_id, 'prediction_file': name, 'sha256': digest(file),
                     'status': receipt['status'], 'retained_stage': receipt['retained_stage']})
    if len(rows) != 1651 or len(list(target.glob('*.md'))) != 1651:
        raise ValueError('Full prediction denominator differs')
    return rows


def launch(manifest, binding, arm, deadline):
    root = Path(manifest['round_root']) / arm / 'evaluation'
    root.mkdir(parents=True, exist_ok=False)
    rows = prepare_predictions(manifest, arm, root / 'predictions')
    atomic_json(root / 'PREDICTION_BINDING.json', rows)
    work = root / 'work'
    work.mkdir()
    name = manifest['run_id'] + '-' + arm.lower().replace('.', '-') + '-score'
    args = ['create', '--name', name, '--label', 'v5-score-owner=' + manifest['run_id'],
            '--network', 'none', '--runtime', 'runc', '--restart', 'no', '--cpus', '8',
            '--memory', '32g', '--memory-swap', '32g', '--pids-limit', '512',
            '--env', 'CUDA_VISIBLE_DEVICES=', '--env', 'NVIDIA_VISIBLE_DEVICES=void',
            '--env', 'PYTHONDONTWRITEBYTECODE=1', '--env', 'OMP_NUM_THREADS=1', '--env', 'OPENBLAS_NUM_THREADS=1',
            '--env', 'MKL_NUM_THREADS=1', '--env', 'PYTHONPATH=/source:/code', '--workdir', '/work']
    mounts = [(str(work), '/work', False), (str(root / 'predictions'), '/pred', True),
              (binding['source']['path'], '/source', True), (binding['config']['path'], '/config/end2end-full.yaml', True),
              (binding['gt']['path'], '/gt/OmniDocBench.json', True), (manifest['code_root'], '/code', True)]
    for src, dst, readonly in mounts:
        args += ['--mount', f'type=bind,src={src},dst={dst}' + (',readonly' if readonly else '')]
    args += ['--entrypoint', SCORER_PYTHON, binding['image'], '-B', '-m', 'versions.v5.evaluate', '--guard',
             '--deadline', str(deadline)]
    cid = docker(*args)
    info = json.loads(docker('inspect', cid))[0]
    if info['HostConfig'].get('Devices') or info['HostConfig'].get('DeviceRequests'):
        raise RuntimeError('Evaluator unexpectedly received a GPU')
    actual = {(x['Source'], x['Destination'], not x['RW']) for x in info['Mounts'] if x['Type'] == 'bind'}
    if actual != set(mounts):
        raise RuntimeError('Evaluator mounts differ')
    receipt = {'arm': arm, 'cid': cid, 'name': name, 'created_at': utc(), 'deadline': deadline,
               'command': ['docker', *args], 'phase': 'scoring', 'evaluation': binding,
               'prediction_binding_sha256': digest(root / 'PREDICTION_BINDING.json')}
    atomic_json(root / 'LAUNCH.json', receipt)
    docker('start', cid)
    state = json.loads(docker('inspect', cid))[0]['State']
    receipt.update(started_at=state['StartedAt'], pid=state['Pid'], identity=process_identity(state['Pid']))
    atomic_json(root / 'STATUS.json', receipt)
    log = open(root / 'scorer.log', 'w')
    subprocess.Popen(['docker', 'logs', '--follow', cid], stdout=log, stderr=subprocess.STDOUT)
    log.close()
    return receipt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest')
    ap.add_argument('--binding')
    ap.add_argument('--guard', action='store_true')
    ap.add_argument('--deadline', type=float)
    args = ap.parse_args()
    if args.guard:
        return guard(args.deadline)
    manifest = read_json(args.manifest)
    binding = read_json(args.binding)
    if manifest['phase'] != 'full' or len(manifest['pages']) != 1651:
        raise ValueError('Only frozen complete full-set predictions may enter this scorer')
    for name in ('gt', 'config'):
        if digest(binding[name]['path']) != binding[name]['sha256']:
            raise ValueError('Evaluator input identity mismatch')
    for name, expected in binding['source']['files'].items():
        if digest(Path(binding['source']['path']) / name) != expected:
            raise ValueError('Evaluator source changed')
    root = Path(manifest['round_root'])
    inference = read_json(root / 'INFERENCE_COMPLETE.json')
    deadline = inference['absolute_deadline_unix'] - 10
    if deadline - time.time() < 60:
        raise RuntimeError('No evaluation time remains within the round ceiling')
    launches = [launch(manifest, binding, arm, deadline) for arm in manifest['gpu_mapping']]
    pending = {row['arm']: row for row in launches}
    results = {}
    while pending:
        for arm, row in list(pending.items()):
            state = json.loads(docker('inspect', row['cid']))[0]
            if state['Config']['Labels'].get('v5-score-owner') != manifest['run_id']:
                raise RuntimeError('Evaluator ownership differs')
            if state['State']['Running']:
                continue
            evaluation_root = root / arm / 'evaluation'
            terminal = read_json(evaluation_root / 'work/SCORER_TERMINAL.json')
            row.update(phase='evaluation_complete' if state['State']['ExitCode'] == 0 else 'evaluation_failed',
                       exit_code=state['State']['ExitCode'], completed_at=utc(), terminal=terminal)
            if row['exit_code'] == 0 and terminal['exit_code'] == 0 and terminal['error'] is None:
                result_path = evaluation_root / 'work/result/pred_quick_match_metric_result.json'
                values = metrics(read_json(result_path))
                results[arm] = {'metrics': values, 'raw_result_sha256': digest(result_path),
                                'raw_result_path': str(result_path), 'denominator': 1651}
                atomic_json(evaluation_root / 'METRICS.json', results[arm])
            else:
                results[arm] = {'error': row}
            atomic_json(evaluation_root / 'STATUS.json', row)
            del pending[arm]
        atomic_json(root / 'EVALUATION_STATUS.json', {'updated_at': utc(), 'running_arms': list(pending),
                                                   'completed': results, 'deadline': deadline})
        combined = read_json(root / 'STATUS.json')
        combined.update(phase='evaluation' if pending else 'complete', updated_at=utc())
        for arm, row in combined['arms'].items():
            row['phase'] = 'scoring' if arm in pending else (
                'complete' if 'metrics' in results.get(arm, {}) else 'evaluation_failed')
            row['evaluation_status_path'] = str(root / arm / 'evaluation/STATUS.json')
            if arm in results:
                row['evaluation_result'] = results[arm]
        atomic_json(root / 'STATUS.json', combined)
        if pending:
            time.sleep(20)
    atomic_json(root / 'EVALUATION_COMPLETE.json', {'completed_at': utc(), 'arms': results,
                                                'success': all('metrics' in row for row in results.values())})
    return 0 if all('metrics' in row for row in results.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
