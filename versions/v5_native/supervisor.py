"""Host controller for task-owned native-input workers; all bindings come from a frozen manifest.

The controller has no inference model and never reads annotation bodies. Only
the separately launched evaluation process receives the evaluator binding.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .io_utils import atomic_json, digest, finalize_page, process_identity, read_json, utc


def docker(*args, timeout=30):
    result = subprocess.run(['docker', *args], text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'Docker {args[0]} failed: {result.stderr[-2000:]}')
    return result.stdout.strip()


def gpu_snapshot():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                             '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True, timeout=10)
    gpus = []
    for line in result.stdout.splitlines():
        index, uuid, memory, utilization = [x.strip() for x in line.split(',')]
        gpus.append({'index': int(index), 'uuid': uuid, 'memory_used_mib': int(memory), 'utilization_percent': int(utilization)})
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                             '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True, timeout=10)
    processes = []
    for line in result.stdout.splitlines():
        fields = [x.strip() for x in line.split(',')]
        if len(fields) == 4:
            processes.append({'gpu_uuid': fields[0], 'pid': int(fields[1]), 'process_name': fields[2], 'memory_mib': fields[3]})
    return {'observed_at': utc(), 'gpus': gpus, 'processes': processes}


class Round:
    def __init__(self, path):
        self.manifest_path = Path(path).resolve()
        self.manifest = read_json(path)
        self.root = Path(self.manifest['round_root'])
        self.protocol = read_json(self.manifest['protocol_path'])
        self.allocations = []
        self.restarts = {arm: self.manifest.get('prior_infrastructure_restarts',{}).get(arm,0)
                         for arm in self.manifest['gpu_mapping']}
        self.errors = []
        self.cached_receipts = {arm: {} for arm in self.manifest['gpu_mapping']}
        self.last_gpu = None
        self.started_unix = None

    def verify(self):
        for path, expected in self.manifest['frozen_files'].items():
            if digest(path) != expected:
                raise RuntimeError('Frozen file changed: ' + path)
        observed = gpu_snapshot()
        for indices in self.manifest['gpu_mapping'].values():
            for index in indices:
                gpu = next(x for x in observed['gpus'] if x['index'] == index)
                if gpu['uuid'] != self.manifest['gpu_uuids'][str(index)]:
                    raise RuntimeError('GPU index/UUID mapping changed')
                if any(x['gpu_uuid'] == gpu['uuid'] for x in observed['processes']) or gpu['memory_used_mib'] > 20:
                    raise RuntimeError(f'GPU {index} is occupied; no other process will be changed')
        if len([gpu for indices in self.manifest['gpu_mapping'].values() for gpu in indices]) > 7:
            raise RuntimeError('GPU ceiling exceeded')
        atomic_json(self.root / 'ADMISSION.json', observed)

    def launch(self, arm, shard, index, attempt=0):
        m = self.manifest
        armroot = self.root / arm
        output = armroot / 'output'
        output.mkdir(parents=True, exist_ok=True)
        worker_id = f'shard-{shard:02d}-attempt-{attempt:02d}'
        control_dir = armroot / 'controls'
        control_dir.mkdir(parents=True, exist_ok=True)
        control_path = control_dir / (worker_id + '.json')
        pages = [page for i, page in enumerate(m['pages']) if i % len(m['gpu_mapping'][arm]) == shard]
        control = {'schema_version': 1, 'arm': arm, 'run_id': m['run_id'], 'pages': pages,
                   'protocol_path': '/code/versions/v5_native/protocol.json', 'protocol_sha256': digest(m['protocol_path']),
                   'model_path': '/model', 'input_root': '/inputs', 'physical_gpu_index': index,
                   'gpu_uuid': m['gpu_uuids'][str(index)], 'absolute_deadline_unix': m.get('absolute_deadline_unix', self.started_unix + m['wall_seconds']),
                   'page_hard_seconds': self.protocol['limits']['page_hard_seconds'], 'startup_hard_seconds': 900}
        if 'expected_common_runtime' in m:
            control['expected_common_runtime'] = m['expected_common_runtime']
        for key in ('worker_module','controllers','external_fixed_scales'):
            if key in m:control[key]=m[key]
        atomic_json(control_path, control)
        name = f"{m['run_id'].lower()}-{arm.lower().replace('.', '-')}-s{shard}-a{attempt}"
        argv = ['create', '--name', name, '--label', 'native-owner=' + m['run_id'], '--network', 'none',
                '--runtime', 'runc', '--restart', 'no', '--read-only', '--cap-drop', 'ALL',
                '--security-opt', 'no-new-privileges', '--cpus', '8', '--memory', '80g', '--memory-swap', '80g',
                '--pids-limit', '2048', '--shm-size', '8g', '--tmpfs', '/tmp:rw,size=8g', '--workdir', '/output',
                '--device', f'/dev/nvidia{index}', '--device', '/dev/nvidiactl', '--device', '/dev/nvidia-uvm',
                '--device', '/dev/nvidia-uvm-tools']
        mounts = [(m['code_root'], '/code', True), (str(control_dir), '/control', True),
                  (str(output), '/output', False), (m['model_root'], '/model', True), (m['input_root'], '/inputs', True),
                  (m['venv_root'], m['venv_root'], True), (m['base_environment'], m['base_environment'], True),
                  (m['driver_root'], '/driver', True)]
        if 'controller_root' in m:mounts.append((m['controller_root'],'/controllers',True))
        for source, target, readonly in mounts:
            argv += ['--mount', f'type=bind,src={source},dst={target}' + (',readonly' if readonly else '')]
        env = {'PYTHONPATH': '/code', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
               'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': '0',
               'LD_LIBRARY_PATH': '/driver:' + m['base_environment'] + '/lib',
               'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
               'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
               'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
               'VLLM_NO_USAGE_STATS': '1', 'VLLM_PLUGINS': 'TeleOCR_vllm',
               'HOME': '/output/cache/' + worker_id, 'HF_HOME': '/output/cache/' + worker_id + '/hf',
               'XDG_CACHE_HOME': '/output/cache/' + worker_id,
               'TRITON_CACHE_DIR': '/output/cache/' + worker_id + '/triton',
               'TORCHINDUCTOR_CACHE_DIR': '/output/cache/' + worker_id + '/inductor'}
        for key, value in env.items():
            argv += ['--env', key + '=' + value]
        argv += ['--entrypoint', m['venv_root'] + '/bin/python', m['image'], '-B', '-m', 'versions.v5_native.guardian',
                 '--control', '/control/' + control_path.name, '--output', '/output', '--worker-id', worker_id]
        cid = docker(*argv)
        inspect = __import__('json').loads(docker('inspect', cid))[0]
        if inspect['Config']['Labels']['native-owner'] != m['run_id'] or inspect['HostConfig']['NetworkMode'] != 'none':
            raise RuntimeError('Container ownership/isolation mismatch')
        actual_mounts = {(x['Source'], x['Destination'], not x['RW']) for x in inspect['Mounts'] if x['Type'] == 'bind'}
        if actual_mounts != set(mounts):
            raise RuntimeError('Container mount identity mismatch')
        directory = armroot / 'allocations' / worker_id
        directory.mkdir(parents=True, exist_ok=False)
        atomic_json(directory / 'CREATE.json', {'argv': ['docker', *argv], 'inspect': inspect, 'created_at': utc()})
        docker('start', cid)
        state = __import__('json').loads(docker('inspect', cid))[0]
        logfile = open(directory / 'worker.log', 'w')
        logproc = subprocess.Popen(['docker', 'logs', '--follow', cid], stdout=logfile, stderr=subprocess.STDOUT)
        allocation = {'arm': arm, 'shard': shard, 'index': index, 'uuid': control['gpu_uuid'], 'attempt': attempt,
                      'worker_id': worker_id, 'cid': cid, 'name': name, 'directory': str(directory),
                      'start_unix': time.time(), 'started_at': state['State']['StartedAt'],
                      'container_pid': state['State']['Pid'], 'host_identity': process_identity(state['State']['Pid']),
                      'ended_unix': None, 'exit_code': None, 'output': str(output), 'worker_status': None,
                      'allocation_state': 'started', 'peak_gpu_memory_mib': 0}
        self.allocations.append(allocation)
        atomic_json(directory / 'START.json', allocation)
        logfile.close()
        if m.get('task_budget_path'):
            budget_path = Path(m['task_budget_path'])
            budget = read_json(budget_path) if budget_path.exists() else {}
            if 'first_actual_gpu_container_started_at' not in budget:
                atomic_json(budget_path.with_name('BUDGET_PRE_GPU_ADMISSION_HISTORY.json'), budget)
                from datetime import datetime
                # Docker uses nanoseconds; Python 3.10 accepts at most microseconds.
                first = datetime.fromisoformat(state['State']['StartedAt'][:26] + '+00:00').timestamp()
                budget.update(first_actual_gpu_container_started_at=state['State']['StartedAt'],
                              first_actual_gpu_launch_unix=first, absolute_deadline_unix=first + 72 * 3600,
                              gpu_seconds_ceiling=96 * 3600, full_benchmark_reserved_gpu_seconds=24 * 3600)
                atomic_json(budget_path, budget)

    def stop(self, allocation, reason):
        if allocation['ended_unix'] is not None:
            return
        state = __import__('json').loads(docker('inspect', allocation['cid']))[0]
        if state['Config'].get('Labels', {}).get('native-owner') != self.manifest['run_id']:
            raise RuntimeError('Refusing to stop a container without this run ownership')
        if state['State']['Running']:
            docker('kill', allocation['cid'])
        atomic_json(Path(allocation['directory']) / 'STOP.json', {'reason': reason, 'at': utc()})

    def finalize_interrupted(self, allocation, reason):
        output = Path(allocation['output'])
        for start_path in (output / 'pages').glob('*/START.json'):
            record = read_json(start_path)
            page_id = start_path.parent.name
            if record['worker_id'] != allocation['worker_id'] or (output / 'receipts' / (page_id + '.json')).exists():
                continue
            record.update(status='hard_timeout' if reason == 'page_hard_deadline' else 'infrastructure_failure',
                          error={'type': reason, 'stage': None}, elapsed_seconds=time.monotonic() - record['start_monotonic'])
            if self.manifest.get('worker_module')=='versions.v5_native.shared_worker':
                from .io_utils import finalize_outputs
                finalize_outputs(start_path.parent,output,page_id,record,self.protocol['arms'])
            else:
                finalize_page(start_path.parent, output, page_id, record)

    def observe(self):
        active = [x for x in self.allocations if x['ended_unix'] is None]
        if not active:
            return
        states = __import__('json').loads(docker('inspect', *[x['cid'] for x in active]))
        by_id = {x['Id']: x for x in states}
        self.last_gpu = gpu_snapshot()
        for allocation in active:
            state = by_id[allocation['cid']]
            path = Path(allocation['output']) / 'workers' / allocation['worker_id'] / 'STATUS.json'
            if path.is_file():
                allocation['worker_status'] = read_json(path)
            memory = next(x['memory_used_mib'] for x in self.last_gpu['gpus'] if x['uuid'] == allocation['uuid'])
            allocation['peak_gpu_memory_mib'] = max(allocation['peak_gpu_memory_mib'], memory)
            if state['State']['Running']:
                continue
            allocation.update(ended_unix=time.time(), exit_code=state['State']['ExitCode'], allocation_state='stopped')
            atomic_json(Path(allocation['directory']) / 'EXIT.json', {'allocation': allocation, 'inspect': state, 'at': utc()})
            if allocation['exit_code'] == 0:
                continue
            hard = path.with_name('HARD_STOP.json')
            reason = read_json(hard)['reason'] if hard.is_file() else 'worker_exit_' + str(allocation['exit_code'])
            self.finalize_interrupted(allocation, reason)
            if reason == 'page_hard_deadline' and self.restarts[allocation['arm']] == 0:
                # The bounded replacement resumes only pages that never began.
                self.restarts[allocation['arm']] += 1
                atomic_json(Path(allocation['directory']) / 'RECOVERY.json', {
                    'diagnosis': 'Page exceeded container hard deadline after work timeout; owned engine was terminated.',
                    'action': 'Replace engine once; finalize timed-out page without rerunning it; continue untouched pages.', 'at': utc()})
                self.launch(allocation['arm'], allocation['shard'], allocation['index'], attempt=1)
            else:
                self.errors.append({'arm': allocation['arm'], 'worker_id': allocation['worker_id'], 'reason': reason, 'at': utc()})

    def snapshot(self):
        now = time.time()
        arms = {}
        for arm, indices in self.manifest['gpu_mapping'].items():
            output = self.root / arm / 'output'
            cache = self.cached_receipts[arm]
            for path in (output / 'receipts').glob('*.json'):
                if path.stem not in cache:
                    cache[path.stem] = read_json(path)
            records = list(cache.values())
            allocations = [x for x in self.allocations if x['arm'] == arm]
            completed = len(records)
            total = len(self.manifest['pages'])
            failures = sum(x['status'] != 'success' for x in records)
            active = [x for x in allocations if x['ended_unix'] is None]
            elapsed = max(0.001, now - self.started_unix)
            rate = completed / elapsed
            phase = 'inference_complete' if completed == total and not active else (
                'inference' if completed else 'model_loading')
            if any(x['arm'] == arm for x in self.errors):
                phase = 'blocked'
            arms[arm] = {'run_id': self.manifest['run_id'], 'phase': phase, 'total_pages': total,
                         'full_set_pages': 1651, 'completed_pages': completed, 'failed_pages': failures,
                         'empty_predictions': sum(x['empty_prediction'] for x in records), 'remaining_pages': total - completed,
                         'physical_gpu_indices': indices, 'gpu_uuids': [self.manifest['gpu_uuids'][str(i)] for i in indices],
                         'started_at': min((x['started_at'] for x in allocations), default=None),
                         'last_progress_at': max((x['completed_at'] for x in records), default=None),
                         'throughput_pages_per_second': rate,
                         'eta_seconds': (total - completed) / rate if completed else None,
                         'eta_basis': 'Completed pages / wall seconds since round launch, including engine startup; early estimate.',
                         'allocated_gpu_seconds': sum((x['ended_unix'] or now) - x['start_unix'] for x in allocations),
                         'peak_gpu_memory_mib': max((x['peak_gpu_memory_mib'] for x in allocations), default=0),
                         'infrastructure_restarts': self.restarts[arm], 'allocations': allocations,
                         'prediction_directory': str(output / 'markdown')}
        status = {'schema_version': 1, 'run_id': self.manifest['run_id'], 'phase': self.manifest['phase'],
                  'updated_at': utc(), 'supervisor_identity': process_identity(), 'remote_root': str(self.root),
                  'started_unix': self.started_unix, 'elapsed_wall_seconds': now - self.started_unix,
                  'absolute_deadline_unix': self.manifest.get('absolute_deadline_unix', self.started_unix + self.manifest['wall_seconds']),
                  'allocated_gpu_seconds': sum(x['allocated_gpu_seconds'] for x in arms.values()),
                  'arms': arms, 'gpu_observation': self.last_gpu, 'errors': self.errors}
        atomic_json(self.root / 'STATUS.json', status)
        if not (self.root / 'LAUNCH_RECEIPT.json').exists():
            observed_arms = [arm for arm, row in arms.items() if row['completed_pages'] and any(
                p['gpu_uuid'] in row['gpu_uuids'] for p in (self.last_gpu or {}).get('processes', []))]
            if len(observed_arms) == len(self.manifest['gpu_mapping']):
                atomic_json(self.root / 'LAUNCH_RECEIPT.json', {**status, 'evidence': 'Each configured track has a completed prediction and a live compute process on its assigned GPU(s).',
                            'stage': 'training_data_generation' if self.manifest['phase'] == 'pilot' else self.manifest['phase']})
        return status

    def run(self):
        self.root.mkdir(parents=True, exist_ok=True)
        lock = self.root / 'SUPERVISOR_START.json'
        with open(lock, 'x') as f:
            f.write('{}\n')
        self.verify()
        self.started_unix = time.time()
        atomic_json(lock, {'identity': process_identity(), 'started_at': utc(), 'started_unix': self.started_unix,
                           'manifest_sha256': digest(self.manifest_path)})
        try:
            for arm, indices in self.manifest['gpu_mapping'].items():
                for shard, index in enumerate(indices):
                    self.launch(arm, shard, index)
            while True:
                self.observe()
                status = self.snapshot()
                if status['allocated_gpu_seconds'] >= self.manifest['gpu_seconds_ceiling'] or time.time() >= status['absolute_deadline_unix']:
                    self.errors.append({'arm': 'all', 'reason': 'operational_ceiling', 'at': utc()})
                if self.errors:
                    for allocation in self.allocations:
                        self.stop(allocation, 'round_blocker')
                    time.sleep(2)
                    self.observe()
                    atomic_json(self.root / 'BLOCKER.json', self.snapshot())
                    return 1
                if all(x['ended_unix'] is not None for x in self.allocations):
                    if not all(row['completed_pages'] == len(self.manifest['pages']) for row in status['arms'].values()):
                        raise RuntimeError('Workers ended without retaining the complete denominator')
                    atomic_json(self.root / 'INFERENCE_COMPLETE.json', status)
                    return 0
                time.sleep(10)
        except BaseException:
            atomic_json(self.root / 'SUPERVISOR_ERROR.json', {'at': utc(), 'traceback': traceback.format_exc()})
            for allocation in self.allocations:
                try:
                    self.stop(allocation, 'supervisor_exception')
                except Exception:
                    pass
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    raise SystemExit(Round(parser.parse_args().manifest).run())
