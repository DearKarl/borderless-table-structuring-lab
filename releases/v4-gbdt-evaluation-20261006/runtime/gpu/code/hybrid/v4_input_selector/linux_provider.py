"""Concrete V31 lease/manual-device provider with an independent host guardian.

Importing this module performs no process, Docker, GPU or model operation.
"""
from datetime import datetime, timezone
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

from .bank import ExistingRuntimeAdapter, OWNER, checked_json, no_unknowns, require
from .provider_wire import Client
from .supervisor import work_alarm
from .worker_contract import atomic_json, file_sha

LEGACY_FILES = {'__init__.py', 'core.py', 'lifecycle.py', 'gpu_backend.py', 'gpu_admission.py', 'image_identity.py', 'bounded_cleanup.py'}
PROVIDER_FILES = {'linux_provider.py', 'provider_wire.py', 'provider_bridge.py', 'provider_host.py'}
APPROVED_UUIDS = {'GPU-PUBLIC-REQUIRES-BINDING-1', 'GPU-PUBLIC-REQUIRES-BINDING-2'}
NATIVE_IMAGE = 'sha256:5dd2b6a864a9565cde9216c8da3db3a8b05c93cbdca9ef417f3956841ce0722f'


def load_legacy(root, files):
    root = Path(root).resolve()
    require(set(files) == LEGACY_FILES, 'Closed seven-file V31 reuse lock required')
    for name, expected in files.items():
        require((root/name).is_file() and (root/name).stat().st_size <= 1024*1024 and file_sha(root/name) == expected,
                'Reused V31 source differs: ' + name)
    package = '_v4_locked_v31'
    require(not any(k == package or k.startswith(package + '.') for k in sys.modules), 'V31 reuse namespace already loaded')
    spec = importlib.util.spec_from_file_location(package, root/'__init__.py', submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    spec.loader.exec_module(module)
    return SimpleNamespace(**{name:importlib.import_module(package + '.' + name) for name in
        ('core','lifecycle','gpu_backend','gpu_admission','image_identity','bounded_cleanup')})


def validate_config(config, config_sha, bank):
    require(set(config) == {'schema','code_root','code_files','legacy_root','legacy_files','roots','driver_binding','resources'}, 'Provider config fields differ')
    require(config['schema'] == 'v4_linux_provider_v1' and no_unknowns(config), 'Provider config unresolved')
    receipt = checked_json(checked_json(bank.master['readiness'])['evidence']['launch_adapter'])
    require(receipt.get('schema') == 'v4_linux_provider_acceptance_v1' and receipt.get('owner') == OWNER
            and receipt.get('provider_config_sha256') == config_sha, 'Root has not accepted this exact provider config')
    require(bank.master['gpu_uuid'] in APPROVED_UUIDS, 'GPU UUID is outside the two explicitly approved devices')
    runtime = bank.template['runtime_binding']
    require(runtime['external_image']['image'] == NATIVE_IMAGE and runtime['interpreter']['executable'] == '/opt/v31-native/bin/python', 'Unreviewed native image/interpreter')
    require(set(config['roots']) == {'vendor','model','aux'} and set(config['resources']) == {'cpus','ram_gib','swap_gib','shm_gib'}, 'Provider roots/resource fields differ')
    for key, value in config['resources'].items():
        require(type(value) in (int,float) and math.isfinite(value) and value >= 0 and (key == 'swap_gib' or value > 0), 'Invalid approved resource limit')
    root = Path(config['code_root']).resolve()
    require(str(root) == config['code_root'] and root.is_dir(), 'Dedicated canonical code staging required')
    require(Path(__file__).resolve() == root/'hybrid/v4_input_selector/linux_provider.py', 'Actual provider module is outside frozen code staging')
    required = {'hybrid/v4_input_selector/' + name for name in PROVIDER_FILES}
    require(required <= set(config['code_files']) and len(config['code_files']) <= 64, 'Provider code lock incomplete/excessive')
    for name, expected in config['code_files'].items():
        path = (root/name).resolve()
        require(path.is_relative_to(root) and path.suffix == '.py' and path.stat().st_size <= 1024*1024
                and file_sha(path) == expected and bank.adapter_sources.get(str(path)) == expected, 'Unfrozen provider/worker code')
    # A dedicated small code package must not hide a reference or private file.
    observed = set()
    for directory, folders, files in os.walk(root, followlinks=False):
        require(len(Path(directory).relative_to(root).parts) <= 4 and not any((Path(directory)/n).is_symlink() for n in folders + files), 'Code staging link/depth differs')
        for name in files:
            observed.add((Path(directory)/name).relative_to(root).as_posix())
            require(len(observed) <= 64, 'Dedicated code staging inventory exceeds cap')
    require(observed == set(config['code_files']), 'Unexpected file in inference code mount')
    for name, expected in config['legacy_files'].items():
        path = str((Path(config['legacy_root'])/name).resolve())
        require(bank.adapter_sources.get(path) == expected, 'Legacy source not bound by accepted adapter sources')
    roots = [root, bank.input_root] + [Path(p).resolve() for p in config['roots'].values()]
    require(all(p.is_absolute() and p.is_dir() for p in roots), 'Missing inference root')
    require(not any(a == z or a.is_relative_to(z) or z.is_relative_to(a) for i,a in enumerate(roots) for z in roots[i+1:]), 'Inference roots must be disjoint')
    require(not any(p == r or p.is_relative_to(r) for p in bank.private_paths for r in roots), 'Private bank/reference path enters inference')
    return config


class Guardian:
    """One allocation, one child host job, one lease set and one owned container.

Only this process owns the real Leases/Admission objects and destructive Docker
operations. The inference host communicates by an inherited private socketpair.
"""
    def __init__(self, bank, ledger, config, legacy, output, global_end, *, probe=False,
                 runner=subprocess.run, clock=time.monotonic, wall=time.time, alarm=work_alarm):
        self.bank, self.ledger, self.config, self.v31 = bank, ledger, config, legacy
        self.output, self.global_end, self.probe = Path(output), global_end, probe
        self.runner, self.clock, self.wall, self.alarm = runner, clock, wall, alarm
        self.hard_end = min(global_end, clock() + 600)
        self.work_end = self.hard_end - 60
        self.lease = self.admission = self.owned = self.binding = None
        self.allocation = self.ipc = self.image_identity = None
        self.final = None
        self.mounts = []
        self.started = False
        self.lease_entered = False
        self.initial_armed = False
        self.armed_pages = set()

    def deadline(self, end):
        # Reuse V31's deadline API without invoking its 240-second emergency clock.
        return self.v31.core.Deadline(0, end, 0, self.clock)

    def command(self, args, end, cap=5):
        return self.runner(args, capture_output=True, text=True, check=False, timeout=self.deadline(end).bound(cap))

    def enter(self, allocation, end):
        require(self.lease is None and self.allocation is None and re.fullmatch('[0-9a-f]{32}', allocation), 'One canonical allocation per guardian')
        if not self.probe:
            self.ledger.require_allocation(self.ledger.snapshot(), allocation)
        self.allocation = allocation
        keys = ['cpu-v4-' + allocation] if self.probe else [self.bank.master['gpu_uuid']]
        self.lease = self.v31.lifecycle.Leases(self.bank.master['lease_directory'], keys)
        with self.alarm(end):
            self.lease.__enter__()
            self.lease_entered = True
            self.admission = self.v31.gpu_admission.Admission(self.bank.master['lease_directory'], keys)
            self.admission.reserve(allocation, [allocation])
        return dict(entered=True, keys=keys)

    def fresh(self, uid, end):
        require(self.lease_entered and self.admission is not None and self.final is None, 'No active shared admission')
        require(not self.probe and uid == self.bank.master['gpu_uuid'], 'Native probe identity differs')
        backend = self.v31.gpu_backend
        with self.alarm(end):
            descriptor = backend.descriptor(checked_json(self.config['driver_binding']))
            driver = backend.verify_host_driver(descriptor, self.deadline(end))
            expected = self.bank.template['runtime_binding']['device']
            cuda = next(row for row in descriptor['files'] if row['soname'] == 'libcuda.so.1')
            require(expected['driver_path'] == '/driver/' + cuda['target_basename'] and expected['driver_sha256'] == cuda['sha256'], 'Worker/host driver binding differs')
            xml = backend.query_xml([uid], 0, driver['driver_version'], self.deadline(end), runner=self.runner)
            observed = xml['gpus'][uid]
            devices = backend.device_stats(observed['minor'])
            backend.validate_devices(devices, observed['minor'])
            self.binding = dict(uuid=observed['uuid'], minor=observed['minor'], devices=devices, driver=driver)
            atomic_json(self.output/'MANUAL_ADMISSION.json', dict(binding=self.binding, xml=xml))
        return dict(uuid=observed['uuid'], memory_used_mib=observed['memory_mib'], compute_pids=observed['processes'], observed_at=self.wall())

    def inspect_state(self, state):
        require(state['Id'] == self.owned.cid and state['Config'].get('Labels', {}).get('hybrid-v3-owner') == self.allocation
                and state['Config']['Labels'].get('v4-bank') == self.bank.master['bank_id'], 'Actual CID/owner/bank labels differ')
        self.v31.lifecycle.verify_isolation(state, self.mounts, self.config['resources'], NATIVE_IMAGE,
            [] if self.probe else [self.binding['uuid']], self.image_identity, backend_binding=None if self.probe else self.binding)
        hc = state['HostConfig']
        require(not hc.get('AutoRemove') and not hc.get('Init') and hc.get('RestartPolicy', {}).get('Name') in ('no','')
                and hc.get('PidMode','') != 'host' and hc.get('IpcMode','') != 'host'
                and not hc.get('Binds') and not hc.get('VolumesFrom') and not hc.get('DeviceCgroupRules'), 'Unexpected lifecycle/namespace/volume configuration')
        require(state['Path'] == '/opt/v31-native/bin/python' and state['Args'] == self.container_args[1:], 'Actual container entrypoint differs')
        require(state['Config'].get('User') == str(os.getuid())+':'+str(os.getgid()), 'Container/host socket owner differs')
        if self.probe:
            require(not hc.get('Devices') and not hc.get('DeviceRequests'), 'CPU lifecycle probe exposed a GPU')
        actual = [(m['Source'],m['Destination'],m['RW']) for m in state['Mounts'] if m['Type']=='bind']
        out = str(self.bank.root/'runs'/self.allocation/'output')
        require(all(not writable or (source == out and target == out) for source,target,writable in actual), 'Only allocation output may be a writable bind')
        return actual

    def inspect_owned(self, allocation, end):
        require(allocation == self.allocation and self.owned is None and (self.probe or self.binding is not None), 'Acquisition order/identity differs')
        run = self.bank.root/'runs'/allocation
        with self.alarm(end):
            self.ipc = Path(tempfile.mkdtemp(prefix='v4p-', dir='/tmp')).resolve()
            os.chmod(self.ipc, 0o700)
            require(len(str(self.ipc/'control.sock').encode()) < 108, 'Host Unix socket path exceeds bound')
            self.mounts = [(str(Path(self.config['code_root']).resolve()), self.config['code_root'], False),
                (str(run/'control'), str(run/'control'), False), (str(run/'output'), str(run/'output'), True),
                (str(self.ipc), '/v4-ipc', False)]
            roots = dict(control=str(run/'control'), output=str(run/'output'), input='/unmounted', vendor='/unmounted', model='/unmounted', aux='/unmounted')
            if not self.probe:
                roots.update(input=str(self.bank.input_root), **self.config['roots'])
                self.mounts.extend((path,path,False) for key,path in roots.items() if key in ('input','vendor','model','aux'))
                self.mounts.append(self.v31.gpu_backend.driver_mount(self.binding))
                # The already observed small weight receipt is a named read-only
                # file, not the private master or a directory-wide evidence mount.
                weight_ref = self.bank.template['runtime_binding']['weight_receipt']
                weight = checked_json(weight_ref)
                require(weight.get('schema') == 'v4_weight_observation_v1' and weight.get('sha256') == self.bank.template['model_files']['model.safetensors']
                        and weight.get('model_loads') == 0 and weight.get('path') == str(Path(roots['model'])/'model.safetensors'), 'Weight observation does not bind the staged native model')
                self.mounts.append((weight_ref['path'],weight_ref['path'],False))
            require(not any(private == Path(source) or private.is_relative_to(Path(source)) for private in self.bank.private_paths for source,_,_ in self.mounts), 'Private host data would be mounted')
            self.image_identity = self.v31.image_identity.resolve_image_identity(NATIVE_IMAGE, self.deadline(end), self.runner)
            self.container_args = ['/opt/v31-native/bin/python','-B','-m','hybrid.v4_input_selector.provider_bridge',
                '--global-end',str(self.global_end),'--initial-work-end',str(end)]
            for key,value in roots.items():
                self.container_args.extend(['--'+key,value])
            if self.probe:
                self.container_args.append('--lifecycle-probe')
            flags = self.v31.lifecycle.limits(self.config['resources'])
            flags += ['--restart','no','--init=false','--user',str(os.getuid())+':'+str(os.getgid()),'--label','v4-bank='+self.bank.master['bank_id'], '--env','PYTHONPATH='+self.config['code_root'],
                      '--env','PYTHONUNBUFFERED=1','--env','HF_HUB_OFFLINE=1','--env','TRANSFORMERS_OFFLINE=1','--env','CUDA_DEVICE_ORDER=PCI_BUS_ID']
            if self.probe:
                flags += ['--runtime','runc','--env','NVIDIA_VISIBLE_DEVICES=void','--env','CUDA_VISIBLE_DEVICES=','--env','NVIDIA_DRIVER_CAPABILITIES=']
            else:
                flags += self.v31.gpu_backend.launch_flags({'schema':2,'gpu_backend':{'kind':'manual','host_binding':self.config['driver_binding']}}, self.binding['uuid'], self.binding)
            flags += self.v31.lifecycle.mount_args(self.mounts, str(run/'output'))
            # Override an image entrypoint rather than silently appending to it.
            flags += ['--entrypoint', self.container_args[0], NATIVE_IMAGE, *self.container_args[1:]]
            self.owned = self.v31.lifecycle.OwnedDocker(allocation, self.deadline(end), self.output, runner=self.runner)
            self.owned.admission = self.admission
            self.owned.create(flags)
            state = self.owned.inspect()
            actual = self.inspect_state(state)
            atomic_json(self.output/'CONTAINER_INSPECT.json', state)
        return dict(job_id=state['Config']['Labels']['v4-bank'], image=state['Image'],
            gpu_uuid=None if self.probe else self.binding['uuid'], owner_token=state['Config']['Labels']['hybrid-v3-owner'],
            observed_at=self.wall(), container_id=state['Id'], inspected_at_utc=datetime.now(timezone.utc).isoformat(),
            mounts=[dict(source=a,target=z,writable=w) for a,z,w in actual])

    def arm(self, phase, deadline, item_id=None):
        require(self.final is None and deadline <= self.global_end - 60 and deadline <= self.clock() + 540, 'Watchdog deadline extension refused')
        if phase == 'startup':
            require(not self.started and not self.initial_armed and deadline <= self.work_end, 'Startup deadline cannot reset')
            self.initial_armed = True
        else:
            require(phase == 'page' and self.started, 'Unexpected watchdog phase')
            require(item_id not in self.armed_pages, 'Repeated page arm cannot reset its deadline')
            if not self.probe:
                state = self.ledger.snapshot()
                self.ledger.require_allocation(state, self.allocation)
                row = state['slots'].get(item_id, {})
                require(row.get('status') == 'uncertain' and row.get('reservation',{}).get('allocation') == self.allocation, 'Page was not durably charged before arm')
            self.armed_pages.add(item_id)
        self.work_end, self.hard_end = deadline, min(self.global_end, deadline + 60)
        atomic_json(self.output/'WATCH_DEADLINE.json', dict(global_end=self.global_end, work_end=self.work_end, hard_end=self.hard_end, phase=phase, item_id=item_id))
        return dict(work_end=self.work_end, hard_end=self.hard_end)

    def start_container(self, end):
        require(self.owned is not None and not self.started and self.final is None, 'Container startup state differs')
        with self.alarm(end):
            if not self.probe:
                xml = self.v31.gpu_backend.query_xml([self.binding['uuid']],0,self.binding['driver']['driver_version'],self.deadline(end),runner=self.runner)
                require(xml['gpus'][self.binding['uuid']]['minor'] == self.binding['minor'] and self.v31.gpu_backend.device_stats(self.binding['minor']) == self.binding['devices'], 'Fresh pre-start UUID/minor/device mapping changed')
                atomic_json(self.output/'GPU_BEFORE_START.json', xml)
            self.owned.deadline = self.deadline(end)
            self.owned.start()
            state = self.owned.inspect()
            self.inspect_state(state)
            require(state['State']['Running'] is True and type(state['State']['Pid']) is int and state['State']['Pid'] > 0, 'Container bridge PID unavailable')
            self.started = True
        return dict(pid=state['State']['Pid'], container_id=state['Id'])

    def release(self, allocation, end):
        require(allocation == self.allocation, 'Cleanup allocation differs')
        if self.final is not None:
            return self.final
        response_end = min(end, self.hard_end - 5, self.clock() + 30)
        # Physical cleanup and durable admission evidence finish with one second
        # left inside the same outer30 for the RPC response. No extra tail.
        end = response_end - 1
        result = dict(allocation=allocation, container_id=None if self.owned is None else self.owned.cid,
                      container_absent=False, lease_released=False, error=None, deadline=end,
                      response_deadline=response_end, events=[])
        self.final = result
        def call(args, reserve, cap=5):
            response = self.command(args, end-reserve, cap)
            result['events'].append(dict(args=args, returncode=response.returncode, stdout=response.stdout, stderr=response.stderr))
            return response
        try:
            with self.alarm(end):
                if self.owned is not None and self.owned.creation_attempted:
                    target = self.owned.cid or 'hybrid-v3-' + allocation
                    inspect = call(['docker','inspect',target], 24, 4)
                    # Recover a timed-out create only by exact name plus label/CID.
                    require(inspect.returncode == 0, 'No fresh owned identity for destructive cleanup')
                    state = json.loads(inspect.stdout)[0]
                    require(state['Config'].get('Labels',{}).get('hybrid-v3-owner') == allocation
                            and state['Config']['Labels'].get('v4-bank') == self.bank.master['bank_id']
                            and state.get('Image') == NATIVE_IMAGE
                            and state.get('Name') == '/hybrid-v3-' + allocation
                            and (self.owned.cid is None or state['Id'] == self.owned.cid), 'Foreign cleanup identity; no signals/removal')
                    self.owned.cid = result['container_id'] = state['Id']
                    cid = state['Id']
                    kill_error = None
                    if state['State']['Running']:
                        killed = call(['docker','kill',cid], 19, 4)
                        if killed.returncode != 0:
                            kill_error = dict(returncode=killed.returncode,stderr=killed.stderr)
                    while True:
                        stopped = call(['docker','inspect',cid], 14, 1)
                        require(stopped.returncode == 0, 'Owned stopped state unavailable')
                        stopped = json.loads(stopped.stdout)[0]
                        require(stopped['Id'] == cid and stopped['Config'].get('Labels',{}).get('hybrid-v3-owner') == allocation, 'Stopped ownership changed')
                        if stopped['State']['Running'] is False and stopped['State']['Pid'] == 0:
                            break
                        time.sleep(min(.05,self.deadline(end-14).remaining()))
                    if kill_error is not None:
                        require(stopped['State'].get('ExitCode') == 0, 'Owned kill failed without a proven normal-exit race')
                        result['recovered_kill_race'] = kill_error
                    require(call(['docker','rm',cid], 10, 3).returncode == 0, 'Owned removal failed')
                    for target in (cid, 'hybrid-v3-' + allocation):
                        absent = call(['docker','inspect',target], 5, 2)
                        require(self.v31.bounded_cleanup.inspect_absent(absent,target), 'CID/name absence is unconfirmed')
                result['container_absent'] = True
                if self.probe:
                    resource = dict(verified=True, probe_no_gpu=True)
                elif self.lease_entered:
                    xml = self.v31.gpu_backend.query_xml([self.bank.master['gpu_uuid']],0,
                        self.binding['driver']['driver_version'] if self.binding else checked_json(self.config['driver_binding'])['driver_version'],
                        self.deadline(end-2),runner=self.runner)
                    resource = dict(verified=True, xml=xml)
                else:
                    resource = dict(verified=True, never_acquired=True)
                if self.ipc is not None:
                    sock = self.ipc/'control.sock'
                    if sock.exists():
                        require(not sock.is_symlink() and stat.S_ISSOCK(sock.lstat().st_mode), 'Unexpected file in owned IPC directory')
                        sock.unlink()
                    self.ipc.rmdir()  # never recursive; unexpected files preserve the block
                # Persist physical absence before clearing the admission marker.
                # This receipt never pretends the lease has already been released.
                atomic_json(self.output/'PHYSICAL_RELEASE.json',dict(allocation=allocation,container_id=result['container_id'],
                    container_absent=True,resource=resource,events=result['events'],deadline=end,observed=self.clock()))
                if self.admission is not None:
                    result['admission'] = self.admission.finish(dict(resource_release_verified=True,budget_exceeded=False,release=resource))
                    require(result['admission']['released'], 'Admission block not cleared')
                if self.lease is not None:
                    self.lease.__exit__(None,None,None)
                    self.lease_entered = False
                result['lease_released'] = True
                require(self.clock() <= end, 'Owned cleanup deadline exceeded')
        except BaseException as exc:
            result['error'] = repr(exc)
            # reserve() already made a durable block before create. Do not clear
            # it, release the held lease, or spend a new deadline on error I/O.
        result['finished'] = self.clock()
        return result


class LeaseProxy:
    def __init__(self, provider, directory, keys):
        self.provider, self.directory, self.keys = provider, directory, keys

    def __enter__(self):
        self.provider.rpc.call('lease_enter',dict(allocation=self.provider.allocation),self.provider.initial_work_end)
        return self

    def __exit__(self, *args):
        proof = self.provider.release_result
        require(proof is not None and proof.get('lease_released') is True and not proof.get('error'), 'Guardian lease remains unresolved')


class SocketTransport:
    def __init__(self, provider, arguments):
        self.provider, self.arguments = provider, arguments
        self.listener = self.sock = self.client = None

    def start(self, deadline):
        p = self.provider
        deadline = min(deadline, p.initial_work_end)
        p.rpc.call('arm',dict(phase='startup',work_end=deadline),deadline)
        ipc = p.rpc.call('ipc',{},deadline)['path']
        self.listener = socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        self.listener.bind(str(Path(ipc)/'control.sock'))
        self.listener.listen(1)
        identity = p.rpc.call('start_container',{},deadline)
        self.listener.settimeout(max(.001,deadline-time.monotonic()))
        self.sock,_ = self.listener.accept()
        pid,uid,gid = struct.unpack('3i', self.sock.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,struct.calcsize('3i')))
        require(pid == identity['pid'], 'Unix peer is not the freshly inspected container bridge PID')
        self.client = Client(self.sock)
        args = {'lifecycle_probe':True} if getattr(p,'probe',False) else {k:self.arguments[k] for k in ('contract','contract_sha256','owner','logs','output_root')}
        return self.client.call('start',args,deadline)

    def page(self, item_id, deadline):
        deadline = min(deadline, self.provider.global_end - 60)
        self.provider.rpc.call('arm',dict(phase='page',item_id=item_id,work_end=deadline),deadline)
        return self.client.call('page',dict(item_id=item_id),deadline)

    def stop(self, deadline):
        try:
            if self.client is None:
                return dict(group_absent=False,cleanup_error='Bridge startup/connection unresolved',signals=[])
            return self.client.call('stop',{},deadline)
        finally:
            if self.sock is not None:
                self.sock.close()
            if self.listener is not None:
                self.listener.close()


class ProviderProxy:
    """The exact callbacks frozen by ExistingRuntimeAdapter; guardian owns lease."""
    def __init__(self, bank, ledger, channel, initial_work_end, global_end):
        self.bank, self.ledger, self.rpc = bank, ledger, Client(channel)
        self.initial_work_end = initial_work_end
        self.global_end = global_end
        self.allocation = None
        self.release_result = None

    def lease_factory(self, directory, keys):
        require(directory == self.bank.master['lease_directory'] and keys == [self.bank.master['gpu_uuid']], 'Proxy lease identity differs')
        self.allocation = self.ledger.snapshot()['active']['id']
        return LeaseProxy(self,directory,keys)

    def fresh_probe(self, uid, deadline):
        return self.rpc.call('fresh',dict(uuid=uid),min(deadline,self.initial_work_end))

    def inspect_owned(self, allocation, deadline):
        self.observed = self.rpc.call('inspect_owned',dict(allocation=allocation),min(deadline,self.initial_work_end))
        return self.observed

    def transport_factory(self, **arguments):
        require(arguments['container_id'] == self.observed['container_id'] and arguments['gpu_uuid'] == self.bank.master['gpu_uuid'], 'Transport CID/UUID differs from actual inspection')
        return SocketTransport(self,arguments)

    def release_owned(self, allocation, deadline):
        self.release_result = self.rpc.call('release',dict(allocation=allocation),deadline)
        return self.release_result

    def adapter(self):
        return ExistingRuntimeAdapter(self.bank,lease_factory=self.lease_factory,fresh_probe=self.fresh_probe,
            inspect_owned=self.inspect_owned,transport_factory=self.transport_factory,release_owned=self.release_owned)
