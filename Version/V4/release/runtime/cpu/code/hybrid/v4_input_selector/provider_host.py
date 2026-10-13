"""Finite same-host V4 guardian entry. No scheduler, requeue, retry or SSH API.

Native submit refuses an unresolved FrozenBank/config before spawning anything.
The separate explicit lifecycle-check mode has no bank/model/GPU capability.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

from .bank import BankLedger, BankSession, FrozenBank, checked_json, require
from .linux_provider import Guardian, ProviderProxy, SocketTransport, load_legacy, validate_config
from .provider_wire import Client, reply, validate_request
from .supervisor import process_identity, work_alarm
from .worker_contract import atomic_json, file_sha, recv_message


def boot():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def native_inputs(args, *, create=False):
    bank = FrozenBank(args.master, args.master_sha256)
    config = checked_json(dict(path=args.config, sha256=args.config_sha256))
    validate_config(config, args.config_sha256, bank)
    ledger = BankLedger(bank, create=create)
    return bank, ledger, config


def signal_child(child, identity, sig):
    # This Popen child is deliberately not polled/reaped before the last signal.
    require(process_identity(child.pid) == identity and identity['pid'] == identity['pgid'] == identity['sid'], 'Host child identity changed; refuse signal')
    os.killpg(child.pid, sig)


def dispatch(guardian, operation, args, deadline):
    if operation == 'release':
        require(set(args) == {'allocation'}, 'Malformed release request')
        return guardian.release(args['allocation'], deadline)
    require(guardian.final is None, 'Guardian allocation already finalized')
    if operation == 'arm':
        require(set(args) in ({'phase','work_end'}, {'phase','work_end','item_id'}), 'Malformed arm request')
        require(args['work_end'] <= deadline, 'RPC expires before its armed work deadline')
        with guardian.alarm(deadline):
            return guardian.arm(args['phase'], args['work_end'], args.get('item_id'))
    end = min(deadline, guardian.work_end)
    require(end > time.monotonic(), 'Guardian work deadline reached')
    with guardian.alarm(end):
        if operation == 'lease_enter' and set(args) == {'allocation'}:
            # enter/fresh/inspect/start carry their own guards; avoid nested alarms.
            pass
        elif operation == 'ipc' and not args:
            require(guardian.ipc is not None, 'No owned IPC directory')
            return dict(path=str(guardian.ipc))
        elif operation not in ('fresh','inspect_owned','start_container'):
            raise ValueError('Unknown guardian operation')
    if operation == 'lease_enter':
        return guardian.enter(args['allocation'], end)
    if operation == 'fresh' and set(args) == {'uuid'}:
        return guardian.fresh(args['uuid'], end)
    if operation == 'inspect_owned' and set(args) == {'allocation'}:
        return guardian.inspect_owned(args['allocation'], end)
    if operation == 'start_container' and not args:
        return guardian.start_container(end)
    raise ValueError('Malformed guardian arguments')


def monitor(guardian, channel, child, identity):
    previous, interrupted = 0, False
    cause, error = 'host_exit', None
    try:
        while True:
            end = guardian.hard_end if guardian.final is not None else guardian.work_end + (25 if interrupted else 0)
            try:
                request = recv_message(channel, deadline=min(end,guardian.global_end))
            except (TimeoutError,socket.timeout):
                if guardian.final is not None:
                    cause = 'host_persistence_deadline'
                    break
                if not interrupted:
                    signal_child(child, identity, signal.SIGTERM)
                    interrupted = True
                    cause = 'original_work_deadline'
                    continue
                cause = 'original_inner_cleanup_deadline'
                break
            previous = validate_request(request, previous)
            try:
                require(not interrupted or request['operation'] == 'release', 'Watchdog interrupted work; cleanup only')
                result = dispatch(guardian,request['operation'],request['arguments'],request['deadline'])
                reply(channel,request,result=result)
            except BaseException as exc:
                reply(channel,request,error=repr(exc))
    except EOFError:
        cause = 'host_channel_eof'
    except BaseException as exc:
        error = repr(exc)
        cause = 'guardian_control_failure'
    finally:
        channel.close()
        if guardian.allocation is not None and guardian.final is None:
            # Missing host cannot defer cleanup to a new clock/watchdog run.
            guardian.release(guardian.allocation,min(guardian.hard_end-5,guardian.clock()+30))
        remaining = min(guardian.hard_end,guardian.global_end) - time.monotonic()
        if remaining > 0:
            try:
                # If wait succeeds, never signal the reaped PID. On timeout the
                # direct Popen child still reserves its identity for one signal.
                child.wait(timeout=min(.25,remaining))
            except subprocess.TimeoutExpired:
                try:
                    signal_child(child,identity,signal.SIGKILL)
                    child.wait(timeout=max(.001,min(1,min(guardian.hard_end,guardian.global_end)-time.monotonic())))
                except BaseException as exc:
                    error = error or repr(exc)
            except BaseException as exc:
                error = error or repr(exc)
        receipt = dict(schema='v4_guardian_exit_v1',cause=cause,error=error,host_returncode=child.returncode,
                       host_identity=identity,global_end=guardian.global_end,hard_end=guardian.hard_end,
                       finalization=guardian.final,finished_monotonic=time.monotonic())
        with guardian.alarm(min(guardian.hard_end,guardian.global_end)):
            atomic_json(guardian.output/'GUARDIAN_EXIT.json',receipt)
    return receipt


def base_argv(args, mode):
    result = [sys.executable,'-B','-m','hybrid.v4_input_selector.provider_host',mode,
              '--config',args.config,'--config-sha256',args.config_sha256,'--job-dir',args.job_dir]
    for key in ('master','master_sha256','phase'):
        if getattr(args,key,None):
            result += ['--'+key.replace('_','-'),getattr(args,key)]
    if args.resume_unstarted:
        result.append('--resume-unstarted')
    return result


def spawn_host(guardian, argv):
    parent, child_socket = socket.socketpair()
    logs = [(guardian.output/name).open('xb') for name in ('HOST_STDOUT.txt','HOST_STDERR.txt')]
    argv += ['--provider-fd',str(child_socket.fileno()),'--global-end',str(guardian.global_end),
             '--initial-work-end',str(guardian.work_end),'--boot-id',boot()]
    try:
        child = subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=logs[0],stderr=logs[1],start_new_session=True,
                                 pass_fds=(child_socket.fileno(),),env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),cwd=Path(__file__).resolve().parents[2])
        identity = process_identity(child.pid)
        atomic_json(guardian.output/'HOST_PROCESS.json',dict(identity=identity,guardian=process_identity(os.getpid()),boot_id=boot(),global_end=guardian.global_end))
    finally:
        child_socket.close()
        for stream in logs:
            stream.close()
    return monitor(guardian,parent,child,identity)


def native_guardian(args):
    import fcntl
    bank, ledger, config = native_inputs(args)
    require(args.boot_id == boot() and math.isfinite(args.global_end) and args.global_end <= ledger.remaining()['deadline_mono'], 'Original host clock/boot differs')
    output = Path(args.job_dir).resolve()
    legacy = load_legacy(config['legacy_root'],config['legacy_files'])
    with (bank.root/'.host-job.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        guardian = Guardian(bank,ledger,config,legacy,output,args.global_end)
        receipt = spawn_host(guardian,base_argv(args,'host-worker'))
    return 0 if receipt['error'] is None and receipt['finalization'] and not receipt['finalization']['error'] else 1


def host_worker(args):
    bank, ledger, config = native_inputs(args)
    require(args.boot_id == boot(), 'Host reboot; refuse budget continuation')
    channel = socket.socket(fileno=args.provider_fd)
    proxy = ProviderProxy(bank,ledger,channel,args.initial_work_end,args.global_end)
    try:
        with BankSession(ledger,proxy.adapter(),resume_unstarted=args.resume_unstarted) as session:
            result = session.run_phase(args.phase)
        # Queue/bank journals already carry the outcome within their five-second
        # tail. Do not append an unbudgeted host result after run_phase returns.
        return 0 if result['state'] == 'completed' else 1
    finally:
        channel.close()


def submit(args):
    bank, ledger, config = native_inputs(args,create=args.create_ledger)
    output = Path(args.job_dir).resolve()
    require(output.is_relative_to(bank.root/'host_jobs') and output != bank.root/'host_jobs', 'Host job output outside canonical bank host_jobs')
    require(ledger.snapshot()['active'] is None, 'Unresolved allocation; inspect existing job, never resubmit')
    output.mkdir(parents=True,exist_ok=False)
    end = ledger.remaining()['deadline_mono']
    argv = base_argv(args,'guardian') + ['--global-end',str(end),'--boot-id',boot()]
    with (output/'GUARDIAN_STDOUT.txt').open('xb') as stdout, (output/'GUARDIAN_STDERR.txt').open('xb') as stderr:
        process = subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),cwd=Path(__file__).resolve().parents[2])
    record = dict(schema='v4_host_submission_v1',state='submitted_not_completed',pid_identity=process_identity(process.pid),
        boot_id=boot(),global_end=end,master=args.master,master_sha256=args.master_sha256,
        config=args.config,config_sha256=args.config_sha256,phase=args.phase,resume_unstarted=args.resume_unstarted)
    atomic_json(output/'SUBMITTED.json',record)
    print(json.dumps(record))
    return 0


def probe_config(args):
    value = checked_json(dict(path=args.config,sha256=args.config_sha256))
    require(set(value) == {'schema','code_root','code_files','legacy_root','legacy_files'} and value['schema'] == 'v4_cpu_provider_probe_v1', 'CPU probe config must not contain native/GPU/master/asset settings')
    for name,h in value['code_files'].items():
        path = (Path(value['code_root'])/name).resolve()
        require(path.is_relative_to(Path(value['code_root']).resolve()) and path.suffix == '.py' and file_sha(path) == h, 'CPU probe code differs')
    value.update(resources=dict(cpus=1,ram_gib=1,swap_gib=0,shm_gib=1))
    return value


def probe_guardian(args):
    config = probe_config(args)
    output = Path(args.job_dir).resolve()
    allocation = args.probe_allocation
    root = output/'synthetic_bank'
    run = root/'runs'/allocation
    (run/'control').mkdir(parents=True,exist_ok=False)
    (run/'output').mkdir()
    (output/'cpu_leases').mkdir()
    bank = SimpleNamespace(root=root,master=dict(bank_id='cpu-provider-'+allocation,lease_directory=str(output/'cpu_leases')),
                           private_paths=[output/'HOST_PRIVATE_CANARY.json'])
    atomic_json(bank.private_paths[0],dict(host_only=True))
    legacy = load_legacy(config['legacy_root'],config['legacy_files'])
    guardian = Guardian(bank,None,config,legacy,output,args.global_end,probe=True)
    guardian.hard_end = min(args.global_end,time.monotonic()+100)
    guardian.work_end = guardian.hard_end-60
    argv = base_argv(args,'probe-worker') + ['--probe-allocation',allocation,'--probe-case',args.probe_case]
    receipt = spawn_host(guardian,argv)
    protocol_path = output/'PROBE_PROTOCOL.json'
    protocol = json.loads(protocol_path.read_bytes()) if protocol_path.is_file() else None
    success = probe_result_ok(receipt,protocol,args.probe_case)
    return 0 if success else 1


def probe_result_ok(receipt, protocol, case):
    """Physical cleanup alone is not a successful two-echo scenario."""
    if case not in ('normal','host-loss') or not isinstance(protocol,dict):
        return False
    final = receipt.get('finalization') or {}
    rows, ready = protocol.get('outputs'), protocol.get('ready') or {}
    return (receipt.get('error') is None and receipt.get('host_returncode') == (0 if case=='normal' else 71)
        and final.get('container_absent') is True and final.get('lease_released') is True and final.get('error') is None
        and protocol.get('model_loads') == 0 and protocol.get('gpu_queries') == 0
        and ready.get('pid') == 1 and ready.get('model_loads') == 0 and ready.get('probe') is True
        and isinstance(rows,list) and len(rows)==2
        and all(isinstance(row,dict) and row.get('pid')==ready['pid'] and row.get('ordinal')==i
            and row.get('item_id')=='echo_'+str(i) and row.get('model_loads')==0 and row.get('probe') is True
            for i,row in enumerate(rows,1)))


def probe_worker(args):
    channel = socket.socket(fileno=args.provider_fd)
    client = Client(channel)
    proxy = SimpleNamespace(rpc=client,initial_work_end=args.initial_work_end,global_end=args.global_end,probe=True)
    client.call('lease_enter',dict(allocation=args.probe_allocation),args.initial_work_end)
    inspected = client.call('inspect_owned',dict(allocation=args.probe_allocation),args.initial_work_end)
    transport = SocketTransport(proxy,{})
    ready = transport.start(args.initial_work_end)
    outputs = [transport.page(name,min(time.monotonic()+10,args.global_end-60)) for name in ('echo_1','echo_2')]
    require(all(x['pid']==ready['pid'] and x['model_loads']==0 for x in outputs) and [x['ordinal'] for x in outputs]==[1,2], 'CPU bridge was not persistent or echo-only')
    atomic_json(Path(args.job_dir)/'PROBE_PROTOCOL.json',dict(ready=ready,outputs=outputs,container=inspected['container_id'],gpu_queries=0,model_loads=0))
    if args.probe_case == 'host-loss':
        os._exit(71)  # explicit CPU fixture fault; guardian must independently clean
    cleanup = transport.stop(min(time.monotonic()+25,args.global_end-35))
    require(cleanup['group_absent'] is True and not cleanup['cleanup_error'], 'CPU bridge stop failed')
    release = client.call('release',dict(allocation=args.probe_allocation),min(time.monotonic()+30,args.global_end-5))
    require(release['container_absent'] and release['lease_released'] and not release['error'], 'CPU provider release failed')
    channel.close()
    return 0


def lifecycle_check(args):
    probe_config(args)
    output = Path(args.job_dir).resolve()
    output.mkdir(parents=True,exist_ok=False)
    overall = time.monotonic()+210
    results = []
    for case in ('normal','host-loss'):
        folder = output/case
        folder.mkdir()
        allocation = uuid.uuid4().hex
        copied = argparse.Namespace(**vars(args))
        copied.job_dir = str(folder)
        case_end = min(overall,time.monotonic()+100)
        argv = base_argv(copied,'probe-guardian') + ['--global-end',str(case_end),
            '--probe-allocation',allocation,'--probe-case',case]
        with (folder/'GUARDIAN_STDOUT.txt').open('xb') as stdout, (folder/'GUARDIAN_STDERR.txt').open('xb') as stderr:
            process = subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True,cwd=Path(__file__).resolve().parents[2])
        atomic_json(folder/'SUBMITTED.json',dict(identity=process_identity(process.pid),global_end=case_end,probe=True,case=case))
        code = process.wait(timeout=max(.001,overall-time.monotonic()))
        receipt = json.loads((folder/'GUARDIAN_EXIT.json').read_bytes())
        protocol = json.loads((folder/'PROBE_PROTOCOL.json').read_bytes())
        require(code == 0 and probe_result_ok(receipt,protocol,case), 'CPU provider lifecycle failed; do not retry')
        results.append(dict(case=case,exit_code=code,receipt_sha256=file_sha(folder/'GUARDIAN_EXIT.json')))
    atomic_json(output/'LIFECYCLE_RESULT.json',dict(schema='v4_cpu_provider_lifecycle_v1',cases=results,model_loads=0,gpu_queries=0,training=False,original_end=overall,finished=time.monotonic()))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['submit','guardian','host-worker','lifecycle-check','probe-guardian','probe-worker'])
    for name in ('config','config-sha256','job-dir'):
        parser.add_argument('--'+name,required=True)
    for name in ('master','master-sha256','phase','boot-id','probe-allocation','probe-case'):
        parser.add_argument('--'+name)
    for name in ('global-end','initial-work-end'):
        parser.add_argument('--'+name,type=float)
    parser.add_argument('--provider-fd',type=int)
    parser.add_argument('--resume-unstarted',action='store_true')
    parser.add_argument('--create-ledger',action='store_true')
    args = parser.parse_args()
    require(sys.platform == 'linux', 'Linux provider entry requires Linux')
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    def cancelled(*_):
        raise InterruptedError('Host job cancelled')
    signal.signal(signal.SIGTERM,cancelled)
    if args.mode in ('submit','guardian','host-worker'):
        require(args.master and args.master_sha256 and args.phase in ('S1','S2'), 'Exact accepted master/phase required')
    else:
        require(not any((args.master,args.master_sha256,args.phase,args.resume_unstarted,args.create_ledger)), 'CPU probe cannot accept native bank arguments')
    if args.mode in ('guardian','host-worker','probe-guardian','probe-worker'):
        require(args.global_end is not None and math.isfinite(args.global_end) and args.global_end > time.monotonic(), 'Missing original finite guardian deadline')
    return {'submit':submit,'guardian':native_guardian,'host-worker':host_worker,'lifecycle-check':lifecycle_check,
            'probe-guardian':probe_guardian,'probe-worker':probe_worker}[args.mode](args)


if __name__ == '__main__':
    raise SystemExit(main())
