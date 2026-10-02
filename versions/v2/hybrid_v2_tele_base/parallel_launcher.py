"""Reserve the second arm, cancel the old serial queue, then run on GPU2. No container signals."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'hybrid-code'))
from hybrid_v2_tele_base.full_state import sha,read,exclusive_json
from hybrid_v2_tele_base import run_full as run
from parallel_contract import resource_only,serial_command,gpu_pair

BASE=run.BASE


def process(pid):
    root=Path('/proc')/str(pid)
    command=[x.decode() for x in (root/'cmdline').read_bytes().split(b'\0') if x]
    fields=(root/'stat').read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'command':command,'state':fields[0],'start_ticks':fields[19]}


def preflight(expected):
    if HERE!=BASE/'parallel-code-v1' or sha(HERE/'PARALLEL_PACKAGE_LOCK.json')!=expected:
        raise RuntimeError('Parallel package boundary or identity')
    for name,value in read(HERE/'PARALLEL_PACKAGE_LOCK.json').items():
        path=(HERE/name).resolve()
        if not path.is_relative_to(HERE) or sha(path)!=value:raise RuntimeError('Parallel package bytes changed')
    pair=read(HERE/'RESOURCE_PAIR.json');original=BASE/'full-code-v2'
    if sha(original/'SYSTEM_FREEZE.json')!=pair['arms']['tele_raw']['freeze_sha256']:
        raise RuntimeError('Original Tele freeze changed')
    parent=read(original/'SYSTEM_FREEZE.json');child=read(HERE/'hybrid-code/SYSTEM_FREEZE.json')
    for name,value in parent['code_files'].items():
        if sha(original/name)!=value:raise RuntimeError('Original Tele frozen code changed')
    resource_only(parent,child,pair['arms']['tele_raw']['gpu_uuid'],pair['arms']['hybrid_v2']['gpu_uuid'])
    frozen,pages,images=run.verify_code(HERE/'hybrid-code',pair['arms']['hybrid_v2']['freeze_sha256'])
    return pair,frozen,pages,images


def reserve(pair):
    """Stop only host supervisor scheduling for the atomic reservation window; Docker keeps running."""
    target=BASE/'full-v2/hybrid_v2';raw=BASE/'full-v2/tele_raw';pid=pair['original_supervisor_pid']
    if target.exists():raise RuntimeError('Hybrid directory already exists; launch is one-shot')
    expected=pair['original_supervisor_command'];fd=os.pidfd_open(pid);stopped=False
    receipt=BASE/'parallel-switch-v1';receipt.mkdir(exist_ok=False)
    before=process(pid);serial_command(before['command'],expected)
    started=time.time();error=None
    try:
        # A detached, PID-fd-pinned fail-safe also releases scheduling if this switch process is killed.
        watchdog="import os,signal,sys,time; time.sleep(10)\ntry: signal.pidfd_send_signal(int(sys.argv[1]),signal.SIGCONT)\nexcept ProcessLookupError: pass\n"
        subprocess.Popen([sys.executable,'-B','-c',watchdog,str(fd)],pass_fds=(fd,),start_new_session=True,
                         stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        signal.pidfd_send_signal(fd,signal.SIGSTOP);stopped=True
        deadline=time.monotonic()+5
        while process(pid)['state'] not in ('T','t'):
            if time.monotonic()>deadline:raise RuntimeError('Supervisor did not stop scheduling')
            time.sleep(.02)
        pinned=process(pid);serial_command(pinned['command'],expected)
        if pinned['start_ticks']!=before['start_ticks']:raise RuntimeError('Supervisor identity changed')
        if target.exists():raise RuntimeError('Serial arm already exists; no parallel launch')
        binding=read(raw/'RUN_BINDING.json')
        if binding['freeze_sha256']!=pair['arms']['tele_raw']['freeze_sha256']:raise RuntimeError('Raw run binding differs')
        sessions=sorted((raw/'sessions').iterdir());session=sessions[-1];start=read(session/'START.json')
        state=json.loads(run.query(['docker','inspect',start['cid']]))[0]
        if not state['State']['Running'] or not state['Id'].startswith(pair['original_container_prefix']):
            raise RuntimeError('Original raw container is not the expected running container')
        if state['Config']['Labels'].get('hybrid.arm')!='tele_raw' or state['Config']['Labels'].get('hybrid.freeze')!=binding['freeze_sha256']:
            raise RuntimeError('Raw container ownership differs')
        target.mkdir(exist_ok=False)
        exclusive_json(target/'PARALLEL_RESERVATION.json',{
            'resource_pair_sha256':sha(HERE/'RESOURCE_PAIR.json'),'old_supervisor':before,'raw_container':state['Id'],
            'raw_container_pid':state['State']['Pid'],'time':time.time(),
            'serial_queue_cancelled_by':'Existing-arm guard before any old Hybrid mutation, writer lease or GPU action',
            'expected_old_exit':'After raw COMPLETE, old --arm both rejects existing Hybrid directory before entering its monitor',
            'baseline_container_signalled':False,'baseline_freeze_changed':False})
        exclusive_json(target/'RUN_BINDING.json',{'freeze_sha256':pair['arms']['hybrid_v2']['freeze_sha256'],
            'arm':'hybrid_v2','input_manifest_sha256':binding['input_manifest_sha256']})
        exclusive_json(receipt/'RESERVED.json',{'supervisor_before':before,'raw_start':start,'raw_container_state':state['State'],
            'hybrid_reservation_sha256':sha(target/'PARALLEL_RESERVATION.json'),'resource_pair_sha256':sha(HERE/'RESOURCE_PAIR.json')})
    except BaseException as exc:error=repr(exc);raise
    finally:
        if stopped:signal.pidfd_send_signal(fd,signal.SIGCONT)
        os.close(fd)
        exclusive_json(receipt/'SUPERVISOR_RELEASE.json',{'pid':pid,'SIGCONT_sent':stopped,'elapsed_seconds':time.time()-started,
            'error':error,'container_stop_or_kill_sent':False,'time':time.time()})
    return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--lock-sha256',required=True)
    parser.add_argument('--action',choices=['launch','resume-hybrid'],required=True);args=parser.parse_args()
    pair,freeze,pages,images=preflight(args.lock_sha256)
    rows=[x.split(',') for x in run.query(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits']).splitlines()]
    hardware=gpu_pair(rows,pair['arms']['tele_raw']['gpu_uuid'],pair['arms']['hybrid_v2']['gpu_uuid'])
    with (BASE/'PARALLEL_SWITCH.lease').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action=='launch':
            before=run.idle();receipt=reserve(pair)
            exclusive_json(receipt/'HARDWARE.json',{'same_hardware':hardware,'hybrid_gpu_idle_before':before})
        else:
            reservation=read(BASE/'full-v2/hybrid_v2/PARALLEL_RESERVATION.json')
            if reservation['resource_pair_sha256']!=sha(HERE/'RESOURCE_PAIR.json'):raise RuntimeError('Wrong resume resource pair')
        return run.run_arm(HERE/'hybrid-code',freeze,pages,images,pair['arms']['hybrid_v2']['freeze_sha256'],'hybrid_v2',True)


if __name__=='__main__':raise SystemExit(main())
