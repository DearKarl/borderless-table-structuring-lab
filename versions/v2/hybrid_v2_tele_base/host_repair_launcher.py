"""Independent host collector repair. No frozen worker file or model recipe is changed."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import sys
import time
import traceback
from host_repair_policy import exclusive_bytes as short_exclusive_bytes,specific_host_fault,validate_binding_digests

HERE=Path(__file__).resolve().parent
BASE=Path('/srv/hybrid-research/inference/hybrid-v2-tele-base')


def process(pid):
    root=Path('/proc')/str(pid);fields=(root/'stat').read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'state':fields[0],'start_ticks':fields[19],
            'command':[x.decode() for x in (root/'cmdline').read_bytes().split(b'\0') if x]}


def takeover(run,state,pins,attempt):
    pid=pins['old_hybrid_supervisor_pid'];old=process(pid)
    if old['start_ticks']!=pins['old_hybrid_start_ticks'] or old['command']!=pins['old_hybrid_command']:
        raise RuntimeError('Old Hybrid supervisor identity differs')
    root=BASE/'full-v2/hybrid_v2';session=root/'sessions/session-0001';start=state.read(session/'START.json')
    if (session/'EXIT.json').exists():raise RuntimeError('Old Hybrid session already exited; diagnose before takeover')
    before=json.loads(run.query(['docker','inspect',start['cid']]))[0]
    identity=pins['arms']['hybrid_v2']['freeze_sha256']
    if not before['State']['Running'] or not before['Id'].startswith(pins['hybrid_container_prefix']):raise RuntimeError('Hybrid container is not running')
    if before['Config']['Labels'].get('hybrid.arm')!='hybrid_v2' or before['Config']['Labels'].get('hybrid.freeze')!=identity:
        raise RuntimeError('Hybrid container ownership differs')
    state.exclusive_json(attempt/'TAKEOVER_READY.json',{'old_supervisor':old,'container_before':before,
        'candidate_pid':os.getpid(),'new_collector_preflight_passed':True,'time':time.time()})
    fd=os.pidfd_open(pid);stopped=False;killed=False
    try:
        again=process(pid)
        if again['start_ticks']!=old['start_ticks'] or again['command']!=old['command']:raise RuntimeError('Supervisor identity changed')
        signal.pidfd_send_signal(fd,signal.SIGSTOP);stopped=True
        deadline=time.monotonic()+5
        while process(pid)['state'] not in ('T','t'):
            if time.monotonic()>deadline:raise RuntimeError('Supervisor did not stop scheduling')
            time.sleep(.02)
        if (session/'EXIT.json').exists():raise RuntimeError('Old monitor exited before takeover')
        # SIGKILL only this PID fd; SIGTERM/KeyboardInterrupt would run the old destructive finally.
        signal.pidfd_send_signal(fd,signal.SIGKILL);killed=True
        if not select.select([fd],[],[],10)[0]:raise RuntimeError('Old host has not exited')
    finally:
        if stopped and not killed:signal.pidfd_send_signal(fd,signal.SIGCONT)
        os.close(fd)
        state.exclusive_json(attempt/'HOST_SIGNAL_RECEIPT.json',{'old_pid':pid,'old_start_ticks':old['start_ticks'],
            'SIGSTOP_sent':stopped,'SIGKILL_sent_to_exact_host_only':killed,'container_signalled':False,
            'docker_attach_signalled':False,'time':time.time()})
    after=json.loads(run.query(['docker','inspect',start['cid']]))[0]
    if not after['State']['Running'] or after['Id']!=before['Id'] or after['State']['Pid']!=before['State']['Pid']:
        raise RuntimeError('Container changed during host takeover; do not initialize another model')
    state.exclusive_json(attempt/'CONTAINER_PRESERVED.json',{'cid':after['Id'],'native_pid_before':before['State']['Pid'],
        'native_pid_after':after['State']['Pid'],'running':True,'native_model_reinitialized':False,'time':time.time()})


def prepare_tele(run,state,pins,pages,attempt):
    root=BASE/'full-v2/tele_raw';session=root/'sessions/session-0001';identity=pins['arms']['tele_raw']['freeze_sha256']
    if Path('/proc/1708844').exists():raise RuntimeError('Original Tele host still exists; no concurrent recovery')
    with (root/'WRITER.lease').open('a') as writer:
        fcntl.flock(writer,fcntl.LOCK_EX|fcntl.LOCK_NB)
        original=state.read(session/'EXIT.json');page=next(p for p in pages if p['key']=='p00205')
        specific_host_fault(original,identity,page['page_id'])
        actual=json.loads(run.query(['docker','inspect',original['cid']]))[0]
        if actual['State']['Running'] or actual['State']['OOMKilled'] or actual['State']['ExitCode']!=137:
            raise RuntimeError('Original stopped Tele container state differs')
        folder=session/'pages/p00205';result=state.read(folder/'PAGE_RESULT.json')
        for key in ('key','page_id','input_sha256'):
            if result[key]!=page[key]:raise RuntimeError('Reusable p00205 original input differs')
        if result['arm']!='tele_raw' or result['freeze_sha256']!=identity or result['status']!='success':raise RuntimeError('p00205 completion identity differs')
        expected=pins['p00205_prediction_sha256']
        if result['prediction_sha256']!=expected or result['bytes']!=1515:raise RuntimeError('p00205 receipt differs from owner snapshot')
        for path in (folder/'prediction.md',folder/'native/p00205/p00205.md'):
            if state.sha(path)!=expected:raise RuntimeError('p00205 native or primary bytes differ')
        before=len(list((root/'terminals').glob('*.json')))
        run.collect(root,session,state.read(session/'ASSIGNED.json')['pages'],identity,'tele_raw')
        remaining=state.pending_pages(root,pages,identity,'tele_raw')
        if any(p['key']=='p00205' for p in remaining):raise RuntimeError('p00205 was not recovered')
        completion=[{'key':p['key'],'page_id':p['page_id'],'receipt_sha256':state.sha(session/'pages'/p['key']/'PAGE_RESULT.json')}
                    for p in pages if (session/'pages'/p['key']/'PAGE_RESULT.json').exists()]
        unfinished=[p['key'] for p in pages if (session/'pages'/p['key']/'STARTED.json').exists() and not (session/'pages'/p['key']/'PAGE_RESULT.json').exists()]
        admission={'original_exit_sha256':state.sha(session/'EXIT.json'),'original_cid':original['cid'],'original_exit_preserved':True,
            'original_failure_reclassified_as_success':False,'arm':'tele_raw','freeze_sha256':identity,
            'reason':'Exact documented host errno36 temporary-filename defect, no model OOM; original native p00205 success reused',
            'original_terminal_count':before,'reconciled_terminal_count':1651-len(remaining),'pending_pages':len(remaining),
            'reused_completed_pages':completion,'interrupted_nonterminal_pages_preserved_and_eligible_to_continue':unfinished,
            'recovery_entry_sha256':state.sha(HERE/'launcher.py'),'policy_sha256':state.sha(HERE/'host_repair_policy.py')}
        proof=BASE/'host-repair-v3/tele_raw/SESSION_0001_ADMISSION.json'
        if proof.exists():
            saved=state.read(proof)
            for key in ('original_exit_sha256','original_cid','freeze_sha256','recovery_entry_sha256','policy_sha256'):
                if saved[key]!=admission[key]:raise RuntimeError('Existing specific recovery admission differs')
        else:state.exclusive_json(proof,admission)
        state.exclusive_json(attempt/'RECONCILED.json',admission)
    original_validator=run.validate_session_exit
    def validator(target):
        if Path(target).resolve()==session.resolve():
            saved=state.read(proof)
            if state.sha(session/'EXIT.json')!=saved['original_exit_sha256']:raise RuntimeError('Original failure evidence changed')
            specific_host_fault(state.read(session/'EXIT.json'),identity,page['page_id'])
            for item in saved['reused_completed_pages']:
                if state.sha(session/'pages'/item['key']/'PAGE_RESULT.json')!=item['receipt_sha256']:
                    raise RuntimeError('Recovered native receipt changed')
            return
        return original_validator(target)
    run.validate_session_exit=validator


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--lock-sha256',required=True)
    parser.add_argument('--action',choices=['recover-tele','takeover-hybrid','resume-hybrid'],required=True);args=parser.parse_args()
    import hashlib
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    if HERE!=BASE/'host-repair-code-v3' or sha(HERE/'CODE_LOCK.json')!=args.lock_sha256:raise RuntimeError('Repair package identity differs')
    for name,value in json.loads((HERE/'CODE_LOCK.json').read_bytes()).items():
        path=(HERE/name).resolve()
        if not path.is_relative_to(HERE) or sha(path)!=value:raise RuntimeError('Repair code changed')
    pins=json.loads((HERE/'BINDINGS.json').read_bytes());validate_binding_digests(pins)
    arm='tele_raw' if args.action=='recover-tele' else 'hybrid_v2'
    code=Path(pins['arms'][arm]['code']);identity=pins['arms'][arm]['freeze_sha256']
    sys.path.insert(0,str(code))
    from hybrid_v2_tele_base import run_full as run,full_state as state
    freeze,pages,images=run.verify_code(code,identity)
    # Process-local host injection only. Original worker source and frozen identities remain unchanged.
    state.exclusive_bytes=short_exclusive_bytes;run.exclusive_bytes=short_exclusive_bytes
    records=BASE/'host-repair-v3'/arm;records.mkdir(parents=True,exist_ok=True)
    with (records/'REPAIR.lease').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        attempts=sorted(records.glob('attempt-*'));attempt=records/('attempt-%04d'%(len(attempts)+1));attempt.mkdir(exist_ok=False)
        state.exclusive_json(attempt/'HOST_START.json',{'pid':os.getpid(),'action':args.action,'command':sys.argv,
            'freeze_sha256':identity,'code_lock_sha256':args.lock_sha256,'time':time.time()})
        try:
            if arm=='tele_raw':prepare_tele(run,state,pins,pages,attempt)
            elif args.action=='takeover-hybrid':takeover(run,state,pins,attempt)
            else:
                if Path('/proc/'+str(pins['old_hybrid_supervisor_pid'])).exists():raise RuntimeError('Old Hybrid host still exists; no unsafe resume')
                if not list(records.glob('attempt-*/HOST_SIGNAL_RECEIPT.json')):raise RuntimeError('No documented host takeover')
            result=run.run_arm(code,freeze,pages,images,identity,arm,True)
            # Original adopt returns 2 after a clean complete container. Only seal in that case; never retry pending work automatically.
            root=BASE/'full-v2'/arm
            if result==2 and not state.pending_pages(root,pages,identity,arm):
                last=sorted((root/'sessions').iterdir())[-1];run.validate_session_exit(last)
                if state.read(last/'SESSION_RESULT.json').get('complete'):result=run.run_arm(code,freeze,pages,images,identity,arm,True)
            state.exclusive_json(attempt/'HOST_RESULT.json',{'exit_code':result,'time':time.time(),'original_failures_preserved':True})
            return result
        except BaseException:
            state.exclusive_json(attempt/'HOST_ERROR.json',{'traceback':traceback.format_exc(),'time':time.time(),
                'recovery':'Use this same repaired entry after diagnosis; never resume the broken legacy collector'})
            raise


if __name__=='__main__':raise SystemExit(main())
