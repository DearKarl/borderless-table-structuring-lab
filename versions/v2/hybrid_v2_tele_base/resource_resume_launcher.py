"""Read-only diagnosis and one-shot same-resource resume after two pinned conflicts."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
from resource_resume_policy import validate_foreign_exit,completed_candidate

HERE=Path(__file__).resolve().parent
BASE=Path('/srv/hybrid-research/inference/hybrid-v2-tele-base')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())


def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def verify_locked_directory(code,lock_name,expected):
    if sha(code/lock_name)!=expected:raise ValueError('Package lock identity differs')
    for name,value in read(code/lock_name).items():
        path=(code/name).resolve()
        if not path.is_relative_to(code) or sha(path)!=value:raise ValueError('Locked package bytes changed')


def audit(run,state,pages,arm,binding):
    root=BASE/'full-v2'/arm;identity=binding['arms'][arm]['freeze_sha256'];pin=binding['events'][arm]
    pending=state.pending_pages(root,pages,identity,arm);pending_keys={p['key'] for p in pending};by_key={p['key']:p for p in pages}
    if read(root/'RUN_BINDING.json')!={'freeze_sha256':identity,'arm':arm,'input_manifest_sha256':binding['input_manifest_sha256']}:
        raise ValueError('Original run binding changed')
    host=read(BASE/pin['host_receipt'])
    if host['pid']!=pin['host_pid'] or host['freeze_sha256']!=identity:raise ValueError('Original host receipt differs')
    if Path('/proc/'+str(pin['host_pid'])).exists():raise ValueError('Original host PID still exists; no second writer')
    conflict=root/'sessions'/pin['session'];exit_path=conflict/'EXIT.json';exit_record=read(exit_path)
    validate_foreign_exit(exit_record,pin,arm,identity,binding['arms'][arm]['gpu_uuid'])
    actual=json.loads(run.query(['docker','inspect',exit_record['cid']]))[0]
    if actual['Id']!=exit_record['cid'] or actual['State']['Running'] or actual['State']['OOMKilled'] or actual['State']['ExitCode']!=137:
        raise ValueError('Original container state changed')
    complete={};interrupted=[];exits=[]
    for session in sorted((root/'sessions').iterdir()):
        assigned=read(session/'ASSIGNED.json')
        if assigned['arm']!=arm or assigned['freeze_sha256']!=identity:raise ValueError('Historical assignment binding differs')
        if len({p['key'] for p in assigned['pages']})!=len(assigned['pages']):raise ValueError('Duplicate assigned page')
        if (session/'EXIT.json').exists():exits.append({'session':session.name,'sha256':sha(session/'EXIT.json')})
        for item in assigned['pages']:
            page=by_key[item['key']]
            if any(item[k]!=page[k] for k in ('page_id','input_sha256')):raise ValueError('Historical assignment input differs')
            folder=session/'pages'/item['key'];receipt=folder/'PAGE_RESULT.json'
            if receipt.exists():
                result=read(receipt);filename=result.get('prediction_file','prediction.md')
                if filename not in ('prediction.md','external-empty.md'):raise ValueError('Unexpected primary filename')
                raw=folder/filename;actual_hash=sha(raw);size=raw.stat().st_size
                completed_candidate(result,page,arm,identity,actual_hash,size)
                if page['key'] in complete:raise ValueError('Multiple completed native results for one page')
                complete[page['key']]={'key':page['key'],'page_id':page['page_id'],'session':session.name,
                    'receipt_sha256':sha(receipt),'prediction_sha256':actual_hash,'bytes':size,'status':result['status'],
                    'committed':page['key'] not in pending_keys}
                if page['key'] not in pending_keys:
                    terminal=read(root/'terminals'/(page['key']+'.json'))
                    if terminal['session_receipt_sha256']!=sha(receipt) or (root/terminal['session_receipt']).resolve()!=receipt.resolve():
                        raise ValueError('Committed page does not reference original native result')
            elif (folder/'STARTED.json').exists():
                started=read(folder/'STARTED.json')
                if any(started[k]!=page[k] for k in ('page_id','input_sha256','key')):raise ValueError('Interrupted page identity differs')
                interrupted.append({'key':page['key'],'session':session.name,'started_sha256':sha(folder/'STARTED.json'),
                    'still_without_terminal':page['key'] in pending_keys})
    if sum(r['committed'] for r in complete.values())!=1651-len(pending):raise ValueError('Terminal lacks completed native evidence')
    return {'arm':arm,'freeze_sha256':identity,'gpu_uuid':binding['arms'][arm]['gpu_uuid'],
        'original_host_receipt_sha256':sha(BASE/pin['host_receipt']),'original_host_pid':pin['host_pid'],
        'specific_foreign_session':str(conflict),'original_exit_sha256':sha(exit_path),'original_exit':exit_record,
        'actual_stopped_container_state':actual['State'],'committed_pages':1651-len(pending),
        'completed_native_pages':len(complete),'complete_uncommitted_pages':[r for r in complete.values() if not r['committed']],
        'completed_pages':list(complete.values()),'interrupted_pages':interrupted,'session_exits':exits,
        'remaining_after_reconciliation':1651-len(complete),'original_failure_preserved':True,'read_only':True}


def old_tele_errno_gate(state,binding,pages):
    code=BASE/'host-repair-code-v3';verify_locked_directory(code,'CODE_LOCK.json',binding['tele_host_repair_lock_sha256'])
    policy=import_file('approved_tele_host_policy_v3',code/'host_repair_policy.py')
    proof=read(BASE/'host-repair-v3/tele_raw/SESSION_0001_ADMISSION.json');session=BASE/'full-v2/tele_raw/sessions/session-0001'
    if proof['recovery_entry_sha256']!=sha(code/'launcher.py') or proof['policy_sha256']!=sha(code/'host_repair_policy.py'):
        raise ValueError('Original errno recovery policy differs')
    if not proof['original_exit_preserved'] or proof['original_failure_reclassified_as_success'] is not False:
        raise ValueError('Original errno failure must remain failed')
    if sha(session/'EXIT.json')!=proof['original_exit_sha256']:raise ValueError('Original errno exit changed')
    p205=next(p for p in pages if p['key']=='p00205')
    policy.specific_host_fault(read(session/'EXIT.json'),binding['arms']['tele_raw']['freeze_sha256'],p205['page_id'])
    for row in proof['reused_completed_pages']:
        if sha(session/'pages'/row['key']/'PAGE_RESULT.json')!=row['receipt_sha256']:raise ValueError('Old recovered receipt changed')
    return session,proof,policy


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--lock-sha256',required=True)
    parser.add_argument('--arm',choices=['tele_raw','hybrid_v2'],required=True);parser.add_argument('--action',choices=['diagnose','resume'],required=True)
    args=parser.parse_args()
    if HERE!=BASE/'resource-resume-code-v1':raise ValueError('Unexpected recovery code location')
    verify_locked_directory(HERE,'CODE_LOCK.json',args.lock_sha256);binding=read(HERE/'BINDINGS.json')
    pair_file=BASE/'parallel-code-v1/RESOURCE_PAIR.json'
    if sha(pair_file)!=binding['resource_pair_sha256'] or read(pair_file)['arms']!=binding['arms']:raise ValueError('Original resource pair differs')
    code=Path(binding['arms'][args.arm]['code']);identity=binding['arms'][args.arm]['freeze_sha256']
    sys.path.insert(0,str(code))
    from hybrid_v2_tele_base import run_full as run,full_state as state
    freeze,pages,images=run.verify_code(code,identity)
    repair=BASE/'host-repair-code-v3';verify_locked_directory(repair,'CODE_LOCK.json',binding['tele_host_repair_lock_sha256'])
    policy=import_file('approved_short_atomic_writer',repair/'host_repair_policy.py');policy.validate_binding_digests(binding)
    if args.action=='diagnose':
        result=audit(run,state,pages,args.arm,binding)
        result['current_gpu_inventory']=run.query(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'])
        print(json.dumps(result,ensure_ascii=False),flush=True);return 0
    state.exclusive_bytes=policy.exclusive_bytes;run.exclusive_bytes=policy.exclusive_bytes
    records=BASE/'resource-resume-v1'/args.arm;records.mkdir(parents=True,exist_ok=True)
    with (records/'RECOVERY.lease').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        attempts=sorted(records.glob('attempt-*'));attempt=records/('attempt-%04d'%(len(attempts)+1));attempt.mkdir(exist_ok=False)
        state.exclusive_json(attempt/'HOST_START.json',{'pid':os.getpid(),'command':sys.argv,'time':time.time(),
            'arm':args.arm,'freeze_sha256':identity,'code_lock_sha256':args.lock_sha256})
        try:
            root=BASE/'full-v2'/args.arm
            with (root/'WRITER.lease').open('a') as writer:
                fcntl.flock(writer,fcntl.LOCK_EX|fcntl.LOCK_NB)
                report=audit(run,state,pages,args.arm,binding);state.exclusive_json(attempt/'READ_ONLY_AUDIT.json',report)
                if Path('/proc/'+str(binding['events'][args.arm]['foreign_pid'])).exists():raise RuntimeError('Original foreign PID currently exists; no signal or launch')
                idle=run.idle()
                prior_errno=old_tele_errno_gate(state,binding,pages) if args.arm=='tele_raw' else None
                for session in sorted((root/'sessions').iterdir()):
                    run.collect(root,session,read(session/'ASSIGNED.json')['pages'],identity,args.arm)
                remaining=state.pending_pages(root,pages,identity,args.arm)
                if len(remaining)!=report['remaining_after_reconciliation']:raise RuntimeError('Reconciled page count differs')
                admission={'arm':args.arm,'freeze_sha256':identity,'specific_foreign_session':report['specific_foreign_session'],
                    'original_exit_sha256':report['original_exit_sha256'],'original_failed_exit_preserved':True,
                    'original_failure_reclassified_as_success':False,'read_only_audit_sha256':sha(attempt/'READ_ONLY_AUDIT.json'),
                    'completed_pages_reused':report['completed_native_pages'],'remaining_pages':len(remaining),
                    'gpu_idle_before_resume':idle,'entry_sha256':sha(HERE/'launcher.py'),'code_lock_sha256':args.lock_sha256,
                    'signals_sent':False,'time':time.time()}
                state.exclusive_json(attempt/'RESOURCE_INTERRUPTION_ADMISSION.json',admission)
            original_validator=run.validate_session_exit
            conflict=Path(report['specific_foreign_session'])
            def validate_session(target):
                target=Path(target)
                if target.resolve()==conflict.resolve():
                    if sha(target/'EXIT.json')!=report['original_exit_sha256']:raise ValueError('Original foreign exit changed')
                    validate_foreign_exit(read(target/'EXIT.json'),binding['events'][args.arm],args.arm,identity,binding['arms'][args.arm]['gpu_uuid'])
                    return
                if prior_errno is not None and target.resolve()==prior_errno[0].resolve():
                    session,proof,old_policy=prior_errno
                    if sha(session/'EXIT.json')!=proof['original_exit_sha256']:raise ValueError('Original errno exit changed')
                    old_policy.specific_host_fault(read(session/'EXIT.json'),identity,next(p['page_id'] for p in pages if p['key']=='p00205'))
                    return
                return original_validator(target)
            run.validate_session_exit=validate_session
            # Original run_arm repeats fresh idle checks under the UUID lease before creating the next worker.
            result=run.run_arm(code,freeze,pages,images,identity,args.arm,True)
            state.exclusive_json(attempt/'HOST_RESULT.json',{'exit_code':result,'time':time.time(),'original_failures_preserved':True})
            return result
        except BaseException:
            state.exclusive_json(attempt/'HOST_ERROR.json',{'traceback':traceback.format_exc(),'time':time.time(),'no_external_process_signalled':True})
            raise


if __name__=='__main__':raise SystemExit(main())
