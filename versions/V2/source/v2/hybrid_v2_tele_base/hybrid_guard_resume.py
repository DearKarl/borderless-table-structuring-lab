"""Only Hybrid session-0002's pinned unknown-ownership guard interruption."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
import time
import traceback
import fcntl

HERE=Path(__file__).resolve().parent
BASE=Path('/srv/hybrid-research/inference/hybrid-v2-tele-base')


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--lock-sha256',required=True);args=parser.parse_args()
    sys.path.insert(0,str(BASE/'parallel-code-v1/hybrid-code'))
    from hybrid_v2_tele_base import full_state as state,run_full as run
    from gpu_ownership_guard import scan
    if HERE!=BASE/'hybrid-guard-resume-v1' or state.sha(HERE/'CODE_LOCK.json')!=args.lock_sha256:raise ValueError('Guard recovery package differs')
    for name,value in state.read(HERE/'CODE_LOCK.json').items():
        path=(HERE/name).resolve()
        if not path.is_relative_to(HERE) or state.sha(path)!=value:raise ValueError('Guard recovery source differs')
    binding=state.read(HERE/'BINDINGS.json');old_score=BASE/'pair-score-v3'
    sys.path.insert(0,str(old_score));score=module('previous_pair_score',old_score/'pair_score.py')
    pair,pair_sha,pages,code,freeze,prior,score_binding=score.preflight(binding['prior_score_lock_sha256'])
    # Run the original full verify_code too: model, source, driver and image inputs are unchanged.
    freeze,pages,images=run.verify_code(code,binding['freeze_sha256'])
    previous_code=BASE/'resource-resume-code-v1';sys.path.insert(0,str(previous_code))
    previous=module('previous_resource_resume',previous_code/'launcher.py')
    previous.verify_locked_directory(previous_code,'CODE_LOCK.json',binding['prior_resume_lock_sha256'])
    previous_binding=state.read(previous_code/'BINDINGS.json')
    previous_binding['events']['hybrid_v2']=binding['event']
    repair=BASE/'host-repair-code-v3';previous.verify_locked_directory(repair,'CODE_LOCK.json',previous_binding['tele_host_repair_lock_sha256'])
    policy=module('validated_short_atomic_guard',repair/'host_repair_policy.py')
    state.exclusive_bytes=policy.exclusive_bytes;run.exclusive_bytes=policy.exclusive_bytes
    for relative,digest in binding['original_evidence'].items():
        if state.sha(BASE/relative)!=digest:raise ValueError('Original interruption evidence changed: '+relative)
    root=BASE/'full-v2/hybrid_v2';target=root/'sessions/session-0002'
    if state.read(target/'EXIT.json')['cid']!=binding['cid']:raise ValueError('Specific full container identity differs')
    directory=BASE/'hybrid-guard-recovery-v1';directory.mkdir(exist_ok=True)
    with (directory/'RECOVERY.lease').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        attempt=directory/'attempt-0001';attempt.mkdir(exist_ok=False)
        state.exclusive_json(attempt/'HOST_START.json',{'pid':os.getpid(),'command':sys.argv,'time':time.time(),
            'arm':'hybrid_v2','freeze_sha256':binding['freeze_sha256'],'code_lock_sha256':args.lock_sha256})
        try:
            with (root/'WRITER.lease').open('a') as writer:
                fcntl.flock(writer,fcntl.LOCK_EX|fcntl.LOCK_NB)
                if {p.name for p in (root/'sessions').iterdir()}!={'session-0001','session-0002'}:raise ValueError('Unexpected existing session; no duplicate launch')
                report=previous.audit(run,state,pages,'hybrid_v2',previous_binding)
                report['ownership_classification']='unknown';report['historical_guard_label_is_not_ownership_proof']=True
                if report['completed_native_pages']!=646 or report['remaining_after_reconciliation']!=1005:raise ValueError('Expected exact 646 reused / 1005 missing pages')
                state.exclusive_json(attempt/'READ_ONLY_AUDIT.json',report)
                if Path('/proc/2294174').exists():raise ValueError('Historical suspect PID currently exists; no signal or launch')
                idle=run.idle()
                for session in sorted((root/'sessions').iterdir()):run.collect(root,session,state.read(session/'ASSIGNED.json')['pages'],binding['freeze_sha256'],'hybrid_v2')
                if len(state.pending_pages(root,pages,binding['freeze_sha256'],'hybrid_v2'))!=1005:raise ValueError('Reconciled coverage differs')
                admission={'arm':'hybrid_v2','freeze_sha256':binding['freeze_sha256'],'specific_session':str(target),
                    'original_exit_sha256':state.sha(target/'EXIT.json'),'original_cid':binding['cid'],
                    'observed_guard_reason':'foreign_gpu_process','ownership_classification':'unknown',
                    'original_failed_exit_preserved':True,'original_failure_reclassified_as_success':False,
                    'completed_pages_reused':646,'remaining_pages':1005,'read_only_audit_sha256':state.sha(attempt/'READ_ONLY_AUDIT.json'),
                    'gpu_idle_before_resume':idle,'code_lock_sha256':args.lock_sha256,'host_pid':os.getpid(),
                    'original_monitor_sha256':binding['original_monitor_sha256'],'guarded_monitor_sha256':state.sha(HERE/'guarded_monitor.py'),
                    'gpu_guard_sha256':state.sha(HERE/'gpu_ownership_guard.py'),'signals_sent':False,'time':time.time()}
                state.exclusive_json(attempt/'GUARD_INTERRUPTION_ADMISSION.json',admission)
            # Only the host monitor's ownership-check block differs; original frozen workers stay untouched.
            namespace=dict(run.__dict__,guard_scan=scan)
            exec(compile((HERE/'guarded_monitor.py').read_bytes(),str(HERE/'guarded_monitor.py'),'exec'),namespace)
            run.monitor=namespace['monitor']
            original_validator=run.validate_session_exit
            def validate(session):
                if Path(session).resolve()==target.resolve():
                    if state.sha(target/'EXIT.json')!=admission['original_exit_sha256']:raise ValueError('Pinned unknown guard interruption changed')
                    return
                return original_validator(session)
            run.validate_session_exit=validate
            result=run.run_arm(code,freeze,pages,images,binding['freeze_sha256'],'hybrid_v2',True)
            state.exclusive_json(attempt/'HOST_RESULT.json',{'exit_code':result,'time':time.time(),'original_failures_preserved':True})
            return result
        except BaseException:
            state.exclusive_json(attempt/'HOST_ERROR.json',{'traceback':traceback.format_exc(),'time':time.time()});raise

if __name__=='__main__':raise SystemExit(main())
