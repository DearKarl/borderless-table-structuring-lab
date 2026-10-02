"""One-shot native Hybrid PageDeadline continuation; keep original validator and worker."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
HERE=Path(__file__).resolve().parent
BASE=Path('/srv/hybrid-research/inference/hybrid-v2-tele-base')


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--lock-sha256',required=True);args=parser.parse_args()
    sys.path.insert(0,str(BASE/'parallel-code-v1/hybrid-code'))
    from hybrid_v2_tele_base import full_state as state,run_full as run
    from hybrid_deadline_evidence import verify_original,audit_completed
    if HERE!=BASE/'hybrid-deadline-resume-v1' or state.sha(HERE/'CODE_LOCK.json')!=args.lock_sha256:raise ValueError('Deadline continuation package differs')
    for name,value in state.read(HERE/'CODE_LOCK.json').items():
        path=(HERE/name).resolve()
        if not path.is_relative_to(HERE) or state.sha(path)!=value:raise ValueError('Deadline continuation source differs')
    pins=state.read(HERE/'BINDINGS.json');prior=BASE/'pair-score-v7';sys.path.insert(0,str(prior))
    score=module('prior_deadline_pair_score',prior/'pair_score.py')
    from hybrid_deadline_history import bind_prior_history
    bind_prior_history(score,HERE,BASE,pins)
    pair,pair_sha,_,_,_,_,_=score.preflight(pins['prior_score_lock_sha256'])
    code=Path(pair['arms']['hybrid_v2']['code']);freeze,pages,images=run.verify_code(code,pins['freeze_sha256'])
    if freeze['policy']['page_seconds']!=900 or freeze['policy']['controlled_page_failure']!='empty_primary_and_pause':raise ValueError('Native deadline policy changed')
    verify_original(BASE,pins)
    repair=BASE/'host-repair-code-v3'
    if state.sha(repair/'CODE_LOCK.json')!=pins['short_writer_lock_sha256']:raise ValueError('Approved short writer lock differs')
    for name,value in state.read(repair/'CODE_LOCK.json').items():
        path=(repair/name).resolve()
        if not path.is_relative_to(repair) or state.sha(path)!=value:raise ValueError('Approved short writer source differs')
    writer_policy=module('deadline_short_writer',repair/'host_repair_policy.py')
    state.exclusive_bytes=writer_policy.exclusive_bytes;run.exclusive_bytes=writer_policy.exclusive_bytes
    root=BASE/'full-v2/hybrid_v2';session=root/'sessions/session-0005';directory=BASE/'hybrid-deadline-recovery-v1';directory.mkdir(exist_ok=True)
    with (directory/'RECOVERY.lease').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        attempt=directory/'attempt-0001';attempt.mkdir(exist_ok=False)
        state.exclusive_json(attempt/'HOST_START.json',{'pid':os.getpid(),'command':sys.argv,'time':time.time(),'arm':'hybrid_v2',
            'freeze_sha256':pins['freeze_sha256'],'code_lock_sha256':args.lock_sha256})
        try:
            with (root/'WRITER.lease').open('a') as writer:
                fcntl.flock(writer,fcntl.LOCK_EX|fcntl.LOCK_NB)
                if Path('/proc/2554534').exists():raise ValueError('Original Hybrid host still exists')
                if {p.name for p in (root/'sessions').iterdir()}!={'session-0001','session-0002','session-0003','session-0004','session-0005'}:raise ValueError('Unexpected prior session; no duplicate launch')
                actual=json.loads(run.query(['docker','inspect',pins['cid']]))[0]
                state_record=actual['State'];labels=actual['Config']['Labels']
                if actual['Id']!=pins['cid'] or state_record['Running'] or state_record['OOMKilled'] or state_record['ExitCode']!=2 or state_record['Status']!='exited':raise ValueError('Original paused container state differs')
                if labels.get('hybrid.arm')!='hybrid_v2' or labels.get('hybrid.freeze')!=pins['freeze_sha256']:raise ValueError('Paused container binding differs')
                if state.read(root/'RUN_BINDING.json')!={'arm':'hybrid_v2','freeze_sha256':pins['freeze_sha256'],'input_manifest_sha256':freeze['input_manifest_sha256']}:raise ValueError('Original run binding changed')
                audit=audit_completed(root,pages,pins['freeze_sha256']);state.exclusive_json(attempt/'READ_ONLY_AUDIT.json',audit)
                idle=run.idle()
                state.exclusive_json(attempt/'NATIVE_DEADLINE_CONTINUATION.json',{'arm':'hybrid_v2','freeze_sha256':pins['freeze_sha256'],
                    'code_lock_sha256':args.lock_sha256,'original_exit_sha256':state.sha(session/'EXIT.json'),
                    'read_only_audit_sha256':state.sha(attempt/'READ_ONLY_AUDIT.json'),'completed_pages_reused':1368,'remaining_pages':283,
                    'timeout_page_retained_failed':'p01367','timeout_seconds':900,'original_validator_unchanged':True,
                    'original_failed_page_preserved':True,'original_failure_reclassified_as_success':False,
                    'gpu_idle_before_resume':idle,'host_pid':os.getpid(),'signals_sent':False,'time':time.time()})
            # Keep the original validator, and reuse the already verified Hybrid ownership monitor unchanged.
            from gpu_ownership_guard import scan
            namespace=dict(run.__dict__,guard_scan=scan)
            exec(compile((HERE/'guarded_monitor.py').read_bytes(),str(HERE/'guarded_monitor.py'),'exec'),namespace)
            run.monitor=namespace['monitor']
            result=run.run_arm(code,freeze,pages,images,pins['freeze_sha256'],'hybrid_v2',True)
            state.exclusive_json(attempt/'HOST_RESULT.json',{'exit_code':result,'time':time.time(),'original_failures_preserved':True})
            return result
        except BaseException:
            state.exclusive_json(attempt/'HOST_ERROR.json',{'traceback':traceback.format_exc(),'time':time.time()});raise

if __name__=='__main__':raise SystemExit(main())
