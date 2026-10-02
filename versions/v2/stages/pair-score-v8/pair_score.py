"""Independent pair-aware evidence seal; reuse the byte-identical official scoring controller."""
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PARALLEL=HERE.parent/'parallel-code-v1'
sys.path.insert(0,str(PARALLEL/'hybrid-code'))
sys.path.insert(0,str(PARALLEL))
from hybrid_v2_tele_base.full_state import read,sha,exclusive_json,pending_pages,validate_session_exit
from hybrid_v2_tele_base import full_score as official
from parallel_contract import resource_only
from recovery_score_gate import verify_recovery
from resource_score_gate import verify_resource_recovery
from guard_score_gate import verify_guard_recovery
from external_score_gate import verify_external_recovery,load_external_pins
from external2_score_gate import verify_external2_recovery,load_external2_pins
from deadline_score_gate import verify_deadline_recovery
from hybrid_deadline_score_gate import verify_hybrid_deadline_recovery,load_hybrid_deadline_pins

BASE=official.BASE


def completed_binding(complete,arm,pair):
    expected={'freeze_sha256':pair['arms'][arm]['freeze_sha256'],'arm':arm,
              'input_manifest_sha256':pair['input_manifest_sha256'],'pages':1651}
    if any(complete.get(k)!=v for k,v in expected.items()):raise ValueError('Completed arm resource identity differs')


def arm_root(pair,arm):
    root=Path(pair['arms'][arm]['output']).resolve()
    if root!=(BASE/'full-v2'/arm).resolve():raise ValueError('Unexpected arm output directory')
    return root


def validate_dual(lock,pair,pair_sha,pages,recovery):
    if lock['freeze_sha256']!=pair_sha or lock['resource_pair_sha256']!=pair_sha or set(lock['arms'])!={'tele_raw','hybrid_v2'}:
        raise ValueError('Pair lock identity differs')
    if lock.get('host_recovery')!=recovery:raise ValueError('Specific host recovery proof changed')
    for name,value in recovery['evidence'].items():
        if sha(Path(name))!=value:raise ValueError('Host recovery evidence changed')
    for arm,record in lock['arms'].items():
        root=arm_root(pair,arm);identity=pair['arms'][arm]['freeze_sha256']
        if record['freeze_sha256']!=identity:raise ValueError('Cross-arm resource freeze substitution')
        completed_binding(read(root/'COMPLETE.json'),arm,pair)
        if pending_pages(root,pages,identity,arm):raise ValueError('Incomplete arm')
        expected={p['page_id']+'.md' for p in pages}
        if set(record['predictions'])!=expected or {p.name for p in (root/'predictions').iterdir()}!=expected:
            raise ValueError('Primary coverage differs')
        for name,value in record['predictions'].items():
            path=root/'predictions'/name
            if path.is_symlink() or sha(path)!=value:raise ValueError('Locked primary changed')
        for name,value in record['evidence'].items():
            path=(root/name).resolve()
            if not path.is_relative_to(root) or sha(path)!=value:raise ValueError('Locked native evidence changed')


def seal(pair,pair_sha,pages,recovery):
    target=BASE/'full-v2/DUAL_PRIMARY_LOCK.json';handles=[]
    with (BASE/'full-v2/PAIR_SEAL.lease').open('a') as sealing:
        fcntl.flock(sealing,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if target.exists():
            validate_dual(read(target),pair,pair_sha,pages,recovery);return target
        lock={'freeze_sha256':pair_sha,'resource_pair_sha256':pair_sha,'complete':True,'page_count_per_arm':1651,'arms':{},'host_recovery':recovery}
        try:
            for arm in ('tele_raw','hybrid_v2'):
                root=arm_root(pair,arm);identity=pair['arms'][arm]['freeze_sha256']
                handle=(root/'WRITER.lease').open('a');handles.append(handle);fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                completed_binding(read(root/'COMPLETE.json'),arm,pair)
                if pending_pages(root,pages,identity,arm):raise ValueError('Arm still has pending pages')
                for session in sorted((root/'sessions').iterdir()):
                    if arm=='tele_raw' and str(session)==recovery['admitted_failed_session']:
                        if sha(session/'EXIT.json')!=recovery['original_exit_sha256']:raise ValueError('Original failed exit changed')
                    elif str(session) in recovery['resource_recovery']['admitted_sessions']:
                        if sha(session/'EXIT.json')!=recovery['resource_recovery']['admitted_sessions'][str(session)]:raise ValueError('Resource interruption exit changed')
                    elif str(session)==recovery['guard_recovery']['admitted_session']:
                        if sha(session/'EXIT.json')!=recovery['guard_recovery']['original_exit_sha256']:raise ValueError('Unknown-ownership interruption exit changed')
                    elif str(session)==recovery['external_recovery']['admitted_session']:
                        if sha(session/'EXIT.json')!=recovery['external_recovery']['original_exit_sha256']:raise ValueError('External interruption exit changed')
                    elif str(session)==recovery['external2_recovery']['admitted_session']:
                        if sha(session/'EXIT.json')!=recovery['external2_recovery']['original_exit_sha256']:raise ValueError('Second external interruption exit changed')
                    else:validate_session_exit(session)
                if {p.name for p in (root/'predictions').iterdir()}!={p['page_id']+'.md' for p in pages}:
                    raise ValueError('Unexpected primary file set')
                terminals=[read(root/'terminals'/(p['key']+'.json')) for p in pages]
                evidence={}
                for sub in ('sessions','terminals'):
                    for path in sorted((root/sub).rglob('*')):
                        relative=path.relative_to(root)
                        if path.is_file() and not any(x in {'.cache','.nv','hf-cache','.paddlex'} for x in relative.parts):
                            evidence[relative.as_posix()]=sha(path)
                for name in ('COMPLETE.json','RUN_BINDING.json','PARALLEL_RESERVATION.json'):
                    if (root/name).exists():evidence[name]=sha(root/name)
                lock['arms'][arm]={'freeze_sha256':identity,'gpu_uuid':pair['arms'][arm]['gpu_uuid'],
                    'predictions':{p['page_id']+'.md':sha(root/'predictions'/(p['page_id']+'.md')) for p in pages},
                    'counts':dict(Counter(t['status'] for t in terminals)),'evidence':evidence}
            exclusive_json(target,lock)
        finally:
            for handle in handles:handle.close()
        return target


def all_recovery(pair,pages,binding):
    recovery=verify_recovery(BASE,pair,pages,binding)
    resource=verify_resource_recovery(BASE,pair,pages,binding)
    recovery['resource_recovery']=resource
    recovery['evidence'].update(resource['evidence'])
    continuation=load_external_pins(BASE,binding)
    second=load_external2_pins(BASE,binding)
    native=load_hybrid_deadline_pins(BASE,binding)
    guard=verify_guard_recovery(BASE,pair,pages,binding,continuation=continuation,allow_session4=True,second_continuation=second,allow_session5=True,native_deadline=native,allow_session6=True)
    recovery['guard_recovery']=guard
    recovery['evidence'].update(guard['evidence'])
    external=verify_external_recovery(BASE,pair,pages,binding,continuation=second,allow_session5=True,native_deadline=native,allow_session6=True)
    recovery['external_recovery']=external
    recovery['evidence'].update(external['evidence'])
    external2=verify_external2_recovery(BASE,pair,pages,binding,native_deadline=native,allow_session6=True)
    recovery['external2_recovery']=external2
    recovery['evidence'].update(external2['evidence'])
    deadline=verify_deadline_recovery(BASE,pair,pages,binding)
    recovery['native_deadline_continuation']=deadline
    recovery['evidence'].update(deadline['evidence'])
    hybrid_deadline=verify_hybrid_deadline_recovery(BASE,pair,pages,binding)
    recovery['hybrid_native_deadline_continuation']=hybrid_deadline
    recovery['evidence'].update(hybrid_deadline['evidence'])
    return recovery


def preflight(expected):
    if HERE!=BASE/'pair-score-v8' or sha(HERE/'PAIR_SCORE_LOCK.json')!=expected:raise ValueError('Score package identity')
    for name,value in read(HERE/'PAIR_SCORE_LOCK.json').items():
        path=(HERE/name).resolve()
        if not path.is_relative_to(HERE) or sha(path)!=value:raise ValueError('Score package changed')
    binding=read(HERE/'SCORE_BINDING.json')
    if sha(PARALLEL/'PARALLEL_PACKAGE_LOCK.json')!=binding['parallel_package_lock_sha256']:
        raise ValueError('Parallel launch package changed')
    for name,value in read(PARALLEL/'PARALLEL_PACKAGE_LOCK.json').items():
        path=(PARALLEL/name).resolve()
        if not path.is_relative_to(PARALLEL) or sha(path)!=value:raise ValueError('Resource package changed')
    pair_file=PARALLEL/'RESOURCE_PAIR.json';pair=read(pair_file);pair_sha=sha(pair_file)
    if pair_sha!=binding['resource_pair_sha256']:raise ValueError('Resource pair changed')
    freezes={}
    for arm in ('tele_raw','hybrid_v2'):
        code=Path(pair['arms'][arm]['code']);freeze_file=code/'SYSTEM_FREEZE.json'
        if sha(freeze_file)!=pair['arms'][arm]['freeze_sha256']:raise ValueError('Arm freeze changed')
        freeze=read(freeze_file);freezes[arm]=freeze
        for name,value in freeze['code_files'].items():
            path=(code/name).resolve()
            if not path.is_relative_to(code) or sha(path)!=value:raise ValueError('Arm frozen code changed')
        if sha(code/'INPUT_MANIFEST.json')!=pair['input_manifest_sha256']:raise ValueError('Arm original input identity differs')
    resource_only(freezes['tele_raw'],freezes['hybrid_v2'],pair['arms']['tele_raw']['gpu_uuid'],pair['arms']['hybrid_v2']['gpu_uuid'])
    if freezes['tele_raw']['official_evaluator']!=pair['official_evaluator'] or freezes['hybrid_v2']['official_evaluator']!=pair['official_evaluator']:
        raise ValueError('Official evaluator differs between arms')
    code=Path(pair['arms']['hybrid_v2']['code'])
    rows=read(code/'INPUT_MANIFEST.json')['pages']
    if len(rows)!=1651 or len({p['page_id'] for p in rows})!=1651:raise ValueError('Original 1651 required')
    pages=[{**p,'key':'p%05d'%i} for i,p in enumerate(rows)]
    recovery=all_recovery(pair,pages,binding)
    return pair,pair_sha,pages,code,freezes['hybrid_v2'],recovery,binding


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--lock-sha256',required=True)
    parser.add_argument('--action',choices=['verify-recovery','seal-both','score'],required=True);parser.add_argument('--arm',choices=['tele_raw','hybrid_v2'])
    args=parser.parse_args();pair,pair_sha,pages,code,freeze,recovery,binding=preflight(args.lock_sha256)
    if args.action=='verify-recovery':
        print(json.dumps({'read_only':True,'recovery_verified':True,'proof':recovery,'scoring_started':False,'seal_created':False}),flush=True)
        return
    if args.action=='seal-both':
        path=seal(pair,pair_sha,pages,recovery);print(json.dumps({'path':str(path),'sha256':sha(path)}),flush=True)
    else:
        if args.arm is None:raise ValueError('Score arm is required')
        # The only adapted dependency is the cross-arm resource identity validator.
        # Official source/config/runner, Docker lifecycle, metrics and aggregation retain original bytes.
        official.validate_dual=lambda lock,identity,rows: validate_dual(lock,pair,identity,rows,all_recovery(pair,rows,binding))
        official.score(code,freeze,pair_sha,pages,args.arm)


if __name__=='__main__':main()
