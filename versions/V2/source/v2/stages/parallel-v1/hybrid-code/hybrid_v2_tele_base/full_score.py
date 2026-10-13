"""Seal both fresh arms before isolated official CPU scoring; no prediction mutation."""
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hybrid_v2_tele_base.full_state import read,sha,exclusive_json,pending_pages,validate_session_exit
from hybrid_v2_tele_base.lock_checks import digest

ROOT=Path('/srv/hybrid-research')
BASE=ROOT/'inference/hybrid-v2-tele-base'


def validate_dual(lock,freeze_sha,pages):
    if lock['freeze_sha256']!=freeze_sha or set(lock['arms'])!={'tele_raw','hybrid_v2'}:raise ValueError('Dual lock identity')
    for arm,record in lock['arms'].items():
        root=BASE/'full-v2'/arm
        if pending_pages(root,pages,freeze_sha,arm):raise ValueError('Incomplete arm')
        if set(record['predictions'])!={p['page_id']+'.md' for p in pages}:raise ValueError('Primary coverage differs')
        for name,expected in record['predictions'].items():
            if sha(root/'predictions'/name)!=expected:raise ValueError('Locked primary differs')
        for name,expected in record['evidence'].items():
            path=(root/name).resolve()
            if not path.is_relative_to(root.resolve()) or sha(path)!=expected:raise ValueError('Locked evidence differs')


def seal(freeze_sha,pages):
    target=BASE/'full-v2/DUAL_PRIMARY_LOCK.json'
    if target.exists():
        lock=read(target);validate_dual(lock,freeze_sha,pages);return target
    lock={'freeze_sha256':freeze_sha,'complete':True,'page_count_per_arm':1651,'arms':{}}
    handles=[]
    try:
        for arm in ('tele_raw','hybrid_v2'):
            root=BASE/'full-v2'/arm
            handle=(root/'WRITER.lease').open('a');handles.append(handle);fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            complete=read(root/'COMPLETE.json')
            if complete['freeze_sha256']!=freeze_sha or complete['pages']!=1651:raise ValueError('Completion identity')
            if pending_pages(root,pages,freeze_sha,arm):raise ValueError('Arm still has pending pages')
            for session in sorted((root/'sessions').iterdir()):validate_session_exit(session)
            if {p.name for p in (root/'predictions').iterdir()}!={p['page_id']+'.md' for p in pages}:raise ValueError('Extra prediction files')
            terminals=[read(root/'terminals'/(p['key']+'.json')) for p in pages]
            evidence={}
            for sub in ('sessions','terminals'):
                for path in sorted((root/sub).rglob('*')):
                    # Runtime caches are not prediction evidence. Native page outputs and all worker logs are included.
                    relative=path.relative_to(root)
                    if path.is_file() and not any(x in {'.cache','.nv','hf-cache','.paddlex'} for x in relative.parts):
                        evidence[relative.as_posix()]=sha(path)
            for name in ('COMPLETE.json','RUN_BINDING.json'):evidence[name]=sha(root/name)
            lock['arms'][arm]={'predictions':{p['page_id']+'.md':sha(root/'predictions'/(p['page_id']+'.md')) for p in pages},
                'counts':dict(Counter(t['status'] for t in terminals)),'evidence':evidence}
        exclusive_json(target,lock)
    finally:
        for handle in handles:handle.close()
    return target


def score(code,freeze,freeze_sha,pages,arm):
    target=BASE/'full-v2/DUAL_PRIMARY_LOCK.json';lock=read(target);validate_dual(lock,freeze_sha,pages)
    pins=freeze['official_evaluator'];official=ROOT/'evaluation/official-full-20260921-v1'
    source=read(code/'OFFICIAL_SOURCE_READY.json')
    if source['source_revision']!=pins['source_revision']:raise ValueError('Evaluator source revision')
    for row in source['files']:
        path=(official/'odb-code'/row['path']).resolve()
        if not path.is_relative_to(official/'odb-code') or sha(path)!=row['sha256']:raise ValueError('Evaluator source changed')
    config=official/'configs/end2end-full.yaml';entry=code/'official_entry.py';gt=official/'OmniDocBench.json'
    if sha(config)!=pins['config_sha256'] or sha(entry)!=pins['entry_sha256'] or sha(gt)!=pins['gt_sha256']:
        raise ValueError('Evaluator config, entry, or GT changed')
    slots=ROOT/'receipts/OFFICIAL_FULL_EVAL_v1';slot=None
    for index in range(2):
        handle=(slots/('CPU_SLOT_%d.lease'%index)).open('a')
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);slot=handle;break
        except BlockingIOError:handle.close()
    if slot is None:raise RuntimeError('No official evaluator CPU slot available')
    out=BASE/'full-v2'/arm/'official-evaluation-v1';out.mkdir(exist_ok=False)
    command=['docker','create','--name','hybrid-v2-full2-official-'+arm,'--network','none','--read-only',
        '--cap-drop','ALL','--security-opt','no-new-privileges','--cpus','24','--memory','128g',
        '--pids-limit','4096','--shm-size','8g','--tmpfs','/tmp:rw,size=8g','--workdir','/work','--entrypoint',pins['python']]
    for key,value in {'PYTHONPATH':'/source','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
            'MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','HOME':'/work'}.items():
        command+=['--env',key+'='+value]
    for path,destination,ro in [(official/'odb-code','/source',True),(official/'configs','/config',True),
            (gt,'/gt/OmniDocBench.json',True),(BASE/'full-v2'/arm/'predictions','/pred',True),
            (entry,'/entry.py',True),(out,'/work',False)]:
        command+=['--mount',f'type=bind,src={path},dst={destination}'+(',readonly' if ro else '')]
    command += [pins['image'],'/entry.py']
    cid=subprocess.check_output(command,text=True).strip()
    exclusive_json(out/'START.json',{'cid':cid,'command':command,'freeze_sha256':freeze_sha,'dual_lock_sha256':sha(target)})
    failure=None;proc=None
    try:
        with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:
            proc=subprocess.run(['docker','start','--attach',cid],stdout=stdout,stderr=stderr,timeout=72*3600)
    except BaseException as exc:failure=repr(exc);raise
    finally:
        state=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0]
        if state['State']['Running']:subprocess.run(['docker','kill',cid],check=True,capture_output=True)
        subprocess.run(['docker','wait',cid],check=True,capture_output=True)
        state=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0]
        exclusive_json(out/'EXIT.json',{'container':state,'controller_failure':failure});slot.close()
    if proc.returncode or state['State']['ExitCode'] or state['State']['OOMKilled']:raise RuntimeError('Official scoring failed')
    score_record=read(out/'OFFICIAL_SCORE.json')
    validate_dual(lock,freeze_sha,pages)
    print(json.dumps({'arm':arm,'official_score':score_record,'output':str(out)}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--code',type=Path,required=True)
    parser.add_argument('--freeze-sha256',required=True);parser.add_argument('--action',choices=['seal-both','score'],required=True)
    parser.add_argument('--arm',choices=['tele_raw','hybrid_v2']);args=parser.parse_args()
    code=args.code.resolve()
    if code!=BASE/'full-code-v2' or sha(code/'SYSTEM_FREEZE.json')!=digest(args.freeze_sha256):raise ValueError('Frozen score code identity')
    freeze=read(code/'SYSTEM_FREEZE.json')
    for name,expected in freeze['code_files'].items():
        path=(code/name).resolve()
        if not path.is_relative_to(code) or sha(path)!=expected:raise ValueError('Score code changed')
    pages=[{**p,'key':'p%05d'%i} for i,p in enumerate(read(code/'INPUT_MANIFEST.json')['pages'])]
    if len(pages)!=1651:raise ValueError('Original 1651 required')
    if args.action=='seal-both':
        target=seal(args.freeze_sha256,pages);print(json.dumps({'path':str(target),'sha256':sha(target)}),flush=True)
    else:
        if args.arm is None:raise ValueError('Score arm is required')
        score(code,freeze,args.freeze_sha256,pages,args.arm)


if __name__=='__main__':main()
