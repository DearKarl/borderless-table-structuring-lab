"""Owned full-run supervisor. Start once; resume validates and adopts prior state."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hybrid_v2_tele_base.full_state import read, sha, exclusive_json, exclusive_bytes, progress, pending_pages, commit_page, phase_limit, validate_session_exit, validate_input_paths
from hybrid_v2_tele_base.lock_checks import digest
from hybrid_v2_tele_base.run_smoke import ROOT, IMAGE, UUID, idle, query

BASE=ROOT/'inference/hybrid-v2-tele-base'


def verify_code(code,expected):
    if sha(code/'SYSTEM_FREEZE.json')!=digest(expected):raise RuntimeError('Freeze identity differs')
    freeze=read(code/'SYSTEM_FREEZE.json')
    for name,expected in freeze['code_files'].items():
        path=(code/name).resolve()
        if not path.is_relative_to(code) or sha(path)!=digest(expected):raise RuntimeError('Frozen code differs')
    manifest=freeze['runtime_manifest']
    if manifest['gpu_uuid']!=UUID or not manifest['capacity']['passed'] or manifest['capacity']['conservative_upper_bound']>4000000000:
        raise RuntimeError('Frozen GPU or capacity gate failed')
    if not read(code/'SMOKE_INDEPENDENT_VERIFICATION_v4.json')['passed']:raise RuntimeError('Smoke gate failed')
    if read(code/'RUNTIME_MANIFEST.json')!=manifest:raise RuntimeError('Worker runtime manifest differs')
    for row in manifest['identity_files']:
        path=Path(row['path']).resolve()
        if not path.is_relative_to(ROOT/'inference') or sha(path)!=digest(row['sha256']):raise RuntimeError('Asset identity differs')
    driver=manifest['driver_lock'];receipt=ROOT/'receipts/GPU_BIND_v1/attempt-v2/DRIVER_BUNDLE.json'
    if sha(receipt)!=digest(driver['receipt_sha256']) or read(receipt)['files']!=driver['files']:raise RuntimeError('Driver receipt differs')
    bundle=ROOT/'inference/driver-bundle-v1/attempt-v2'
    for row in driver['files']:
        if row['present']:
            if not Path(row['target']).resolve().is_relative_to(bundle):raise RuntimeError('Driver target boundary')
            if sha(row['target'])!=digest(row['sha256']) or sha(row['source'])!=row['sha256']:raise RuntimeError('Driver changed')
            if (bundle/row['soname']).resolve()!=Path(row['target']).resolve():raise RuntimeError('Driver symlink changed')
    original=code/'INPUT_MANIFEST.json'
    if sha(original)!=digest(freeze['input_manifest_sha256']):raise RuntimeError('Original manifest changed')
    rows=read(original)['pages']
    if len(rows)!=1651 or len({p['page_id'] for p in rows})!=1651:raise RuntimeError('Expected original 1651 unique pages')
    image_root=ROOT/'inference/official-full-20260921-v1/images'
    validate_input_paths(rows,image_root)
    pages=[{**p,'key':'p%05d'%i,'image_path':'/images/'+Path(p['image_path']).name} for i,p in enumerate(rows)]
    return freeze,pages,image_root


def collect(root,session,pages,freeze_sha,arm):
    count=0
    for page in pages:
        if (session/'pages'/page['key']/'PAGE_RESULT.json').exists():
            count+=commit_page(root,session,page,freeze_sha,arm)
    return count


def monitor(root,session,pages,freeze_sha,arm,cid,adopted):
    started=time.monotonic();phase=None;phase_started=started;reason=None;foreign=[];process=None;current={'phase':'load'};supervisor_error=None
    if not adopted:
        stdout=(session/'stdout.log').open('ab');stderr=(session/'stderr.log').open('ab')
        process=subprocess.Popen(['docker','start','--attach',cid],stdout=stdout,stderr=stderr)
    try:
        while True:
            state=json.loads(query(['docker','inspect',cid]))[0]
            collect(root,session,pages,freeze_sha,arm)
            if state['State']['Status'] not in ('created','running'):break
            try:current=read(session/'PROGRESS.json')
            except FileNotFoundError:current={'phase':'load'}
            identity=(current['phase'],current.get('key'))
            if identity!=phase:
                phase=identity;phase_started=time.monotonic()-max(0,time.time()-current.get('time',time.time()))
                with (root/'PROGRESS.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps({'session':session.name,'time':time.time(),**current},ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
                progress(root/'LATEST.json',{'session':session.name,'time':time.time(),**current})
            if state['State']['Running']:
                owned={x.strip() for x in query(['docker','top',cid,'-eo','pid']).splitlines()[1:]}
                active=query(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'])
                foreign=[x for x in active.splitlines() if x.split(',')[0].strip()==UUID and x.split(',')[1].strip() not in owned]
                if foreign:reason='foreign_gpu_process';break
            limit=phase_limit(current['phase'])
            if time.monotonic()-phase_started>limit:reason='page_timeout' if current['phase']=='page' else 'load_or_shutdown_timeout';break
            if time.monotonic()-started>72*3600:reason='supervisor_wall_limit';break
            time.sleep(5)
    except BaseException:
        reason='supervisor_error';supervisor_error=traceback.format_exc()
        raise
    finally:
        state=json.loads(query(['docker','inspect',cid]))[0]
        if state['State']['Running']:subprocess.run(['docker','kill',cid],check=True,capture_output=True)
        if state['State']['Status']!='created':subprocess.run(['docker','wait',cid],check=True,capture_output=True)
        state=json.loads(query(['docker','inspect',cid]))[0]
        if process is not None:
            process.wait(timeout=30);stdout.close();stderr.close()
        exclusive_json(session/'EXIT.json',{'cid':cid,'container':state,'reason':reason,'foreign_processes':foreign,
                                          'last_progress':current,'adopted_existing_container':adopted,'time':time.time(),
                                          'supervisor_error':supervisor_error})
    if reason=='page_timeout' and not state['State']['OOMKilled']:
        page=next(p for p in pages if p['key']==current['key'])
        folder=session/'pages'/page['key'];started_record=read(folder/'STARTED.json')
        if any(started_record[k]!=page[k] for k in ('key','page_id','input_sha256')):raise RuntimeError('Timeout page identity mismatch')
        if not (folder/'PAGE_RESULT.json').exists():
            exclusive_bytes(folder/'external-empty.md',b'')
            exclusive_json(folder/'PAGE_RESULT.json',{
                **{k:page[k] for k in ('key','page_id','input_sha256')},'arm':arm,'freeze_sha256':freeze_sha,
                'status':'failed','prediction_file':'external-empty.md','prediction_sha256':sha(folder/'external-empty.md'),
                'bytes':0,'error':{'type':'SupervisorPageDeadline','message':'Owned container stopped after page deadline'},
                'external_exit_sha256':sha(session/'EXIT.json'),'native_outputs_preserved':True})
    collect(root,session,pages,freeze_sha,arm)
    return state,reason


def run_arm(code,freeze,pages,image_root,freeze_sha,arm,resume):
    root=BASE/'full-v2'/arm
    if root.exists() and not resume:raise RuntimeError('Existing arm requires explicit --resume, never restart blindly')
    root.mkdir(parents=True,exist_ok=True)
    with (root/'WRITER.lease').open('a') as writer:
        fcntl.flock(writer,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in ('sessions','terminals','predictions'):(root/name).mkdir(exist_ok=True)
        binding={'freeze_sha256':freeze_sha,'arm':arm,'input_manifest_sha256':freeze['input_manifest_sha256']}
        if (root/'RUN_BINDING.json').exists():
            if read(root/'RUN_BINDING.json')!=binding:raise RuntimeError('Resume recipe differs')
        else:exclusive_json(root/'RUN_BINDING.json',binding)
        sessions=sorted((root/'sessions').iterdir());orphan=None
        for old in sessions:
            assigned=read(old/'ASSIGNED.json')
            collect(root,old,assigned['pages'],freeze_sha,arm)
            if (old/'START.json').exists() and not (old/'EXIT.json').exists():
                if orphan is not None:raise RuntimeError('More than one unresolved session')
                orphan=old
        pending=pending_pages(root,pages,freeze_sha,arm)
        if not pending and orphan is None:
            if not sessions or not (sessions[-1]/'EXIT.json').exists():raise RuntimeError('Last session exit is not verified')
            validate_session_exit(sessions[-1])
            if not (root/'COMPLETE.json').exists():exclusive_json(root/'COMPLETE.json',{**binding,'pages':1651})
            return 0
        lease_path=ROOT/'receipts/REMOTE_NATIVE_SMOKE_v1'/(UUID+'.lease')
        with lease_path.open('a') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if orphan is not None:
                start=read(orphan/'START.json');cid=start['cid'];state=json.loads(query(['docker','inspect',cid]))[0]
                if state['Config']['Labels'].get('hybrid.freeze')!=freeze_sha or state['Config']['Labels'].get('hybrid.arm')!=arm:
                    raise RuntimeError('Orphan container ownership differs')
                created=state['State']['Status']=='created'
                if created:idle()
                state,reason=monitor(root,orphan,read(orphan/'ASSIGNED.json')['pages'],freeze_sha,arm,cid,not created)
                # A second explicit resume seals or continues the fully reconciled session.
                return 2
            if sessions:
                last=sessions[-1]
                if not (last/'EXIT.json').exists():raise RuntimeError('Unresolved launch state')
                validate_session_exit(last)
            before=idle()
            entry=next(g for g in ET.fromstring(query(['nvidia-smi','-q','-x'])).findall('gpu') if g.findtext('uuid')==UUID)
            if entry.findtext('mig_mode/current_mig')!='Disabled':raise RuntimeError('MIG unsupported')
            minor=int(entry.findtext('minor_number'));devices=[f'/dev/nvidia{minor}','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools']
            if not all(Path(x).is_char_device() for x in devices):raise RuntimeError('Device missing')
            if os.minor(Path(devices[0]).stat().st_rdev)!=minor:raise RuntimeError('GPU minor mismatch')
            session=root/'sessions'/('session-%04d'%(len(sessions)+1));session.mkdir(exist_ok=False)
            assigned={**binding,'session_id':arm+'/'+session.name,'pages':pending}
            exclusive_json(session/'ASSIGNED.json',assigned)
            command=['docker','create','--label','hybrid.freeze='+freeze_sha,'--label','hybrid.arm='+arm,
                '--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--cpus','8','--memory','48g',
                '--pids-limit','512','--shm-size','8g','--tmpfs','/tmp:rw,size=2g']
            for d in devices:command+=['--device',d+':'+d]
            for k,v in {'CUDA_VISIBLE_DEVICES':UUID,'HOME':'/output','HF_HOME':'/output/hf-cache','FTLANG_CACHE':'/output/ftlang',
                'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONPATH':'/code','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1',
                'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','LD_LIBRARY_PATH':'/driver','PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK':'True'}.items():command+=['--env',k+'='+v]
            envroot=ROOT/'inference/environments';manifest=freeze['runtime_manifest']
            mounts=[(code,'/code',True),(session,'/output',False),(image_root,'/images',True),
                    (ROOT/'inference/driver-bundle-v1/attempt-v2','/driver',True)]
            mounts += [(envroot/n,str(envroot/n),True) for n in ('tele-v2','tele-v3','paddle-v2','cpython-3.12.14-20260901')]
            mounts += [(Path(manifest[k]),manifest[k],True) for k in ('tele_source','tele_model','paddle_model')]
            for src,dst,ro in mounts:command+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
            command += [IMAGE,manifest['tele_python'],'-B','-m','hybrid_v2_tele_base.full_worker','--arm',arm,
                        '--assigned','/output/ASSIGNED.json','--freeze','/code/SYSTEM_FREEZE.json']
            cid=query(command)
            exclusive_json(session/'START.json',{'cid':cid,'command':command,'before':before,'lease':str(lease_path),**binding})
            idle()
            state,reason=monitor(root,session,pending,freeze_sha,arm,cid,False)
        remaining=pending_pages(root,pages,freeze_sha,arm)
        if not remaining and state['State']['ExitCode']==0 and not state['State']['OOMKilled'] and reason is None:
            exclusive_json(root/'COMPLETE.json',{**binding,'pages':1651,'sessions':len(sessions)+1})
            return 0
        return 2


def main():
    p=argparse.ArgumentParser();p.add_argument('--code',type=Path,required=True);p.add_argument('--freeze-sha256',required=True)
    p.add_argument('--arm',choices=['tele_raw','hybrid_v2','both'],required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    code=a.code.resolve()
    if code.parent!=BASE or code.name!='full-code-v2':raise RuntimeError('Unexpected code root')
    freeze,pages,image_root=verify_code(code,a.freeze_sha256)
    for arm in (['tele_raw','hybrid_v2'] if a.arm=='both' else [a.arm]):
        result=run_arm(code,freeze,pages,image_root,a.freeze_sha256,arm,a.resume)
        if result:return result
    return 0


if __name__=='__main__':raise SystemExit(main())
