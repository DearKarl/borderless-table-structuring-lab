"""Host-owned isolated Docker smoke, fresh UUID lease; never touches customer jobs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hybrid_v2_tele_base.lock_checks import digest

ROOT = Path('/srv/hybrid-research')
BASE = ROOT/'inference/hybrid-v2-tele-base/attempt-001'
IMAGE = 'sha256:787dd9a15d41c442e78a5faa9cdc721f154cbf3d4cf8624edb41026c3052b75a'
UUID = os.environ.get('HYBRID_GPU_UUID', 'GPU-EXPLICIT-UUID')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while b:=stream.read(8*1024*1024): h.update(b)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x') as stream: json.dump(value,stream,indent=2)


def query(args): return subprocess.check_output(args,text=True).strip()


def idle():
    inventory=query(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'])
    processes=query(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'])
    rows=[list(map(str.strip,line.split(','))) for line in inventory.splitlines()]
    row=next((r for r in rows if r[1]==UUID),None)
    if row is None or row[0]!='1' or int(row[2])>10 or int(row[3])!=0 or UUID in processes:
        raise RuntimeError('Allocated GPU 1 is not freshly idle; do not launch')
    return {'inventory':inventory,'processes':processes}


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--lock-sha256',required=True)
    a=p.parse_args()
    code=Path(a.bundle).resolve()
    if sys.platform!='linux' or code.parent!=BASE or not code.name.startswith('smoke-code-'):
        raise RuntimeError('Unexpected immutable code path')
    if sha(code/'CODE_LOCK.json')!=digest(a.lock_sha256): raise RuntimeError('Code lock mismatch')
    lock=json.loads((code/'CODE_LOCK.json').read_bytes())
    for relative,expected in lock.items():
        path=(code/relative).resolve()
        if not path.is_relative_to(code) or sha(path)!=digest(expected): raise RuntimeError('Code/manifest mismatch')
    manifest=json.loads((code/'SMOKE_MANIFEST.json').read_bytes())
    if manifest['gpu_uuid']!=UUID or not manifest['capacity']['passed']: raise RuntimeError('Manifest gate failed')
    # Bind the same real assets checked by preflight, including auxiliary resources.
    for row in manifest['identity_files']:
        if not Path(row['path']).resolve().is_relative_to(ROOT/'inference') or sha(row['path'])!=digest(row['sha256']):
            raise RuntimeError('Source/model/aux changed')
    driver_lock=manifest['driver_lock']
    driver_receipt=ROOT/'receipts/GPU_BIND_v1/attempt-v2/DRIVER_BUNDLE.json'
    if sha(driver_receipt)!=digest(driver_lock['receipt_sha256']):
        raise RuntimeError('Driver receipt mismatch')
    actual_driver=json.loads(driver_receipt.read_bytes())
    if actual_driver['files']!=driver_lock['files']:raise RuntimeError('Driver file manifest differs')
    driver_bundle=ROOT/'inference/driver-bundle-v1/attempt-v2'
    for row in driver_lock['files']:
        if row['present']:
            expected=digest(row['sha256'])
            if not Path(row['target']).resolve().is_relative_to(driver_bundle):raise RuntimeError('Driver target escapes bundle')
            if sha(row['target'])!=expected or sha(row['source'])!=expected:raise RuntimeError('Driver source or copy changed')
            if (driver_bundle/row['soname']).resolve()!=Path(row['target']).resolve():raise RuntimeError('Driver SONAME target changed')
    lease_path=ROOT/'receipts/REMOTE_NATIVE_SMOKE_v1'/(UUID+'.lease')
    with lease_path.open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        before=idle()
        entry=next(g for g in ET.fromstring(query(['nvidia-smi','-q','-x'])).findall('gpu') if g.findtext('uuid')==UUID)
        if entry.findtext('mig_mode/current_mig')!='Disabled': raise RuntimeError('MIG unsupported')
        minor=int(entry.findtext('minor_number'))
        devices=[f'/dev/nvidia{minor}','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools']
        if not all(Path(d).is_char_device() for d in devices): raise RuntimeError('GPU device missing')
        if os.minor(Path(devices[0]).stat().st_rdev)!=minor: raise RuntimeError('GPU minor mismatch')
        output=BASE/('smoke-output-'+code.name.removeprefix('smoke-code-'));output.mkdir(exist_ok=False)
        command=['docker','create','--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges',
            '--cpus','8','--memory','48g','--pids-limit','512','--shm-size','8g','--tmpfs','/tmp:rw,size=2g']
        for d in devices:command+=['--device',d+':'+d]
        for k,v in {'CUDA_VISIBLE_DEVICES':UUID,'HOME':'/output','HF_HOME':'/output/hf-cache',
            'FTLANG_CACHE':'/output/ftlang','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONPATH':'/code',
            'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4',
            'LD_LIBRARY_PATH':'/driver','PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK':'True'}.items():command+=['--env',k+'='+v]
        envroot=ROOT/'inference/environments'
        mounts=[(code,'/code',True),(output,'/output',False),
                (ROOT/'inference/driver-bundle-v1/attempt-v2','/driver',True)]
        mounts += [(envroot/n,str(envroot/n),True) for n in ('tele-v2','tele-v3','paddle-v2','cpython-3.12.14-20260901')]
        mounts += [(Path(manifest[k]),manifest[k],True) for k in ('tele_source','tele_model','paddle_model')]
        for src,dst,ro in mounts:command+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
        command += [IMAGE,manifest['tele_python'],'-B','-m','hybrid_v2_tele_base.smoke','--manifest','/code/SMOKE_MANIFEST.json']
        cid=query(command)
        save(output/'START.json',{'cid':cid,'command':command,'lease':str(lease_path),'before':before,
             'manifest_sha256':sha(code/'SMOKE_MANIFEST.json'),'code_lock_sha256':a.lock_sha256})
        cli=None;timeout=False; foreign_processes=[]
        try:
            idle()  # Recheck immediately before this owned container is started.
            with (output/'stdout.log').open('x') as stdout,(output/'stderr.log').open('x') as stderr:
                attached=subprocess.Popen(['docker','start','--attach',cid],stdout=stdout,stderr=stderr)
                started=time.monotonic()
                while attached.poll() is None:
                    state=json.loads(query(['docker','inspect',cid]))[0]
                    if state['State']['Running']:
                        owned={line.strip() for line in query(['docker','top',cid,'-eo','pid']).splitlines()[1:]}
                        active=query(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'])
                        foreign_processes=[line for line in active.splitlines()
                                           if line.split(',')[0].strip()==UUID and line.split(',')[1].strip() not in owned]
                        if foreign_processes:break
                    if time.monotonic()-started>14400:timeout=True;break
                    try:attached.wait(timeout=10)
                    except subprocess.TimeoutExpired:pass
                cli=attached.poll()
        finally:
            state=json.loads(query(['docker','inspect',cid]))[0]
            if state['State']['Running']:subprocess.run(['docker','kill',cid],check=True,capture_output=True)
            if state['State']['Status']!='created':subprocess.run(['docker','wait',cid],check=True,capture_output=True)
            state=json.loads(query(['docker','inspect',cid]))[0]
            save(output/'EXIT.json',{'container':state,'cli_exit':cli,'timeout':timeout,'foreign_processes':foreign_processes,
                 'finished_at':time.time(),'after_processes':query(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'])})
        # Lease stays open until the owned container is confirmed stopped.
        if state['State']['Running']:raise RuntimeError('Owned container still running')
    print(json.dumps({'output':str(output),'exit_code':cli,'timeout':timeout}))
    return 0 if cli==0 and not timeout and not state['State']['OOMKilled'] else 1


if __name__=='__main__':raise SystemExit(main())
