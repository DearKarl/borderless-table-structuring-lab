"""Root executes on Linux. One CPU-only native Docker probe; no Tele load."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

IMAGE='sha256:5dd2b6a864a9565cde9216c8da3db3a8b05c93cbdca9ef417f3956841ce0722f'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n',encoding='utf-8')
def main():
    p=argparse.ArgumentParser()
    for k in ('code','overlay-root','model','fixture','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();out=Path(a.output).resolve();out.mkdir(exist_ok=False)
    owner=uuid.uuid4().hex;name='v4-dep-'+owner;cid=None;absent=False;rc=None;result=None;error=None
    overlay=Path(a.overlay_root).resolve();fixture=Path(a.fixture).resolve()
    mounts=[(Path(a.code).resolve(),'/v4-code'),(overlay/'site','/v4-deps'),
        (overlay/'OVERLAY.json','/v4-overlay.json'),(Path(a.model).resolve(),'/v4-model'),(fixture,'/v4-fixture.json')]
    if any(',' in str(s) or not s.exists() for s,_ in mounts):raise ValueError('Missing/unsafe mount')
    deadline=time.monotonic()+60
    def call(argv,**kw):return subprocess.run(argv,text=True,capture_output=True,timeout=max(.01,deadline-time.monotonic()),**kw)
    try:
        argv=['docker','create','--name',name,'--label','v4-dependency-owner='+owner,
            '--read-only','--network','none','--cpus','1','--memory','2g','--memory-swap','2g',
            '--pids-limit','128','--runtime','runc','--cap-drop','ALL','--security-opt','no-new-privileges',
            '--user',str(os.getuid())+':'+str(os.getgid()),'--tmpfs','/tmp:rw,noexec,nosuid,size=64m',
            '--env','NVIDIA_VISIBLE_DEVICES=void','--env','CUDA_VISIBLE_DEVICES=',
            '--env','PYTHONDONTWRITEBYTECODE=1','--env','PYTHONPATH=/v4-code',
            '--env','OMP_NUM_THREADS=1','--env','OPENBLAS_NUM_THREADS=1','--env','MKL_NUM_THREADS=1']
        for source,target in mounts:argv+=['--mount','type=bind,source='+str(source)+',target='+target+',readonly']
        argv+=['--entrypoint','/opt/v31-native/bin/python',IMAGE,'-B','-m','hybrid.v4_selected_eval.dependencies',
            '--overlay','/v4-deps','--overlay-manifest','/v4-overlay.json','--overlay-sha',sha(overlay/'OVERLAY.json'),
            '--model','/v4-model','--fixture','/v4-fixture.json','--fixture-sha',sha(fixture)]
        made=call(argv,check=True);cid=made.stdout.strip()
        if not re.fullmatch('[0-9a-f]{64}',cid):raise ValueError('Invalid created CID')
        state=json.loads(call(['docker','inspect',cid],check=True).stdout)[0];save(out/'INSPECT.json',state)
        h=state['HostConfig']
        if (state['Image']!=IMAGE or state['Config']['Labels'].get('v4-dependency-owner')!=owner
            or not h['ReadonlyRootfs'] or h['NetworkMode']!='none' or h.get('Devices') or h.get('DeviceRequests')
            or h['NanoCpus']!=1000000000 or h['Memory']!=2*1024**3 or any(x['RW'] for x in state['Mounts'])):
            raise ValueError('CPU isolation inspection failed')
        process=call(['docker','start','--attach',cid]);rc=process.returncode
        (out/'STDOUT.txt').write_text(process.stdout,encoding='utf-8');(out/'STDERR.txt').write_text(process.stderr,encoding='utf-8')
        if rc!=0:raise ValueError('Native dependency probe failed')
        result=json.loads(process.stdout)
        if result.get('passed') is not True or result.get('parity') is not True:raise ValueError('No passing prediction receipt')
    except BaseException as exc:error=repr(exc)
    finally:
        # Use the exact unique name if create timed out before returning its CID.
        target=cid or name;cleanup_end=time.monotonic()+25
        def cleanup(argv):return subprocess.run(argv,text=True,capture_output=True,timeout=max(.01,cleanup_end-time.monotonic()))
        try:
            inspected=cleanup(['docker','inspect',target])
            if inspected.returncode==0:
                state=json.loads(inspected.stdout)[0]
                if state['Config']['Labels'].get('v4-dependency-owner')!=owner or state['Image']!=IMAGE:raise ValueError('Cleanup ownership differs')
                exact=state['Id']
                if state['State']['Running']:cleanup(['docker','kill',exact])
                cleanup(['docker','rm',exact])
            checked=cleanup(['docker','inspect',target])
            absent=(checked.returncode==1 and checked.stdout.strip() in ('','[]') and
                re.fullmatch(r'(?i:(?:error:\s*|error response from daemon:\s*)?no such (?:object|container):\s*)'+re.escape(target),checked.stderr.strip()) is not None)
        except BaseException as exc:error=error or repr(exc)
        receipt=dict(schema='v4_native_gbdt_probe_envelope_v1',image=IMAGE,exit_code=rc,
            container_id=cid,container_absent=absent,network='none',readonly=True,gpu_devices=[],
            inspect_sha256=sha(out/'INSPECT.json') if (out/'INSPECT.json').exists() else None,
            result=result,error=error,passed=error is None and absent and rc==0)
        save(out/'RECEIPT.json',receipt)
    print(json.dumps(dict(receipt=str(out/'RECEIPT.json'),sha256=sha(out/'RECEIPT.json'),passed=receipt['passed'])))
    return 0 if receipt['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
