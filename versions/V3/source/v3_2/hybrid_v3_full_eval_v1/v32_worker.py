"""Isolated resident Ovis worker; one construction, ordered requests, fatal system errors."""
import json
import os
from pathlib import Path
import signal
import time
import traceback
from .core import ContractError,DeadlineError,closed,read,sha,write,progress
from .request_binding import rgb_binding
from .v32_protocol import ContentRejected,validate

def main():
    control=read('/control.json');out=Path('/output');root=out/'experts';ipc=root/'ipc'
    runtime=control['runtime'];provider=None;resources=None;sequence=1
    def expired(*_):raise DeadlineError('Ovis phase/host deadline')
    signal.signal(signal.SIGALRM,expired)
    def arm(cap):
        left=min(cap,control['absolute_stop_monotonic']-time.monotonic())
        if left<=0:raise DeadlineError('Ovis host deadline')
        signal.setitimer(signal.ITIMER_REAL,left)
    def event(name,**kw):
        with (root/'LEDGER.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({'event':name,'monotonic':time.monotonic(),**kw})+'\n');f.flush()
    try:
        start=time.monotonic();progress(root/'PROGRESS.json',{'phase':'load','monotonic':start});arm(control['budget']['load_seconds'])
        from .v32_gpu import adapt
        adapt(control,out)
        from .gpu_backend import worker_guard
        actual=worker_guard(control,'ovis',out)['actual_uuid']
        from .v32_resources import resource_audit
        resources=resource_audit(runtime)
        from .asset_binding import load_boundary
        with resources.integrity_boundary(runtime):boundary=load_boundary(runtime,'ovis')
        write(root/'LOAD_BOUNDARY.json',boundary)
        from .v32_provider import Ovis
        provider=Ovis(control,root,event)
        if resources.result()['unknown_resources']:raise ContractError('Unknown parent Ovis resource')
        signal.setitimer(signal.ITIMER_REAL,0)
        write(root/'READY.json',{'run_id':control['run_id'],'pid':os.getpid(),'gpu_uuid':actual,'components':['text'],'model_loads':1,
            'host_environment_binding_sha256':runtime['environments']['ovis']['host_environment_binding']['sha256']})
        seen=set();counts={}
        while True:
            if time.monotonic()>=control['absolute_stop_monotonic']:raise DeadlineError('Ovis host deadline')
            if (root/'CLOSE.json').exists():
                if read(root/'CLOSE.json')!={'run_id':control['run_id']}:raise ContractError('Ovis close identity')
                break
            file=ipc/f'{sequence:08d}.request.json'
            if not file.exists():time.sleep(.05);continue
            request=read(file);v=runtime['v32'];page=next((p for p in control['inputs']['pages'] if p['page_id']==request['page_id']),None)
            if (request['run_id']!=control['run_id'] or page is None or request['input_sha256']!=page['file_sha256']
                or any(request[k]!=v[k] for k in ('config_sha256','source_sha256','model_sha256'))):raise ContractError('Ovis request identity differs')
            if request['request_id'] in seen:raise ContractError('Ovis duplicate request')
            seen.add(request['request_id']);key=request['page_id'];counts[key]=counts.get(key,0)+1
            crop=v.get('execution')=='crop'
            if counts[key]>min(256,control['budget']['expert_calls_per_page']) or sequence>(1 if crop else 512):raise ContractError('Ovis cumulative request budget')
            png=closed(ipc,request['png_name'])
            if sha(png)!=request['png_sha256']:raise ContractError('Ovis PNG hash mismatch')
            from PIL import Image
            with Image.open(png) as source:
                source.load();image=source.copy()
            try:
                if rgb_binding(image)!=request['crop']:raise ContractError('Ovis actual RGB mismatch')
                event('CALL_START',engine='ovis',request_id=request['request_id'],page_id=key);arm(120)
                response=provider.call(request,image)
                # Persist original result even if the real EOS contract is incompatible.
                write(ipc/f'{sequence:08d}.raw-response.json',response)
                try:validate(request,response)
                except ContentRejected:
                    if crop:raise
                if resources.result()['unknown_resources']:raise ContractError('Unknown Ovis request resource')
                pending=ipc/f'{sequence:08d}.reply-pending';write(pending,response);pending.rename(ipc/f'{sequence:08d}.response.json')
                event('CALL_RESULT',engine='ovis',request_id=request['request_id'],returned=True)
                signal.setitimer(signal.ITIMER_REAL,0)
            finally:image.close()
            sequence+=1
        write(root/'SESSION_RESULT.json',{'normal_exit':True,'run_id':control['run_id'],'model_loads':1,'requests':sequence-1})
        return 0
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL,0)
        write(root/'ERROR.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);errors=[]
        if provider is not None:
            try:provider.close()
            except BaseException as exc:errors.append({'type':type(exc).__name__,'message':str(exc)})
        write(root/'FINAL_AUDIT.json',{'resource_audit':resources.result() if resources else {},'close_errors':errors,
            'host_environment_binding_sha256':runtime['environments']['ovis']['host_environment_binding']['sha256']})
        if errors:raise ContractError('Ovis close failed')

if __name__=='__main__':raise SystemExit(main())
