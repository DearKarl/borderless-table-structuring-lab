"""One resident native Tele process and optional resident Paddle process per session."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time
import traceback
from .full_state import read, sha, exclusive_json, exclusive_bytes, progress
from .runtime_load import load_runtime
from .smoke import observe, persist_page_audit, save
from .expert_process import ExpertProcess
from .region_protocol import RunBlocked, PauseAfterFallback


class PageDeadline(TimeoutError):pass


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['tele_raw','hybrid_v2'],required=True)
    parser.add_argument('--assigned',required=True);parser.add_argument('--freeze',required=True)
    args=parser.parse_args();out=Path('/output');assigned=read(args.assigned);freeze=read(args.freeze)
    freeze_sha=sha(args.freeze);manifest=freeze['runtime_manifest']
    if assigned['freeze_sha256']!=freeze_sha or assigned['arm']!=args.arm:raise RunBlocked('Assignment identity differs')
    summary={'arm':args.arm,'freeze_sha256':freeze_sha,'model_loads':{'tele':0,'paddle':0},'complete':False}
    expert=None;rt=None;page=None
    def stage(phase,**extra):
        progress(out/'PROGRESS.json',{'phase':phase,'time':time.time(),'arm':args.arm,**extra})
        print(json.dumps({'phase':phase,'time':time.time(),**extra}),flush=True)
    try:
        stage('load_tele')
        rt=load_runtime(manifest,out,summary)
        for name in ('PROCESSOR.json','TELE_LOAD.json'):
            if read(out/name)!=freeze['native_runtime'][name]:raise RunBlocked('Native Tele load metadata changed: '+name)
        if args.arm=='hybrid_v2':
            stage('load_paddle')
            expert=ExpertProcess(manifest['paddle_python'],manifest['paddle_model'],out/'paddle-worker','/code',
                                 manifest='/code/RUNTIME_MANIFEST.json')
            summary['model_loads']['paddle']=expert.load['model_loads']
            for key,value in freeze['native_runtime']['paddle_load'].items():
                if expert.load.get(key)!=value:raise RunBlocked('Native Paddle load metadata changed: '+key)
        completed=0
        for page in assigned['pages']:
            if (out/'STOP_REQUEST.json').exists():
                summary['pause_reason']='owner_or_resource_stop';break
            folder=out/'pages'/page['key'];folder.mkdir(parents=True,exist_ok=False)
            if sha(page['image_path'])!=page['input_sha256']:raise RunBlocked('Original image identity mismatch')
            exclusive_json(folder/'STARTED.json',{**page,'time':time.time(),'freeze_sha256':freeze_sha,'arm':args.arm})
            stage('page',key=page['key'],page_id=page['page_id'],completed=completed,total=len(assigned['pages']))
            events=[];captured={};rt['generation'].clear()
            native_args=dict(backend='transformers',model=rt['model'],processor=rt['processor'],batch_size=1,max_concurrency=1)
            if args.arm=='tele_raw':client=rt['TeleOCRClient'](**native_args)
            else:
                client=rt['hybrid_class'](**native_args,hybrid_mode='on',formula_expert=expert,emit=events.append)
                client.set_context(assigned['session_id'],page['page_id'],page['input_sha256'])
            observe(client,folder,captured)
            primary_error=None;primary_traceback=None;started=time.monotonic()
            try:
                def expired(*_):raise PageDeadline('Native page deadline 900 seconds')
                signal.signal(signal.SIGALRM,expired);signal.alarm(900)
                rt['do_parse'](str(folder/'native'),[page['key']],[rt['read_fn'](page['image_path'])],[None],predictor=client)
            except BaseException as exc:
                primary_error=exc;primary_traceback=traceback.format_exc()
            finally:signal.alarm(0)
            # Logs are independent of terminal classification; unknown errors never become failed predictions.
            persistence_error=None
            try:persist_page_audit(folder,[('GENERATION.json',rt['generation']),('REGIONS.json',events),
                    ('BLOCKS.json',captured)],primary_error,primary_traceback)
            except BaseException as exc:persistence_error=exc
            if primary_error is None and persistence_error is not None:raise persistence_error
            if primary_error is not None and not isinstance(primary_error,(PageDeadline,PauseAfterFallback)):
                raise primary_error
            resources=rt['resources'].result();save(folder/'RESOURCE_AUDIT.json',resources)
            if resources['unknown_resources']:raise RunBlocked('Unknown runtime auxiliary resource')
            failed=primary_error is not None
            if failed:
                data=b'';status='failed'
            else:
                data=(folder/'native'/page['key']/(page['key']+'.md')).read_bytes()
                # Preserve native truncated content exactly, matching the official allow_truncated_content setting.
                truncated=any(x.get('returned') and x['termination']['stop']!='eos' for x in rt['generation'])
                status='truncated' if truncated else 'success'
            exclusive_bytes(folder/'prediction.md',data)
            exclusive_json(folder/'PAGE_RESULT.json',{
                **{k:page[k] for k in ('key','page_id','input_sha256')},'arm':args.arm,'freeze_sha256':freeze_sha,
                'status':status,'prediction_sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),
                'elapsed_seconds':time.monotonic()-started,'error':None if not failed else
                    {'type':type(primary_error).__name__,'message':str(primary_error),'traceback':primary_traceback},
                'routes':{name:sum(e.get('route')==name for e in events) for name in ('tele_native','paddle_success','tele_fallback')},
                'tele_calls':len(rt['generation']), 'paddle_worker_session':None if expert is None else expert.load['worker_session'],
                'model_loads':summary['model_loads']})
            completed+=1;stage('page_complete',key=page['key'],completed=completed,total=len(assigned['pages']),status=status)
            if failed:
                summary['pause_reason']=type(primary_error).__name__;break
        else:summary['complete']=True
        summary['completed_this_session']=completed
    except BaseException as exc:
        summary.update(blocked=True,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
        if expert is not None:
            expert.close(force=not summary['complete'])
            if summary['complete'] and expert.process.returncode!=0:
                summary.update(complete=False,blocked=True,error='Expert shutdown failed')
        if rt is not None:
            save(out/'RESOURCE_AUDIT.json',rt['resources'].result())
            summary['peak_tele_allocated_bytes']=rt['torch'].cuda.max_memory_allocated()
            summary['peak_tele_reserved_bytes']=rt['torch'].cuda.max_memory_reserved()
        save(out/'SESSION_RESULT.json',summary)
        stage('complete' if summary['complete'] else 'blocked' if summary.get('blocked') else 'paused')
    return 0 if summary['complete'] else 2 if summary.get('pause_reason') else 1


if __name__=='__main__':raise SystemExit(main())
