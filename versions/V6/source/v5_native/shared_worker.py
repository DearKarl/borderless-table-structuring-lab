"""Seven paired outputs from one native page, with four frozen direct actors."""
import argparse
import json
import signal
import time
import traceback
from pathlib import Path

from .actor import DirectActors,InputObserver
from .assembly import assemble,targets
from .io_utils import PageDeadline,atomic_json,commit_stage,digest,finalize_outputs,process_identity,read_json,require_common_runtime,utc
from .native import native_page
from .regions import region_crop
from .runtime import runtime_receipt

VERSIONS={'D0':(None,'D0'),'GBDT_V6.1.1':('gbdt','DT'),'GBDT_V6.1.2':('gbdt','DF'),
          'GBDT_V6.1.3':('gbdt','DTF'),'MLP_V6.2.1':('mlp','DT'),'MLP_V6.2.2':('mlp','DF'),'MLP_V6.2.3':('mlp','DTF')}


def finalize_shared(folder,output,identifier,record):
    """Retain each arm's last atomic stage even after a killed partial family."""
    return finalize_outputs(folder,output,identifier,record,VERSIONS)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--control',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--worker-id',required=True);args=ap.parse_args()
    control=read_json(args.control);protocol=read_json(control['protocol_path'])
    if control['arm']!='shared' or digest(control['protocol_path'])!=control['protocol_sha256']:
        raise RuntimeError('Shared worker requires its frozen native/direct protocol')
    if set(protocol['arms'])!=set(VERSIONS):raise RuntimeError('Seven output identities differ')
    output=Path(args.output);wd=output/'workers'/args.worker_id;wd.mkdir(parents=True,exist_ok=True)
    progress={'worker_id':args.worker_id,'run_id':control['run_id'],'identity':process_identity(),'started_at':utc(),
              'phase':'model_loading','current_page':None,'completed_in_process':0,'physical_gpu_index':control['physical_gpu_index'],
              'gpu_uuid':control['gpu_uuid']}
    def status(**kwargs):
        progress.update(kwargs,updated_at=utc(),heartbeat_monotonic=time.monotonic());atomic_json(wd/'STATUS.json',progress)
    status()
    from PIL import Image
    Image.MAX_IMAGE_PIXELS=None
    import TeleOCR.config as config
    config.MAX_MODEL_LEN=protocol['engine']['max_model_len'];config.GPU_MEMORY_UTILIZATION=protocol['engine']['gpu_memory_utilization']
    from TeleOCR.vlm_utils.TeleOCR_model import TeleOCRMODEL_SERVICE
    from TeleOCR.src.vlm_middle_json_mkcontent import union_make
    client=TeleOCRMODEL_SERVICE.get_model(protocol['backend'],control['model_path'],None,**protocol['engine'])
    client.client.use_tqdm=False
    runtime=runtime_receipt(client,protocol)
    if runtime['cuda_device_count']!=1 or runtime['cuda_device_uuid'].removeprefix('GPU-')!=control['gpu_uuid'].removeprefix('GPU-'):
        raise RuntimeError('Physical GPU assignment changed')
    require_common_runtime(runtime,control.get('expected_common_runtime',{}));atomic_json(wd/'RUNTIME.json',runtime)
    events=None;stage=None;page_start=None
    def emit(row):
        if events is not None:
            events.write(json.dumps({'at':utc(),'stage':stage,'elapsed_seconds':time.monotonic()-page_start,**row},default=str)+'\n');events.flush()
    observer=InputObserver(client,emit)
    actors=DirectActors(client,observer,control['controllers'])
    atomic_json(wd/'MODELS_LOADED.json',{'at':utc(),'device':'cpu','models':{
        key:{'sha256':model.sha256,'family':model.model['family'],'kind':model.model['kind'],'contract':model.model['contract']}
        for key,model in actors.models.items()},'runtime_scale_search':False})
    def alarm(*_):raise PageDeadline('Shared page work deadline reached')
    signal.signal(signal.SIGALRM,alarm);status(phase='ready',model_ready_at=utc());failures=0
    for sample in control['pages']:
        identifier=sample['page_id']
        if (output/'receipts'/(identifier+'.json')).exists():continue
        if time.time()>=control['absolute_deadline_unix']:return 30
        folder=output/'pages'/identifier;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'START.json').exists():raise RuntimeError('Started benchmark page cannot be replayed')
        source=Path(control['input_root'])/sample['file']
        if digest(source)!=sample['input_sha256']:raise RuntimeError('Frozen page input changed')
        observer.begin_page();actors.begin_page();page_start=time.monotonic();stage='native'
        record={'arm':'shared','run_id':control['run_id'],'worker_id':args.worker_id,'started_at':utc(),
                'start_monotonic':page_start,'status':'success','error':None,'input_sha256':sample['input_sha256'],
                'completed_stages':[],'timing':{},'physical_gpu_index':control['physical_gpu_index'],'gpu_uuid':control['gpu_uuid']}
        atomic_json(folder/'START.json',record);events=open(folder/'events.jsonl','a',encoding='utf-8')
        status(phase='inference',current_page=identifier,page_start_monotonic=page_start,stage=stage)
        signal.setitimer(signal.ITIMER_REAL,min(590,max(.001,control['absolute_deadline_unix']-time.time())))
        fatal=None
        try:
            start=time.monotonic();middle,markdown,image=native_page(client,source,folder,sample.get('source_page_index'))
            record['timing']['native_seconds']=time.monotonic()-start;record['native_model_requests']=len(observer.submitted_request_ids)
            commit_stage(folder,'native',middle,markdown)
            for version in VERSIONS:commit_stage(folder/'arms'/version,'native',middle,markdown)
            record['completed_stages'].append('native');regions=targets(middle)
            atomic_json(folder/'ELIGIBLE_REGIONS.json',{'regions':regions,'render_dpi':200,'eligibility':'Native table/interline_equation spans with content and bounding box'})
            prepared={};crops={};stage='features_and_reference_preprocessing';status(stage=stage)
            for region in regions:
                i=region['index'];start=time.monotonic();crop=region_crop(image,middle['pdf_info'][0],region,1.)
                crop_seconds=time.monotonic()-start;data=actors.prepare(crop,region['native'],region['kind'])
                data.update(crop_seconds=crop_seconds,region_id=i,kind=region['kind'],bbox=region['bbox'])
                prepared[i]=data;crops[i]=crop;emit({'event':'region_features',**data})
            atomic_json(folder/'FEATURES.json',prepared)
            for family in ('gbdt','mlp'):
                stage=family+'_regional';status(stage=stage);family_start=time.monotonic();candidates={}
                for region in regions:
                    i=region['index'];emit({'event':'actor_started','family':family,'region_id':i,'kind':region['kind']})
                    row=actors.apply(crops[i],region['native'],region['kind'],prepared[i],family,i)
                    candidates[i]=row;atomic_json(folder/(family+'_CANDIDATES.json'),candidates);emit({'event':'actor_completed',**row})
                record['timing'][family+'_regional_seconds']=time.monotonic()-family_start
                stage=family+'_assembly';status(stage=stage);start=time.monotonic()
                for version,(owner,arm) in VERSIONS.items():
                    if owner!=family:continue
                    changed,audit=assemble(middle,candidates,arm)
                    md=union_make(changed['pdf_info'],'images') if any(x['applied'] for x in audit) else markdown
                    atomic_json(folder/'arms'/version/'ASSEMBLY.json',{'version':version,'audit':audit,'shared_component_candidates':True})
                    commit_stage(folder/'arms'/version,'direct_actor',changed,md)
                record['timing'][family+'_assembly_seconds']=time.monotonic()-start;record['completed_stages'].append(family)
            failures=0
        except PageDeadline as exc:
            record.update(status='timeout',error={'type':type(exc).__name__,'stage':stage,'message':str(exc)})
        except Exception as exc:
            record.update(status='failed',error={'type':type(exc).__name__,'stage':stage,'message':str(exc)[:2000]})
            emit({'event':'exception','traceback':traceback.format_exc()})
            if type(exc).__name__ in ('EngineDeadError','OutOfMemoryError','AttributeError','KeyError','RuntimeError'):fatal=exc
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            try:observer.cancel()
            except Exception as exc:fatal=exc
            record.update(elapsed_seconds=time.monotonic()-page_start,request_count=len(observer.request_ids),
                          submitted_request_count=len(observer.submitted_request_ids))
            finalize_shared(folder,output,identifier,record);events.close();events=None
            status(current_page=None,page_start_monotonic=None,completed_in_process=progress['completed_in_process']+1)
        if record['status']!='success':failures+=1
        if fatal is not None or failures>=3:
            status(phase='systemic_failure',error=str(fatal));return 20
    status(phase='inference_complete',finished_at=utc());return 0


if __name__=='__main__':raise SystemExit(main())
