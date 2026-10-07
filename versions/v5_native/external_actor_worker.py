"""Actual direct-scale validation/test actions on saved external native regions."""
import argparse,json,signal,time,traceback
from pathlib import Path

from .actor import DirectActors,InputObserver,text_identity
from .features import scaled
from .io_utils import PageDeadline,atomic_json,commit_stage,digest,finalize_page,process_identity,read_json,require_common_runtime,utc
from .native import pixel_identity,realized_view
from .regions import runaway
from .runtime import runtime_receipt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--control',required=True);ap.add_argument('--output',required=True);ap.add_argument('--worker-id',required=True);args=ap.parse_args()
    control=read_json(args.control);protocol=read_json(control['protocol_path'])
    assert control['arm']=='actor_validation' and digest(control['protocol_path'])==control['protocol_sha256']
    output=Path(args.output);wd=output/'workers'/args.worker_id;wd.mkdir(parents=True,exist_ok=True)
    progress={'worker_id':args.worker_id,'run_id':control['run_id'],'identity':process_identity(),'phase':'model_loading',
              'started_at':utc(),'current_page':None,'completed_in_process':0,'physical_gpu_index':control['physical_gpu_index'],'gpu_uuid':control['gpu_uuid']}
    def status(**kwargs):
        progress.update(kwargs,updated_at=utc(),heartbeat_monotonic=time.monotonic());atomic_json(wd/'STATUS.json',progress)
    status()
    from PIL import Image
    Image.MAX_IMAGE_PIXELS=None
    import TeleOCR.config as config
    config.MAX_MODEL_LEN=protocol['engine']['max_model_len'];config.GPU_MEMORY_UTILIZATION=protocol['engine']['gpu_memory_utilization']
    from TeleOCR.vlm_utils.TeleOCR_model import TeleOCRMODEL_SERVICE
    client=TeleOCRMODEL_SERVICE.get_model(protocol['backend'],control['model_path'],None,**protocol['engine']);client.client.use_tqdm=False
    runtime=runtime_receipt(client,protocol)
    if runtime['cuda_device_count']!=1 or runtime['cuda_device_uuid'].removeprefix('GPU-')!=control['gpu_uuid'].removeprefix('GPU-'):
        raise RuntimeError('Physical GPU assignment changed')
    require_common_runtime(runtime,control.get('expected_common_runtime',{}));atomic_json(wd/'RUNTIME.json',runtime)
    events=None;stage=None;started=None
    def emit(row):
        if events is not None:
            events.write(json.dumps({'at':utc(),'stage':stage,'elapsed_seconds':time.monotonic()-started,**row},default=str)+'\n');events.flush()
    observer=InputObserver(client,emit);actors=DirectActors(client,observer,control['controllers'])
    atomic_json(wd/'MODELS_LOADED.json',{'at':utc(),'device':'cpu','models':{k:m.sha256 for k,m in actors.models.items()},'runtime_scale_search':False})
    def alarm(*_):raise PageDeadline('External actor region deadline')
    signal.signal(signal.SIGALRM,alarm);status(phase='ready',model_ready_at=utc());failures=0
    for row in control['pages']:
        rid=row['page_id'];assert row['split'] in ('validation','test') and row['unit_contract']=='source_page_native_predicted_region_v1'
        if (output/'receipts'/(rid+'.json')).exists():continue
        if time.time()>=control['absolute_deadline_unix']:return 30
        folder=output/'pages'/rid;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'START.json').exists():raise RuntimeError('Started actor sample cannot be replayed')
        source=Path(control['input_root'])/row['file'];assert digest(source)==row['input_sha256']
        with Image.open(source) as saved:crop=saved.copy()
        assert pixel_identity(crop)==row['native_pixel_identity']
        started=time.monotonic();observer.begin_page();actors.begin_page();stage='features'
        record={'arm':'actor_validation','run_id':control['run_id'],'worker_id':args.worker_id,'started_at':utc(),'start_monotonic':started,
                'input_sha256':row['input_sha256'],'status':'success','error':None,'native_ocr_requests':0}
        atomic_json(folder/'START.json',record);events=open(folder/'events.jsonl','a',encoding='utf-8')
        status(phase='inference',current_page=rid,page_start_monotonic=started,stage=stage)
        signal.setitimer(signal.ITIMER_REAL,min(590,max(.001,control['absolute_deadline_unix']-time.time())))
        probe={'region_id':rid,'kind':row['kind'],'split':row['split'],'source_group':row['source_group'],'baseline':row['baseline'],
               'scope':'external_source_page_native_region','actions':[]}
        fatal=None
        try:
            commit_stage(folder,'reused_native_region',{'unit_contract':row['unit_contract']},row['baseline'])
            prepared=actors.prepare(crop,row['baseline'],row['kind']);probe['prepared']=prepared
            import numpy as np
            if not np.allclose(prepared['features'],row['features'],rtol=0,atol=1e-7):raise RuntimeError('Native actor features changed')
            atomic_json(folder/'ACTOR.json',probe)
            for family in ('gbdt','mlp'):
                stage=family;status(stage=stage)
                action=actors.apply(crop,row['baseline'],row['kind'],prepared,family,rid)
                probe['actions'].append(action);atomic_json(folder/'ACTOR.json',probe)
            stage='validation_selected_fixed';status(stage=stage)
            action=actors.apply_fixed(crop,row['baseline'],row['kind'],prepared,control['external_fixed_scales'][row['kind']],rid)
            probe['actions'].append(action);atomic_json(folder/'ACTOR.json',probe);failures=0
        except PageDeadline as exc:record.update(status='timeout',error={'type':type(exc).__name__,'stage':stage,'message':str(exc)})
        except Exception as exc:
            record.update(status='failed',error={'type':type(exc).__name__,'stage':stage,'message':str(exc)[:2000]});emit({'event':'exception','traceback':traceback.format_exc()})
            if type(exc).__name__ in ('RuntimeError','EngineDeadError','OutOfMemoryError','AttributeError','KeyError'):fatal=exc
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            try:observer.cancel()
            except Exception as exc:fatal=exc
            record.update(elapsed_seconds=time.monotonic()-started,request_count=len(observer.request_ids),
                          submitted_request_count=len(observer.submitted_request_ids))
            finalize_page(folder,output,rid,record);events.close();events=None
            status(current_page=None,page_start_monotonic=None,completed_in_process=progress['completed_in_process']+1)
        if record['status']!='success':failures+=1
        if fatal is not None or failures>=3:status(phase='systemic_failure',error=str(fatal));return 20
    status(phase='inference_complete',finished_at=utc());return 0


if __name__=='__main__':raise SystemExit(main())
