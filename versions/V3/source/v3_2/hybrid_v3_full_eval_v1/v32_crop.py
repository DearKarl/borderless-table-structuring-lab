"""One fixed first-crop attempt, one Ovis load/request, no Tele and no retry."""
import time
import uuid
from pathlib import Path
from .core import ContractError,Deadline,new_dir,read,sha,write,closed
from .runner import preflight,source_check
from .lifecycle import Leases,gpu_idle
from .gpu_admission import Admission
from .group_cleanup import finalize_group
from .bounded_cleanup import CleanupClock
from .image_identity import effective_references,resolve_identities
from . import gpu_backend
from .v32_protocol import validate

CROP_SHA='f5ec2f18fc57a3cc7fecfe1e0e9b06a0d30e295266317a57cfc12691a632e8dd'
INPUT_SHA='e23d5c145173b92041a7a0ca1a4d0faa17fabfdd731d1a1544e64ae203607b9b'
PIXEL_SHA='71c18f145e3207f8f1e93c6bbfac943f9a17bfb790d1e2dac52887a5b807892d'

def infer(inputs,runtime,budget,crop,output):
    start=time.monotonic()
    from .v32_cluster import bound_budget
    b=bound_budget(read(runtime),read(budget),start);clock=Deadline(start,b['total_seconds'],b['cleanup_seconds'])
    plan=preflight('v32-text',inputs,runtime,budget,clock,'on');plan['budget']=b;r=plan['runtime']
    if r['v32'].get('execution')!='crop' or r['v32'].get('parameter_status')!='bootstrap_header':raise ContractError('Dedicated first-crop bootstrap required')
    if len(plan['inputs']['pages'])!=1:raise ContractError('One crop parent required')
    page=plan['inputs']['pages'][0]
    if page['page_id']!='p00169' or page['file_sha256']!=INPUT_SHA or sha(crop,clock)!=CROP_SHA:raise ContractError('Fixed first-crop identity mismatch')
    if gpu_backend.kind(r)!='manual':raise ContractError('V32 first crop requires explicit manual binding')
    out=new_dir(output);run_id=uuid.uuid4().hex
    control={**plan,'profile':'v32-text','mode':'on','run_id':run_id,'gpu_binding_schema':1,'image_identity_schema':1,
             'host_started_monotonic':start,'absolute_stop_monotonic':start+b['total_seconds']-b['cleanup_seconds']}
    write(out/'CONTROL.json',control);failure=None;group=None;admitted=None;service=None
    with Leases(r['lease_directory'],r['gpu_uuids']):
        admission=Admission(r['lease_directory'],r['gpu_uuids'])
        try:
            ids=resolve_identities(run_id,effective_references(r,'on'),clock);write(out/'IMAGE_IDENTITIES.json',ids)
            write(out/'GPU_BEFORE.json',gpu_idle(r['gpu_uuids'],0,clock))
            bindings=gpu_backend.resolve_manual(control,Path(runtime).resolve().parent,out,ids,clock)
            binding=bindings['roles']['ovis']
            mounts=[(out,'/output',True),(out/'CONTROL.json','/control.json',False),
                    (Path(__file__).resolve().parent,'/framework/hybrid_v3_full_eval_v1',False),gpu_backend.driver_mount(binding)]
            from .v32_cluster import asset_root
            mounts.extend((asset_root(r,Path(runtime).resolve().parent,n),'/assets/'+n,False) for n in r['assets'])
            from .v32_service import ExpertService
            service=ExpertService(control,clock,out,mounts,ids['roles']['ovis'],binding)
            admission.reserve(run_id,[service.docker.token]);service.docker.admission=admission;service.start()
            root=out/'experts/ipc';(root/'00000001.png').write_bytes(Path(crop).read_bytes())
            v=r['v32'];request={'run_id':run_id,'page_id':'p00169','slot_id':'p00169:native:11','request_id':run_id+'/crop/1',
                'input_sha256':INPUT_SHA,'crop':{'width':795,'height':36,'mode':'RGB','rgb_sha256':PIXEL_SHA},
                'png_sha256':CROP_SHA,'png_name':'00000001.png',**{k:v[k] for k in ('model_sha256','config_sha256','source_sha256')}}
            pending=root/'00000001.pending';write(pending,request);pending.rename(root/'00000001.request.json')
            end=min(control['absolute_stop_monotonic'],time.monotonic()+120)
            while not (root/'00000001.response.json').exists():
                service.healthy();clock.remaining()
                if time.monotonic()>=end:raise ContractError('First-crop request deadline')
                time.sleep(.05)
            response=read(root/'00000001.response.json');clean=validate(request,response)
            if any(f['cache_hit'] for f in response['engine_evidence']['payload']['mm_features']):raise ContractError('First crop must observe real tensors')
            write(out/'CROP_RESULT.json',{'request':request,'response':response,'cleaned_text':clean,'protocol_accepted':True,'quality_claim':False})
            service.close()
            if source_check(clock)!=plan['source']:raise ContractError('Crop source changed')
        except BaseException as exc:failure={'type':type(exc).__name__,'message':str(exc)}
        finally:
            cleanup_clock=CleanupClock(clock)
            def release(c):return {'verified':True,'gpu_uuids':r['gpu_uuids'],'memory_mib':gpu_idle(r['gpu_uuids'],0,c),'observed_monotonic':time.monotonic()}
            if service is not None:
                group=finalize_group([service.docker],release,clock=cleanup_clock);admitted=admission.finish(group)
                write(out/'GROUP_CLEANUP.json',group);write(out/'ADMISSION_RESULT.json',admitted)
    ok=failure is None and group and group['complete'] and admitted['released'] and not admitted['blocked']
    files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
    result={'run_id':run_id,'success':bool(ok),'failure':failure,'elapsed_seconds':time.monotonic()-start,
            'source_sha256':plan['source']['sha256'],'evidence':files,'formal_collection_eligible':False,'quality_claim':False}
    write(out/'CROP_LOCK.json',result)
    if not ok:raise ContractError('First crop failed; preserve attempt and cumulative load budget')
    return result
