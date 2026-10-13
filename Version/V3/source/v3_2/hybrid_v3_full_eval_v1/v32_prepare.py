"""Native preparation reuse plus an explicitly separate Ovis deployment."""
import copy
import shutil
from pathlib import Path
from .core import ContractError,canonical,closed,new_dir,read,sha,write,verify_tree

def prepare(configuration_path,output):
    from .asset_binder import prepare as native_prepare,source_path
    cfg=read(configuration_path);base=Path(configuration_path).resolve().parent;out=new_dir(output)
    active=cfg.get('mode')=='on';env=read(source_path(base,cfg['environment_lock']))
    ledger=read(source_path(base,cfg['parameter_ledger']))
    native=copy.deepcopy(cfg);native['profile']='native';native['mode']='off'
    native['asset_roots']={k:str(source_path(base,v)) for k,v in cfg['asset_roots'].items() if k!='ovis_model'}
    native['gpu_uuids']=cfg['gpu_uuids'][:1]
    native['validation_budget']={**cfg['validation_budget'],'gpu_allowlist':native['gpu_uuids']}
    native['environment_lock']=str(out/'native-env.json');write(native['environment_lock'],{'native':env['native']})
    native['parameter_ledger']=str(out/'native-ledger.json');write(native['parameter_ledger'],[x for x in ledger if x['component']!='ovis'])
    if 'gpu_backend' in native:
        native['gpu_backend']['host_binding']['path']=str(source_path(base,cfg['gpu_backend']['host_binding']['path']))
    write(out/'native-configuration.json',native)
    native_prepare(out/'native-configuration.json',out/'native-plan')
    plan=read(out/'native-plan/bind-plan.json')
    if 'gpu_backend' in plan:
        ref=plan['gpu_backend']['host_binding'];shutil.copyfile(out/'native-plan'/ref['path'],out/ref['path'])
    plan.update(validation_profile='v32-text',validation_mode=cfg['mode'],validation_budget=cfg['validation_budget'],
                gpu_uuids=cfg['gpu_uuids'],environments=env,parameter_ledger=ledger,v32=copy.deepcopy(cfg['v32']))
    if active:
        root=source_path(base,cfg['asset_roots']['ovis_model'])
        files={r['name']:r['sha256'] for r in read(Path(__file__).with_name('v32_model_lock.json'))['files']}
        verify_tree(root,files)
        plan['assets']['ovis_model']={'path':str(root),'files':files,'subset':False,'kind':'cluster_ro_bind'}
    write(out/'bind-plan.json',plan)
    return {'status':'plan_created_not_runtime_verified','plan_sha256':sha(out/'bind-plan.json'),'model_loads':0}

def bind(plan_path,output):
    from .asset_binder import bind as native_bind,source_path
    from .asset_binding import capacity
    from .contracts import runtime as validate_runtime
    plan=read(plan_path);base=Path(plan_path).resolve().parent
    # A new, explicit native intermediate; no original receipt or V31 artifact is changed.
    staging=new_dir(str(output)+'-native-plan');native=copy.deepcopy(plan)
    native.pop('v32');native['validation_profile']='native';native['validation_mode']='off'
    native['assets'].pop('ovis_model',None);native['environments'].pop('ovis',None)
    native['parameter_ledger']=[x for x in native['parameter_ledger'] if x['component']!='ovis']
    native['gpu_uuids']=native['gpu_uuids'][:1]
    native['validation_budget']={**native['validation_budget'],'gpu_allowlist':native['gpu_uuids']}
    for item in native['assets'].values():item['path']=str(source_path(base,item['path']))
    if 'gpu_backend' in native:
        ref=native['gpu_backend']['host_binding'];shutil.copyfile(closed(base,ref['path']),staging/ref['path'])
    write(staging/'plan.json',native);native_bind(staging/'plan.json',output)
    out=Path(output);r=read(out/'runtime.json')
    if 'ovis_model' in plan['assets']:
        item=plan['assets']['ovis_model'];root=source_path(base,item['path']);verify_tree(root,item['files'])
        if item.get('kind')!='cluster_ro_bind':raise ContractError('Explicit cluster Ovis asset required')
        r['assets']['ovis_model']={'role':'ovis_model','path':str(root),'files':item['files'],'kind':'cluster_ro_bind'}
    r.update(v32=plan['v32'],environments=plan['environments'],parameter_ledger=plan['parameter_ledger'],gpu_uuids=plan['gpu_uuids'])
    if 'ovis_model' in r['assets']:
        r['v32'].update(model_sha256=canonical(r['assets']['ovis_model']['files']),source_sha256=canonical(r['environments']['ovis']['image_files']))
    from .v32_cluster import asset_root
    total=capacity(r,{k:asset_root(r,out,k) for k in r['assets']});r['native']['capacity']={'passed':True,'conservative_upper_bound':total}
    (out/'runtime.json').rename(out/'NATIVE_INTERMEDIATE_RUNTIME.json');write(out/'runtime.json',r)
    validate_runtime(out/'runtime.json','v32-text',plan['validation_budget'],mode=plan['validation_mode'])
    (out/'ASSET_BINDING.json').rename(out/'NATIVE_INTERMEDIATE_BINDING.json')
    result={'passed':True,'runtime_sha256':sha(out/'runtime.json'),'parameter_status':r['v32']['parameter_status'],
            'unique_trainable_total':total,'runtime_verified':False,'model_loads':0,'installation':False}
    write(out/'ASSET_BINDING.json',result);return result
