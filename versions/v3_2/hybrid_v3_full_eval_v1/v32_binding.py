"""Explicit Ovis assets/environment and first-crop-only parameter bootstrap."""
import copy
from pathlib import Path
from .core import ContractError, canonical, closed, digest, read, sha
from .v32_protocol import CONFIG_SHA, manifest

def capacity(runtime,roots):
    from .asset_binding import capacity as base_capacity
    r=copy.deepcopy(runtime);v=r.pop('v32');rows=r['parameter_ledger']
    ovis=[x for x in rows if x['component']=='ovis'];r['parameter_ledger']=[x for x in rows if x['component']!='ovis']
    total=base_capacity(r,roots)
    active='ovis_model' in runtime['assets']
    if not active:
        if ovis or v.get('parameter_status') not in ('not_deployed',None):raise ContractError('Ovis count without deployed model')
        return total
    if v.get('parameter_status')=='bootstrap_header':
        if v.get('execution')!='crop' or ovis:raise ContractError('Bootstrap cannot authorize formal on or measured ledger')
        if v.get('header_stored_elements')!=852985920:raise ContractError('Ovis header estimate changed')
        count=852985920
    else:
        if v.get('parameter_status')!='measured' or len(ovis)!=1:raise ContractError('Measured Ovis evidence required')
        row=ovis[0];ref=row['evidence'];path=closed(roots[ref['asset']],ref['path'])
        if sha(path)!=digest(ref['sha256']):raise ContractError('Ovis measurement changed')
        evidence=read(path)
        for key in ('component','stored_elements','unique_trainable','buffers','model_files_sha256'):
            if row.get(key)!=evidence.get(key):raise ContractError('Ovis ledger/measurement mismatch')
        if (evidence.get('method')!='loaded_unique_parameters' or not evidence.get('structure_sha256')
            or not evidence.get('identity',{}).get('load_id') or not evidence.get('device',{}).get('uuid')):
            raise ContractError('Ovis same-load evidence missing')
        count=row['unique_trainable']
        if type(count) is not int or count<=0:raise ContractError('Ovis count invalid')
        if row['model_files_sha256']!=canonical(runtime['assets']['ovis_model']['files']):raise ContractError('Ovis count/model mismatch')
    if total+count>4_000_000_000:raise ContractError('V32 total parameter cap exceeded')
    return total+count

def validate(runtime,root,profile,budget,mode,verify=True):
    from .asset_binding import validate_v31
    v=runtime['v32'];active=mode=='on';crop=v.get('execution')=='crop'
    if profile!='v32-text' or v.get('components')!=['text'] or v.get('config_sha256')!=CONFIG_SHA:
        raise ContractError('V32 profile/config mismatch')
    if mode not in ('on','off','pass-through') or crop and not active:raise ContractError('V32 execution mode mismatch')
    if set(runtime['environments'])!=({'native','ovis'} if active else {'native'}):raise ContractError('V32 effective environment set differs')
    if len(runtime['gpu_uuids'])!=(1 if crop or not active else 2):raise ContractError('V32 effective GPU count differs')
    if type(runtime['idle_memory_mib']) is not int or runtime['idle_memory_mib']!=0:raise ContractError('V32 requires integer zero idle memory')
    if budget['expert_calls_per_page']>256 or budget['native_calls_per_page']>256:raise ContractError('V32 call budget exceeded')
    if budget['cpus']!=8 or budget['ram_gib']!=48 or budget['swap_gib']!=0:raise ContractError('V32 native envelope differs')
    if crop and (budget['pages']!=1 or budget['model_loads']!=1 or budget['expert_calls_per_page']!=1 or budget['total_seconds']>900):
        raise ContractError('First crop budget differs')
    # Reuse the exact native binding without representing Ovis as a Paddle component.
    r=copy.deepcopy(runtime);r.pop('v32');r['gpu_uuids']=r['gpu_uuids'][:1]
    r['environments'].pop('ovis',None);r['assets'].pop('ovis_model',None)
    r['parameter_ledger']=[x for x in r['parameter_ledger'] if x['component']!='ovis']
    validate_v31(r,root,'native',budget,'off',verify)
    if active:
        env=runtime['environments']['ovis'];cfg=manifest()
        from .v32_cluster import descriptor
        descriptor(env)
        from .image_identity import parse_reference
        parse_reference(env['image'])
        if env['image']==runtime['image']:raise ContractError('Ovis environment must be independent')
        for k in ('python','resolved_python','site_packages'):
            if not isinstance(env.get(k),str) or not env[k].startswith('/') or env[k].startswith(('/assets/','/output/','/framework/')):
                raise ContractError('Ovis environment path must be explicit')
        if env['resolved_python'] not in env['image_files']:raise ContractError('Ovis interpreter not locked')
        for n,h in env['image_files'].items():
            if not n.startswith('/') or '/../' in n:raise ContractError('Invalid Ovis image path')
            digest(h)
        if any(env['packages'].get(n)!=ver for n,ver in {'vllm':'0.22.1','torch':'2.11.0','transformers':'5.5.1'}.items()):raise ContractError('Ovis package versions changed')
        if not {'ovis_model','environment','parameter_evidence','auxiliary_models'}<=set(env['asset_roles']):raise ContractError('Ovis load boundary incomplete')
        expected={x['name']:x['sha256'] for x in read(Path(__file__).with_name('v32_model_lock.json'))['files']}
        if runtime['assets']['ovis_model']['files']!=expected:raise ContractError('Frozen Ovis model changed')
        if (v.get('model_sha256')!=canonical(expected) or v.get('source_sha256')!=canonical(env['image_files'])
            or v.get('model_revision')!=cfg['model_revision']):raise ContractError('Ovis source/model identity differs')
        if not crop and v.get('parameter_status')!='measured':raise ContractError('Formal on rejects bootstrap parameters')
    elif 'ovis_model' in runtime['assets']:raise ContractError('Off/pass must not deploy Ovis')
    from .v32_cluster import asset_root
    capacity(runtime,{k:asset_root(runtime,root,k) for k in runtime['assets']})
    return True
