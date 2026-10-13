"""GT-free full/batch input binding for exactly one experimental1651 run."""
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import shutil
from hybrid.v4_input_selector.worker_contract import atomic_json,bound_file,file_sha,validate_contract
from .contracts import checked_json,source_files,verify_code,selected_items,prepare_frozen_model
from .dependencies import IMAGE,need,require_receipt,inventory
from .host_manifest import GPU,check_mounts,read,ref,no_text
from .experimental import LIMITS,MODEL_FILES,CALIBRATION_SHA,digest,validate_auth,validate_batch,prepare_experimental

def rows_from(raw):
    if isinstance(raw,list):rows=raw
    else:
        choices=[v for v in raw.values() if isinstance(v,list) and len(v)==1651 and all(isinstance(x,dict) and 'original_page_id' in x for x in v)]
        need(len(choices)==1,'Exactly one full1651 original-ID list required');rows=choices[0]
    need(len(rows)==1651,'Exactly1651 input rows required');return rows

def one(row,keys):
    values=[row[k] for k in keys if k in row]
    need(values and all(v==values[0] for v in values),'Missing/ambiguous input field');return values[0]

def full_inputs(raw,input_root):
    from PIL import Image,ImageOps
    pages=[]
    for row in rows_from(raw):
        name=one(row,('file','path','input_path','image_path','relative_path'));path=Path(name)
        if path.is_absolute():
            need(path.resolve().is_relative_to(Path(input_root).resolve()),'Input outside root')
            name=path.resolve().relative_to(Path(input_root).resolve()).as_posix()
        elif name.startswith('data/'):name=name[5:]
        path=bound_file(input_root,name);h=one(row,('sha256','input_sha256','file_sha256'))
        need(file_sha(path)==h,'Original raster SHA differs')
        pid=path.stem;original=row['original_page_id']
        need(re.fullmatch('p[0-9]{5}',pid) is not None and isinstance(original,str)
            and original not in ('','.','..') and '/' not in original and '\\' not in original,'Unsafe original/input identity')
        if Path(original).suffix.lower() in ('.png','.jpg','.jpeg','.webp','.tif','.tiff','.bmp'):original=Path(original).stem
        with Image.open(path) as image:
            upright=ImageOps.exif_transpose(image)
            try:size=list(upright.size)
            finally:upright.close()
        source=dict(original_file_sha256=h,source_type='raster',original_page_ordinal=0,oriented_size=size,
            pdf_cropbox=None,pdf_rotation=None,pdf_content_type='unknown')
        pages.append(dict(page_id=pid,original_page_id=original,file=name,input_sha256=h,source=source))
    pages.sort(key=lambda r:r['page_id'])
    need(len({p['page_id'] for p in pages})==len({p['original_page_id'] for p in pages})==1651,'Duplicate full page identity')
    projection=selected_items([{k:v for k,v in p.items() if k!='original_page_id'} for p in pages])
    return pages,projection

def runtime(m,projection_ref):
    args=dict(mode='experimental',projection=projection_ref['path'],model=m['gbdt_root'],overlay='/v4-deps')
    args.update({'projection-sha':projection_ref['sha256'],'code-root':m['provider']['code_root'],
        'experiment-auth':m['experiment_auth']['path'],'experiment-auth-sha':m['experiment_auth']['sha256'],
        'rounding-evidence':m['rounding']['path'],'rounding-evidence-sha':m['rounding']['sha256'],
        'overlay-manifest':m['overlay_manifest']['path'],'overlay-sha':m['overlay_manifest']['sha256'],
        'dependency-receipt':m['dependency_receipt']['path'],'dependency-receipt-sha':m['dependency_receipt']['sha256'],
        'fixture-sha':m['fixture_sha256']})
    return dict(schema='v4_selector_runtime_v1',activated=True,arguments=args)

def validate(m,*,check_inputs=True):
    need(m['schema']=='v4_experimental_host_v1' and m['limits']==LIMITS and m['gpu_uuid']==GPU,'Experimental host scope differs')
    auth=validate_auth(read(m['experiment_auth']));code=Path(m['provider']['code_root'])
    need(Path(__file__).resolve()==code/'hybrid/v4_selected_eval/experimental_manifest.py','Execute bound GPU code staging')
    verify_code(code,auth['code_files']);need(inventory(code)==m['provider']['code_files']
        and all(k.endswith('.py') for k in m['provider']['code_files']),'Unbound code mount entry')
    no_text(auth);no_text(m['full_input'])
    _,effective,_=prepare_experimental(m['gbdt_root'],m['experiment_auth']['path'],m['experiment_auth']['sha256'],code_root=code)
    need(m['effective_policy']==effective,'Effective experimental policy differs')
    need(m['run_id']==auth['run_id'] and len(m['items'])==len(m['pages'])==1651,'Host/auth scope differs')
    need([digest(i) for i in m['items']]==auth['item_hashes'] and digest(m['full_input'])==auth['full_input_sha256']
        and m['full_input']['items']==m['items'],'Full input identity differs')
    need(m['items']==selected_items([{k:v for k,v in p.items() if k!='original_page_id'} for p in m['pages']])['items'],'Official page/input mapping differs')
    need(len({p['original_page_id'] for p in m['pages']})==1651,'Original page IDs not unique')
    need(inventory(m['gbdt_root'])==MODEL_FILES,'Original model three files changed');prepare_frozen_model(m['gbdt_root'])
    need(read(m['calibration_report'])['accepted'] is False and m['calibration_report']['sha256']==CALIBRATION_SHA,'Original failed calibration report required')
    need(m['rounding']['sha256']==auth['rounding_evidence_sha256'],'Rounding authorization differs');read(m['rounding'])
    need(inventory(m['overlay_root'])==read(m['overlay_manifest'])['files'],'Verified dependency bytes changed')
    dependency=require_receipt(read(m['dependency_receipt']),overlay_sha=m['overlay_manifest']['sha256'],fixture_sha=m['fixture_sha256'])
    need(set(dependency.get('native_origins',{}))=={'numpy','torch','pypdfium2'},'Native origin evidence missing')
    for name,row in dependency['native_origins'].items():
        need(row['path'].startswith('/opt/v31-native/lib/python3.12/site-packages/'+name+'/') and len(row['sha256'])==64,'Native origin differs')
    for name,row in dependency['modules'].items():
        prefix='/opt/v31-native/lib/python3.12/site-packages/numpy/' if name=='numpy' else '/v4-deps/'
        need(row.get('path','').startswith(prefix),'Dependency origin differs')
    need(m['template']==read(m['template_ref']) and m['template']['runtime_binding']['device']['uuid']==GPU
        and m['template']['runtime_binding']['external_image']['image']==IMAGE,'Accepted native GPU0/image template required')
    for r in m['static_refs']:read(r)
    need(m['evaluation_binding']['sha256']==auth['evaluation_binding_sha256'],'Scoring protocol binding differs')
    # The binding JSON is metadata only; GT bytes are not opened during inference.
    need(read(m['evaluation_binding'])==m['evaluation'],'Evaluation binding metadata differs')
    if check_inputs:need(inventory(m['input_root'])=={i['file']:i['input_sha256'] for i in m['items']},'Input-only mount inventory/hash differs')
    cfg=m['provider']
    need(set(cfg['roots'])=={'vendor','model','aux'} and set(cfg['resources'])=={'cpus','ram_gib','swap_gib','shm_gib'},'Native roots/resources differ')
    need(all(type(v) in (int,float) and math.isfinite(v) and v>=0 and (k=='swap_gib' or v>0) for k,v in cfg['resources'].items()),'Invalid resources')
    need(set(cfg['legacy_files'])=={'__init__.py','core.py','lifecycle.py','gpu_backend.py','gpu_admission.py','image_identity.py','bounded_cleanup.py'},'Legacy closure differs')
    for name,h in cfg['legacy_files'].items():need(file_sha(bound_file(cfg['legacy_root'],name))==h,'Legacy bytes differ')
    roots=[Path(p).resolve() for p in (code,m['input_root'],m['gbdt_root'],m['overlay_root'],*cfg['roots'].values())]
    need(all(p.is_dir() for p in roots) and all(Path(p).is_absolute() for p in (m['root'],m['lease_directory'],*cfg['roots'].values())),'Absolute existing roots required')
    need(not any(a==b or a.is_relative_to(b) or b.is_relative_to(a) for i,a in enumerate(roots) for b in roots[i+1:]),'Inference mounts overlap')
    check_mounts(m,[code,m['input_root'],m['gbdt_root'],m['overlay_root'],*m['provider']['roots'].values(),
        m['experiment_auth']['path'],m['rounding']['path'],m['overlay_manifest']['path'],m['dependency_receipt']['path']])
    return m

def load(path,sha,*,check_inputs=True):return validate(checked_json(path,sha),check_inputs=check_inputs)

def batch(m,indices,allocation,seconds):
    auth=read(m['experiment_auth']);items=[m['items'][i] for i in indices]
    projection=dict(schema='v4_experimental_batch_v1',experiment_auth_sha256=m['experiment_auth']['sha256'],
        run_id=m['run_id'],full_input_sha256=auth['full_input_sha256'],indices=indices,items=items)
    validate_batch(projection,auth,m['experiment_auth']['sha256'])
    run=Path(m['root'])/'runs'/allocation;(run/'control').mkdir(parents=True,exist_ok=False);(run/'output').mkdir()
    atomic_json(run/'control/INPUT_PROJECTION.json',projection);projection_ref=ref(run/'control/INPUT_PROJECTION.json')
    atomic_json(run/'control/SELECTOR_RUNTIME.json',runtime(m,projection_ref))
    contract=deepcopy(m['template']);contract.update(job_id='exp-'+allocation,wall_seconds=seconds,
        items=[dict(i,action='A') for i in items])
    contract['runtime_binding']['external_image']['receipt']=dict(path=str(run/'control/image.json'),sha256='0'*64)
    validate_contract(contract);atomic_json(run/'control/contract.json',contract)
    return dict(m,projection=projection_ref,runtime=ref(run/'control/SELECTOR_RUNTIME.json'),batch_contract=ref(run/'control/contract.json'))

def read_batch(m,allocation,indices,expected):
    run=Path(m['root'])/'runs'/allocation;projection_ref=ref(run/'control/INPUT_PROJECTION.json')
    projection=read(projection_ref);validate_batch(projection,read(m['experiment_auth']),m['experiment_auth']['sha256'])
    need(projection['indices']==indices and projection['items']==[m['items'][i] for i in indices],'Allocation batch differs')
    runtime_ref=ref(run/'control/SELECTOR_RUNTIME.json');need(read(runtime_ref)==runtime(m,projection_ref),'Batch runtime differs')
    need(expected==dict(projection=projection_ref,runtime=runtime_ref,contract=ref(run/'control/contract.json')),'Batch bytes changed after reservation')
    return dict(m,projection=projection_ref,runtime=runtime_ref)

def bind(request,sha):
    c=checked_json(request,sha);need(c['schema']=='v4_experimental_bind_request_v1','Experimental bind request required')
    need(c.get('user_authorized_full1651_despite_failed_calibration') is True,'Explicit experimental authorization required')
    native=read(c['native_binding']);report=read(c['calibration_report'])
    need(c['calibration_report']['sha256']==CALIBRATION_SHA and report['accepted'] is False,'Actual failed report bytes required')
    require_receipt(read(native['dependency_receipt']),overlay_sha=native['overlay_manifest']['sha256'],fixture_sha=native['fixture_sha256'])
    root=Path(c['root']).resolve();root.mkdir(parents=False,exist_ok=False);metadata=root/'metadata';metadata.mkdir()
    code=Path(c['code_root']).resolve();provider=deepcopy(native['provider']);provider['code_root']=str(code);provider['code_files']=inventory(code)
    pages,projection=full_inputs(read(c['source_roster']),c['source_input_root'])
    inputs=root/'inference-inputs';inputs.mkdir()
    for item in projection['items']:
        dest=bound_file(inputs,item['file']);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(bound_file(c['source_input_root'],item['file']),dest)
    auth=dict(schema='v4_experiment_auth_v1',run_id=c['run_id'],experimental_only=True,adaptive_eligible=False,
        calibration_accepted=False,training_allowed=False,user_authorized_full1651_despite_failed_calibration=True,
        authorization_source=c['authorization_source'],margin=0.0,fixed_action='B',model_files=MODEL_FILES,limits=LIMITS,
        calibration_report_sha256=c['calibration_report']['sha256'],source_roster_sha256=c['source_roster']['sha256'],
        full_input_sha256=digest(projection),item_hashes=[digest(i) for i in projection['items']],
        code_files={p.relative_to(code).as_posix():file_sha(p) for p in source_files(code)},
        rounding_evidence_sha256=native['rounding']['sha256'],evaluation_binding_sha256=c['evaluation_binding']['sha256'])
    validate_auth(auth);atomic_json(metadata/'EXPERIMENT_AUTH.json',auth)
    evaluation=read(c['evaluation_binding'])
    m=dict(schema='v4_experimental_host_v1',run_id=c['run_id'],task_id=c['run_id'],root=str(root),gpu_uuid=GPU,
        limits=LIMITS,provider=provider,template=native['template'],template_ref=native['template_ref'],
        lease_directory=native['lease_directory'],gbdt_root=c['gbdt_root'],input_root=str(inputs),
        overlay_root=native['overlay_root'],fixture_sha256=native['fixture_sha256'],
        experiment_auth=ref(metadata/'EXPERIMENT_AUTH.json'),calibration_report=c['calibration_report'],
        evaluation_binding=c['evaluation_binding'],evaluation=evaluation,pages=pages,items=projection['items'],full_input=projection,
        static_refs=[c['native_binding'],c['source_roster'],provider['driver_binding'],native['template']['runtime_binding']['weight_receipt']],
        private_paths=[str(root/'MANIFEST.json'),str(root/'official'),evaluation['gt']['path'],evaluation['source']['path'],
            c['source_roster']['path'],c['calibration_report']['path'],*native['private_paths']])
    for key in ('rounding','overlay_manifest','dependency_receipt'):
        target=metadata/(key+'.json');shutil.copyfile(native[key]['path'],target);m[key]=ref(target)
    # Standard-library validation/caching only, no estimator import on the host.
    _,binding,_=prepare_experimental(m['gbdt_root'],m['experiment_auth']['path'],m['experiment_auth']['sha256'],code_root=code)
    m['effective_policy']=binding
    validate(m);manifest=root/'MANIFEST.json';mh=atomic_json(manifest,m)
    from .experimental_state import State
    State(root,mh).create(m['items'])
    return dict(manifest=str(manifest),manifest_sha256=mh,experiment_auth=m['experiment_auth'],status='experimental_bound_not_submitted')
