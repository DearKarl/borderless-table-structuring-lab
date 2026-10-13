"""Host-only binding. Inference receives only sixteen inputs and small locks."""
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
from hybrid.v4_input_selector.scale_policy import verified_bundle
from hybrid.v4_input_selector.worker_contract import atomic_json,bound_file,file_sha,validate_contract
from .contracts import checked_json,verify_code,MODEL_SHA,ORIGINAL_MANIFEST_SHA,MARGINS
from .dependencies import IMAGE,need,require_receipt,inventory
from .host_state import State,LIMITS
from .prepare import RECIPE

GPU='GPU-PUBLIC-REQUIRES-BINDING-0'

def ref(path):return dict(path=str(Path(path).resolve()),sha256=file_sha(path))
def read(value):return checked_json(value['path'],value['sha256'])

def input_projection(roster,plan,plan_sha):
    need(roster.get('schema')=='v4_raster_roster_v1' and roster.get('plan_sha256')==plan_sha,'Roster/plan binding differs')
    need(roster.get('recipe')==RECIPE and len(roster['pages'])==8 and len(roster['items'])==16,'Exact raster recipe/8 pairs required')
    need([r['page_id'] for r in roster['pages']]==[r['page_id'] for r in plan['pages']],'Frozen page order differs')
    expected=[]
    for row,frozen in zip(roster['pages'],plan['pages']):
        need(row['source_group']==frozen['group_id'] and row['source_pdf_sha256']==frozen['pdf_sha256'] and row['source_member']==frozen['member'],'Frozen source provenance differs')
        need(row['file']=='inputs/'+row['page_id']+'.png' and row['source']['source_type']=='raster'
             and row['source']['original_file_sha256']==row['input_sha256'],'Raster source identity differs')
        for action in ('A','B'):
            expected.append(dict(item_id=row['page_id']+'_'+action,action=action,
                **{k:row[k] for k in ('page_id','file','input_sha256','source')}))
    need(roster['items']==expected and len({i['item_id'] for i in expected})==16,'Input-only item equality failed')
    return dict(schema=roster['schema'],plan_sha256=plan_sha,items=deepcopy(expected))

def check_mounts(m,sources):
    private=[Path(x).resolve() for x in m['private_paths']]
    for source in sources:
        source=Path(source).resolve()
        need(not any(p==source or p.is_relative_to(source) or source.is_relative_to(p) for p in private),'Private silver/roster/archive would enter inference mount')

def no_text(value):
    if isinstance(value,dict):
        need(not {'natural_text','transcript','reference_text','gold','gt'}&set(value),'Reference text in inference metadata')
        for v in value.values():no_text(v)
    elif isinstance(value,list):
        for v in value:no_text(v)

def validate(m):
    need(m['schema']=='v4_calibration_host_v1' and m['gpu_uuid']==GPU and m['limits']==LIMITS,'Calibration manifest scope differs')
    need(m['template']['runtime_binding']['external_image']['image']==IMAGE,'Native image differs')
    need(m['template']['runtime_binding']['device']['uuid']==GPU,'Native template GPU identity differs')
    cfg=m['provider'];code=Path(cfg['code_root']).resolve()
    need(str(code)==cfg['code_root'] and Path(__file__).resolve()==code/'hybrid/v4_selected_eval/host_manifest.py','Execute exact staged code')
    code_files=inventory(code)
    need(code_files==cfg['code_files'] and all(k.endswith('.py') for k in code_files),'Dedicated code mount contains unbound files')
    plan=read(m['plan']);verify_code(code,plan['code_files']);no_text(plan)
    need(plan['known_overlap_reviewed'] is True and plan['model_sha256']==MODEL_SHA and plan['margins']==list(MARGINS),'Plan not accepted/frozen')
    roster=read(m['roster']);projection=read(m['projection'])
    need(projection==input_projection(roster,plan,m['plan']['sha256']),'Projection differs from full host roster')
    probe_contract=deepcopy(m['template']);probe_contract.update(job_id='raster-cal-binding',wall_seconds=14400,items=projection['items'])
    probe_contract['runtime_binding']['external_image']['receipt']=dict(path=str(Path(m['root'])/'binding-image-placeholder.json'),sha256='0'*64)
    validate_contract(probe_contract)
    need(read(m['rounding'])['schema']=='v4_native_rounding_evidence_v1' and m['rounding']['sha256']==plan['rounding_evidence_sha256'],'Rounding lock differs')
    need(m['template']==read(m['template_ref']),'Native template changed')
    verified_bundle(m['gbdt_root'],expected_manifest_sha256=ORIGINAL_MANIFEST_SHA)
    need(inventory(m['gbdt_root'])==m['gbdt_files'] and len(m['gbdt_files'])==3,'Small GBDT directory inventory differs')
    need(inventory(m['overlay_root'])==read(m['overlay_manifest'])['files'],'Dependency overlay bytes differ')
    dependency=require_receipt(read(m['dependency_receipt']),overlay_sha=m['overlay_manifest']['sha256'],fixture_sha=m['fixture_sha256'])
    need(set(dependency.get('native_origins',{}))=={'numpy','torch','pypdfium2'},'Native package origin evidence missing')
    for name,row in dependency['native_origins'].items():
        need(row['path'].startswith('/opt/v31-native/lib/python3.12/site-packages/'+name+'/')
            and len(row['sha256'])==64,'Native origin receipt differs')
    for name,row in dependency['modules'].items():
        prefix='/opt/v31-native/lib/python3.12/site-packages/numpy/' if name=='numpy' else '/v4-deps/'
        need(row.get('path','').startswith(prefix),'Dependency module import path differs')
    inputs={i['file']:i['input_sha256'] for i in projection['items']}
    need(len(inputs)==8 and inventory(m['input_root'])==inputs,'Input mount must contain only the eight original rasters')
    need(m['runtime']['sha256']==file_sha(m['runtime']['path']),'Runtime binding changed')
    runtime=read(m['runtime']);need(runtime==runtime_config(m),'Runtime/environment arguments differ')
    need(set(cfg['roots'])=={'vendor','model','aux'} and set(cfg['resources'])=={'cpus','ram_gib','swap_gib','shm_gib'},'Native resources/roots differ')
    need(all(Path(p).is_absolute() for p in cfg['roots'].values()),'Native roots must be absolute')
    need(all(type(v) in (int,float) and math.isfinite(v) and v>=0 and (k=='swap_gib' or v>0) for k,v in cfg['resources'].items()),'Invalid resource values')
    need(set(cfg['legacy_files'])=={'__init__.py','core.py','lifecycle.py','gpu_backend.py','gpu_admission.py','image_identity.py','bounded_cleanup.py'},'Closed V31 reuse files required')
    for name,h in cfg['legacy_files'].items():need(file_sha(bound_file(cfg['legacy_root'],name))==h,'Legacy source differs')
    read(cfg['driver_binding'])
    need({plan['archive_sha256'],plan['parquet_sha256']}<={r['sha256'] for r in m['private_refs']},'Raw dataset private bindings missing')
    for r in m['private_refs']:need(file_sha(r['path'])==r['sha256'],'Host-only input changed')
    need(file_sha(m['cpu_python']['path'])==m['cpu_python']['sha256'],'Bound CPU interpreter changed')
    need(Path(m['root']).is_absolute() and Path(m['lease_directory']).is_absolute(),'Absolute host paths required')
    roots=[Path(p).resolve() for p in (cfg['code_root'],m['input_root'],m['gbdt_root'],m['overlay_root'],*cfg['roots'].values())]
    need(all(p.is_dir() for p in roots),'Inference root missing')
    need(not any(a==b or a.is_relative_to(b) or b.is_relative_to(a) for i,a in enumerate(roots) for b in roots[i+1:]),'Inference directory mounts overlap')
    check_mounts(m,[code,m['input_root'],m['gbdt_root'],m['overlay_root'],*cfg['roots'].values(),
        m['plan']['path'],m['projection']['path'],m['rounding']['path'],m['overlay_manifest']['path'],
        m['dependency_receipt']['path'],m['runtime']['path']])
    return m

def runtime_config(m):
    args=dict(mode='calibration',projection=m['projection']['path'],model=m['gbdt_root'],overlay='/v4-deps')
    args.update({'projection-sha':m['projection']['sha256'],'code-root':m['provider']['code_root'],
        'plan':m['plan']['path'],'plan-sha':m['plan']['sha256'],
        'rounding-evidence':m['rounding']['path'],'rounding-evidence-sha':m['rounding']['sha256'],
        'overlay-manifest':m['overlay_manifest']['path'],'overlay-sha':m['overlay_manifest']['sha256'],
        'dependency-receipt':m['dependency_receipt']['path'],'dependency-receipt-sha':m['dependency_receipt']['sha256'],
        'fixture-sha':m['fixture_sha256']})
    return dict(schema='v4_selector_runtime_v1',activated=True,arguments=args)

def load(path,sha):return validate(checked_json(path,sha))

def bind(config,config_sha):
    c=checked_json(config,config_sha)
    need(c['schema']=='v4_calibration_bind_request_v1','Unsupported bind request')
    # Refuse absent/failed CPU receipt before creating the run or reserving a GPU.
    require_receipt(read(c['dependency_receipt']),overlay_sha=c['overlay_manifest']['sha256'],fixture_sha=c['fixture_sha256'])
    root=Path(c['root']).resolve();root.mkdir(parents=False,exist_ok=False)
    control=root/'metadata';control.mkdir();input_root=root/'inference-inputs';input_root.mkdir()
    plan=read(c['plan']);roster=read(c['roster']);projection=input_projection(roster,plan,c['plan']['sha256'])
    m=dict(schema='v4_calibration_host_v1',task_id=c['task_id'],root=str(root),gpu_uuid=GPU,limits=LIMITS,
        lease_directory=c['lease_directory'],provider=c['provider'],template=read(c['template_ref']),
        template_ref=c['template_ref'],roster=c['roster'],data_root=c['data_root'],gbdt_root=c['gbdt_root'],
        gbdt_files=inventory(c['gbdt_root']),overlay_root=c['overlay_root'],fixture_sha256=c['fixture_sha256'],
        input_root=str(input_root),private_refs=c['private_refs'],
        private_paths=[str(Path(c['data_root']).resolve()),c['roster']['path'],str(root/'MANIFEST.json'),
            *[r['path'] for r in c['private_refs']]],cpu_python=c['cpu_python'])
    for key in ('plan','rounding','overlay_manifest','dependency_receipt'):
        read(c[key]);path=control/(key+'.json');shutil.copyfile(c[key]['path'],path);m[key]=ref(path)
    for name,h in {i['file']:i['input_sha256'] for i in projection['items']}.items():
        source=bound_file(c['data_root'],name);need(file_sha(source)==h,'Original raster hash differs')
        dest=bound_file(input_root,name);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    atomic_json(control/'INPUT_PROJECTION.json',projection);m['projection']=ref(control/'INPUT_PROJECTION.json')
    runtime=runtime_config(m);atomic_json(control/'SELECTOR_RUNTIME.json',runtime);m['runtime']=ref(control/'SELECTOR_RUNTIME.json')
    validate(m)
    manifest=root/'MANIFEST.json';manifest_sha=atomic_json(manifest,m)
    State(root,manifest_sha).create(projection['items'],c.get('cpu_seconds_already_spent',0))
    return dict(manifest=str(manifest),manifest_sha256=manifest_sha,status='bound_not_submitted',runtime_sha256=m['runtime']['sha256'])
