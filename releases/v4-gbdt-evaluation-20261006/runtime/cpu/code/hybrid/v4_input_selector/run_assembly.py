"""Pure preparation assembler. Emits drafts/gaps only; no master or launcher."""
import argparse
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath

from .bank import PLAN_SHA, PUBLIC_PLAN_SHA, PUBLIC_SPLIT, PUBLIC_STARTS
from .schema import SourceRecord
from .worker_contract import SOURCE_KEYS, atomic_json, digest, safe_id, hash_value

BINDINGS={'asset_lock','old_worker_template','runtime_binding','versions','provider_draft','provider_code',
          'feature_contract','metric_binding','budgets'}
PLACEMENTS={'bank_id','bank_root','input_root','gpu_uuid','weight_receipt_path'}
BUDGET=dict(pages=60,families=20,family_split=dict(fit=12,calibration=4,held_out=4),
            held_pages_max=12,smoke_pages=2,smoke_starts_max=6,bank_starts_max=180,
            wall_seconds=43200,gpu_seconds=43200,page_seconds=600,work_seconds=540,
            inner_seconds=25,outer_seconds=30,persist_seconds=5)

def require(value,message):
    if not value: raise ValueError(message)

def load_binding(binding):
    require(isinstance(binding,dict) and set(binding)=={'path','sha256'},'Exact assembly input path/hash required')
    path=Path(binding['path'])
    require(path.is_absolute() and path.is_file() and not path.is_symlink() and path.stat().st_size<=16*1024*1024,
            'Assembly input unavailable/excessive/link')
    raw=path.read_bytes()
    require(digest(raw)==binding['sha256'],'Assembly input changed: '+str(path))
    return json.loads(raw)

def assemble(request):
    public=request.get('schema')=='nju_public_assembly_request_v1'
    count=48 if public else 60
    ceiling=PUBLIC_STARTS if public else 180
    budget=deepcopy(BUDGET)
    if public:
        budget.update(pages=48,families_min=24,max_pages_per_family=2,page_split=PUBLIC_SPLIT,
                      held_pages_max=8,bank_starts_max=144,smoke_wall_seconds=3600,smoke_gpu_seconds=3600,
                      cpu_analysis_seconds=1800,seed=20261003)
        del budget['families'];del budget['family_split']
    require(set(request)=={'schema','bindings','placements','roster','owner_reported_eligible_pages'}
            and request['schema'] in ('v4_native_assembly_request_v1','nju_public_assembly_request_v1'),'Unknown assembly request')
    require(set(request['bindings'])==BINDINGS and set(request['placements'])==PLACEMENTS,'Assembly field set differs')
    require(type(request['owner_reported_eligible_pages']) is int and 0<=request['owner_reported_eligible_pages']<=count,
            'Invalid owner-reported eligibility count')
    data={name:load_binding(binding) for name,binding in request['bindings'].items()}
    b=data['budgets']
    require(b['page_starts']['S1_max']==6 and b['page_starts']['S2_cumulative_max']==ceiling
            and b['resources']['page_timeout_seconds']==600 and b['resources']['S1_S2_wall_hours_max']==12
            and b['resources']['S1_S2_gpu_hours_max']==12 and b['resources']['automatic_retries'] is False,
            'Frozen budget differs')
    if public:
        require(b['resources'].get('S1_wall_hours_max')==1 and b['resources'].get('S1_gpu_hours_max')==1
                and b['resources'].get('cpu_analysis_seconds')==1800 and b.get('split_pages')==PUBLIC_SPLIT,
                'Public smoke/CPU/split contract differs')
        require(data['metric_binding'].get('score_version')=='PublicTranscriptEdit_v1', 'Wrong public scorer')
    assets=data['asset_lock']
    vendor=deepcopy(assets['model_files']['tele_source'])
    old=data['old_worker_template']['vendor_files']
    require(set(vendor)<=set(old) and all(old[n]==h for n,h in vendor.items()),'Deployed vendor/source freeze differs')
    delta=sorted(set(old)-set(vendor))
    require(delta==['TeleOCR-vllm/TeleOCR_vllm/__init__.py','TeleOCR-vllm/TeleOCR_vllm/qwen2_5_vl.py','infer.py'],
            'Unreviewed old-template/deployment difference')
    versions=deepcopy(data['versions'])
    require('pillow' in versions and 'Pillow' not in versions,'Ambiguous package alias')
    versions['Pillow']=versions.pop('pillow')
    profile=deepcopy(data['runtime_binding'])
    profile['device']['uuid']=request['placements']['gpu_uuid']
    profile['external_image']['receipt']=None  # Live allocation-only placeholder.
    profile['weight_receipt']['path']=request['placements']['weight_receipt_path']
    native=assets['native_runtime']
    template=dict(schema='v4_tele_worker_v2',vendor_files=vendor,model_files=deepcopy(assets['model_files']['tele_model']),
        aux_files=deepcopy(assets['image_auxiliaries']['native']),runtime_versions=versions,
        expected=dict(model_class=native['TELE_LOAD.json']['model_class'],processor_class=native['PROCESSOR.json']['class'],
            dtype=native['TELE_LOAD.json']['dtype'],model_max_length=128000,processor_merge_size=2),runtime_binding=profile)
    provider=deepcopy(data['provider_draft'])
    for name in ('code_root','code_files','legacy_root','legacy_files'):
        provider[name]=deepcopy(data['provider_code'][name])
    gaps=[]
    def gap(field,reason):
        gaps.append(dict(field=field,reason=reason,owner='Adjutant/root'))
    for name,value in request['placements'].items():
        if value is None: gap('placements.'+name,'Root must bind the exact target; no guessed path/device')
        elif name=='bank_id': require(safe_id(value),'Malformed bank ID')
        elif name!='gpu_uuid': require(PurePosixPath(value).is_absolute(),'Absolute native target path required')
    for name,value in provider['resources'].items():
        if value is None: gap('provider.resources.'+name,'Native host quantity not frozen by current PLAN/BUDGET')
    if request['roster'] is None:
        rows=[dict(page_id=None,source_family=None,split=None,
            input=dict(path=None,sha256=None,source={k:None for k in sorted(SOURCE_KEYS)}),
            reference=(dict(path=None,sha256=None,score_version='PublicTranscriptEdit_v1') if public else
                       dict(path=None,sha256=None,states=dict(text=None,table=None,formula=None)))) for _ in range(count)]
    else:
        roster=load_binding(request['roster'])
        require(set(roster)=={'schema','pages'} and roster['schema'] in (('nju_public_roster_v1',) if public else ('v4_roster_draft_v1','v4_frozen_roster_v1')),
                'Unrecognized roster input')
        rows=deepcopy(roster['pages'])
        require(len(rows)==count,'Preparation must not silently change the frozen page target')
    availability=[]
    families={}
    complete=True
    for n,row in enumerate(rows):
        prefix='roster.pages['+str(n)+']'
        for field in ('page_id','source_family','split'):
            if row.get(field) is None:
                complete=False
                gap(prefix+'.'+field,'Unfrozen original identity/source-separated allocation')
            elif field!='split': require(safe_id(row[field]),'Malformed roster identity')
        split=row.get('split');family=row.get('source_family')
        if split is not None:
            require(split in ('fit','calibration','held_out'),'Unknown split')
        if family is not None and split is not None:
            require(families.setdefault(family,split)==split,'Source family crosses splits')
        for section in ('input','reference'):
            for name in ('path','sha256'):
                if row[section].get(name) is None:
                    gap(prefix+'.'+section+'.'+name,'Root-owned source/reference binding missing')
                elif name=='sha256': require(hash_value(row[section][name]),'Malformed roster hash')
        source=row['input']['source']
        require(isinstance(source,dict) and set(source)==SOURCE_KEYS,'Unexpected source field set')
        for name in sorted(SOURCE_KEYS):
            optional_raster=name in ('pdf_cropbox','pdf_rotation') and source.get('source_type')=='raster'
            if source[name] is None and not optional_raster:
                gap(prefix+'.input.source.'+name,'Original source fact not frozen')
        try:
            record=SourceRecord(**source)
            actions=list(record.available_actions())
        except (ValueError,TypeError):
            actions=None
            gap(prefix+'.input.source','Incomplete/invalid original source geometry/type; no guessed action availability')
        if actions is not None and row['input']['sha256'] is not None:
            require(source['original_file_sha256']==row['input']['sha256'],'Original input/source hash mismatch')
        if public:
            require(row['reference'].get('score_version')=='PublicTranscriptEdit_v1','Public reference score version differs')
            if actions is not None:
                require(record.source_type=='original_pdf' and record.original_page_ordinal==0,'Supplied public single-page PDF index must be zero')
        else:
            for kind in ('text','table','formula'):
                if row['reference']['states'].get(kind) not in ('present','verified_absent'):
                    gap(prefix+'.reference.states.'+kind,'Unlabeled/unreviewed category cannot enter primary target')
        availability.append(dict(position=n,opaque_page_if_frozen='p'+str(n).zfill(3),actions=actions))
    if complete:
        require(len({row['page_id'] for row in rows})==count,'Duplicate page ID')
        if public:
            require({s:sum(r['split']==s for r in rows) for s in PUBLIC_SPLIT}==PUBLIC_SPLIT
                    and len(families)>=24 and max(sum(r['source_family']==g for r in rows) for g in families)<=2,
                    'Public source-group/page partition differs')
        else:
            require({s:list(families.values()).count(s) for s in BUDGET['family_split']}==BUDGET['family_split'],
                'Current bank requires exactly 20 source families with12/4/4 allocation')
            require(sum(row['split']=='held_out' for row in rows)<=12,'Held page cap exceeded')
    else:
        gap('roster.source_split','Frozen source-group/page allocation incomplete')
    known_actions=all(row['actions'] is not None for row in availability)
    smoke_positions=[n for n,row in enumerate(rows) if row.get('split')=='fit'][:2] if complete else None
    report=dict(budget=budget,actual_available_slots=sum(len(r['actions']) for r in availability) if known_actions else None,
                smoke_positions=smoke_positions,
                smoke_available_starts=sum(len(availability[n]['actions']) for n in smoke_positions)
                    if smoke_positions is not None and all(availability[n]['actions'] is not None for n in smoke_positions) else None,
                availability=availability,unused_slots_not_transferred=True,owner_reported_eligible_pages=request['owner_reported_eligible_pages'])
    require(report['actual_available_slots'] is None or report['actual_available_slots']<=ceiling,'Bank ceiling exceeded')
    require(report['smoke_available_starts'] is None or report['smoke_available_starts']<=6,'Smoke ceiling exceeded')
    gap('roster_acceptance',('Root public-source integrity, silver-reference and source/split protocol acceptance absent; not generated' if public else 'Root independent reference/provenance/source/split acceptance absent; not generated'))
    for name in ('runtime_native','parameter_inventory','actions_metrics','feature_boundary','launch_adapter'):
        gap('readiness.evidence.'+name,'Final root acceptance binding is not generated by this preparation tool')
    gap('readiness','No accepted readiness or master; drafts cannot launch')
    for name in ('roster','worker_template','adapter_sources'):
        gap('host.'+name,'Final staged host path/hash binding absent; draft is not an accepted master')
    gap('metric_evaluation_dispatch','Consumer API is provided; root must bind final result/reference staging and bounded evaluator dispatch')
    host=dict(schema='nju_public_host_configuration_draft_v1' if public else 'v4_host_configuration_draft_v1',plan_sha256=PUBLIC_PLAN_SHA if public else PLAN_SHA,
        bank_id=request['placements']['bank_id'],roster=None,worker_template=None,adapter_sources=None,
        bank_root=request['placements']['bank_root'],input_root=request['placements']['input_root'],
        lease_directory='/absolute/required-shared-lease-directory',
        gpu_uuid=request['placements']['gpu_uuid'],roster_acceptance=None,readiness=None)
    return {'WORKER_TEMPLATE_DRAFT.json':dict(schema='v4_worker_template_draft_v1',template=template),
        'PROVIDER_CONFIG_DRAFT.json':dict(schema='v4_provider_configuration_draft_v1',configuration=provider),
        'HOST_CONFIG_DRAFT.json':host,'ROSTER_FIELDS_DRAFT.json':dict(schema='nju_public_roster_v1' if public else 'v4_roster_draft_v1',pages=rows),
        'SLOT_REPORT_DRAFT.json':report,'GAPS.json':dict(launch_ready=False,accepted_master_generated=False,fields=gaps),
        'ASSEMBLY_AUDIT.json':dict(schema='v4_assembly_audit_v1',input_bindings=request['bindings'],
            vendor_count=len(vendor),old_template_vendor_count=len(old),excluded_unstaged_template_entries=delta,
            model_count=len(template['model_files']),aux_count=len(template['aux_files']),weight_bytes_read=False,
            package_key_alias=dict(pillow='Pillow'),recipes_changed=False,source_split_invented=False,
            metric_binding=data['metric_binding'],feature_contract=data['feature_contract'])}

def main():
    parser=argparse.ArgumentParser(description='Generate drafts only; never launch or accept a master')
    parser.add_argument('--request',required=True)
    parser.add_argument('--request-sha256',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    request=load_binding(dict(path=args.request,sha256=args.request_sha256))
    result=assemble(request)
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    for name,value in result.items(): atomic_json(output/name,value)
    print(json.dumps(dict(status='draft_only',output=str(output),gap_fields=len(result['GAPS.json']['fields']),launch_ready=False)))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
