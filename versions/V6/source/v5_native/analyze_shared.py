"""Audit shared inference activation and observed timing without ground truth."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

from .io_utils import atomic_json,digest,read_json,utc

VERSIONS={'GBDT_V6.1.1':('gbdt',('table',)),'GBDT_V6.1.2':('gbdt',('equation',)),
          'GBDT_V6.1.3':('gbdt',('table','equation')),'MLP_V6.2.1':('mlp',('table',)),
          'MLP_V6.2.2':('mlp',('equation',)),'MLP_V6.2.3':('mlp',('table','equation'))}
STAGES=('eligible','controller_evaluated','nominal_scale_changed','boundary_clipped','requested_dimensions_changed',
        'processor_input_changed','processor_input_unchanged','processor_input_unknown','ocr_requested',
        'ocr_submitted','ocr_physically_completed','ocr_failed','ocr_submission_unknown','ocr_reused',
        'actual_tensor_verified','valid_reread','candidate_canonical_changed','applied_canonical_edit')


def quantiles(values):
    values=sorted(x for x in values if isinstance(x,(int,float)) and math.isfinite(x))
    if not values:return {'n':0,'min':None,'p50':None,'p90':None,'p95':None,'max':None,'sum':0}
    def percentile(q):
        i=(len(values)-1)*q;lo=int(i);hi=min(lo+1,len(values)-1)
        return values[lo]+(values[hi]-values[lo])*(i-lo)
    return {'n':len(values),'min':values[0],'p50':percentile(.5),'p90':percentile(.9),'p95':percentile(.95),'max':values[-1],'sum':sum(values)}


def canonical_bytes(path):
    value=Path(path).read_text(encoding='utf-8').replace('\r\n','\n').replace('\r','\n').strip()
    return hashlib.sha256(value.encode()).hexdigest()


def analyze(manifest):
    m=read_json(manifest)
    if len(m['pages'])!=1651:raise ValueError('Full denominator is required')
    sets={v:{k:{s:set() for s in STAGES} for k in kinds} for v,(_,kinds) in VERSIONS.items()}
    details={v:{k:{'predict_api_calls':0,'predict_batches':0,'predicted_rows':0,'physical_completed_ocr_requests':0,
                    'submitted_ocr_requests':0,'failed_ocr_attempts':0,'reused_actions':0,'checkpoint_sha256':set(),'raw_log_scales':[],'bounded_scales':[],
                    'controller_seconds':[],'resize_seconds':[],'preview_seconds':[],'physical_ocr_seconds':[],
                    'requested_width':[],'requested_height':[],'prepared_width':[],'prepared_height':[],
                    'pixel_tensor_rows':[],'grid_height':[],'grid_width':[],
                    'rejections':collections.Counter(),'standalone_regional_seconds_estimate':0.} for k in kinds} for v,(_,kinds) in VERSIONS.items()}
    pages={v:{'raw_changed':set(),'canonical_changed':set(),'empty':0,'failures':0,'latency':[],
               'standalone_seconds_estimate':0.} for v in ('D0',*VERSIONS)}
    global_timing=collections.defaultdict(list);unknown_native=[];physical_family=collections.Counter();status=collections.Counter()
    page_rows=[];native_submitted_total=0;submitted_total=0
    for page in m['pages']:
        pid=page['page_id'];out=Path(page['source_run_root'])/'shared/output';folder=out/'pages'/pid
        record=read_json(out/'receipts'/(pid+'.json'));status[record['status']]+=1
        native_file=out/'arms/D0/markdown'/(pid+'.md');native_hash=digest(native_file);native_canonical=canonical_bytes(native_file)
        for name,value in record.get('timing',{}).items():global_timing[name].append(value)
        render_path=folder/'capture/RENDER.json'
        if render_path.exists():
            render=read_json(render_path)
            if render['configured_dpi']!=200:raise ValueError('Native rendering DPI changed')
            for name,value in zip(('native_image_width','native_image_height'),render['size']):global_timing[name].append(value)
        global_timing['page_seconds'].append(record['elapsed_seconds'])
        global_timing['successful_page_seconds' if record['status']=='success' else 'failed_page_seconds'].append(record['elapsed_seconds'])
        eligible_path=folder/'ELIGIBLE_REGIONS.json';regions=read_json(eligible_path)['regions'] if eligible_path.exists() else []
        if not eligible_path.exists():unknown_native.append(pid)
        features=read_json(folder/'FEATURES.json') if (folder/'FEATURES.json').exists() else {}
        candidates={family:{int(k):v for k,v in read_json(folder/(family+'_CANDIDATES.json')).items()}
                    if (folder/(family+'_CANDIDATES.json')).exists() else {} for family in ('gbdt','mlp')}
        telemetry=collections.defaultdict(dict);submitted=collections.Counter();context=None
        if (folder/'events.jsonl').exists():
            for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines():
                try:event=json.loads(line)
                except json.JSONDecodeError:continue
                if event.get('family') in ('gbdt','mlp') and 'region_id' in event:
                    context=(event['family'],event['region_id']);telemetry[context][event['event']]=event
                if event.get('event')=='region_features':features.setdefault(str(event['region_id']),event)
                if event.get('event')=='model_request_submitted':
                    submitted_total+=1
                    if context is None:native_submitted_total+=1
                    else:submitted[context]+=1
        for feature in features.values():
            for key in ('crop_seconds','feature_seconds','reference_preprocess_seconds'):
                if key in feature:global_timing[key].append(feature[key])
        for family,rows in candidates.items():
            for row in rows.values():physical_family[family]+=row['physical_requests']
        for (family,index),evidence in telemetry.items():
            if index not in candidates[family]:physical_family[family]+=evidence.get('regional_ocr_completed',{}).get('physical_requests',0)
        for version in ('D0',*VERSIONS):
            rr=read_json(out/'arms'/version/'receipts'/(pid+'.json'));file=out/'arms'/version/'markdown'/(pid+'.md')
            if digest(file)!=rr['prediction_sha256']:raise ValueError('Prediction changed before analysis')
            pp=pages[version];pp['failures']+=rr['status']!='success';pp['empty']+=rr['empty_prediction'];pp['latency'].append(rr['elapsed_seconds'])
            if digest(file)!=native_hash:pp['raw_changed'].add(pid)
            if canonical_bytes(file)!=native_canonical:pp['canonical_changed'].add(pid)
            pp['standalone_seconds_estimate']+=record.get('timing',{}).get('native_seconds',0)
            if version=='D0':continue
            family,kinds=VERSIONS[version];audit_path=folder/'arms'/version/'ASSEMBLY.json'
            audit=read_json(audit_path)['audit'] if audit_path.exists() else [];applied={x['index'] for x in audit if x['applied']}
            for region in regions:
                kind=region['kind'];index=region['index']
                if kind not in kinds:continue
                ss=sets[version][kind];dd=details[version][kind];key=(pid,index);ss['eligible'].add(key)
                row=candidates[family].get(index)
                evidence=telemetry.get((family,index),{});failure=evidence.get('regional_ocr_failed')
                if row is None and 'controller_evaluated' in evidence:
                    called=evidence['controller_evaluated'];prepared=evidence.get('actor_action_prepared',{})
                    completed=evidence.get('regional_ocr_completed',{})
                    row={**called,**prepared,'decision':called['decision'],'accepted':False,
                         'logical_ocr_requested':bool(prepared),'physical_requests':completed.get('physical_requests',0),
                         'reused':completed.get('reused',False),'partial_action_record':True}
                if row is None:
                    ss['processor_input_unknown'].add(key);continue
                decision=row['decision'];dd['checkpoint_sha256'].add(row['checkpoint_sha256'])
                for name in ('predict_api_calls','predict_batches','predicted_rows'):dd[name]+=row[name]
                if row['predicted_rows']:ss['controller_evaluated'].add(key)
                for name in ('nominal_scale_changed','requested_dimensions_changed'):
                    if row.get(name):ss[name].add(key)
                if decision.get('clipped'):ss['boundary_clipped'].add(key)
                if 'raw_log_scale' in decision:dd['raw_log_scales'].append(decision['raw_log_scale'])
                if decision.get('scale') is not None:dd['bounded_scales'].append(decision['scale'])
                changed=row.get('input_changed_vs_1x');ss['processor_input_unknown' if changed is None else ('processor_input_changed' if changed else 'processor_input_unchanged')].add(key)
                if row.get('logical_ocr_requested'):ss['ocr_requested'].add(key)
                submitted_count=submitted.get((family,index),0);dd['submitted_ocr_requests']+=submitted_count
                if submitted_count:ss['ocr_submitted'].add(key)
                if failure:
                    ss['ocr_failed'].add(key);dd['failed_ocr_attempts']+=failure['attempted_requests']
                elif row.get('partial_action_record') and row.get('logical_ocr_requested') and not evidence.get('regional_ocr_completed'):
                    ss['ocr_submission_unknown'].add(key)
                if row['physical_requests']:ss['ocr_physically_completed'].add(key)
                if row.get('reused'):ss['ocr_reused'].add(key);dd['reused_actions']+=1
                if row.get('actual_tensor_identity_verified') or evidence.get('regional_ocr_completed',{}).get('actual_tensor_identity_verified'):
                    ss['actual_tensor_verified'].add(key)
                dd['physical_completed_ocr_requests']+=row['physical_requests']
                size=decision.get('requested_dimensions')
                if size:
                    dd['requested_width'].append(size[0]);dd['requested_height'].append(size[1])
                realized=row.get('realized',{});size=realized.get('prepared',{}).get('size')
                if size:
                    dd['prepared_width'].append(size[0]);dd['prepared_height'].append(size[1])
                tensors=realized.get('tensors',{});shape=tensors.get('pixel_values',{}).get('shape')
                if shape:dd['pixel_tensor_rows'].append(shape[0])
                grid=tensors.get('image_grid_thw',{}).get('values')
                if grid:
                    dd['grid_height'].append(grid[0][1]);dd['grid_width'].append(grid[0][2])
                if row['accepted']:ss['valid_reread'].add(key)
                if row.get('canonical_output_changed'):ss['candidate_canonical_changed'].add(key)
                if index in applied and row.get('canonical_output_changed'):ss['applied_canonical_edit'].add(key)
                if row.get('rejection'):dd['rejections'][row['rejection']]+=1
                for name in ('controller_seconds','resize_seconds','preview_seconds'):dd[name].append(row.get(name,0.))
                dd['physical_ocr_seconds'].append(row.get('ocr_seconds',0.))
                base=features.get(str(index),{});estimate=sum(base.get(k,0.) for k in ('crop_seconds','feature_seconds','reference_preprocess_seconds'))
                estimate+=sum(row.get(k,0.) for k in ('controller_seconds','resize_seconds','preview_seconds','ocr_seconds'))
                if row.get('reused'):
                    owner=row.get('reuse_from');original=candidates.get(owner['family'],{}).get(owner['region_id']) if owner else None
                    if original is not None:estimate+=original.get('ocr_seconds',0.)
                dd['standalone_regional_seconds_estimate']+=estimate;pp['standalone_seconds_estimate']+=estimate
            # Assembly for each version was not independently timed. Attribute the
            # family's measured three-output assembly equally, explicitly estimated.
            pp['standalone_seconds_estimate']+=record.get('timing',{}).get(family+'_assembly_seconds',0.)/3
        page_rows.append({'page_id':pid,'source_run_id':page['source_run_id'],'status':record['status'],
                          'elapsed_seconds':record['elapsed_seconds'],'request_count':record.get('request_count'),
                          'native_request_count':record.get('native_model_requests'),
                          'prediction_hashes':{v:digest(out/'arms'/v/'markdown'/(pid+'.md')) for v in ('D0',*VERSIONS)},
                          'regions':[{**{k:row.get(k) for k in ('family','kind','region_id','checkpoint_sha256','before','after',
                              'reference_1x','realized','actual_tensor_identity_verified','input_changed_vs_1x','reused','reuse_from')},
                              'native_crop':features.get(str(index),{}).get('native_crop')}
                              for family,rows in candidates.items() for index,row in rows.items()]})
    result={'at':utc(),'manifest_sha256':digest(manifest),'page_denominator':1651,'render_dpi':200,'render_dpi_changed_pages':0,
            'native_model_requests_submitted':native_submitted_total,'total_model_requests_submitted':submitted_total,
            'native_detection_unavailable_pages':len(unknown_native),'missed_detections':'Unknown without offline GT correspondence; absence of eligible native detection is not measured recall',
            'physical_completed_regional_requests_by_family':dict(physical_family),'inference_status':dict(status),
            'shared_stage_seconds':{k:quantiles(v) for k,v in global_timing.items() if not k.startswith('native_image_')},
            'native_image_dimensions':{k:quantiles(v) for k,v in global_timing.items() if k.startswith('native_image_')},'versions':{},
            'reuse_note':'Per-arm controller and OCR figures are logical attribution. Each family physically evaluates each eligible region once; table/formula/combined outputs reuse those decisions. Combined funnel page counts are unions. Cross-family OCR reuse is counted separately.',
            'cost_note':'Stage totals may overlap and are not elapsed wall time. Per-version standalone estimates reattribute shared feature/1x preprocessing and reused OCR, excluding unmeasured independent startup and failed partial work.',
            'failure_accounting_note':'Durable prediction, action-preparation and submission events retain failed-action evidence. Requested, successfully submitted, completed and failed requests are separate. Interrupted actions without a terminal event are marked unknown.',
            'quality_outcomes':'Pending separate paired official scoring; changed output is not assumed improved.'}
    for version,pp in pages.items():
        item={'denominator':1651,'raw_changed_pages':len(pp['raw_changed']),'canonical_changed_pages':len(pp['canonical_changed']),
              'empty_predictions':pp['empty'],'shared_track_failed_pages':pp['failures'],'shared_page_latency_seconds':quantiles(pp['latency']),
              'standalone_seconds_estimate':pp['standalone_seconds_estimate'],'by_kind':{}}
        if version in VERSIONS:
            for kind,ss in sets[version].items():
                dd=details[version][kind];converted={}
                for key,value in dd.items():
                    if isinstance(value,set):converted[key]=sorted(value)
                    elif isinstance(value,list):converted[key]=quantiles(value)
                    else:converted[key]=dict(value) if isinstance(value,collections.Counter) else value
                item['by_kind'][kind]={'funnel':{s:{'regions':len(v),'unique_pages':len({x[0] for x in v}),'page_denominator':1651} for s,v in ss.items()},**converted}
            item['combined_funnel']={stage:{'regions':len(set().union(*(ss[stage] for ss in sets[version].values()))),
                                                 'unique_pages':len({x[0] for ss in sets[version].values() for x in ss[stage]}),'page_denominator':1651} for stage in STAGES}
        result['versions'][version]=item
    result['family_physical_controller_totals']={family:{key:sum(d[key] for d in details[version].values())
        for key in ('predict_api_calls','predict_batches','predicted_rows','submitted_ocr_requests','physical_completed_ocr_requests','reused_actions')}
        for family,version in (('gbdt','GBDT_V6.1.3'),('mlp','MLP_V6.2.3'))}
    return result,page_rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--private-pages',required=True);a=p.parse_args()
    result,private=analyze(a.manifest);atomic_json(a.output,result);atomic_json(a.private_pages,private)
    print(json.dumps({'output':a.output,'page_denominator':1651,'versions':len(result['versions'])}))
