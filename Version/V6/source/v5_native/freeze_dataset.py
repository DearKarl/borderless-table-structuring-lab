"""Freeze matched external scale targets and validation-only fixed controls."""
import argparse
import collections
import json
import math
from pathlib import Path

from .direct_controller import direct_target
from .features import FEATURE_NAMES
from .io_utils import atomic_json,digest,utc


def freeze(pools,scores,output,events):
    if len(pools)!=len(scores):raise ValueError('One completed score bundle per input pool is required')
    rows=[];bindings=[];seen=set();groups={};pixels={};pdfs={};excluded=[];responses=[]
    for pool_path,score_path in zip(pools,scores):
        pool=json.loads(Path(pool_path).read_text(encoding='utf-8'))
        measured=json.loads(Path(score_path).read_text(encoding='utf-8'))
        if pool.get('GT_access') is not False or measured['scope']!='external_only' or not measured.get('completed_at'):
            raise ValueError('Only completed external measurements and no-GT inference manifests may be frozen')
        index={r['region_id']:r for r in measured['rows']}
        if len(index)!=len(measured['rows']) or set(index)!={r['region_id'] for r in pool['rows']}:
            raise ValueError('Every selected region, including failures, must have one score disposition')
        bindings.append({'pool_sha256':digest(pool_path),'scores_sha256':digest(score_path),
                         'pool_path':str(Path(pool_path).resolve()),'scores_path':str(Path(score_path).resolve())})
        for sample in pool['rows']:
            rid=sample['region_id'];score=index[rid]
            if rid in seen:raise ValueError('Duplicate native region identifier')
            seen.add(rid)
            if sample['unit_contract']!='source_page_native_predicted_region_v1':raise ValueError('Wrong supervision unit')
            if sample['split'] not in ('train','validation','test'):raise ValueError('Unknown split')
            for key in ('kind','source_group','split','features'):
                if sample[key]!=score[key]:raise ValueError('Measurement and source lineage differ: '+key)
            if len(sample['features'])!=len(FEATURE_NAMES) or not all(math.isfinite(x) for x in sample['features']):
                raise ValueError('Nonfinite native feature')
            group=sample['kind']+':'+sample['source_group']
            if group in groups and groups[group]!=sample['split']:raise ValueError('Source document crosses split')
            groups[group]=sample['split']
            pixel=sample['native_pixel_identity']['pixel_sha256']
            if pixel in pixels:raise ValueError('Exact duplicate native region pixels require an explicit disposition')
            pixels[pixel]=rid
            # Capture identity catches repeated source pages despite differently spelled group IDs.
            page=sample['native_capture_pixel_sha256']
            if page in pdfs and pdfs[page]!=sample['split']:raise ValueError('Identical source page crosses split')
            pdfs[page]=sample['split']
            baseline=score.get('baseline_metrics',{})
            native=baseline.get('quality') if baseline.get('quality_valid') else None
            recalculated=direct_target(native,score.get('candidates',[]))
            if recalculated['target_valid']!=score['target_valid']:raise ValueError('Target validity differs')
            for key in ('target_scale','target_log_scale','target_candidate_index'):
                if key in recalculated and recalculated[key]!=score[key]:raise ValueError('Frozen target differs: '+key)
            responses.append(score)
            if not recalculated['target_valid']:
                excluded.append({'region_id':rid,'kind':sample['kind'],'split':sample['split'],
                                 'reason':score.get('reason','no_valid_measured_reread')})
                continue
            rows.append({k:sample[k] for k in ('region_id','kind','source_group','split','features')}|
                        recalculated|{'weight':1,'input_sha256':sample['input_sha256'],
                                      'native_pixel_sha256':pixel,'measurement_probe_sha256':score['measurement_probe_sha256']})
    group_counts=collections.Counter((r['kind'],r['source_group']) for r in rows)
    if max(group_counts.values(),default=0)>4:raise ValueError('Source contribution cap exceeded')
    if len(rows)>3072:raise ValueError('Main pool budget exceeded')
    # Fixed policies use the same validation regions for every candidate scale.
    # Invalid/rejected/missing rereads fall back to a valid native score. Unknown
    # baseline or selected-output metrics exclude the row from ALL fixed choices.
    fixed={};fixed_details={}
    scales=[1.,1.25,.535299427146522,.8626400854839283,.9294914874958631,
            1.468988298324275,2.0370816641840115,2.633920303088907]
    for kind in ('table','equation'):
        validation=[r for r in responses if r['kind']==kind and r['split']=='validation']
        common=[];unknown=[]
        for row in validation:
            baseline=row.get('baseline_metrics',{});native=baseline.get('quality') if baseline.get('quality_valid') else None
            candidates={c['scale']:c for c in row.get('candidates',[])};values=[]
            for scale in scales:
                candidate=candidates.get(scale)
                if candidate and candidate.get('accepted'):
                    value=candidate.get('quality') if candidate.get('quality_valid') else None
                else:value=native
                values.append(value)
            if any(v is None or not math.isfinite(v) for v in values):unknown.append(row['region_id'])
            else:common.append((row['region_id'],values))
        if not common:raise ValueError('No common validation quality coverage for fixed control')
        means=[sum(r[1][i] for r in common)/len(common) for i in range(len(scales))]
        best=max(means);tied=[i for i,x in enumerate(means) if best-x<=1e-6]
        chosen=min(tied,key=lambda i:(abs(math.log(scales[i])),scales[i]))
        fixed[kind]=scales[chosen]
        fixed_details[kind]={'selection_split':'validation','region_macro_mean':True,'selected_scale':scales[chosen],
                             'candidate_scales':scales,'mean_retained_quality':means,'common_rows':len(common),
                             'validation_denominator':len(validation),'unknown_region_ids':unknown,
                             'common_region_ids':[r[0] for r in common],'tie_tolerance':1e-6}
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    provenance={'scope':'external_only','unit_contract':'source_page_native_predicted_region_v1',
                'source_bindings':bindings,'features':FEATURE_NAMES,'one_target_per_region':True,
                'region_weight':1,'families_share_exact_rows':True,'target':'best_valid_measured_reread_log_scale',
                'benchmark_access':False,'runtime_scale_search':False,'auxiliary_gain_head':False}
    split_hashes={};counts={}
    for split in ('train','validation','test'):
        subset=[r for r in rows if r['split']==split]
        atomic_json(out/(split+'.json'),{'provenance':provenance,'split':split,'rows':subset})
        split_hashes[split]=digest(out/(split+'.json'))
        counts[split]={'regions':len(subset),'source_groups':len({(r['kind'],r['source_group']) for r in subset}),
                       'by_kind':dict(collections.Counter(r['kind'] for r in subset))}
    atomic_json(out/'FIXED_CONTROLS.json',{'scales':fixed,'selection':fixed_details,'test_used_for_selection':False})
    atomic_json(out/'EXCLUSIONS.json',{'selected_regions':len(seen),'excluded':excluded,'quality_improvement_filter':False})
    receipt={'at':utc(),'event':'dataset_frozen','selected_regions':len(seen),'valid_targets':len(rows),
             'excluded_targets':len(excluded),'source_groups':len(group_counts),'splits':counts,'split_sha256':split_hashes,
             'features':FEATURE_NAMES,'non_beneficial_retained':sum(r['non_beneficial'] is True for r in rows),
             'non_benefit_unknown':sum(r['non_beneficial'] is None for r in rows),
             'multiple_distinct_tied_inputs':sum(r['distinct_tied_inputs']>1 for r in rows),
             'fixed_control_sha256':digest(out/'FIXED_CONTROLS.json'),'source_bindings':bindings,
             'public_safe':False,'device':'cpu','checkpoint_fits_started':False}
    atomic_json(out/'DATASET_FROZEN.json',receipt)
    with open(events,'a',encoding='utf-8') as f:f.write(json.dumps(receipt)+'\n')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pool',action='append',required=True);p.add_argument('--scores',action='append',required=True)
    p.add_argument('--output',required=True);p.add_argument('--phase-events',required=True);a=p.parse_args()
    print(json.dumps(freeze(a.pool,a.scores,a.output,a.phase_events),indent=2))
