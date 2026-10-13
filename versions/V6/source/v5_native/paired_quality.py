"""Offline official component outcomes and conservative native-region correspondence."""
import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path

from .io_utils import atomic_json,digest,read_json,utc
from .analyze_shared import VERSIONS

COMPONENTS={'table':{'TEDS':1,'TEDS_structure_only':1,'Edit_dist':-1},
            'equation':{'CDM':1,'Edit_dist':-1}}


def outcome(before,after,direction):
    if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in (before,after)):return 'unmatchable'
    delta=direction*(after-before)
    return 'improved' if delta>1e-6 else ('worsened' if delta< -1e-6 else 'tied')


def canonical(text,kind):
    value=str(text or '').replace('\r\n','\n').replace('\r','\n').strip()
    if kind=='equation':
        for left,right in (('$$','$$'),('\\[','\\]')):
            if value.startswith(left) and value.endswith(right):value=value[len(left):-len(right)].strip()
    else:
        value=re.sub(r'^\s*<(?:html|body)[^>]*>\s*','',value,flags=re.I)
        value=re.sub(r'^\s*<(?:html|body)[^>]*>\s*','',value,flags=re.I)
        value=re.sub(r'\s*</(?:html|body)>\s*$','',value,flags=re.I)
        value=re.sub(r'\s*</(?:html|body)>\s*$','',value,flags=re.I)
        value=re.sub(r'>\s+<','><',value)
    return value


def sample_key(row):
    indices=row.get('gt_idx')
    if not indices:return None
    truth=str(row.get('gt',''));page=Path(str(row.get('image_name',row.get('img_id','')))).name
    return (page,json.dumps(indices,sort_keys=True),hashlib.sha256(truth.encode()).hexdigest())


def load_samples(folder,kind):
    name='display_formula' if kind=='equation' else 'table'
    path=folder/('pred_quick_match_'+name+'_result.json')
    rows=read_json(path) if path.exists() else []
    index=collections.defaultdict(list);texts=collections.defaultdict(list)
    for row in rows:
        key=sample_key(row)
        if key is not None:index[key].append(row)
        page=Path(str(row.get('image_name',row.get('img_id','')))).name
        body=canonical(row.get('pred',''),kind)
        if body:texts[(page,body)].append(row)
    return rows,index,texts


def paired_samples(before,after,kind):
    brows,bindex,_=before;arows,aindex,_=after;keys=set(bindex)|set(aindex);result={}
    # Units with empty GT identity are retained as explicitly unmatchable rows.
    unidentified=max(sum(sample_key(x) is None for x in brows),sum(sample_key(x) is None for x in arows))
    for metric,direction in COMPONENTS[kind].items():
        counts=collections.Counter({'improved':0,'worsened':0,'tied':0,'unmatchable':unidentified});deltas=[]
        for key in keys:
            b,a=bindex.get(key,[]),aindex.get(key,[])
            if len(b)!=1 or len(a)!=1:counts['unmatchable']+=1;continue
            x=b[0].get('metric',{}).get(metric);y=a[0].get('metric',{}).get(metric);label=outcome(x,y,direction);counts[label]+=1
            if label!='unmatchable':deltas.append(y-x)
        result[metric]={'outcomes':dict(counts),'unit_denominator':sum(counts.values()),
                        'baseline_scored_rows':len(brows),'version_scored_rows':len(arows),
                        'mean_raw_delta_on_paired_units':sum(deltas)/len(deltas) if deltas else None,
                        'correspondence':'Same page, exact GT-index group and GT-text hash; unique in both outputs',
                        'direction':'higher' if direction==1 else 'lower','tie_tolerance':1e-6}
    return result


def page_values(folder):
    values={}
    for kind in ('text_block','display_formula','table','reading_order'):
        path=folder/('pred_quick_match_'+kind+'_per_page_edit.json')
        values[kind+'_edit']=read_json(path) if path.exists() else {}
    for kind in ('table','equation'):
        rows,_,_=load_samples(folder,kind)
        for metric,direction in COMPONENTS[kind].items():
            if metric=='Edit_dist':continue
            groups=collections.defaultdict(list)
            for row in rows:
                value=row.get('metric',{}).get(metric)
                if isinstance(value,(int,float)) and math.isfinite(value):groups[str(row.get('image_name',row.get('img_id','')))].append(value)
            values[kind+'_'+metric]={key:sum(v)/len(v) for key,v in groups.items()}
    return values


def analyze(manifest):
    m=read_json(manifest);root=Path(m['round_root'])
    if len(m['pages'])!=1651:raise ValueError('All1651 pages must remain in the comparison')
    folders={version:root/version/'evaluation/work/result' for version in ('D0',*VERSIONS)}
    sample_cache={v:{k:load_samples(f,k) for k in COMPONENTS} for v,f in folders.items()}
    pages={v:page_values(f) for v,f in folders.items()};result={'at':utc(),'manifest_sha256':digest(manifest),
        'page_denominator':1651,'comparison':'Each version minus its shared D0','ground_truth_use':'Offline scoring only',
        'native_region_correspondence':'Unique exact canonical native-prediction text to D0 official scored sample, same GT identity in version, and final prediction text must match expected retained region. Ambiguous/mixed/split cases remain unmatchable.',
        'canonicalization':'Line endings and outer whitespace; formula outer display delimiters; table outer html/body wrappers and whitespace between tags only',
        'page_metric_note':'Saved official per-page edit metrics; TEDS/CDM page values are diagnostic means of emitted sample metrics. No page-level Overall is constructed.',
        'versions':{}};private=[]
    for version,(family,kinds) in VERSIONS.items():
        combined_regions=collections.Counter({'improved':0,'worsened':0,'tied':0,'unmatchable':0})
        combined_pages={label:set() for label in combined_regions}
        item={'official_sample_outcomes':{k:paired_samples(sample_cache['D0'][k],sample_cache[version][k],k) for k in COMPONENTS},
              'page_outcomes':{},'eligible_native_region_outcomes':{}}
        for metric,before in pages['D0'].items():
            after=pages[version].get(metric,{});common=set(before)&set(after);direction=-1 if metric.endswith('_edit') else 1
            counts=collections.Counter({'improved':0,'worsened':0,'tied':0,'unmatchable':1651-len(common)})
            for page in common:counts[outcome(before[page],after[page],direction)]+=1
            item['page_outcomes'][metric]={'outcomes':dict(counts),'page_denominator':1651,'tie_tolerance':1e-6}
        for kind in kinds:
            counts={metric:collections.Counter({'improved':0,'worsened':0,'tied':0,'unmatchable':0}) for metric in COMPONENTS[kind]}
            page_sets={metric:{label:set() for label in ('improved','worsened','tied','unmatchable')} for metric in counts}
            for page in m['pages']:
                pid=page['page_id'];folder=Path(page['source_run_root'])/'shared/output/pages'/pid;ep=folder/'ELIGIBLE_REGIONS.json'
                if not ep.exists():continue
                regions=[r for r in read_json(ep)['regions'] if r['kind']==kind]
                candidate_path=folder/(family+'_CANDIDATES.json');candidates=read_json(candidate_path) if candidate_path.exists() else {}
                ap=folder/'arms'/version/'ASSEMBLY.json';applied={r['index'] for r in read_json(ap)['audit'] if r['applied']} if ap.exists() else set()
                btexts=sample_cache['D0'][kind][2];aindex=sample_cache[version][kind][1]
                names={str(page['original_page_id']),Path(str(page['original_page_id'])).name}
                names|={name+ext for name in list(names) for ext in ('.png','.jpg','.jpeg','.pdf')}
                for region in regions:
                    body=canonical(region['native'],kind);matched=[row for name in names for row in btexts.get((name,body),[])]
                    before=matched[0] if len(matched)==1 else None;key=sample_key(before) if before else None
                    afters=aindex.get(key,[]) if key else [];after=afters[0] if len(afters)==1 else None
                    chosen=candidates.get(str(region['index']));expected=chosen['output'] if region['index'] in applied and chosen else region['native']
                    if after and canonical(after.get('pred',''),kind)!=canonical(expected,kind):after=None
                    labels={}
                    for metric,direction in COMPONENTS[kind].items():
                        label=outcome(before.get('metric',{}).get(metric) if before else None,after.get('metric',{}).get(metric) if after else None,direction)
                        counts[metric][label]+=1;page_sets[metric][label].add(pid);labels[metric]=label
                    primary=labels['TEDS' if kind=='table' else 'CDM']
                    combined_regions[primary]+=1;combined_pages[primary].add(pid)
                    private.append({'version':version,'page_id':pid,'native_region_index':region['index'],'kind':kind,'outcomes':labels,'applied':region['index'] in applied})
            item['eligible_native_region_outcomes'][kind]={metric:{'regions':dict(counter),'unique_pages_by_outcome':{label:len(ids) for label,ids in page_sets[metric].items()},
                        'region_denominator':sum(counter.values()),'page_denominator':1651,
                        'page_note':'A page can contain both improved and worsened regions; outcome page sets are not mutually exclusive.'} for metric,counter in counts.items()}
        item['combined_primary_component_outcomes']={'regions':dict(combined_regions),
            'unique_pages_by_outcome':{label:len(ids) for label,ids in combined_pages.items()},
            'region_denominator':sum(combined_regions.values()),'page_denominator':1651,
            'definition':'Union across tables by TEDS and formulas by CDM; diagnostic outcome counts, not a composite quality score. Page outcome sets can overlap.'}
        result['versions'][version]=item
    return result,private


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--private-regions',required=True);a=p.parse_args()
    result,private=analyze(a.manifest);atomic_json(a.output,result);atomic_json(a.private_regions,private)
    print(json.dumps({'versions':len(result['versions']),'page_denominator':1651,'output':a.output}))
