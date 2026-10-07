"""Source-cluster paired external quality summaries; no policy selection or refit."""
import argparse
import collections
import itertools
import json
from pathlib import Path

import numpy as np

from .io_utils import atomic_json,digest,read_json,utc


def paired(rows,left,right,repetitions=2000):
    def quality(row,name):
        metric=row['baseline_metrics'] if name=='native' else row['actors'][name]['metrics']
        return metric.get('quality') if metric.get('quality_valid') else None
    pairs=[]
    for row in rows:
        a,b=quality(row,left),quality(row,right)
        if a is not None and b is not None:pairs.append((row['source_group'],a-b))
    summary={'comparison':left+' minus '+right,'region_denominator':len(rows),'paired_regions':len(pairs),
             'unmatchable_regions':len(rows)-len(pairs),'positive_favors':left,'tie_tolerance':1e-6}
    if not pairs:return {**summary,'mean_delta':None,'source_cluster_bootstrap_95ci':None,'reason':'No common valid metric coverage'}
    values=np.asarray([d for _,d in pairs]);groups=collections.defaultdict(list)
    for group,delta in pairs:groups[group].append(delta)
    blocks=list(groups.values());rng=np.random.default_rng(0);draws=[]
    for _ in range(repetitions):
        indices=rng.integers(0,len(blocks),size=len(blocks));sample=[value for i in indices for value in blocks[i]]
        draws.append(float(np.mean(sample)))
    return {**summary,'mean_delta':float(np.mean(values)),'source_groups':len(groups),
            'improved':int(np.sum(values>1e-6)),'worsened':int(np.sum(values< -1e-6)),'tied':int(np.sum(np.abs(values)<=1e-6)),
            'source_cluster_bootstrap_95ci':[float(x) for x in np.percentile(draws,[2.5,97.5])],
            'bootstrap_repetitions':repetitions,'bootstrap_seed':0,'bootstrap_unit':'Source document; all paired regions within resampled documents retained',
            'estimand':'Equal-region macro quality delta on common metric coverage; no significance or multiple-comparison correction claim'}


def summarize(path):
    raw=read_json(path)
    if not raw.get('completed_at') or len(raw['rows'])!=raw['regions_denominator']:raise ValueError('Completed all-region actor scoring is required')
    result={'at':utc(),'scope':'external_source_page_native_regions','score_sha256':digest(path),
            'quality_units':'0-1','policy_refitted':False,'benchmark_used':False,'splits':{}}
    for split in ('validation','test'):
        result['splits'][split]={}
        for kind in ('table','equation'):
            rows=[r for r in raw['rows'] if r['split']==split and r['kind']==kind]
            metrics={}
            for family in ('native','gbdt','mlp','fixed'):
                data=[r['baseline_metrics'] if family=='native' else r['actors'][family]['metrics'] for r in rows]
                valid=[m['quality'] for m in data if m['quality_valid']]
                metrics[family]={'mean_quality':float(np.mean(valid)) if valid else None,'valid_regions':len(valid),
                                 'denominator':len(rows),'unmatchable_regions':len(rows)-len(valid)}
                if family!='native':
                    actions=[r['actors'][family] for r in rows]
                    metrics[family].update(completed_actions=sum(a['actual_action_completed'] for a in actions),
                                          accepted_actions=sum(a['accepted'] for a in actions),
                                          outcomes=dict(collections.Counter(a['outcome'] for a in actions)))
            comparisons=[paired(rows,a,b) for a,b in [('gbdt','native'),('mlp','native'),('fixed','native'),('gbdt','fixed'),('mlp','fixed'),('mlp','gbdt')]]
            result['splits'][split][kind]={'metric':'TEDS' if kind=='table' else 'CDM','region_denominator':len(rows),
                                         'source_groups':len({r['source_group'] for r in rows}),'metrics':metrics,
                                         'paired_comparisons':comparisons,'inference_status':dict(collections.Counter(r['inference_status'] for r in rows))}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scores',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=summarize(a.scores);atomic_json(a.output,result);print(json.dumps(result,indent=2))
