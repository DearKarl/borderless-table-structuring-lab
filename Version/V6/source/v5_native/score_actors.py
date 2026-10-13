"""Offline quality of actual external actor outputs, retaining all held-out regions."""
import argparse
import concurrent.futures
import json
import signal
import time
from pathlib import Path

from .external_metrics import score
from .io_utils import atomic_json,digest,utc


def one_case(arguments):
    row,actions,output=arguments;rid=row['region_id'];folder=Path(actions)/'pages'/rid
    def deadline(*_):raise TimeoutError('Individual metric exceeded60seconds')
    signal.signal(signal.SIGALRM,deadline)
    def measure(text,label):
        signal.setitimer(signal.ITIMER_REAL,60);start=time.monotonic()
        try:return {**score(row['kind'],row['canonical_ground_truth'],text,Path(output)/'metric_work'/rid,label),'seconds':time.monotonic()-start}
        except Exception as exc:return {'quality':None,'quality_valid':False,'reason':type(exc).__name__+': '+str(exc),'seconds':time.monotonic()-start}
        finally:signal.setitimer(signal.ITIMER_REAL,0)
    probe=json.loads((folder/'ACTOR.json').read_text()) if (folder/'ACTOR.json').exists() else None
    receipt_path=Path(actions)/'receipts'/(rid+'.json');receipt=json.loads(receipt_path.read_text())
    if probe is not None:
        assert probe['baseline']==row['baseline'] and probe['region_id']==rid and probe['split']==row['split']
    baseline=measure(row['baseline'],rid+'-native');bank={row['baseline']:baseline}
    result={k:row[k] for k in ('region_id','kind','split','source_group')}
    result.update(baseline_metrics=baseline,inference_status=receipt['status'],inference_error=receipt['error'],actors={})
    actions_by_family={a['family']:a for a in probe['actions']} if probe else {}
    for family in ('gbdt','mlp','fixed'):
        action=actions_by_family.get(family);accepted=bool(action and action.get('accepted'))
        retained=action['output'] if accepted else row['baseline']
        if retained not in bank:bank[retained]=measure(retained,rid+'-'+family)
        metric=bank[retained];native=baseline['quality'] if baseline['quality_valid'] else None
        quality=metric['quality'] if metric['quality_valid'] else None
        delta=quality-native if quality is not None and native is not None else None
        result['actors'][family]={'metrics':metric,'accepted':accepted,'actual_action_completed':action is not None,
            'fallback_reason':None if accepted else (action.get('rejection') if action else 'actor_not_completed_retained_native'),
            'delta_over_native':delta,'outcome':'unmatchable' if delta is None else ('improved' if delta>1e-6 else ('worsened' if delta< -1e-6 else 'tied')),
            'predict_api_calls':action.get('predict_api_calls',0) if action else None,
            'physical_requests':action.get('physical_requests') if action else None,
            'reused':action.get('reused') if action else None,
            'scale':action.get('decision',{}).get('scale') if action else None}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--labels',required=True);p.add_argument('--actions',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    labels=json.loads(Path(a.labels).read_text());assert labels['scope']=='external_only'
    rows=labels['rows'];assert all(r['split'] in ('validation','test') for r in rows)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    result={'started_at':utc(),'scope':'external_only','device':'cpu','labels_sha256':digest(a.labels),
            'regions_denominator':len(rows),'rows':[],'quality_filter':False}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures=[executor.submit(one_case,(row,a.actions,a.output)) for row in rows]
        for future in concurrent.futures.as_completed(futures):
            result['rows'].append(future.result());atomic_json(out/'ACTOR_QUALITY.json',result)
            print('scored',len(result['rows']),flush=True)
    index={r['region_id']:i for i,r in enumerate(rows)};result['rows'].sort(key=lambda r:index[r['region_id']])
    result['completed_at']=utc();atomic_json(out/'ACTOR_QUALITY.json',result)


if __name__=='__main__':main()
