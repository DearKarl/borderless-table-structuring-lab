"""Offline task scoring and sampled scale targets; no model runtime access."""
import argparse
import concurrent.futures
import json
import pathlib
import signal
import time

from .direct_controller import direct_target
from .external_metrics import score
from .io_utils import atomic_json,digest,utc


def score_case(arguments):
    row,measurements,output=arguments;out=pathlib.Path(output)
    def deadline(*_):raise TimeoutError('Individual metric exceeded 60 seconds')
    signal.signal(signal.SIGALRM,deadline)
    def measure(row,prediction,identifier):
        signal.setitimer(signal.ITIMER_REAL,60);start=time.monotonic()
        try:return {**score(row['kind'],row['canonical_ground_truth'],prediction,out/'metric_work'/row['region_id'],identifier),'seconds':time.monotonic()-start}
        except Exception as exc:return {'quality':None,'quality_valid':False,'reason':type(exc).__name__+': '+str(exc),'seconds':time.monotonic()-start}
        finally:signal.setitimer(signal.ITIMER_REAL,0)
    rid=row['region_id'];folder=pathlib.Path(measurements)/'pages'/rid
    item={k:row[k] for k in ('region_id','kind','split','source_group','features')}
    if not (folder/'PROBE.json').exists():
        item.update(target_valid=False,reason='no_measurement_probe')
    else:
        probe=json.loads((folder/'PROBE.json').read_text())
        assert probe['scope']=='external_source_page_native_region'
        baseline=measure(row,probe['baseline'],rid+'-native')
        candidates=[]
        for r in probe['candidates']:
            if r.get('alias_of') is not None:
                metrics=candidates[r['alias_of']]['metrics'];metrics={**metrics,'reused_metric':True}
            else:metrics=measure(row,r['output'],rid+'-view-'+str(r['index']))
            candidates.append({**r,**{k:metrics[k] for k in ('quality','quality_valid')},'metrics':metrics})
        native_quality=baseline['quality'] if baseline['quality_valid'] else None
        item.update(baseline_metrics=baseline,candidates=candidates,
                    **direct_target(native_quality,candidates),measurement_probe_sha256=digest(folder/'PROBE.json'))
    return item


def main():
    p=argparse.ArgumentParser();p.add_argument('--labels',required=True);p.add_argument('--measurements',required=True)
    p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if not 1<=a.workers<=4:raise ValueError('External CPU metric pool is bounded to four processes')
    labels=json.loads(pathlib.Path(a.labels).read_text())['rows'];out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True)
    result={'started_at':utc(),'scope':'external_only','device':'cpu','workers':a.workers,'label_bundle_sha256':digest(a.labels),'rows':[]}
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as executor:
        futures=[executor.submit(score_case,(row,a.measurements,a.output)) for row in labels]
        for future in concurrent.futures.as_completed(futures):
            item=future.result();result['rows'].append(item);atomic_json(out/'SCALE_LABELS.json',result)
            print(item['region_id'],item['target_valid'],flush=True)
    order={r['region_id']:i for i,r in enumerate(labels)}
    result['rows'].sort(key=lambda r:order[r['region_id']])
    result.update(completed_at=utc(),valid_targets=sum(r['target_valid'] for r in result['rows']))
    atomic_json(out/'SCALE_LABELS.json',result)


if __name__=='__main__':main()
