"""CPU-only metric self-render validation; no inference or controller fit."""
import argparse,json,pathlib,signal,time
from .external_metrics import formula,score
from .io_utils import atomic_json,digest,utc


def main():
    p=argparse.ArgumentParser();p.add_argument('--labels',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    rows=json.loads(pathlib.Path(a.labels).read_text())['rows'];out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True)
    result={'started_at':utc(),'device':'cpu','label_bundle_sha256':digest(a.labels),'rows':[]}
    def deadline(*_):raise TimeoutError('External label metric validation exceeded 60 seconds')
    signal.signal(signal.SIGALRM,deadline)
    for row in rows:
        case={'region_id':row['region_id'],'kind':row['kind'],'split':row['split']};start=time.monotonic()
        signal.setitimer(signal.ITIMER_REAL,60)
        try:
            truth=formula(row['ground_truth'],require_outer=True) if row['kind']=='equation' else row['ground_truth']
            metrics=score(row['kind'],truth,truth,out/'metric_work',row['region_id'])
            case.update(canonical_ground_truth=truth,metrics=metrics,
                        label_valid=metrics['quality_valid'] and abs(metrics['quality']-1)<=1e-6)
            if not case['label_valid']:case['reason']='GT self-score is not one or metric failed'
        except Exception as exc:case.update(label_valid=False,reason=type(exc).__name__+': '+str(exc))
        finally:signal.setitimer(signal.ITIMER_REAL,0)
        case['elapsed_seconds']=time.monotonic()-start;result['rows'].append(case)
        atomic_json(out/'LABEL_VALIDITY.json',result)
        print(row['region_id'],case['label_valid'],case.get('reason',''),flush=True)
    result.update(completed_at=utc(),label_valid=sum(r['label_valid'] for r in result['rows']))
    atomic_json(out/'LABEL_VALIDITY.json',result)


if __name__=='__main__':main()
