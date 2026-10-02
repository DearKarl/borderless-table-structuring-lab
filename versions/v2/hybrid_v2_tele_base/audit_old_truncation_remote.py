"""Read-only audit of exactly the already-scored fresh v1 inputs. No evaluation or writes."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT=Path('/srv/hybrid-research')
OLD=ROOT/'artifacts/official-full-20260921-v1'
LOCKS=OLD/'hybrid_v1/final-fresh-empty-v1-r2-score-inputs-v1'
EXPECTED={'M':'35d554dd58534ee75babe1dfae7cc53d9b30ea321c7a55f25319584d107e6b6d',
          'P':'26b7434c1aff95689b539e79201b98f655a8c036fd8c8eaafc818fb5b3b8955f',
          'H':'8bfe61e41eefbe951ac239406a2e8d6fd27a50db5399715802d2beb618848fd7'}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while b:=stream.read(8*1024*1024):h.update(b)
    return h.hexdigest()


def main():
    result={'read_only':True,'inference_started':False,'evaluation_started':False,'GT_read':False,'arms':{}}
    for arm,expected in EXPECTED.items():
        path=LOCKS/(arm+'_COMPLETE_INPUT_LOCK.json')
        if sha(path)!=expected:raise RuntimeError('Existing scoring lock changed')
        lock=json.loads(path.read_bytes());prediction=Path(lock['prediction_directory'])
        if not prediction.resolve().is_relative_to(OLD) or len(lock['pages'])!=1651:raise RuntimeError('Input boundary or coverage')
        start_path=OLD/('hybrid_v1_fresh_'+arm)/'official-evaluation-v1/START.json'
        start=json.loads(start_path.read_bytes())
        if start['input_lock_sha256']!=expected:raise RuntimeError('Score START is not linked to this input lock')
        counts=Counter();selected=[]
        for row in lock['pages']:
            raw=prediction/(row['page_id']+'.md')
            if raw.is_symlink() or sha(raw)!=row['prediction_sha256']:raise RuntimeError('Actual primary differs')
            size=raw.stat().st_size;counts[row['status']]+=1
            if row['status']!='success':
                selected.append({'page_id':row['page_id'],'status':row['status'],'primary_bytes':size,
                    'prediction_sha256':row['prediction_sha256'],'primary_path':str(raw)})
        result['arms'][arm]={'lock_path':str(path),'lock_sha256':expected,'score_start_sha256':sha(start_path),
            'score_mounts':[x for x in start['command'] if x.startswith('type=bind,') and 'dst=/pred' in x],
            'all_1651_actual_primary_hashes_verified':True,'counts':dict(counts),'non_success_pages':selected,
            'truncated_empty_primary':sum(x['status']=='truncated' and x['primary_bytes']==0 for x in selected),
            'truncated_nonempty_primary':sum(x['status']=='truncated' and x['primary_bytes']>0 for x in selected)}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
