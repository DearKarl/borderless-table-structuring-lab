"""CPU scoring with the original strict output parser and frozen TEDS source."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
from pathlib import Path
import time
from .execution import checked_json,sha,write,now
from versions.generator_pilot import large_score

def initialize(path,expected):
    if sha(path)!=expected:raise ValueError('Frozen table scorer changed')
    spec=importlib.util.spec_from_file_location('tdatr_frozen_teds',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    large_score.SCORERS=[module.TEDS(),module.TEDS(structure_only=True)]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding',required=True);args=ap.parse_args()
    binding=json.loads(Path(args.binding).read_text())
    protocol=checked_json(binding['protocol'],binding['protocol_sha256'])
    gold=checked_json(binding['gold'],binding['gold_sha256'])
    predictions=checked_json(binding['predictions'],binding['predictions_sha256'])
    if gold['cohort'] not in ['dev','confirmation'] or gold['denominator']!={'dev':256,'confirmation':512}[gold['cohort']]:
        raise ValueError('Original reference denominator differs')
    if protocol['gold_hashes'][gold['cohort']]!=binding['gold_sha256']:raise ValueError('References not in the frozen protocol')
    if predictions['status'] not in ['complete','failed'] or predictions['cohort']!=gold['cohort'] or predictions['denominator']!=gold['denominator']:
        raise ValueError('Inference is incomplete or for another cohort')
    original={r['id'] for r in gold['rows']};by_id={r['id']:r for r in predictions['rows']}
    if len(original)!=gold['denominator'] or len(by_id)!=len(predictions['rows']) or set(by_id)-original:
        raise ValueError('Inference membership differs')
    begun=time.monotonic();tasks=[];dispositions=[]
    for row in gold['rows']:
        predicted=by_id.get(row['id'],{})
        tasks.append((row,predicted.get('prediction')))
        dispositions.append(dict(id=row['id'],inference_status=predicted.get('status','missing')))
    pool=ProcessPoolExecutor(max_workers=4,initializer=initialize,
        initargs=(binding['scorer_source'],protocol['scoring_source_sha256']))
    try:scores=[f.result(timeout=300) for f in [pool.submit(large_score.score_record,task) for task in tasks]]
    except BaseException:
        for process in pool._processes.values():process.terminate()
        pool.shutdown(wait=True,cancel_futures=True);raise
    else:pool.shutdown(wait=True)
    if len(scores)!=gold['denominator']:raise ValueError('Scoring lost an original source')
    keys=['teds','structure_teds','character_distance','cell_text_exact_match','exact_grid','exact_table']
    aggregate={key:sum(row[key] for row in scores)/len(scores) for key in keys}
    aggregate.update({key+'_count':sum(row[key] for row in scores) for key in ['malformed','missing','unmatchable_grid','truncated']})
    result=dict(at_utc=now(),status='complete',inference_status=predictions['status'],cohort=gold['cohort'],denominator=len(scores),
        run_id=predictions['run_id'],recipe=predictions['recipe'],version=predictions['version'],seed=None,epoch=predictions.get('epoch'),
        rows=scores,aggregate=aggregate,output_dispositions=dispositions,scoring_seconds=time.monotonic()-begun,
        prediction_file_sha256=binding['predictions_sha256'],gold_file_sha256=binding['gold_sha256'],protocol_sha256=binding['protocol_sha256'],
        scorer_sha256=sha(binding['scorer_source']),all_sources_retained=True,reference_text_normalized=False,GPU_allocations=0)
    path=Path(binding['output'])
    if path.exists():raise FileExistsError('No score overwrite or automatic retry')
    write(path,result)

if __name__=='__main__':main()
