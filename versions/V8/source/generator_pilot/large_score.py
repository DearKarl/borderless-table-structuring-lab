"""Offline frozen TEDS scoring with complete held-out source denominators."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
import time
from .common import ROOT,OUT,read,sha,now
from .strict_output import validate_prediction

SCORERS=None


def initialize_scorers():
    global SCORERS
    sys.path.insert(0,str(OUT/'scoring-deps'))
    path=ROOT/'evaluation/omnidocbench/src/metrics/table_metric.py'
    spec=importlib.util.spec_from_file_location('large_frozen_teds',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    SCORERS=[module.TEDS(),module.TEDS(structure_only=True)]


def score_record(value):
    gold,record=value
    import Levenshtein
    truth=gold['grid'];gold_cells={(c['row'],c['col'],c['rowspan'],c['colspan']):c['text'] for c in truth['cells']}
    gold_text='\u241f'.join(c['text'] for c in truth['cells'])
    try:
        grid=validate_prediction(record)
        predicted=record['decoded_html'];predicted=predicted if predicted.lstrip().startswith('<html') else '<html><body>'+predicted+'</body></html>'
        full,structure=[scorer.evaluate(predicted,gold['gold_html']) for scorer in SCORERS]
        cells={(c['row'],c['col'],c['rowspan'],c['colspan']):c['text'] for c in grid['cells']}
        union=set(cells)|set(gold_cells);exact=sum(k in cells and k in gold_cells and cells[k]==gold_cells[k] for k in union)
        text='\u241f'.join(c['text'] for c in grid['cells']);distance=Levenshtein.distance(text,gold_text)/max(1,len(text),len(gold_text))
        return dict(id=gold['id'],source=gold['source'],teds=full,structure_teds=structure,character_distance=distance,
            cell_text_exact_match=exact/max(1,len(union)),exact_grid=set(cells)==set(gold_cells),exact_table=cells==gold_cells,
            malformed=False,missing=False,unmatchable_grid=set(cells)!=set(gold_cells),truncated=not record.get('eos_reached',False),reason=None)
    except (ValueError,TypeError,KeyError,IndexError) as error:
        return dict(id=gold['id'],source=gold['source'],teds=0.,structure_teds=0.,character_distance=1.,cell_text_exact_match=0.,
            exact_grid=False,exact_table=False,malformed=True,missing=record is None,unmatchable_grid=True,
            truncated=bool(record and not record.get('eos_reached',False)),reason=str(error))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gold',required=True);parser.add_argument('--predictions',required=True)
    parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True);parser.add_argument('--legacy',action='store_true')
    args=parser.parse_args();gold=read(args.gold);predictions=read(args.predictions);protocol=read(args.protocol)
    scorer_path=ROOT/'evaluation/omnidocbench/src/metrics/table_metric.py'
    if sha(scorer_path)!=protocol['scoring_source_sha256']:raise ValueError('Frozen scorer changed')
    if sha(Path(args.gold))!=protocol['gold_hashes'][gold['cohort']]:raise ValueError('Reference cohort changed')
    if len(gold['rows'])!=gold['denominator']:raise ValueError('Gold denominator differs')
    if predictions['cohort']!=gold['cohort'] or predictions['denominator']!=gold['denominator']:raise ValueError('Inference and scoring cohorts differ')
    if predictions['status'] not in ['complete','failed']:raise ValueError('Inference still running')
    by_id={r['id']:r for r in predictions['rows']}
    if len(by_id)!=len(predictions['rows']) or set(by_id)-{r['id'] for r in gold['rows']}:raise ValueError('Prediction membership differs')
    if args.legacy and predictions['recipe']!='base':raise ValueError('Legacy comparator requires base model')
    tasks=[];details=[]
    for row in gold['rows']:
        result=by_id.get(row['id'],{});use_candidate=args.legacy and result.get('legacy_rule_accepted',False)
        record=result.get('fixed_candidate') if use_candidate else result.get('prediction')
        tasks.append((row,record));details.append(dict(id=row['id'],inference_status=result.get('status','missing'),legacy_candidate_used=bool(use_candidate)))
    started=time.monotonic();scores=[]
    # Four offline workers, fixed before quality; a stalled scorer fails the
    # stage instead of silently replacing a measured value or dropping a source.
    pool=ProcessPoolExecutor(max_workers=4,initializer=initialize_scorers)
    try:
        futures=[pool.submit(score_record,task) for task in tasks]
        for future in futures:scores.append(future.result(timeout=300))
    except BaseException:
        for process in pool._processes.values():process.terminate()
        pool.shutdown(wait=True,cancel_futures=True);raise
    else:pool.shutdown(wait=True)
    if len(scores)!=gold['denominator']:raise ValueError('Incomplete scored denominator')
    keys=['teds','structure_teds','character_distance','cell_text_exact_match','exact_grid','exact_table']
    aggregate={key:sum(row[key] for row in scores)/len(scores) for key in keys}
    aggregate.update({key+'_count':sum(row[key] for row in scores) for key in ['malformed','missing','unmatchable_grid','truncated']})
    output=Path(args.output)
    result=dict(at_utc=now(),status='complete',inference_status=predictions['status'],cohort=gold['cohort'],denominator=len(scores),
        run_id=predictions['run_id'],recipe='legacy_simple_rule' if args.legacy else predictions['recipe'],seed=predictions.get('seed'),epoch=predictions.get('epoch'),
        rows=scores,aggregate=aggregate,output_dispositions=details,scoring_seconds=time.monotonic()-started,
        prediction_file_sha256=sha(Path(args.predictions)),gold_file_sha256=sha(Path(args.gold)),protocol_sha256=sha(Path(args.protocol)),
        scorer_sha256=sha(scorer_path),all_sources_retained=True,reference_text_normalized=False)
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(dict(recipe=result['recipe'],seed=result['seed'],epoch=result['epoch'],cohort=result['cohort'],denominator=len(scores),aggregate=aggregate))


if __name__=='__main__':main()
