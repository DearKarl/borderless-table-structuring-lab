"""Freeze one fusion checkpoint per version using only the original DEV cohort."""
import argparse
import json
from pathlib import Path
from .execution import checked_json,sha,now

def select(endpoints):
    keyed={}
    for item in endpoints:
        score=item['score'];key=(item['version'],score['epoch'])
        if key in keyed:raise ValueError('Duplicate DEV endpoint')
        if (score['status']!='complete' or score['cohort']!='dev' or score['denominator']!=256
            or len(score['rows'])!=256 or len({r['id'] for r in score['rows']})!=256):raise ValueError('Incomplete scored DEV denominator')
        keyed[key]=item
    if set(keyed)!={(v,e) for v in ['V8.4','V8.5'] for e in [1,2,3]}:raise ValueError('All six fusion DEV endpoints are required')
    result={}
    for version in ['V8.4','V8.5']:
        best=max((keyed[version,e] for e in [1,2,3]),key=lambda x:(x['score']['aggregate']['teds'],-x['score']['epoch']))
        result[version]=dict(epoch=best['score']['epoch'],checkpoint_sha256=best['checkpoint_sha256'],
            checkpoint_path=best['checkpoint_path'],dev_teds=best['score']['aggregate']['teds'],
            score_sha256=best['score_sha256'],inference_status=best['score']['inference_status'])
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    manifest=json.loads(Path(args.manifest).read_text());endpoints=[]
    for item in manifest['endpoints']:
        score=checked_json(item['score_file'],item['score_sha256'])
        if score['protocol_sha256']!=manifest['protocol_sha256'] or score['gold_file_sha256']!=manifest['dev_gold_sha256']:
            raise ValueError('DEV scoring contract differs')
        prediction=checked_json(item['predictions_file'],score['prediction_file_sha256'])
        if prediction['version']!=item['version'] or prediction['checkpoint_sha256']!=item['checkpoint_sha256']:
            raise ValueError('DEV model identity differs')
        if sha(item['checkpoint_path'])!=item['checkpoint_sha256']:raise ValueError('Evaluated checkpoint changed')
        endpoints.append(dict(item,score=score))
    result=dict(status='DEV_only_checkpoints_frozen',at_utc=now(),protocol_sha256=manifest['protocol_sha256'],
        endpoint_manifest_sha256=sha(args.manifest),selected_checkpoints=select(endpoints),
        tie_rule='Earlier fusion epoch on exact mean full-denominator TEDS ties',
        confirmation_used_for_selection=False,benchmark_used_for_selection=False,
        full_benchmark_required_regardless_confirmation=True,no_seed_selection=True)
    with Path(args.output).open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')

if __name__=='__main__':main()
