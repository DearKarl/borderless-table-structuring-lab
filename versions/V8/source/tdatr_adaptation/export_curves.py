"""Export numeric per-update history from completed hash-bound formal fits."""
import argparse,csv,json,math
from pathlib import Path
from .portable_load import digest


def main():
    parser=argparse.ArgumentParser()
    for name in ['results','download','output']:parser.add_argument('--'+name,required=True)
    args=parser.parse_args();results=json.loads(Path(args.results).read_text());download=Path(args.download)
    manifest=json.loads((download/'REMOTE_MANIFEST.json').read_text());output=Path(args.output)
    if results['status']!='complete_scores_and_descriptive_analysis' or results['protocol_sha256']!=manifest['protocol_sha256']:
        raise ValueError('Completed result analysis required')
    output.mkdir(parents=True,exist_ok=True);files={}
    basic=['step','stage','stage_step','epoch','epoch_examples','exposures','unique_tables','target_tokens','cell_count',
        'pre_clip_gradient_norm','update_seconds','peak_gpu_allocated_bytes']
    loss=['ce','total','box','iou','mask','relation','weighted_localization']
    groups=['attention_adapter','merger','localizer']
    fields=basic+['loss_'+name for name in loss]+['gradient_norm_'+name for name in groups]+['learning_rate_adapter_and_head','learning_rate_merger']
    for version,tag in [('V8.4','v84'),('V8.5','v85')]:
        relative='train-'+tag+'-001/output/SCALARS.jsonl';path=download/relative
        if digest(path)!=manifest['files'][relative]['sha256']:raise ValueError('Original scalar history changed')
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        if [row['step'] for row in rows]!=list(range(1,7501)) or rows[-1]['exposures']!=120000:raise ValueError('Complete update history required')
        target=output/(version+'.csv')
        with target.open('x',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for row in rows:
                item={name:row[name] for name in basic}
                item.update({'loss_'+name:row['losses'].get(name) for name in loss})
                item.update({'gradient_norm_'+name:row['gradient_norms'].get(name) for name in groups})
                item.update(learning_rate_adapter_and_head=row['learning_rates'][0],learning_rate_merger=row['learning_rates'][1])
                if any(isinstance(value,(int,float)) and not math.isfinite(value) for value in item.values()):raise ValueError('Nonfinite training history value')
                writer.writerow(item)
        files[target.name]=dict(sha256=digest(target),rows=7500,source_sha256=manifest['files'][relative]['sha256'])
    (output/'MANIFEST.json').write_text(json.dumps(dict(status='complete_numeric_history',result_sha256=digest(args.results),files=files,
        blank_cells='Unavailable during this stage; never imputed as zero',raw_GT_or_prediction_bodies_included=False),indent=2)+'\n')
    print(json.dumps(dict(status='complete_numeric_history',files=2,updates_per_file=7500)))


if __name__=='__main__':main()
