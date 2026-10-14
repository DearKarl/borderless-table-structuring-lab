"""CPU-only full-cohort tokenizer audit and source-bound detail-task database."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import traceback


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--binding', required=True); args=ap.parse_args()
    binding=json.loads(Path(args.binding).read_text()); root=Path(binding['output_root'])
    root.mkdir(exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    import torch
    from transformers import AutoProcessor
    from safetensors import safe_open
    from .cell_tokens import full_cell_token_ranges
    from .inputs import encode_text, NATIVE_PROMPT, ids_blob
    from .targets import TASKS, render, verified_lines
    from versions.generator_pilot.targets import serialize
    torch.set_num_threads(4)
    assert sha(binding['common_inputs']) == binding['common_inputs_sha256']
    inputs=json.loads(Path(binding['common_inputs']).read_text())
    assert len(inputs['train'])==20000 and len({r['source'] for r in inputs['train']})==11217
    assert len(inputs['dev'])==256
    rows=inputs['train']+inputs['dev']; by_id={r['id']:r for r in rows}
    assert len(by_id)==20256
    assert not ({r['source'] for r in inputs['train']} & {r['source'] for r in inputs['dev']})
    annotations={}
    for name in ['lines','geometry']:
        path=Path(binding[name+'_path']); assert sha(path)==binding[name+'_sha256']
        records={}
        with path.open() as stream:
            for line in stream:
                row=json.loads(line)
                if row['id'] in records or row['id'] not in by_id: raise ValueError('Annotation membership mismatch')
                records[row['id']]=row
        annotations[name]=records
    processor=AutoProcessor.from_pretrained(binding['model_root'], local_files_only=True, trust_remote_code=False)
    database=root/'DETAIL_INPUTS.sqlite'
    db=None
    report=dict(status='running', completed=0, expected=len(rows), native_cache_exact=0, full_cell_ranges_verified=0,
        eligible=Counter(), unavailable=Counter(), maximum_tokens={}, source_manifest_hashes={},
        task_sequence_lengths=defaultdict(list), adjacent_shared_tokens=0,
        GPU_allocations=0, formal_optimizer_updates=0, cohort_changed=False)
    if binding.get('recovery'):
        from .audit_recovery import recover
        db,report=recover(binding,rows,root)
    else:
        db=sqlite3.connect(database)
        db.execute('CREATE TABLE examples(id TEXT PRIMARY KEY, partition TEXT, record TEXT, cell_ranges TEXT, logical_spans TEXT, eligible_tasks TEXT)')
        db.execute('CREATE TABLE tasks(id TEXT, task TEXT, input_ids BLOB, target_start INT, eos_position INT, sequence_tokens INT, target_tokens INT, prompt_sha256 TEXT, target_sha256 TEXT, PRIMARY KEY(id,task))')
    start_index=report['completed']
    begin=time.monotonic()
    def save():
        report.update(at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-begin)
        summary={k:v for k,v in report.items() if k!='task_sequence_lengths'}
        temporary=root/'AUDIT.tmp'; temporary.write_text(json.dumps(summary,indent=2)); temporary.replace(root/'AUDIT.json')
    @lru_cache(maxsize=2)
    def source_rows(stage):
        stage=Path(stage); manifest=stage/'LABEL_MANIFEST.json'
        report['source_manifest_hashes'][str(manifest)]=sha(manifest)
        return {r['id']:r for r in json.loads(manifest.read_text())['rows']}
    try:
        save()
        for index, row in enumerate(rows):
            if index<start_index:continue
            if row['status']!='input_eligible':
                report['unavailable'][row['partition']+':missing_native_input']+=1
                report['completed']=index+1
                continue
            cache=Path(row['cache_path']); assert sha(cache)==row['cache_sha256']
            original=source_rows(str(cache.parent.parent))[row['id']]
            canonical=original['canonical']; table=canonical['table']; target=canonical['target_otsl']
            if serialize(table)!=target or hashlib.sha256(target.encode()).hexdigest()!=row['target_sha256']:
                raise ValueError('Frozen canonical table target mismatch')
            with safe_open(cache, framework='pt', device='cpu') as sf:
                native_ids=sf.get_tensor('input_ids'); native_labels=sf.get_tensor('labels')
                grid=sf.get_tensor('image_grid_thw'); native_mask=sf.get_tensor('attention_mask')
            if tuple(grid.shape)!=(1,3) or int(grid[0,0])!=1: raise ValueError('Expected one native image')
            image_tokens=int(grid.prod())//processor.image_processor.merge_size**2
            base=encode_text(processor,image_tokens,NATIVE_PROMPT,target)
            if (native_ids.tolist()!=[base['input_ids']] or native_labels.tolist()!=[base['labels']] or
                native_mask.tolist()!=[[1]*len(base['input_ids'])]):
                raise ValueError('Native text reconstruction differs from original default-processor cache')
            report['native_cache_exact']+=1
            ranges=full_cell_token_ranges(target,base['offsets'],sequence_start=base['counts']['target_start'])
            if len(ranges['cells'])!=len(table['cells']): raise ValueError('Native cell correspondence mismatch')
            report['full_cell_ranges_verified']+=1
            report['adjacent_shared_tokens']+=ranges['adjacent_shared_tokens']
            spans=[[c['row'],c['row']+c['rowspan']-1,c['col'],c['col']+c['colspan']-1] for c in table['cells']]
            line_annotation=annotations['lines'][row['id']]; geometry=annotations['geometry'][row['id']]
            complete_lines=line_annotation.get('status')=='processed' and line_annotation.get('full_table_text_verified')
            boxes=row['normalized_boxes']; query=None
            if complete_lines:
                lines=verified_lines(table,line_annotation)
                # The first valid source cell region is deterministic. A query
                # must include some but not all text lines and cut no line.
                for candidate in boxes:
                    inside=[x for x in lines if x['box'][0]>=candidate[0] and x['box'][1]>=candidate[1] and x['box'][2]<=candidate[2] and x['box'][3]<=candidate[3]]
                    partial=[x for x in lines if min(x['box'][2],candidate[2])>max(x['box'][0],candidate[0]) and min(x['box'][3],candidate[3])>max(x['box'][1],candidate[1]) and x not in inside]
                    if inside and len(inside)<len(lines) and not partial:
                        query=candidate; break
            eligible=[]
            for task in TASKS:
                key=row['partition']+':'+task
                reason=None
                if task in TASKS[:4] and not complete_lines: reason='source_lines_unavailable'
                elif task.startswith('box_') and query is None: reason='no_complete_proper_subset_query'
                elif task=='row_column_detection' and geometry['status']!='verified': reason='row_column_geometry_unavailable'
                if reason:
                    report['unavailable'][key+':'+reason]+=1; continue
                prompt,answer=render(task,table,boxes,line_annotation=line_annotation,query_box=query,
                    row_boxes=geometry.get('row_boxes'),column_boxes=geometry.get('column_boxes'))
                try: encoded=encode_text(processor,image_tokens,prompt,answer)
                except ValueError as error:
                    if str(error)=='Sequence exceeds the frozen token ceiling; truncation forbidden':
                        report['unavailable'][key+':token_limit']+=1; continue
                    raise
                counts=encoded['counts']
                db.execute('INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?)',(row['id'],task,ids_blob(encoded['input_ids']),
                    counts['target_start'],counts['eos_position'],counts['sequence_tokens'],counts['target_tokens'],
                    encoded['prompt_sha256'],encoded['target_sha256']))
                eligible.append(task);report['eligible'][key]+=1
                report['task_sequence_lengths'][key].append(counts['sequence_tokens'])
                report['maximum_tokens'][key]=max(report['maximum_tokens'].get(key,0),counts['sequence_tokens'])
            if not eligible: raise ValueError('Original cohort member has no feasible detail task')
            db.execute('INSERT INTO examples VALUES(?,?,?,?,?,?)',(row['id'],row['partition'],json.dumps(row),
                json.dumps([c['token_range'] for c in ranges['cells']]),json.dumps(spans),json.dumps(eligible)))
            report['completed']=index+1
            if (index+1)%100==0: db.commit();save()
        db.commit();db.close()
        if report['native_cache_exact']!=20244: raise ValueError('Native input denominator differs')
        report.update(status='passed',detail_database_sha256=sha(database),detail_database_bytes=database.stat().st_size,
            sequence_length_quantiles={k:{str(q):sorted(v)[round((len(v)-1)*q)] for q in [.5,.9,.99,1]} for k,v in report['task_sequence_lengths'].items()})
        save()
    except Exception as error:
        db.commit();db.close()
        report.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        save();raise


if __name__=='__main__':main()
