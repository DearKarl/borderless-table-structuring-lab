"""Versioned recovery of pre-inference callback failure; original outputs preserved."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
import signal
import sys
import time
import traceback
from .execution import checked_json,sha,write,now,verify_runtime
from .benchmark_recovery_contract import native_keyword_adapter,validate_prior
from versions.generator_pilot.large_benchmark import non_table_identity,table_replacements
from versions.generator_pilot.large_pdf_native import open_native_pdf

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding',required=True);args=ap.parse_args()
    binding=json.loads(Path(args.binding).read_text());root=Path(binding['output_root']);root.mkdir(exist_ok=False)
    verify_runtime(binding)
    manifest=checked_json(binding['benchmark_inputs'],binding['benchmark_inputs_sha256']);rows=manifest['pages']
    if manifest['full_denominator']!=1651 or len(rows)!=1651 or manifest['labels_present'] is not False:
        raise ValueError('Complete label-free benchmark required')
    if len({r['page_id'] for r in rows})!=1651:raise ValueError('Repeated page')
    for row in rows:
        if set(row)!={'page_id','original_page_id','input_path','input_sha256'}:raise ValueError('Unexpected benchmark fields')
        if any(Path(row[k]).name!=row[k] for k in ['page_id','original_page_id']):raise ValueError('Unsafe page identifier')
        if sha(row['input_path'])!=row['input_sha256']:raise ValueError('Original source image changed')
    prior_rows=validate_prior(binding,rows)
    selection=checked_json(binding['nomination_path'],binding['nomination_sha256'])
    if selection['selected_checkpoints'][binding['version']]['checkpoint_sha256']!=binding['checkpoint_sha256']:
        raise ValueError('Checkpoint was not frozen by DEV selection')
    index=checked_json(binding['shared_cache_index'],binding['shared_cache_index_sha256'])
    if index['selection_sha256']!=binding['shared_selection_sha256']:raise ValueError('Original shared cache binding differs')
    caches={r['page_id']:r for r in index['rows']}
    if len(index['rows'])!=1651 or set(caches)!={r['page_id'] for r in rows}:raise ValueError('Shared cache denominator differs')
    checked_json(binding['pdf_correction'],binding['pdf_correction_sha256'])
    for path in binding['tele_source_paths']:sys.path.insert(0,path)
    import TeleOCR
    for name,expected in binding['tele_source_hashes'].items():
        if sha(Path(TeleOCR.__file__).parent.parent/name)!=expected:raise ValueError('Pinned native processing source changed')
    import torch
    from transformers import AutoProcessor
    from TeleOCR.vlm_utils.TeleOCR_client import TeleOCRClient
    from TeleOCR.src.model_output_to_middle_json import blocks_to_page_info
    from TeleOCR.src.vlm_middle_json_mkcontent import union_make
    from TeleOCR.data_reader_writer import ImageDataWriter
    from versions.generator_pilot.validate_smoke_inputs import tensor_summary
    from .inference import load_adapted,localize_generated
    start=time.monotonic();records=[];calls=[];active_page=None;abort=None;capture=None
    report=dict(status='running',mode='model',version=binding['version'],full_denominator=1651,shard_pages=1651,rows=records,
        benchmark_inputs_sha256=binding['benchmark_inputs_sha256'],checkpoint_sha256=binding['checkpoint_sha256'],
        selection_sha256=binding['nomination_sha256'],shared_selection_sha256=binding['shared_selection_sha256'],
        shared_cache_index_sha256=binding['shared_cache_index_sha256'],labels_present=False,backend='transformers',
        presence_frequency_penalties_applied=False,historical_vllm_equivalence=False,optimizer_updates=0,
        table_input='Byte-verified original 200 DPI model-predicted native crops and default processor',
        non_table_policy='Exactly the preserved original-base content and geometry',model_run_id=binding['version'],recovery_id=binding['recovery_id'],
        recovery_from_benchmark_sha256=binding['recovery_from_benchmark_sha256'],
        recomposed_non_table_pages=1189,failed_pages_to_attempt=462,original_model_calls=0)
    def save():
        report.update(at_utc=now(),elapsed_seconds=time.monotonic()-start,model_calls=len(calls),pages_completed=len(records))
        write(root/'BENCHMARK.json',report)
    def empty(record,folder,status):
        target=folder/'candidate.md'
        if target.exists():
            partial=folder/'candidate.partial.md'
            if partial.exists():raise ValueError('Refusing to overwrite preserved partial output')
            target.rename(partial)
            record['partial_composition']=dict(file=str(partial),sha256=sha(partial),not_scored=True)
        target.write_text('')
        record['outputs']={'candidate':dict(file=str(target),sha256=sha(target),status=status,empty=True)}
    try:
        save();processor=AutoProcessor.from_pretrained(binding['model_root'],local_files_only=True,trust_remote_code=False)
        model,capture=load_adapted(binding)
        client=TeleOCRClient(backend='transformers',model=model,processor=processor,batch_size=1,use_tqdm=False)
        original=client.client._predict_one_batch
        def observed(images,prompts,parameters,**kwargs):
            if len(images)!=1:raise ValueError('Native microbatch changed')
            tensors=processor(text=prompts,images=images,padding=True,return_tensors='pt')
            event=dict(started_at_utc=now(),page_id=active_page,role='candidate_table',
                pixel_sha256=hashlib.sha256(images[0].tobytes()).hexdigest(),size=list(images[0].size),
                input_hashes={k:tensor_summary(tensors[k])['sha256'] for k in ['input_ids','pixel_values','image_grid_thw']},
                generate_kwargs=client.client.build_generate_kwargs(parameters),status='started')
            calls.append(event);begin=time.monotonic();signal.alarm(1200)
            try:
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):result=original(images,prompts,parameters,**kwargs)
                event['generation_seconds']=time.monotonic()-begin
                if len(result)!=1 or not isinstance(result[0],str):raise ValueError('Native backend output schema changed')
                if len(calls)==1:
                    first=dict(event)
                    first.update(status='generation_completed',actual_generation_completed=True,
                        generation_completed_at_utc=now(),worker_pid=__import__('os').getpid(),checkpoint_sha256=binding['checkpoint_sha256'],
                        benchmark_inputs_sha256=binding['benchmark_inputs_sha256'],selected_gpu=binding['selected_gpu'])
                    write(root/'FIRST_INFERENCE_CALL.json',first)
                try:event['localization']=localize_generated(model,processor,tensors.to(device='cuda:0',dtype=torch.bfloat16),result[0],capture)
                except ValueError as error:event['localization']=dict(status='unavailable',reason=str(error),labels_present=False)
                event['status']='complete';return result
            except Exception as error:event.update(status='failed',error_type=type(error).__name__);raise
            finally:
                signal.alarm(0);event.update(seconds=time.monotonic()-begin,finished_at_utc=now())
                with (root/'CALLS.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
        client.client._predict_one_batch=native_keyword_adapter(observed)
        for row in rows:
            active_page=row['page_id'];folder=root/'pages'/active_page;folder.mkdir(parents=True,exist_ok=False)
            record=dict(page_id=active_page,original_page_id=row['original_page_id'],input_sha256=row['input_sha256'],
                status='running',outputs={},tables=[],started_at_utc=now())
            pdf_doc=None;begin=time.monotonic();call_start=len(calls)
            try:
                if abort:raise RuntimeError('Not attempted after worker failure: '+abort)
                cached=caches[active_page]
                if cached['input_sha256']!=row['input_sha256']:raise ValueError('Shared source identity differs')
                if not cached.get('cache'):raise RuntimeError('Shared native input unavailable: '+cached.get('reason',cached['status']))
                path=Path(cached['cache']['path'])
                if sha(path)!=cached['cache']['sha256']:raise ValueError('Shared cache bytes changed')
                # Only this previously generated, hash-bound project cache is
                # deserialized. External or unbound pickle files are rejected.
                with path.open('rb') as stream:value=pickle.load(stream)
                if value['input_sha256']!=row['input_sha256'] or value['selection_sha256']!=binding['shared_selection_sha256']:
                    raise ValueError('Cached object provenance differs')
                base_blocks=value['base_blocks'];base_middle=value['base_middle'];replacements={}
                pdf_doc=open_native_pdf(value['pdf']);writer=ImageDataWriter(str(folder/'images'))
                if value['indices'] and prior_rows[active_page]['status']!='failed':
                    raise ValueError('Refusing to repeat a previously completed model page')
                for position,index_ in enumerate(value['indices']):
                    if base_blocks[index_].type!='table':raise ValueError('Shared specialist cache contains non-table input')
                    before=len(calls)
                    raw=client.client.predict(value['prepared'][position],value['prompts'][position],value['parameters'][position])
                    event=calls[before];base_event=value['base_events'][position]
                    if any(event[k]!=base_event[k] for k in ['input_hashes','pixel_sha256','generate_kwargs']):
                        raise ValueError('Candidate and original-base native input or decoder differs')
                    replacements[index_]=raw
                    record['tables'].append(dict(block_index=index_,native_input_exact=True,changed=raw!=base_blocks[index_].content,
                        localization_status=event['localization']['status']))
                    (folder/'RAW_TABLE_OUTPUTS.json').write_text(json.dumps(replacements,ensure_ascii=False))
                processed=client.helper.post_process(copy.deepcopy(table_replacements(base_blocks,replacements)))
                middle=blocks_to_page_info(processed,value['image_info'],pdf_doc[0],writer,0)
                if non_table_identity(middle)!=non_table_identity(base_middle):raise ValueError('Non-table output or geometry changed')
                target=folder/'candidate.md';target.write_text(union_make([middle],'images'),encoding='utf-8')
                (folder/'candidate.middle.json').write_text(json.dumps(middle,ensure_ascii=False,indent=2));writer.save_all_images()
                record.update(status='complete',non_table_middle_exact=True,predicted_tables=len(value['indices']),shared_cache=cached['cache'],
                    outputs={'candidate':dict(file=str(target),sha256=sha(target),status='complete',empty=target.stat().st_size==0)})
            except Exception as error:
                record.update(status='not_attempted_after_worker_failure' if abort else 'failed',error_type=type(error).__name__,reason=str(error),traceback=traceback.format_exc())
                empty(record,folder,record['status'])
                if isinstance(error,(torch.cuda.OutOfMemoryError,TimeoutError,ValueError,AttributeError,TypeError)):abort=type(error).__name__+': '+str(error)
                torch.cuda.empty_cache()
            finally:
                signal.alarm(0)
                if pdf_doc is not None:pdf_doc.close()
                record.update(seconds=time.monotonic()-begin,model_calls=len(calls)-call_start,finished_at_utc=now())
                write(folder/'RECEIPT.json',record);records.append(record);save()
        report.update(status='failed' if abort or any(r['status']!='complete' for r in records) else 'complete',
            worker_abort=abort,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated());save()
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,reason=str(error),traceback=traceback.format_exc())
        recorded={r['page_id'] for r in records}
        for row in rows:
            if row['page_id'] in recorded:continue
            folder=root/'pages'/row['page_id'];folder.mkdir(parents=True,exist_ok=True)
            record=dict(page_id=row['page_id'],original_page_id=row['original_page_id'],input_sha256=row['input_sha256'],
                status='not_attempted_after_initialization_failure',reason=str(error),outputs={},tables=[])
            empty(record,folder,record['status']);records.append(record)
        save();raise
    finally:
        signal.alarm(0)
        if capture is not None:capture.close()

if __name__=='__main__':main()
