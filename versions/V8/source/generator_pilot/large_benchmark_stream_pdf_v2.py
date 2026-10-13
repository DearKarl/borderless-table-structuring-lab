"""Versioned streaming consumer of the original frozen six-model benchmark.

Shared preprocessing, base output and legacy output are generated once. Trusted,
locally produced cache objects are hash-verified before deserialization. Each
selected model receives identical table pixels, tensors, prompts and decoding
parameters; all non-table content and geometry are shared with the base output.
"""
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
from datetime import datetime,timezone

from large_benchmark import digest, non_table_identity, table_replacements
from large_pdf_native import open_native_pdf
from large_failed_prefix import load_prior, preserve_failure
from large_stream_cache import StreamingCache


def read(path):
    return json.loads(Path(path).read_text())


def main(mode):
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parent;binding=read(root/'STAGE_BINDING.json')
    from importlib.metadata import version
    if {name:version(name) for name in binding['runtime_packages']}!=binding['runtime_packages']:
        raise ValueError('Pinned runtime differs')
    if digest(root/'BENCHMARK_INPUTS.json')!=binding['benchmark_inputs_sha256']:raise ValueError('Input manifest changed')
    manifest=read(root/'BENCHMARK_INPUTS.json');rows=manifest['pages']
    if manifest['full_denominator']!=1651 or manifest['labels_present'] is not False:raise ValueError('Benchmark contract differs')
    if not rows or len({r['page_id'] for r in rows})!=len(rows):raise ValueError('Duplicate or empty membership')
    if mode=='model' and len(rows)!=1651:raise ValueError('Every selected model requires all1651 pages')
    for row in rows:
        if set(row)!={'page_id','original_page_id','input_path','input_sha256'}:raise ValueError('Unexpected label-free fields')
        if any(Path(row[k]).name!=row[k] for k in ['page_id','original_page_id']):raise ValueError('Unsafe identifier')
        if digest(row['input_path'])!=row['input_sha256']:raise ValueError('Source image changed')
    for source in binding['tele_source_paths']:sys.path.insert(0,source)
    import TeleOCR
    for name,expected in binding['tele_source_hashes'].items():
        if digest(Path(TeleOCR.__file__).parent.parent/name)!=expected:raise ValueError('TeleOCR source changed')
    if (mode!='model' or binding.get('preferred_stream_gpu') not in [5,6]
        or binding['eligible_physical_gpu_indices']!=[1,2,3,binding['preferred_stream_gpu']]
        or not binding['stream_amendment_sha256']):raise ValueError('Streaming GPU5/6 and sealed-input fallback binding missing')
    if digest(binding['checkpoint_path'])!=binding['checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
    if digest(root/'STREAM_EXTENSION_FROZEN.json')!=binding['stream_extension_sha256']:raise ValueError('Streaming extension binding changed')
    caches=StreamingCache(root,binding,rows)
    if digest(root/'PDF_CORRECTION.json')!=binding['pdf_correction_sha256']:raise ValueError('PDF correction binding changed')
    prior_rows,prior_calls=load_prior(root,binding,rows)
    if args.preflight:
        caches.record_consumption=False
        first_shard=caches.shards[rows[0]['page_id']]
        if not (Path(first_shard['root'])/'EXECUTION.json').exists():raise ValueError('First shared shard must be terminal before early model staging')
        caches.terminal_shard(first_shard)
        (root/'CPU_PREFLIGHT.json').write_text(json.dumps(dict(status='passed',pages=len(rows),full_denominator=1651,
            labels_present=False,selection_sha256=binding['selection_sha256'],gpu_allocations=0,mode=mode,
            first_terminal_shard_cache_hashes_verified=True,stream_dependencies_sha256=binding['stream_dependencies_sha256'])))
        return
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText,AutoProcessor
    from safetensors.torch import load_file
    from large_adapters import attach,restore
    from supervision import gpu_uuid_key
    from validate_smoke_inputs import tensor_summary
    from TeleOCR.vlm_utils.TeleOCR_client import TeleOCRClient
    from TeleOCR.vlm_utils.structs import ContentBlock
    from TeleOCR.tools.read_file import read_fn
    from TeleOCR.tools.pdf_image_tools import load_images_from_pdf,convert_pdf_bytes_to_bytes
    from TeleOCR.tools.enum_class import ImageType
    from TeleOCR.src.model_output_to_middle_json import blocks_to_page_info
    from TeleOCR.src.vlm_middle_json_mkcontent import union_make
    from TeleOCR.data_reader_writer import ImageDataWriter
    from versions.v5_native.features import extract as image_features
    from versions.v5_native.regions import runaway
    from versions.v7.features import extract as rule_features,simple_rule
    import TeleOCR.config as config
    if torch.cuda.device_count()!=1 or gpu_uuid_key(torch.cuda.get_device_properties(0).uuid)!=gpu_uuid_key(binding['selected_gpu']['uuid']):
        raise ValueError('Physical GPU identity differs')
    torch.manual_seed(0);torch.set_num_threads(4)
    def timeout(signum,frame):raise TimeoutError('Benchmark model call exceeded1200 seconds')
    signal.signal(signal.SIGALRM,timeout)
    policies=['base','legacy_simple_rule'] if mode=='shared' else ['candidate']
    start=time.monotonic();records=[];calls=[];active_page=None;active_role=None;abort=None
    report=dict(status='running',full_denominator=1651,shard_pages=len(rows),rows=records,mode=mode,
        benchmark_inputs_sha256=binding['benchmark_inputs_sha256'],selection_sha256=binding['selection_sha256'],
        model_run_id=binding.get('model_run_id'),checkpoint_sha256=binding.get('checkpoint_sha256'),
        shared_cache_index_sha256=None,stream_dependencies_sha256=binding['stream_dependencies_sha256'],stream_amendment_sha256=binding['stream_amendment_sha256'],labels_present=False,backend='transformers',
        presence_frequency_penalties_applied=False,historical_vllm_equivalence=False,optimizer_updates=0,
        table_input='Cached default200DPI predicted native helper inputs; exact base pixel/tensor/prompt checks',
        non_table_policy='Same original-base output and geometry for all six models and legacy control')
    def write():
        report.update(elapsed_seconds=time.monotonic()-start,model_calls=len(calls)+prior_calls,new_model_calls=len(calls),inherited_model_calls=prior_calls,inherited_attempted_failures=len(prior_rows),pages_completed=len(records))
        temp=root/'BENCHMARK.tmp';temp.write_text(json.dumps(report,indent=2));temp.replace(root/'BENCHMARK.json')
    def dependency_wait(reason):
        report['dependency_wait']=reason;write()
    caches.wait_callback=dependency_wait
    def empty(record,folder,status):
        for policy in policies:
            if policy in record['outputs']:continue
            target=folder/(policy+'.md');target.write_text('')
            record['outputs'][policy]=dict(file=str(target),sha256=digest(target),status=status,empty=True)
    try:
        write();processor=AutoProcessor.from_pretrained(binding['model_root'],local_files_only=True,trust_remote_code=False)
        model=AutoModelForImageTextToText.from_pretrained(binding['model_root'],local_files_only=True,trust_remote_code=True,
            dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa')
        model.config.max_position_embeddings=16384
        if mode=='model':
            state=load_file(binding['checkpoint_path']);factors=None
            if binding['model_recipe']!='V7.3.0':
                factors={key[:-10]:(value,state[key[:-10]+'.initial_b']) for key,value in state.items() if key.endswith('.initial_a')}
            attach(model,factors=factors,grounding=binding['model_recipe']=='V7.3.2');restore(model,state);del state
        model.eval().requires_grad_(False)
        client=TeleOCRClient(backend='transformers',model=model,processor=processor,batch_size=1,use_tqdm=False)
        original=client.client._predict_one_batch
        def observed(image_objs,chat_prompts,sampling_params,**kwargs):
            if len(image_objs)!=1:raise ValueError('Registered microbatch is one')
            tensors=processor(text=chat_prompts,images=image_objs,padding=True,return_tensors='pt')
            event=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),page_id=active_page,role=active_role,pixel_sha256=hashlib.sha256(image_objs[0].tobytes()).hexdigest(),
                size=list(image_objs[0].size),input_hashes={k:tensor_summary(tensors[k])['sha256'] for k in ['input_ids','pixel_values','image_grid_thw']},
                generate_kwargs=client.client.build_generate_kwargs(sampling_params),status='started')
            del tensors;calls.append(event);begin=time.monotonic();signal.alarm(1200)
            try:
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):result=original(image_objs,chat_prompts,sampling_params,**kwargs)
                event['status']='complete';return result
            except Exception as error:event.update(status='failed',error_type=type(error).__name__);raise
            finally:
                signal.alarm(0);event['seconds']=time.monotonic()-begin;event['finished_at_utc']=datetime.now(timezone.utc).isoformat()
                with (root/'CALLS.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
        client.client._predict_one_batch=observed
        for row in rows:
            active_page=row['page_id'];folder=root/'pages'/active_page;folder.mkdir(parents=True,exist_ok=False)
            if active_page in prior_rows:
                records.append(preserve_failure(prior_rows[active_page],folder));write();continue
            record=dict(page_id=active_page,original_page_id=row['original_page_id'],input_sha256=row['input_sha256'],status='running',outputs={},tables=[],started_at_utc=datetime.now(timezone.utc).isoformat())
            pdf_doc=None;begin=time.monotonic();call_start=len(calls);base_middle=None
            try:
                if abort:raise RuntimeError('Not attempted after worker failure: '+abort)
                if mode=='shared':
                    active_role='base_layout';pdf=convert_pdf_bytes_to_bytes(read_fn(row['input_path']),None)
                    images,pdf_doc=load_images_from_pdf(pdf,image_type=ImageType.PIL,threads=config.PDF_TOOLS_WORKER_MAX_NUM)
                    if len(images)!=1:raise ValueError('Benchmark source must contain one page')
                    image_info=images[0];image=image_info['img_pil']
                    blocks=client.batch_layout_detect([image])[0]
                    prepared,prompts,parameters,indices=client.helper.prepare_for_extract(image,blocks)
                    active_role='base_content';first_content=len(calls)
                    outputs=client.client.batch_predict(prepared,prompts,parameters);base_events=calls[first_content:]
                    if len(outputs)!=len(indices) or len(base_events)!=len(indices):raise ValueError('Native content cardinality differs')
                    for index,output in zip(indices,outputs):blocks[index].content=output
                    base_blocks=copy.deepcopy(blocks)
                else:
                    cached=caches[active_page]
                    if cached['input_sha256']!=row['input_sha256']:raise ValueError('Cache source identity differs')
                    if not cached.get('cache'):raise RuntimeError('Shared native input unavailable: '+cached.get('reason',cached['status']))
                    cache_path=Path(cached['cache']['path'])
                    if digest(cache_path)!=cached['cache']['sha256']:raise ValueError('Cache changed after CPU gate')
                    # Only deserialize our hash-bound, locally generated trusted cache.
                    with cache_path.open('rb') as stream:value=pickle.load(stream)
                    if value['input_sha256']!=row['input_sha256'] or value['selection_sha256']!=binding['selection_sha256']:
                        raise ValueError('Cached object provenance differs')
                    image_info=value['image_info'];base_blocks=value['base_blocks'];base_middle=value['base_middle']
                    prepared=value['prepared'];prompts=value['prompts'];parameters=value['parameters'];indices=value['indices'];base_events=value['base_events']
                    pdf_doc=open_native_pdf(value['pdf'])
                image_writer=ImageDataWriter(str(folder/'images'))
                def compose(policy,raw):
                    processed=client.helper.post_process(copy.deepcopy(raw))
                    middle=blocks_to_page_info(processed,image_info,pdf_doc[0],image_writer,0)
                    if base_middle is not None and non_table_identity(middle)!=non_table_identity(base_middle):raise ValueError('Non-table output or geometry changed')
                    target=folder/(policy+'.md');target.write_text(union_make([middle],'images'),encoding='utf-8')
                    (folder/(policy+'.middle.json')).write_text(json.dumps(middle,ensure_ascii=False,indent=2))
                    record['outputs'][policy]=dict(file=str(target),sha256=digest(target),status='complete',empty=target.stat().st_size==0)
                    return middle
                if mode=='shared':
                    base_middle=compose('base',base_blocks)
                    positions=[p for p,i in enumerate(indices) if base_blocks[i].type=='table']
                    cache_path=folder/'native-cache.pkl'
                    value=dict(input_sha256=row['input_sha256'],selection_sha256=binding['selection_sha256'],pdf=pdf,
                        image_info=image_info,base_blocks=base_blocks,base_middle=base_middle,
                        prepared=[prepared[p] for p in positions],prompts=[prompts[p] for p in positions],
                        parameters=[parameters[p] for p in positions],indices=[indices[p] for p in positions],base_events=[base_events[p] for p in positions])
                    with cache_path.open('xb') as stream:pickle.dump(value,stream,protocol=4)
                    record['cache']=dict(path=str(cache_path),sha256=digest(cache_path),bytes=cache_path.stat().st_size)
                replacements={}
                for position,index in enumerate(indices):
                    if base_blocks[index].type!='table':continue
                    native_image=prepared[position]
                    if mode=='model':
                        active_role='candidate_table';event_index=len(calls)
                        raw=client.client.predict(native_image,prompts[position],parameters[position]);event=calls[event_index]
                        if any(event[k]!=base_events[position][k] for k in ['input_hashes','pixel_sha256','generate_kwargs']):
                            raise ValueError('Candidate and original-base native input or decoder differs')
                        replacements[index]=raw
                        record['tables'].append(dict(block_index=index,native_input_exact=True,changed=raw!=base_blocks[index].content))
                        (folder/'RAW_TABLE_OUTPUTS.json').write_text(json.dumps(replacements,ensure_ascii=False),encoding='utf-8')
                    else:
                        active_role='legacy_table';scale=.535299427146522
                        resized=native_image.resize((max(1,round(native_image.width*scale)),max(1,round(native_image.height*scale))),Image.Resampling.LANCZOS)
                        fixed,fp,fs,fi=client.helper.prepare_for_extract(resized,[ContentBlock('table',[0,0,1,1],angle=0)])
                        if fi!=[0]:raise ValueError('Legacy helper cardinality differs')
                        raw=client.client.predict(fixed[0],fp[0],fs[0])
                        native_html=client.helper.post_process([ContentBlock('table',[0,0,1,1],content=base_blocks[index].content)])[0].content or ''
                        fixed_html=client.helper.post_process([ContentBlock('table',[0,0,1,1],content=raw)])[0].content or ''
                        features=rule_features(image_features(native_image,native_html,'table'),native_html,fixed_html,bool(fixed_html and runaway(fixed_html,'table') is None))
                        accepted=simple_rule(features)
                        if accepted:replacements[index]=raw
                        record['tables'].append(dict(block_index=index,legacy_accepted=accepted,legacy_changed=accepted and raw!=base_blocks[index].content))
                compose('legacy_simple_rule' if mode=='shared' else 'candidate',table_replacements(base_blocks,replacements));image_writer.save_all_images()
                record.update(status='complete',non_table_middle_exact=True,predicted_tables=sum(b.type=='table' for b in base_blocks))
                if mode=='model':record['shared_cache']=cached['cache']
            except Exception as error:
                record.update(status='not_attempted_after_worker_failure' if abort else 'failed',error_type=type(error).__name__,reason=str(error),traceback=traceback.format_exc())
                empty(record,folder,record['status'])
                if isinstance(error,(torch.cuda.OutOfMemoryError,TimeoutError,ValueError,AttributeError)):abort=type(error).__name__+': '+str(error)
                torch.cuda.empty_cache()
            finally:
                signal.alarm(0)
                if pdf_doc is not None:pdf_doc.close()
                record.update(seconds=time.monotonic()-begin,model_calls=len(calls)-call_start,finished_at_utc=datetime.now(timezone.utc).isoformat())
                (folder/'RECEIPT.json').write_text(json.dumps(record,indent=2));records.append(record);write()
        if not abort:report['shared_cache_index_sha256']=caches.finalize()
        report.update(status='failed' if abort or prior_rows else 'complete',worker_abort=abort,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,reason=str(error),traceback=traceback.format_exc())
        recorded={r['page_id'] for r in records}
        for row in rows:
            if row['page_id'] in recorded:continue
            folder=root/'pages'/row['page_id'];folder.mkdir(parents=True,exist_ok=True)
            if row['page_id'] in prior_rows:
                records.append(preserve_failure(prior_rows[row['page_id']],folder));continue
            record=dict(page_id=row['page_id'],original_page_id=row['original_page_id'],input_sha256=row['input_sha256'],
                status='not_attempted_after_initialization_failure',reason=str(error),outputs={},tables=[])
            empty(record,folder,record['status']);records.append(record)
    finally:signal.alarm(0);write()
