"""Matched full-page benchmark shard; only predicted tables use the nominee.

All three outputs share one original-base layout and non-table recognition pass.
Images pass through TeleOCR read_fn, PDF conversion and the default PDF renderer.
References and ground-truth coordinates are never loaded by this worker.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import traceback


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table_replacements(blocks,replacements):
    result=copy.deepcopy(blocks)
    for index,content in replacements.items():
        if not isinstance(index,int) or not 0<=index<len(result) or result[index]['type']!='table':
            raise ValueError('Attempted replacement of a non-table block')
        if not isinstance(content,str):raise ValueError('Replacement must be generated text')
        result[index]['content']=content
    return result


def non_table_identity(value):
    """Preserve geometry/order and every non-table field in middle JSON."""
    if isinstance(value,list):return [non_table_identity(x) for x in value]
    if isinstance(value,dict):
        return {k:('[TABLE_OUTPUT]' if k=='html' and value.get('type')=='table' else non_table_identity(v)) for k,v in value.items()}
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parent;binding=json.loads((root/'STAGE_BINDING.json').read_text())
    from importlib.metadata import version
    if {name:version(name) for name in binding['runtime_packages']}!=binding['runtime_packages']:
        raise ValueError('Pinned runtime differs')
    path=root/'BENCHMARK_INPUTS.json'
    if digest(path)!=binding['benchmark_inputs_sha256']:raise ValueError('Benchmark manifest changed')
    manifest=json.loads(path.read_text());rows=manifest['pages']
    if manifest['full_denominator']!=1651 or manifest['labels_present'] is not False:raise ValueError('Full benchmark contract differs')
    allowed={'page_id','original_page_id','input_path','input_sha256'}
    if not rows or len({r['page_id'] for r in rows})!=len(rows) or any(set(r)!=allowed for r in rows):
        raise ValueError('Invalid label-free shard')
    for row in rows:
        if Path(row['page_id']).name!=row['page_id'] or Path(row['original_page_id']).name!=row['original_page_id']:
            raise ValueError('Unsafe identifier')
        if digest(row['input_path'])!=row['input_sha256']:raise ValueError('Benchmark image changed')
    if binding['nominee_seed']!=0 or not binding['nomination_sha256']:raise ValueError('DEV-only seed0 nominee required')
    if digest(binding['checkpoint_path'])!=binding['checkpoint_sha256']:raise ValueError('Nominee checkpoint changed')
    for source in binding['tele_source_paths']:sys.path.insert(0,source)
    import TeleOCR
    for name,expected in binding['tele_source_hashes'].items():
        if digest(Path(TeleOCR.__file__).parent.parent/name)!=expected:raise ValueError('TeleOCR source changed')
    if args.preflight:
        (root/'CPU_PREFLIGHT.json').write_text(json.dumps(dict(status='passed',pages=len(rows),full_denominator=1651,
            labels_present=False,nomination_sha256=binding['nomination_sha256'],gpu_allocations=0)))
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
    from TeleOCR.tools.pdf_image_tools import load_images_from_pdf,convert_pdf_bytes_to_bytes,get_page_size
    from TeleOCR.tools.enum_class import ImageType
    from TeleOCR.src.model_output_to_middle_json import blocks_to_page_info
    from TeleOCR.src.vlm_middle_json_mkcontent import union_make
    from TeleOCR.data_reader_writer import ImageDataWriter
    from versions.v5_native.features import extract as image_features
    from versions.v5_native.regions import runaway
    from versions.v7.features import extract as rule_features,simple_rule
    import TeleOCR.config as config
    if torch.cuda.device_count()!=1 or gpu_uuid_key(torch.cuda.get_device_properties(0).uuid)!=gpu_uuid_key(binding['selected_gpu']['uuid']):
        raise ValueError('GPU identity differs')
    torch.manual_seed(0);torch.set_num_threads(4)
    def timeout(signum,frame):raise TimeoutError('Benchmark model call exceeded1200 seconds')
    signal.signal(signal.SIGALRM,timeout)
    start=time.monotonic();records=[];calls=[];active_page=None;active_role=None;abort=None
    report=dict(status='running',full_denominator=1651,shard_pages=len(rows),rows=records,
        benchmark_inputs_sha256=binding['benchmark_inputs_sha256'],nomination_sha256=binding['nomination_sha256'],
        checkpoint_sha256=binding['checkpoint_sha256'],nominee_recipe=binding['nominee_recipe'],nominee_seed=0,
        labels_present=False,backend='transformers',presence_frequency_penalties_applied=False,historical_vllm_equivalence=False,
        table_input='Default200DPI predicted native helper crop; identical prepared image/tensor/prompt for base and nominee',
        non_table_policy='Original base output shared byte-for-byte before and after middle-JSON composition',
        model_calls=0,optimizer_updates=0)
    def write():
        report.update(elapsed_seconds=time.monotonic()-start,model_calls=len(calls),pages_completed=len(records))
        temp=root/'BENCHMARK.tmp';temp.write_text(json.dumps(report,indent=2));temp.replace(root/'BENCHMARK.json')
    def load_model():
        model=AutoModelForImageTextToText.from_pretrained(binding['model_root'],local_files_only=True,trust_remote_code=True,
            dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa')
        model.config.max_position_embeddings=16384;return model
    try:
        write();processor=AutoProcessor.from_pretrained(binding['model_root'],local_files_only=True,trust_remote_code=False)
        base=load_model();nominee=load_model();state=load_file(binding['checkpoint_path']);factors=None
        if binding['nominee_recipe']!='V7.3.0':
            factors={key[:-10]:(value,state[key[:-10]+'.initial_b']) for key,value in state.items() if key.endswith('.initial_a')}
        attach(nominee,factors=factors,grounding=binding['nominee_recipe']=='V7.3.2');restore(nominee,state);del state
        base.eval().requires_grad_(False);nominee.eval().requires_grad_(False)
        base_client=TeleOCRClient(backend='transformers',model=base,processor=processor,batch_size=1,use_tqdm=False)
        candidate_client=TeleOCRClient(backend='transformers',model=nominee,processor=processor,batch_size=1,use_tqdm=False)

        # Observe the actual default Transformers method, adding only BF16
        # autocast (required by the trained FP32 merger), timing and hashes.
        def instrument(client,role):
            original=client.client._predict_one_batch
            def observed(image_objs,chat_prompts,sampling_params,**kwargs):
                if len(image_objs)!=1:raise ValueError('Registered benchmark microbatch is one')
                tensors=processor(text=chat_prompts,images=image_objs,padding=True,return_tensors='pt')
                event=dict(page_id=active_page,role=active_role or role,
                    pixel_sha256=hashlib.sha256(image_objs[0].tobytes()).hexdigest(),size=list(image_objs[0].size),
                    input_hashes={k:tensor_summary(tensors[k])['sha256'] for k in ['input_ids','pixel_values','image_grid_thw']},
                    generate_kwargs=client.client.build_generate_kwargs(sampling_params),status='started')
                del tensors;calls.append(event);begin=time.monotonic();signal.alarm(1200)
                try:
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):result=original(image_objs,chat_prompts,sampling_params,**kwargs)
                    event['status']='complete';return result
                except Exception as error:event.update(status='failed',error_type=type(error).__name__);raise
                finally:
                    signal.alarm(0);event['seconds']=time.monotonic()-begin
                    with (root/'CALLS.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(event)+'\n')
            client.client._predict_one_batch=observed
        instrument(base_client,'base');instrument(candidate_client,'nominee')
        for row in rows:
            active_page=row['page_id'];folder=root/'pages'/active_page;folder.mkdir(parents=True,exist_ok=False)
            record=dict(page_id=active_page,original_page_id=row['original_page_id'],status='running',outputs={},tables=[])
            if abort:
                record.update(status='not_attempted_after_worker_failure',reason=abort)
                for policy in ['base','nominee','legacy_simple_rule']:
                    target=folder/(policy+'.md');target.write_text('');record['outputs'][policy]=dict(file=str(target),sha256=digest(target),status=record['status'],empty=True)
                records.append(record);write();continue
            (folder/'START.json').write_text(json.dumps(dict(started_unix=time.time(),input_sha256=row['input_sha256'])))
            pdf_doc=None;begin=time.monotonic();call_start=len(calls);base_blocks=None;base_middle=None
            try:
                active_role='base_layout'
                pdf=convert_pdf_bytes_to_bytes(read_fn(row['input_path']),None)
                images,pdf_doc=load_images_from_pdf(pdf,image_type=ImageType.PIL,threads=config.PDF_TOOLS_WORKER_MAX_NUM)
                if len(images)!=1:raise ValueError('Benchmark source must be exactly one page')
                image=images[0]['img_pil'];image.save(folder/'rendered.png')
                record['render']=dict(configured_dpi=200,renderer_scale=images[0]['scale'],size=list(image.size),mode=image.mode,
                    pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),png_sha256=digest(folder/'rendered.png'),pdf_page_size=list(get_page_size(pdf_doc[0])))
                # This is the upstream stepping path for one page, split at
                # its raw content boundary to reuse layout/non-table calls.
                blocks=base_client.batch_layout_detect([image])[0]
                prepared,prompts,parameters,indices=base_client.helper.prepare_for_extract(image,blocks)
                active_role='base_content';first_content=len(calls)
                outputs=base_client.client.batch_predict(prepared,prompts,parameters)
                if len(outputs)!=len(indices):raise ValueError('Native content cardinality differs')
                base_events=calls[first_content:]
                if len(base_events)!=len(indices):raise ValueError('One recorded model call per native region required')
                for index,output in zip(indices,outputs):blocks[index].content=output
                base_blocks=copy.deepcopy(blocks)
                image_writer=ImageDataWriter(str(folder/'images'))
                def compose(policy,raw):
                    processed=base_client.helper.post_process(copy.deepcopy(raw))
                    middle=blocks_to_page_info(processed,images[0],pdf_doc[0],image_writer,0)
                    if base_middle is not None and non_table_identity(middle)!=non_table_identity(base_middle):
                        raise ValueError('Non-table middle JSON changed')
                    target=folder/(policy+'.md');target.write_text(union_make([middle],'images'),encoding='utf-8')
                    (folder/(policy+'.middle.json')).write_text(json.dumps(middle,ensure_ascii=False,indent=2))
                    record['outputs'][policy]=dict(file=str(target),sha256=digest(target),status='complete',empty=target.stat().st_size==0)
                    return middle
                base_middle=compose('base',base_blocks);nominee_raw={};legacy_raw={}
                for position,index in enumerate(indices):
                    if blocks[index].type!='table':continue
                    native_image=prepared[position];active_role='nominee_table';event_index=len(calls)
                    raw=candidate_client.client.predict(native_image,prompts[position],parameters[position])
                    if calls[event_index]['input_hashes']!=base_events[position]['input_hashes'] or calls[event_index]['pixel_sha256']!=base_events[position]['pixel_sha256']:
                        raise ValueError('Base and nominee table inputs differ')
                    nominee_raw[index]=raw
                    # Use precisely the same fixed legacy candidate and rule
                    # as the registered DEV/confirmation comparator.
                    active_role='legacy_table';scale=.535299427146522
                    resized=native_image.resize((max(1,round(native_image.width*scale)),max(1,round(native_image.height*scale))),Image.Resampling.LANCZOS)
                    fixed,fp,fs,fi=base_client.helper.prepare_for_extract(resized,[ContentBlock('table',[0,0,1,1],angle=0)])
                    if fi!=[0]:raise ValueError('Legacy helper cardinality differs')
                    fixed_raw=base_client.client.predict(fixed[0],fp[0],fs[0])
                    native_html=base_client.helper.post_process([ContentBlock('table',[0,0,1,1],content=blocks[index].content)])[0].content or ''
                    fixed_html=base_client.helper.post_process([ContentBlock('table',[0,0,1,1],content=fixed_raw)])[0].content or ''
                    features=rule_features(image_features(native_image,native_html,'table'),native_html,fixed_html,bool(fixed_html and runaway(fixed_html,'table') is None))
                    accepted=simple_rule(features)
                    if accepted:legacy_raw[index]=fixed_raw
                    record['tables'].append(dict(block_index=index,native_pixel_sha256=base_events[position]['pixel_sha256'],
                        nominee_input_exact=True,nominee_changed=raw!=blocks[index].content,legacy_accepted=accepted,
                        legacy_changed=accepted and fixed_raw!=blocks[index].content))
                compose('nominee',table_replacements(base_blocks,nominee_raw))
                compose('legacy_simple_rule',table_replacements(base_blocks,legacy_raw));image_writer.save_all_images()
                record.update(status='complete',non_table_middle_exact=True,predicted_tables=sum(b.type=='table' for b in base_blocks))
            except Exception as error:
                record.update(status='failed',error_type=type(error).__name__,reason=str(error),traceback=traceback.format_exc())
                # Preserve completed base output; do not substitute it for a
                # failed direct nominee. All policies retain this page.
                for policy in ['base','nominee','legacy_simple_rule']:
                    if policy not in record['outputs']:
                        target=folder/(policy+'.md');target.write_text('');record['outputs'][policy]=dict(file=str(target),sha256=digest(target),status='failed',empty=True)
                if isinstance(error,(torch.cuda.OutOfMemoryError,TimeoutError,ValueError)):abort=type(error).__name__+': '+str(error)
                torch.cuda.empty_cache()
            finally:
                signal.alarm(0)
                if pdf_doc is not None:pdf_doc.close()
                record.update(seconds=time.monotonic()-begin,model_calls=len(calls)-call_start)
                (folder/'RECEIPT.json').write_text(json.dumps(record,indent=2));records.append(record);write()
        report.update(status='failed' if abort else 'complete',worker_abort=abort,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,reason=str(error),traceback=traceback.format_exc());raise
    finally:signal.alarm(0);write()


if __name__=='__main__':main()
