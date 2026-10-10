"""Label-free, default Transformers generation for one frozen model/cohort."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import traceback


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parent;binding=json.loads((root/'STAGE_BINDING.json').read_text())
    from importlib.metadata import version
    observed_runtime={name:version(name) for name in binding['runtime_packages']}
    if observed_runtime!=binding['runtime_packages']:raise ValueError('Pinned runtime package versions differ')
    inputs_path=root/'INFERENCE_INPUTS.json'
    if digest(inputs_path)!=binding['inference_inputs_sha256']:raise ValueError('Inference manifest changed')
    manifest=json.loads(inputs_path.read_text());rows=manifest['rows']
    if len(rows)!=manifest['denominator'] or len({r['id'] for r in rows})!=len(rows):raise ValueError('Cohort denominator differs')
    if manifest['cohort'] not in ['dev','confirmation']:raise ValueError('Unregistered source cohort')
    if len(rows)!={'dev':256,'confirmation':512}[manifest['cohort']]:raise ValueError('Cohort size differs')
    allowed={'id','source','status','reason','native_image_path','native_image_sha256','native_pixel_sha256','processor_tensor_hashes','prompt_ids_sha256'}
    if any(set(row)-allowed for row in rows):raise ValueError('Inference schema contains unregistered fields')
    for row in rows:
        if row['status']=='input_eligible' and digest(row['native_image_path'])!=row['native_image_sha256']:
            raise ValueError('Native crop changed')
    if binding.get('checkpoint_path') and digest(binding['checkpoint_path'])!=binding['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    for path in binding['tele_source_paths']:sys.path.insert(0,path)
    import TeleOCR
    for name,expected in binding['tele_source_hashes'].items():
        if digest(Path(TeleOCR.__file__).parent.parent/name)!=expected:raise ValueError('TeleOCR source changed')
    if args.preflight:
        (root/'CPU_PREFLIGHT.json').write_text(json.dumps(dict(status='passed',cohort=manifest['cohort'],denominator=len(rows),
            missing_inputs=sum(r['status']!='input_eligible' for r in rows),labels_present=False,gpu_allocations=0)))
        return
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText,AutoProcessor
    from safetensors.torch import load_file
    from TeleOCR.vlm_utils.TeleOCR_client import TeleOCRClient
    from TeleOCR.vlm_utils.structs import ContentBlock
    from TeleOCR.vlm_utils.post_process.otsl2html import convert_otsl_to_html
    from supervision import gpu_uuid_key
    from validate_smoke_inputs import tensor_summary
    from large_adapters import attach,restore
    if torch.cuda.device_count()!=1 or gpu_uuid_key(torch.cuda.get_device_properties(0).uuid)!=gpu_uuid_key(binding['selected_gpu']['uuid']):
        raise ValueError('GPU identity differs')
    torch.manual_seed(0);torch.set_num_threads(4)
    def timeout_handler(signum,frame):raise TimeoutError('Generation exceeded1200 seconds')
    signal.signal(signal.SIGALRM,timeout_handler)
    start=time.monotonic();records=[];calls=0;abort=None
    report=dict(status='running',run_id=binding['run_id'],cohort=manifest['cohort'],recipe=binding['recipe'],
        seed=binding.get('seed'),epoch=binding.get('epoch'),denominator=len(rows),rows=records,
        backend='transformers',presence_frequency_penalties_applied=False,historical_vllm_equivalence=False,
        labels_present=False,optimizer_updates=0,inference_inputs_sha256=binding['inference_inputs_sha256'],checkpoint_sha256=binding.get('checkpoint_sha256'))
    report.update(execution_protocol_sha256=binding['execution_protocol_sha256'],nomination_sha256=binding.get('nomination_sha256'))
    def write():
        report.update(elapsed_seconds=time.monotonic()-start,model_calls=calls)
        temporary=root/'PREDICTIONS.tmp';temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2));temporary.replace(root/'PREDICTIONS.json')
    try:
        write()
        model=AutoModelForImageTextToText.from_pretrained(binding['model_root'],local_files_only=True,trust_remote_code=True,
            dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa')
        if binding.get('checkpoint_path'):
            state=load_file(binding['checkpoint_path']);factors=None
            if binding['recipe']!='V7.3.0':
                factors={key[:-10]:(value,state[key[:-10]+'.initial_b']) for key,value in state.items() if key.endswith('.initial_a')}
            attach(model,factors=factors,grounding=binding['recipe']=='V7.3.2');restore(model,state);del state
        elif binding['recipe']!='base':raise ValueError('Non-base model missing checkpoint')
        model.eval();model.requires_grad_(False);model.config.max_position_embeddings=16384
        processor=AutoProcessor.from_pretrained(binding['model_root'],local_files_only=True,trust_remote_code=False)
        client=TeleOCRClient(backend='transformers',model=model,processor=processor,batch_size=1,use_tqdm=False)

        def recognize(image,row,*,original):
            nonlocal calls
            begin=time.monotonic();signal.alarm(1200)
            try:
                prepared,prompts,parameters,indices=client.helper.prepare_for_extract(image,[ContentBlock('table',[0,0,1,1],angle=0)])
                if indices!=[0]:raise ValueError('Default helper cardinality differs')
                if original and (prepared[0].size!=image.size or prepared[0].tobytes()!=image.tobytes()):
                    raise ValueError('Repeated native helper changed frozen crop')
                prompt=processor.apply_chat_template(client.client.build_messages(prompts[0]),tokenize=False,add_generation_prompt=True)
                values=processor(text=[prompt],images=prepared,padding=True,return_tensors='pt')
                tensors={k:tensor_summary(values[k]) for k in ['pixel_values','image_grid_thw']}
                if original:
                    if any(tensors[k]['sha256']!=row['processor_tensor_hashes'][k] for k in tensors):raise ValueError('Inference processor differs from frozen training path')
                    if tensor_summary(values['input_ids'])['sha256']!=row['prompt_ids_sha256']:raise ValueError('Inference prompt differs')
                prompt_tokens=int(values['input_ids'].shape[1])
                if prompt_tokens>=16384:raise ValueError('No generation context remains')
                kwargs=client.client.build_generate_kwargs(parameters[0]);values=values.to(device='cuda:0',dtype=torch.bfloat16)
                calls+=1
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):output=model.generate(**values,use_cache=True,**kwargs)
                ids=output[0,prompt_tokens:].cpu().tolist();raw=processor.decode([i for i in ids if i not in client.client.skip_token_ids],
                    skip_special_tokens=False,clean_up_tokenization_spaces=False)
                try:html=convert_otsl_to_html(raw);error=None
                except Exception as exc:html='';error=type(exc).__name__+': '+str(exc)
                return dict(raw_otsl=raw,decoded_html=html,decode_error=error,eos_reached=bool(ids and ids[-1]==processor.tokenizer.eos_token_id),
                    generated_tokens=len(ids),prompt_tokens=prompt_tokens,generate_kwargs=kwargs,actual_tensors=tensors,
                    input_size=list(image.size),helper_size=list(prepared[0].size),seconds=time.monotonic()-begin)
            finally:signal.alarm(0)

        for row in rows:
            record=dict(id=row['id'],source=row['source'],status='pending',model_calls=0)
            if row['status']!='input_eligible':record.update(status='missing_input',reason=row.get('reason'))
            elif abort:record.update(status='not_attempted_after_worker_failure',reason=abort)
            else:
                before=calls
                try:
                    with Image.open(row['native_image_path']) as image:image.load();image=image.copy()
                    record['prediction']=recognize(image,row,original=True)
                    if binding.get('include_legacy_candidate'):
                        if binding['recipe']!='base':raise ValueError('Legacy reread requires original base')
                        scale=.535299427146522
                        resized=image.resize((max(1,round(image.width*scale)),max(1,round(image.height*scale))),Image.Resampling.LANCZOS)
                        record['fixed_candidate']=recognize(resized,row,original=False)
                        from versions.v5_native.features import extract as image_features
                        from versions.v5_native.regions import runaway
                        from versions.v7.features import extract as rule_features,simple_rule
                        native=record['prediction']['decoded_html'];candidate=record['fixed_candidate']['decoded_html']
                        values=image_features(image,native,'table')
                        features=rule_features(values,native,candidate,bool(candidate and runaway(candidate,'table') is None))
                        record['legacy_rule_accepted']=simple_rule(features)
                    record['status']='complete'
                except Exception as error:
                    record.update(status='failed',error_type=type(error).__name__,reason=str(error),traceback=traceback.format_exc())
                    # Decoder malformation is captured inside recognize and
                    # remains a scored outcome. Exceptions escaping that call
                    # indicate execution/input failure, not model quality.
                    abort=type(error).__name__+': '+str(error)
                    torch.cuda.empty_cache()
                record['model_calls']=calls-before
            records.append(record);write()
        report.update(status='failed' if abort else 'complete',worker_abort=abort,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,reason=str(error),traceback=traceback.format_exc());raise
    finally:signal.alarm(0);write()


if __name__=='__main__':main()
