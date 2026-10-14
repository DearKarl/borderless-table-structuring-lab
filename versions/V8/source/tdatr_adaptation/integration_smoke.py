"""Discarded GA/detail/fusion/export/native-generation checks before formal fits."""
import argparse
import gc
import json
from pathlib import Path
import signal
import sys
import time
import traceback
from .execution import Inputs,checked_json,sha,write,now,verify_runtime,load_base

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding',required=True);args=ap.parse_args()
    binding=json.loads(Path(args.binding).read_text());root=Path(binding['output_root']);root.mkdir(exist_ok=False)
    verify_runtime(binding);inputs=Inputs(binding)
    import torch
    from transformers import AutoProcessor
    from safetensors.torch import load_file
    from .model import attach,attach_localizer,objective,optimizer_for,verify_gradients,exact_save
    from .inference import load_adapted,localize_generated
    from versions.generator_pilot.validate_smoke_inputs import tensor_summary
    from PIL import Image
    for path in binding['tele_source_paths']:sys.path.insert(0,path)
    from TeleOCR.vlm_utils.TeleOCR_client import TeleOCRClient
    from TeleOCR.vlm_utils.structs import ContentBlock
    processor=AutoProcessor.from_pretrained(binding['model_root'],local_files_only=True,trust_remote_code=False)
    first=inputs.common['train'][0]['id']
    largest=inputs.db.execute("SELECT id,task,sequence_tokens FROM tasks WHERE id IN (SELECT id FROM examples WHERE partition='train') ORDER BY sequence_tokens DESC,id,task LIMIT 1").fetchone()
    calibration=checked_json(binding['calibration_receipt'],binding['calibration_receipt_sha256'])
    if calibration['status']!='passed' or sha(binding['calibration_factors'])!=binding['calibration_factors_sha256']:
        raise ValueError('Verified formal initialization required for export test')
    if sha(binding['shared_localizer'])!=binding['shared_localizer_sha256']:raise ValueError('Shared head initialization changed')
    began=time.monotonic();capture=None
    report=dict(status='running',formal_training=False,formal_optimizer_updates=0,discarded_optimizer_updates=0,
        arms={},maximum_detail=dict(largest),input_database_sha256=binding['detail_database_sha256'])
    def save():report.update(at_utc=now(),elapsed_seconds=time.monotonic()-began);write(root/'SMOKE.json',report)
    try:
        save()
        for version in ['V8.4','V8.5']:
            arm=dict(status='running',checks=[]);report['arms'][version]=arm;save()
            model=load_base(binding);factors=None
            if version=='V8.5':
                state=load_file(binding['calibration_factors'])
                factors={k[:-2]:(v,state[k[:-2]+'.B']) for k,v in state.items() if k.endswith('.A')}
            attach(model,factors=factors,grounding=False)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train()
            optimizer=optimizer_for(model);signal.alarm(1200)
            batch=inputs.batch(largest['id'],largest['task'])
            loss,losses,_,_=objective(model,batch);loss.backward();arm['detail_gradients']=verify_gradients(model)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            optimizer.step();report['discarded_optimizer_updates']+=1;optimizer.zero_grad(set_to_none=True)
            arm['maximum_detail_loss']=float(loss.detach());del optimizer,batch,loss,losses;signal.alarm(0)
            arm['checks'].append('complete_audit_maximum_detail_forward_backward_update')
            capture=attach_localizer(model,load_file(binding['shared_localizer']));optimizer=optimizer_for(model)
            signal.alarm(1200);example=inputs.example(first);batch=inputs.native(first)
            loss,losses,_,_=objective(model,batch,capture=capture,cell_ranges=json.loads(example['cell_ranges']),
                boxes=inputs.records[first]['normalized_boxes'],logical_spans=json.loads(example['logical_spans']))
            loss.backward();arm['fusion_gradients']=verify_gradients(model)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            optimizer.step();report['discarded_optimizer_updates']+=1;optimizer.zero_grad(set_to_none=True);capture.clear()
            del optimizer,loss,losses;signal.alarm(0);model.gradient_checkpointing_disable();model.eval()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                expected=model(**{k:v for k,v in batch.items() if k!='labels'},use_cache=False).logits.cpu()
            path=root/(version+'-discarded.safetensors');exact_save(model,path)
            capture.close();capture=None;del model,batch,factors;gc.collect();torch.cuda.empty_cache()
            model,capture=load_adapted(dict(binding,version=version,checkpoint_path=str(path),checkpoint_sha256=sha(path)))
            batch=inputs.native(first)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                actual=model(**{k:v for k,v in batch.items() if k!='labels'},use_cache=False).logits.cpu()
            if not torch.equal(expected,actual):raise ValueError('Export/reconstruction changed native logits')
            del expected,actual,batch
            arm['checks'].extend(['detail_to_fusion_optimizer_reset_and_update','fresh_model_checkpoint_reload_logits_exact'])
            client=TeleOCRClient(backend='transformers',model=model,processor=processor,batch_size=1,use_tqdm=False)
            record=inputs.records[first]
            with Image.open(record['native_image_path']) as image:image.load();image=image.copy()
            prepared,prompts,parameters,indices=client.helper.prepare_for_extract(image,[ContentBlock('table',[0,0,1,1],angle=0)])
            if indices!=[0] or prepared[0].tobytes()!=image.tobytes():raise ValueError('Default native helper changed smoke input')
            prompt=processor.apply_chat_template(client.client.build_messages(prompts[0]),tokenize=False,add_generation_prompt=True)
            values=processor(text=[prompt],images=prepared,padding=True,return_tensors='pt')
            # Compare directly against the original native cached image tensors.
            from safetensors import safe_open
            with safe_open(record['cache_path'],framework='pt',device='cpu') as cache:
                for key in ['pixel_values','image_grid_thw']:
                    if not torch.equal(values[key],cache.get_tensor(key)):raise ValueError('Native processor smoke tensors differ')
            prompt_tokens=values['input_ids'].shape[1];values=values.to(device='cuda:0',dtype=torch.bfloat16)
            kwargs=client.client.build_generate_kwargs(parameters[0]);signal.alarm(1200)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):output=model.generate(**values,use_cache=True,**kwargs)
            ids=output[0,prompt_tokens:].cpu().tolist();raw=processor.decode([i for i in ids if i not in client.client.skip_token_ids],
                skip_special_tokens=False,clean_up_tokenization_spaces=False)
            localization=localize_generated(model,processor,values,raw,capture)
            if localization['status']!='complete' or localization['labels_present']:raise ValueError('Label-free localization did not execute')
            signal.alarm(0)
            arm.update(status='passed',generation=dict(generated_tokens=len(ids),eos_reached=bool(ids and ids[-1]==processor.tokenizer.eos_token_id),
                labels_present=False,localization_cells=localization['cells'],localization_seconds=localization['seconds'],generate_kwargs=kwargs),
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),checkpoint_discarded=True)
            arm['checks'].append('default_native_generation_and_annotation_free_localization')
            capture.close();capture=None;del model,client,values,output;gc.collect();torch.cuda.empty_cache();save()
        report['status']='passed';save()
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc());save();raise
    finally:
        signal.alarm(0)
        if capture is not None:capture.close()
        inputs.close()

if __name__=='__main__':main()
