"""Discarded actual-model integration checks; never a formal fit or selection run."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import traceback


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding',required=True);args=ap.parse_args()
    binding=json.loads(Path(args.binding).read_text());root=Path(binding['output_root'])
    root.mkdir(exist_ok=False)
    import torch
    from transformers import AutoModelForImageTextToText,AutoProcessor
    from safetensors.torch import load_file
    from .inputs import encode_text, task_batch, ids_blob
    from .targets import render
    from .cell_tokens import full_cell_token_ranges
    from .backbone import native_fusion_forward
    from .model import attach,attach_localizer,objective,optimizer_for,verify_gradients,exact_save,restore
    from versions.generator_pilot.supervision import gpu_uuid_key
    assert torch.cuda.device_count()==1
    assert gpu_uuid_key(torch.cuda.get_device_properties(0).uuid)==gpu_uuid_key(binding['selected_gpu']['uuid'])
    assert binding['selected_gpu']['physical_index'] in [1,2,3,5,6]
    torch.set_num_threads(4)
    begun=time.monotonic()
    report=dict(status='running',formal_training=False,formal_optimizer_updates=0,discarded_optimizer_updates=0,
        torch_initial_seed=torch.initial_seed(),selected_gpu=binding['selected_gpu'],checks=[],samples=[])
    def save():
        report.update(at_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-begun)
        temporary=root/'SMOKE.tmp';temporary.write_text(json.dumps(report,indent=2));temporary.replace(root/'SMOKE.json')
    try:
        save()
        source=Path(binding['common_inputs']);assert hashlib.sha256(source.read_bytes()).hexdigest()==binding['common_inputs_sha256']
        records=json.loads(source.read_text())['train']
        selected=[records[0],max(records,key=lambda r:len(r['anchor_positions'])),
            max(records,key=lambda r:r['audit']['counts']['sequence_tokens'])]
        selected=list({r['id']:r for r in selected}.values())
        processor=AutoProcessor.from_pretrained(binding['model_root'],local_files_only=True,trust_remote_code=False)
        model=AutoModelForImageTextToText.from_pretrained(binding['model_root'],local_files_only=True,
            trust_remote_code=True,dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa')
        model.config.use_cache=False
        def batch(record):
            path=Path(record['cache_path'])
            assert hashlib.sha256(path.read_bytes()).hexdigest()==record['cache_sha256']
            return {k:v.to('cuda:0',dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in load_file(str(path)).items()}
        first=batch(selected[0]);model.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            before=model(**{k:v for k,v in first.items() if k!='labels'},use_cache=False).logits.cpu()
        attach(model,grounding=False)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            after=model(**{k:v for k,v in first.items() if k!='labels'},use_cache=False).logits.cpu()
        assert torch.equal(before,after)
        del before,after
        report['checks'].append('ordinary_LoRA_initial_native_logits_exact');save()
        record=selected[0]
        manifest=json.loads((Path(record['cache_path']).parent.parent/'LABEL_MANIFEST.json').read_text())
        canonical=next(r['canonical'] for r in manifest['rows'] if r['id']==record['id'])
        prompt,target=render('cell_detection',canonical['table'],record['normalized_boxes'])
        encoded=encode_text(processor,int(first['image_grid_thw'].prod())//4,prompt,target)
        detail=task_batch(first,dict(input_ids=ids_blob(encoded['input_ids']),**encoded['counts']))
        detail={k:v.to('cuda:0') for k,v in detail.items()}
        assert torch.equal(detail['pixel_values'],first['pixel_values'])
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.train();optimizer=optimizer_for(model)
        loss,losses,_,_=objective(model,detail);loss.backward()
        report['detail_gradients']=verify_gradients(model)
        torch.nn.utils.clip_grad_norm_(list(p for p in model.parameters() if p.requires_grad),1,error_if_nonfinite=True)
        optimizer.step();report['discarded_optimizer_updates']+=1
        report['detail_loss']=float(loss.detach());report['checks'].append('detail_native_image_preserved_forward_backward_update')
        optimizer.zero_grad(set_to_none=True);del optimizer,loss,losses,detail,first
        capture=attach_localizer(model);optimizer=optimizer_for(model)
        for number,record in enumerate(selected):
            current=batch(record)
            manifest=json.loads((Path(record['cache_path']).parent.parent/'LABEL_MANIFEST.json').read_text())
            canonical=next(r['canonical'] for r in manifest['rows'] if r['id']==record['id'])
            table=canonical['table'];target=canonical['target_otsl']
            tokenized=processor.tokenizer(target,add_special_tokens=False,return_offsets_mapping=True)
            start=record['audit']['counts']['target_start']
            ranges=full_cell_token_ranges(target,tokenized['offset_mapping'],sequence_start=start)
            cells=[x['token_range'] for x in ranges['cells']]
            spans=[[c['row'],c['row']+c['rowspan']-1,c['col'],c['col']+c['colspan']-1] for c in table['cells']]
            optimizer.zero_grad(set_to_none=True)
            loss,losses,_,_=objective(model,current,capture=capture,cell_ranges=cells,
                boxes=record['normalized_boxes'],logical_spans=spans)
            maps=capture.maps()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                index,_=model.model.visual.get_window_index(current['image_grid_thw'])
                assert torch.equal(model.model.visual.merger(capture.fine)[index.argsort()],capture.coarse)
            loss.backward();gradients=verify_gradients(model)
            report['samples'].append(dict(id=record['id'],tokens=int(current['input_ids'].shape[1]),cells=len(cells),
                fine_shape=list(maps[0].shape),coarse_shape=list(maps[1].shape),
                losses={k:float(v.detach()) for k,v in losses.items()},gradient_norms=gradients,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),native_merger_restoration_exact=True))
            if number==0:
                torch.nn.utils.clip_grad_norm_(list(p for p in model.parameters() if p.requires_grad),1,error_if_nonfinite=True)
                optimizer.step();report['discarded_optimizer_updates']+=1
            optimizer.zero_grad(set_to_none=True);capture.clear()
            del maps,loss,losses,current
            save()
        state=exact_save(model,root/'discarded-state.safetensors')
        with torch.no_grad():
            next(p for p in model.parameters() if p.requires_grad).add_(1)
        restore(model,state)
        for name,p in model.named_parameters():
            if p.requires_grad and not torch.equal(p.detach().cpu(),state[name]):raise ValueError('Restore mismatch')
        report['checks'].extend(['native_two_resolution_spatial_maps_exact','fusion_all_trainable_groups_finite',
            'largest_cell_and_token_cases_passed','full_trainable_checkpoint_save_restore_exact'])
        capture.close()
        report.update(status='passed',checkpoint_discarded=True,formal_optimizer_updates=0,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
        save()
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc());save();raise


if __name__=='__main__': main()
