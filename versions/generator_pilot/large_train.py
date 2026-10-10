"""One fixed full-corpus fit; common inputs and protocol must already be frozen.

This worker does not select data/checkpoints, generate predictions, or access
confirmation references. Epoch checkpoints are evaluated by a separate worker.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lr_multiplier(update,total=3750):
    warmup=math.ceil(total*.05)
    if not 1<=update<=total:raise ValueError('Update outside frozen schedule')
    return update/warmup if update<=warmup else .5*(1+math.cos(math.pi*(update-warmup)/(total-warmup)))


def resources():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,timeout=10)
    return {parts[0].strip():dict(utilization_percent=int(parts[1]),device_memory_used_mib=int(parts[2]))
        for line in raw.strip().splitlines() if len(parts:=line.split(','))==3}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parent
    binding=json.loads((root/'STAGE_BINDING.json').read_text())
    from importlib.metadata import version
    observed_runtime={name:version(name) for name in binding['runtime_packages']}
    if observed_runtime!=binding['runtime_packages']:raise ValueError('Pinned runtime package versions differ')
    for path in binding['tele_source_paths']:sys.path.insert(0,path)
    protocol_path=root/'EXECUTION_PROTOCOL.json'
    if digest(protocol_path)!=binding['execution_protocol_sha256']:raise ValueError('Protocol changed')
    protocol=json.loads(protocol_path.read_text())
    if protocol['status']!='full_dataset_and_execution_frozen':raise ValueError('Full freeze required')
    inputs_path=Path(binding['common_inputs_path'])
    if digest(inputs_path)!=protocol['common_inputs_sha256']:raise ValueError('Common inputs changed')
    inputs=json.loads(inputs_path.read_text())
    train=inputs['train'];dev=inputs['dev'];by_id={r['id']:r for r in train+dev}
    source_counts=Counter(r['source'] for r in train)
    if len(train)!=20000 or len({r['id'] for r in train})!=20000 or len(source_counts)<5000 or max(source_counts.values())>4:
        raise ValueError('Training corpus target or source cap differs')
    if len(dev)!=256 or len({r['source'] for r in dev})!=256 or set(source_counts)&{r['source'] for r in dev}:
        raise ValueError('Development source isolation differs')
    if len(by_id)!=20256:raise ValueError('Repeated table identifiers')
    for partition in ['train','dev']:
        panel=inputs['panels'][partition]
        if len(panel)!=128 or len(set(panel))!=128:raise ValueError('Diagnostic panel differs')
        if any(by_id[i]['partition']!=partition or by_id[i]['status']!='input_eligible' for i in panel):
            raise ValueError('Invalid diagnostic panel input')
    if binding['recipe']!='V7.3.0':
        receipt_path=Path(binding['calibration_receipt_path'])
        if digest(receipt_path)!=binding['calibration_receipt_sha256']:raise ValueError('Calibration receipt changed')
        calibration=json.loads(receipt_path.read_text())
        if calibration['status']!='passed' or calibration['examples_completed']!=256 or calibration['optimizer_updates']!=0:
            raise ValueError('Formal calibration incomplete')
        if calibration['calibration_inputs_sha256']!=protocol['calibration_inputs_sha256']:raise ValueError('Calibration subset differs')
        if calibration['factor_sha256']!=binding['calibration_factors_sha256']:raise ValueError('Calibration factor identity differs')
    if args.preflight:
        (root/'CPU_PREFLIGHT.json').write_text(json.dumps(dict(status='passed',train_tables=20000,train_sources=len(source_counts),
            development_sources=256,panel_examples=256,gpu_allocations=0,optimizer_updates=0)))
        return
    import numpy as np
    import torch
    from transformers import AutoModelForImageTextToText
    from safetensors.torch import load_file,save_file
    from large_adapters import attach,generation_loss,geometry_loss,combined_objective,checkpoint_state,LargeAttentionAdapter
    from supervision import gpu_uuid_key
    if torch.cuda.device_count()!=1 or gpu_uuid_key(torch.cuda.get_device_properties(0).uuid)!=gpu_uuid_key(binding['selected_gpu']['uuid']):
        raise ValueError('GPU identity differs')
    seed=binding['seed'];recipe=binding['recipe']
    if seed not in [0,1] or recipe not in ['V7.3.0','V7.3.1','V7.3.2']:raise ValueError('Unregistered fit')
    torch.set_num_threads(4);torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    def timeout_handler(signum,frame):raise TimeoutError('Training/diagnostic example exceeded frozen call timeout')
    signal.signal(signal.SIGALRM,timeout_handler)
    begun=time.monotonic();training_seconds=diagnostic_seconds=0.;updates=exposures=target_tokens=0;visited=set()
    run_id=binding['run_id']
    report=dict(status='running',run_id=run_id,recipe=recipe,seed=seed,optimizer_updates=0,exposures=0,training_seconds=0.,diagnostic_seconds=0.,
        formal_training=True,protocol_sha256=binding['execution_protocol_sha256'],checkpoints=[])
    def write():
        report.update(optimizer_updates=updates,exposures=exposures,training_seconds=training_seconds,
            diagnostic_seconds=diagnostic_seconds,total_wall_seconds=time.monotonic()-begun)
        temporary=root/'TRAINING.tmp';temporary.write_text(json.dumps(report,indent=2));temporary.replace(root/'TRAINING.json')
    def get_batch(record):
        path=Path(record['cache_path'])
        if digest(path)!=record['cache_sha256']:raise ValueError('Frozen processor cache changed')
        values=load_file(str(path))
        return {k:v.to('cuda:0',dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in values.items()}
    try:
        report['phase']='original_base_loading';write()
        model=AutoModelForImageTextToText.from_pretrained(binding['model_root'],local_files_only=True,trust_remote_code=True,
            dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa')
        model.config.use_cache=False
        model.config.max_position_embeddings=16384
        factors=None
        if recipe!='V7.3.0':
            factor_path=Path(binding['calibration_factors_path'])
            if digest(factor_path)!=binding['calibration_factors_sha256']:raise ValueError('Formal256 gradient initialization differs')
            factor_state=load_file(str(factor_path))
            factors={name[:-2]:(value,factor_state[name[:-2]+'.B']) for name,value in factor_state.items() if name.endswith('.A')}
        params=attach(model,factors=factors,grounding=recipe=='V7.3.2')
        if protocol['gradient_checkpointing']:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        optimizer=torch.optim.AdamW([
            dict(params=[p for n,p in params.items() if not n.startswith('model.visual.merger.')],lr=1e-4,base_lr=1e-4),
            dict(params=[p for n,p in params.items() if n.startswith('model.visual.merger.')],lr=1e-5,base_lr=1e-5)],
            betas=(.9,.999),eps=1e-8,weight_decay=.01)
        capture={}
        if recipe=='V7.3.2':
            hook=model.model.language_model.norm.register_forward_hook(lambda module,args,output:capture.update(hidden=output))
        torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)

        def panel_loss(step,stream):
            nonlocal diagnostic_seconds
            torch.cuda.synchronize();begin=time.monotonic();mode=model.training
            rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all();py_rng=random.getstate();np_rng=np.random.get_state()
            model.eval()
            try:
                for partition,ids in inputs['panels'].items():
                    losses=[];num=den=0.
                    for rid in ids:
                        signal.alarm(protocol['diagnostic_example_timeout_seconds'])
                        batch=get_batch(by_id[rid])
                        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):output=model(**{k:v for k,v in batch.items() if k!='labels'},use_cache=False)
                        ce,n,d=generation_loss(output.logits,batch['labels'])
                        losses.append(float(ce));num+=float(n);den+=int(d)
                        capture.clear();del output,batch,ce
                        signal.alarm(0)
                    stream.write(json.dumps(dict(run_id=run_id,recipe=recipe,seed=seed,step=step,partition=partition,examples=len(ids),
                        common_generation_ce=sum(losses)/len(losses),generation_nll_numerator=num,valid_target_tokens=den,
                        dropout=False,checkpoint_selection_used=False))+'\n');stream.flush()
            finally:
                signal.alarm(0)
                model.train(mode);torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng);random.setstate(py_rng);np.random.set_state(np_rng)
                torch.cuda.synchronize();diagnostic_seconds+=time.monotonic()-begin

        orders=inputs['epoch_orders'][str(seed)]
        if len(orders)!=3 or any(len(o)!=20000 or set(o)!={r['id'] for r in train} for o in orders):
            raise ValueError('Per-seed epoch order is not three complete permutations')
        model.train();report['phase']='training';write()
        with (root/'SCALARS.jsonl').open('x') as scalars,(root/'DIAGNOSTICS.jsonl').open('x') as diagnostics:
            scalars.write(json.dumps(dict(run_id=run_id,recipe=recipe,seed=seed,step=0,microsteps=0,epoch=0,unique_examples_visited=0,
                exposures=0,valid_target_tokens=0,common_generation_ce=None,geometry_l1=None,total_objective=None,
                cumulative_target_tokens=0,generation_nll_numerator=0.,generation_nll_denominator=0,valid_cell_count=0,
                pre_clip_gradient_norm=None,clipped=False,update_seconds=0.,target_tokens_per_second=None,
                adapter_learning_rate=0.,projection_learning_rate=0.,training_seconds=0.,nonfinite=False,skipped=False))+'\n');scalars.flush()
            panel_loss(0,diagnostics)
            for epoch,order in enumerate(orders,1):
                for offset in range(0,20000,16):
                    torch.cuda.synchronize();begin=time.monotonic();optimizer.zero_grad(set_to_none=True)
                    step=updates+1;multiplier=lr_multiplier(step)
                    for group in optimizer.param_groups:group['lr']=group['base_lr']*multiplier
                    ce_values=[];geo_values=[];objective_values=[];num=den=valid_cells=0
                    for rid in order[offset:offset+16]:
                        signal.alarm(protocol['training_example_timeout_seconds'])
                        record=by_id[rid];batch=get_batch(record)
                        with torch.autocast('cuda',dtype=torch.bfloat16):output=model(**{k:v for k,v in batch.items() if k!='labels'},use_cache=False)
                        ce,n,d=generation_loss(output.logits,batch['labels']);geo=None
                        if recipe=='V7.3.2':
                            geo,_=geometry_loss(model.cell_box_head,capture['hidden'],record['anchor_positions'],record['normalized_boxes'])
                            valid_cells+=len(record['anchor_positions'])
                        loss=combined_objective(ce,geo,.1 if geo is not None else 0.)
                        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
                        ce_values.append(float(ce.detach()));objective_values.append(float(loss.detach()))
                        if geo is not None:geo_values.append(float(geo.detach()))
                        num+=float(n.detach());den+=int(d.detach())
                        (loss/16).backward();exposures+=1;visited.add(rid)
                        capture.clear();del output,ce,geo,loss,batch,n,d
                        signal.alarm(0)
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params.values()):
                        raise FloatingPointError('Missing/nonfinite trainable gradient')
                    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise ValueError('Frozen base received gradient')
                    summaries={}
                    if step%50==0:
                        for category,prefix in [('projection','model.visual.merger.'),('head','cell_box_head.')]:
                            group=[p.grad.float().square().sum() for n,p in params.items() if n.startswith(prefix)]
                            summaries[category+'_gradient_norm']=float(torch.stack(group).sum().sqrt()) if group else None
                    norm=float(torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True))
                    optimizer.step();updates=step;target_tokens+=den
                    if step%50==0:
                        squared=[];base_squared=[]
                        with torch.no_grad():
                            for module in model.modules():
                                if not isinstance(module,LargeAttentionAdapter):continue
                                a=module.lora_a;b=module.lora_b
                                if module.initial_a is not None:
                                    a=torch.cat([a,module.initial_a]);b=torch.cat([b,-module.initial_b],dim=1)
                                squared.append(4*((b.T@b)*(a@a.T).T).sum())
                                base_squared.append(module.base.weight.float().square().sum())
                        update_norm=float(torch.stack(squared).sum().clamp_min(0).sqrt())
                        base_norm=float(torch.stack(base_squared).sum().sqrt())
                        summaries.update(adapter_update_frobenius=update_norm,adapter_update_to_attention_base_ratio=update_norm/base_norm,
                            allocated_gpu_bytes=torch.cuda.memory_allocated(),peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
                        summaries.update(resources()[binding['selected_gpu']['uuid']])
                    torch.cuda.synchronize();seconds=time.monotonic()-begin;training_seconds+=seconds
                    scalars.write(json.dumps(dict(run_id=run_id,recipe=recipe,seed=seed,step=step,microsteps=exposures,epoch=epoch-1+(offset+16)/20000,
                        unique_examples_visited=len(visited),exposures=exposures,valid_target_tokens=den,cumulative_target_tokens=target_tokens,
                        generation_nll_numerator=num,generation_nll_denominator=den,common_generation_ce=sum(ce_values)/16,
                        geometry_l1=sum(geo_values)/16 if geo_values else None,valid_cell_count=valid_cells,
                        total_objective=sum(objective_values)/16,adapter_learning_rate=1e-4*multiplier,projection_learning_rate=1e-5*multiplier,
                        pre_clip_gradient_norm=norm,clipped=norm>1.,nonfinite=False,skipped=False,
                        update_seconds=seconds,training_seconds=training_seconds,target_tokens_per_second=den/seconds,**summaries))+'\n');scalars.flush();write()
                    if step%250==0:panel_loss(step,diagnostics);write()
                checkpoint=root/f'epoch-{epoch}.safetensors';state=checkpoint_state(model);save_file(state,str(checkpoint))
                reloaded=load_file(str(checkpoint))
                if set(reloaded)!=set(state) or any(not torch.equal(value,reloaded[name]) for name,value in state.items()):
                    raise ValueError('Serialized epoch parameters differ')
                del state,reloaded
                recovery=root/f'epoch-{epoch}-recovery.pt'
                torch.save(dict(optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                    python_rng=random.getstate(),numpy_rng=np.random.get_state(),updates=updates,exposures=exposures,
                    target_tokens=target_tokens,training_seconds=training_seconds,diagnostic_seconds=diagnostic_seconds,
                    common_inputs_sha256=protocol['common_inputs_sha256'],protocol_sha256=binding['execution_protocol_sha256'],
                    automatic_resume=False),recovery)
                report['checkpoints'].append(dict(epoch=epoch,step=updates,file=checkpoint.name,sha256=digest(checkpoint),bytes=checkpoint.stat().st_size,
                    serialized_state_exact=True,recovery_file=recovery.name,recovery_sha256=digest(recovery),recovery_bytes=recovery.stat().st_size))
                write()
        if updates!=3750 or exposures!=60000 or len(visited)!=20000:raise ValueError('Completed schedule differs')
        report.update(status='passed',phase='complete',peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            checkpoint_selection_performed=False,confirmation_accessed=False)
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        raise
    finally:signal.alarm(0);write()


if __name__=='__main__':main()
