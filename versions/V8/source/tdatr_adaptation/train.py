"""One formal detail-to-fusion fit with frozen, matched full-corpus exposures."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import signal
import time
import traceback
from .execution import Inputs,checked_json,sha,write,now,verify_runtime,load_base,rng_snapshot

def lr_multiplier(step,total=3750):
    warmup=math.ceil(.05*total)
    if not 1<=step<=total:raise ValueError('Update outside frozen stage schedule')
    return step/warmup if step<=warmup else .5*(1+math.cos(math.pi*(step-warmup)/(total-warmup)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding',required=True);args=ap.parse_args()
    binding=json.loads(Path(args.binding).read_text());root=Path(binding['output_root']);root.mkdir(exist_ok=False)
    protocol=checked_json(binding['protocol_path'],binding['protocol_sha256'])
    if protocol['status']!='frozen' or binding['version'] not in ['V8.4','V8.5']:raise ValueError('Formal campaign freeze missing')
    verify_runtime(binding)
    plan=checked_json(binding['exposures_path'],binding['exposures_sha256'])
    if plan['status']!='frozen' or plan['optimizer_updates_per_version']!=7500:raise ValueError('Formal exposures differ')
    inputs=Inputs(binding)
    ids={r['id'] for r in inputs.common['train']}
    if len(plan['epochs'])!=6 or any(len(e['examples'])!=20000 or {i for i,t in e['examples']}!=ids for e in plan['epochs']):
        raise ValueError('Every epoch must retain every original TRAIN table once')
    if [(e['stage'],e['epoch']) for e in plan['epochs']]!=[(s,e) for s in ['detail','fusion'] for e in [1,2,3]]:
        raise ValueError('Sequential stages or epoch order differ')
    import torch
    from safetensors.torch import load_file
    from .model import attach,attach_localizer,objective,optimizer_for,verify_gradients,exact_save
    model=optimizer=capture=None;begun=time.monotonic();updates=exposures=0;training_seconds=diagnostic_seconds=0.
    report=dict(status='running',version=binding['version'],phase='loading_original_base',formal_training=True,
        optimizer_updates=0,exposures=0,checkpoint_selection='DEV only; fusion epoch checkpoints; mean full-denominator TEDS; earliest epoch on ties',
        protocol_sha256=binding['protocol_sha256'],exposures_sha256=binding['exposures_sha256'],selected_gpu=binding['selected_gpu'],
        worker_pid=os.getpid(),checkpoints=[],task_exposures={},automatic_retry=False)
    def save():
        report.update(at_utc=now(),optimizer_updates=updates,exposures=exposures,training_seconds=training_seconds,
            diagnostic_seconds=diagnostic_seconds,elapsed_seconds=time.monotonic()-begun)
        write(root/'TRAINING.json',report)
    try:
        save();report['initial_rng']=rng_snapshot(root/'initial-rng.safetensors')
        model=load_base(binding);factors=None
        if binding['version']=='V8.5':
            calibration=checked_json(binding['calibration_receipt'],binding['calibration_receipt_sha256'])
            if calibration['status']!='passed' or calibration['examples_completed']!=256 or calibration['optimizer_updates']!=0:
                raise ValueError('First-stage calibration incomplete')
            if calibration['exposures_sha256']!=binding['exposures_sha256'] or calibration['factor_sha256']!=binding['calibration_factors_sha256']:
                raise ValueError('Calibration objective or factor identity changed')
            if sha(binding['calibration_factors'])!=binding['calibration_factors_sha256']:raise ValueError('Calibration factors changed')
            state=load_file(binding['calibration_factors'])
            factors={k[:-2]:(v,state[k[:-2]+'.B']) for k,v in state.items() if k.endswith('.A')}
        attach(model,factors=factors,grounding=False);del factors
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.train();optimizer=optimizer_for(model);stage_step=0;tasks=Counter();visited=set()
        def checkpoint(label,selection_eligible):
            path=root/(label+'.safetensors');exact_save(model,path)
            rng=rng_snapshot(root/(label+'-rng.safetensors'))
            torch.save(dict(optimizer=optimizer.state_dict(),optimizer_updates=updates,exposures=exposures,
                stage=report['phase'],stage_step=stage_step,automatic_resume=False),root/(label+'-optimizer.pt'))
            row=dict(label=label,path=str(path),sha256=sha(path),bytes=path.stat().st_size,optimizer_updates=updates,
                exposures=exposures,DEV_selection_eligible=selection_eligible,rng=rng,at_utc=now())
            report['checkpoints'].append(row);save()
        def diagnose(stream):
            nonlocal diagnostic_seconds
            state=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all();mode=model.training
            torch.cuda.synchronize();start=time.monotonic();model.eval()
            try:
                for partition,identifiers in inputs.common['panels'].items():
                    if len(identifiers)!=128:raise ValueError('Original diagnostic panels changed')
                    losses=[];numerator=denominator=0.
                    for identifier in identifiers:
                        signal.alarm(1200);batch=inputs.native(identifier)
                        with torch.no_grad():loss,_,n,d=objective(model,batch)
                        losses.append(float(loss));numerator+=float(n);denominator+=int(d)
                        del batch,loss,n,d;signal.alarm(0)
                    stream.write(json.dumps(dict(at_utc=now(),step=updates,partition=partition,examples=len(losses),
                        native_generation_ce=sum(losses)/len(losses),generation_nll_numerator=numerator,
                        target_tokens=denominator,dropout=False,used_for_checkpoint_selection=False))+'\n');stream.flush()
            finally:
                signal.alarm(0);model.train(mode);torch.set_rng_state(state);torch.cuda.set_rng_state_all(cuda)
                torch.cuda.synchronize();diagnostic_seconds+=time.monotonic()-start
        with (root/'SCALARS.jsonl').open('x') as scalars,(root/'DIAGNOSTICS.jsonl').open('x') as diagnostics:
            diagnose(diagnostics)
            for epoch in plan['epochs']:
                if epoch['stage']=='fusion' and capture is None:
                    if updates!=3750 or exposures!=60000:raise ValueError('Fusion began before complete detail stage')
                    if sha(binding['shared_localizer'])!=binding['shared_localizer_sha256']:raise ValueError('Shared head initialization changed')
                    capture=attach_localizer(model,load_file(binding['shared_localizer']))
                    # Each stage receives its own complete 3-epoch AdamW schedule.
                    # This reset is frozen identically for both versions.
                    del optimizer;optimizer=optimizer_for(model);stage_step=0
                report.update(phase=epoch['stage'],epoch=epoch['epoch']);save()
                epoch_seen=set()
                for offset in range(0,20000,16):
                    torch.cuda.synchronize();start=time.monotonic();optimizer.zero_grad(set_to_none=True)
                    multiplier=lr_multiplier(stage_step+1)
                    for group in optimizer.param_groups:group['lr']=group['base_lr']*multiplier
                    accumulated=Counter();numerator=denominator=cells=0
                    for identifier,task in epoch['examples'][offset:offset+16]:
                        if identifier in epoch_seen:raise ValueError('Duplicate within epoch')
                        if (epoch['stage']=='fusion')!=(task=='native_otsl'):raise ValueError('Task assigned to wrong stage')
                        signal.alarm(1200);batch=inputs.batch(identifier,task);kwargs={}
                        if capture is not None:
                            example=inputs.example(identifier)
                            kwargs=dict(capture=capture,cell_ranges=json.loads(example['cell_ranges']),
                                boxes=inputs.records[identifier]['normalized_boxes'],logical_spans=json.loads(example['logical_spans']))
                            cells+=len(kwargs['cell_ranges'])
                        total,losses,n,d=objective(model,batch,**kwargs)
                        for name,value in losses.items():accumulated[name]+=float(value.detach())/16
                        accumulated['total']+=float(total.detach())/16;numerator+=float(n.detach());denominator+=int(d.detach())
                        (total/16).backward();exposures+=1;tasks[task]+=1;visited.add(identifier);epoch_seen.add(identifier)
                        if capture is not None:capture.clear()
                        del batch,total,losses,n,d,kwargs;signal.alarm(0)
                    gradients=verify_gradients(model)
                    norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True))
                    optimizer.step();updates+=1;stage_step+=1
                    if updates==1:
                        report['first_optimizer_update']=dict(at_utc=now(),worker_pid=os.getpid(),selected_gpu=binding['selected_gpu'],exposures=exposures)
                        write(root/'FIRST_OPTIMIZER_UPDATE.json',report['first_optimizer_update'])
                    torch.cuda.synchronize();seconds=time.monotonic()-start;training_seconds+=seconds
                    row=dict(at_utc=now(),step=updates,stage=epoch['stage'],stage_step=stage_step,epoch=epoch['epoch'],epoch_examples=offset+16,
                        exposures=exposures,unique_tables=len(visited),losses=dict(accumulated),gradient_norms=gradients,pre_clip_gradient_norm=norm,
                        learning_rates=[g['lr'] for g in optimizer.param_groups],update_seconds=seconds,target_tokens=denominator,
                        generation_nll_numerator=numerator,cell_count=cells,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),skipped=False)
                    scalars.write(json.dumps(row)+'\n');scalars.flush()
                    report.update(task_exposures=dict(tasks),last_update=row);save()
                    if updates%250==0:diagnose(diagnostics)
                if epoch_seen!=ids:raise ValueError('Epoch lost an original cohort member')
                checkpoint(epoch['stage']+'-epoch-'+str(epoch['epoch']),epoch['stage']=='fusion')
        if updates!=7500 or exposures!=120000 or dict(tasks)!=plan['task_exposures']:raise ValueError('Formal fit exposure mismatch')
        report.update(status='complete',phase='complete',peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated());save()
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        if model is not None and optimizer is not None:
            try:checkpoint('failure-state',False)
            except Exception as recovery_error:report['failure_checkpoint_error']=repr(recovery_error)
        save();raise
    finally:
        signal.alarm(0)
        if capture is not None:capture.close()
        inputs.close()

if __name__=='__main__':main()
