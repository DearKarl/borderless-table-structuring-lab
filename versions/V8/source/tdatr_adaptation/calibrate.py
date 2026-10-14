"""LoRA-GA initialization from the frozen first-stage detail objective on TRAIN256."""
import argparse
import json
from pathlib import Path
import signal
import time
import traceback
from .execution import Inputs,checked_json,sha,write,now,verify_runtime,load_base,rng_snapshot

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding',required=True);args=ap.parse_args()
    binding=json.loads(Path(args.binding).read_text());root=Path(binding['output_root']);root.mkdir(exist_ok=False)
    verify_runtime(binding);inputs=Inputs(binding)
    plan=checked_json(binding['exposures_path'],binding['exposures_sha256']);examples=plan['calibration']
    if plan['status']!='frozen' or len(examples)!=256 or len({i for i,t in examples})!=256:
        raise ValueError('Frozen TRAIN256 detail calibration missing')
    if any(inputs.records[i]['partition']!='train' or task=='native_otsl' for i,task in examples):raise ValueError('Calibration scope differs')
    import torch
    from safetensors.torch import save_file
    from versions.generator_pilot.supervision import expected_adapter_modules
    from .model import attach,objective
    from versions.generator_pilot.large_adapters import gradient_factors
    began=time.monotonic()
    report=dict(status='running',phase='loading_original_base',examples_completed=0,optimizer_updates=0,
        exposures_sha256=binding['exposures_sha256'],calibration_membership_sha256=plan['calibration_membership_sha256'],
        reduction='Mean of 256 per-example detail assistant+EOS mean gradients; native BF16 gradients accumulated in FP32',
        initialization='ArB2r/stable/gamma16, rank8, q32/niter4 SVD, exact initial-factor compensation',
        original_base=True,dropout=False,automatic_retry=False)
    def save():report.update(at_utc=now(),elapsed_seconds=time.monotonic()-began);write(root/'CALIBRATION.json',report)
    try:
        report['initial_rng']=rng_snapshot(root/'initial-rng.safetensors');save()
        model=load_base(binding);model.requires_grad_(False)
        selected={r['name']:model.get_submodule(r['name']).weight for r in expected_adapter_modules()}
        for weight in selected.values():weight.requires_grad_(True)
        means={name:torch.zeros_like(weight,dtype=torch.float32) for name,weight in selected.items()}
        model.train()
        for module in model.modules():
            if isinstance(module,torch.nn.Dropout):module.p=0.
            if isinstance(getattr(module,'attention_dropout',None),(float,int)) and module.attention_dropout!=0:
                raise ValueError('Unexpected nonzero base attention dropout')
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        report['phase']='first_stage_detail_gradient_estimation';save()
        for number,(identifier,task) in enumerate(examples,1):
            signal.alarm(1200);model.zero_grad(set_to_none=True);batch=inputs.batch(identifier,task)
            loss,_,_,_=objective(model,batch);loss.backward()
            for name,weight in selected.items():
                if weight.grad is None or not torch.isfinite(weight.grad).all():raise FloatingPointError('Missing/nonfinite initialization gradient')
                means[name].add_(weight.grad.float(),alpha=1/256)
            report.update(examples_completed=number,last_detail_ce=float(loss.detach()),last_task=task);save()
            del batch,loss;signal.alarm(0)
        model.zero_grad(set_to_none=True);report['phase']='gradient_svd';save()
        report['svd_rng_before']=rng_snapshot(root/'svd-rng-before.safetensors')
        signal.alarm(1200);factors={name:gradient_factors(value) for name,value in means.items()};signal.alarm(0)
        report['gradient_norms']={name:float(value.norm()) for name,value in means.items()};del means,selected
        state={name+suffix:value.detach().cpu().contiguous() for name,pair in factors.items() for suffix,value in zip(['.A','.B'],pair)}
        path=root/'calibration-factors.safetensors';save_file(state,str(path));del state
        model.gradient_checkpointing_disable();model.eval();references=[]
        for identifier,task in examples[:3]:
            signal.alarm(1200);batch=inputs.batch(identifier,task)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                references.append(model(**{k:v for k,v in batch.items() if k!='labels'},use_cache=False).logits.cpu())
            del batch;signal.alarm(0)
        attach(model,factors=factors,grounding=False);model.eval()
        for (identifier,task),expected in zip(examples[:3],references):
            signal.alarm(1200);batch=inputs.batch(identifier,task)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                actual=model(**{k:v for k,v in batch.items() if k!='labels'},use_cache=False).logits.cpu()
            if not torch.equal(expected,actual):raise ValueError('GA factors changed initial detail logits')
            del batch,actual;signal.alarm(0)
        report.update(status='passed',phase='complete',initial_logits_exact=True,initial_function_checked_examples=examples[:3],
            factor_sha256=sha(path),factor_bytes=path.stat().st_size,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
        save()
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc());save();raise
    finally:signal.alarm(0);inputs.close()

if __name__=='__main__':main()
