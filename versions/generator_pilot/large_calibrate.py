"""Estimate one shared generation-gradient initialization on frozen TRAIN256."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import signal
import time
import traceback


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parent;binding=json.loads((root/'STAGE_BINDING.json').read_text())
    from importlib.metadata import version
    observed_runtime={name:version(name) for name in binding['runtime_packages']}
    if observed_runtime!=binding['runtime_packages']:raise ValueError('Pinned runtime package versions differ')
    for path in binding['tele_source_paths']:sys.path.insert(0,path)
    manifest_path=root/'CALIBRATION_INPUTS.json'
    if digest(manifest_path)!=binding['calibration_inputs_sha256']:raise ValueError('Calibration inputs changed')
    manifest=json.loads(manifest_path.read_text());rows=manifest['rows']
    if len(rows)!=256 or len({r['id'] for r in rows})!=256 or any(r['partition']!='train' or r['status']!='input_eligible' for r in rows):
        raise ValueError('Calibration must use256 unique accepted TRAIN inputs')
    if manifest['seed']!=0 or manifest['status']!='full_dataset_frozen':raise ValueError('Formal calibration freeze missing')
    for row in rows:
        if digest(row['cache_path'])!=row['cache_sha256']:raise ValueError('Calibration processor cache changed')
    if args.preflight:
        (root/'CPU_PREFLIGHT.json').write_text(json.dumps(dict(status='passed',examples=256,gpu_allocations=0,optimizer_updates=0)))
        return
    import torch
    from transformers import AutoModelForImageTextToText
    from safetensors.torch import load_file,save_file
    from supervision import expected_adapter_modules,gpu_uuid_key
    from large_adapters import generation_loss,gradient_factors,attach
    if torch.cuda.device_count()!=1 or gpu_uuid_key(torch.cuda.get_device_properties(0).uuid)!=gpu_uuid_key(binding['selected_gpu']['uuid']):
        raise ValueError('GPU binding differs')
    torch.set_num_threads(4);torch.manual_seed(0);start=time.monotonic()
    def timeout_handler(signum,frame):raise TimeoutError('Calibration call exceeded1200 seconds')
    signal.signal(signal.SIGALRM,timeout_handler)
    report=dict(status='running',phase='loading_original_base',examples_completed=0,optimizer_updates=0,
        gradient_estimation_seed=0,calibration_inputs_sha256=binding['calibration_inputs_sha256'],
        reduction='Mean of256 per-example assistant+EOS mean generation gradients; BF16 base gradients accumulated in FP32',
        initialization='ArB2r stable gamma16; shared factors across B/C and both seeds')
    def write():
        report['elapsed_seconds']=time.monotonic()-start
        temporary=root/'CALIBRATION.tmp';temporary.write_text(json.dumps(report,indent=2));temporary.replace(root/'CALIBRATION.json')
    def batch(row):
        return {k:v.to('cuda:0',dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in load_file(row['cache_path']).items()}
    try:
        write()
        model=AutoModelForImageTextToText.from_pretrained(binding['model_root'],local_files_only=True,trust_remote_code=True,
            dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa')
        model.config.use_cache=False;model.requires_grad_(False)
        model.config.max_position_embeddings=16384
        selected={r['name']:model.get_submodule(r['name']).weight for r in expected_adapter_modules()}
        for weight in selected.values():weight.requires_grad_(True)
        means={name:torch.zeros_like(weight,dtype=torch.float32) for name,weight in selected.items()}
        # Checkpointing requires training mode. There are no learned adapters yet;
        # disable base dropout explicitly so calibration remains dropout-free.
        model.train()
        for module in model.modules():
            if isinstance(module,torch.nn.Dropout):module.p=0.
            if hasattr(module,'attention_dropout') and isinstance(module.attention_dropout,(float,int)):
                if module.attention_dropout!=0:raise ValueError('Unexpected nonzero base attention dropout')
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        report['phase']='generation_gradient_estimation';write()
        for index,row in enumerate(rows):
            signal.alarm(1200)
            model.zero_grad(set_to_none=True);current=batch(row);labels=current.pop('labels')
            with torch.autocast('cuda',dtype=torch.bfloat16):output=model(**current,use_cache=False)
            loss,_,_=generation_loss(output.logits,labels)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite calibration loss')
            loss.backward()
            for name,weight in selected.items():
                if weight.grad is None or not torch.isfinite(weight.grad).all():raise FloatingPointError('Missing/nonfinite calibration gradient')
                means[name].add_(weight.grad.float(),alpha=1/256)
            report.update(examples_completed=index+1,last_generation_ce=float(loss.detach()));write()
            del output,loss,current,labels
            signal.alarm(0)
        model.zero_grad(set_to_none=True);report['phase']='gradient_svd';write()
        with torch.random.fork_rng(devices=[0]):
            signal.alarm(1200)
            torch.manual_seed(0);factors={name:gradient_factors(value) for name,value in means.items()}
            signal.alarm(0)
        norms={name:float(value.norm()) for name,value in means.items()};del means,selected
        state={name+suffix:value.detach().cpu().contiguous() for name,pair in factors.items() for suffix,value in zip(['.A','.B'],pair)}
        factor_path=root/'calibration-factors.safetensors';save_file(state,str(factor_path))
        # Bind formal factors to actual base-logit identity on three predetermined
        # calibration records. No quality-based sample choice or optimizer step.
        model.gradient_checkpointing_disable();model.eval();references=[]
        for row in rows[:3]:
            signal.alarm(1200)
            current=batch(row);current.pop('labels')
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):references.append(model(**current,use_cache=False).logits.cpu())
            del current
            signal.alarm(0)
        attach(model,factors=factors);model.eval()
        for row,expected in zip(rows[:3],references):
            signal.alarm(1200)
            current=batch(row);current.pop('labels')
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):actual=model(**current,use_cache=False).logits.cpu()
            if not torch.equal(expected,actual):raise ValueError('Formal factors change initial base logits')
            del actual,current
            signal.alarm(0)
        report.update(status='passed',phase='complete',factor_sha256=digest(factor_path),factor_bytes=factor_path.stat().st_size,
            gradient_norms=norms,initial_function_checked_ids=[r['id'] for r in rows[:3]],initial_logits_exact=True,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),optimizer_updates=0)
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc());raise
    finally:signal.alarm(0);write()


if __name__=='__main__':main()
