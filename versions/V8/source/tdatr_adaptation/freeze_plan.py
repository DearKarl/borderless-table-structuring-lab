"""Freeze shared full-corpus exposures and localization initialization on CPU."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sqlite3
from .execution import checked_json,sha,write,now

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding',required=True);args=ap.parse_args()
    binding=json.loads(Path(args.binding).read_text());root=Path(binding['output_root']);root.mkdir(exist_ok=False)
    audit=checked_json(binding['audit_path'],binding['audit_sha256'])
    if audit['status']!='passed' or audit['completed']!=20256 or audit['native_cache_exact']!=20244:raise ValueError('Complete audit required')
    if sha(binding['detail_database'])!=audit['detail_database_sha256']:raise ValueError('Audited database changed')
    common=checked_json(binding['common_inputs'],binding['common_inputs_sha256'])
    ids=[r['id'] for r in common['train']]
    if len(ids)!=20000 or len(set(ids))!=20000 or len({r['source'] for r in common['train']})!=11217:raise ValueError('Original TRAIN cohort changed')
    with sqlite3.connect('file:'+binding['detail_database']+'?mode=ro',uri=True) as db:
        eligible={i:json.loads(t) for i,t in db.execute("SELECT id,eligible_tasks FROM examples WHERE partition='train'")}
    if set(eligible)!=set(ids):raise ValueError('TRAIN membership differs from complete audit')
    entropy=random.SystemRandom();exposures=Counter();used={i:set() for i in ids};epochs=[]
    for stage in ['detail','fusion']:
        for epoch in range(1,4):
            order=ids.copy();entropy.shuffle(order);examples=[]
            for identifier in order:
                task='native_otsl'
                if stage=='detail':
                    choices=[t for t in eligible[identifier] if t not in used[identifier]] or eligible[identifier]
                    minimum=min(exposures[t] for t in choices)
                    task=entropy.choice([t for t in choices if exposures[t]==minimum]);used[identifier].add(task)
                exposures[task]+=1;examples.append([identifier,task])
            epochs.append(dict(stage=stage,epoch=epoch,examples=examples))
    calibration=checked_json(binding['calibration_inputs'],binding['calibration_inputs_sha256'])
    calibration_ids=[r['id'] for r in calibration['rows']]
    if len(calibration_ids)!=256 or len(set(calibration_ids))!=256 or not set(calibration_ids)<=set(ids):raise ValueError('Original TRAIN256 calibration differs')
    first_tasks=dict(epochs[0]['examples'])
    plan=dict(status='frozen',at_utc=now(),epochs=epochs,task_exposures=dict(exposures),
        train_tables=20000,train_sources=11217,exposures_per_version=120000,optimizer_updates_per_version=7500,
        detail_epochs=3,fusion_epochs=3,microbatch=1,gradient_accumulation=16,
        randomness='SystemRandom OS entropy; realized orders and tasks frozen and identical across both versions; no seed sweep',
        assignment='Greedy minimum global task exposure among available tasks, avoiding repeat per table where feasible; entropy breaks ties',
        calibration=[[i,first_tasks[i]] for i in calibration_ids],calibration_membership_sha256=binding['calibration_inputs_sha256'],
        common_inputs_sha256=binding['common_inputs_sha256'],detail_database_sha256=audit['detail_database_sha256'],audit_sha256=binding['audit_sha256'])
    write(root/'EXPOSURES.json',plan)
    import torch
    from safetensors.torch import save_file
    from .sgcl import CellLocalizer
    head=CellLocalizer(language_dim=1024,decoder_layers=28,vision_channels=[1280,1024])
    state={k:v.detach().cpu().contiguous() for k,v in head.state_dict().items()}
    save_file(state,str(root/'shared-localizer-initial.safetensors'))
    save_file({'torch_cpu':torch.get_rng_state()},str(root/'head-initialization-rng.safetensors'))
    write(root/'FREEZE.json',dict(status='passed',at_utc=now(),exposures_sha256=sha(root/'EXPOSURES.json'),
        shared_localizer_sha256=sha(root/'shared-localizer-initial.safetensors'),head_rng_sha256=sha(root/'head-initialization-rng.safetensors'),
        realized_framework_initial_seed=torch.initial_seed(),head_parameters=sum(p.numel() for p in head.parameters()),
        task_exposures=dict(exposures),formal_optimizer_updates=0,GPU_allocations=0))

if __name__=='__main__':main()
