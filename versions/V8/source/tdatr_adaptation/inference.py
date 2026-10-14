"""Exact adapted checkpoint reconstruction and annotation-free localization."""
import time
import torch
from safetensors.torch import load_file
from .execution import sha,load_base
from .model import attach,attach_localizer,restore
from .backbone import native_fusion_forward
from .inputs import NATIVE_PROMPT,encode_text,task_batch,ids_blob
from .cell_tokens import full_cell_token_ranges

def load_adapted(binding):
    if sha(binding['checkpoint_path'])!=binding['checkpoint_sha256']:raise ValueError('Frozen checkpoint changed')
    state=load_file(binding['checkpoint_path']);factors=None
    if binding['version']=='V8.5':
        factors={key[:-10]:(value,state[key[:-10]+'.initial_b']) for key,value in state.items() if key.endswith('.initial_a')}
        if len(factors)!=112:raise ValueError('Compensated initialization buffers incomplete')
    elif binding['version']!='V8.4':raise ValueError('Unknown adapted version')
    elif any(k.endswith(('.initial_a','.initial_b')) for k in state):raise ValueError('Ordinary LoRA cannot contain GA compensation')
    model=load_base(binding);attach(model,factors=factors,grounding=False)
    capture=attach_localizer(model);restore(model,state)
    for name,p in model.named_parameters():
        if p.requires_grad and not torch.equal(p.detach().cpu(),state[name]):raise ValueError('Reloaded trainable tensor differs')
    model.eval();model.requires_grad_(False)
    return model,capture

def localize_generated(model,processor,native_values,raw_otsl,capture):
    """Replay the model's own decoded OTSL, using no reference cells or geometry.

    This native-VLM adaptation reports an additional forward's cost. It does not
    edit the generated structure/text or replace the official HTML conversion.
    Invalid generated OTSL or an overlong replay remains an explicit unavailable
    localization outcome while the original textual prediction is retained.
    """
    start=time.monotonic()
    try:
        encoded=encode_text(processor,int(native_values['image_grid_thw'].prod())//4,NATIVE_PROMPT,raw_otsl)
        ranges=full_cell_token_ranges(raw_otsl,encoded['offsets'],sequence_start=encoded['counts']['target_start'])
        current=task_batch(native_values,dict(input_ids=ids_blob(encoded['input_ids']),**encoded['counts']))
        current={k:v.to('cuda:0') for k,v in current.items()}
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            _,prediction=native_fusion_forward(model,current,[c['token_range'] for c in ranges['cells']],capture)
        result=dict(status='complete',cells=len(ranges['cells']),labels_present=False,
            boxes_xyxy=prediction['boxes'][-1].cpu().tolist(),additional_model_calls=1)
        return dict(**result,seconds=time.monotonic()-start)
    finally:capture.clear()
