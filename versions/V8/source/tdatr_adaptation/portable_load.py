"""Reconstruct a packaged TDATR-adapted model on a supplied original base."""
import hashlib
import json
from pathlib import Path
import time


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):result.update(block)
    return result.hexdigest()


def load(package,base,*,device='cuda:0'):
    """Return model, processor, feature capture, and manifest after exact checks.

    The package stores all trained adapter, visual-merger and localization tensors.
    V8.5 also needs its initial factors for residual compensation. A generic PEFT
    loader or a BF16 merge is not an equivalent reconstruction.
    """
    import torch
    from importlib.metadata import version
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText,AutoProcessor
    from .model import attach,attach_localizer,restore
    package=Path(package).resolve();base=Path(base).resolve()
    manifest=json.loads((package/'MANIFEST.json').read_text())
    if manifest['format']!='tdatr_native_lora_complete_v1' or manifest['version'] not in ['V8.4','V8.5']:
        raise ValueError('Unrecognized complete adapted model package')
    for name in ['torch','torchvision','transformers','safetensors','tokenizers','accelerate']:
        if version(name)!=manifest['runtime_packages'][name]:raise ValueError('Reconstruction runtime differs: '+name)
    code=Path(__file__).resolve().parents[2]
    for root,hashes in [(base,manifest['base_file_hashes']),(code,manifest['reconstruction_source_hashes'])]:
        for name,expected in hashes.items():
            path=(root/name).resolve()
            if not path.is_relative_to(root) or digest(path)!=expected:raise ValueError('Frozen model/source asset differs: '+name)
    checkpoint=package/'adapted.safetensors'
    if digest(checkpoint)!=manifest['checkpoint_sha256']:raise ValueError('Packaged checkpoint changed')
    state=load_file(str(checkpoint));factors=None
    if manifest['version']=='V8.5':
        factors={name[:-10]:(value,state[name[:-10]+'.initial_b']) for name,value in state.items() if name.endswith('.initial_a')}
        if len(factors)!=112:raise ValueError('Compensated initialization factors incomplete')
    elif any(name.endswith(('.initial_a','.initial_b')) for name in state):
        raise ValueError('Ordinary LoRA contains unexpected compensation')
    model=AutoModelForImageTextToText.from_pretrained(str(base),local_files_only=True,trust_remote_code=True,
        dtype=torch.bfloat16,device_map={'':device},attn_implementation='sdpa')
    attach(model,factors=factors,grounding=False);capture=attach_localizer(model);restore(model,state)
    destinations=dict(model.named_parameters());destinations.update(dict(model.named_buffers()))
    if any(not torch.equal(destinations[name].detach().cpu(),value) for name,value in state.items()):
        capture.close();raise ValueError('Restored model tensors differ')
    model.eval().requires_grad_(False);model.config.max_position_embeddings=16384;model.config.use_cache=False
    processor=AutoProcessor.from_pretrained(str(base),local_files_only=True,trust_remote_code=False)
    return model,processor,capture,manifest


def generation_context(model):
    import torch
    return torch.autocast(device_type=next(model.parameters()).device.type,dtype=torch.bfloat16)


def localize(model,processor,native_values,raw_otsl,capture):
    """Use only the model's own generated OTSL and its original native image.

    This extra forward returns boxes without altering the text prediction. Invalid
    OTSL or a replay above the context limit raises ValueError; callers retain the
    unchanged text and report unavailable localization.
    """
    import torch
    from .backbone import native_fusion_forward
    from .inputs import NATIVE_PROMPT,encode_text,task_batch,ids_blob
    from .cell_tokens import full_cell_token_ranges
    begun=time.monotonic();device=next(model.parameters()).device
    try:
        encoded=encode_text(processor,int(native_values['image_grid_thw'].prod())//4,NATIVE_PROMPT,raw_otsl)
        ranges=full_cell_token_ranges(raw_otsl,encoded['offsets'],sequence_start=encoded['counts']['target_start'])
        batch=task_batch(native_values,dict(input_ids=ids_blob(encoded['input_ids']),**encoded['counts']))
        batch={key:value.to(device) for key,value in batch.items()}
        with torch.inference_mode(),generation_context(model):
            _,prediction=native_fusion_forward(model,batch,[cell['token_range'] for cell in ranges['cells']],capture)
        boxes=prediction['boxes'][-1].cpu().tolist()
        return dict(status='complete',cells=len(boxes),boxes_xyxy=boxes,labels_present=False,
            additional_model_calls=1,seconds=time.monotonic()-begun)
    finally:capture.clear()
