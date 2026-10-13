"""Load a complete large-campaign adapter/projection package on its exact base.

This format is not a generic PEFT adapter: LoRA-GA reconstruction also subtracts
the stored initial low-rank factors. Use BF16 autocast for model forward/generate
because the trained visual merger remains FP32.
"""
import hashlib
import json
from pathlib import Path


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):value.update(block)
    return value.hexdigest()


def load(package,base,*,device='cpu'):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText,AutoProcessor
    try:from .large_adapters import attach,restore
    except ImportError:from large_adapters import attach,restore
    try:from .large_names import package_recipe
    except ImportError:from large_names import package_recipe
    package=Path(package);base=Path(base);manifest=json.loads((package/'MANIFEST.json').read_text())
    if manifest['format']!='large_lora_with_projection_and_initial_factor_compensation_v1':raise ValueError('Unknown package format')
    recipe=package_recipe(manifest)
    from importlib.metadata import version
    for name in ['torch','torchvision','transformers','safetensors','tokenizers','accelerate']:
        if version(name)!=manifest['runtime_packages'][name]:raise ValueError('Pinned reconstruction runtime differs: '+name)
    checkpoint=package/'adapter.safetensors'
    if digest(checkpoint)!=manifest['adapter_sha256']:raise ValueError('Adapter package changed')
    for name,expected in manifest['base_file_hashes'].items():
        path=base/name
        if base.resolve() not in path.resolve().parents or digest(path)!=expected:raise ValueError('Original base asset differs')
    for name,expected in manifest['reconstruction_source_hashes'].items():
        path=Path(__file__).resolve().parent/name
        if path.parent!=Path(__file__).resolve().parent or digest(path)!=expected:raise ValueError('Reconstruction source differs')
    model=AutoModelForImageTextToText.from_pretrained(str(base),local_files_only=True,trust_remote_code=True,
        dtype=torch.bfloat16,device_map={'':device},attn_implementation='sdpa')
    state=load_file(str(checkpoint));factors=None
    if recipe!='V7.3.0':
        factors={key[:-10]:(value,state[key[:-10]+'.initial_b']) for key,value in state.items() if key.endswith('.initial_a')}
        if len(factors)!=112:raise ValueError('GA package is missing initial factors')
    attach(model,factors=factors,grounding=recipe=='V7.3.2');restore(model,state)
    destinations=dict(model.named_parameters());destinations.update(dict(model.named_buffers()))
    if any(not torch.equal(destinations[name].detach().cpu(),value) for name,value in state.items()):raise ValueError('Reconstructed model state differs')
    model.eval().requires_grad_(False);model.config.max_position_embeddings=16384
    processor=AutoProcessor.from_pretrained(str(base),local_files_only=True,trust_remote_code=False)
    return model,processor,manifest


def generation_context(model):
    """Pair this context with torch.inference_mode() when using the loaded model."""
    import torch
    return torch.autocast(device_type=next(model.parameters()).device.type,dtype=torch.bfloat16)
