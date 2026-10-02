"""Load exact NaviDC checkpoint on CPU, count unique parameters; no generation."""
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import torch
from transformers import AutoConfig
from hybrid.native_tables.runtime.generation import check_loading_info
from btsl.model import verify_model, runtime


def main():
    model_dir, output = map(Path, sys.argv[1:])
    runtime(); verify_model(model_dir)
    source = model_dir / 'modeling_naviocr.py'
    spec = importlib.util.spec_from_file_location('_sha_bound_navidc_model', source)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    model, info = module.Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_dir), config=config,
        local_files_only=True, use_safetensors=True, ignore_mismatched_sizes=False, output_loading_info=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    check_loading_info(info)
    assert all(p.device.type == 'cpu' for p in model.parameters())
    assert model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
    parameters = list(model.named_parameters(remove_duplicate=True)); buffers = list(model.named_buffers(remove_duplicate=True))
    params = sum(p.numel() for _, p in parameters); buffer_count = sum(p.numel() for _, p in buffers)
    with (model_dir / 'model.safetensors').open('rb') as f:
        header = json.loads(f.read(struct.unpack('<Q', f.read(8))[0]))
    stored = sum(math.prod(v['shape']) for k,v in header.items() if k != '__metadata__')
    result = {'unique_loaded_parameters': params, 'unique_loaded_buffers': buffer_count,
              'stored_tensor_elements': stored, 'conservative_upper_bound': max(stored, params + buffer_count),
              'tied_embedding_verified': True, 'loading_info': info, 'GPU_used': False, 'forward_calls': 0,
              'parameters': [{'name': n, 'shape': list(p.shape), 'numel': p.numel()} for n,p in parameters],
              'buffers': [{'name': n, 'shape': list(p.shape), 'numel': p.numel()} for n,p in buffers]}
    with output.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('parameters','buffers')}), flush=True)


if __name__ == '__main__':
    main()
