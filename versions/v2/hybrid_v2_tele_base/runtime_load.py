"""Initialization extracted verbatim as AST from GPU-passed smoke v4. No recipe changes."""
from .smoke import sys, importlib, ResourceAudit, RunBlocked, save, deadline, signal, install_tele_audit, make_client_class

def load_runtime(manifest, out, summary):
    sys.path.insert(0, manifest['tele_source'])
    resources = ResourceAudit(manifest['identity_files'])
    resources.bind_tele_loaders()
    for name, version in manifest['tele_packages'].items():
        if importlib.metadata.version(name) != version:
            raise RunBlocked('Package mismatch: ' + name)
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    save(out / 'ENVIRONMENT.json', {'packages': {d.metadata['Name']: {'version': d.version, 'path': str(d._path)} for d in importlib.metadata.distributions()}, 'sys_path': sys.path})
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RunBlocked('Expected one visible BF16-capable GPU')
    gpu = torch.cuda.get_device_properties(0)
    actual_uuid = str(gpu.uuid)
    if actual_uuid.removeprefix('GPU-').lower() != manifest['gpu_uuid'].removeprefix('GPU-').lower():
        raise RunBlocked('Actual CUDA UUID differs from host lease')
    summary['gpu'] = {'uuid': actual_uuid, 'name': gpu.name, 'torch_cuda': torch.version.cuda}
    import TeleOCR.config as config
    config.BACKEND = 'transformers'
    config.model_path = manifest['tele_model']
    config.LAYOUT_MODE = 'Detection'
    config.PDF_TOOLS = 'pypdfium2'
    config.PDF_TOOLS_WORKER_MAX_NUM = 4
    config.MAX_PIXELS = 64000000
    from TeleOCR.tools.read_file import read_fn
    from TeleOCR.engine import do_parse
    from TeleOCR.vlm_utils.TeleOCR_client import TeleOCRClient
    deadline(600)
    processor = AutoProcessor.from_pretrained(manifest['tele_model'], trust_remote_code=True, local_files_only=True)
    save(out / 'PROCESSOR.json', {'class': type(processor).__name__, 'image_processor_class': type(processor.image_processor).__name__, 'image_processor_config': processor.image_processor.to_dict(), 'tokenizer_class': type(processor.tokenizer).__name__, 'tokenizer_init_kwargs': processor.tokenizer.init_kwargs, 'chat_template': processor.chat_template})
    model, loading = AutoModelForImageTextToText.from_pretrained(manifest['tele_model'], trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa', output_loading_info=True)
    for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
        if key not in loading or loading[key]:
            raise RunBlocked('Nonempty or missing loading audit: ' + key)
    model = model.to('cuda:0').eval()
    signal.alarm(0)
    if type(model).__name__ != 'Qwen2_5_VLForConditionalGeneration':
        raise RunBlocked('Tele model class differs')
    if any((x.dtype != torch.bfloat16 for x in model.parameters() if x.is_floating_point())):
        raise RunBlocked('Tele parameter dtype mismatch')
    summary['model_loads']['tele'] = 1
    save(out / 'TELE_LOAD.json', {'loading_info': loading, 'generation_config': model.generation_config.to_dict(), 'config': model.config.to_dict(), 'parameter_count': sum((p.numel() for p in model.parameters())), 'model_class': type(model).__name__, 'dtype': str(model.dtype)})
    generation = []
    install_tele_audit(model, generation.append)
    hybrid_class = make_client_class(TeleOCRClient)
    return locals()
