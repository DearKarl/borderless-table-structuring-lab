"""GT-free native whole-page M/P/T adapters; single owned GPU worker."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parent))
from complementarity_ovis_smoke import digest, supervise, write
from page_paths import component_name


def load_mineru(root, out):
    write(out/'linux-bootstrap.json',{'platform':sys.platform,'Windows_DLL_bootstrap_used':False})
    os.environ['MINERU_LMDEPLOY_BACKEND'] = 'turbomind'
    os.environ['MINERU_MODEL_SOURCE'] = 'local'
    from mineru.config import config, VlmConfig
    config.model.base_dir = str(root / 'models')
    config.model.source = 'local'
    config.model.small_backend = 'torch'
    config.model.vlm = VlmConfig(engine='lmdeploy', max_concurrency=1)
    from mineru.model.vlm.runtime import ModelSingleton
    original_create = ModelSingleton._create_model
    def resource_config(self, backend, model_path, server_url, **kwargs):
        if backend == 'lmdeploy-engine':
            model_path = str(root / 'models/MinerU2.5-Pro-2605-1.2B')
            write(out/'actual-local-model-path.json', {'backend':backend,'model_path':model_path,'source':'controller_SHA_verified_read_only_mount'})
            kwargs.update(batch_size=1, max_batch_size=1, dtype='bfloat16', session_len=8192, cache_max_entry_count=0.5, quant_policy=0)
        return original_create(self, backend, model_path, server_url, **kwargs)
    ModelSingleton._create_model = resource_config
    from mineru.model.vlm.client import get_vlm_predictor
    predictor, backend = get_vlm_predictor(config.model.vlm)
    predictor.client.use_tqdm = False
    engine = predictor._mineru_runtime_handles['lmdeploy_engine']
    import inspect
    actual = {'requested_backend':'turbomind','actual_backend':engine.async_engine.backend,
              'pipeline_class':type(engine).__name__, 'engine_class':type(engine.async_engine.engine).__name__,
              'config_class':type(engine.backend_config).__name__,
              'engine_source':inspect.getfile(type(engine.async_engine.engine)),
              'config_source':inspect.getfile(type(engine.backend_config)),
              'backend_config':vars(engine.backend_config),
              'native_model_class':type(getattr(engine.async_engine.engine,'model_comm',None)).__name__,
              'source_model_class':type(getattr(engine.async_engine.engine,'source_model',None)).__name__,
              'turbomind_pyd':getattr(sys.modules.get('_turbomind'),'__file__',None)}
    write(out/'actual-backend.json',actual)
    write(out/'loaded-linux-libraries.json',{'maps':Path('/proc/self/maps').read_text(),'ldd':subprocess.run(['ldd',actual['turbomind_pyd']],capture_output=True,text=True).__dict__ if actual['turbomind_pyd'] else None})
    if actual['actual_backend'] != 'turbomind' or actual['config_class'] != 'TurbomindEngineConfig':
        raise RuntimeError('Refusing LMDeploy silent backend fallback: '+str(actual))
    effective_engine = engine.async_engine.engine.engine_config
    write(out/'effective-engine-config.json',vars(effective_engine))
    assert effective_engine.dtype == 'bfloat16' and effective_engine.max_batch_size == 1
    assert effective_engine.quant_policy == 0, 'Quantized inference differs from the frozen profile'
    assert effective_engine.session_len == 8192 and effective_engine.cache_max_entry_count == 0.5
    original_infer = engine.infer
    calls = []
    effective_configs = []
    async_engine = engine.async_engine
    original_determine = async_engine._determine_gen_config
    def capture_determine(input_ids, gen_config=None):
        result = original_determine(input_ids, gen_config)
        effective_configs.append({'input_tokens':len(input_ids), 'parameters':dict(vars(result)),
                                  'tokenizer_eos_token_id':async_engine.tokenizer.eos_token_id,
                                  'hf_generation_config':async_engine.hf_gen_cfg})
        write(out/'effective-generation-configs.json',effective_configs)
        return result
    async_engine._determine_gen_config = capture_determine
    def capture_infer(*inputs, **options):
        start = time.monotonic()
        first_config = len(effective_configs)
        responses = original_infer(*inputs, **options)
        calls.append({'seconds':time.monotonic()-start,
                      'responses':[{key:getattr(response,key,None) for key in
                         ['text','generate_token_len','input_token_len','finish_reason','token_ids','index']}
                         for response in responses],
                      'generation_configs':[vars(c) for c in options.get('gen_config',[])],
                      'effective_generation_configs':effective_configs[first_config:]})
        write(out/'generation-calls.json',calls)
        return responses
    engine.infer = capture_infer
    write(out / 'backend.json', {'backend': backend, 'tier': 'advanced', 'effort': 'xhigh',
                               'dtype_requested': 'bfloat16', 'batch_size': 1, 'max_batch_size': 1,
                               'image_analysis': True, 'ocr_mode': 'ocr'})
    from mineru.parser import parse
    from mineru.parser.writer import FileBasedDataWriter
    def run(page, page_out):
        first = len(calls)
        result = parse(page['image_path'], tier='advanced', ocr_mode='ocr',
                       image_analysis=True, vlm_config=config.model.vlm)
        result.save(FileBasedDataWriter(str(page_out)))
        write(page_out/'generation-calls.json',calls[first:])
        markdown = (page_out/'markdown.md').read_bytes().decode('utf-8')
        return markdown
    run._capture_buffers = [calls,effective_configs]
    return run


def load_tele(root, out):
    source = root / 'models/tele-source/TeleOCR-9921cffe380efe4e2fa010258b3d0c3cb70bab2d'
    sys.path.insert(0, str(source))
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from TeleOCR.vlm_utils.TeleOCR_client import TeleOCRClient
    from TeleOCR.engine import do_parse
    from TeleOCR.tools.read_file import read_fn
    from TeleOCR import config
    config.PDF_TOOLS_WORKER_MAX_NUM = 4
    config.BACKEND = 'transformers'
    config.model_path = str(root / 'models/tele')
    processor = AutoProcessor.from_pretrained(config.model_path, trust_remote_code=True, local_files_only=True)
    model, info = AutoModelForImageTextToText.from_pretrained(config.model_path, trust_remote_code=True,
                      local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa', output_loading_info=True)
    write(out / 'loading-info.json', info)
    if any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):
        raise RuntimeError('Unexplained TeleOCR loading keys')
    model = model.cuda().eval()
    write(out/'model-generation-config.json', model.generation_config.to_dict())
    predictor = TeleOCRClient(backend='transformers', model=model, processor=processor,
                              batch_size=1, max_concurrency=1)
    calls = []
    original_generate = model.generate
    def capture_generate(*args, **kwargs):
        start = time.monotonic()
        result = original_generate(*args, **kwargs)
        input_ids = kwargs.get('input_ids')
        n = input_ids.shape[-1] if input_ids is not None else 0
        rows = result.sequences if hasattr(result, 'sequences') else result
        generated = rows[:, n:]
        eos = kwargs.get('eos_token_id', model.generation_config.eos_token_id)
        eos = eos if isinstance(eos, list) else [eos]
        token_rows = generated.tolist()
        cap = kwargs.get('max_new_tokens') or max(0, kwargs.get('max_length', model.generation_config.max_length)-n)
        record = {'seconds': time.monotonic()-start, 'input_tokens': n,
                  'output_tokens': [len(row)-n for row in rows],
                  'token_ids': token_rows,
                  'raw': processor.batch_decode(generated, skip_special_tokens=True),
                  'raw_with_special_tokens': processor.batch_decode(generated, skip_special_tokens=False),
                  'effective_eos_token_id': eos,
                  'effective_pad_token_id': kwargs.get('pad_token_id',model.generation_config.pad_token_id),
                  'effective_output_token_cap': cap,
                  'stop_reasons':['eos' if any(t in eos for t in tokens) else 'length' if len(tokens)>=cap else 'other'
                                  for tokens in token_rows],
                  'generation_parameters': {k: v for k, v in kwargs.items() if isinstance(v, (int,float,bool,str,type(None)))}}
        calls.append(record)
        write(out / 'generation-calls.json', calls)
        return result
    model.generate = capture_generate
    def run(page, page_out):
        start_call = len(calls)
        result = do_parse(str(page_out), [page['page_id']], [read_fn(page['image_path'])],
                          valid_page_ids=[None], predictor=predictor)
        write(page_out / 'native.json', result)
        write(page_out / 'generation-calls.json', calls[start_call:])
        return (page_out / page['page_id'] / (page['page_id'] + '.md')).read_text(encoding='utf-8')
    run._capture_buffers = [calls]
    return run


def worker(args):
    from device_proof import verify_device
    from capture_utils import persist_buffers
    verify_device(Path(args.output))
    root, out = Path(args.root), Path(args.output)
    stage = out / 'stage.json'
    write(stage, {'phase': 'load', 'started': time.time()})
    import torch
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), 'GPU required; no CPU fallback'
    params = {'arm': args.arm, 'adapter_sha256': digest(Path(__file__)),
              'supervisor_sha256': digest(Path(__file__).with_name('complementarity_ovis_smoke.py')),
              'manifest_sha256': digest(Path(args.manifest)), 'batch_size': 1,
              'tier': 'advanced/xhigh' if args.arm == 'mineru' else 'native whole-page',
              'dtype': 'bfloat16', 'torch': torch.__version__,
              'packages': {x: importlib.metadata.version(x) for x in ['transformers', 'pillow']}}
    write(out / 'parameters.json', params)
    start = time.monotonic()
    run = {'mineru': load_mineru, 'tele': load_tele}[args.arm](root, out)
    torch.cuda.synchronize()
    write(out / 'loaded.json', {'seconds': time.monotonic()-start})
    previous_error = None
    for page in json.loads(Path(args.manifest).read_text(encoding='utf-8'))['pages']:
        page_out = out / component_name(page['page_id'])
        page_out.mkdir()
        write(out/'events'/component_name(page['page_id'], '.started.json'),{'page_id':page['page_id'],'started':time.time()})
        write(stage, {'phase': 'page', 'page_id': page['page_id'], 'started': time.time()})
        assert digest(Path(page['image_path'])) == page['input_sha256']
        write(out/'generation-calls.json',[])
        if args.arm=='mineru':write(out/'effective-generation-configs.json',[])
        start = time.monotonic()
        record = {'page_id': page['page_id'], 'input_sha256': page['input_sha256'],
                  'params_sha256': digest(out / 'parameters.json')}
        try:
            torch.cuda.reset_peak_memory_stats()
            markdown = run(page, page_out)
            torch.cuda.synchronize()
            (page_out / 'prediction.md').write_bytes(markdown.encode('utf-8'))
            if args.arm=='mineru':assert digest(page_out/'prediction.md')==digest(page_out/'markdown.md')
            record.update(status='returned', nonempty=bool(markdown.strip()), characters=len(markdown),
                          generation_calls=len(json.loads((page_out/'generation-calls.json').read_text(encoding='utf-8'))),
                          peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(),
                          prediction_sha256=digest(page_out / 'prediction.md'))
            previous_error = None
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
            (page_out / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
            record.update(status='failed', error=error,
                          arm_paused=isinstance(exc, torch.OutOfMemoryError) or error == previous_error)
            previous_error = error
        finally:
            persist_buffers(page_out,run._capture_buffers)
        record['seconds'] = time.monotonic()-start
        write(page_out / 'receipt.json', record)
        print(json.dumps(record), flush=True)
        for buffer in run._capture_buffers:buffer.clear()
        if record.get('arm_paused'):
            break
    write(stage, {'phase': 'finished', 'started': time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--arm', required=True, choices=['mineru', 'tele'])
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    worker(args) if args.worker else supervise(args, __file__)
