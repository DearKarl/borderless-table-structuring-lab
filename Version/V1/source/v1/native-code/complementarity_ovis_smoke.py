"""GT-free Ovis whole-page smoke, native Transformers variant (amendment E)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from page_paths import atomic_write_json, component_name


def write(path, value):
    atomic_write_json(path, value)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(text):
    text = '\n\n'.join(x for x in text.strip().split('\n\n')
                       if not x.strip().startswith('<img src="images/bbox_'))
    n = len(text)
    if n < 8000:
        return text
    for unit_len in range(1, min(200, n - 1) + 1):
        if text[n - 1] != text[n - 1 - unit_len]:
            continue
        match_len, idx = 1, n - 2
        while idx >= unit_len and text[idx] == text[idx - unit_len]:
            match_len += 1
            idx -= 1
        total_len = match_len + unit_len
        if total_len // unit_len >= 5 and total_len >= 100:
            return text[:n - total_len + unit_len] + text[n - total_len % unit_len:]
    return text


def worker(args):
    root, out = Path(args.root), Path(args.output)
    stage = out / 'stage.json'
    write(stage, {'phase': 'load', 'started': time.time()})
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from PIL import Image
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), 'GPU required; CPU fallback forbidden'
    model_path = root / 'models/ovis'
    prompt = '\nExtract all readable content from the image in natural human reading order and output the result as a single Markdown document. For charts or images, represent them using an HTML image tag: <img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, where left, top, right, bottom are bounding box coordinates scaled to [0, 1000). Format formulas as LaTeX. Format tables as HTML: <table>...</table>. Transcribe all other text as standard Markdown. Preserve the original text without translation or paraphrasing.'
    params = {'repo': 'ATH-MaaS/OvisOCR2', 'revision': '1fc9221b7823a371d6e97f92d527cc847e24e107',
              'backend': 'Windows native Transformers Qwen3_5ForConditionalGeneration',
              'torch': torch.__version__, 'transformers': transformers.__version__,
              'dtype': 'bfloat16', 'attention': 'sdpa', 'gdn': 'transformers torch reference GPU',
              'max_new_tokens': 16384, 'do_sample': False, 'enable_thinking': False,
              'eos_token_id': [248046, 248044], 'pad_token_id': 248044,
              'stop_alignment': 'vLLM 0.22.1 tokenizer EOS union GenerationConfig EOS',
              'min_pixels': 448**2, 'max_pixels': 2880**2, 'batch_size': 1,
              'prompt': prompt, 'adapter_sha256': digest(Path(__file__)),
              'manifest_sha256': digest(Path(args.manifest))}
    write(out / 'parameters.json', params)
    params_hash = digest(out / 'parameters.json')
    started = time.monotonic()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation='sdpa',
        local_files_only=True, output_loading_info=True)
    write(out / 'loading-info.json', loading)
    if loading.get('missing_keys') or loading.get('unexpected_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):
        raise RuntimeError('Unexplained loading keys; owner review required')
    model = model.to('cuda').eval()
    write(out / 'eos-audit.json', {'loaded_generation_config_eos': model.generation_config.eos_token_id,
                                 'tokenizer_eos': processor.tokenizer.eos_token_id,
                                 'tokenizer_pad': processor.tokenizer.pad_token_id,
                                 'explicit_eos': [248046, 248044]})
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - started
    write(out / 'loaded.json', {'seconds': load_seconds, 'device': torch.cuda.get_device_name(0),
                               'dtype': str(next(model.parameters()).dtype)})
    pages = json.loads(Path(args.manifest).read_text(encoding='utf-8'))['pages']
    prompt_text = processor.apply_chat_template(
        [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    (out / 'rendered-prompt.txt').write_text(prompt_text, encoding='utf-8')
    previous_failure = None
    for page in pages:
        page_id = page['page_id']
        page_out = out / component_name(page_id)
        page_out.mkdir()
        write(stage, {'phase': 'page', 'page_id': page_id, 'started': time.time()})
        assert digest(Path(page['image_path'])) == page['input_sha256']
        record = {'page_id': page_id, 'model': 'O', 'input_sha256': page['input_sha256'],
                  'params_sha256': params_hash, 'generation_calls': 0, 'load_seconds': load_seconds}
        start = time.monotonic()
        try:
            torch.cuda.reset_peak_memory_stats()
            image = Image.open(page['image_path']).convert('RGB')
            inputs = processor(text=[prompt_text], images=[image], return_tensors='pt',
                               images_kwargs={'min_pixels': 448**2, 'max_pixels': 2880**2}).to('cuda')
            record['image_grid_thw'] = inputs['image_grid_thw'].tolist()
            record['input_tokens'] = inputs['input_ids'].shape[1]
            record['generation_calls'] = 1
            from transformers.generation.streamers import BaseStreamer
            class TokenRecorder(BaseStreamer):
                def __init__(self, path):
                    self.file = path.open('w', encoding='utf-8', buffering=1)
                    self.first = True
                def put(self, value):
                    if self.first:
                        self.first = False
                        return
                    self.file.write(json.dumps(value.tolist()) + '\n')
                def end(self):
                    self.file.close()
            streamer = TokenRecorder(page_out / 'generated-tokens.partial.jsonl')
            with torch.inference_mode():
                try:
                    output = model.generate(**inputs, do_sample=False, max_new_tokens=16384,
                                            eos_token_id=[248046, 248044], pad_token_id=248044,
                                            streamer=streamer)
                finally:
                    streamer.end()
            torch.cuda.synchronize()
            generated = output[0, inputs['input_ids'].shape[1]:]
            tokens = generated.tolist()
            write(page_out / 'generated-tokens.json', tokens)
            stop_positions = [i for i, token in enumerate(tokens) if token in [248046, 248044]]
            (page_out / 'raw-with-special-tokens.txt').write_text(
                processor.decode(generated, skip_special_tokens=False), encoding='utf-8')
            raw = processor.decode(generated, skip_special_tokens=True)
            (page_out / 'raw.txt').write_text(raw, encoding='utf-8')
            (page_out / 'prediction.md').write_text(clean(raw), encoding='utf-8')
            record.update(status='success', output_tokens=len(generated),
                          truncated=len(generated) >= 16384,
                          first_stop_position=stop_positions[0] if stop_positions else None,
                          stop_token=tokens[stop_positions[0]] if stop_positions else None,
                          stop_reason='eos' if stop_positions else 'token_cap',
                          peak_vram_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                          raw_sha256=digest(page_out / 'raw.txt'),
                          prediction_sha256=digest(page_out / 'prediction.md'))
            previous_failure = None
            del inputs, output, generated
        except Exception as exc:
            failure = type(exc).__name__ + ': ' + str(exc)
            (page_out / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
            record.update(status='failed', error=failure)
            repeat = previous_failure == failure
            previous_failure = failure
            if isinstance(exc, torch.OutOfMemoryError) or repeat:
                record['arm_paused'] = True
        record['seconds'] = time.monotonic() - start
        write(page_out / 'receipt.json', record)
        print(json.dumps(record), flush=True)
        if record.get('arm_paused'):
            break
    write(stage, {'phase': 'finished', 'started': time.time()})


def supervise(args, worker_script=None):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'adapter-source.py').write_bytes(Path(worker_script or __file__).read_bytes())
    if worker_script:
        (out / 'supervisor-source.py').write_bytes(Path(__file__).read_bytes())
    root = Path(args.root)
    env = dict(os.environ)
    for variable, folder in [('TRITON_CACHE_DIR', 'triton'), ('TRITON_DUMP_DIR', 'triton-dump'),
                             ('TRITON_OVERRIDE_DIR', 'triton-override'),
                             ('TORCHINDUCTOR_CACHE_DIR', 'torchinductor'), ('CUDA_CACHE_PATH', 'cuda')]:
        cache = root / 'cache' / folder
        cache.mkdir(parents=True, exist_ok=True)
        env[variable] = str(cache)
    env.update(HF_HOME=str(root / 'cache/huggingface'), HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
               TORCH_HOME=str(root / 'cache/torch'), OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
               MINERU_HOME=str(root / 'cache/mineru'), PADDLE_PDX_CACHE_HOME=str(root / 'cache/paddlex'),
               PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True',
               TEMP=str(root / 'downloads'), TMP=str(root / 'downloads'))
    cmd = [sys.executable, '-I', '-B', '-X', 'utf8', str(Path(worker_script or __file__).resolve()), '--worker',
           '--root', args.root, '--manifest', args.manifest, '--output', args.output]
    if getattr(args, 'arm', None):
        cmd += ['--arm', args.arm]
    with (out / 'worker.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        launched = time.time()
        timeout = None
        last_gpu_sample = 0
        gpu_samples = []
        while process.poll() is None:
            stage = {'phase': 'load', 'started': launched}
            try:
                stage = json.loads((out / 'stage.json').read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            if time.time() - last_gpu_sample >= 5:
                try:
                    sample = subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
                                             '--format=csv,noheader,nounits'], capture_output=True,
                                            text=True, timeout=4)
                    gpu_samples.append({'utc_epoch':time.time(),'stage':stage,'output':sample.stdout.strip(),
                                        'exit_code':sample.returncode})
                    write(out/'gpu-system-samples.json',gpu_samples)
                except (OSError, subprocess.TimeoutExpired):
                    pass
                last_gpu_sample = time.time()
            limit = 900 if stage['phase'] == 'page' else 600
            if time.time() - stage['started'] > limit:
                timeout = stage
                if os.name == 'nt':
                    # Windows venv python.exe can be a redirector with a real
                    # interpreter child. Terminate this owned tree, not just it.
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                   stdout=log, stderr=subprocess.STDOUT, check=False)
                else:
                    process.kill()
                process.wait()
                break
            time.sleep(2)
    write(out / 'supervisor.json', {'command': cmd, 'exit_code': process.returncode,
                                  'timeout': timeout, 'wall_seconds': time.time() - launched})
    print(json.dumps({'exit_code': process.returncode, 'timeout': timeout}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--worker', action='store_true')
    arguments = parser.parse_args()
    worker(arguments) if arguments.worker else supervise(arguments)
