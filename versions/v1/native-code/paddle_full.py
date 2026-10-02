"""Official PaddleOCRVL v1.6 whole-page native pipeline, GT-free inputs."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
from page_paths import component_name
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from complementarity_ovis_smoke import digest, supervise, write


def worker(args):
    from device_proof import verify_device
    from capture_utils import persist_buffers
    verify_device(Path(args.output))
    root, out = Path(args.root), Path(args.output)
    stage = out/'stage.json'
    write(stage, {'phase':'load','started':time.time()})
    os.environ['OMP_NUM_THREADS'] = '1'
    import paddle
    import paddlex
    import yaml
    from paddleocr import PaddleOCRVL
    assert paddle.is_compiled_with_cuda(), 'CUDA Paddle required'
    paddle.set_device('gpu:0')
    config_path = Path(paddlex.__file__).parent/'configs/pipelines/PaddleOCR-VL-1.6.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    config['batch_size'] = 1
    config['SubModules']['LayoutDetection']['batch_size'] = 1
    config['SubModules']['LayoutDetection']['model_dir'] = str(root/'models/paddle_layout')
    config['SubModules']['VLRecognition']['batch_size'] = 1
    config['SubModules']['VLRecognition']['model_dir'] = str(root/'models/paddle')
    config['use_queues'] = False  # single-page serial resource scheduling
    write(out/'parameters.json', {'model':'PaddleOCR-VL-1.6',
          'revision':'c5630abae1d940eafe0697512a0325494b02ab42',
          'layout_revision':'7b48a7566925fa464281f930c58eee04fe2c862a',
          'paddle':paddle.__version__, 'paddleocr':importlib.metadata.version('paddleocr'),
          'paddlex':importlib.metadata.version('paddlex'), 'pipeline_config':config,
          'adapter_sha256':digest(Path(__file__)), 'manifest_sha256':digest(Path(args.manifest)),
          'conversion':'official PaddleX convert_from_hf=True; no custom conversion'})
    from paddlex.inference.models.doc_vlm.predictor import DocVLMLocalPredictor
    original_build = DocVLMLocalPredictor._build
    calls = []
    def capture_build(self, **kwargs):
        model, processor = original_build(self, **kwargs)
        if self.dtype != 'bfloat16':
            raise RuntimeError('Expected native GPU bfloat16; refusing silent dtype fallback: '+self.dtype)
        original_generate = model.generate
        native_generation = getattr(model,'generation_config',None)
        native_dict = native_generation.to_dict() if hasattr(native_generation,'to_dict') else {}
        write(out/'model-generation-config.json',native_dict)
        def generate(*inputs, **options):
            start = time.monotonic()
            result = original_generate(*inputs, **options)
            tokens = result[0] if isinstance(result, tuple) else result
            token_rows = tokens.tolist() if hasattr(tokens,'tolist') else []
            eos = options.get('eos_token_id',native_dict.get('eos_token_id',model.config.eos_token_id))
            eos = eos if isinstance(eos,list) else [eos]
            cap = options.get('max_new_tokens',native_dict.get('max_new_tokens'))
            record = {'seconds':time.monotonic()-start,
                      'tokens':token_rows,
                      'effective_eos_token_id':eos,
                      'effective_pad_token_id':options.get('pad_token_id',native_dict.get('pad_token_id',model.config.pad_token_id)),
                      'effective_output_token_cap':cap,
                      'stop_reasons':['eos' if any(t in eos for t in row) else 'length' if cap is not None and len(row)>=cap else 'other'
                                      for row in token_rows],
                      'parameters':{k:v for k,v in options.items() if isinstance(v,(str,int,float,bool,type(None)))}}
            calls.append(record)
            write(out/'generation-calls.json',calls)
            return result
        model.generate = generate
        write(out/'loaded-model-details.json', {'dtype':self.dtype,'model_class':type(model).__name__,
                                               'processor_class':type(processor).__name__})
        return model, processor
    DocVLMLocalPredictor._build = capture_build
    start = time.monotonic()
    pipeline = PaddleOCRVL(pipeline_version='v1.6', paddlex_config=config, device='gpu:0')
    paddle.device.cuda.synchronize()
    write(out/'loaded.json', {'seconds':time.monotonic()-start})
    pipeline.export_paddlex_config_to_yaml(str(out/'effective-pipeline.yaml'))
    previous = None
    for page in json.loads(Path(args.manifest).read_text(encoding='utf-8'))['pages']:
        folder = out/component_name(page['page_id'])
        folder.mkdir()
        write(out/'events'/component_name(page['page_id'], '.started.json'),{'page_id':page['page_id'],'started':time.time()})
        write(stage, {'phase':'page','page_id':page['page_id'],'started':time.time()})
        assert digest(Path(page['image_path'])) == page['input_sha256']
        record = {'page_id':page['page_id'],'input_sha256':page['input_sha256'],
                  'params_sha256':digest(out/'parameters.json')}
        write(out/'generation-calls.json',[])
        start, first = time.monotonic(), len(calls)
        try:
            paddle.device.cuda.reset_max_memory_allocated()
            paddle.device.cuda.reset_max_memory_reserved()
            results = list(pipeline.predict(page['image_path']))
            assert len(results) == 1, 'Expected one whole-page result'
            result = results[0]
            result.save_to_json(str(folder/'native.json'))
            result.save_to_markdown(str(folder))
            markdown = result.markdown['markdown_texts']
            (folder/'prediction.md').write_text(markdown,encoding='utf-8')
            paddle.device.cuda.synchronize()
            write(folder/'generation-calls.json',calls[first:])
            record.update(status='returned',nonempty=bool(markdown.strip()),characters=len(markdown),
                          generation_calls=len(calls)-first,prediction_sha256=digest(folder/'prediction.md'),
                          peak_paddle_allocated_bytes=paddle.device.cuda.max_memory_allocated(),
                          peak_paddle_reserved_bytes=paddle.device.cuda.max_memory_reserved())
            previous = None
        except Exception as exc:
            error = type(exc).__name__+': '+str(exc)
            (folder/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
            record.update(status='failed',error=error,arm_paused=error==previous or 'out of memory' in error.lower())
            previous = error
        finally:
            persist_buffers(folder,[calls[first:]])
        record['seconds'] = time.monotonic()-start
        write(folder/'receipt.json',record)
        print(json.dumps(record),flush=True)
        calls.clear()
        if record.get('arm_paused'):
            break
    write(stage, {'phase':'finished','started':time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ['root','manifest','output']:
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--worker',action='store_true')
    args = parser.parse_args()
    worker(args) if args.worker else supervise(args,__file__)
