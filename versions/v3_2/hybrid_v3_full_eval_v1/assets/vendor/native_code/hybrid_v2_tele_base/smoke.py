"""Real official whole-page smoke; failures remain failures, no full-run entry here."""
import argparse
import base64
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from .region_protocol import RunBlocked, image_binding
from .audit import install_tele_audit, audit_snapshot
from .tele_adapter import make_client_class
from .expert_process import ExpertProcess
from .runtime_resources import ResourceAudit


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8*1024*1024): h.update(chunk)
    return h.hexdigest()


def save(path, value):
    serialized = json.dumps(audit_snapshot(value), ensure_ascii=False, indent=2, allow_nan=False)
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(serialized)


def deadline(seconds):
    def expired(*_): raise RunBlocked('Owned smoke phase deadline exceeded')
    signal.signal(signal.SIGALRM, expired); signal.alarm(seconds)


def persist_page_audit(folder, records, primary_error=None, primary_traceback=None):
    audit_errors = []
    for filename,value in records:
        try: save(folder/filename,value)
        except BaseException as exc:
            audit_errors.append({'file':filename,'error_type':type(exc).__name__,'error':str(exc),
                                 'traceback':traceback.format_exc()})
    if primary_error is not None or audit_errors:
        save(folder/'PAGE_ERRORS.json',{'primary_error':None if primary_error is None else
            {'type':type(primary_error).__name__,'error':str(primary_error),'traceback':primary_traceback},
            'audit_errors':audit_errors})
        if primary_error is not None: raise primary_error
        raise RunBlocked('Page inference returned but audit persistence failed; see PAGE_ERRORS.json')


def observe(client, folder, captured):
    """Observers return original objects and do not perform crop/rotate/resize again."""
    import numpy as np
    helper = client.helper
    prepare, post = helper.prepare_for_extract, helper.post_process
    def preparation(image, blocks, not_extract_list=None):
        returned = prepare(image, blocks, not_extract_list)
        crops, prompts, params, indices = returned
        width, height = image.size
        if width*height > 64000000:
            scale = (64000000/(width*height))**0.5
            width, height = max(1, round(width*scale)), max(1, round(height*scale))
        records = []
        for crop, index in zip(crops, indices):
            pts = np.array(blocks[index].bbox, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= width; pts[:, 1] *= height
            row = {'original_block_index': index, 'prepared_crop': image_binding(crop),
                   'normalized_bbox': blocks[index].bbox, 'angle': blocks[index].angle,
                   'source_size': [width, height], 'integer_points_from_frozen_helper_rule': pts.astype(np.int32).tolist()}
            crop.save(folder / ('crop-%04d.png' % index))
            records.append(row)
        captured['prepared'] = records
        captured['layout'] = copy.deepcopy(blocks)
        return returned
    def postprocess(blocks):
        captured['before_postprocess'] = copy.deepcopy(blocks)
        returned = post(blocks)
        captured['after_postprocess'] = copy.deepcopy(returned)
        return returned
    helper.prepare_for_extract, helper.post_process = preparation, postprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    a = p.parse_args()
    manifest = json.loads(Path(a.manifest).read_bytes())
    out = Path('/output')
    summary = {'passed':False,'real_gpu':True,'model_loads':{'tele':0,'paddle':0},'checks':{},'pages':[]}
    expert = None; resources = None
    try:
        for row in manifest['identity_files']:
            if sha(row['path']) != row['sha256']: raise RunBlocked('Source/model/aux identity mismatch: '+row['path'])
        if not manifest['capacity']['passed'] or manifest['capacity']['conservative_upper_bound'] > 4000000000:
            raise RunBlocked('Capacity not bounded')
        sys.path.insert(0, manifest['tele_source'])
        resources = ResourceAudit(manifest['identity_files'])
        resources.bind_tele_loaders()
        for name, version in manifest['tele_packages'].items():
            if importlib.metadata.version(name) != version: raise RunBlocked('Package mismatch: '+name)
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        save(out/'ENVIRONMENT.json', {'packages':{d.metadata['Name']:{'version':d.version,'path':str(d._path)}
             for d in importlib.metadata.distributions()},'sys_path':sys.path})
        if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
            raise RunBlocked('Expected one visible BF16-capable GPU')
        gpu = torch.cuda.get_device_properties(0)
        actual_uuid = str(gpu.uuid)
        if actual_uuid.removeprefix('GPU-').lower() != manifest['gpu_uuid'].removeprefix('GPU-').lower():
            raise RunBlocked('Actual CUDA UUID differs from host lease')
        summary['gpu'] = {'uuid':actual_uuid,'name':gpu.name,'torch_cuda':torch.version.cuda}
        import TeleOCR.config as config
        config.BACKEND = 'transformers'; config.model_path = manifest['tele_model']
        config.LAYOUT_MODE = 'Detection'; config.PDF_TOOLS = 'pypdfium2'
        config.PDF_TOOLS_WORKER_MAX_NUM = 4; config.MAX_PIXELS = 64000000
        from TeleOCR.tools.read_file import read_fn
        from TeleOCR.engine import do_parse
        from TeleOCR.vlm_utils.TeleOCR_client import TeleOCRClient
        deadline(600)
        processor = AutoProcessor.from_pretrained(manifest['tele_model'], trust_remote_code=True, local_files_only=True)
        save(out/'PROCESSOR.json', {'class':type(processor).__name__,
            'image_processor_class':type(processor.image_processor).__name__,
            'image_processor_config':processor.image_processor.to_dict(),
            'tokenizer_class':type(processor.tokenizer).__name__,
            'tokenizer_init_kwargs':processor.tokenizer.init_kwargs,
            'chat_template':processor.chat_template})
        model, loading = AutoModelForImageTextToText.from_pretrained(manifest['tele_model'],
            trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', output_loading_info=True)
        for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'):
            if key not in loading or loading[key]: raise RunBlocked('Nonempty or missing loading audit: '+key)
        model = model.to('cuda:0').eval(); signal.alarm(0)
        if type(model).__name__ != 'Qwen2_5_VLForConditionalGeneration': raise RunBlocked('Tele model class differs')
        if any(x.dtype != torch.bfloat16 for x in model.parameters() if x.is_floating_point()):
            raise RunBlocked('Tele parameter dtype mismatch')
        summary['model_loads']['tele'] = 1
        save(out/'TELE_LOAD.json', {'loading_info':loading,'generation_config':model.generation_config.to_dict(),
            'config':model.config.to_dict(),'parameter_count':sum(p.numel() for p in model.parameters()),
            'model_class':type(model).__name__,'dtype':str(model.dtype)})
        generation = []
        install_tele_audit(model, generation.append)
        hybrid_class = make_client_class(TeleOCRClient)
        controls = {}
        for mode in ('official','off','pass-through','on'):
            if mode == 'on':
                expert = ExpertProcess(manifest['paddle_python'], manifest['paddle_model'],
                    out/'paddle-worker', '/code', load_timeout=600, region_timeout=180,
                    smoke_controls=True, manifest=a.manifest)
                summary['model_loads']['paddle'] = expert.load['model_loads']
            for page in manifest['pages']:
                folder = out/mode/page['id']; folder.mkdir(parents=True, exist_ok=False)
                if sha(page['path']) != page['sha256']: raise RunBlocked('Input identity mismatch')
                events, captured = [], {}
                kwargs = dict(backend='transformers',model=model,processor=processor,batch_size=1,max_concurrency=1)
                client = TeleOCRClient(**kwargs) if mode == 'official' else hybrid_class(**kwargs,
                    hybrid_mode=mode, formula_expert=expert, emit=events.append)
                if mode != 'official': client.set_context(manifest['run_id'],page['id'],page['sha256'])
                observe(client, folder, captured)
                start = len(generation); started = time.monotonic()
                primary_error = None; primary_traceback = None
                try:
                    deadline(900)
                    result = do_parse(str(folder/'native'), [page['id']], [read_fn(page['path'])],
                                      [None], predictor=client)
                except BaseException as exc:
                    primary_error = exc; primary_traceback = traceback.format_exc()
                finally:
                    signal.alarm(0)
                persist_page_audit(folder,[('GENERATION.json',generation[start:]),('REGIONS.json',events),
                    ('BLOCKS.json',captured)],primary_error,primary_traceback)
                md = (folder/'native'/page['id']/(page['id']+'.md')).read_bytes()
                normalized = {'blocks':captured,'markdown_sha256':hashlib.sha256(md).hexdigest(), 'middle':result}
                if mode == 'official': controls[page['id']] = normalized
                elif mode in ('off','pass-through'):
                    if normalized != controls[page['id']]: raise RunBlocked(mode+' differs from official whole-page output')
                else:
                    original = controls[page['id']]['blocks']
                    if captured['layout'] != original['layout'] or captured['prepared'] != original['prepared']:
                        raise RunBlocked('Hybrid layout or crop preparation differs')
                    before = original['before_postprocess']; after = captured['before_postprocess']
                    if len(before) != len(after): raise RunBlocked('Hybrid block count changed')
                    for x,y in zip(before,after):
                        if {k:v for k,v in x.items() if k!='content'} != {k:v for k,v in y.items() if k!='content'}:
                            raise RunBlocked('Hybrid metadata changed')
                        if x['type'] != 'equation' and x != y: raise RunBlocked('Non-equation output changed')
                summary['pages'].append({'mode':mode,'page':page['id'],'seconds':time.monotonic()-started,
                    'paddle_loads_so_far':summary['model_loads']['paddle'],
                    'routes':[{k:e[k] for k in ('route','block_index') if k in e} for e in events if e.get('event')=='route']})
        routes = [r for p in summary['pages'] if p['mode']=='on' for r in p['routes']]
        count = sum(r.get('route')=='paddle_success' for r in routes)
        zero = any(p['mode']=='on' and not any(r.get('route','').startswith('paddle') or r.get('route')=='tele_fallback'
                   for r in p['routes']) for p in summary['pages'])
        summary['checks'].update(official_off_pass_through_equal=True, nonformula_equal=True,
            paddle_success_regions=count, zero_route_page=zero,
            peak_tele_allocated_bytes=torch.cuda.max_memory_allocated(), peak_tele_reserved_bytes=torch.cuda.max_memory_reserved())
        if count < 3 or len(manifest['pages']) < 2 or not zero: raise RunBlocked('Real formula/zero-route coverage incomplete')
        references = []
        requests = sorted((out/'paddle-worker').glob('region-*.request.json'))
        for request_path in requests:
            original_request = json.loads(request_path.read_bytes())
            prefix = request_path.name.removesuffix('.request.json')
            response = json.loads((request_path.parent/(prefix+'.response.json')).read_bytes())
            original_request['png_base64'] = base64.b64encode((request_path.parent/(prefix+'.png')).read_bytes()).decode('ascii')
            original_request['op'] = 'native_control'
            control = expert.recognize(original_request)
            if control.get('status') != 'native_control' or response.get('status') != 'ok':
                raise RunBlocked('Direct-control comparison requires successful wrapped/native results')
            if control['raw_text'] != response['raw_text']:
                raise RunBlocked('Direct native/wrapper text mismatch')
            for key in ('input_tensors','token_ids','eos_ids','budget','stop'):
                if control['generation'][key] != response['generation'][key]:
                    raise RunBlocked('Direct native/wrapper audit mismatch: '+key)
            references.append({'region_id':control['region_id'],'same_text_tokens_and_tensors':True})
        save(out/'DIRECT_CONTROL.json',references)
        resource_result = resources.result()
        save(out/'RESOURCE_AUDIT.json',resource_result)
        if resource_result['unknown_resources']: raise RunBlocked('Unknown runtime resources require capacity audit')
        summary['checks']['paddle_direct_control_passed'] = len(references) >= 3
        summary['coverage_passed'] = True
        summary['passed'] = len(references) >= 3 and summary['model_loads'] == {'tele':1,'paddle':1}
    except BaseException as exc:
        summary.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
        if expert is not None:
            expert.close(force=not summary['passed'])
            if summary['passed'] and expert.process.returncode != 0:
                summary.update(passed=False,error='Paddle worker did not shut down cleanly')
        if resources is not None and not (out/'RESOURCE_AUDIT.json').exists():
            save(out/'RESOURCE_AUDIT.json',resources.result())
        save(out/'SMOKE_RESULT.json',summary)
    return 0 if summary['passed'] else 1


if __name__ == '__main__': raise SystemExit(main())
