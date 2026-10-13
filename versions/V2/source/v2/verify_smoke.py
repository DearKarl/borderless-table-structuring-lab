"""Independent read-only evidence verification. Does not import either model framework."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import traceback
from PIL import Image


def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise ValueError(message)


def verify(bundle,output):
    manifest=read(bundle/'SMOKE_MANIFEST.json')
    for relative,digest in read(bundle/'CODE_LOCK.json').items():
        p=(bundle/relative).resolve()
        require(p.is_relative_to(bundle.resolve()) and sha(p)==digest,'Frozen package identity')
    finish=read(output/'EXIT.json')
    state=finish['container']['State'];host=finish['container']['HostConfig']
    require(finish['cli_exit']==0 and not finish['timeout'] and not finish['foreign_processes'],'Supervisor failed')
    require(state['ExitCode']==0 and not state['Running'] and not state['OOMKilled'],'Container failure')
    require(host['NetworkMode']=='none' and host['ReadonlyRootfs'] and not host['Privileged'],'Container isolation')
    require(finish['container']['Image']=='sha256:787dd9a15d41c442e78a5faa9cdc721f154cbf3d4cf8624edb41026c3052b75a','Image changed')
    start=read(output/'START.json')
    require(start['manifest_sha256']==sha(bundle/'SMOKE_MANIFEST.json'),'Manifest launch binding')
    result=read(output/'SMOKE_RESULT.json')
    require(result['passed'] and result['model_loads']=={'tele':1,'paddle':1},'Model load/runner gate')
    require(result['gpu']['uuid'].removeprefix('GPU-').lower()==manifest['gpu_uuid'].removeprefix('GPU-').lower(),'GPU UUID')
    tele=read(output/'TELE_LOAD.json')
    require(tele['dtype']=='torch.bfloat16','Tele dtype')
    require(all(k in tele['loading_info'] and not tele['loading_info'][k]
                for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),'Loading keys')
    require(not read(output/'RESOURCE_AUDIT.json')['unknown_resources'],'Tele auxiliary coverage')
    worker=output/'paddle-worker'
    load=read(worker/'native/load.json')
    require(load['model_loads']==1 and load['dtype']=='bfloat16' and not load['Paddle_layout_loaded'],'Paddle native model')
    require(not read(worker/'native/RESOURCE_AUDIT.json')['unknown_resources'],'Paddle auxiliary coverage')
    require(read(worker/'PROCESS_EXIT.json')['exit_code']==0,'Paddle worker lifecycle')
    responses={}; controls={}; request_count=0
    for path in sorted(worker.glob('region-*.request.json')):
        request_count+=1
        request=read(path);prefix=path.name.removesuffix('.request.json')
        response=read(worker/(prefix+'.response.json'))
        with Image.open(worker/(prefix+'.png')) as im:
            require(im.mode=='RGB' and [im.height,im.width,3]==request['shape'],'RGB shape')
            require(hashlib.sha256(im.tobytes()).hexdigest()==request['crop_sha256'],'Crop pixels')
        for key in ('run_id','page_id','region_id','crop_sha256','image_mode','shape','config_sha256'):
            require(request[key]==response[key],'Request/response identity: '+key)
        require(response['worker_session']==load['worker_session'],'Worker session changed')
        require(response['status'] in ('ok','native_control'),'Unexpected expert failure in passing smoke')
        gen=response['generation'];ids=gen['token_ids']
        require(gen['returned'] and gen['eos_ids']==[2] and gen['budget']==4096 and len(ids)<=4096 and 2 in ids,'Paddle EOS budget')
        require(gen['decode_strategy']=='greedy_search' and gen['trunc_input'] and gen['pad_token_id']==0,'Paddle native decoding')
        target=controls if response['status']=='native_control' else responses
        require(response['region_id'] not in target,'Repeated terminal region')
        target[response['region_id']]=response
    require(len(responses)>=3 and set(responses)==set(controls),'Direct native control coverage')
    for region,response in responses.items():
        control=controls[region]
        require(control['raw_text']==response['raw_text'] and bool(response['raw_text'].strip()),'Native text identity')
        for key in ('input_tensors','token_ids','eos_ids','budget','stop'):
            require(control['generation'][key]==response['generation'][key],'Native tensor/token identity')
    zero=0;route_ids=[];generations=0
    require(len(manifest['pages'])>=2,'Page coverage')
    for page in manifest['pages']:
        original=output/'official'/page['id'];baseline=read(original/'BLOCKS.json')
        for mode in ('official','off','pass-through','on'):
            folder=output/mode/page['id'];blocks=read(folder/'BLOCKS.json')
            if mode in ('off','pass-through'):
                require(blocks==baseline,'Native blocks changed in '+mode)
                for suffix in ('.md','_middle.json'):
                    left=original/'native'/page['id']/(page['id']+suffix)
                    right=folder/'native'/page['id']/(page['id']+suffix)
                    require(left.read_bytes()==right.read_bytes(),'Whole-page native file differs: '+mode+suffix)
            for generation in read(folder/'GENERATION.json'):
                generations+=1
                require(generation['returned'],'Tele generation failure')
                cfg=generation['effective_generation_config']
                require(cfg['max_length']==128000 and cfg['max_new_tokens'] is None,'Tele actual sequence budget')
                require(cfg['do_sample'] is False and cfg['no_repeat_ngram_size']==100 and cfg['repetition_penalty']==1.,'Tele native decode config')
            if mode=='on':
                require(blocks['layout']==baseline['layout'] and blocks['prepared']==baseline['prepared'],'Hybrid layout/preparation changed')
                require(len(blocks['before_postprocess'])==len(baseline['before_postprocess']),'Hybrid block count')
                for a,b in zip(baseline['before_postprocess'],blocks['before_postprocess']):
                    require({k:v for k,v in a.items() if k!='content'}=={k:v for k,v in b.items() if k!='content'},'Metadata changed')
                    if a['type']!='equation':require(a==b,'Nonformula text changed')
                require([x for x in baseline['after_postprocess'] if x['type']!='equation']==
                        [x for x in blocks['after_postprocess'] if x['type']!='equation'],
                        'Postprocessed nonformula blocks changed')
                routes=[x for x in read(folder/'REGIONS.json') if x.get('event')=='route']
                expert_routes=[x for x in routes if x['route']!='tele_native']
                if not expert_routes:zero+=1
                for route in expert_routes:
                    require(route['route']=='paddle_success' and route['block']['type']=='equation','Actual equation routing')
                    require(route['region_id']==f"{page['sha256']}:0:{route['block_index']}",'Original region index identity')
                    route_ids.append(route['region_id'])
    require(zero>0 and Counter(route_ids)==Counter(responses.keys()),'Route coverage/worker identity mismatch')
    return {'passed':True,'pages':len(manifest['pages']),'paddle_regions':len(responses),
            'zero_route_pages':zero,'expert_requests_including_controls':request_count,'tele_generations':generations,
            'static_capacity_upper_bound':manifest['capacity']['conservative_upper_bound'],
            'model_loads':result['model_loads'],'gpu_uuid':manifest['gpu_uuid']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
    try: result=verify(a.bundle,a.output)
    except Exception as exc:result={'passed':False,'error':str(exc),'traceback':traceback.format_exc()}
    result['verifier_sha256']=sha(Path(__file__))
    result['evidence_files']={str(x.relative_to(a.output)):sha(x) for x in sorted(a.output.rglob('*')) if x.is_file()}
    with a.receipt.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({'passed':result['passed'],'receipt':str(a.receipt),'sha256':sha(a.receipt)}))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
