"""Read-only observation of the actual vLLM process_inputs -> enqueue objects."""
import copy
import dataclasses
import enum
import hashlib
from .core import ContractError, canonical

def snapshot(value):
    if value is None or type(value) in (str,int,float,bool):return value
    if isinstance(value,enum.Enum):return {"enum":type(value).__name__,"value":snapshot(value.value)}
    if isinstance(value,(set,frozenset)):return sorted(snapshot(x) for x in value)
    if isinstance(value,(list,tuple)):return [snapshot(x) for x in value]
    if isinstance(value,dict):
        if any(not isinstance(k,str) for k in value):raise ContractError('Unknown payload key')
        return {k:snapshot(v) for k,v in value.items()}
    if hasattr(value,'detach') and hasattr(value,'shape'):
        raw=value.detach().cpu().contiguous()
        import torch
        return {'dtype':str(value.dtype),'shape':list(value.shape),'device':str(value.device),
                'contiguous':bool(value.is_contiguous()),
                'bytes_sha256':hashlib.sha256(raw.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
    if dataclasses.is_dataclass(value):return {f.name:snapshot(getattr(value,f.name)) for f in dataclasses.fields(value)}
    fields=getattr(value,'__struct_fields__',None)
    if fields:return {k:snapshot(getattr(value,k)) for k in fields}
    if hasattr(value,'__dict__'):return {k:snapshot(v) for k,v in vars(value).items()}
    raise ContractError('Unobservable payload type: '+type(value).__name__)

class EngineObserver:
    def __init__(self,llm,config_sha256,tensor_snapshot=snapshot):
        self.engine=llm.llm_engine;self.config=config_sha256;self.snap=tensor_snapshot
        self.pending=None;self.record=None;self.obj=None;self.cache={};self.ids=set()
        self.process=self.engine.input_processor.process_inputs
        self.enqueue=self.engine.engine_core.add_request
        self.engine.input_processor.process_inputs=self.process_inputs
        self.engine.engine_core.add_request=self.add_request

    def begin(self,identity):
        if self.pending is not None:raise ContractError('Concurrent Ovis request')
        self.pending=copy.deepcopy(identity);self.record=None;self.obj=None

    def features(self,request):
        features=request.mm_features
        if not isinstance(features,list) or len(features)!=1:raise ContractError('Exactly one actual image feature required')
        rows=[]
        for feature in features:
            if feature.modality!='image' or not feature.identifier or not feature.mm_hash:raise ContractError('Unknown multimodal feature')
            key=canonical([self.config,feature.modality,feature.identifier,feature.mm_hash])
            row={'modality':feature.modality,'identifier':feature.identifier,'mm_hash':feature.mm_hash,
                 'placeholder':self.snap(feature.mm_position),'cache_key':key}
            if feature.data is None:
                if key not in self.cache:raise ContractError('Unobserved engine cache entry')
                row.update(cache_hit=True,reference=self.cache[key]['request_id'],tensors=self.cache[key]['tensors'])
            else:
                data=feature.data.get_data()
                if set(data)!={'pixel_values','image_grid_thw'}:raise ContractError('Unknown or missing actual image tensor fields')
                tensors={k:self.snap(v) for k,v in data.items()}
                if any(not isinstance(v,dict) or not {'dtype','shape','device','bytes_sha256'}<=set(v) for v in tensors.values()):
                    raise ContractError('Actual multimodal tensors not observable')
                row.update(cache_hit=False,reference=None,tensors=tensors)
            rows.append(row)
        return rows

    def payload(self,request):
        fields=getattr(request,'__struct_fields__',None) or tuple(vars(request))
        result={k:self.snap(getattr(request,k)) for k in fields if k not in ('request_id','external_req_id','mm_features')}
        result['mm_features']=self.features(request)
        return result

    def gate(self,request):
        ids=request.prompt_token_ids;s=request.sampling_params
        if not isinstance(ids,list) or not ids or any(type(t) is not int or t<0 for t in ids):raise ContractError('Actual prompt IDs missing')
        if len(ids)+16384>32768:raise ContractError('Actual prompt exceeds pre-enqueue budget')
        if (s.max_tokens!=16384 or s.temperature!=0 or s.n!=1 or s.ignore_eos is not False
            or s.stop not in (None,[])
            or type(getattr(s,'_eos_token_id',None)) is not int or s._eos_token_id!=248046
            or type(getattr(s,'stop_token_ids',None)) is not list or s.stop_token_ids!=[248044]
            or any(type(t) is not int for t in s.stop_token_ids)
            or type(getattr(s,'all_stop_token_ids',None)) is not set or s.all_stop_token_ids!={248044,248046}
            or any(type(t) is not int for t in s.all_stop_token_ids)):
            raise ContractError('Actual Ovis sampling changed')
        if getattr(request,'pooling_params',None) is not None or getattr(request,'prompt_embeds',None) is not None:
            raise ContractError('Unexpected non-token generation input')

    def process_inputs(self,*args,**kwargs):
        if self.pending is None or self.obj is not None:raise ContractError('Unbound/repeated process_inputs')
        request=self.process(*args,**kwargs)
        self.gate(request)
        if getattr(request,'external_req_id',None) is not None:raise ContractError('Preassigned engine request')
        payload=self.payload(request)
        self.obj=request
        self.record={'identity':copy.deepcopy(self.pending),'request_id_before':request.request_id,
                     'payload':payload,'payload_sha256':canonical(payload),'submitted':False,
                     'prompt_length':len(request.prompt_token_ids),'prompt_ids_sha256':canonical(request.prompt_token_ids)}
        return request

    def add_request(self,request,*args,**kwargs):
        if request is not self.obj or self.record is None or self.record['submitted']:raise ContractError('Unobserved/duplicate engine submission')
        self.gate(request)
        if self.payload(request)!=self.record['payload']:raise ContractError('Actual engine payload changed before enqueue')
        before=self.record['request_id_before']
        if request.external_req_id!=before or not isinstance(request.request_id,str) or not request.request_id:
            raise ContractError('Engine request ID mapping invalid')
        if request.request_id in self.ids:raise ContractError('Replayed engine ID')
        self.ids.add(request.request_id)
        self.record.update(request_id=request.request_id,external_req_id=request.external_req_id,enqueue_attempted=True)
        result=self.enqueue(request,*args,**kwargs)
        self.record['submitted']=True
        for f in self.record['payload']['mm_features']:
            if not f['cache_hit']:self.cache[f['cache_key']]={'request_id':request.request_id,'tensors':copy.deepcopy(f['tensors'])}
        return result

    def finish(self,output):
        r=self.record
        if not r or not r['submitted'] or output.request_id!=r['external_req_id']:
            raise ContractError('Output external request ID differs from observed engine mapping')
        if list(output.prompt_token_ids or [])!=r['payload']['prompt_token_ids']:raise ContractError('Output prompt IDs differ')
        r=copy.deepcopy(r);r['output_request_id']=output.request_id
        self.pending=None;self.obj=None;self.record=None
        return r

    def restore(self):
        self.engine.input_processor.process_inputs=self.process
        self.engine.engine_core.add_request=self.enqueue
