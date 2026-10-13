"""One real LLM construction; same-load counts and actual engine observation."""
import functools
import time
from .core import ContractError, canonical, read, write
from .v32_protocol import CONFIG_SHA, IDENTITY, manifest
from .v32_observer import EngineObserver
from .v32_parameter_contract import validate_loaded_count


def eos_binding(llm,tokenizer,cfg,identity,root):
    """Read the existing resident objects; preserve actual values before rejection."""
    engine=llm.llm_engine;renderer=engine.renderer;processor=engine.input_processor
    fields=processor.generation_config_fields;errors=[]
    def bounded(value):
        if value is None or type(value) in (bool,int,float):return value
        if type(value) is str:return value if len(value)<=256 else {'invalid_type':'oversized_string'}
        if type(value) is list and len(value)<=16:return [bounded(v) for v in value]
        return {'invalid_type':type(value).__name__}
    def read_value(name,reader):
        try:return bounded(reader())
        except Exception as exc:
            errors.append({'field':name,'error_type':type(exc).__name__});return None
    fields_sha=None
    try:
        if type(fields) is not dict:raise TypeError('generation fields must be dict')
        fields_sha=canonical(fields)
    except (TypeError,ValueError,OverflowError):errors.append({'field':'generation_config_fields','error_type':'invalid_canonical_dict'})
    binding={'identity':dict(identity),'tokenizer_class':type(tokenizer).__name__,'tokenizer_module':type(tokenizer).__module__,
        'tokenizer_eos_token':read_value('tokenizer_eos_token',lambda:tokenizer.eos_token),
        'tokenizer_eos_token_id':read_value('tokenizer_eos_token_id',lambda:tokenizer.eos_token_id),
        'tokenizer_pad_token':read_value('tokenizer_pad_token',lambda:tokenizer.pad_token),
        'tokenizer_pad_token_id':read_value('tokenizer_pad_token_id',lambda:tokenizer.pad_token_id),
        'im_end_token_id':read_value('im_end_token_id',lambda:tokenizer.convert_tokens_to_ids('<|im_end|>')),
        'endoftext_token_id':read_value('endoftext_token_id',lambda:tokenizer.convert_tokens_to_ids('<|endoftext|>')),
        'renderer_eos_token_id':read_value('renderer_eos_token_id',lambda:renderer.get_eos_token_id()),
        'hf_text_config_eos_token_id':read_value('hf_text_config_eos_token_id',lambda:engine.model_config.hf_text_config.eos_token_id),
        'generation_eos_token_id':read_value('generation_eos_token_id',lambda:fields.get('eos_token_id')),
        'generation_config_fields_sha256':fields_sha,
        'extra_stop_sources':sorted(k for k in ('stop','stop_token_ids','stop_strings') if type(fields) is dict and k in fields),
        'manifest_eos_token_id':bounded(cfg.get('eos_token_id')),
        'same_renderer_tokenizer':getattr(renderer,'tokenizer',None) is tokenizer,
        'same_input_processor_renderer':getattr(processor,'renderer',None) is renderer,
        'read_errors':errors,'validated':False,
        'provenance':{'tokenizer_config_sha256':'49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c',
            'model_config_sha256':'b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204',
            'eos_static_receipt_sha256':'85828c817825c130c36efa469ca54d1b7130d528d89c087100038d97de0b43eb',
            'tokenizer_extra_receipt_sha256':'9dc667b4759c2d70a986666117137c6b671d5c04d5b5f6d256a70171135224b4',
            'sampling_params_sha256':'c5f9be229115230eccc33788d30fb2b399d71d7c1093962e9bf8f125fceb19a6'}}
    write(root/'EOS_BINDING_RAW.json',binding)
    expected={'tokenizer_eos_token_id':248046,'tokenizer_pad_token_id':248044,'im_end_token_id':248046,
        'endoftext_token_id':248044,'renderer_eos_token_id':248046,'hf_text_config_eos_token_id':248044,
        'generation_eos_token_id':248044,'manifest_eos_token_id':248044}
    if (errors or binding['extra_stop_sources'] or not fields_sha
        or binding['tokenizer_eos_token']!='<|im_end|>' or binding['tokenizer_pad_token']!='<|endoftext|>'
        or not binding['same_renderer_tokenizer'] or not binding['same_input_processor_renderer']
        or any(type(binding[k]) is not int or binding[k]!=v for k,v in expected.items())):
        raise ContractError('Resident Ovis EOS binding differs; see EOS_BINDING_RAW.json')
    return {**binding,'validated':True}

class Ovis:
    def __init__(self,control,root,event):
        self.control,self.root,self.event=control,root,event
        r=control['runtime'];cfg=manifest();self.cfg=cfg
        from vllm import LLM,SamplingParams
        import torch
        if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability(0)[0]<8:raise ContractError('Ovis requires one BF16-capable GPU')
        self.identity={'run_id':control['run_id'],'load_id':control['run_id']+'/ovis/1',
                       'model_sha256':r['v32']['model_sha256'],'source_sha256':control['source']['sha256'],
                       'environment_sha256':r['v32']['source_sha256']}
        event('LOAD_START',engine='ovis',load_id=self.identity['load_id'])
        self.llm=LLM(model='/assets/ovis_model',tokenizer='/assets/ovis_model',trust_remote_code=False,dtype='bfloat16',
            tensor_parallel_size=1,gpu_memory_utilization=.8,gdn_prefill_backend='triton',max_model_len=32768,
            max_num_seqs=1,limit_mm_per_prompt={'image':1},generation_config='vllm',seed=0,
            worker_cls='hybrid_v3_full_eval_v1.v32_engine_worker.AuditedWorker')
        remaining=min(60,control['absolute_stop_monotonic']-time.monotonic())
        if remaining<=0:raise ContractError('No time for same-load count')
        counts=self.llm.collective_rpc('v32_count_loaded_parameters',timeout=remaining,args=(self.identity,))
        if len(counts)!=1:raise ContractError('Ovis tensor-parallel worker count changed')
        count=counts[0];write(root/'LOADED_PARAMETERS_RAW.json',count)
        if count['identity']!=self.identity or count['device']['uuid'].lower()!=r['gpu_uuids'][-1].lower():raise ContractError('Ovis loaded count identity differs')
        if count['resource_audit']['unknown_resources']:raise ContractError('Ovis actual worker resource audit failed')
        mapping=validate_loaded_count(count)
        count.update(component='ovis',model_files_sha256=r['v32']['model_sha256'],measurement_source_sha256=control['source']['sha256'])
        if r['v32']['parameter_status']!='bootstrap_header':
            row=next(x for x in r['parameter_ledger'] if x['component']=='ovis')
            evidence=read('/assets/'+row['evidence']['asset']+'/'+row['evidence']['path'])
            if any(count[k]!=evidence[k] for k in ('stored_elements','unique_trainable','buffers','structure_sha256')):
                raise ContractError('Same-load Ovis count differs from accepted first load')
        other=sum(x['unique_trainable'] for x in r['parameter_ledger'] if x['component']!='ovis')
        if other+count['unique_trainable']>4_000_000_000:raise ContractError('Loaded V32 capacity cap exceeded')
        write(root/'LOADED_PARAMETER_MAPPING.json',mapping)
        write(root/'LOADED_PARAMETERS.json',count)
        self.tokenizer=self.llm.get_tokenizer()
        self.eos_binding=eos_binding(self.llm,self.tokenizer,cfg,self.identity,root)
        self.generation_eos_token_id=self.eos_binding['generation_eos_token_id']
        self.prompt=self.tokenizer.apply_chat_template([{'role':'user','content':[{'type':'image'},
            {'type':'text','text':cfg['official_prompt']}]}],**cfg['chat_template_options'])
        self.sampling=SamplingParams(**cfg['sampling'],seed=0)
        self.observer=EngineObserver(self.llm,CONFIG_SHA)
        write(root/'ENGINE_CONFIG.json',{'config':cfg,'prompt':self.prompt,'sampling_repr':repr(self.sampling),
              'worker_class':'hybrid_v3_full_eval_v1.v32_engine_worker.AuditedWorker','load_identity':self.identity,'eos_binding':self.eos_binding})
        event('LOAD_RESULT',engine='ovis',returned=True,load_id=self.identity['load_id'])
    def call(self,request,image):
        identity={k:request[k] for k in IDENTITY};self.observer.begin(identity)
        result=self.llm.generate([{'prompt':self.prompt,'multi_modal_data':{'image':image},
            'mm_processor_kwargs':self.cfg['image_processor_call']}],self.sampling,use_tqdm=False)
        if len(result)!=1 or len(result[0].outputs)!=1:raise ContractError('Ovis output cardinality differs')
        output=result[0];part=output.outputs[0];evidence=self.observer.finish(output)
        return {**identity,'raw_text':part.text,'engine_evidence':evidence,
                'generation':{'finished':output.finished,'finish_reason':part.finish_reason,'stop_reason':part.stop_reason,
                    'token_ids':list(part.token_ids),'eos_token_id':self.generation_eos_token_id,'config_sha256':CONFIG_SHA}}
    def close(self):
        if hasattr(self,'observer'):self.observer.restore()
        if hasattr(self,'llm'):self.llm.llm_engine.engine_core.shutdown()
